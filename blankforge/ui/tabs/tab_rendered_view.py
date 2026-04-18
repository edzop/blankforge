from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton,
    QSlider, QToolButton, QVBoxLayout, QWidget,
)

from blankforge.data.model import BoardModel
from blankforge.geometry.board import BoardMesh
from blankforge.ui.widgets.gl_viewport import GLViewport


# ---------------------------------------------------------------------------
# Wireframe line geometry
# ---------------------------------------------------------------------------

def _build_wireframe_lines(
    model: BoardModel,
    n_stations: int,
    n_contour: int,
    show_long: bool,
    show_lat: bool,
    show_cross: bool = False,
) -> np.ndarray:
    from blankforge.geometry.curves import (
        BoardCurveEvaluator, RailProfileEvaluator, resolve_thickness_curve,
    )
    L = model.parameters.length_mm
    width_eval = BoardCurveEvaluator(model.curves.width)
    thick_eval = BoardCurveEvaluator(
        resolve_thickness_curve(model.curves.thickness, model.parameters.thickness_mm)
    )
    rocker_eval = BoardCurveEvaluator(model.curves.rocker)
    rail_eval = RailProfileEvaluator(model.curves.rail)

    stations = np.linspace(0, L, n_stations)

    rings: list[np.ndarray] = []
    for pos in stations:
        hw = float(width_eval(pos))
        ht = float(thick_eval(pos))
        rocker = float(rocker_eval(pos))
        profile = rail_eval.at(pos)
        pts = rail_eval.cross_section_points(pos, hw, ht, n_points=n_contour, profile=profile)
        left = pts[::-1].copy()
        left[:, 0] *= -1
        ring2d = np.vstack([pts, left])
        ring3d = np.zeros((len(ring2d), 3), dtype=np.float32)
        ring3d[:, 0] = float(pos)
        ring3d[:, 1] = ring2d[:, 0]
        ring3d[:, 2] = ring2d[:, 1] + rocker
        rings.append(ring3d)

    pairs: list[np.ndarray] = []

    if show_lat and rings:
        for ring in rings:
            n = len(ring)
            for i in range(n):
                pairs.append(ring[i])
                pairs.append(ring[(i + 1) % n])

    if show_long and len(rings) >= 2:
        n_ring = len(rings[0])
        step = max(1, n_ring // n_contour)
        for j in range(0, n_ring, step):
            for i in range(len(rings) - 1):
                if j < len(rings[i]) and j < len(rings[i + 1]):
                    pairs.append(rings[i][j])
                    pairs.append(rings[i + 1][j])

    if show_cross and len(rings) >= 2:
        for i in range(len(rings) - 1):
            n = len(rings[i])
            for j in range(n):
                pairs.append(rings[i][j])
                pairs.append(rings[i + 1][(j + 1) % n])
                pairs.append(rings[i][j])
                pairs.append(rings[i + 1][(j - 1) % n])

    return np.array(pairs, dtype=np.float32) if pairs else np.zeros((0, 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------

_VIEW_PRESETS = [
    ("Persp", 45.0, 25.0),
    ("Top",   0.0,  89.0),
    ("Side",  0.0,   0.5),
    ("Front", 90.0,  0.5),
]


class RenderedViewTab(QWidget):
    mesh_quality_changed = Signal(int, int)  # (resolution, n_contour)

    def __init__(self, model: BoardModel, model_changed: Signal,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._model_changed = model_changed
        self._mesh: BoardMesh | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        content = QHBoxLayout()
        content.setSpacing(0)
        self._viewport = GLViewport(self)
        content.addWidget(self._viewport, stretch=1)

        # Narrow sidebar-toggle strip (always visible, sits between viewport and sidebar)
        self._toggle_strip = self._build_toggle_strip()
        content.addWidget(self._toggle_strip)

        self._sidebar = self._build_sidebar()
        content.addWidget(self._sidebar)
        layout.addLayout(content, stretch=1)

        # Floating view controls overlay (positioned near gizmo by GLViewport)
        self._viewport.set_controls_widget(self._build_view_controls())

    def _build_toggle_strip(self) -> QWidget:
        strip = QWidget()
        strip.setFixedWidth(18)
        strip.setStyleSheet("QWidget { background: #1a1c22; border-left: 1px solid #333; border-right: 1px solid #333; }")
        vbox = QVBoxLayout(strip)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addStretch()
        self._toggle_btn = QToolButton()
        self._toggle_btn.setText("»")
        self._toggle_btn.setFixedSize(18, 40)
        self._toggle_btn.setStyleSheet(
            "QToolButton { background: #252830; border: none; color: #7a8aaa; font-size: 11px; }"
            "QToolButton:hover { background: #2e3340; color: #c0cce0; }"
        )
        self._toggle_btn.setToolTip("Collapse settings panel")
        self._toggle_btn.clicked.connect(self._toggle_sidebar)
        vbox.addWidget(self._toggle_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        vbox.addStretch()
        return strip

    def _build_view_controls(self) -> QWidget:
        container = QWidget()
        container.setStyleSheet(
            "QWidget { background: transparent; }"
            "QPushButton {"
            "  background: rgba(20,22,28,200);"
            "  border: 1px solid rgba(80,90,110,160);"
            "  color: #bdc8dc;"
            "  font-size: 10px;"
            "  padding: 3px 6px;"
            "  border-radius: 3px;"
            "}"
            "QPushButton:hover { background: rgba(50,65,100,220); color: #fff; }"
            "QPushButton:pressed { background: rgba(40,55,90,240); }"
        )
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(2)

        reset_btn = QPushButton("Reset")
        reset_btn.setFixedWidth(68)
        reset_btn.clicked.connect(self._viewport.reset_view)
        vbox.addWidget(reset_btn)

        for label, az, el in _VIEW_PRESETS[1:]:  # Top, Side, Front
            btn = QPushButton(label)
            btn.setFixedWidth(68)
            btn.clicked.connect(lambda _, a=az, e=el: self._viewport.set_preset_view(a, e))
            vbox.addWidget(btn)

        fit_btn = QPushButton("Fit")
        fit_btn.setFixedWidth(68)
        fit_btn.clicked.connect(self._viewport.fit_view)
        vbox.addWidget(fit_btn)

        container.adjustSize()
        return container

    def _sep(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(1)
        w.setStyleSheet("background: #383838;")
        return w

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(210)
        sidebar.setStyleSheet(
            "QWidget { background: #1e2025; border-left: 1px solid #333; }"
            "QLabel { border: none; } QCheckBox { border: none; }"
        )

        vbox = QVBoxLayout(sidebar)
        vbox.setContentsMargins(10, 10, 10, 10)
        vbox.setSpacing(8)
        vbox.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Mesh quality — smooths out shaded-mode curves by increasing tessellation
        vbox.addWidget(QLabel("<b>Mesh Quality</b>"))
        mq_row = QHBoxLayout()
        mq_row.addWidget(QLabel("Resolution:"))
        self._mesh_quality = QSlider(Qt.Orientation.Horizontal)
        self._mesh_quality.setMinimum(20)
        self._mesh_quality.setMaximum(200)
        self._mesh_quality.setValue(50)
        mq_row.addWidget(self._mesh_quality, stretch=1)
        self._mesh_quality_label = QLabel("50")
        self._mesh_quality_label.setFixedWidth(32)
        mq_row.addWidget(self._mesh_quality_label)
        vbox.addLayout(mq_row)
        self._mesh_quality.sliderReleased.connect(self._on_mesh_quality_changed)
        self._mesh_quality.valueChanged.connect(
            lambda v: self._mesh_quality_label.setText(str(v))
        )

        # Shading
        vbox.addWidget(QLabel("<b>Shading</b>"))
        self._shade_combo = QComboBox()
        self._shade_combo.addItems(["Solid", "Wireframe", "Heatmap"])
        self._shade_combo.currentIndexChanged.connect(self._on_shade_changed)
        vbox.addWidget(self._shade_combo)

        # Wireframe controls (hidden when not in Wireframe mode)
        self._wf_box = QWidget()
        wf_layout = QVBoxLayout(self._wf_box)
        wf_layout.setContentsMargins(0, 4, 0, 0)
        wf_layout.setSpacing(4)

        self._wf_long = QCheckBox("Longitudinal")
        self._wf_long.setChecked(True)
        self._wf_lat = QCheckBox("Latitudinal")
        self._wf_lat.setChecked(True)
        self._wf_cross = QCheckBox("Crosshatch")
        self._wf_cross.setChecked(False)
        wf_layout.addWidget(self._wf_long)
        wf_layout.addWidget(self._wf_lat)
        wf_layout.addWidget(self._wf_cross)

        density_row = QHBoxLayout()
        density_row.addWidget(QLabel("Density:"))
        self._wf_density = QSlider(Qt.Orientation.Horizontal)
        self._wf_density.setMinimum(4)
        self._wf_density.setMaximum(40)
        self._wf_density.setValue(12)
        density_row.addWidget(self._wf_density, stretch=1)
        self._wf_density_label = QLabel("12")
        self._wf_density_label.setFixedWidth(24)
        density_row.addWidget(self._wf_density_label)
        wf_layout.addLayout(density_row)

        self._wf_long.toggled.connect(self._on_wireframe_settings_changed)
        self._wf_lat.toggled.connect(self._on_wireframe_settings_changed)
        self._wf_cross.toggled.connect(self._on_wireframe_settings_changed)
        self._wf_density.valueChanged.connect(self._on_density_changed)

        vbox.addWidget(self._wf_box)
        self._wf_box.setVisible(False)

        # Heatmap controls (shown only in Heatmap mode)
        self._hm_box = QWidget()
        hm_layout = QVBoxLayout(self._hm_box)
        hm_layout.setContentsMargins(0, 4, 0, 0)
        hm_layout.setSpacing(4)
        sensitivity_row = QHBoxLayout()
        sensitivity_row.addWidget(QLabel("Sensitivity:"))
        self._hm_sensitivity = QSlider(Qt.Orientation.Horizontal)
        self._hm_sensitivity.setMinimum(1)
        self._hm_sensitivity.setMaximum(100)
        self._hm_sensitivity.setValue(100)
        sensitivity_row.addWidget(self._hm_sensitivity, stretch=1)
        self._hm_sensitivity_label = QLabel("1.00")
        self._hm_sensitivity_label.setFixedWidth(32)
        sensitivity_row.addWidget(self._hm_sensitivity_label)
        hm_layout.addLayout(sensitivity_row)
        self._hm_sensitivity.valueChanged.connect(self._on_heatmap_sensitivity_changed)
        vbox.addWidget(self._hm_box)
        self._hm_box.setVisible(False)

        # Blender
        from blankforge.renderer.blender import BlenderRenderer
        if BlenderRenderer().is_available():
            vbox.addWidget(self._sep())
            vbox.addWidget(QLabel("<b>Blender Render</b>"))
            render_btn = QPushButton("Render with Blender…")
            render_btn.clicked.connect(self._render_blender)
            vbox.addWidget(render_btn)

        vbox.addStretch()
        return sidebar

    def _toggle_sidebar(self) -> None:
        visible = not self._sidebar.isVisible()
        self._sidebar.setVisible(visible)
        self._toggle_btn.setText("»" if visible else "«")
        self._toggle_btn.setToolTip("Collapse settings panel" if visible else "Expand settings panel")

    def update_mesh(self, mesh: BoardMesh) -> None:
        self._mesh = mesh
        self._viewport.update_mesh(mesh)
        idx = self._shade_combo.currentIndex()
        if idx == 1:
            self._rebuild_wireframe_lines()
        elif idx == 2:
            self._viewport.set_heatmap(True, self._hm_sensitivity.value() / 100.0)

    def _on_shade_changed(self, idx: int) -> None:
        self._viewport.set_wireframe(idx == 1)
        self._viewport.set_heatmap(idx == 2, self._hm_sensitivity.value() / 100.0)
        self._wf_box.setVisible(idx == 1)
        self._hm_box.setVisible(idx == 2)
        if idx == 1:
            self._rebuild_wireframe_lines()
        else:
            self._viewport.update_wireframe_lines(None)

    def _on_heatmap_sensitivity_changed(self, val: int) -> None:
        sensitivity = val / 100.0
        self._hm_sensitivity_label.setText(f"{sensitivity:.2f}")
        if self._shade_combo.currentIndex() == 2:
            self._viewport.set_heatmap(True, sensitivity)

    def _on_wireframe_settings_changed(self) -> None:
        if self._shade_combo.currentIndex() == 1:
            self._rebuild_wireframe_lines()

    def _on_density_changed(self, val: int) -> None:
        self._wf_density_label.setText(str(val))
        if self._shade_combo.currentIndex() == 1:
            self._rebuild_wireframe_lines()

    def _on_mesh_quality_changed(self) -> None:
        # Scale both station count (length axis) and cross-section points together
        resolution = self._mesh_quality.value()
        n_contour = max(16, int(resolution * 0.64))  # preserve rough aspect of defaults (50→32)
        self.mesh_quality_changed.emit(resolution, n_contour)

    def _rebuild_wireframe_lines(self) -> None:
        density = self._wf_density.value()
        verts = _build_wireframe_lines(
            self._model,
            n_stations=density,
            n_contour=density,
            show_long=self._wf_long.isChecked(),
            show_lat=self._wf_lat.isChecked(),
            show_cross=self._wf_cross.isChecked(),
        )
        self._viewport.update_wireframe_lines(verts if len(verts) > 0 else None)

    def _render_blender(self) -> None:
        from pathlib import Path
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        from blankforge.renderer.blender import BlenderRenderer
        from blankforge.geometry.board import BoardGeometryBuilder

        path, _ = QFileDialog.getSaveFileName(self, "Save Render", "", "PNG Images (*.png)")
        if not path:
            return
        mesh, _ = BoardGeometryBuilder(use_occt=False).build(self._model, resolution=60)
        try:
            BlenderRenderer().render(mesh, Path(path), width=1920, height=1080)
        except Exception as e:
            QMessageBox.critical(self, "Blender Render Failed", str(e))
