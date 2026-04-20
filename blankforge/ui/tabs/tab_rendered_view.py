from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from blankforge.data.model import BoardModel
from blankforge.geometry.board import BoardMesh
from blankforge.ui.widgets.gl_viewport import GLViewport
from blankforge.ui.widgets.viewport_controls import (
    BOARD_PRESETS,
    ShadingControlsSidebar,
    SidebarToggleStrip,
    ViewPresetsWidget,
)


# ---------------------------------------------------------------------------
# Wireframe line geometry (board-specific)
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

    # Longitudinal lines must follow the board's curves closely.
    LONG_STATIONS = 200

    def _make_rings(positions: np.ndarray) -> list[np.ndarray]:
        rings = []
        for pos in positions:
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
        return rings

    pairs: list[np.ndarray] = []

    if show_lat:
        lat_rings = _make_rings(np.linspace(0, L, n_stations))
        for ring in lat_rings:
            n = len(ring)
            for i in range(n):
                pairs.append(ring[i])
                pairs.append(ring[(i + 1) % n])

    if show_long:
        long_rings = _make_rings(np.linspace(0, L, LONG_STATIONS))
        if len(long_rings) >= 2:
            n_ring = len(long_rings[0])
            step = max(1, n_ring // n_contour)
            for j in range(0, n_ring, step):
                for i in range(len(long_rings) - 1):
                    if j < len(long_rings[i]) and j < len(long_rings[i + 1]):
                        pairs.append(long_rings[i][j])
                        pairs.append(long_rings[i + 1][j])

    if show_cross:
        cross_rings = _make_rings(np.linspace(0, L, n_stations))
        if len(cross_rings) >= 2:
            for i in range(len(cross_rings) - 1):
                n      = len(cross_rings[i])
                n_next = len(cross_rings[i + 1])
                for j in range(n):
                    pairs.append(cross_rings[i][j])
                    pairs.append(cross_rings[i + 1][(j + 1) % n_next])
                    pairs.append(cross_rings[i][j])
                    pairs.append(cross_rings[i + 1][(j - 1) % n_next])

    return np.array(pairs, dtype=np.float32) if pairs else np.zeros((0, 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------

class RenderedViewTab(QWidget):
    mesh_quality_changed = Signal(int, int)  # (resolution, n_contour)

    def __init__(self, model: BoardModel, model_changed: Signal,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._model_changed = model_changed
        self._mesh: BoardMesh | None = None       # board-only mesh
        self._show_fins: bool = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        content = QHBoxLayout()
        content.setSpacing(0)

        self._viewport = GLViewport(self)
        content.addWidget(self._viewport, stretch=1)

        # Shading sidebar (board mode — show detail controls)
        self._shading = ShadingControlsSidebar(
            self._viewport,
            wireframe_builder=self._build_wireframe_lines,
            show_wireframe_detail=True,
            initial_solid=100,
        )
        # Prepend board-specific mesh quality section at the top of the sidebar
        self._shading.prepend_section(self._build_quality_section())

        # Toggle strip + sidebar
        toggle = SidebarToggleStrip(self._shading)
        content.addWidget(toggle)
        content.addWidget(self._shading)

        layout.addLayout(content, stretch=1)

        # Floating view controls overlay
        overlay = ViewPresetsWidget(self._viewport, presets=BOARD_PRESETS)
        self._viewport.set_controls_widget(overlay)

    def _build_quality_section(self) -> QWidget:
        from PySide6.QtWidgets import QSlider
        from PySide6.QtCore import Qt
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(6)
        vbox.addWidget(QLabel("<b>Mesh Quality</b>"))

        row_w = QWidget()
        row = QHBoxLayout(row_w)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Resolution:"))
        self._mesh_quality = QSlider(Qt.Orientation.Horizontal)
        self._mesh_quality.setMinimum(20)
        self._mesh_quality.setMaximum(200)
        self._mesh_quality.setValue(50)
        row.addWidget(self._mesh_quality, stretch=1)
        self._mesh_quality_label = QLabel("50")
        self._mesh_quality_label.setFixedWidth(32)
        row.addWidget(self._mesh_quality_label)
        vbox.addWidget(row_w)

        self._mesh_quality.sliderReleased.connect(self._on_mesh_quality_changed)
        self._mesh_quality.valueChanged.connect(
            lambda v: self._mesh_quality_label.setText(str(v))
        )

        # ── Fins overlay ─────────────────────────────────────────────
        self._fins_check = QCheckBox("Show Fins")
        self._fins_check.setChecked(False)
        self._fins_check.setToolTip(
            "Overlay the fins defined in the Fins tab onto the board.\n"
            "Fins are placed at the bottom of the board at their configured positions."
        )
        self._fins_check.toggled.connect(self._on_fins_toggled)
        vbox.addWidget(self._fins_check)

        # Blender button (optional)
        from blankforge.renderer.blender import BlenderRenderer
        if BlenderRenderer().is_available():
            from PySide6.QtWidgets import QPushButton
            render_btn = QPushButton("Render with Blender…")
            render_btn.clicked.connect(self._render_blender)
            vbox.addWidget(render_btn)

        return container

    def _on_mesh_quality_changed(self) -> None:
        resolution = self._mesh_quality.value()
        n_contour = max(16, int(resolution * 0.64))
        self.mesh_quality_changed.emit(resolution, n_contour)

    # ── Mesh update ───────────────────────────────────────────────────

    def update_mesh(self, mesh: BoardMesh) -> None:
        self._mesh = mesh
        self._viewport.update_mesh(self._combined_mesh())
        self._shading.on_mesh_updated()

    def _on_fins_toggled(self, checked: bool) -> None:
        self._show_fins = checked
        if self._mesh is not None:
            self._viewport.update_mesh(self._combined_mesh())
            self._shading.on_mesh_updated()

    def _combined_mesh(self) -> BoardMesh:
        """Return board mesh, optionally merged with all fin meshes."""
        if not self._show_fins or self._model.fins is None or self._mesh is None:
            return self._mesh
        from blankforge.geometry.fin import (
            build_fin_mesh, merge_meshes, transform_fin_to_board,
        )
        parts = [self._mesh]
        for fin in self._model.fins.fins:
            fm = build_fin_mesh(fin, n_height=30, n_chord=12)
            if fm is not None:
                parts.append(transform_fin_to_board(fm, fin, self._model))
        result = merge_meshes(*parts)
        return result if result is not None else self._mesh

    def _build_wireframe_lines(self) -> np.ndarray | None:
        return _build_wireframe_lines(
            self._model,
            n_stations=self._shading.wf_density,
            n_contour=self._shading.wf_density,
            show_long=self._shading.wf_long,
            show_lat=self._shading.wf_lat,
            show_cross=self._shading.wf_cross,
        )

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
