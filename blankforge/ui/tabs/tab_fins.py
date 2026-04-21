"""Fins tab — split 2-D planform editor + 3-D GL preview with shared shading controls."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Info-icon helper
# ---------------------------------------------------------------------------

def _with_info(widget: QWidget, tip: str) -> QWidget:
    """Wrap *widget* with a hoverable ⓘ info icon carrying *tip* as tooltip."""
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    row.addWidget(widget, 1)
    lbl = QLabel("ⓘ")
    lbl.setToolTip(tip)
    lbl.setStyleSheet("color: #5888c8; font-size: 11px;")
    lbl.setCursor(Qt.CursorShape.WhatsThisCursor)
    row.addWidget(lbl)
    return container

from blankforge.data.fin_model import (
    BOX_PRESETS,
    FIN_TEMPLATES,
    FinDef,
    FinFoil,
    FinSetup,
    _thruster_setup,
    _single_setup,
    _two_plus_one_setup,
    _twin_setup,
)
from blankforge.data.model import BoardModel
from blankforge.geometry.fin import apply_cant_toe, build_fin_mesh
from blankforge.ui.widgets.fin_2d_editor import FinOutlineEditor
from blankforge.ui.widgets.gl_viewport import GLViewport
from blankforge.ui.widgets.viewport_controls import (
    FIN_PRESETS,
    ShadingControlsSidebar,
    SidebarToggleStrip,
    ViewPresetsWidget,
)


# Map setup type → factory
_SETUP_FACTORIES = {
    "thruster": lambda: _thruster_setup("FCS II"),
    "single":   lambda: _single_setup("Futures"),
    "twin":     lambda: _twin_setup("FCS II"),
    "2+1":      lambda: _two_plus_one_setup("Futures"),
}
_SETUP_LABELS = ["thruster", "single", "twin", "2+1"]


# ---------------------------------------------------------------------------
# Fin wireframe: extract unique mesh edges
# ---------------------------------------------------------------------------

def _fin_wireframe_from_mesh(mesh) -> np.ndarray | None:
    """Build line-segment pairs from unique mesh edges."""
    if mesh is None:
        return None
    tris = mesh.triangles  # (M, 3)
    verts = mesh.vertices  # (N, 3)
    # Build edge set
    edges = set()
    for tri in tris:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))
    if not edges:
        return None
    pairs = []
    for a, b in edges:
        pairs.append(verts[a])
        pairs.append(verts[b])
    return np.array(pairs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------

class FinsTab(QWidget):
    """Full fins editing tab."""

    def __init__(
        self, model: BoardModel, model_changed_signal: Signal, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._external_changed = model_changed_signal
        self._current_mesh = None  # latest fin BoardMesh for wireframe
        self._build_ui()
        self._refresh_from_model()

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ---- Left panel: setup selector + 2D editor + fin properties ----
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(8, 8, 8, 8)

        # Setup type selector
        setup_row = QHBoxLayout()
        setup_row.addWidget(QLabel("Setup:"))
        self._setup_combo = QComboBox()
        for lbl in _SETUP_LABELS:
            self._setup_combo.addItem(lbl)
        self._setup_combo.currentIndexChanged.connect(self._on_setup_changed)
        setup_row.addWidget(self._setup_combo, 1)
        left_lay.addLayout(setup_row)

        # Per-fin selector
        fin_row = QHBoxLayout()
        fin_row.addWidget(QLabel("Fin:"))
        self._fin_combo = QComboBox()
        self._fin_combo.currentIndexChanged.connect(self._on_fin_selected)
        fin_row.addWidget(self._fin_combo, 1)
        left_lay.addLayout(fin_row)

        # 2D outline editor
        self._editor = FinOutlineEditor(FinDef())  # replaced in _refresh
        self._editor.fin_changed.connect(self._on_fin_changed)
        self._editor.point_selected.connect(self._on_point_selected)
        left_lay.addWidget(self._editor, 1)

        # Fin properties panel
        left_lay.addWidget(self._build_properties_panel())

        splitter.addWidget(left)

        # ---- Right panel: 3D GL viewport + shading sidebar ----
        right = QWidget()
        right_lay = QHBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        self._gl = GLViewport()
        right_lay.addWidget(self._gl, 1)

        # Shared shading sidebar (fin mode — no board-detail controls)
        self._shading = ShadingControlsSidebar(
            self._gl,
            wireframe_builder=self._build_wireframe_lines,
            show_wireframe_detail=False,
            initial_solid=100,
            thickness_axis="y",
        )
        toggle = SidebarToggleStrip(self._shading)
        right_lay.addWidget(toggle)
        right_lay.addWidget(self._shading)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        # Floating view controls (Iso / Top / Side + Reset / Fit)
        overlay = ViewPresetsWidget(self._gl, presets=FIN_PRESETS)
        self._gl.set_controls_widget(overlay)

    def _build_properties_panel(self) -> QGroupBox:
        grp = QGroupBox("Fin Properties")
        lay = QFormLayout(grp)
        lay.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._name_lbl = QLabel("—")
        lay.addRow("Name:", self._name_lbl)

        self._template_combo = QComboBox()
        for name in FIN_TEMPLATES:
            self._template_combo.addItem(name)
        self._template_combo.currentTextChanged.connect(self._apply_template)
        lay.addRow("Template:", _with_info(self._template_combo,
            "Load a preset fin outline. You can then drag control points to customise it."))

        self._box_combo = QComboBox()
        for bt in BOX_PRESETS:
            self._box_combo.addItem(bt)
        self._box_combo.currentTextChanged.connect(self._on_box_changed)
        lay.addRow("Box:", _with_info(self._box_combo,
            "The fin plug / box system:\n"
            "• FCS / FCS II — twin-tab screws, removable\n"
            "• Futures — single-screw, removable\n"
            "• Glassed-in — fin is permanently laminated to the board"))

        self._foil_spin = QDoubleSpinBox()
        self._foil_spin.setRange(2.0, 50.0)
        self._foil_spin.setSuffix(" %")
        self._foil_spin.setSingleStep(0.5)
        self._foil_spin.valueChanged.connect(self._on_foil_changed)
        lay.addRow("Foil thickness:", _with_info(self._foil_spin,
            "Maximum cross-section thickness as a percentage of the local chord length.\n"
            "Typical values: 10–15 % (performance fins), 15–20 % (beginner/longboard fins)."))

        # ── Selected-point controls ──────────────────────────────────

        # Sharpness: edge taper rate (0=round/soft, 1=knife)
        sharp_row = QHBoxLayout()
        self._sharpness_slider = QSlider(Qt.Orientation.Horizontal)
        self._sharpness_slider.setRange(0, 100)
        self._sharpness_slider.setValue(0)
        self._sharpness_slider.valueChanged.connect(self._on_sharpness_changed)
        self._sharpness_lbl = QLabel("0%")
        sharp_row.addWidget(self._sharpness_slider, 1)
        sharp_row.addWidget(self._sharpness_lbl)
        self._sharp_container = QWidget()
        self._sharp_container.setLayout(sharp_row)
        self._sharp_container.setEnabled(False)
        lay.addRow("Sharpness:", _with_info(self._sharp_container,
            "Controls how quickly the foil edge tapers at this height.\n"
            "0 % — gradual taper: soft, round edge (more forgiving)\n"
            "100 % — abrupt taper: knife-sharp edge (high performance)\n"
            "The maximum foil thickness in the centre is not affected."))

        # Influence: spline tension (0=smooth curve, 1=sharp corner)
        inf_row = QHBoxLayout()
        self._influence_slider = QSlider(Qt.Orientation.Horizontal)
        self._influence_slider.setRange(0, 100)
        self._influence_slider.setValue(0)
        self._influence_slider.valueChanged.connect(self._on_influence_changed)
        self._influence_lbl = QLabel("0%")
        inf_row.addWidget(self._influence_slider, 1)
        inf_row.addWidget(self._influence_lbl)
        self._inf_container = QWidget()
        self._inf_container.setLayout(inf_row)
        self._inf_container.setEnabled(False)
        lay.addRow("Influence:", _with_info(self._inf_container,
            "Controls the spline tension (curve tightness) at this control point.\n"
            "0 % — smooth Catmull-Rom curve passing through the point\n"
            "100 % — sharp corner: the outline bends abruptly at this point\n"
            "Visible as the control point fill colour: blue → yellow-green."))

        # Placement
        self._x_tail_spin = QDoubleSpinBox()
        self._x_tail_spin.setRange(0, 5000)
        self._x_tail_spin.setSuffix(" mm")
        self._x_tail_spin.setSingleStep(5)
        self._x_tail_spin.valueChanged.connect(self._on_placement_changed)
        lay.addRow("X from tail:", _with_info(self._x_tail_spin,
            "Distance from the tail of the board to the fin's leading-edge base (mm).\n"
            "Typical: side fins 100–150 mm, centre fins 300–400 mm from tail."))

        self._y_center_spin = QDoubleSpinBox()
        self._y_center_spin.setRange(-500, 500)
        self._y_center_spin.setSuffix(" mm")
        self._y_center_spin.setSingleStep(5)
        self._y_center_spin.valueChanged.connect(self._on_placement_changed)
        lay.addRow("Y offset:", _with_info(self._y_center_spin,
            "Lateral distance from the board centreline to the fin base midpoint (mm).\n"
            "Positive = toward the right rail. Centre fin = 0 mm."))

        self._cant_spin = QDoubleSpinBox()
        self._cant_spin.setRange(-20, 20)
        self._cant_spin.setSuffix(" °")
        self._cant_spin.setSingleStep(0.5)
        self._cant_spin.valueChanged.connect(self._on_placement_changed)
        lay.addRow("Cant:", _with_info(self._cant_spin,
            "Cant angle — how much the fin tilts outward from vertical (degrees).\n"
            "0° = perfectly upright. 3–5° outward is typical for side fins.\n"
            "Higher cant increases lift and looseness; lower cant adds drive.\n"
            "The 3-D preview tilts the fin to show this angle."))

        self._toe_spin = QDoubleSpinBox()
        self._toe_spin.setRange(-10, 10)
        self._toe_spin.setSuffix(" °")
        self._toe_spin.setSingleStep(0.5)
        self._toe_spin.valueChanged.connect(self._on_placement_changed)
        lay.addRow("Toe:", _with_info(self._toe_spin,
            "Toe angle — how much the fin's leading edge points inward toward the nose (degrees).\n"
            "0° = parallel to centreline. 2–4° toe-in is typical for side fins.\n"
            "More toe-in increases pivot and looseness; less adds hold and speed.\n"
            "The 3-D preview rotates the fin to show this angle."))

        return grp

    # ------------------------------------------------------------------
    # Refresh from model
    # ------------------------------------------------------------------

    def refresh_from_model(self) -> None:
        self._refresh_from_model()

    def _refresh_from_model(self) -> None:
        fins = self._model.fins
        if fins is None:
            return
        self._setup_combo.blockSignals(True)
        idx = _SETUP_LABELS.index(fins.setup_type) if fins.setup_type in _SETUP_LABELS else 0
        self._setup_combo.setCurrentIndex(idx)
        self._setup_combo.blockSignals(False)
        self._rebuild_fin_combo(fins)
        if fins.fins:
            self._load_fin(0)

    def _rebuild_fin_combo(self, fins: FinSetup) -> None:
        self._fin_combo.blockSignals(True)
        self._fin_combo.clear()
        for f in fins.fins:
            self._fin_combo.addItem(f.name)
        self._fin_combo.blockSignals(False)

    def _load_fin(self, idx: int) -> None:
        fins = self._model.fins
        if fins is None or idx < 0 or idx >= len(fins.fins):
            return
        fin = fins.fins[idx]
        self._editor.set_fin(fin)
        self._name_lbl.setText(fin.name)

        self._box_combo.blockSignals(True)
        bi = self._box_combo.findText(fin.box.box_type)
        if bi >= 0:
            self._box_combo.setCurrentIndex(bi)
        self._box_combo.blockSignals(False)

        self._foil_spin.blockSignals(True)
        self._foil_spin.setValue(fin.foil.thickness_ratio * 100.0)
        self._foil_spin.blockSignals(False)

        for spin, val in [
            (self._x_tail_spin,  fin.placement.x_from_tail_mm),
            (self._y_center_spin, fin.placement.y_from_center_mm),
            (self._cant_spin,    fin.placement.cant_deg),
            (self._toe_spin,     fin.placement.toe_deg),
        ]:
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

        self._sharp_container.setEnabled(False)
        self._inf_container.setEnabled(False)
        self._rebuild_3d()

    # ------------------------------------------------------------------
    # 3D rebuild
    # ------------------------------------------------------------------

    def _rebuild_3d(self) -> None:
        fin = self._current_fin()
        if fin is None:
            return
        mesh = build_fin_mesh(fin, n_height=40, n_chord=16)
        if mesh is not None:
            # Apply cant and toe so user sees the visual effect immediately
            mesh = apply_cant_toe(mesh,
                                   fin.placement.cant_deg,
                                   fin.placement.toe_deg)
        self._current_mesh = mesh
        if mesh is not None:
            self._gl.update_mesh(mesh)
            self._shading.on_mesh_updated()

    def _build_wireframe_lines(self) -> np.ndarray | None:
        return _fin_wireframe_from_mesh(self._current_mesh)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_fin(self) -> FinDef | None:
        fins = self._model.fins
        if fins is None:
            return None
        idx = self._fin_combo.currentIndex()
        if idx < 0 or idx >= len(fins.fins):
            return None
        return fins.fins[idx]

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_setup_changed(self, combo_idx: int) -> None:
        name = _SETUP_LABELS[combo_idx]
        factory = _SETUP_FACTORIES.get(name)
        if factory is None:
            return
        self._model.fins = factory()
        self._rebuild_fin_combo(self._model.fins)
        if self._model.fins.fins:
            self._load_fin(0)
        self._external_changed.emit()

    def _on_fin_selected(self, idx: int) -> None:
        self._load_fin(idx)

    def _on_fin_changed(self) -> None:
        self._rebuild_3d()

    def _on_point_selected(self, idx: int) -> None:
        fin = self._current_fin()
        if fin is None or idx < 0:
            self._sharp_container.setEnabled(False)
            self._inf_container.setEnabled(False)
            return
        pt = fin.points[idx]
        self._sharp_container.setEnabled(True)
        self._inf_container.setEnabled(True)

        self._sharpness_slider.blockSignals(True)
        self._sharpness_slider.setValue(int(pt.sharpness * 100))
        self._sharpness_slider.blockSignals(False)
        self._sharpness_lbl.setText(f"{int(pt.sharpness * 100)}%")

        self._influence_slider.blockSignals(True)
        self._influence_slider.setValue(int(pt.influence * 100))
        self._influence_slider.blockSignals(False)
        self._influence_lbl.setText(f"{int(pt.influence * 100)}%")

    def _on_sharpness_changed(self, val: int) -> None:
        self._sharpness_lbl.setText(f"{val}%")
        pt_idx = self._editor.selected_index()
        if pt_idx is not None:
            self._editor.set_point_sharpness(pt_idx, val / 100.0)

    def _on_influence_changed(self, val: int) -> None:
        self._influence_lbl.setText(f"{val}%")
        pt_idx = self._editor.selected_index()
        if pt_idx is not None:
            self._editor.set_point_influence(pt_idx, val / 100.0)

    def _on_box_changed(self, text: str) -> None:
        fin = self._current_fin()
        if fin is None:
            return
        from blankforge.data.fin_model import FinBox
        fin.box = FinBox.from_preset(text)
        self._rebuild_3d()

    def _on_foil_changed(self, val: float) -> None:
        fin = self._current_fin()
        if fin is None:
            return
        fin.foil.thickness_ratio = val / 100.0
        self._rebuild_3d()

    def _on_placement_changed(self) -> None:
        fin = self._current_fin()
        if fin is None:
            return
        fin.placement.x_from_tail_mm = self._x_tail_spin.value()
        fin.placement.y_from_center_mm = self._y_center_spin.value()
        fin.placement.cant_deg = self._cant_spin.value()
        fin.placement.toe_deg = self._toe_spin.value()
        # Rebuild 3D so cant/toe rotation is shown immediately
        self._rebuild_3d()

    def _apply_template(self, template_name: str) -> None:
        fin = self._current_fin()
        if fin is None:
            return
        factory = FIN_TEMPLATES.get(template_name)
        if factory is None:
            return
        fin.points = factory()  # type: ignore[operator]
        self._editor.set_fin(fin)
        self._editor._fitted = False
        self._rebuild_3d()
