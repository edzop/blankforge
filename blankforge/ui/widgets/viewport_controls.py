"""Reusable viewport overlay and sidebar widgets shared by all 3-D views.

ViewPresetsWidget  — floating overlay with Reset / preset / Fit buttons
ShadingControlsSidebar — collapsible sidebar: Solid / Wireframe / Heatmap sliders

Usage
-----
Both RenderedViewTab and FinsTab instantiate these; any fix or improvement in
this file is immediately reflected in both views.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QPushButton,
    QSlider, QToolButton, QVBoxLayout, QWidget,
)

from blankforge.ui.widgets.gl_viewport import GLViewport


def _apply_dark_palette(widget: QWidget, bg: str = "#1e2025", fg: str = "#c0c8d8") -> None:
    """Paint *widget* with a solid dark background using the Qt palette mechanism.

    This is more reliable than stylesheet background rules on QWidget subclasses
    because Qt's palette path is independent of the style-sheet cascade.
    """
    widget.setAutoFillBackground(True)
    pal = widget.palette()
    pal.setColor(QPalette.ColorRole.Window,     QColor(bg))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(fg))
    pal.setColor(QPalette.ColorRole.Base,       QColor(bg))
    pal.setColor(QPalette.ColorRole.Text,       QColor(fg))
    pal.setColor(QPalette.ColorRole.Button,     QColor(bg))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(fg))
    widget.setPalette(pal)


# ---------------------------------------------------------------------------
# Floating view-preset overlay
# ---------------------------------------------------------------------------

_OVERLAY_STYLE = (
    "QWidget { background: transparent; }"
    "QPushButton {"
    "  background: rgba(20,22,28,200);"
    "  border: 1px solid rgba(80,90,110,160);"
    "  color: #bdc8dc; font-size: 10px;"
    "  padding: 3px 6px; border-radius: 3px;"
    "}"
    "QPushButton:hover { background: rgba(50,65,100,220); color: #fff; }"
    "QPushButton:pressed { background: rgba(40,55,90,240); }"
)

# Standard preset lists — (label, azimuth, elevation) or tuples with a special key
BOARD_PRESETS = [
    ("Top",    0.0,  89.0),
    ("Side",   0.0,   0.5),
    ("Front", 90.0,   0.5),
]

FIN_PRESETS = [
    ("Iso",   45.0, 25.0),
    ("Top",    0.0, 89.0),
    ("Side",   0.0,  0.5),
    ("Front", 90.0,  0.5),
]


class ViewPresetsWidget(QWidget):
    """Floating overlay placed over a GLViewport (via set_controls_widget).

    Parameters
    ----------
    viewport : GLViewport
    presets  : list of (label, azimuth_deg, elevation_deg)
               Use the module-level BOARD_PRESETS / FIN_PRESETS constants or
               supply your own list.
    """

    def __init__(
        self,
        viewport: GLViewport,
        presets: list[tuple[str, float, float]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._viewport = viewport
        self._presets = presets or BOARD_PRESETS
        self.setStyleSheet(_OVERLAY_STYLE)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(2)

        reset_btn = self._btn("Reset")
        reset_btn.clicked.connect(self._viewport.reset_view)
        vbox.addWidget(reset_btn)

        for label, az, el in self._presets:
            btn = self._btn(label)
            btn.clicked.connect(lambda _, a=az, e=el: self._viewport.set_preset_view(a, e))
            vbox.addWidget(btn)

        fit_btn = self._btn("Fit")
        fit_btn.clicked.connect(self._viewport.fit_view)
        vbox.addWidget(fit_btn)

        self.adjustSize()

    def _btn(self, label: str) -> QPushButton:
        b = QPushButton(label)
        b.setFixedWidth(68)
        return b


# ---------------------------------------------------------------------------
# Shading controls sidebar
# ---------------------------------------------------------------------------

# Child-widget styling only — the container background is handled by the palette,
# not by a stylesheet background rule (which is unreliable on QWidget subclasses).
_SIDEBAR_CHILD_STYLE = (
    "QLabel    { color: #c0c8d8; border: none; background: transparent; }"
    "QCheckBox { color: #c0c8d8; border: none; background: transparent; }"
    "QGroupBox { color: #c0c8d8; border: 1px solid #383848; background: transparent; }"
    "QSlider::groove:horizontal {"
    "  background: #383848; height: 4px; border-radius: 2px;"
    "}"
    "QSlider::handle:horizontal {"
    "  background: #5878b8; width: 12px; height: 12px; margin: -4px 0;"
    "  border-radius: 6px;"
    "}"
)


class ShadingControlsSidebar(QWidget):
    """Collapsible sidebar: Solid / Wireframe / Heatmap sliders.

    Parameters
    ----------
    viewport              : GLViewport to control
    wireframe_builder     : Callable[[], np.ndarray | None]
                            Called whenever the wireframe needs to be rebuilt.
                            Should return a flat (N, 3) float32 array of line-
                            segment vertex pairs, or None to clear.
    show_wireframe_detail : If True, adds Longitudinal / Latitudinal / Crosshatch
                            checkboxes and a Density slider (used by the board view).
    initial_solid         : Starting value for the Solid Opacity slider (0–100).
    """

    def __init__(
        self,
        viewport: GLViewport,
        wireframe_builder: Callable[[], np.ndarray | None] | None = None,
        show_wireframe_detail: bool = False,
        initial_solid: int = 100,
        thickness_axis: str = "z",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._viewport = viewport
        self._wireframe_builder = wireframe_builder
        self._show_detail = show_wireframe_detail
        self._thickness_axis = thickness_axis
        self.setFixedWidth(220)
        # Dark background via palette — reliable across all Qt styles/themes
        _apply_dark_palette(self)
        # Style text colour and sliders for all descendants
        self.setStyleSheet(_SIDEBAR_CHILD_STYLE)
        self._build(initial_solid)

    # ------------------------------------------------------------------
    # Public properties (board tab reads these in its wireframe builder)
    # ------------------------------------------------------------------

    @property
    def wf_long(self) -> bool:
        return self._wf_long.isChecked() if self._show_detail else True

    @property
    def wf_lat(self) -> bool:
        return self._wf_lat.isChecked() if self._show_detail else True

    @property
    def wf_cross(self) -> bool:
        return self._wf_cross.isChecked() if self._show_detail else False

    @property
    def wf_density(self) -> int:
        return self._wf_density.value() if self._show_detail else 12

    @property
    def hm_sensitivity(self) -> float:
        return self._hm_sensitivity.value() / 100.0

    @property
    def solid_value(self) -> int:
        return self._solid_sl.value()

    @property
    def wf_value(self) -> int:
        return self._wf_sl.value()

    @property
    def hm_value(self) -> int:
        return self._hm_sl.value()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def set_wireframe_builder(self, fn: Callable[[], np.ndarray | None]) -> None:
        self._wireframe_builder = fn

    def prepend_section(self, widget: QWidget) -> None:
        """Insert *widget* (plus a separator) before the Surface section."""
        sep = self._sep()
        # _vbox is set at the end of _build(); insert at top (index 0, 1)
        self._vbox.insertWidget(0, sep)
        self._vbox.insertWidget(0, widget)

    def on_mesh_updated(self) -> None:
        """Re-apply active shading after the mesh has been swapped."""
        if self._wf_sl.value() > 0:
            self._rebuild_wireframe()
        if self._hm_sl.value() > 0:
            self._viewport.set_heatmap_blend(
                self._hm_sl.value() / 100.0,
                self.hm_sensitivity,
                self.hm_mode,
                self._thickness_axis,
            )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _sep(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(1)
        w.setAutoFillBackground(True)
        pal = w.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#383848"))
        w.setPalette(pal)
        return w

    def _slider_row(
        self, label: str, lo: int, hi: int, val: int, fmt: str = "{}"
    ) -> tuple[QSlider, QLabel, QHBoxLayout]:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setMinimum(lo)
        sl.setMaximum(hi)
        sl.setValue(val)
        row.addWidget(sl, stretch=1)
        lbl = QLabel(fmt.format(val))
        lbl.setFixedWidth(32)
        row.addWidget(lbl)
        return sl, lbl, row

    def _build(self, initial_solid: int) -> None:
        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(10, 10, 10, 10)
        vbox.setSpacing(8)
        vbox.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._vbox = vbox  # exposed for prepend_section()

        # ── Surface ───────────────────────────────────────────────────
        vbox.addWidget(QLabel("<b>Surface</b>"))
        self._solid_sl, self._solid_lbl, solid_row = \
            self._slider_row("Opacity:", 0, 100, initial_solid, "{}%")
        self._solid_lbl.setText(f"{initial_solid}%")
        vbox.addLayout(solid_row)
        self._solid_sl.valueChanged.connect(self._on_solid_changed)

        vbox.addWidget(self._sep())

        # ── Wireframe ─────────────────────────────────────────────────
        vbox.addWidget(QLabel("<b>Wireframe</b>"))
        self._wf_sl, self._wf_sl_lbl, wf_row = \
            self._slider_row("Intensity:", 0, 100, 0, "{}%")
        self._wf_sl_lbl.setText("0%")
        vbox.addLayout(wf_row)
        self._wf_sl.valueChanged.connect(self._on_wf_intensity_changed)

        # Optional board-specific detail controls
        if self._show_detail:
            self._wf_detail = QWidget()
            wf_det_lay = QVBoxLayout(self._wf_detail)
            wf_det_lay.setContentsMargins(0, 2, 0, 0)
            wf_det_lay.setSpacing(4)

            self._wf_long = QCheckBox("Longitudinal")
            self._wf_long.setChecked(True)
            self._wf_lat = QCheckBox("Latitudinal")
            self._wf_lat.setChecked(True)
            self._wf_cross = QCheckBox("Crosshatch")
            self._wf_cross.setChecked(False)
            wf_det_lay.addWidget(self._wf_long)
            wf_det_lay.addWidget(self._wf_lat)
            wf_det_lay.addWidget(self._wf_cross)

            self._wf_density, self._wf_density_label, density_row = \
                self._slider_row("Density:", 4, 40, 12)
            wf_det_lay.addLayout(density_row)

            self._wf_long.toggled.connect(self._on_wf_settings_changed)
            self._wf_lat.toggled.connect(self._on_wf_settings_changed)
            self._wf_cross.toggled.connect(self._on_wf_settings_changed)
            self._wf_density.valueChanged.connect(self._on_density_changed)

            vbox.addWidget(self._wf_detail)
            self._wf_detail.setVisible(False)
        else:
            # Stubs so property accessors don't crash
            self._wf_long = _StubCheckBox(True)
            self._wf_lat = _StubCheckBox(True)
            self._wf_cross = _StubCheckBox(False)
            self._wf_density = _StubSlider(12)

        vbox.addWidget(self._sep())

        # ── Heatmap ───────────────────────────────────────────────────
        vbox.addWidget(QLabel("<b>Heatmap</b>"))

        from PySide6.QtWidgets import QComboBox
        mode_row_w = QWidget()
        mode_row = QHBoxLayout(mode_row_w)
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.addWidget(QLabel("Mode:"))
        self._hm_mode = QComboBox()
        self._hm_mode.addItems(["Curvature", "Thickness"])
        mode_row.addWidget(self._hm_mode, stretch=1)
        vbox.addWidget(mode_row_w)
        self._hm_mode.currentIndexChanged.connect(self._on_hm_mode_changed)

        self._hm_sl, self._hm_sl_lbl, hm_row = \
            self._slider_row("Intensity:", 0, 100, 0, "{}%")
        self._hm_sl_lbl.setText("0%")
        vbox.addLayout(hm_row)
        self._hm_sl.valueChanged.connect(self._on_hm_intensity_changed)

        self._hm_detail = QWidget()
        hm_det_lay = QVBoxLayout(self._hm_detail)
        hm_det_lay.setContentsMargins(0, 2, 0, 0)
        hm_det_lay.setSpacing(4)
        self._hm_sensitivity, self._hm_sensitivity_label, sens_row = \
            self._slider_row("Sensitivity:", 1, 100, 100)
        hm_det_lay.addLayout(sens_row)
        self._hm_sensitivity.valueChanged.connect(self._on_hm_sensitivity_changed)
        vbox.addWidget(self._hm_detail)
        self._hm_detail.setVisible(False)

        vbox.addStretch()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_solid_changed(self, val: int) -> None:
        self._solid_lbl.setText(f"{val}%")
        self._viewport.set_solid_alpha(val / 100.0)

    def _on_wf_intensity_changed(self, val: int) -> None:
        self._wf_sl_lbl.setText(f"{val}%")
        self._viewport.set_line_alpha(val / 100.0)
        if self._show_detail:
            self._wf_detail.setVisible(val > 0)
        if val > 0:
            self._rebuild_wireframe()
        else:
            self._viewport.update_wireframe_lines(None)

    def _on_wf_settings_changed(self) -> None:
        if self._wf_sl.value() > 0:
            self._rebuild_wireframe()

    def _on_density_changed(self, val: int) -> None:
        self._wf_density_label.setText(str(val))
        if self._wf_sl.value() > 0:
            self._rebuild_wireframe()

    @property
    def hm_mode(self) -> str:
        return self._hm_mode.currentText().lower()

    def _on_hm_mode_changed(self) -> None:
        if self._hm_sl.value() > 0:
            self._viewport.set_heatmap_blend(
                self._hm_sl.value() / 100.0, self.hm_sensitivity,
                self.hm_mode, self._thickness_axis,
            )

    def _on_hm_intensity_changed(self, val: int) -> None:
        self._hm_sl_lbl.setText(f"{val}%")
        self._viewport.set_heatmap_blend(
            val / 100.0, self.hm_sensitivity, self.hm_mode, self._thickness_axis
        )
        self._hm_detail.setVisible(val > 0)

    def _on_hm_sensitivity_changed(self, val: int) -> None:
        self._hm_sensitivity_label.setText(f"{val / 100:.2f}")
        if self._hm_sl.value() > 0:
            self._viewport.set_heatmap_blend(
                self._hm_sl.value() / 100.0, val / 100.0,
                self.hm_mode, self._thickness_axis,
            )

    def _rebuild_wireframe(self) -> None:
        if self._wireframe_builder is None:
            return
        verts = self._wireframe_builder()
        if verts is not None and len(verts) > 0:
            self._viewport.update_wireframe_lines(verts)
        else:
            self._viewport.update_wireframe_lines(None)


# ---------------------------------------------------------------------------
# Toggle strip (shared between tabs)
# ---------------------------------------------------------------------------

class SidebarToggleStrip(QWidget):
    """Narrow strip between viewport and sidebar with a collapse/expand button."""

    def __init__(self, sidebar: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sidebar = sidebar
        self.setFixedWidth(18)
        _apply_dark_palette(self, bg="#1a1c22")
        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addStretch()
        self._btn = QToolButton()
        self._btn.setText("»")
        self._btn.setFixedSize(18, 40)
        self._btn.setStyleSheet(
            "QToolButton { background: #252830; border: none; color: #7a8aaa; font-size: 11px; }"
            "QToolButton:hover { background: #2e3340; color: #c0cce0; }"
        )
        self._btn.setToolTip("Collapse settings panel")
        self._btn.clicked.connect(self._toggle)
        vbox.addWidget(self._btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        vbox.addStretch()

    def _toggle(self) -> None:
        visible = not self._sidebar.isVisible()
        self._sidebar.setVisible(visible)
        self._btn.setText("»" if visible else "«")
        self._btn.setToolTip(
            "Collapse settings panel" if visible else "Expand settings panel"
        )


# ---------------------------------------------------------------------------
# Internal stub helpers (avoid AttributeError when detail=False)
# ---------------------------------------------------------------------------

class _StubCheckBox:
    def __init__(self, default: bool) -> None:
        self._v = default
    def isChecked(self) -> bool:
        return self._v


class _StubSlider:
    def __init__(self, default: int) -> None:
        self._v = default
    def value(self) -> int:
        return self._v
