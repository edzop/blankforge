from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QSlider, QVBoxLayout, QWidget,
)

from blankforge.data.model import BoardModel, RailProfile, RailStation
from blankforge.geometry.curves import BoardCurveEvaluator, RailProfileEvaluator
from blankforge.ui.widgets.value_sliders import LabeledSlider


class ProfileCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(300, 200)
        self._profile: RailProfile = RailProfile()
        self._half_width: float = 255.0
        self._half_thickness: float = 30.0
        self._ghost_profiles: list[tuple[RailProfile, float, float, float]] = []  # (profile, hw, ht, alpha)

    def set_profile(
        self,
        profile: RailProfile,
        half_width: float,
        half_thickness: float,
        ghosts: list[tuple[RailProfile, float, float, float]] | None = None,
    ) -> None:
        self._profile = profile
        self._half_width = half_width
        self._half_thickness = half_thickness
        self._ghost_profiles = ghosts or []
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(30, 32, 36))

        # Draw ghosts first
        for ghost_profile, ghost_hw, ghost_ht, alpha in self._ghost_profiles:
            self._draw_cross_section(p, ghost_profile, ghost_hw, ghost_ht, alpha)

        # Draw current station
        self._draw_cross_section(p, self._profile, self._half_width, self._half_thickness, 1.0)

        # Draw centerline
        cx, cy = self.width() / 2, self.height() / 2
        pen = QPen(QColor(60, 65, 75))
        pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(int(cx), 0, int(cx), self.height())
        p.drawLine(0, int(cy), self.width(), int(cy))

    def _draw_cross_section(
        self,
        p: QPainter,
        profile: RailProfile,
        hw: float,
        ht: float,
        alpha: float,
    ) -> None:
        evaluator = RailProfileEvaluator([])
        pts = evaluator.cross_section_points(0, hw, ht, n_points=64)
        if pts is None or len(pts) == 0:
            return

        # Scale to widget
        w, h = self.width(), self.height()
        margin = 0.12
        scale = min(w * (1 - 2 * margin) / (hw * 2 + 1), h * (1 - 2 * margin) / (ht * 2 + 1))
        cx, cy = w / 2, h / 2

        thickness = float(ht * 2)

        def to_screen(y, z):
            sx = cx + y * scale
            sy = cy - (z - thickness / 2) * scale
            return sx, sy

        # Build full cross-section (right + mirrored left)
        right = pts
        left = pts[::-1].copy()
        left[:, 0] *= -1

        all_pts = np.vstack([right, left])

        path = QPainterPath()
        for i, (y, z) in enumerate(all_pts):
            sx, sy = to_screen(y, z)
            if i == 0:
                path.moveTo(sx, sy)
            else:
                path.lineTo(sx, sy)
        path.closeSubpath()

        color = QColor(80, 140, 200, int(alpha * 255))
        fill = QColor(60, 100, 160, int(alpha * 80))
        p.fillPath(path, fill)
        pen = QPen(color)
        pen.setWidthF(2.0 if alpha >= 1.0 else 1.0)
        p.setPen(pen)
        p.drawPath(path)


class ProfileViewTab(QWidget):
    def __init__(self, model: BoardModel, model_changed: Signal, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._model_changed = model_changed
        self._updating = False
        self._station_idx = 0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(QLabel("<b>Profile View — Rail Cross-Section</b>"))

        self._canvas = ProfileCanvas()
        layout.addWidget(self._canvas, stretch=1)

        # Station navigation
        nav = QHBoxLayout()
        self._prev_btn = QPushButton("← Prev")
        self._next_btn = QPushButton("Next →")
        self._station_label = QLabel("Station 0")
        self._station_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._slider, stretch=1)
        nav.addWidget(self._station_label)
        nav.addWidget(self._next_btn)
        layout.addLayout(nav)

        # Rail controls
        rail_box = QGroupBox("Rail Profile — Current Station")
        rail_form = QVBoxLayout(rail_box)
        self._apex_slider = LabeledSlider("Apex Ratio", 0.0, 1.0, decimals=2)
        self._concave_slider = LabeledSlider("Upper Concave", -1.0, 1.0, decimals=2)
        self._angle_slider = LabeledSlider("Rail Angle (°)", 0.0, 90.0, decimals=1)
        self._softness_slider = LabeledSlider("Softness", 0.0, 1.0, decimals=2)
        for w in [self._apex_slider, self._concave_slider, self._angle_slider, self._softness_slider]:
            rail_form.addWidget(w)
        layout.addWidget(rail_box)

        # Signals
        self._prev_btn.clicked.connect(self._go_prev)
        self._next_btn.clicked.connect(self._go_next)
        self._slider.valueChanged.connect(self._on_station_slider)
        self._apex_slider.value_changed.connect(self._on_rail_changed)
        self._concave_slider.value_changed.connect(self._on_rail_changed)
        self._angle_slider.value_changed.connect(self._on_rail_changed)
        self._softness_slider.value_changed.connect(self._on_rail_changed)

        self.refresh_from_model()

    def refresh_from_model(self) -> None:
        self._updating = True
        n = len(self._model.curves.rail)
        self._slider.setMaximum(max(0, n - 1))
        self._station_idx = min(self._station_idx, max(0, n - 1))
        self._slider.setValue(self._station_idx)
        self._update_station_display()
        self._updating = False

    def _update_station_display(self) -> None:
        rail = self._model.curves.rail
        n = len(rail)
        if n == 0:
            self._station_label.setText("No stations")
            return

        idx = self._station_idx
        self._station_label.setText(f"Station {idx + 1} / {n}")

        station = rail[idx]
        prof = station.profile

        self._updating = True
        self._apex_slider.set_value(prof.apex_ratio)
        self._concave_slider.set_value(prof.upper_concave)
        self._angle_slider.set_value(prof.lower_rail_angle)
        self._softness_slider.set_value(prof.softness)
        self._updating = False

        # Compute geometry for canvas
        width_eval = BoardCurveEvaluator(self._model.curves.width)
        thick_eval = BoardCurveEvaluator(self._model.curves.thickness)
        pos = station.position_mm

        hw = float(width_eval(pos))
        ht = float(thick_eval(pos))

        # Build ghost stations
        ghosts = []
        for offset in [-2, -1, 1, 2]:
            ghost_idx = idx + offset
            if 0 <= ghost_idx < n:
                ghost_s = rail[ghost_idx]
                ghost_hw = float(width_eval(ghost_s.position_mm))
                ghost_ht = float(thick_eval(ghost_s.position_mm))
                alpha = 0.2 if abs(offset) == 2 else 0.45
                ghosts.append((ghost_s.profile, ghost_hw, ghost_ht, alpha))

        self._canvas.set_profile(prof, hw, ht, ghosts)

    def _go_prev(self) -> None:
        if self._station_idx > 0:
            self._station_idx -= 1
            self._slider.setValue(self._station_idx)
            self._update_station_display()

    def _go_next(self) -> None:
        n = len(self._model.curves.rail)
        if self._station_idx < n - 1:
            self._station_idx += 1
            self._slider.setValue(self._station_idx)
            self._update_station_display()

    def _on_station_slider(self, val: int) -> None:
        self._station_idx = val
        self._update_station_display()

    def _on_rail_changed(self, *_) -> None:
        if self._updating:
            return
        rail = self._model.curves.rail
        if not rail or self._station_idx >= len(rail):
            return
        prof = rail[self._station_idx].profile
        prof.apex_ratio = self._apex_slider.value()
        prof.upper_concave = self._concave_slider.value()
        prof.lower_rail_angle = self._angle_slider.value()
        prof.softness = self._softness_slider.value()
        self._update_station_display()
        self._model_changed.emit()
