from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
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
        self._rocker: float = 0.0
        # ghost: (profile, hw, ht, rocker, alpha, is_next)
        self._ghost_profiles: list[tuple[RailProfile, float, float, float, float, bool]] = []

    def set_profile(
        self,
        profile: RailProfile,
        half_width: float,
        half_thickness: float,
        rocker: float = 0.0,
        ghosts: list[tuple[RailProfile, float, float, float, float, bool]] | None = None,
    ) -> None:
        self._profile = profile
        self._half_width = half_width
        self._half_thickness = half_thickness
        self._rocker = rocker
        self._ghost_profiles = ghosts or []
        self.update()

    def _compute_scale(self) -> float:
        """Single scale shared by current + all ghosts so relative sizes are accurate."""
        w, h = self.width(), self.height()
        margin = 0.12
        max_hw = self._half_width
        max_ht = self._half_thickness
        rockers = [self._rocker]
        for g in self._ghost_profiles:
            max_hw = max(max_hw, g[1])
            max_ht = max(max_ht, g[2])
            rockers.append(g[3])
        # Vertical extent must accommodate thickest profile + rocker variation between stations
        rocker_range = max(rockers) - min(rockers)
        total_v = max_ht * 2.0 + rocker_range
        return min(
            w * (1 - 2 * margin) / (max_hw * 2.0 + 1.0),
            h * (1 - 2 * margin) / (total_v + 1.0),
        )

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(30, 32, 36))

        scale = self._compute_scale()

        # Draw scale grid for the current station, using the shared scale
        self._draw_scale_grid(p, scale)

        # Draw ghosts first (back to front so current sits on top)
        for ghost in self._ghost_profiles:
            self._draw_cross_section(p, ghost[0], ghost[1], ghost[2], ghost[3], ghost[4], ghost[5], scale)

        # Draw current station (rocker_offset = 0)
        self._draw_cross_section(p, self._profile, self._half_width, self._half_thickness,
                                 self._rocker, 1.0, None, scale)

        # Draw centerline
        cx, cy = self.width() / 2, self.height() / 2
        pen = QPen(QColor(60, 65, 75))
        pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(int(cx), 0, int(cx), self.height())
        p.drawLine(0, int(cy), self.width(), int(cy))

        # Color key (top-right)
        self._draw_color_key(p)

    def _draw_scale_grid(self, p: QPainter, scale: float) -> None:
        from PySide6.QtGui import QFontMetrics
        w, h = self.width(), self.height()
        hw = self._half_width
        ht = self._half_thickness
        t = ht * 2.0
        if hw < 1 or ht < 1:
            return

        cx, cy = w / 2, h / 2
        thickness = t

        def sx(y_mm):  return cx + y_mm * scale
        def sy(z_mm):  return cy - (z_mm - thickness / 2) * scale

        pen = QPen(QColor(80, 90, 110, 90))
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidthF(0.8)
        p.setPen(pen)

        from PySide6.QtWidgets import QApplication
        font = QApplication.font()
        p.setFont(font)
        fm = QFontMetrics(font)

        # Horizontal grid: z = 0 (hull) and z = t (deck)
        for z_mm, label in [(0.0, f"0"), (t, f"{t:.0f} mm")]:
            sy_val = sy(z_mm)
            p.drawLine(int(sx(-hw)), int(sy_val), int(sx(hw)), int(sy_val))
            p.setPen(QColor(170, 200, 230, 230))
            p.drawText(int(sx(hw)) + 4, int(sy_val) + fm.ascent() // 2, label)
            p.setPen(pen)

        # Vertical grid: y = -hw (left rail), y = 0 (centerline already drawn), y = hw (right rail)
        for y_mm, label in [(-hw, f"{hw*2:.0f} mm"), (hw, "")]:
            sx_val = sx(y_mm)
            p.drawLine(int(sx_val), int(sy(0)), int(sx_val), int(sy(t)))
            if label:
                p.setPen(QColor(170, 200, 230, 230))
                p.drawText(int(sx(-hw)) - fm.horizontalAdvance(label) - 4,
                           int(sy(t / 2)) + fm.ascent() // 2, label)
                p.setPen(pen)

    def _draw_color_key(self, p: QPainter) -> None:
        from PySide6.QtGui import QFont, QFontMetrics
        font = QFont("monospace", 8)
        p.setFont(font)
        fm = QFontMetrics(font)
        entries = [
            (QColor(60, 170, 100), "Previous"),
            (QColor(80, 140, 200), "Current"),
            (QColor(210, 190, 60), "Next"),
        ]
        pad, swatch, gap = 6, 10, 4
        line_h = max(fm.height(), swatch) + 3
        box_w = swatch + gap + max(fm.horizontalAdvance(e[1]) for e in entries) + pad * 2
        box_h = line_h * len(entries) + pad * 2
        x = self.width() - box_w - 6
        y = 6
        p.setBrush(QColor(20, 22, 28, 200))
        p.setPen(QPen(QColor(70, 80, 100, 180)))
        p.drawRoundedRect(x, y, box_w, box_h, 5, 5)
        for i, (color, label) in enumerate(entries):
            ey = y + pad + i * line_h
            p.setBrush(color)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(x + pad, ey + (line_h - swatch) // 2, swatch, swatch, 2, 2)
            p.setPen(QColor(190, 210, 240))
            p.drawText(x + pad + swatch + gap, ey + fm.ascent() + (line_h - fm.height()) // 2, label)

    def _draw_cross_section(
        self,
        p: QPainter,
        profile: RailProfile,
        hw: float,
        ht: float,
        rocker: float,
        alpha: float,
        is_next: bool | None = None,
        scale: float | None = None,
    ) -> None:
        evaluator = RailProfileEvaluator([])
        pts = evaluator.cross_section_points(0, hw, ht, n_points=64, profile=profile)
        if pts is None or len(pts) == 0:
            return

        w, h = self.width(), self.height()
        if scale is None:
            scale = self._compute_scale()
        cx, cy = w / 2, h / 2
        thickness = float(ht * 2)
        # Vertical offset from current station's centroid: higher rocker → up on screen
        rocker_offset = rocker - self._rocker

        def to_screen(y, z):
            sx = cx + y * scale
            # z=t/2 is the station's own centroid; shift by rocker delta
            sy = cy - (z - thickness / 2 + rocker_offset) * scale
            return sx, sy

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

        # Current = blue; previous = green; next = yellow
        if is_next is None:
            r, g, b = 80, 140, 200
        elif is_next:
            r, g, b = 210, 190, 60
        else:
            r, g, b = 60, 170, 100

        color = QColor(r, g, b, int(alpha * 255))
        fill = QColor(r, g, b, int(alpha * 60))
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

        layout.addWidget(QLabel("<b>Rails — Cross-Section</b>"))

        self._canvas = ProfileCanvas()
        layout.addWidget(self._canvas, stretch=1)

        # Station navigation
        nav = QHBoxLayout()
        self._prev_btn = QPushButton("← Prev")
        self._next_btn = QPushButton("Next →")
        self._station_label = QLabel("Station 0")
        self._station_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        nav.addWidget(self._station_label, stretch=1)
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._next_btn)
        layout.addLayout(nav)

        # Rail controls
        rail_box = QGroupBox("Rail Profile — Current Station")
        rail_form = QVBoxLayout(rail_box)

        copy_all_row = QHBoxLayout()
        copy_all_row.addStretch()
        self._copy_all_prev_btn = QPushButton("← All")
        self._reset_all_btn = QPushButton("↺ All")
        self._copy_all_next_btn = QPushButton("All →")
        self._copy_all_prev_btn.setToolTip("Copy all parameters from previous station")
        self._reset_all_btn.setToolTip("Reset all parameters to default")
        self._copy_all_next_btn.setToolTip("Copy all parameters from next station")
        for b in (self._copy_all_prev_btn, self._reset_all_btn, self._copy_all_next_btn):
            b.setFixedWidth(52)
        # Prev = toward tail (idx+1), Next = toward nose (idx-1) — matches tail-left / nose-right layout
        self._copy_all_prev_btn.clicked.connect(lambda: self._copy_all_from_station(+1))
        self._reset_all_btn.clicked.connect(self._reset_all_to_default)
        self._copy_all_next_btn.clicked.connect(lambda: self._copy_all_from_station(-1))
        copy_all_row.addWidget(self._copy_all_prev_btn)
        copy_all_row.addWidget(self._reset_all_btn)
        copy_all_row.addWidget(self._copy_all_next_btn)
        rail_form.addLayout(copy_all_row)

        self._apex_slider = LabeledSlider("Apex Ratio", 0.0, 1.0, decimals=2)
        self._deck_concave_slider = LabeledSlider("Deck Concave", -1.0, 1.0, decimals=2)
        self._lower_concave_slider = LabeledSlider("Lower Concave", -1.0, 1.0, decimals=2)
        self._rail_ratio_slider = LabeledSlider("Rail Ratio", 0.0, 1.0, decimals=2)
        self._softness_slider = LabeledSlider("Softness", 0.0, 1.0, decimals=2)
        for slider, field in [
            (self._apex_slider, "apex_ratio"),
            (self._deck_concave_slider, "deck_concave"),
            (self._lower_concave_slider, "lower_concave"),
            (self._rail_ratio_slider, "rail_ratio"),
            (self._softness_slider, "softness"),
        ]:
            rail_form.addWidget(self._make_slider_row(slider, field))
        layout.addWidget(rail_box)

        # Signals
        self._prev_btn.clicked.connect(self._go_prev)
        self._next_btn.clicked.connect(self._go_next)
        for sl in [self._apex_slider, self._deck_concave_slider, self._lower_concave_slider,
                   self._rail_ratio_slider, self._softness_slider]:
            sl.value_changed.connect(self._on_rail_changed)

        self.refresh_from_model()

    def refresh_from_model(self) -> None:
        self._updating = True
        n = len(self._model.curves.rail)
        self._station_idx = min(self._station_idx, max(0, n - 1))
        self._update_station_display()
        self._updating = False

    def _update_station_display(self) -> None:
        rail = self._model.curves.rail
        n = len(rail)
        if n == 0:
            self._station_label.setText("No stations")
            return

        idx = self._station_idx
        if idx == 0:
            label = f"Nose Station (1 / {n})"
        elif idx == n - 1:
            label = f"Tail Station ({n} / {n})"
        else:
            label = f"Station {idx + 1} / {n}"
        self._station_label.setText(label)

        station = rail[idx]
        prof = station.profile

        self._updating = True
        self._apex_slider.set_value(prof.apex_ratio)
        self._deck_concave_slider.set_value(prof.deck_concave)
        self._lower_concave_slider.set_value(prof.lower_concave)
        self._rail_ratio_slider.set_value(prof.rail_ratio)
        self._softness_slider.set_value(prof.softness)
        self._updating = False

        # Compute geometry for canvas
        width_eval = BoardCurveEvaluator(self._model.curves.width)
        thick_eval = BoardCurveEvaluator(self._model.curves.thickness)
        rocker_eval = BoardCurveEvaluator(self._model.curves.rocker)
        pos = station.position_mm

        hw = float(width_eval(pos))
        ht = float(thick_eval(pos))
        rk = float(rocker_eval(pos))

        # Build ghost stations (previous = darker, next = lighter)
        # Each ghost carries its own rocker so the canvas can offset it vertically
        # relative to the current station — preserving relative size and height.
        ghosts = []
        for offset in [-2, -1, 1, 2]:
            ghost_idx = idx + offset
            if 0 <= ghost_idx < n:
                ghost_s = rail[ghost_idx]
                ghost_hw = float(width_eval(ghost_s.position_mm))
                ghost_ht = float(thick_eval(ghost_s.position_mm))
                ghost_rk = float(rocker_eval(ghost_s.position_mm))
                alpha = 0.2 if abs(offset) == 2 else 0.45
                is_next = offset > 0
                ghosts.append((ghost_s.profile, ghost_hw, ghost_ht, ghost_rk, alpha, is_next))

        self._canvas.set_profile(prof, hw, ht, rk, ghosts)

    def _make_slider_row(self, slider: LabeledSlider, field: str) -> QWidget:
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(3)
        h.addWidget(slider, stretch=1)
        prev_btn = QPushButton("◀")
        prev_btn.setFixedWidth(26)
        prev_btn.setToolTip("Copy from previous station")
        reset_btn = QPushButton("↺")
        reset_btn.setFixedWidth(26)
        reset_btn.setToolTip("Reset to default")
        next_btn = QPushButton("▶")
        next_btn.setFixedWidth(26)
        next_btn.setToolTip("Copy from next station")
        # Prev = toward tail (idx+1), Next = toward nose (idx-1) — matches tail-left / nose-right layout
        prev_btn.clicked.connect(lambda _=False, s=slider, f=field: self._copy_field_from_station(+1, s, f))
        reset_btn.clicked.connect(lambda _=False, s=slider, f=field: self._reset_field(s, f))
        next_btn.clicked.connect(lambda _=False, s=slider, f=field: self._copy_field_from_station(-1, s, f))
        h.addWidget(prev_btn)
        h.addWidget(reset_btn)
        h.addWidget(next_btn)
        return container

    def _copy_field_from_station(self, offset: int, slider: LabeledSlider, field: str) -> None:
        rail = self._model.curves.rail
        src_idx = self._station_idx + offset
        if not (0 <= src_idx < len(rail)):
            return
        slider.set_value(getattr(rail[src_idx].profile, field))
        self._on_rail_changed()

    def _copy_all_from_station(self, offset: int) -> None:
        rail = self._model.curves.rail
        src_idx = self._station_idx + offset
        if not (0 <= src_idx < len(rail)):
            return
        src = rail[src_idx].profile
        self._updating = True
        self._apex_slider.set_value(src.apex_ratio)
        self._deck_concave_slider.set_value(src.deck_concave)
        self._lower_concave_slider.set_value(src.lower_concave)
        self._rail_ratio_slider.set_value(src.rail_ratio)
        self._softness_slider.set_value(src.softness)
        self._updating = False
        self._on_rail_changed()

    def _go_prev(self) -> None:
        # Prev moves toward the tail (higher index = position L = tail)
        n = len(self._model.curves.rail)
        if self._station_idx < n - 1:
            self._station_idx += 1
            self._update_station_display()

    def _go_next(self) -> None:
        # Next moves toward the nose (lower index = position 0 = nose)
        if self._station_idx > 0:
            self._station_idx -= 1
            self._update_station_display()

    def _reset_field(self, slider: LabeledSlider, field: str) -> None:
        slider.set_value(getattr(RailProfile(), field))
        self._on_rail_changed()

    def _reset_all_to_default(self) -> None:
        default = RailProfile()
        self._updating = True
        self._apex_slider.set_value(default.apex_ratio)
        self._deck_concave_slider.set_value(default.deck_concave)
        self._lower_concave_slider.set_value(default.lower_concave)
        self._rail_ratio_slider.set_value(default.rail_ratio)
        self._softness_slider.set_value(default.softness)
        self._updating = False
        self._on_rail_changed()

    def _on_rail_changed(self, *_) -> None:
        if self._updating:
            return
        rail = self._model.curves.rail
        if not rail or self._station_idx >= len(rail):
            return
        prof = rail[self._station_idx].profile
        prof.apex_ratio = self._apex_slider.value()
        prof.deck_concave = self._deck_concave_slider.value()
        prof.lower_concave = self._lower_concave_slider.value()
        prof.rail_ratio = self._rail_ratio_slider.value()
        prof.softness = self._softness_slider.value()
        self._update_station_display()
        self._model_changed.emit()
