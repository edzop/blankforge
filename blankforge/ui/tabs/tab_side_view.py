from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import PchipInterpolator
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from blankforge.data.model import BoardModel, ControlPoint, RailStation
from blankforge.geometry.curves import BoardCurveEvaluator, RailProfileEvaluator, resolve_thickness_curve
from blankforge.ui.widgets.value_sliders import LabeledSlider

HIT_RADIUS_PX = 8
MIN_POINTS = 2

# 4-tuple type: (pos_mm, hull_rocker_mm, full_thickness_mm, thickness_mode)
# hull_rocker = Z of the board's bottom surface at this station
# thickness_mode is "fixed" or "ratio"
_Pt = tuple[float, float, float, str]


def _sync_rail_stations(model: BoardModel) -> None:
    evaluator = RailProfileEvaluator(model.curves.rail)
    positions = sorted({pt.position_mm for pt in model.curves.rocker.points})
    model.curves.rail = [
        RailStation(position_mm=pos, profile=evaluator.at(pos))
        for pos in positions
    ]


def _interp(xs: list[float], ys: list[float], x_eval: np.ndarray) -> np.ndarray:
    if len(xs) < 2:
        return np.full(len(x_eval), ys[0] if ys else 0.0)
    result = PchipInterpolator(xs, ys, extrapolate=False)(x_eval)
    lo, hi = ys[0], ys[-1]
    result = np.where(np.isnan(result), np.where(x_eval < xs[0], lo, hi), result)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

class SideProfileCanvas(QWidget):
    point_selected = Signal(int)
    point_changed = Signal()
    drag_finished = Signal()
    point_added = Signal(float, float, float)   # pos_mm, hull_rocker_mm, full_thickness_mm
    point_deleted = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pts: list[_Pt] = []
        self._board_length: float = 1880.0
        self._selected_pos: float | None = None
        self._drag_start_mouse: tuple[float, float] | None = None
        self._drag_start_vals: tuple[float, float] | None = None
        self._drag_active: bool = False
        self.setMinimumSize(400, 180)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data(self, pts: list[_Pt], board_length: float) -> None:
        self._pts = sorted(pts, key=lambda p: p[0])
        self._board_length = board_length
        self.update()

    @property
    def selected_index(self) -> int:
        if self._selected_pos is None or not self._pts:
            return -1
        return self._nearest_idx(self._selected_pos)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _scales(self) -> tuple[float, float]:
        # Proportional rendering: same px/mm in both axes so thickness/length
        # visually matches the actual ratio (and matches the 3D rendered view).
        ml, mt, mr, mb = self._margins()
        W = max(1.0, self.width() - ml - mr)
        H = max(1.0, self.height() - mt - mb)
        sx_fit = W / max(self._board_length, 1.0)
        tops = [pt[1] + pt[2] for pt in self._pts] if self._pts else [80.0]
        y_range = max(max(tops) * 1.15, 1.0)
        sy_fit = H / y_range
        scale = min(sx_fit, sy_fit)  # whichever axis constrains, both share it
        return scale, scale

    def _margins(self) -> tuple[int, int, int, int]:
        return 40, 16, 16, 48  # left, top, right, bottom

    def _content_bottom(self) -> float:
        # Center the (typically thin) board profile vertically in the canvas
        ml, mt, mr, mb = self._margins()
        scale, _ = self._scales()
        tops = [pt[1] + pt[2] for pt in self._pts] if self._pts else [80.0]
        content_h = max(tops) * scale * 1.15
        available = self.height() - mt - mb
        top_pad = max(0.0, (available - content_h) / 2)
        return self.height() - mb - top_pad

    def _to_screen(self, pos_mm: float, z_mm: float) -> tuple[float, float]:
        ml = self._margins()[0]
        sx, sy = self._scales()
        base_y = self._content_bottom()
        return ml + (self._board_length - pos_mm) * sx, base_y - z_mm * sy

    def _from_screen(self, px: float, py: float) -> tuple[float, float]:
        ml = self._margins()[0]
        sx, sy = self._scales()
        base_y = self._content_bottom()
        return self._board_length - (px - ml) / sx, (base_y - py) / sy

    def _nearest_idx(self, pos_mm: float) -> int:
        return min(range(len(self._pts)), key=lambda i: abs(self._pts[i][0] - pos_mm))

    def _hit_test(self, mx: float, my: float) -> int:
        for i, pt in enumerate(self._pts):
            sx, sy = self._to_screen(pt[0], pt[1])
            if math.hypot(mx - sx, my - sy) <= HIT_RADIUS_PX:
                return i
        return -1

    def _is_endpoint(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self._pts):
            return False
        pos = self._pts[idx][0]
        return abs(pos) < 0.5 or abs(pos - self._board_length) < 0.5

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(28, 30, 36))

        if len(self._pts) < 2:
            return

        xs = [pt[0] for pt in self._pts]
        hull_rockers = [pt[1] for pt in self._pts]
        thicks = [pt[2] for pt in self._pts]

        n = max(120, len(self._pts) * 25)
        x_eval = np.linspace(xs[0], xs[-1], n)
        r_eval = _interp(xs, hull_rockers, x_eval)
        t_eval = _interp(xs, thicks, x_eval)
        hull_eval = r_eval
        deck_eval = r_eval + t_eval

        self._draw_grid(p, hull_eval, deck_eval, x_eval)
        self._draw_profile(p, x_eval, r_eval, hull_eval, deck_eval)
        self._draw_handles(p)
        self._draw_overlay(p, r_eval, t_eval)

    def _draw_grid(self, p: QPainter, hull_eval, deck_eval, x_eval) -> None:
        pen = QPen(QColor(65, 75, 95, 100))
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidthF(0.7)
        p.setPen(pen)

        from PySide6.QtWidgets import QApplication
        font = QApplication.font()
        p.setFont(font)
        fm = QFontMetrics(font)

        max_deck = float(deck_eval.max())
        sx_nose, sy_hull = self._to_screen(0.0, 0.0)
        sx_tail, _ = self._to_screen(self._board_length, 0.0)
        _, sy_deck = self._to_screen(0.0, max_deck)
        sx_left = min(sx_nose, sx_tail)

        # Top label (max thickness): above its line; bottom label ("0"): below its line.
        # Both end up outside the board profile so they don't overlap each other or the curve.
        for sy, label, above in [(sy_deck, f"{max_deck:.0f} mm", True), (sy_hull, "0", False)]:
            p.setPen(pen)
            p.drawLine(int(sx_nose), int(sy), int(sx_tail), int(sy))
            p.setPen(QColor(170, 200, 230, 230))
            label_y = int(sy) - 3 if above else int(sy) + fm.ascent() + 3
            p.drawText(int(sx_left) + 2, label_y, label)

        pen2 = QPen(QColor(65, 75, 95, 80))
        pen2.setStyle(Qt.PenStyle.DashLine)
        pen2.setWidthF(0.7)
        # Nose/Tail labels stacked below the "0" label so nothing overlaps
        nose_tail_y = int(sy_hull) + fm.ascent() * 2 + 10
        for x_mm, lbl in [(0.0, "Nose"), (self._board_length, "Tail")]:
            sxv, _ = self._to_screen(x_mm, 0)
            p.setPen(pen2)
            p.drawLine(int(sxv), int(sy_deck) - 4, int(sxv), int(sy_hull) + 4)
            p.setPen(QColor(170, 200, 230, 230))
            p.drawText(int(sxv) - fm.horizontalAdvance(lbl) // 2, nose_tail_y, lbl)

    def _draw_profile(self, p: QPainter, x_eval, r_eval, hull_eval, deck_eval) -> None:
        path = QPainterPath()
        for i in range(len(x_eval)):
            sx, sy = self._to_screen(float(x_eval[i]), float(hull_eval[i]))
            path.moveTo(sx, sy) if i == 0 else path.lineTo(sx, sy)
        for i in range(len(x_eval) - 1, -1, -1):
            sx, sy = self._to_screen(float(x_eval[i]), float(deck_eval[i]))
            path.lineTo(sx, sy)
        path.closeSubpath()
        p.fillPath(path, QColor(80, 130, 200, 110))

        pen = QPen(QColor(60, 190, 150))
        pen.setWidthF(1.4)
        p.setPen(pen)
        for z_eval in (hull_eval, deck_eval):
            surf_path = QPainterPath()
            for i in range(len(x_eval)):
                sx, sy = self._to_screen(float(x_eval[i]), float(z_eval[i]))
                surf_path.moveTo(sx, sy) if i == 0 else surf_path.lineTo(sx, sy)
            p.drawPath(surf_path)

        rocker_path = QPainterPath()
        for i in range(len(x_eval)):
            sx, sy = self._to_screen(float(x_eval[i]), float(r_eval[i]))
            rocker_path.moveTo(sx, sy) if i == 0 else rocker_path.lineTo(sx, sy)
        pen.setColor(QColor(80, 140, 230))
        pen.setWidthF(1.8)
        p.setPen(pen)
        p.drawPath(rocker_path)

    def _draw_handles(self, p: QPainter) -> None:
        sel_idx = self.selected_index
        for i, pt in enumerate(self._pts):
            sx, sy = self._to_screen(pt[0], pt[1])
            is_ep = self._is_endpoint(i)
            if i == sel_idx:
                color = QColor(100, 225, 140)
                border = QColor(180, 255, 200)
                r = 7.0
            elif is_ep:
                color = QColor(180, 130, 60)
                border = QColor(220, 170, 80)
                r = 5.0
            else:
                color = QColor(200, 165, 50)
                border = QColor(240, 210, 80)
                r = 5.0
            p.setBrush(color)
            p.setPen(QPen(border, 1.5 if i == sel_idx else 1.0))
            from PySide6.QtCore import QPointF
            p.drawEllipse(QPointF(sx, sy), r, r)

    def _draw_overlay(self, p: QPainter, r_eval, t_eval) -> None:
        font = QFont("monospace", 8)
        p.setFont(font)
        p.setPen(QColor(160, 180, 210, 200))
        max_r = float(r_eval.max())
        max_t = float(t_eval.max())
        lines = [
            f"Max rocker: {max_r:.1f} mm  ({max_r/25.4:.2f} in)",
            f"Max thickness: {max_t:.1f} mm  ({max_t/25.4:.2f} in)",
        ]
        fm = QFontMetrics(font)
        x, y = 4, 12
        for line in lines:
            p.drawText(x, y, line)
            y += fm.height() + 1

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        mx, my = event.position().x(), event.position().y()
        idx = self._hit_test(mx, my)
        if idx >= 0:
            self._selected_pos = self._pts[idx][0]
            self._drag_start_mouse = (mx, my)
            self._drag_start_vals = (self._pts[idx][0], self._pts[idx][1])
            self.point_selected.emit(idx)
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start_mouse is None:
            return
        self._drag_active = True
        mx, my = event.position().x(), event.position().y()
        dx = mx - self._drag_start_mouse[0]
        dy = my - self._drag_start_mouse[1]
        sx, sy = self._scales()

        old_idx = self._nearest_idx(self._selected_pos)
        is_ep = self._is_endpoint(old_idx)

        if is_ep:
            new_pos = self._drag_start_vals[0]  # lock x for nose/tail
        else:
            new_pos = max(0.0, min(self._board_length, self._drag_start_vals[0] - dx / sx))

        new_rocker = max(0.0, self._drag_start_vals[1] - dy / sy)

        old_pt = self._pts[old_idx]
        self._pts[old_idx] = (new_pos, new_rocker, old_pt[2], old_pt[3])
        self._pts.sort(key=lambda pt: pt[0])
        self._selected_pos = new_pos

        self.point_changed.emit()
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        was_dragging = self._drag_active
        self._drag_start_mouse = None
        self._drag_start_vals = None
        self._drag_active = False
        if was_dragging:
            self.drag_finished.emit()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        mx, my = event.position().x(), event.position().y()
        for pt in self._pts:
            sx, sy = self._to_screen(pt[0], pt[1])
            if math.hypot(mx - sx, my - sy) < HIT_RADIUS_PX * 2.5:
                return
        pos, z = self._from_screen(mx, my)
        pos = max(0.001, min(self._board_length - 0.001, pos))  # never at exact nose/tail
        z = max(0.0, z)
        xs = [pt[0] for pt in self._pts]
        t_interp = _interp(xs, [pt[2] for pt in self._pts], np.array([pos]))[0]
        self.point_added.emit(pos, z, float(t_interp))

    def contextMenuEvent(self, event) -> None:
        from PySide6.QtWidgets import QMenu
        mx, my = float(event.pos().x()), float(event.pos().y())
        idx = self._hit_test(mx, my)
        if idx >= 0 and len(self._pts) > MIN_POINTS and not self._is_endpoint(idx):
            menu = QMenu(self)
            act = menu.addAction("Delete point")
            if menu.exec(event.globalPos()) == act:
                self.point_deleted.emit(idx)


# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------

class SideViewTab(QWidget):
    def __init__(self, model: BoardModel, model_changed: Signal,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._model_changed = model_changed
        self._updating = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(QLabel(
            "<b>Side View — Rocker &amp; Thickness</b>  "
            "(drag points; double-click canvas to add; right-click point to delete)"
        ))

        self._canvas = SideProfileCanvas()
        layout.addWidget(self._canvas, stretch=1)

        self._pos_slider = LabeledSlider("Position (ratio)", 0.0, 1.0, decimals=3)
        self._rocker_slider = LabeledSlider("Rocker (mm)", 0.0, 200.0, decimals=1)

        # Thickness mode row
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Thickness mode:"))
        self._thick_mode = QComboBox()
        self._thick_mode.addItems(["Fixed mm", "Ratio"])
        self._thick_mode.setFixedWidth(100)
        mode_row.addWidget(self._thick_mode)
        mode_row.addStretch()

        self._thick_slider = LabeledSlider("Thickness (mm)", 2.0, 300.0, decimals=1)

        layout.addWidget(self._pos_slider)
        layout.addWidget(self._rocker_slider)
        layout.addLayout(mode_row)
        layout.addWidget(self._thick_slider)

        self._canvas.point_selected.connect(self._on_point_selected)
        self._canvas.point_changed.connect(self._on_canvas_changed)
        self._canvas.drag_finished.connect(self._on_drag_finished)
        self._canvas.point_added.connect(self._on_point_added)
        self._canvas.point_deleted.connect(self._on_point_deleted)
        self._pos_slider.value_changed.connect(self._on_slider_changed)
        self._rocker_slider.value_changed.connect(self._on_slider_changed)
        self._thick_slider.value_changed.connect(self._on_slider_changed)
        self._thick_mode.currentIndexChanged.connect(self._on_mode_changed)

        self.refresh_from_model()

    # ------------------------------------------------------------------
    # Model <-> merged points sync
    # ------------------------------------------------------------------

    def _merged_from_model(self) -> list[_Pt]:
        """Returns (pos, hull_rocker_mm, full_thickness_mm, mode). Model stores hull rocker directly."""
        rocker_pts = self._model.curves.rocker.sorted_points()
        if not rocker_pts:
            L = self._model.parameters.length_mm
            return [(0.0, 30.0, 32.0, "fixed"), (L * 0.5, 3.0, 60.0, "fixed"), (L, 20.0, 30.0, "fixed")]

        param_t = self._model.parameters.thickness_mm
        thick_by_pos = {cp.position_mm: cp for cp in self._model.curves.thickness.sorted_points()}

        result: list[_Pt] = []
        for pt in rocker_pts:
            thick_cp = thick_by_pos.get(pt.position_mm)
            if thick_cp is None:
                # Fallback: evaluate via interpolation (fixed mode)
                thick_eval = BoardCurveEvaluator(
                    resolve_thickness_curve(self._model.curves.thickness,
                                            self._model.parameters.thickness_mm)
                )
                half_t = float(thick_eval(pt.position_mm))
                mode = "fixed"
            else:
                mode = getattr(thick_cp, "mode", "fixed")
                if mode == "ratio":
                    half_t = thick_cp.value_mm * param_t / 2.0
                else:
                    half_t = thick_cp.value_mm
            # pt.value_mm IS hull_rocker (bottom surface Z); store directly
            result.append((pt.position_mm, pt.value_mm, half_t * 2.0, mode))
        return result

    def _write_to_model(self, pts: list[_Pt]) -> None:
        """pts = (pos, hull_rocker_mm, full_thickness, mode). Model stores hull_rocker directly."""
        param_t = self._model.parameters.thickness_mm
        self._model.curves.rocker.points = [
            ControlPoint(position_mm=p[0], value_mm=p[1])
            for p in pts
        ]
        thick_points = []
        for p in pts:
            mode = p[3] if len(p) > 3 else "fixed"
            if mode == "ratio":
                value = p[2] / param_t if param_t > 0 else 1.0
            else:
                value = p[2] / 2.0
            thick_points.append(ControlPoint(position_mm=p[0], value_mm=value, mode=mode))
        self._model.curves.thickness.points = thick_points

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh_from_model(self) -> None:
        self._updating = True
        L = self._model.parameters.length_mm
        pts = self._merged_from_model()
        self._canvas.set_data(pts, L)
        if pts:
            if self._canvas._selected_pos is None:
                self._canvas._selected_pos = pts[0][0]
            idx = min(range(len(pts)), key=lambda i: abs(pts[i][0] - self._canvas._selected_pos))
            self._load_sliders(pts[idx])
        self._updating = False

    def _load_sliders(self, pt: _Pt) -> None:
        self._updating = True
        L = self._model.parameters.length_mm
        mode = pt[3] if len(pt) > 3 else "fixed"

        # Position — disabled for nose/tail
        is_endpoint = abs(pt[0]) < 0.5 or abs(pt[0] - L) < 0.5
        self._pos_slider.set_value(pt[0] / L)
        self._pos_slider.setEnabled(not is_endpoint)

        # Rocker in absolute mm — independent of thickness parameter
        self._rocker_slider.set_value(pt[1])

        # Thickness — mode-dependent
        self._thick_mode.setCurrentIndex(0 if mode == "fixed" else 1)
        if mode == "ratio":
            param_t = self._model.parameters.thickness_mm
            self._thick_slider.set_label("Thickness (ratio)")
            self._thick_slider.set_range(0.0, 1.0)
            self._thick_slider.set_value(pt[2] / param_t if param_t > 0 else 1.0)
        else:
            self._thick_slider.set_label("Thickness (mm)")
            self._thick_slider.set_range(2.0, 300.0)
            self._thick_slider.set_value(pt[2])

        self._updating = False

    # ------------------------------------------------------------------
    # Signals from canvas
    # ------------------------------------------------------------------

    def _on_point_selected(self, idx: int) -> None:
        pts = self._canvas._pts
        if 0 <= idx < len(pts):
            self._load_sliders(pts[idx])

    def _on_canvas_changed(self) -> None:
        pts = self._canvas._pts
        idx = self._canvas.selected_index
        if 0 <= idx < len(pts):
            self._updating = True
            L = self._model.parameters.length_mm
            p = pts[idx]
            self._pos_slider.set_value(p[0] / L)
            self._rocker_slider.set_value(p[1])
            self._updating = False
        self._write_to_model(pts)
        _sync_rail_stations(self._model)

    def _on_drag_finished(self) -> None:
        self._write_to_model(self._canvas._pts)
        _sync_rail_stations(self._model)
        self._model_changed.emit()

    def _on_point_added(self, pos: float, rocker: float, ht: float) -> None:
        pts = list(self._canvas._pts)
        pts.append((pos, rocker, ht, "fixed"))
        pts.sort(key=lambda p: p[0])
        self._canvas.set_data(pts, self._model.parameters.length_mm)
        self._canvas._selected_pos = pos
        self._write_to_model(pts)
        _sync_rail_stations(self._model)
        self._model_changed.emit()

    def _on_point_deleted(self, idx: int) -> None:
        pts = list(self._canvas._pts)
        if len(pts) <= MIN_POINTS:
            return
        pts.pop(idx)
        self._canvas.set_data(pts, self._model.parameters.length_mm)
        self._write_to_model(pts)
        _sync_rail_stations(self._model)
        self._model_changed.emit()

    # ------------------------------------------------------------------
    # Signals from sliders
    # ------------------------------------------------------------------

    def _on_mode_changed(self, combo_idx: int) -> None:
        if self._updating:
            return
        pts = list(self._canvas._pts)
        idx = self._canvas.selected_index
        if idx < 0 or idx >= len(pts):
            return
        mode = "ratio" if combo_idx == 1 else "fixed"
        old = pts[idx]
        pts[idx] = (old[0], old[1], old[2], mode)
        self._canvas._pts = pts

        # Reconfigure slider without triggering _on_slider_changed
        self._updating = True
        param_t = self._model.parameters.thickness_mm
        if mode == "ratio":
            self._thick_slider.set_label("Thickness (ratio)")
            self._thick_slider.set_range(0.0, 1.0)
            self._thick_slider.set_value(old[2] / param_t if param_t > 0 else 1.0)
        else:
            self._thick_slider.set_label("Thickness (mm)")
            self._thick_slider.set_range(2.0, 300.0)
            self._thick_slider.set_value(old[2])
        self._updating = False

        self._write_to_model(pts)
        _sync_rail_stations(self._model)
        self._model_changed.emit()

    def _on_slider_changed(self) -> None:
        if self._updating:
            return
        pts = list(self._canvas._pts)
        idx = self._canvas.selected_index
        if idx < 0 or idx >= len(pts):
            return

        L = self._model.parameters.length_mm
        param_t = self._model.parameters.thickness_mm
        mode = pts[idx][3] if len(pts[idx]) > 3 else "fixed"

        is_endpoint = self._canvas._is_endpoint(idx)
        if is_endpoint:
            new_pos = pts[idx][0]
        else:
            new_pos = max(0.0, min(L, self._pos_slider.value() * L))

        # Rocker slider is absolute mm — independent of thickness
        new_hull_rocker = max(0.0, self._rocker_slider.value())

        if mode == "ratio":
            new_full_t = self._thick_slider.value() * param_t
        else:
            new_full_t = self._thick_slider.value()

        pts[idx] = (new_pos, new_hull_rocker, new_full_t, mode)
        pts.sort(key=lambda p: p[0])
        self._canvas.set_data(pts, L)
        self._canvas._selected_pos = new_pos
        self._write_to_model(pts)
        _sync_rail_stations(self._model)
        self._model_changed.emit()
