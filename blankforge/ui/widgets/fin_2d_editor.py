"""2-D planform editor for a single FinDef.

Shows the fin outline (side view, looking at the planform face) with
interactive control points.  Sharpness is encoded as a colour gradient
along the outline: blue = smooth (0) → yellow = sharp (1).
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (
    QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen,
)
from PySide6.QtWidgets import QMenu, QWidget

from blankforge.data.fin_model import FinDef, FinPoint
from blankforge.geometry.fin import eval_fin_outline

HIT_RADIUS_PX = 9
POINT_RADIUS_PX = 6


def _sharpness_color(s: float) -> QColor:
    """Outline color: blue (s=0, blunt/full foil) → orange-red (s=1, knife-edge)."""
    s = max(0.0, min(1.0, s))
    r = int(40  + s * 210)   # 40  → 250
    g = int(120 - s * 100)   # 120 → 20
    b = int(230 - s * 210)   # 230 → 20
    return QColor(r, g, b)


def _influence_color(v: float) -> QColor:
    """Control-point color: blue (v=0, smooth curve) → yellow-green (v=1, sharp corner)."""
    v = max(0.0, min(1.0, v))
    r = int(20  + v * 235)
    g = int(80  + v * 155)
    b = int(220 - v * 200)
    return QColor(r, g, b)


class FinOutlineEditor(QWidget):
    """Interactive 2-D fin planform editor."""

    # Signals
    point_selected = Signal(int)   # index into fin.points
    fin_changed = Signal()         # outline was modified — rebuild mesh

    def __init__(self, fin: FinDef, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fin = fin
        self._selected_idx: int | None = None
        self._hover_idx: int | None = None
        self._drag_active = False
        self._drag_start_screen: QPointF | None = None
        self._drag_start_world: tuple[float, float] | None = None
        self._pan_start: QPointF | None = None
        self._pan_offset_start: QPointF | None = None

        # View transform: world → screen
        self._offset = QPointF(0.0, 0.0)  # pixels
        self._scale = 1.0                 # pixels per mm
        self._fitted = False

        self.setMinimumSize(280, 200)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_fin(self, fin: FinDef) -> None:
        self._fin = fin
        self._selected_idx = None
        self._fitted = False
        self.update()

    def selected_index(self) -> int | None:
        return self._selected_idx

    def set_point_sharpness(self, idx: int, sharpness: float) -> None:
        """Set foil-edge sharpness (0=blunt, 1=knife-edge) for point idx."""
        if 0 <= idx < len(self._fin.points):
            self._fin.points[idx].sharpness = float(sharpness)
            self.update()
            self.fin_changed.emit()

    def set_point_influence(self, idx: int, influence: float) -> None:
        """Set spline tension/influence (0=smooth curve, 1=sharp corner) for point idx."""
        if 0 <= idx < len(self._fin.points):
            self._fin.points[idx].influence = float(influence)
            self.update()
            self.fin_changed.emit()

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def _fit(self) -> None:
        pts = self._fin.points
        if not pts:
            self._scale = 1.0
            self._offset = QPointF(40.0, self.height() - 40.0)
            self._fitted = True
            return
        xs = [p.x_mm for p in pts]
        ys = [p.y_mm for p in pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_span = (x_max - x_min) or 1.0
        y_span = (y_max - y_min) or 1.0
        margin = 0.12
        w, h = self.width(), self.height()
        sx = w * (1 - 2 * margin) / x_span
        sy = h * (1 - 2 * margin) / y_span
        self._scale = min(sx, sy)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        # Y is flipped: world Y+ = up = screen Y-
        self._offset = QPointF(
            w / 2 - cx * self._scale,
            h / 2 + cy * self._scale,
        )
        self._fitted = True

    def _to_screen(self, x_mm: float, y_mm: float) -> QPointF:
        return QPointF(
            x_mm * self._scale + self._offset.x(),
            -y_mm * self._scale + self._offset.y(),
        )

    def _to_world(self, sx: float, sy: float) -> tuple[float, float]:
        x_mm = (sx - self._offset.x()) / self._scale
        y_mm = -(sy - self._offset.y()) / self._scale
        return x_mm, y_mm

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        if not self._fitted:
            self._fit()

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Dark background
        p.fillRect(self.rect(), QColor(22, 24, 30))

        # Grid
        self._draw_grid(p)

        pts = self._fin.points
        if len(pts) >= 2:
            self._draw_outline(p, pts)
            self._draw_base(p, pts)

        self._draw_control_points(p, pts)
        self._draw_labels(p)

    def _draw_grid(self, p: QPainter) -> None:
        w, h = self.width(), self.height()
        p.setPen(QPen(QColor(40, 44, 54), 1))
        # 10 mm grid
        step = 10.0
        sx0, _ = self._to_world(0, 0)
        sx1, _ = self._to_world(w, 0)
        xstart = math.floor(sx0 / step) * step
        x = xstart
        while True:
            sx = self._to_screen(x, 0).x()
            if sx > w:
                break
            if 0 <= sx <= w:
                p.drawLine(int(sx), 0, int(sx), h)
            x += step
        _, sy0 = self._to_world(0, 0)
        _, sy1 = self._to_world(0, h)
        y_lo, y_hi = min(sy0, sy1), max(sy0, sy1)
        ystart = math.floor(y_lo / step) * step
        y = ystart
        while True:
            pt = self._to_screen(0, y)
            if pt.y() < 0:
                break
            if 0 <= pt.y() <= h:
                p.drawLine(0, int(pt.y()), w, int(pt.y()))
            y += step

    def _draw_outline(self, p: QPainter, pts: list[FinPoint]) -> None:
        """Draw the spline outline; line colour = sharpness (foil thinness)."""
        outline = eval_fin_outline(self._fin, samples_per_segment=20)
        if len(outline) < 2:
            return

        n_ctrl = len(pts)
        spline_per_seg = 20
        n_pts = len(outline)

        seg_count = n_ctrl - 1
        for seg in range(seg_count):
            i_start = seg * spline_per_seg
            i_end   = min(i_start + spline_per_seg + 1, n_pts)
            s0 = pts[seg].sharpness
            s1 = pts[seg + 1].sharpness
            for k in range(i_start, i_end - 1):
                t = (k - i_start) / max(spline_per_seg, 1)
                col = _sharpness_color(s0 + t * (s1 - s0))
                p.setPen(QPen(col, 2.5))
                a = self._to_screen(outline[k, 0], outline[k, 1])
                b = self._to_screen(outline[k + 1, 0], outline[k + 1, 1])
                p.drawLine(a, b)

    def _draw_base(self, p: QPainter, pts: list[FinPoint]) -> None:
        """Draw the flat base line connecting first and last points."""
        a = self._to_screen(pts[0].x_mm, pts[0].y_mm)
        b = self._to_screen(pts[-1].x_mm, pts[-1].y_mm)
        p.setPen(QPen(QColor(120, 130, 145), 1.5, Qt.PenStyle.DashLine))
        p.drawLine(a, b)

    def _draw_control_points(self, p: QPainter, pts: list[FinPoint]) -> None:
        for i, pt in enumerate(pts):
            sp = self._to_screen(pt.x_mm, pt.y_mm)
            is_sel = (i == self._selected_idx)
            is_hov = (i == self._hover_idx)

            # Control point fill colour = influence (spline tension)
            col = _influence_color(pt.influence)
            if is_sel:
                p.setPen(QPen(QColor(255, 255, 255), 2.0))
                p.setBrush(col)
                p.drawEllipse(sp, POINT_RADIUS_PX + 2, POINT_RADIUS_PX + 2)
            elif is_hov:
                p.setPen(QPen(QColor(200, 210, 220), 1.5))
                p.setBrush(col.lighter(130))
                p.drawEllipse(sp, POINT_RADIUS_PX + 1, POINT_RADIUS_PX + 1)
            else:
                p.setPen(QPen(QColor(140, 150, 165), 1.0))
                p.setBrush(col)
                p.drawEllipse(sp, POINT_RADIUS_PX, POINT_RADIUS_PX)

    def _draw_labels(self, p: QPainter) -> None:
        p.setFont(QFont("sans-serif", 8))
        p.setPen(QColor(100, 110, 125))
        pts = self._fin.points
        if pts:
            h_mm = max((pt.y_mm for pt in pts), default=0)
            w_mm = max((pt.x_mm for pt in pts), default=0) - min((pt.x_mm for pt in pts), default=0)
            tip_sp = self._to_screen(pts[len(pts) // 2].x_mm, h_mm)
            p.drawText(int(tip_sp.x()) + 6, int(tip_sp.y()) - 4,
                       f"H={h_mm:.0f} mm  B={w_mm:.0f} mm")

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    def _hit_index(self, pos: QPointF) -> int | None:
        for i, pt in enumerate(self._fin.points):
            sp = self._to_screen(pt.x_mm, pt.y_mm)
            dx, dy = pos.x() - sp.x(), pos.y() - sp.y()
            if math.hypot(dx, dy) <= HIT_RADIUS_PX:
                return i
        return None

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        self._fitted = False
        self.update()

    def mousePressEvent(self, event) -> None:
        pos = QPointF(event.pos())
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = pos
            self._pan_offset_start = QPointF(self._offset)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._hit_index(pos)
            if idx is not None:
                self._selected_idx = idx
                self._drag_active = True
                self._drag_start_screen = pos
                pt = self._fin.points[idx]
                self._drag_start_world = (pt.x_mm, pt.y_mm)
                self.point_selected.emit(idx)
            else:
                self._selected_idx = None
                self.point_selected.emit(-1)
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = False
            self._drag_start_screen = None
            self._drag_start_world = None
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = None

    def mouseMoveEvent(self, event) -> None:
        pos = QPointF(event.pos())

        if self._pan_start is not None and self._pan_offset_start is not None:
            delta = pos - self._pan_start
            self._offset = self._pan_offset_start + delta
            self.update()
            return

        self._hover_idx = self._hit_index(pos)

        if self._drag_active and self._selected_idx is not None:
            idx = self._selected_idx
            pt = self._fin.points[idx]
            wx, wy = self._to_world(pos.x(), pos.y())

            # Lock base endpoints to y=0
            if pt.y_mm == 0.0 or idx == 0 or idx == len(self._fin.points) - 1:
                wy = 0.0

            pt.x_mm = wx
            pt.y_mm = max(0.0, wy)
            self.update()
            self.fin_changed.emit()
        else:
            self.update()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1 / 1.15)
        pos = QPointF(event.position())
        # Zoom toward cursor
        wx, wy = self._to_world(pos.x(), pos.y())
        self._scale *= factor
        new_sx = wx * self._scale + self._offset.x()
        new_sy = -wy * self._scale + self._offset.y()
        self._offset += QPointF(pos.x() - new_sx, pos.y() - new_sy)
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = QPointF(event.pos())
        wx, wy = self._to_world(pos.x(), pos.y())
        wy = max(0.0, wy)

        # Insert new point in sorted position by y
        new_pt = FinPoint(x_mm=wx, y_mm=wy)
        pts = self._fin.points

        # Find insertion index: keep sorted by y (height)
        insert_at = len(pts)
        for i, p in enumerate(pts):
            if p.y_mm > wy:
                insert_at = i
                break

        # Don't insert before or after base endpoints
        insert_at = max(1, min(insert_at, len(pts) - 1))
        pts.insert(insert_at, new_pt)
        self._selected_idx = insert_at
        self.point_selected.emit(insert_at)
        self.update()
        self.fin_changed.emit()

    def _context_menu(self, pos) -> None:
        idx = self._hit_index(QPointF(pos))
        if idx is None:
            return
        pts = self._fin.points
        # Don't allow deleting first or last point
        if idx == 0 or idx == len(pts) - 1:
            return
        menu = QMenu(self)
        act = menu.addAction(f"Delete point {idx}")
        chosen = menu.exec(self.mapToGlobal(pos))
        if chosen == act:
            pts.pop(idx)
            if self._selected_idx == idx:
                self._selected_idx = None
                self.point_selected.emit(-1)
            self.update()
            self.fin_changed.emit()
