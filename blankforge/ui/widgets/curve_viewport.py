from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor, QFont, QPainter, QPainterPath, QPen, QWheelEvent,
)
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QMenu, QWidget

from blankforge.data.model import ControlPoint, CurveData
from blankforge.geometry.curves import BoardCurveEvaluator

HIT_RADIUS_PX = 8
POINT_RADIUS_PX = 6


class ViewTransform:
    def __init__(self) -> None:
        self.offset = QPointF(0, 0)
        self.scale = 1.0

    def world_to_screen(self, wx: float, wy: float) -> QPointF:
        return QPointF(wx * self.scale + self.offset.x(), wy * self.scale + self.offset.y())

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return (sx - self.offset.x()) / self.scale, (sy - self.offset.y()) / self.scale

    def fit(self, xs: list[float], ys: list[float], widget_w: int, widget_h: int, margin: float = 0.1) -> None:
        if not xs or not ys:
            return
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_span = (x_max - x_min) or 1.0
        y_span = (y_max - y_min) or 1.0
        scale_x = widget_w * (1 - 2 * margin) / x_span
        scale_y = widget_h * (1 - 2 * margin) / y_span
        self.scale = min(scale_x, scale_y)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        self.offset = QPointF(
            widget_w / 2 - cx * self.scale,
            widget_h / 2 - cy * self.scale,
        )


class CurveViewport(QWidget):
    point_selected = Signal(int)
    point_moved = Signal(int, float, float)
    point_added = Signal(float, float)
    point_deleted = Signal(int)

    # Subclasses set these to configure axis appearance
    x_label: str = "Position (mm)"
    y_label: str = "Value (mm)"
    y_flip: bool = False  # Set True for rocker (positive = up = lower screen Y)
    symmetric: bool = False  # Set True for width (draws mirrored silhouette)

    def __init__(self, curve_data: CurveData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(300, 200)
        self.setMouseTracking(True)
        self._curve = curve_data
        self._transform = ViewTransform()
        self._selected_idx: int | None = None
        self._hover_idx: int | None = None
        self._drag_active = False
        self._drag_start_world: tuple[float, float] | None = None
        self._drag_start_pt: tuple[float, float] | None = None
        self._pan_start: QPointF | None = None
        self._pan_offset_start: QPointF | None = None
        self._fitted = False
        self._board_length: float = 2000.0

    def set_curve_data(self, curve: CurveData) -> None:
        self._curve = curve
        self._fitted = False
        self.update()

    def set_board_length(self, length_mm: float) -> None:
        self._board_length = length_mm
        self._fitted = False
        self.update()

    def _ensure_fitted(self) -> None:
        if self._fitted:
            return
        pts = self._curve.sorted_points()
        if not pts:
            xs = [0, self._board_length]
            ys = [-50, 300]
        else:
            xs = [p.position_mm for p in pts] + [0, self._board_length]
            ys = [p.value_mm for p in pts]
            if self.symmetric:
                ys = [-y for y in ys] + ys
            else:
                ys = ys + [0]
        # When y_flip=True the transform works in negated-Y space; negate ys so fit() centers correctly
        fit_ys = [-y for y in ys] if self.y_flip else ys
        self._transform.fit(xs, fit_ys, self.width(), self.height())
        self._fitted = True

    def _screen_y(self, wy: float) -> float:
        """Map world Y → screen Y, optionally flipping."""
        if self.y_flip:
            wy = -wy
        return wy * self._transform.scale + self._transform.offset.y()

    def _world_y_from_screen(self, sy: float) -> float:
        wy = (sy - self._transform.offset.y()) / self._transform.scale
        if self.y_flip:
            wy = -wy
        return wy

    def _to_screen(self, wx: float, wy: float) -> QPointF:
        sx = wx * self._transform.scale + self._transform.offset.x()
        sy = self._screen_y(wy)
        return QPointF(sx, sy)

    def _from_screen(self, sx: float, sy: float) -> tuple[float, float]:
        wx = (sx - self._transform.offset.x()) / self._transform.scale
        wy = self._world_y_from_screen(sy)
        return wx, wy

    def _sorted_points(self) -> list[ControlPoint]:
        return self._curve.sorted_points()

    def _hit_test(self, sx: float, sy: float) -> int | None:
        for i, pt in enumerate(self._sorted_points()):
            sp = self._to_screen(pt.position_mm, pt.value_mm)
            dx = sx - sp.x()
            dy = sy - sp.y()
            if (dx * dx + dy * dy) ** 0.5 < HIT_RADIUS_PX:
                return i
        return None

    def _overlay_lines(self) -> list[str]:
        """Override in subclasses to add info text in the top-right corner."""
        return []

    def paintEvent(self, event) -> None:
        self._ensure_fitted()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_background(p)
        self._draw_grid(p)
        self._draw_curve(p)
        self._draw_points(p)
        self._draw_axes_labels(p)
        self._draw_overlay(p)

    def _draw_overlay(self, p: QPainter) -> None:
        lines = self._overlay_lines()
        if not lines:
            return
        font = QFont("monospace", 9)
        font.setBold(True)
        p.setFont(font)
        fm = QFontMetrics(font)
        line_h = fm.height() + 3
        pad = 8
        max_w = max(fm.horizontalAdvance(ln) for ln in lines)
        box_w = max_w + pad * 2
        box_h = line_h * len(lines) + pad * 2
        x = self.width() - box_w - 8
        y = 8
        # Semi-transparent background pill
        p.setBrush(QColor(20, 22, 28, 200))
        p.setPen(QPen(QColor(70, 80, 100, 180)))
        p.drawRoundedRect(x, y, box_w, box_h, 6, 6)
        p.setPen(QColor(190, 210, 240))
        for i, line in enumerate(lines):
            p.drawText(x + pad, y + pad + fm.ascent() + i * line_h, line)

    def _draw_background(self, p: QPainter) -> None:
        p.fillRect(self.rect(), QColor(30, 32, 36))

    def _draw_grid(self, p: QPainter) -> None:
        pen = QPen(QColor(50, 52, 60))
        pen.setWidth(1)
        p.setPen(pen)
        # Vertical gridlines every 200mm
        step = 200.0
        x = 0.0
        while x <= self._board_length + step:
            sx = self._to_screen(x, 0).x()
            if 0 <= sx <= self.width():
                p.drawLine(int(sx), 0, int(sx), self.height())
            x += step
        # Horizontal zero line
        sy_zero = int(self._to_screen(0, 0).y())
        pen_zero = QPen(QColor(70, 72, 80))
        p.setPen(pen_zero)
        p.drawLine(0, sy_zero, self.width(), sy_zero)

    def _draw_curve(self, p: QPainter) -> None:
        pts = self._sorted_points()
        if len(pts) < 2:
            return
        evaluator = BoardCurveEvaluator(self._curve)
        xs = np.linspace(0, self._board_length, 200)
        ys = evaluator(xs)

        path = QPainterPath()
        first = True
        for x, y in zip(xs, ys):
            sp = self._to_screen(x, y)
            if first:
                path.moveTo(sp)
                first = False
            else:
                path.lineTo(sp)

        if self.symmetric:
            # Draw mirrored (negative Y)
            path_mirror = QPainterPath()
            first = True
            for x, y in zip(xs, ys):
                sp = self._to_screen(x, -y)
                if first:
                    path_mirror.moveTo(sp)
                    first = False
                else:
                    path_mirror.lineTo(sp)
            # Fill between curves
            fill_path = QPainterPath(path)
            for x, y in zip(reversed(xs), reversed(ys)):
                fill_path.lineTo(self._to_screen(x, -y))
            fill_path.closeSubpath()
            p.fillPath(fill_path, QColor(60, 100, 140, 60))
            pen_m = QPen(QColor(80, 130, 180, 150))
            pen_m.setWidth(1)
            p.setPen(pen_m)
            p.drawPath(path_mirror)

        pen = QPen(QColor(80, 160, 220))
        pen.setWidthF(2.0)
        p.setPen(pen)
        p.drawPath(path)

    def _draw_points(self, p: QPainter) -> None:
        for i, pt in enumerate(self._sorted_points()):
            sp = self._to_screen(pt.position_mm, pt.value_mm)
            is_sel = i == self._selected_idx
            is_hov = i == self._hover_idx
            if is_sel:
                color = QColor(255, 200, 50)
            elif is_hov:
                color = QColor(150, 210, 255)
            else:
                color = QColor(100, 180, 255)
            p.setBrush(color)
            pen = QPen(QColor(255, 255, 255, 180))
            pen.setWidthF(1.5)
            p.setPen(pen)
            p.drawEllipse(sp, POINT_RADIUS_PX, POINT_RADIUS_PX)

    def _draw_axes_labels(self, p: QPainter) -> None:
        font = QFont("monospace", 8)
        p.setFont(font)
        p.setPen(QColor(120, 125, 135))
        # X axis ticks
        step = 200.0
        x = 0.0
        while x <= self._board_length + step:
            sp = self._to_screen(x, 0)
            sy_zero = sp.y()
            if 0 <= sp.x() <= self.width():
                p.drawText(QRectF(sp.x() - 20, sy_zero + 4, 40, 14), Qt.AlignmentFlag.AlignCenter, f"{int(x)}")
            x += step

    def mousePressEvent(self, event) -> None:
        sx, sy = event.position().x(), event.position().y()
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = event.position()
            self._pan_offset_start = QPointF(self._transform.offset)
            return
        if event.button() == Qt.MouseButton.RightButton:
            idx = self._hit_test(sx, sy)
            if idx is not None and len(self._sorted_points()) > 2:
                menu = QMenu(self)
                act = menu.addAction("Delete point")
                if menu.exec(event.globalPosition().toPoint()) == act:
                    self._selected_idx = None
                    self.point_deleted.emit(idx)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._hit_test(sx, sy)
            if idx is not None:
                self._selected_idx = idx
                self._drag_active = True
                self._drag_start_world = self._from_screen(sx, sy)
                pt = self._sorted_points()[idx]
                self._drag_start_pt = (pt.position_mm, pt.value_mm)
                self.point_selected.emit(idx)
            else:
                # Double-click adds a point
                if event.type().name == "MouseButtonDblClick":
                    wx, wy = self._from_screen(sx, sy)
                    self.point_added.emit(wx, wy)
            self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            sx, sy = event.position().x(), event.position().y()
            idx = self._hit_test(sx, sy)
            if idx is None:
                wx, wy = self._from_screen(sx, sy)
                self.point_added.emit(max(0.0, wx), max(0.0, wy))

    def mouseMoveEvent(self, event) -> None:
        sx, sy = event.position().x(), event.position().y()
        if self._pan_start is not None:
            delta = event.position() - self._pan_start
            self._transform.offset = QPointF(
                self._pan_offset_start.x() + delta.x(),
                self._pan_offset_start.y() + delta.y(),
            )
            self.update()
            return
        if self._drag_active and self._drag_start_world is not None:
            wx, wy = self._from_screen(sx, sy)
            dx = wx - self._drag_start_world[0]
            dy = wy - self._drag_start_world[1]
            new_pos = self._drag_start_pt[0] + dx
            new_val = self._drag_start_pt[1] + dy
            new_pos = max(0.0, min(self._board_length, new_pos))
            new_val = max(0.0, new_val)
            self.point_moved.emit(self._selected_idx, new_pos, new_val)
            self.update()
            return
        old_hover = self._hover_idx
        self._hover_idx = self._hit_test(sx, sy)
        if self._hover_idx != old_hover:
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = None
            self._pan_offset_start = None
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = False
            self._drag_start_world = None
            self._drag_start_pt = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        sx, sy = event.position().x(), event.position().y()
        wx, wy = self._from_screen(sx, sy)
        self._transform.scale *= factor
        # Keep mouse world position fixed
        self._transform.offset = QPointF(
            sx - wx * self._transform.scale,
            sy - self._screen_y_raw(wy),
        )
        self.update()

    def _screen_y_raw(self, wy: float) -> float:
        if self.y_flip:
            wy = -wy
        return wy * self._transform.scale + self._transform.offset.y()

    def resizeEvent(self, event) -> None:
        self._fitted = False
        super().resizeEvent(event)

    def selected_index(self) -> int | None:
        return self._selected_idx

    def set_selected_index(self, idx: int | None) -> None:
        self._selected_idx = idx
        self.update()
