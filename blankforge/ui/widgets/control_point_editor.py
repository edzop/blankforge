from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from blankforge.data.model import ControlPoint, CurveData
from blankforge.ui.widgets.curve_viewport import CurveViewport
from blankforge.ui.widgets.value_sliders import LabeledSlider


class ControlPointSliderPanel(QWidget):
    value_changed = Signal(float, float, float)  # (pos_mm, val_mm, influence)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._pos_scale: float = 1.0  # pos_mm = slider_value * _pos_scale
        self._val_scale: float = 1.0  # val_mm = slider_value * _val_scale
        # Ratio sliders cap at 1.0 — to go higher, raise the global parameter
        self._pos_slider = LabeledSlider("Position", 0.0, 1.0, decimals=3)
        self._val_slider = LabeledSlider("Value", 0.0, 1.0, decimals=3)
        self._infl_slider = LabeledSlider("Influence", 0.0, 1.0, decimals=2)
        layout.addWidget(self._pos_slider)
        layout.addWidget(self._val_slider)
        layout.addWidget(self._infl_slider)

        self._blocked = False
        self._pos_slider.value_changed.connect(self._emit)
        self._val_slider.value_changed.connect(self._emit)
        self._infl_slider.value_changed.connect(self._emit)
        self.setEnabled(False)

    def set_scales(self, pos_scale: float, val_scale: float) -> None:
        self._pos_scale = max(pos_scale, 1e-9)
        self._val_scale = max(val_scale, 1e-9)

    def _emit(self, _=None) -> None:
        if not self._blocked:
            self.value_changed.emit(
                self._pos_slider.value() * self._pos_scale,
                self._val_slider.value() * self._val_scale,
                self._infl_slider.value(),
            )

    def set_point(self, pt: ControlPoint, board_length: float, max_value: float) -> None:
        self._blocked = True
        self._pos_slider.set_value(pt.position_mm / self._pos_scale)
        self._val_slider.set_value(pt.value_mm / self._val_scale)
        self._infl_slider.set_value(getattr(pt, "influence", 1.0))
        self._blocked = False
        self.setEnabled(True)

    def configure_labels(self, pos_label: str, val_label: str) -> None:
        self._pos_slider.set_label(pos_label)
        self._val_slider.set_label(val_label)


class ControlPointEditor(QWidget):
    curve_changed = Signal()

    def __init__(
        self,
        curve_data: CurveData,
        viewport_class: type[CurveViewport] = CurveViewport,
        board_length: float = 2000.0,
        max_value: float = 600.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._curve = curve_data
        self._board_length = board_length
        self._max_value = max_value
        # Track selection by position_mm, not by sorted index.
        # Sorted index is fragile: float quantization in sliders can cause
        # tiny position drifts that change sort order and lose the selection.
        self._selected_pos_mm: float | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._viewport = viewport_class(curve_data, self)
        self._viewport.set_board_length(board_length)
        layout.addWidget(self._viewport, stretch=1)

        self._slider_panel = ControlPointSliderPanel()
        layout.addWidget(self._slider_panel)

        self._viewport.point_selected.connect(self._on_point_selected)
        self._viewport.point_moved.connect(self._on_point_moved)
        self._viewport.point_added.connect(self._on_point_added)
        self._viewport.point_deleted.connect(self._on_point_deleted)
        self._viewport.drag_finished.connect(self._on_drag_finished)
        self._slider_panel.value_changed.connect(self._on_slider_changed)

        # Auto-select first point so sliders are immediately usable
        self._select_first_point()

    def _idx_by_pos(self, pts: list[ControlPoint]) -> int | None:
        if self._selected_pos_mm is None or not pts:
            return None
        return min(range(len(pts)), key=lambda i: abs(pts[i].position_mm - self._selected_pos_mm))

    def _select_first_point(self) -> None:
        pts = self._curve.sorted_points()
        if pts:
            self._selected_pos_mm = pts[0].position_mm
            self._viewport.set_selected_index(0)
            self._slider_panel.set_point(pts[0], self._board_length, self._max_value)

    def set_curve_data(self, curve: CurveData) -> None:
        self._curve = curve
        self._viewport.set_curve_data(curve)
        if not self._viewport._drag_active:
            pts = self._curve.sorted_points()
            idx = self._idx_by_pos(pts)
            if idx is not None:
                self._viewport.set_selected_index(idx)
                self._slider_panel.set_point(pts[idx], self._board_length, self._max_value)
            else:
                self._select_first_point()
        self.curve_changed.emit()

    def set_board_length(self, length_mm: float) -> None:
        self._board_length = length_mm
        self._viewport.set_board_length(length_mm)

    def set_scales(self, pos_scale: float, val_scale: float) -> None:
        self._slider_panel.set_scales(pos_scale, val_scale)
        pts = self._curve.sorted_points()
        idx = self._idx_by_pos(pts)
        if idx is not None:
            self._slider_panel.set_point(pts[idx], self._board_length, self._max_value)

    def viewport(self) -> CurveViewport:
        return self._viewport

    def _on_point_selected(self, idx: int) -> None:
        pts = self._curve.sorted_points()
        if 0 <= idx < len(pts):
            self._selected_pos_mm = pts[idx].position_mm
            self._slider_panel.set_point(pts[idx], self._board_length, self._max_value)

    def _on_point_moved(self, idx: int, pos_mm: float, val_mm: float) -> None:
        pts = self._curve.sorted_points()
        if 0 <= idx < len(pts):
            pts[idx].position_mm = pos_mm
            pts[idx].value_mm = val_mm
            self._curve.points = pts
            self._selected_pos_mm = pos_mm
            self._slider_panel.set_point(pts[idx], self._board_length, self._max_value)
            # Don't emit curve_changed on every move — wait for drag_finished

    def _on_drag_finished(self) -> None:
        self.curve_changed.emit()

    def _on_point_added(self, pos_mm: float, val_mm: float) -> None:
        self._curve.points.append(ControlPoint(position_mm=pos_mm, value_mm=val_mm))
        self._viewport.update()
        self.curve_changed.emit()

    def _on_point_deleted(self, idx: int) -> None:
        pts = self._curve.sorted_points()
        if 0 <= idx < len(pts):
            del pts[idx]
            self._curve.points = pts
            self._viewport.update()
            self._select_first_point()
            self.curve_changed.emit()

    def _on_slider_changed(self, pos_mm: float, val_mm: float, influence: float) -> None:
        idx = self._idx_by_pos(self._curve.sorted_points())
        if idx is None:
            return
        pts = self._curve.sorted_points()
        if 0 <= idx < len(pts):
            pts[idx].position_mm = pos_mm
            pts[idx].value_mm = val_mm
            pts[idx].influence = influence
            self._curve.points = pts
            self._selected_pos_mm = pos_mm
            self._viewport.update()
            self.curve_changed.emit()
