from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from blankforge.data.model import ControlPoint, CurveData
from blankforge.ui.widgets.curve_viewport import CurveViewport
from blankforge.ui.widgets.value_sliders import LabeledSlider


class ControlPointSliderPanel(QWidget):
    value_changed = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._pos_slider = LabeledSlider("Position (mm)", 0, 10000, decimals=0)
        self._val_slider = LabeledSlider("Value (mm)", 0, 2000, decimals=1)
        layout.addWidget(self._pos_slider)
        layout.addWidget(self._val_slider)

        self._blocked = False
        self._pos_slider.value_changed.connect(self._emit)
        self._val_slider.value_changed.connect(self._emit)
        self.setEnabled(False)

    def _emit(self, _=None) -> None:
        if not self._blocked:
            self.value_changed.emit(self._pos_slider.value(), self._val_slider.value())

    def set_point(self, pt: ControlPoint, board_length: float, max_value: float) -> None:
        self._blocked = True
        self._pos_slider._spinbox.setMaximum(board_length)
        self._pos_slider._slider.setMaximum(int(board_length))
        self._val_slider._spinbox.setMaximum(max_value)
        self._val_slider._slider.setMaximum(int(max_value * 10))
        self._pos_slider.set_value(pt.position_mm)
        self._val_slider.set_value(pt.value_mm)
        self._blocked = False
        self.setEnabled(True)

    def configure_labels(self, pos_label: str, val_label: str) -> None:
        self._pos_slider.set_label(pos_label)
        self._val_slider.set_label(val_label)

    def configure_ranges(self, pos_max: float, val_max: float) -> None:
        self._pos_slider._spinbox.setMaximum(pos_max)
        self._pos_slider._slider.setMaximum(int(pos_max))
        self._val_slider._spinbox.setMaximum(val_max)
        self._val_slider._slider.setMaximum(int(val_max * 10))


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
        self._slider_panel.value_changed.connect(self._on_slider_changed)

        # Auto-select first point so sliders are immediately usable
        self._select_first_point()

    def _select_first_point(self) -> None:
        pts = self._curve.sorted_points()
        if pts:
            self._viewport.set_selected_index(0)
            self._slider_panel.set_point(pts[0], self._board_length, self._max_value)

    def set_curve_data(self, curve: CurveData) -> None:
        self._curve = curve
        self._viewport.set_curve_data(curve)
        self._select_first_point()
        self.curve_changed.emit()

    def set_board_length(self, length_mm: float) -> None:
        self._board_length = length_mm
        self._viewport.set_board_length(length_mm)

    def viewport(self) -> CurveViewport:
        return self._viewport

    def _on_point_selected(self, idx: int) -> None:
        pts = self._curve.sorted_points()
        if 0 <= idx < len(pts):
            self._slider_panel.set_point(pts[idx], self._board_length, self._max_value)

    def _on_point_moved(self, idx: int, pos_mm: float, val_mm: float) -> None:
        pts = self._curve.sorted_points()
        if 0 <= idx < len(pts):
            pts[idx].position_mm = pos_mm
            pts[idx].value_mm = val_mm
            self._curve.points = pts
            self._slider_panel.set_point(pts[idx], self._board_length, self._max_value)
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

    def _on_slider_changed(self, pos_mm: float, val_mm: float) -> None:
        idx = self._viewport.selected_index()
        if idx is None:
            return
        pts = self._curve.sorted_points()
        if 0 <= idx < len(pts):
            pts[idx].position_mm = pos_mm
            pts[idx].value_mm = val_mm
            self._curve.points = pts
            self._viewport.update()
            self.curve_changed.emit()
