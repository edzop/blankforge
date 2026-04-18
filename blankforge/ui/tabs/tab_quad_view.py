from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QWidget

from blankforge.data.model import BoardModel
from blankforge.geometry.board import BoardMesh
from blankforge.ui.widgets.curve_viewport import CurveViewport
from blankforge.ui.widgets.gl_viewport import GLViewport


class ReadOnlyTopCanvas(CurveViewport):
    symmetric = True
    x_label = "Position (mm)"
    y_label = "Width (mm)"

    def mousePressEvent(self, e): pass
    def mouseDoubleClickEvent(self, e): pass
    def mouseMoveEvent(self, e): pass
    def mouseReleaseEvent(self, e): pass


class ReadOnlyRockerCanvas(CurveViewport):
    x_label = "Position (mm)"
    y_label = "Rocker (mm)"

    def mousePressEvent(self, e): pass
    def mouseDoubleClickEvent(self, e): pass
    def mouseMoveEvent(self, e): pass
    def mouseReleaseEvent(self, e): pass


class ReadOnlyThicknessCanvas(CurveViewport):
    x_label = "Position (mm)"
    y_label = "Thickness (mm)"

    def mousePressEvent(self, e): pass
    def mouseDoubleClickEvent(self, e): pass
    def mouseMoveEvent(self, e): pass
    def mouseReleaseEvent(self, e): pass


def _labeled(title: str, widget: QWidget) -> QWidget:
    from PySide6.QtWidgets import QVBoxLayout
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(2)
    lbl = QLabel(f"<b>{title}</b>")
    layout.addWidget(lbl)
    layout.addWidget(widget, stretch=1)
    return container


class QuadViewTab(QWidget):
    def __init__(self, model: BoardModel, model_changed: Signal, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._model_changed = model_changed
        self._build_ui()

    def _build_ui(self) -> None:
        grid = QGridLayout(self)
        grid.setSpacing(4)
        grid.setContentsMargins(4, 4, 4, 4)

        self._top_canvas = ReadOnlyTopCanvas(self._model.curves.width)
        self._top_canvas.set_board_length(self._model.parameters.length_mm)

        self._rocker_canvas = ReadOnlyRockerCanvas(self._model.curves.rocker)
        self._rocker_canvas.set_board_length(self._model.parameters.length_mm)

        self._thick_canvas = ReadOnlyThicknessCanvas(self._model.curves.thickness)
        self._thick_canvas.set_board_length(self._model.parameters.length_mm)

        self._gl = GLViewport(self)

        grid.addWidget(_labeled("Top View", self._top_canvas), 0, 0)
        grid.addWidget(_labeled("Rendered View", self._gl), 0, 1)
        grid.addWidget(_labeled("Side View — Rocker", self._rocker_canvas), 1, 0)
        grid.addWidget(_labeled("Side View — Thickness", self._thick_canvas), 1, 1)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

    def update_mesh(self, mesh: BoardMesh) -> None:
        self._gl.update_mesh(mesh)

    def refresh_from_model(self) -> None:
        L = self._model.parameters.length_mm
        self._top_canvas.set_board_length(L)
        self._rocker_canvas.set_board_length(L)
        self._thick_canvas.set_board_length(L)
        self._top_canvas.set_curve_data(self._model.curves.width)
        self._rocker_canvas.set_curve_data(self._model.curves.rocker)
        self._thick_canvas.set_curve_data(self._model.curves.thickness)
