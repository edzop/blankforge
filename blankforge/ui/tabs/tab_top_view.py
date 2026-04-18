from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from blankforge.data.model import BoardModel
from blankforge.ui.widgets.control_point_editor import ControlPointEditor
from blankforge.ui.widgets.curve_viewport import CurveViewport


class TopViewCanvas(CurveViewport):
    x_label = "Position along board (mm)"
    y_label = "Half-Width (mm)"
    symmetric = True


class TopViewTab(QWidget):
    def __init__(self, model: BoardModel, model_changed: Signal, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._model_changed = model_changed
        self._updating = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        layout.addWidget(QLabel("<b>Top View — Width Curve</b>  (drag points; double-click canvas to add; right-click point to delete)"))

        self._editor = ControlPointEditor(
            curve_data=self._model.curves.width,
            viewport_class=TopViewCanvas,
            board_length=self._model.parameters.length_mm,
            max_value=600.0,
            parent=self,
        )
        layout.addWidget(self._editor, stretch=1)

        self._editor.curve_changed.connect(self._on_curve_changed)

    def refresh_from_model(self) -> None:
        self._updating = True
        self._editor.set_board_length(self._model.parameters.length_mm)
        self._editor.set_curve_data(self._model.curves.width)
        self._updating = False

    def _on_curve_changed(self) -> None:
        if not self._updating:
            self._model_changed.emit()
