from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QLabel, QVBoxLayout, QWidget,
)

from blankforge.data.model import BoardModel
from blankforge.ui.widgets.value_sliders import LabeledSlider

TAIL_SHAPES = ["squaretail", "roundtail", "swallowtail", "dovetail"]
TAIL_LABELS = ["Square Tail", "Round Tail", "Swallow Tail", "Dove Tail"]


class ParametersTab(QWidget):
    def __init__(self, model: BoardModel, model_changed: Signal, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._model_changed = model_changed
        self._updating = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        layout.addWidget(QLabel("<h2>Board Parameters</h2>"))

        dims_box = QGroupBox("Global Dimensions")
        dims_form = QFormLayout(dims_box)
        dims_form.setSpacing(8)

        self._length = LabeledSlider("Length (mm)", 500, 4000, decimals=0)
        self._width = LabeledSlider("Width (mm)", 200, 900, decimals=0)
        self._thickness = LabeledSlider("Thickness (mm)", 20, 150, decimals=1)
        self._rocker = LabeledSlider("Rocker (mm)", 0, 200, decimals=1)

        for w in [self._length, self._width, self._thickness, self._rocker]:
            dims_form.addRow("", w)

        layout.addWidget(dims_box)

        tail_box = QGroupBox("Tail Configuration")
        tail_form = QFormLayout(tail_box)
        tail_form.setSpacing(8)

        self._tail_shape = QComboBox()
        self._tail_shape.addItems(TAIL_LABELS)
        tail_form.addRow("Shape:", self._tail_shape)

        self._tail_width = LabeledSlider("Tail Width (mm)", 100, 600, decimals=0)
        self._tail_length = LabeledSlider("Tail Length (mm)", 50, 400, decimals=0)
        self._tail_thickness = LabeledSlider("Tail Thickness (mm)", 15, 100, decimals=1)

        for w in [self._tail_width, self._tail_length, self._tail_thickness]:
            tail_form.addRow("", w)

        layout.addWidget(tail_box)
        layout.addStretch()

        # Connect signals
        self._length.value_changed.connect(self._on_changed)
        self._width.value_changed.connect(self._on_changed)
        self._thickness.value_changed.connect(self._on_changed)
        self._rocker.value_changed.connect(self._on_changed)
        self._tail_shape.currentIndexChanged.connect(self._on_changed)
        self._tail_width.value_changed.connect(self._on_changed)
        self._tail_length.value_changed.connect(self._on_changed)
        self._tail_thickness.value_changed.connect(self._on_changed)

        self.refresh_from_model()

    def refresh_from_model(self) -> None:
        self._updating = True
        p = self._model.parameters
        t = self._model.tail
        self._length.set_value(p.length_mm)
        self._width.set_value(p.width_mm)
        self._thickness.set_value(p.thickness_mm)
        self._rocker.set_value(p.rocker_mm)
        shape_idx = TAIL_SHAPES.index(t.shape) if t.shape in TAIL_SHAPES else 0
        self._tail_shape.setCurrentIndex(shape_idx)
        self._tail_width.set_value(t.width_mm)
        self._tail_length.set_value(t.length_mm)
        self._tail_thickness.set_value(t.thickness_mm)
        self._updating = False

    def _on_changed(self, *_) -> None:
        if self._updating:
            return
        p = self._model.parameters
        p.length_mm = self._length.value()
        p.width_mm = self._width.value()
        p.thickness_mm = self._thickness.value()
        p.rocker_mm = self._rocker.value()
        t = self._model.tail
        t.shape = TAIL_SHAPES[self._tail_shape.currentIndex()]
        t.width_mm = self._tail_width.value()
        t.length_mm = self._tail_length.value()
        t.thickness_mm = self._tail_thickness.value()
        self._model_changed.emit()
