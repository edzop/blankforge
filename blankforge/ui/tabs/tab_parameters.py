from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout, QGroupBox, QLabel, QVBoxLayout, QWidget,
)

from blankforge.data.model import BoardModel
from blankforge.ui.widgets.value_sliders import LabeledSlider


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

        MM_TO_IN = 1.0 / 25.4
        self._length = LabeledSlider("Length", 500, 4000, decimals=0, suffix="mm",
                                     imperial_factor=MM_TO_IN, imperial_suffix="in", imperial_decimals=1)
        self._width = LabeledSlider("Width", 200, 900, decimals=0, suffix="mm",
                                    imperial_factor=MM_TO_IN, imperial_suffix="in", imperial_decimals=2)
        self._thickness = LabeledSlider("Thickness", 20, 150, decimals=1, suffix="mm",
                                        imperial_factor=MM_TO_IN, imperial_suffix="in", imperial_decimals=2)
        self._rocker = LabeledSlider("Rocker", 0, 200, decimals=1, suffix="mm",
                                     imperial_factor=MM_TO_IN, imperial_suffix="in", imperial_decimals=2)

        for w in [self._length, self._width, self._thickness, self._rocker]:
            dims_form.addRow("", w)

        layout.addWidget(dims_box)
        layout.addStretch()

        self._length.value_changed.connect(self._on_changed)
        self._width.value_changed.connect(self._on_changed)
        self._thickness.value_changed.connect(self._on_changed)
        self._rocker.value_changed.connect(self._on_changed)

        self.refresh_from_model()

    def refresh_from_model(self) -> None:
        self._updating = True
        p = self._model.parameters
        self._length.set_value(p.length_mm)
        self._width.set_value(p.width_mm)
        self._thickness.set_value(p.thickness_mm)
        self._rocker.set_value(p.rocker_mm)
        self._updating = False

    def _on_changed(self, *_) -> None:
        if self._updating:
            return
        p = self._model.parameters
        new_L, old_L = self._length.value(), p.length_mm
        new_W, old_W = self._width.value(), p.width_mm
        new_T, old_T = self._thickness.value(), p.thickness_mm
        new_R, old_R = self._rocker.value(), p.rocker_mm

        # Length: scale all control-point X positions so they stay at the same board ratio
        if abs(new_L - old_L) > 0.1 and old_L > 0:
            ratio = new_L / old_L
            for curve in (self._model.curves.width, self._model.curves.rocker,
                          self._model.curves.thickness):
                for cp in curve.points:
                    cp.position_mm = cp.position_mm * ratio
            for station in self._model.curves.rail:
                station.position_mm = station.position_mm * ratio

        # Width: scale width-curve values so the board's wide point follows the new param
        if abs(new_W - old_W) > 0.1 and old_W > 0:
            ratio = new_W / old_W
            for cp in self._model.curves.width.points:
                cp.value_mm = cp.value_mm * ratio

        # Thickness: scale fixed-mode thickness values; ratio-mode points already track param
        if abs(new_T - old_T) > 0.05 and old_T > 0:
            ratio = new_T / old_T
            for cp in self._model.curves.thickness.points:
                if cp.mode == "fixed":
                    cp.value_mm = cp.value_mm * ratio

        # Rocker: scale rocker curve values
        if abs(new_R - old_R) > 0.05 and old_R > 0:
            ratio = new_R / old_R
            for cp in self._model.curves.rocker.points:
                cp.value_mm = cp.value_mm * ratio

        p.length_mm = new_L
        p.width_mm = new_W
        p.thickness_mm = new_T
        p.rocker_mm = new_R
        self._model_changed.emit()
