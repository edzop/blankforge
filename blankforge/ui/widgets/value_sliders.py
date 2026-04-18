from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QSlider, QWidget


class LabeledSlider(QWidget):
    value_changed = Signal(float)

    def __init__(
        self,
        label: str,
        min_val: float,
        max_val: float,
        decimals: int = 1,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._decimals = decimals
        self._scale = 10 ** decimals
        self._blocked = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._label = QLabel(label)
        self._label.setMinimumWidth(130)
        layout.addWidget(self._label)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(int(min_val * self._scale))
        self._slider.setMaximum(int(max_val * self._scale))
        layout.addWidget(self._slider, stretch=1)

        self._spinbox = QDoubleSpinBox()
        self._spinbox.setMinimum(min_val)
        self._spinbox.setMaximum(max_val)
        self._spinbox.setDecimals(decimals)
        self._spinbox.setMinimumWidth(80)
        layout.addWidget(self._spinbox)

        self._slider.valueChanged.connect(self._on_slider_changed)
        self._spinbox.valueChanged.connect(self._on_spin_changed)

    def _on_slider_changed(self, slider_val: int) -> None:
        if self._blocked:
            return
        val = slider_val / self._scale
        self._blocked = True
        self._spinbox.setValue(val)
        self._blocked = False
        self.value_changed.emit(val)

    def _on_spin_changed(self, val: float) -> None:
        if self._blocked:
            return
        self._blocked = True
        self._slider.setValue(int(val * self._scale))
        self._blocked = False
        self.value_changed.emit(val)

    def value(self) -> float:
        return self._spinbox.value()

    def set_value(self, v: float) -> None:
        self._blocked = True
        clamped = max(self._min, min(self._max, v))
        self._spinbox.setValue(clamped)
        self._slider.setValue(int(clamped * self._scale))
        self._blocked = False

    def set_label(self, text: str) -> None:
        self._label.setText(text)
