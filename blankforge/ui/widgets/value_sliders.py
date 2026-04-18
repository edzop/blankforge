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
        suffix: str = "",
        imperial_factor: float | None = None,
        imperial_suffix: str = "in",
        imperial_decimals: int = 2,
        show_ft_in: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._decimals = decimals
        self._scale = 10 ** decimals
        self._blocked = False
        self._imperial_factor = imperial_factor

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
        if suffix:
            self._spinbox.setSuffix(f" {suffix}")
        layout.addWidget(self._spinbox)

        if imperial_factor is not None:
            self._imperial_spin = QDoubleSpinBox()
            self._imperial_spin.setMinimum(min_val * imperial_factor)
            self._imperial_spin.setMaximum(max_val * imperial_factor)
            self._imperial_spin.setDecimals(imperial_decimals)
            self._imperial_spin.setSuffix(f" {imperial_suffix}")
            self._imperial_spin.setMinimumWidth(80)
            layout.addWidget(self._imperial_spin)
            self._imperial_spin.valueChanged.connect(self._on_imperial_changed)
        else:
            self._imperial_spin = None

        # Read-only feet+inches display (only useful for length-style values)
        if show_ft_in and imperial_factor is not None:
            self._ft_in_label = QLabel("")
            self._ft_in_label.setMinimumWidth(72)
            self._ft_in_label.setStyleSheet("color: #aaa;")
            layout.addWidget(self._ft_in_label)
        else:
            self._ft_in_label = None

        self._slider.valueChanged.connect(self._on_slider_changed)
        self._spinbox.valueChanged.connect(self._on_spin_changed)

    def _update_ft_in(self, val: float) -> None:
        if self._ft_in_label is None:
            return
        total_in = val * self._imperial_factor
        ft = int(total_in // 12)
        inches = total_in - ft * 12
        self._ft_in_label.setText(f"{ft}' {inches:.1f}\"")

    def _on_slider_changed(self, slider_val: int) -> None:
        if self._blocked:
            return
        val = slider_val / self._scale
        self._blocked = True
        self._spinbox.setValue(val)
        if self._imperial_spin is not None:
            self._imperial_spin.setValue(val * self._imperial_factor)
        self._update_ft_in(val)
        self._blocked = False
        self.value_changed.emit(val)

    def _on_spin_changed(self, val: float) -> None:
        if self._blocked:
            return
        self._blocked = True
        self._slider.setValue(int(val * self._scale))
        if self._imperial_spin is not None:
            self._imperial_spin.setValue(val * self._imperial_factor)
        self._update_ft_in(val)
        self._blocked = False
        self.value_changed.emit(val)

    def _on_imperial_changed(self, imperial_val: float) -> None:
        if self._blocked:
            return
        val = max(self._min, min(self._max, imperial_val / self._imperial_factor))
        self._blocked = True
        self._spinbox.setValue(val)
        self._slider.setValue(int(val * self._scale))
        self._update_ft_in(val)
        self._blocked = False
        self.value_changed.emit(val)

    def value(self) -> float:
        return self._spinbox.value()

    def set_value(self, v: float) -> None:
        self._blocked = True
        clamped = max(self._min, min(self._max, v))
        self._spinbox.setValue(clamped)
        self._slider.setValue(int(clamped * self._scale))
        if self._imperial_spin is not None:
            self._imperial_spin.setValue(clamped * self._imperial_factor)
        self._update_ft_in(clamped)
        self._blocked = False

    def set_range(self, min_val: float, max_val: float) -> None:
        self._min = min_val
        self._max = max_val
        self._slider.setMinimum(int(min_val * self._scale))
        self._slider.setMaximum(int(max_val * self._scale))
        self._spinbox.setMinimum(min_val)
        self._spinbox.setMaximum(max_val)
        if self._imperial_spin is not None:
            self._imperial_spin.setMinimum(min_val * self._imperial_factor)
            self._imperial_spin.setMaximum(max_val * self._imperial_factor)

    def set_label(self, text: str) -> None:
        self._label.setText(text)
