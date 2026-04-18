from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QVBoxLayout, QWidget,
)

from blankforge.data.model import BoardModel
from blankforge.geometry.board import BoardStats

# Conversion constants
MM_TO_IN = 1 / 25.4
MM_TO_FT = 1 / 304.8
L_TO_GAL = 0.264172          # litres → US gallons
CM2_TO_FT2 = 0.00107639      # cm² → ft²


def _mm_to_ftin(mm: float) -> str:
    total_in = mm * MM_TO_IN
    feet = int(total_in // 12)
    inches = total_in % 12
    return f"{feet}'{inches:.1f}\""


def _mm_to_in(mm: float) -> str:
    return f"{mm * MM_TO_IN:.2f}\""


class StatisticsTab(QWidget):
    def __init__(self, model: BoardModel, model_changed: Signal, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._model_changed = model_changed
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        layout.addWidget(QLabel(
            "<h2>Board Statistics</h2>"
            "<p>Computed from geometry. Updates automatically when curves change.</p>"
        ))

        box = QGroupBox("Computed Values")
        box_layout = QVBoxLayout(box)

        # Header row
        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Measurement</b>"), stretch=2)
        header.addWidget(QLabel("<b>Metric</b>"), stretch=1)
        header.addWidget(QLabel("<b>Imperial</b>"), stretch=1)
        box_layout.addLayout(header)

        sep = QLabel()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #444;")
        box_layout.addWidget(sep)

        self._metric_labels: dict[str, QLabel] = {}
        self._imperial_labels: dict[str, QLabel] = {}

        rows = [
            ("volume",           "Volume"),
            ("surface_area",     "Surface Area"),
            ("length",           "Length"),
            ("width",            "Width"),
            ("thickness",        "Thickness"),
            ("nose_width_1in",   "Nose Width @ 1\" from nose"),
            ("nose_width_2in",   "Nose Width @ 2\" from nose"),
            ("tail_width_1in",   "Tail Width @ 1\" from tail"),
            ("tail_width_2in",   "Tail Width @ 2\" from tail"),
        ]

        for key, label_text in rows:
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            m_val = QLabel("—")
            i_val = QLabel("—")
            m_val.setStyleSheet("font-family: monospace; font-weight: bold;")
            i_val.setStyleSheet("font-family: monospace; color: #aaa;")
            row.addWidget(lbl, stretch=2)
            row.addWidget(m_val, stretch=1)
            row.addWidget(i_val, stretch=1)
            box_layout.addLayout(row)
            self._metric_labels[key] = m_val
            self._imperial_labels[key] = i_val

        layout.addWidget(box)

        self._status = QLabel("Waiting for geometry build…")
        self._status.setStyleSheet("color: #888;")
        layout.addWidget(self._status)
        layout.addStretch()

    def update_stats(self, stats: BoardStats) -> None:
        volume_l = stats.volume_cm3 / 1000.0
        surface_m2 = stats.surface_area_cm2 / 10000.0

        self._metric_labels["volume"].setText(f"{volume_l:.2f} L")
        self._imperial_labels["volume"].setText(f"{volume_l * L_TO_GAL:.2f} gal")

        self._metric_labels["surface_area"].setText(f"{surface_m2:.3f} m²")
        self._imperial_labels["surface_area"].setText(f"{stats.surface_area_cm2 * CM2_TO_FT2:.2f} ft²")

        self._metric_labels["length"].setText(f"{stats.length_mm:.0f} mm")
        self._imperial_labels["length"].setText(_mm_to_ftin(stats.length_mm))

        self._metric_labels["width"].setText(f"{stats.width_mm:.0f} mm")
        self._imperial_labels["width"].setText(_mm_to_in(stats.width_mm))

        self._metric_labels["thickness"].setText(f"{stats.thickness_mm:.1f} mm")
        self._imperial_labels["thickness"].setText(_mm_to_in(stats.thickness_mm))

        self._metric_labels["nose_width_1in"].setText(f"{stats.nose_width_1in_mm:.1f} mm")
        self._imperial_labels["nose_width_1in"].setText(_mm_to_in(stats.nose_width_1in_mm))

        self._metric_labels["nose_width_2in"].setText(f"{stats.nose_width_2in_mm:.1f} mm")
        self._imperial_labels["nose_width_2in"].setText(_mm_to_in(stats.nose_width_2in_mm))

        self._metric_labels["tail_width_1in"].setText(f"{stats.tail_width_1in_mm:.1f} mm")
        self._imperial_labels["tail_width_1in"].setText(_mm_to_in(stats.tail_width_1in_mm))

        self._metric_labels["tail_width_2in"].setText(f"{stats.tail_width_2in_mm:.1f} mm")
        self._imperial_labels["tail_width_2in"].setText(_mm_to_in(stats.tail_width_2in_mm))

        self._status.setText("Up to date.")
        self._status.setStyleSheet("color: #4fa;")

    def set_building(self) -> None:
        self._status.setText("Building geometry…")
        self._status.setStyleSheet("color: #fa4;")
