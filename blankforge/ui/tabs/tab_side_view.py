from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QSplitter, QVBoxLayout, QWidget
from PySide6.QtCore import Qt

from blankforge.data.model import BoardModel
from blankforge.ui.widgets.control_point_editor import ControlPointEditor
from blankforge.ui.widgets.curve_viewport import CurveViewport


class RockerCanvas(CurveViewport):
    x_label = "Position along board (mm)"
    y_label = "Rocker (mm)"
    y_flip = True  # positive rocker = nose/tail UP = higher on screen

    def _overlay_lines(self) -> list[str]:
        pts = self._curve.sorted_points()
        if not pts:
            return []
        max_r = max(p.value_mm for p in pts)
        return [
            f"Max Rocker",
            f"{max_r:.1f} mm",
            f"{max_r / 25.4:.2f} in",
        ]


class ThicknessCanvas(CurveViewport):
    x_label = "Position along board (mm)"
    y_label = "Half-Thickness (mm)"

    def _overlay_lines(self) -> list[str]:
        pts = self._curve.sorted_points()
        if not pts:
            return []
        max_ht = max(p.value_mm for p in pts)
        max_t = max_ht * 2  # stored as half-thickness
        return [
            f"Max Thickness",
            f"{max_t:.1f} mm",
            f"{max_t / 25.4:.2f} in",
        ]


class SideViewTab(QWidget):
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
        layout.addWidget(QLabel("<b>Side View — Rocker &amp; Thickness Curves</b>"))

        splitter = QSplitter(Qt.Orientation.Vertical)

        rocker_widget = QWidget()
        rocker_layout = QVBoxLayout(rocker_widget)
        rocker_layout.setContentsMargins(0, 0, 0, 0)
        rocker_layout.addWidget(QLabel("  Rocker (mm)"))
        self._rocker_editor = ControlPointEditor(
            curve_data=self._model.curves.rocker,
            viewport_class=RockerCanvas,
            board_length=self._model.parameters.length_mm,
            max_value=200.0,
        )
        rocker_layout.addWidget(self._rocker_editor)
        splitter.addWidget(rocker_widget)

        thick_widget = QWidget()
        thick_layout = QVBoxLayout(thick_widget)
        thick_layout.setContentsMargins(0, 0, 0, 0)
        thick_layout.addWidget(QLabel("  Half-Thickness (mm)"))
        self._thick_editor = ControlPointEditor(
            curve_data=self._model.curves.thickness,
            viewport_class=ThicknessCanvas,
            board_length=self._model.parameters.length_mm,
            max_value=150.0,
        )
        thick_layout.addWidget(self._thick_editor)
        splitter.addWidget(thick_widget)

        splitter.setSizes([1, 1])
        layout.addWidget(splitter, stretch=1)

        self._rocker_editor.curve_changed.connect(self._on_rocker_changed)
        self._thick_editor.curve_changed.connect(self._on_thick_changed)

    def refresh_from_model(self) -> None:
        self._updating = True
        L = self._model.parameters.length_mm
        self._rocker_editor.set_board_length(L)
        self._thick_editor.set_board_length(L)
        self._rocker_editor.set_curve_data(self._model.curves.rocker)
        self._thick_editor.set_curve_data(self._model.curves.thickness)
        self._updating = False

    def _on_rocker_changed(self) -> None:
        if not self._updating:
            self._model_changed.emit()

    def _on_thick_changed(self) -> None:
        if not self._updating:
            self._model_changed.emit()
