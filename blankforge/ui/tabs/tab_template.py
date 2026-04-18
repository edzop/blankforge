from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QPainterPath, QPen, QColor
from PySide6.QtWidgets import (
    QButtonGroup, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

from blankforge.data.model import BoardModel

TEMPLATES = [
    ("Longboard", "longboard", "9'0\" — Smooth, flowing, for long rides"),
    ("Shortboard", "shortboard", "6'2\" — Performance, response, maneuverability"),
    ("Midlength", "midlength", "7'6\" — Versatile all-rounder"),
    ("Custom", "custom", "Start from scratch with your own dimensions"),
]


class TemplateSilhouette(QWidget):
    """Small widget that draws a simplified top-view board silhouette."""

    def __init__(self, template: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._template = template
        self.setFixedSize(80, 160)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        margin = 8

        # Silhouette proportions by template
        props = {
            "longboard":  {"nose_w": 0.72, "wide_w": 1.0, "wide_y": 0.42, "tail_w": 0.68},
            "shortboard": {"nose_w": 0.45, "wide_w": 1.0, "wide_y": 0.40, "tail_w": 0.52},
            "midlength":  {"nose_w": 0.60, "wide_w": 1.0, "wide_y": 0.43, "tail_w": 0.61},
            "custom":     {"nose_w": 0.55, "wide_w": 1.0, "wide_y": 0.45, "tail_w": 0.55},
        }
        pr = props.get(self._template, props["shortboard"])
        cx = w / 2

        def x(frac): return cx + frac * (w / 2 - margin)
        def nx(frac): return cx - frac * (w / 2 - margin)

        # Y positions: top = nose, bottom = tail
        y_nose = margin
        y_wide = h * pr["wide_y"]
        y_tail = h - margin

        path = QPainterPath()
        path.moveTo(cx, y_nose)
        path.cubicTo(x(pr["nose_w"] * 0.5), y_nose + (y_wide - y_nose) * 0.2,
                     x(pr["wide_w"]), y_wide - 20,
                     x(pr["wide_w"]), y_wide)
        path.cubicTo(x(pr["wide_w"]), y_wide + 20,
                     x(pr["tail_w"]), y_tail - (y_tail - y_wide) * 0.3,
                     cx, y_tail)
        path.cubicTo(nx(pr["tail_w"]), y_tail - (y_tail - y_wide) * 0.3,
                     nx(pr["wide_w"]), y_wide + 20,
                     nx(pr["wide_w"]), y_wide)
        path.cubicTo(nx(pr["wide_w"]), y_wide - 20,
                     nx(pr["nose_w"] * 0.5), y_nose + (y_wide - y_nose) * 0.2,
                     cx, y_nose)
        path.closeSubpath()

        p.fillPath(path, QColor(70, 120, 180, 200))
        pen = QPen(QColor(120, 180, 240))
        pen.setWidth(2)
        p.setPen(pen)
        p.drawPath(path)


class TemplateButton(QPushButton):
    def __init__(self, name: str, template: str, desc: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._template = template
        self.setCheckable(True)
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        inner = QVBoxLayout(self)
        inner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.setSpacing(8)

        sil = TemplateSilhouette(template)
        inner.addWidget(sil, alignment=Qt.AlignmentFlag.AlignCenter)

        lbl = QLabel(f"<b>{name}</b>")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(lbl)

        desc_lbl = QLabel(desc)
        desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("color: #aaa; font-size: 10px;")
        inner.addWidget(desc_lbl)

        self.setStyleSheet("""
            QPushButton { border: 2px solid #444; border-radius: 8px; padding: 8px; background: #252830; }
            QPushButton:checked { border: 2px solid #4fa; background: #1a2a2a; }
            QPushButton:hover { border-color: #777; }
        """)


class TemplateTab(QWidget):
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

        title = QLabel("<h2>Board Template</h2><p>Select a starting shape. This sets default dimensions and curve presets.</p>")
        title.setWordWrap(True)
        layout.addWidget(title)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        self._buttons: dict[str, TemplateButton] = {}

        for name, template, desc in TEMPLATES:
            btn = TemplateButton(name, template, desc)
            self._btn_group.addButton(btn)
            self._buttons[template] = btn
            btn_layout.addWidget(btn)
            btn.clicked.connect(lambda checked, t=template: self._on_template_selected(t))

        layout.addLayout(btn_layout)
        layout.addStretch()

        self.refresh_from_model()

    def refresh_from_model(self) -> None:
        self._updating = True
        current = self._model.meta.template
        if current in self._buttons:
            self._buttons[current].setChecked(True)
        self._updating = False

    def _on_template_selected(self, template: str) -> None:
        if self._updating:
            return
        new_model = BoardModel.from_template(template)
        self._model.meta = new_model.meta
        self._model.parameters = new_model.parameters
        self._model.tail = new_model.tail
        self._model.curves = new_model.curves
        self._model_changed.emit()
