from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPainterPath, QPen, QColor
from PySide6.QtWidgets import (
    QButtonGroup, QDialog, QDialogButtonBox, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from blankforge.data.model import BoardModel

TEMPLATES = [
    ("Longboard", "longboard", "9'0\" — Smooth, flowing, for long rides"),
    ("Shortboard", "shortboard", "6'2\" — Performance, response, maneuverability"),
    ("Midlength", "midlength", "7'6\" — Versatile all-rounder"),
    ("Custom", "custom", "Start from scratch with your own dimensions"),
]


class TemplateSilhouette(QWidget):
    def __init__(self, template: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._template = template
        self.setFixedSize(80, 160)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        margin = 8
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
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        inner = QVBoxLayout(self)
        inner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.setSpacing(8)
        inner.addWidget(TemplateSilhouette(template), alignment=Qt.AlignmentFlag.AlignCenter)
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


class NewBoardDialog(QDialog):
    """Modal wizard shown when creating a new file."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Board — Choose Template")
        self.setMinimumWidth(720)
        self._selected: str = "shortboard"

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        layout.addWidget(QLabel(
            "<h2>New Board</h2>"
            "<p>Choose a starting template. You can adjust all parameters after creation.</p>"
        ))

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
            btn.clicked.connect(lambda _checked, t=template: self._on_select(t))

        self._buttons["shortboard"].setChecked(True)
        layout.addLayout(btn_layout)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_select(self, template: str) -> None:
        self._selected = template

    def selected_template(self) -> str:
        return self._selected
