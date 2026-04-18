from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from blankforge.data.model import BoardModel
from blankforge.data.serializer import SurfboardSerializer
from blankforge.ui.widgets.value_sliders import LabeledSlider


class ExportTab(QWidget):
    def __init__(self, model: BoardModel, model_changed: Signal, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._model_changed = model_changed
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(QLabel("<h2>Export</h2>"))

        # JSON
        json_box = QGroupBox(".surfboard (JSON — native format)")
        jl = QVBoxLayout(json_box)
        self._json_path, json_browse, json_export = self._path_row(jl, "surfboard")
        json_export.setText("Save .surfboard")
        json_export.clicked.connect(self._export_json)
        layout.addWidget(json_box)

        # STL
        stl_box = QGroupBox("STL (3D printing / CNC)")
        sl = QVBoxLayout(stl_box)
        self._stl_res = LabeledSlider("Resolution", 10, 100, decimals=0)
        self._stl_res.set_value(50)
        sl.addWidget(self._stl_res)
        self._stl_path, stl_browse, stl_export = self._path_row(sl, "stl")
        stl_export.setText("Export STL")
        stl_export.clicked.connect(self._export_stl)
        layout.addWidget(stl_box)

        # OBJ
        obj_box = QGroupBox("OBJ (general 3D interchange)")
        ol = QVBoxLayout(obj_box)
        self._obj_res = LabeledSlider("Resolution", 10, 100, decimals=0)
        self._obj_res.set_value(50)
        ol.addWidget(self._obj_res)
        self._obj_path, obj_browse, obj_export = self._path_row(ol, "obj")
        obj_export.setText("Export OBJ")
        obj_export.clicked.connect(self._export_obj)
        layout.addWidget(obj_box)

        self._status = QLabel("")
        layout.addWidget(self._status)
        layout.addStretch()

    def _path_row(self, parent_layout, ext: str):
        row = QHBoxLayout()
        path_edit = QLineEdit()
        path_edit.setPlaceholderText(f"Output path (.{ext})")
        browse = QPushButton("Browse…")
        export = QPushButton("Export")
        row.addWidget(path_edit, stretch=1)
        row.addWidget(browse)
        row.addWidget(export)
        parent_layout.addLayout(row)
        browse.clicked.connect(lambda: self._browse(path_edit, ext))
        return path_edit, browse, export

    def _browse(self, edit: QLineEdit, ext: str) -> None:
        filter_map = {
            "surfboard": "Surfboard Files (*.surfboard)",
            "stl": "STL Files (*.stl)",
            "obj": "OBJ Files (*.obj)",
        }
        path, _ = QFileDialog.getSaveFileName(self, "Choose output path", "", filter_map.get(ext, "*"))
        if path:
            edit.setText(path)

    def _export_json(self) -> None:
        path = self._json_path.text().strip()
        if not path:
            self._status.setText("Please specify a path.")
            return
        try:
            SurfboardSerializer.save(self._model, Path(path))
            self._status.setText(f"Saved: {path}")
            self._status.setStyleSheet("color: #4fa;")
        except Exception as e:
            self._status.setText(f"Error: {e}")
            self._status.setStyleSheet("color: #f44;")

    def _export_stl(self) -> None:
        path = self._stl_path.text().strip()
        if not path:
            self._status.setText("Please specify a path.")
            return
        try:
            SurfboardSerializer.export_stl(self._model, Path(path), int(self._stl_res.value()))
            self._status.setText(f"Exported STL: {path}")
            self._status.setStyleSheet("color: #4fa;")
        except Exception as e:
            self._status.setText(f"Error: {e}")
            self._status.setStyleSheet("color: #f44;")

    def _export_obj(self) -> None:
        path = self._obj_path.text().strip()
        if not path:
            self._status.setText("Please specify a path.")
            return
        try:
            SurfboardSerializer.export_obj(self._model, Path(path), int(self._obj_res.value()))
            self._status.setText(f"Exported OBJ: {path}")
            self._status.setStyleSheet("color: #4fa;")
        except Exception as e:
            self._status.setText(f"Error: {e}")
            self._status.setStyleSheet("color: #f44;")
