from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from blankforge.data.model import BoardModel
from blankforge.geometry.board import BoardMesh
from blankforge.ui.widgets.gl_viewport import GLViewport


class RenderedViewTab(QWidget):
    def __init__(self, model: BoardModel, model_changed: Signal, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._model_changed = model_changed
        self._wireframe = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._viewport = GLViewport(self)
        layout.addWidget(self._viewport, stretch=1)

        toolbar = QHBoxLayout()
        reset_btn = QPushButton("Reset View")
        reset_btn.setFixedWidth(100)
        reset_btn.clicked.connect(self._viewport.reset_view)
        toolbar.addWidget(reset_btn)

        self._shade_combo = QComboBox()
        self._shade_combo.addItems(["Solid", "Wireframe"])
        self._shade_combo.currentIndexChanged.connect(self._on_shade_changed)
        toolbar.addWidget(QLabel("Shading:"))
        toolbar.addWidget(self._shade_combo)

        from blankforge.renderer.blender import BlenderRenderer
        blender = BlenderRenderer()
        if blender.is_available():
            render_btn = QPushButton("Render with Blender…")
            render_btn.clicked.connect(self._render_blender)
            toolbar.addWidget(render_btn)

        toolbar.addStretch()
        layout.addLayout(toolbar)

    def update_mesh(self, mesh: BoardMesh) -> None:
        self._viewport.update_mesh(mesh)

    def _on_shade_changed(self, idx: int) -> None:
        self._viewport.set_wireframe(idx == 1)

    def _render_blender(self) -> None:
        from pathlib import Path
        from PySide6.QtWidgets import QFileDialog
        from blankforge.renderer.blender import BlenderRenderer
        from blankforge.geometry.board import BoardGeometryBuilder

        path, _ = QFileDialog.getSaveFileName(self, "Save Render", "", "PNG Images (*.png)")
        if not path:
            return
        mesh, _ = BoardGeometryBuilder(use_occt=False).build(self._model, resolution=60)
        renderer = BlenderRenderer()
        try:
            renderer.render(mesh, Path(path), width=1920, height=1080)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Blender Render Failed", str(e))
