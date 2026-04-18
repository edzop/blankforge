from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from blankforge.geometry.board import BoardMesh
from blankforge.renderer.opengl import CameraState, OpenGLBoardRenderer


class GLViewport(QOpenGLWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._renderer = OpenGLBoardRenderer()
        self._camera = CameraState()
        self._last_mouse: QPoint | None = None
        self._drag_mode: str = "none"  # "rotate" or "pan"
        self._mesh: BoardMesh | None = None
        self.setMinimumSize(300, 200)

    def initializeGL(self) -> None:
        self._renderer.initialize_gl()
        if self._mesh is not None:
            self._renderer.update_mesh(self._mesh)

    def resizeGL(self, w: int, h: int) -> None:
        from blankforge.renderer.opengl import _import_gl
        gl = _import_gl()
        if gl:
            gl.glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        self._renderer.paint(self.width(), self.height(), self._camera)

    def update_mesh(self, mesh: BoardMesh) -> None:
        self._mesh = mesh
        L = float(mesh.vertices[:, 0].max()) if len(mesh.vertices) > 0 else 2000.0
        self._camera.reset_for_board(L)
        if self.isValid():
            self.makeCurrent()
            self._renderer.update_mesh(mesh)
            self.doneCurrent()
        self.update()

    def camera(self) -> CameraState:
        return self._camera

    def reset_view(self) -> None:
        if self._mesh is not None:
            L = float(self._mesh.vertices[:, 0].max())
            self._camera.reset_for_board(L)
        else:
            self._camera = CameraState()
        self.update()

    def set_wireframe(self, enabled: bool) -> None:
        from blankforge.renderer.opengl import _import_gl
        gl = _import_gl()
        if gl:
            self.makeCurrent()
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE if enabled else gl.GL_FILL)
            self.doneCurrent()
        self.update()

    def mousePressEvent(self, event) -> None:
        self._last_mouse = event.pos()
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_mode = "rotate"
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._drag_mode = "pan"

    def mouseReleaseEvent(self, event) -> None:
        self._drag_mode = "none"
        self._last_mouse = None

    def mouseMoveEvent(self, event) -> None:
        if self._last_mouse is None or self._drag_mode == "none":
            return
        delta = event.pos() - self._last_mouse
        self._last_mouse = event.pos()
        dx, dy = delta.x(), delta.y()
        if self._drag_mode == "rotate":
            self._camera.azimuth -= dx * 0.5
            self._camera.elevation = max(-89, min(89, self._camera.elevation + dy * 0.3))
        elif self._drag_mode == "pan":
            import math
            az = math.radians(self._camera.azimuth)
            el = math.radians(self._camera.elevation)
            right = np.array([math.cos(az), -math.sin(az), 0], dtype=np.float32)
            up = np.array([
                -math.sin(el) * math.sin(az),
                -math.sin(el) * math.cos(az),
                math.cos(el),
            ], dtype=np.float32)
            pan_scale = self._camera.distance * 0.001
            self._camera.target += right * dx * pan_scale - up * dy * pan_scale
        self.update()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self._camera.distance = max(100, self._camera.distance * factor)
        self.update()
