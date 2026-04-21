from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from blankforge.geometry.board import BoardMesh
from blankforge.renderer.opengl import CameraState, OpenGLBoardRenderer


# Sentinel: distinguishes "no pending update" from "pending update to None"
class _Unset:
    pass
_UNSET = _Unset()


class _GizmoOverlay(QWidget):
    """Transparent overlay widget that draws the XYZ orientation gizmo."""

    _AXES = [
        ((1, 0, 0), QColor(220, 55, 55),  "X"),
        ((0, 1, 0), QColor(55, 195, 55),  "Y"),
        ((0, 0, 1), QColor(55, 110, 230), "Z"),
    ]

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._camera: CameraState | None = None
        self.setFixedSize(84, 84)

    def set_camera(self, camera: CameraState) -> None:
        self._camera = camera
        self.update()

    def paintEvent(self, event) -> None:
        if self._camera is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Subtle dark backing circle
        p.setBrush(QColor(18, 20, 26, 160))
        p.setPen(QPen(QColor(60, 70, 90, 120), 1.0))
        p.drawEllipse(2, 2, 80, 80)

        az = math.radians(self._camera.azimuth)
        el = math.radians(self._camera.elevation)

        # View-space right and up vectors (derived from camera angles)
        r = (-math.cos(az), math.sin(az), 0.0)
        u = (-math.sin(az) * math.sin(el), -math.cos(az) * math.sin(el), math.cos(el))
        fwd = (math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el))

        cx, cy, arm = 42.0, 42.0, 27.0

        def project(axis):
            sx = sum(r[i] * axis[i] for i in range(3))
            sy = -sum(u[i] * axis[i] for i in range(3))
            depth = sum(fwd[i] * axis[i] for i in range(3))
            return sx, sy, depth

        # Sort back-to-front so front-facing axes draw on top
        axes_proj = [(proj := project(ax), col, lbl, ax)
                     for ax, col, lbl in self._AXES]
        axes_proj.sort(key=lambda t: t[0][2])  # ascending depth = back first

        font = QFont("sans-serif", 8, QFont.Weight.Bold)
        p.setFont(font)

        for (sx, sy, depth), col, lbl, _ in axes_proj:
            alpha = 255 if depth >= -0.15 else 110
            ex, ey = cx + sx * arm, cy + sy * arm

            # Line
            line_col = QColor(col.red(), col.green(), col.blue(), alpha)
            p.setPen(QPen(line_col, 2.5))
            p.drawLine(QPointF(cx, cy), QPointF(ex, ey))

            # Dot at tip
            p.setBrush(line_col)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(ex, ey), 4.5, 4.5)

            # Label
            p.setPen(QColor(col.red(), col.green(), col.blue(), min(255, alpha + 60)))
            p.drawText(int(ex) + 6, int(ey) + 5, lbl)


def _compute_thickness_colors(
    mesh: BoardMesh, sensitivity: float = 1.0, thickness_axis: str = "z"
) -> np.ndarray:
    """Per-vertex thickness mapped to blue=thick, red=thin.

    thickness_axis: which axis measures thickness (span). The remaining two
    axes form the spatial binning grid. Use 'z' for boards (binning X,Y) or
    'y' for fins (binning X,Z — Y is foil depth).
    """
    verts = mesh.vertices
    n = len(verts)

    axis_idx = {"x": 0, "y": 1, "z": 2}[thickness_axis]
    bin_axes = [i for i in (0, 1, 2) if i != axis_idx]
    a = verts[:, bin_axes[0]]
    b = verts[:, bin_axes[1]]
    t_vals = verts[:, axis_idx]

    nx, ny = 150, 80
    alo, ahi = float(a.min()), float(a.max())
    blo, bhi = float(b.min()), float(b.max())
    a_range = ahi - alo if ahi > alo else 1.0
    b_range = bhi - blo if bhi > blo else 1.0

    ia = np.clip(((a - alo) / a_range * nx).astype(int), 0, nx - 1)
    ib = np.clip(((b - blo) / b_range * ny).astype(int), 0, ny - 1)
    flat = ia * ny + ib
    n_cells = nx * ny

    t_top = np.full(n_cells, -np.inf)
    t_bot = np.full(n_cells,  np.inf)
    np.maximum.at(t_top, flat, t_vals)
    np.minimum.at(t_bot, flat, t_vals)

    thickness_cells = np.where(
        np.isfinite(t_top) & np.isfinite(t_bot),
        np.maximum(t_top - t_bot, 0.0), 0.0,
    )
    thick = thickness_cells[flat]
    t_lo, t_hi = float(thick.min()), float(thick.max())
    if t_hi > t_lo:
        t = (thick - t_lo) / (t_hi - t_lo)
    else:
        t = np.ones(n, dtype=np.float32)

    thinness = np.clip((1.0 - t) / max(sensitivity, 0.01), 0.0, 1.0).astype(np.float32)
    colors = np.zeros((n, 3), dtype=np.float32)
    colors[:, 0] = thinness          # red when thin
    colors[:, 2] = 1.0 - thinness   # blue when thick
    return colors


def _compute_curvature_colors(mesh: BoardMesh, sensitivity: float = 1.0) -> np.ndarray:
    """Per-vertex mean curvature mapped to blue→red. sensitivity in (0,1]: lower = amplifies subtle differences."""
    norms = mesh.normals
    tris = mesh.triangles
    n_verts = len(mesh.vertices)

    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    a, b = edges[:, 0], edges[:, 1]
    dots = np.clip((norms[a] * norms[b]).sum(axis=1), -1.0, 1.0)
    angles = np.arccos(dots)

    deviation = np.zeros(n_verts, dtype=np.float64)
    count = np.zeros(n_verts, dtype=np.float64)
    np.add.at(deviation, a, angles)
    np.add.at(count, a, 1.0)
    np.add.at(deviation, b, angles)
    np.add.at(count, b, 1.0)

    curv = np.where(count > 0, deviation / count, 0.0).astype(np.float32)
    lo, hi = curv.min(), curv.max()
    if hi > lo:
        curv = (curv - lo) / (hi - lo)

    # sensitivity < 1 raises the ceiling: values above `sensitivity` saturate to red
    curv = np.clip(curv / max(sensitivity, 0.01), 0.0, 1.0)

    colors = np.zeros((n_verts, 3), dtype=np.float32)
    colors[:, 0] = curv          # red channel
    colors[:, 2] = 1.0 - curv   # blue channel

    return colors


class GLViewport(QOpenGLWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._renderer = OpenGLBoardRenderer()
        self._camera = CameraState()
        self._last_mouse: QPoint | None = None
        self._drag_mode: str = "none"  # "rotate" or "pan"
        self._mesh: BoardMesh | None = None
        self.setMinimumSize(300, 200)

        # Pending GPU state — flushed at the top of paintGL / initializeGL
        # so all GL calls happen inside Qt-managed callbacks where the
        # context is guaranteed current. Never call makeCurrent() manually.
        self._pending_mesh: BoardMesh | None = None
        self._pending_wireframe_lines: np.ndarray | None | _Unset = _UNSET
        self._pending_vertex_colors: np.ndarray | None | _Unset = _UNSET
        self._pending_solid_alpha: float | _Unset = _UNSET
        self._pending_heatmap_blend: float | _Unset = _UNSET
        self._pending_line_alpha: float | _Unset = _UNSET

        self._gizmo = _GizmoOverlay(self)
        self._gizmo.set_camera(self._camera)
        self._gizmo.raise_()
        self._controls: QWidget | None = None

    # ------------------------------------------------------------------
    # Qt GL callbacks — context is current inside all three of these
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        self._renderer.initialize_gl()
        self._flush_pending()

    def resizeGL(self, w: int, h: int) -> None:
        from blankforge.renderer.opengl import _import_gl
        gl = _import_gl()
        if gl:
            gl.glViewport(0, 0, w, h)
        self._reposition_overlays(w)

    def paintGL(self) -> None:
        self._flush_pending()
        self._renderer.paint(self.width(), self.height(), self._camera)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_pending(self) -> None:
        """Apply all queued GPU uploads. Must only be called from initializeGL or paintGL."""
        if self._pending_mesh is not None:
            self._renderer.update_mesh(self._pending_mesh)
            self._pending_mesh = None
        if not isinstance(self._pending_wireframe_lines, _Unset):
            self._renderer.update_wireframe_lines(self._pending_wireframe_lines)
            self._pending_wireframe_lines = _UNSET
        if not isinstance(self._pending_vertex_colors, _Unset):
            self._renderer.update_vertex_colors(self._pending_vertex_colors)
            self._pending_vertex_colors = _UNSET
        if not isinstance(self._pending_solid_alpha, _Unset):
            self._renderer.set_solid_alpha(self._pending_solid_alpha)
            self._pending_solid_alpha = _UNSET
        if not isinstance(self._pending_heatmap_blend, _Unset):
            self._renderer.set_heatmap_blend(self._pending_heatmap_blend)
            self._pending_heatmap_blend = _UNSET
        if not isinstance(self._pending_line_alpha, _Unset):
            self._renderer.set_line_alpha(self._pending_line_alpha)
            self._pending_line_alpha = _UNSET

    def _refresh(self) -> None:
        self._gizmo.set_camera(self._camera)
        self.update()

    def _aspect(self) -> float:
        w, h = self.width(), self.height()
        if w < 10 or h < 10:
            return 1.5  # widget not yet laid out; use a safe default
        return w / h

    def set_controls_widget(self, widget: QWidget) -> None:
        self._controls = widget
        widget.setParent(self)
        widget.raise_()
        self._reposition_overlays(self.width())

    def _reposition_overlays(self, w: int) -> None:
        margin = 8
        gw = self._gizmo.width()
        self._gizmo.move(w - gw - margin, margin)
        if self._controls is not None:
            self._controls.adjustSize()
            cw = self._controls.width()
            self._controls.move(w - cw - margin, margin + self._gizmo.height() + margin)

    # ------------------------------------------------------------------
    # Public API — queue state, never touch GL directly
    # ------------------------------------------------------------------

    def update_mesh(self, mesh: BoardMesh) -> None:
        # Only auto-frame the camera the first time a mesh is set.
        # Later updates (parameter tweaks, edits) preserve the current view —
        # use the Reset / Fit / preset buttons to re-frame explicitly.
        is_first = self._mesh is None
        self._mesh = mesh
        if is_first:
            L = float(mesh.vertices[:, 0].max()) if len(mesh.vertices) > 0 else 2000.0
            self._camera.reset_for_board(L, self._aspect())
        self._pending_mesh = mesh
        self._refresh()

    def camera(self) -> CameraState:
        return self._camera

    def reset_view(self) -> None:
        if self._mesh is not None:
            L = float(self._mesh.vertices[:, 0].max())
            self._camera.reset_for_board(L, self._aspect())
        else:
            self._camera = CameraState()
        self._refresh()

    def fit_view(self) -> None:
        if self._mesh is not None:
            L = float(self._mesh.vertices[:, 0].max())
            self._camera.fit_board(L, self._aspect())
        self._refresh()

    def set_preset_view(self, azimuth: float, elevation: float) -> None:
        self._camera.azimuth = azimuth
        self._camera.elevation = elevation
        self._camera.use_ortho = True
        if self._mesh is not None:
            L = float(self._mesh.vertices[:, 0].max())
            self._camera.fit_board(L, self._aspect())
        self._refresh()

    def set_solid_alpha(self, alpha: float) -> None:
        self._pending_solid_alpha = alpha
        self.update()

    def set_line_alpha(self, alpha: float) -> None:
        self._pending_line_alpha = alpha
        self.update()

    def set_heatmap_blend(
        self, blend: float, sensitivity: float = 1.0,
        mode: str = "curvature", thickness_axis: str = "z",
    ) -> None:
        self._pending_heatmap_blend = blend
        if blend > 0.0 and self._mesh is not None:
            if mode == "thickness":
                self._pending_vertex_colors = _compute_thickness_colors(
                    self._mesh, sensitivity, thickness_axis
                )
            else:
                self._pending_vertex_colors = _compute_curvature_colors(self._mesh, sensitivity)
        self.update()

    def update_wireframe_lines(self, verts: np.ndarray | None) -> None:
        self._pending_wireframe_lines = verts
        self.update()

    # ------------------------------------------------------------------
    # Mouse / wheel interaction
    # ------------------------------------------------------------------

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
            self._camera.use_ortho = False
            self._camera.azimuth -= dx * 0.5
            self._camera.elevation = max(-89, min(89, self._camera.elevation + dy * 0.3))
        elif self._drag_mode == "pan":
            az = math.radians(self._camera.azimuth)
            el = math.radians(self._camera.elevation)
            right = np.array([math.cos(az), -math.sin(az), 0], dtype=np.float32)
            up = np.array([
                -math.sin(el) * math.sin(az),
                -math.sin(el) * math.cos(az),
                math.cos(el),
            ], dtype=np.float32)
            pan_scale = self._camera.distance * 0.001
            # Grab-and-drag convention: mouse down → scene moves down with the cursor
            self._camera.target += right * dx * pan_scale + up * dy * pan_scale
        self._refresh()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self._camera.distance = max(100, self._camera.distance * factor)
        self._refresh()
