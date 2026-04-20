from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from blankforge.geometry.board import BoardMesh
from blankforge.renderer.base import AbstractRenderer, RendererOutput

# OpenGL / PySide6 imports are deferred to avoid import errors at module load
_gl_initialized = False
_gl = None


def _import_gl():
    global _gl, _gl_initialized
    if _gl_initialized:
        return _gl
    try:
        import OpenGL.GL as gl
        _gl = gl
    except ImportError:
        _gl = None
    _gl_initialized = True
    return _gl


VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;

uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat3 uNormalMatrix;

out vec3 vNormal;
out vec3 vFragPos;
out vec3 vVertexColor;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vFragPos = vec3(uModel * vec4(aPos, 1.0));
    vNormal = normalize(uNormalMatrix * aNormal);
    vVertexColor = aColor;
}
"""

# Separate line shader — vertex + geometry + fragment.
# The geometry shader expands each line segment into a screen-space quad
# so lines have a consistent pixel width regardless of depth.
LINE_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 2) in vec3 aColor;

uniform mat4 uMVP;
out vec3 vColor;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vColor = aColor;
}
"""

LINE_GEOMETRY_SHADER = """
#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

uniform vec2 uViewport;
uniform float uLineWidth;

in vec3 vColor[];
out vec3 gColor;

void main() {
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;

    // NDC coordinates of each endpoint
    vec2 ndc0 = p0.xy / p0.w;
    vec2 ndc1 = p1.xy / p1.w;

    // Screen-space direction — skip degenerate segments (nose/tail tip,
    // or two identical points) to avoid NaN from normalize(zero).
    vec2 delta = (ndc1 - ndc0) * uViewport;
    if (dot(delta, delta) < 0.0001) return;

    vec2 dir  = normalize(delta);
    vec2 perp = vec2(-dir.y, dir.x) * uLineWidth / uViewport;

    gColor = vColor[0];
    gl_Position = vec4(p0.xy + perp * p0.w, p0.zw); EmitVertex();
    gl_Position = vec4(p0.xy - perp * p0.w, p0.zw); EmitVertex();
    gColor = vColor[1];
    gl_Position = vec4(p1.xy + perp * p1.w, p1.zw); EmitVertex();
    gl_Position = vec4(p1.xy - perp * p1.w, p1.zw); EmitVertex();
    EndPrimitive();
}
"""

LINE_FRAGMENT_SHADER = """
#version 330 core
in vec3 gColor;
uniform float uLineAlpha;
out vec4 FragColor;
void main() {
    FragColor = vec4(gColor, uLineAlpha);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 vNormal;
in vec3 vFragPos;
in vec3 vVertexColor;

uniform vec3 uCameraPos;
uniform vec3 uSurfaceColor;
uniform float uHeatmapBlend;  // 0 = surface colour only, 1 = full heatmap
uniform float uAlpha;         // overall mesh opacity

// 3-point lighting
uniform vec3 uKeyDir;
uniform vec3 uKeyColor;
uniform float uKeyStrength;

uniform vec3 uFillDir;
uniform vec3 uFillColor;
uniform float uFillStrength;

uniform vec3 uRimDir;
uniform vec3 uRimColor;
uniform float uRimStrength;

out vec4 FragColor;

vec3 light_contrib(vec3 N, vec3 V, vec3 L, vec3 lcolor, float strength,
                   vec3 surf, float diff_power, float spec_gloss, float spec_strength) {
    vec3 Ln = normalize(L);
    float diff = max(dot(N, Ln), 0.0);
    vec3 H = normalize(Ln + V);
    float spec = pow(max(dot(N, H), 0.0), spec_gloss) * spec_strength;
    return lcolor * strength * (surf * diff * diff_power + vec3(spec));
}

void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCameraPos - vFragPos);

    vec3 surf = mix(uSurfaceColor, vVertexColor, uHeatmapBlend);

    vec3 ambient = surf * vec3(0.03, 0.04, 0.07);
    vec3 key  = light_contrib(N, V, uKeyDir,  uKeyColor,  uKeyStrength,  surf, 1.0, 128.0, 0.65);
    vec3 fill = light_contrib(N, V, uFillDir, uFillColor, uFillStrength, surf, 0.7, 1.0,   0.0);
    vec3 rim  = light_contrib(N, V, uRimDir,  uRimColor,  uRimStrength,  surf, 0.5, 40.0,  0.25);

    vec3 result = ambient + key + fill + rim;
    result = result / (result + vec3(0.70));
    result = pow(clamp(result, 0.0, 1.0), vec3(1.0 / 2.2));
    FragColor = vec4(result, uAlpha);
}
"""


@dataclass
class CameraState:
    azimuth: float = 45.0
    elevation: float = 20.0
    distance: float = 3000.0
    target: np.ndarray = field(default_factory=lambda: np.array([940.0, 0.0, 50.0], dtype=np.float32))
    fov: float = 45.0
    use_ortho: bool = False

    def view_matrix(self) -> np.ndarray:
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        eye = self.target + self.distance * np.array([
            math.cos(el) * math.sin(az),
            math.cos(el) * math.cos(az),
            math.sin(el),
        ], dtype=np.float32)
        return _look_at(eye, self.target, np.array([0, 0, 1], dtype=np.float32))

    def projection_matrix(self, aspect: float) -> np.ndarray:
        if self.use_ortho:
            h = self.distance * math.tan(math.radians(self.fov / 2))
            w = h * aspect
            return _ortho(-w, w, -h, h, 10.0, 20000.0)
        return _perspective(self.fov, aspect, 10.0, 20000.0)

    def eye_position(self) -> np.ndarray:
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        return self.target + self.distance * np.array([
            math.cos(el) * math.sin(az),
            math.cos(el) * math.cos(az),
            math.sin(el),
        ], dtype=np.float32)

    def reset_for_board(self, length_mm: float, aspect: float = 1.5) -> None:
        self.target = np.array([length_mm / 2, 0.0, 30.0], dtype=np.float32)
        half_fov = math.tan(math.radians(self.fov / 2.0))
        d_for_length = (length_mm / 2.0) / (max(aspect, 0.01) * half_fov)
        self.distance = d_for_length * 1.15  # slightly looser for perspective view
        self.azimuth = 45.0
        self.elevation = 25.0
        self.use_ortho = False

    def fit_board(self, length_mm: float, aspect: float = 1.5) -> None:
        self.target = np.array([length_mm / 2, 0.0, 30.0], dtype=np.float32)
        # Fit so the board's length spans the view width with a small margin.
        half_fov = math.tan(math.radians(self.fov / 2.0))
        d_for_length = (length_mm / 2.0) / (max(aspect, 0.01) * half_fov)
        self.distance = d_for_length * 1.08  # 8% margin


class OpenGLBoardRenderer(AbstractRenderer):
    def __init__(self) -> None:
        self._vao = None
        self._vbo_verts = None
        self._vbo_norms = None
        self._vbo_colors = None
        self._ebo = None
        self._vao_lines = None
        self._vbo_lines = None
        self._n_line_verts = 0
        self._shader_program = None
        self._line_shader_program = None
        self._line_width_px = 1.8  # screen-space pixel width for wireframe lines
        self._n_indices = 0
        self._n_verts = 0
        # Blendable layer intensities
        self._solid_alpha: float = 1.0      # mesh opacity  0-1
        self._heatmap_blend: float = 0.0    # heatmap mix   0-1
        self._line_alpha: float = 0.0       # wireframe opacity 0-1
        self._initialized = False

    def name(self) -> str:
        return "PyOpenGL"

    def is_available(self) -> bool:
        return _import_gl() is not None

    def initialize_gl(self) -> None:
        gl = _import_gl()
        if gl is None:
            return
        self._shader_program = _compile_program(gl, VERTEX_SHADER, FRAGMENT_SHADER)
        self._line_shader_program = _compile_program(
            gl, LINE_VERTEX_SHADER, LINE_FRAGMENT_SHADER,
            geom_src=LINE_GEOMETRY_SHADER,
        )
        self._vao = gl.glGenVertexArrays(1)
        self._vbo_verts = gl.glGenBuffers(1)
        self._vbo_norms = gl.glGenBuffers(1)
        self._vbo_colors = gl.glGenBuffers(1)
        self._ebo = gl.glGenBuffers(1)
        self._vao_lines = gl.glGenVertexArrays(1)
        self._vbo_lines = gl.glGenBuffers(1)
        self._initialized = True

    def update_mesh(self, mesh: BoardMesh) -> None:
        if not self._initialized:
            self.initialize_gl()
        gl = _import_gl()
        if gl is None:
            return
        self._n_indices = len(mesh.triangles) * 3
        self._n_verts = len(mesh.vertices)
        self._use_vertex_colors = False

        gl.glBindVertexArray(self._vao)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_verts)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, mesh.vertices.nbytes, mesh.vertices, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_norms)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, mesh.normals.nbytes, mesh.normals, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        default_colors = np.ones((self._n_verts, 3), dtype=np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_colors)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, default_colors.nbytes, default_colors, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        indices = mesh.triangles.astype(np.uint32)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

        gl.glBindVertexArray(0)

    def update_wireframe_lines(self, verts: np.ndarray | None) -> None:
        """Upload line vertex pairs (Nx3 float32, drawn as GL_LINES)."""
        gl = _import_gl()
        if gl is None or not self._initialized:
            return
        if verts is None or len(verts) == 0:
            self._n_line_verts = 0
            return
        verts = verts.astype(np.float32)
        dummy_norms = np.zeros_like(verts)
        dummy_colors = np.ones_like(verts)

        gl.glBindVertexArray(self._vao_lines)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_lines)
        stride = 9 * 4  # 3 pos + 3 norm + 3 color, interleaved
        combined = np.hstack([verts, dummy_norms, dummy_colors]).astype(np.float32)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, combined.nbytes, combined, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
        import ctypes
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(24))
        gl.glBindVertexArray(0)
        self._n_line_verts = len(verts)

    def set_solid_alpha(self, alpha: float) -> None:
        self._solid_alpha = float(alpha)

    def set_heatmap_blend(self, blend: float) -> None:
        self._heatmap_blend = float(blend)

    def set_line_alpha(self, alpha: float) -> None:
        self._line_alpha = float(alpha)

    def update_vertex_colors(self, colors: np.ndarray | None) -> None:
        gl = _import_gl()
        if gl is None or not self._initialized:
            return
        if colors is None:
            return
        gl.glBindVertexArray(self._vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_colors)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors.nbytes, colors.astype(np.float32), gl.GL_DYNAMIC_DRAW)
        gl.glBindVertexArray(0)

    def paint(self, width: int, height: int, camera: CameraState) -> None:
        if not self._initialized or self._n_indices == 0:
            return
        gl = _import_gl()
        if gl is None:
            return

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.12, 0.12, 0.15, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        aspect = width / max(height, 1)
        proj = camera.projection_matrix(aspect)
        view = camera.view_matrix()
        model = np.eye(4, dtype=np.float32)
        mvp = proj @ view @ model
        normal_mat = np.linalg.inv(model[:3, :3]).T.astype(np.float32)

        # --- Draw mesh ---
        if self._solid_alpha > 0.0:
            gl.glUseProgram(self._shader_program)
            _set_uniform_mat4(gl, self._shader_program, "uMVP", mvp)
            _set_uniform_mat4(gl, self._shader_program, "uModel", model)
            _set_uniform_mat3(gl, self._shader_program, "uNormalMatrix", normal_mat)
            _set_uniform_vec3(gl, self._shader_program, "uCameraPos", camera.eye_position())
            _set_uniform_vec3(gl, self._shader_program, "uSurfaceColor",
                              np.array([0.86, 0.83, 0.76], dtype=np.float32))
            _set_uniform_float(gl, self._shader_program, "uHeatmapBlend", self._heatmap_blend)
            _set_uniform_float(gl, self._shader_program, "uAlpha", self._solid_alpha)

            _set_uniform_vec3(gl, self._shader_program, "uKeyDir",
                              _norm(np.array([-0.50, -0.60, 0.90], dtype=np.float32)))
            _set_uniform_vec3(gl, self._shader_program, "uKeyColor",
                              np.array([1.00, 0.92, 0.72], dtype=np.float32))
            _set_uniform_float(gl, self._shader_program, "uKeyStrength", 1.1)
            _set_uniform_vec3(gl, self._shader_program, "uFillDir",
                              _norm(np.array([0.75, 0.35, 0.30], dtype=np.float32)))
            _set_uniform_vec3(gl, self._shader_program, "uFillColor",
                              np.array([0.45, 0.58, 0.82], dtype=np.float32))
            _set_uniform_float(gl, self._shader_program, "uFillStrength", 0.22)
            _set_uniform_vec3(gl, self._shader_program, "uRimDir",
                              _norm(np.array([0.20, 0.90, -0.35], dtype=np.float32)))
            _set_uniform_vec3(gl, self._shader_program, "uRimColor",
                              np.array([0.60, 0.75, 1.00], dtype=np.float32))
            _set_uniform_float(gl, self._shader_program, "uRimStrength", 0.50)

            transparent = self._solid_alpha < 0.999
            if transparent:
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
                gl.glDepthMask(gl.GL_FALSE)
            gl.glBindVertexArray(self._vao)
            gl.glDrawElements(gl.GL_TRIANGLES, self._n_indices, gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)
            if transparent:
                gl.glDepthMask(gl.GL_TRUE)
                gl.glDisable(gl.GL_BLEND)

        # --- Draw wireframe lines ---
        if self._n_line_verts > 0 and self._line_alpha > 0.0:
            gl.glUseProgram(self._line_shader_program)
            _set_uniform_mat4(gl, self._line_shader_program, "uMVP", mvp)
            _set_uniform_vec2(gl, self._line_shader_program, "uViewport",
                              np.array([width, height], dtype=np.float32))
            _set_uniform_float(gl, self._line_shader_program, "uLineWidth", self._line_width_px)
            _set_uniform_float(gl, self._line_shader_program, "uLineAlpha", self._line_alpha)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glBindVertexArray(self._vao_lines)
            gl.glDrawArrays(gl.GL_LINES, 0, self._n_line_verts)
            gl.glBindVertexArray(0)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_BLEND)

    def render(self, mesh, output_path, width=1920, height=1080, view="perspective",
               background_color=(0.15, 0.15, 0.15)) -> RendererOutput:
        # For headless rendering we delegate to the Qt offscreen path
        img = self.render_to_image(mesh, width, height, view)
        if img is not None:
            _save_rgba_png(img, output_path)
        return RendererOutput(image_path=Path(output_path), width=width, height=height)

    def render_to_image(self, mesh: BoardMesh, width: int, height: int,
                        view: str = "perspective") -> np.ndarray | None:
        try:
            from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
            from PySide6.QtCore import QSize
            fmt = QSurfaceFormat()
            fmt.setVersion(3, 3)
            fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
            fmt.setDepthBufferSize(24)
            surface = QOffscreenSurface()
            surface.setFormat(fmt)
            surface.create()
            ctx = QOpenGLContext()
            ctx.setFormat(fmt)
            ctx.create()
            ctx.makeCurrent(surface)

            gl = _import_gl()
            if gl is None:
                return None

            fbo_id = gl.glGenFramebuffers(1)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo_id)
            color_rb = gl.glGenRenderbuffers(1)
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, color_rb)
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGBA8, width, height)
            gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, color_rb)
            depth_rb = gl.glGenRenderbuffers(1)
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_rb)
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, width, height)
            gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth_rb)
            gl.glViewport(0, 0, width, height)

            self.initialize_gl()
            self.update_mesh(mesh)

            camera = CameraState()
            L = float(mesh.vertices[:, 0].max())
            camera.reset_for_board(L)
            if view == "top":
                camera.elevation = 88.0
                camera.azimuth = 0.0
            elif view == "side":
                camera.elevation = 0.0
                camera.azimuth = 90.0
            elif view == "profile":
                camera.elevation = 0.0
                camera.azimuth = 0.0

            self.paint(width, height, camera)

            pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
            img = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)
            img = img[::-1]  # flip Y

            ctx.doneCurrent()
            return img
        except Exception as e:
            print(f"Offscreen render failed: {e}")
            return None


def _save_rgba_png(img: np.ndarray, path) -> None:
    from PySide6.QtGui import QImage
    h, w = img.shape[:2]
    qi = QImage(img.data, w, h, QImage.Format.Format_RGBA8888)
    qi.save(str(path))


def _compile_shader(gl, shader_type, source):
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)
    if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
        log = gl.glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compile error: {log}")
    return shader


def _compile_program(gl, vert_src, frag_src, geom_src=None):
    vert = _compile_shader(gl, gl.GL_VERTEX_SHADER, vert_src)
    frag = _compile_shader(gl, gl.GL_FRAGMENT_SHADER, frag_src)
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vert)
    gl.glAttachShader(prog, frag)
    if geom_src is not None:
        geom = _compile_shader(gl, gl.GL_GEOMETRY_SHADER, geom_src)
        gl.glAttachShader(prog, geom)
    gl.glLinkProgram(prog)
    if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
        log = gl.glGetProgramInfoLog(prog).decode()
        raise RuntimeError(f"Program link error: {log}")
    gl.glDeleteShader(vert)
    gl.glDeleteShader(frag)
    if geom_src is not None:
        gl.glDeleteShader(geom)
    return prog


def _set_uniform_mat4(gl, prog, name, mat):
    loc = gl.glGetUniformLocation(prog, name)
    if loc >= 0:
        gl.glUniformMatrix4fv(loc, 1, gl.GL_TRUE, mat.astype(np.float32))


def _set_uniform_mat3(gl, prog, name, mat):
    loc = gl.glGetUniformLocation(prog, name)
    if loc >= 0:
        gl.glUniformMatrix3fv(loc, 1, gl.GL_TRUE, mat.astype(np.float32))


def _set_uniform_vec2(gl, prog, name, vec):
    loc = gl.glGetUniformLocation(prog, name)
    if loc >= 0:
        gl.glUniform2fv(loc, 1, vec.astype(np.float32))


def _set_uniform_vec3(gl, prog, name, vec):
    loc = gl.glGetUniformLocation(prog, name)
    if loc >= 0:
        gl.glUniform3fv(loc, 1, vec.astype(np.float32))


def _set_uniform_float(gl, prog, name, value: float):
    loc = gl.glGetUniformLocation(prog, name)
    if loc >= 0:
        gl.glUniform1f(loc, float(value))


def _set_uniform_bool(gl, prog, name, value: bool):
    loc = gl.glGetUniformLocation(prog, name)
    if loc >= 0:
        gl.glUniform1i(loc, int(value))


def _norm(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = center - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-6:
        r = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        r = r / r_norm
    u = np.cross(r, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = r
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(r, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _ortho(left: float, right: float, bottom: float, top: float, near: float, far: float) -> np.ndarray:
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    m[3, 3] = 1.0
    return m
