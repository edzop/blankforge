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

uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat3 uNormalMatrix;

out vec3 vNormal;
out vec3 vFragPos;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vFragPos = vec3(uModel * vec4(aPos, 1.0));
    vNormal = normalize(uNormalMatrix * aNormal);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 vNormal;
in vec3 vFragPos;

uniform vec3 uCameraPos;
uniform vec3 uSurfaceColor;

// 3-point lighting
uniform vec3 uKeyDir;      // direction FROM fragment TO key light
uniform vec3 uKeyColor;
uniform float uKeyStrength;

uniform vec3 uFillDir;
uniform vec3 uFillColor;
uniform float uFillStrength;

uniform vec3 uRimDir;      // direction FROM fragment TO rim light (behind subject)
uniform vec3 uRimColor;
uniform float uRimStrength;

out vec4 FragColor;

// Blinn-Phong contribution from a single light
vec3 light_contrib(vec3 N, vec3 V, vec3 L, vec3 color, float strength,
                   float diff_power, float spec_gloss, float spec_strength) {
    vec3 Ln = normalize(L);
    float diff = max(dot(N, Ln), 0.0);
    vec3 H = normalize(Ln + V);
    float spec = pow(max(dot(N, H), 0.0), spec_gloss) * spec_strength;
    return color * strength * (uSurfaceColor * diff * diff_power + vec3(spec));
}

void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCameraPos - vFragPos);

    // Soft ambient — slightly sky-tinted
    vec3 ambient = uSurfaceColor * vec3(0.08, 0.10, 0.13);

    // Key: warm white, strong diffuse + sharp specular (main shadow-casting light)
    vec3 key  = light_contrib(N, V, uKeyDir,  uKeyColor,  uKeyStrength,
                               1.0, 96.0, 0.45);

    // Fill: cool blue-grey, soft diffuse only, kills harsh shadows
    vec3 fill = light_contrib(N, V, uFillDir, uFillColor, uFillStrength,
                               0.9, 1.0, 0.0);

    // Rim: from behind, broad soft glow + thin specular to separate from bg
    vec3 rim  = light_contrib(N, V, uRimDir,  uRimColor,  uRimStrength,
                               0.7, 24.0, 0.15);

    vec3 result = ambient + key + fill + rim;

    // Reinhard tone mapping + gamma correction
    result = result / (result + vec3(0.55));
    result = pow(clamp(result, 0.0, 1.0), vec3(1.0 / 2.2));

    FragColor = vec4(result, 1.0);
}
"""


@dataclass
class CameraState:
    azimuth: float = 45.0
    elevation: float = 20.0
    distance: float = 3000.0
    target: np.ndarray = field(default_factory=lambda: np.array([940.0, 0.0, 50.0], dtype=np.float32))
    fov: float = 45.0

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
        return _perspective(self.fov, aspect, 10.0, 20000.0)

    def eye_position(self) -> np.ndarray:
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        return self.target + self.distance * np.array([
            math.cos(el) * math.sin(az),
            math.cos(el) * math.cos(az),
            math.sin(el),
        ], dtype=np.float32)

    def reset_for_board(self, length_mm: float) -> None:
        self.target = np.array([length_mm / 2, 0.0, 30.0], dtype=np.float32)
        self.distance = length_mm * 1.6
        self.azimuth = 45.0
        self.elevation = 25.0


class OpenGLBoardRenderer(AbstractRenderer):
    def __init__(self) -> None:
        self._vao = None
        self._vbo_verts = None
        self._vbo_norms = None
        self._ebo = None
        self._shader_program = None
        self._n_indices = 0
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
        self._vao = gl.glGenVertexArrays(1)
        self._vbo_verts = gl.glGenBuffers(1)
        self._vbo_norms = gl.glGenBuffers(1)
        self._ebo = gl.glGenBuffers(1)
        self._initialized = True

    def update_mesh(self, mesh: BoardMesh) -> None:
        if not self._initialized:
            self.initialize_gl()
        gl = _import_gl()
        if gl is None:
            return
        self._n_indices = len(mesh.triangles) * 3

        gl.glBindVertexArray(self._vao)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_verts)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, mesh.vertices.nbytes, mesh.vertices, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_norms)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, mesh.normals.nbytes, mesh.normals, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        indices = mesh.triangles.astype(np.uint32)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

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

        gl.glUseProgram(self._shader_program)

        aspect = width / max(height, 1)
        proj = camera.projection_matrix(aspect)
        view = camera.view_matrix()
        model = np.eye(4, dtype=np.float32)
        mvp = proj @ view @ model
        normal_mat = np.linalg.inv(model[:3, :3]).T.astype(np.float32)

        _set_uniform_mat4(gl, self._shader_program, "uMVP", mvp)
        _set_uniform_mat4(gl, self._shader_program, "uModel", model)
        _set_uniform_mat3(gl, self._shader_program, "uNormalMatrix", normal_mat)
        _set_uniform_vec3(gl, self._shader_program, "uCameraPos", camera.eye_position())
        _set_uniform_vec3(gl, self._shader_program, "uSurfaceColor", np.array([0.86, 0.83, 0.76], dtype=np.float32))

        # Key light — warm, upper-left of default camera view
        _set_uniform_vec3(gl, self._shader_program, "uKeyDir",
                          _norm(np.array([-0.55, -0.65, 0.85], dtype=np.float32)))
        _set_uniform_vec3(gl, self._shader_program, "uKeyColor",
                          np.array([1.00, 0.95, 0.82], dtype=np.float32))
        _set_uniform_float(gl, self._shader_program, "uKeyStrength", 1.0)

        # Fill light — cool blue-grey, lower-right, softens key shadows
        _set_uniform_vec3(gl, self._shader_program, "uFillDir",
                          _norm(np.array([0.80, 0.40, 0.25], dtype=np.float32)))
        _set_uniform_vec3(gl, self._shader_program, "uFillColor",
                          np.array([0.60, 0.70, 0.90], dtype=np.float32))
        _set_uniform_float(gl, self._shader_program, "uFillStrength", 0.38)

        # Rim light — from behind, slightly cool, kisses the rails
        _set_uniform_vec3(gl, self._shader_program, "uRimDir",
                          _norm(np.array([0.30, 0.85, -0.40], dtype=np.float32)))
        _set_uniform_vec3(gl, self._shader_program, "uRimColor",
                          np.array([0.75, 0.85, 1.00], dtype=np.float32))
        _set_uniform_float(gl, self._shader_program, "uRimStrength", 0.55)

        gl.glBindVertexArray(self._vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self._n_indices, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

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


def _compile_program(gl, vert_src, frag_src):
    vert = _compile_shader(gl, gl.GL_VERTEX_SHADER, vert_src)
    frag = _compile_shader(gl, gl.GL_FRAGMENT_SHADER, frag_src)
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vert)
    gl.glAttachShader(prog, frag)
    gl.glLinkProgram(prog)
    if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
        log = gl.glGetProgramInfoLog(prog).decode()
        raise RuntimeError(f"Program link error: {log}")
    gl.glDeleteShader(vert)
    gl.glDeleteShader(frag)
    return prog


def _set_uniform_mat4(gl, prog, name, mat):
    loc = gl.glGetUniformLocation(prog, name)
    if loc >= 0:
        gl.glUniformMatrix4fv(loc, 1, gl.GL_TRUE, mat.astype(np.float32))


def _set_uniform_mat3(gl, prog, name, mat):
    loc = gl.glGetUniformLocation(prog, name)
    if loc >= 0:
        gl.glUniformMatrix3fv(loc, 1, gl.GL_TRUE, mat.astype(np.float32))


def _set_uniform_vec3(gl, prog, name, vec):
    loc = gl.glGetUniformLocation(prog, name)
    if loc >= 0:
        gl.glUniform3fv(loc, 1, vec.astype(np.float32))


def _set_uniform_float(gl, prog, name, value: float):
    loc = gl.glGetUniformLocation(prog, name)
    if loc >= 0:
        gl.glUniform1f(loc, float(value))


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
