from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from blankforge.data.model import BoardModel, TailConfig
from blankforge.geometry.curves import BoardCurveEvaluator, RailProfileEvaluator


class BoardMesh(NamedTuple):
    vertices: np.ndarray   # (N, 3) float32 XYZ — X=length, Y=lateral, Z=vertical
    normals: np.ndarray    # (N, 3) float32
    triangles: np.ndarray  # (M, 3) int32 indices


class BoardStats(NamedTuple):
    volume_cm3: float
    surface_area_cm2: float
    length_mm: float
    width_mm: float
    thickness_mm: float
    nose_width_mm: float
    tail_width_mm: float


def _occt_available() -> bool:
    try:
        import OCC.Core.BRep  # noqa: F401
        return True
    except ImportError:
        return False


def _tail_hw(x: float, L: float, hw_curve: float, tail: TailConfig) -> float:
    """
    Returns the effective half-width at position x, applying the tail shape
    in the last tail.length_mm of the board.

    Shapes:
      squaretail — constant width to the very end (wide, flat cutoff)
      roundtail  — tapers smoothly to a rounded point
      swallowtail — stays wide; at the very end the two lobes are prominent
      dovetail   — moderate taper with slight concave suggestion
    """
    tail_start = L - tail.length_mm
    if x <= tail_start or tail.length_mm < 1.0:
        return hw_curve

    t = (x - tail_start) / tail.length_mm  # 0 → 1 within tail region
    hw_entry = hw_curve  # width at tail_start (from curve)
    hw_end = tail.width_mm / 2.0  # target full-tail half-width

    if tail.shape == "squaretail":
        # Blend quickly to tail width, then hold flat — squared-off end
        blend = min(1.0, t * 3.0)
        target = hw_end
    elif tail.shape == "roundtail":
        # Taper to ~0 following a quarter-circle (pronounced taper, round point)
        target = hw_end * math.cos(t * math.pi / 2)
        blend = t
    elif tail.shape == "swallowtail":
        # Stay wide (lobes); the width at the very end is slightly WIDER than
        # the nominal tail width to suggest the two protruding wings
        target = hw_end * (1.0 + 0.15 * math.sin(t * math.pi))
        blend = t
    elif tail.shape == "dovetail":
        # Moderate taper — ends noticeably narrower than squaretail
        target = hw_end * (1.0 - 0.45 * t)
        blend = t
    else:
        return hw_curve

    return hw_entry * (1.0 - blend) + target * blend


def _build_with_numpy(model: BoardModel, resolution: int) -> tuple[BoardMesh, BoardStats]:
    L = model.parameters.length_mm
    stations = np.linspace(0, L, resolution)

    width_eval = BoardCurveEvaluator(model.curves.width)
    rocker_eval = BoardCurveEvaluator(model.curves.rocker)
    thick_eval = BoardCurveEvaluator(model.curves.thickness)
    rail_eval = RailProfileEvaluator(model.curves.rail)

    # n_contour: number of points around each cross-section (even, for symmetry)
    n_contour = 32  # per side → 64 total (left + right)

    all_verts: list[np.ndarray] = []
    all_rings: list[np.ndarray] = []  # indices into all_verts for each ring

    for x in stations:
        hw = float(width_eval(x))    # half-width
        hw = _tail_hw(x, L, hw, model.tail)  # apply tail shape override
        ht = float(thick_eval(x))    # half-thickness
        rk = float(rocker_eval(x))   # rocker height (bottom of board rises by this)

        profile_pts = rail_eval.cross_section_points(x, hw, ht, n_points=n_contour * 2)
        n_pts = len(profile_pts)

        # Build full closed ring: right side + mirrored left side
        # profile_pts are (y, z) with y >= 0
        ring_right = profile_pts  # (y, z) right side
        ring_left = profile_pts[::-1].copy()
        ring_left[:, 0] *= -1  # mirror Y

        # Combine into full closed ring (y, z) in 2D
        ring_2d = np.vstack([ring_right, ring_left])

        # Convert to 3D: X=station position, Y=lateral, Z=vertical+rocker
        n_ring = len(ring_2d)
        ring_3d = np.zeros((n_ring, 3), dtype=np.float32)
        ring_3d[:, 0] = x
        ring_3d[:, 1] = ring_2d[:, 0]
        ring_3d[:, 2] = ring_2d[:, 1] + rk

        start_idx = sum(len(r) for r in all_rings)
        all_verts.append(ring_3d)
        all_rings.append(np.arange(start_idx, start_idx + n_ring))

    # Build triangles by connecting adjacent rings
    triangles: list[list[int]] = []
    for i in range(len(all_rings) - 1):
        ring_a = all_rings[i]
        ring_b = all_rings[i + 1]
        n = len(ring_a)
        for j in range(n):
            j_next = (j + 1) % n
            a0, a1 = int(ring_a[j]), int(ring_a[j_next])
            b0, b1 = int(ring_b[j]), int(ring_b[j_next])
            triangles.append([a0, b0, a1])
            triangles.append([a1, b0, b1])

    # Nose cap: fan triangulate the first ring to a nose point
    nose_pt_idx = sum(len(v) for v in all_verts)
    nose_center = np.array([[0.0, 0.0, float(rocker_eval(0))]], dtype=np.float32)
    all_verts.append(nose_center)
    nose_ring = all_rings[0]
    n = len(nose_ring)
    for j in range(n):
        j_next = (j + 1) % n
        triangles.append([int(nose_ring[j]), nose_pt_idx, int(nose_ring[j_next])])

    # Tail cap: fan triangulate the last ring to a tail center
    tail_pt_idx = sum(len(v) for v in all_verts) - 1 + 1
    tail_center = np.array([[L, 0.0, float(rocker_eval(L))]], dtype=np.float32)
    all_verts.append(tail_center)
    tail_ring = all_rings[-1]
    n = len(tail_ring)
    for j in range(n):
        j_next = (j + 1) % n
        triangles.append([int(tail_ring[j_next]), tail_pt_idx, int(tail_ring[j])])

    vertices = np.vstack(all_verts).astype(np.float32)
    tris = np.array(triangles, dtype=np.int32)

    # Compute per-vertex normals
    normals = _compute_normals(vertices, tris)

    # Compute stats
    volume_mm3 = _mesh_volume(vertices, tris)
    surface_mm2 = _mesh_surface_area(vertices, tris)

    stats = BoardStats(
        volume_cm3=abs(volume_mm3) / 1000.0,
        surface_area_cm2=surface_mm2 / 100.0,
        length_mm=L,
        width_mm=model.parameters.width_mm,
        thickness_mm=model.parameters.thickness_mm,
        nose_width_mm=float(width_eval(300.0)) * 2,
        tail_width_mm=float(width_eval(L - 300.0)) * 2,
    )
    return BoardMesh(vertices=vertices, normals=normals, triangles=tris), stats


def _compute_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, triangles[:, i], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (normals / norms).astype(np.float32)


def _mesh_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """Signed volume via divergence theorem."""
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross = np.cross(v1, v2)
    return float(np.sum(v0 * cross) / 6.0)


def _mesh_surface_area(vertices: np.ndarray, triangles: np.ndarray) -> float:
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return float(np.sum(np.linalg.norm(cross, axis=1)) / 2.0)


class BoardGeometryBuilder:
    def __init__(self, use_occt: bool = True) -> None:
        self._use_occt = use_occt and _occt_available()

    def build(self, model: BoardModel, resolution: int = 50) -> tuple[BoardMesh, BoardStats]:
        if self._use_occt:
            try:
                return _build_with_occt(model, resolution)
            except Exception:
                pass
        return _build_with_numpy(model, resolution)


def _build_with_occt(model: BoardModel, resolution: int) -> tuple[BoardMesh, BoardStats]:
    raise NotImplementedError("OCCT path not yet implemented — using numpy fallback")
