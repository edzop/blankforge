"""Fin geometry: spline evaluation and 3-D mesh builder.

Coordinate convention for the fin local frame
----------------------------------------------
  X  — chord direction (LE = 0, positive toward TE)
  Y  — lateral (depth into the fin face); 0 = planform plane, ±max = foil surfaces
  Z  — fin height (0 = base, positive = tip)

The returned BoardMesh uses the same NamedTuple as the board mesh so it can
be fed directly into GLViewport.
"""
from __future__ import annotations

import numpy as np

from blankforge.data.fin_model import FinDef
from blankforge.geometry.board import BoardMesh


# ---------------------------------------------------------------------------
# Tension-controlled Hermite spline (Kochanek-Bartels, bias=0, continuity=0)
# ---------------------------------------------------------------------------

def _kb_tangent(
    pts: np.ndarray,    # (N, 2) control points
    i: int,
    influence: float,   # tension: 0 = Catmull-Rom, 1 = sharp corner
) -> np.ndarray:
    """Compute the KB tangent at point i scaled by (1 - influence)."""
    n = len(pts)
    if n < 2:
        return np.zeros(2)
    if i == 0:
        raw = pts[1] - pts[0]
    elif i == n - 1:
        raw = pts[-1] - pts[-2]
    else:
        raw = 0.5 * (pts[i + 1] - pts[i - 1])
    return raw * (1.0 - influence)


def eval_fin_outline(fin: FinDef, samples_per_segment: int = 24) -> np.ndarray:
    """Evaluate the fin planform outline as a polyline of (x, y) points.

    Returns shape (N, 2) float32.  The last point is NOT equal to the first
    (open polyline from LE base to TE base).
    """
    pts_data = fin.points
    if len(pts_data) < 2:
        return np.zeros((2, 2), dtype=np.float32)

    pts_xy = np.array([[p.x_mm, p.y_mm] for p in pts_data], dtype=np.float64)
    influence = np.array([p.influence for p in pts_data], dtype=np.float64)
    n = len(pts_xy)

    # Pre-compute tangents using influence (spline tension) per point
    tangents = np.array([
        _kb_tangent(pts_xy, i, influence[i]) for i in range(n)
    ])

    result: list[np.ndarray] = []
    for seg in range(n - 1):
        p0, p1 = pts_xy[seg], pts_xy[seg + 1]
        t0, t1 = tangents[seg], tangents[seg + 1]
        ts = np.linspace(0.0, 1.0, samples_per_segment, endpoint=(seg == n - 2))
        h00 = 2 * ts**3 - 3 * ts**2 + 1
        h10 = ts**3 - 2 * ts**2 + ts
        h01 = -2 * ts**3 + 3 * ts**2
        h11 = ts**3 - ts**2
        seg_pts = (
            np.outer(h00, p0) + np.outer(h10, t0) +
            np.outer(h01, p1) + np.outer(h11, t1)
        )
        result.append(seg_pts)

    return np.vstack(result).astype(np.float32)


# ---------------------------------------------------------------------------
# NACA symmetric thickness distribution
# ---------------------------------------------------------------------------

def _naca_half_thickness(x_norm: np.ndarray, max_t: float) -> np.ndarray:
    """NACA 4-digit symmetric thickness (half-thickness, fraction of chord).

    x_norm: array in [0, 1] from LE to TE
    max_t:  maximum thickness as fraction of chord (e.g. 0.12)
    Returns half-thickness as fraction of chord.
    """
    x = np.clip(x_norm, 0.0, 1.0)
    # NACA formula; clamp sqrt argument
    sqx = np.sqrt(np.maximum(x, 0.0))
    t = 5.0 * max_t * (
        0.2969 * sqx
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    return np.maximum(t, 0.0)


# ---------------------------------------------------------------------------
# Fin mesh builder
# ---------------------------------------------------------------------------

def apply_cant_toe(mesh: BoardMesh, cant_deg: float, toe_deg: float) -> BoardMesh:
    """Rotate a fin mesh to show cant and toe angles visually.

    cant_deg : tilt around the chord (X) axis — positive = tip leans toward +Y
    toe_deg  : rotation around the height (Z) axis — positive = LE turns toward +Y
    """
    if abs(cant_deg) < 0.01 and abs(toe_deg) < 0.01:
        return mesh
    verts = mesh.vertices.astype(np.float64)
    norms = mesh.normals.astype(np.float64)

    c, s = np.cos(np.radians(cant_deg)), np.sin(np.radians(cant_deg))
    R_cant = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    ct, st = np.cos(np.radians(toe_deg)), np.sin(np.radians(toe_deg))
    R_toe = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])

    R = R_cant @ R_toe
    verts = (R @ verts.T).T
    norms = (R @ norms.T).T
    return BoardMesh(vertices=verts.astype(np.float32),
                     normals=norms.astype(np.float32),
                     triangles=mesh.triangles)


def transform_fin_to_board(fin_mesh: BoardMesh, fin: "FinDef", board_model) -> BoardMesh:
    """Transform a fin mesh from its local frame into board world coordinates.

    Fin local frame  : X=chord (0=LE), Y=foil depth, Z=height (0=base, up=tip)
    Board world frame: X=length (0=nose→L=tail), Y=lateral, Z=vertical (up=deck)
    The fin mounts on the bottom; its tip points downward (−Z in board frame).
    The fin is rotated around the lateral (Y) axis to align with the board
    bottom tangent, compensating for rocker at the mounting position.
    """
    from blankforge.geometry.curves import BoardCurveEvaluator

    L = board_model.parameters.length_mm
    x_board = L - fin.placement.x_from_tail_mm
    y_board = fin.placement.y_from_center_mm

    rocker_eval = BoardCurveEvaluator(board_model.curves.rocker)
    z_bottom = float(rocker_eval(x_board))

    # Rocker slope at the fin mounting point via central finite difference
    eps = 0.5  # mm
    slope = (float(rocker_eval(x_board + eps)) - float(rocker_eval(x_board - eps))) / (2.0 * eps)
    angle = np.arctan(slope)
    cos_r, sin_r = float(np.cos(angle)), float(np.sin(angle))

    # Apply cant and toe in the fin's local frame first
    rotated = apply_cant_toe(fin_mesh,
                              fin.placement.cant_deg,
                              fin.placement.toe_deg)

    verts = rotated.vertices.astype(np.float64)
    norms = rotated.normals.astype(np.float64)

    # Board-frame displacements, centered at the mounting point.
    # Local Z (height) is negated because the fin hangs below the board bottom.
    dx = verts[:, 0]   # chord → fore-aft
    dy = verts[:, 1]   # depth → lateral
    dz = -verts[:, 2]  # height → -Z (downward)

    # Ry(-angle): rotates the fin so its height axis aligns with the board
    # bottom outward normal, accounting for rocker slope.
    #   (0, 0, -1) maps to (sin_r, 0, -cos_r) — tilts toward tail for +slope.
    dx_r = dx * cos_r - dz * sin_r
    dz_r = dx * sin_r + dz * cos_r

    out = np.zeros_like(verts)
    out[:, 0] = x_board + dx_r
    out[:, 1] = y_board + dy
    out[:, 2] = z_bottom + dz_r

    # Same rotation applied to normals (no translation needed)
    nx = norms[:, 0]
    ny = norms[:, 1]
    nz = -norms[:, 2]
    nx_r = nx * cos_r - nz * sin_r
    nz_r = nx * sin_r + nz * cos_r

    out_n = np.zeros_like(norms)
    out_n[:, 0] = nx_r
    out_n[:, 1] = ny
    out_n[:, 2] = nz_r

    return BoardMesh(vertices=out.astype(np.float32),
                     normals=out_n.astype(np.float32),
                     triangles=rotated.triangles)


def merge_meshes(*meshes: BoardMesh) -> BoardMesh | None:
    """Concatenate multiple BoardMesh objects into a single mesh."""
    valid = [m for m in meshes if m is not None]
    if not valid:
        return None
    verts, norms, tris = [], [], []
    offset = 0
    for m in valid:
        verts.append(m.vertices)
        norms.append(m.normals)
        tris.append(m.triangles + offset)
        offset += len(m.vertices)
    return BoardMesh(
        vertices=np.vstack(verts).astype(np.float32),
        normals=np.vstack(norms).astype(np.float32),
        triangles=np.vstack(tris).astype(np.int32),
    )


# ---------------------------------------------------------------------------
# Fin box plug geometry
# ---------------------------------------------------------------------------

# How deep (in mm) the plug tab extends below the fin base (into the board).
_PLUG_DEPTH = 10.0

# Screw boss protrusion below the plug's bottom face.
_SCREW_PROTRUDE = 2.5
_SCREW_RADIUS   = 2.2
_SCREW_SIDES    = 10

# Offset from each end of the slot to the screw-centre for two-screw boxes.
_SCREW_INSET = 6.0


def _box_mesh(
    x0: float, x1: float,
    y0: float, y1: float,
    z0: float, z1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed axis-aligned rectangular prism.  Outward-facing triangle winding."""
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],   # 0-3 bottom
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],   # 4-7 top
    ], dtype=np.float32)
    t = np.array([
        [0, 2, 1], [0, 3, 2],  # bottom  (-Z)
        [4, 5, 6], [4, 6, 7],  # top     (+Z)
        [0, 5, 4], [0, 1, 5],  # front   (-Y)
        [2, 3, 7], [2, 7, 6],  # back    (+Y)
        [0, 7, 3], [0, 4, 7],  # left    (-X)
        [1, 2, 6], [1, 6, 5],  # right   (+X)
    ], dtype=np.int32)
    return v, t


def _cylinder_mesh(
    cx: float, cy: float,
    z_bot: float, z_top: float,
    radius: float = _SCREW_RADIUS,
    n: int = _SCREW_SIDES,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed cylinder (both caps).  Outward-facing triangle winding."""
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    cos_a, sin_a = np.cos(a), np.sin(a)

    # Layout: [0]         = bottom centre
    #         [1 .. n]    = bottom ring
    #         [n+1]       = top centre
    #         [n+2 .. 2n+1] = top ring
    v = np.empty((2 * n + 2, 3), dtype=np.float32)
    v[0] = [cx, cy, z_bot]
    v[1:n + 1, 0] = cx + cos_a * radius
    v[1:n + 1, 1] = cy + sin_a * radius
    v[1:n + 1, 2] = z_bot
    v[n + 1] = [cx, cy, z_top]
    v[n + 2:, 0] = cx + cos_a * radius
    v[n + 2:, 1] = cy + sin_a * radius
    v[n + 2:, 2] = z_top

    tris: list[list[int]] = []
    for i in range(n):
        j = (i + 1) % n
        # Bottom cap  (normal −Z)
        tris.append([0, 1 + j, 1 + i])
        # Top cap  (normal +Z)
        tris.append([n + 1, n + 2 + i, n + 2 + j])
        # Side quad
        tris.append([1 + i, 1 + j, n + 2 + j])
        tris.append([1 + i, n + 2 + j, n + 2 + i])

    return v, np.array(tris, dtype=np.int32)


def _compute_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals for an arbitrary triangle mesh."""
    norms = np.zeros_like(vertices)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)
    for k in range(3):
        np.add.at(norms, triangles[:, k], face_n)
    mag = np.linalg.norm(norms, axis=1, keepdims=True)
    mag = np.where(mag < 1e-9, 1.0, mag)
    return (norms / mag).astype(np.float32)


def _build_fin_box_mesh(fin: "FinDef") -> "BoardMesh | None":
    """Build the fin-plug geometry that inserts into the board box.

    Returns None for glassed-in fins (no removable plug).

    The plug extends from Z=0 (fin base / board surface) down to Z=-_PLUG_DEPTH.
    Screw-boss cylinders protrude below the plug's bottom face to indicate
    screw positions for each box standard.
    """
    box = fin.box
    if box.box_type == "Glassed-in" or box.slot_length_mm <= 0:
        return None

    if len(fin.points) < 2:
        return None

    # Centre the plug on the fin base chord.
    x_le = float(fin.points[0].x_mm)
    x_te = float(fin.points[-1].x_mm)
    chord_cx = (x_le + x_te) * 0.5

    half_l = box.slot_length_mm * 0.5
    half_w = box.slot_width_mm * 0.5

    px0, px1 = chord_cx - half_l, chord_cx + half_l
    py0, py1 = -half_w, half_w
    pz0, pz1 = -_PLUG_DEPTH, 0.0

    all_verts: list[np.ndarray] = []
    all_tris:  list[np.ndarray] = []
    offset = 0

    # --- plug body ---
    bv, bt = _box_mesh(px0, px1, py0, py1, pz0, pz1)
    all_verts.append(bv)
    all_tris.append(bt + offset)
    offset += len(bv)

    # --- screw bosses below the plug bottom face ---
    screw_z_top = pz0
    screw_z_bot = pz0 - _SCREW_PROTRUDE

    if box.box_type in ("FCS", "FCS II"):
        inset = min(_SCREW_INSET, half_l - _SCREW_RADIUS - 1.0)
        for sx in (chord_cx - inset, chord_cx + inset):
            sv, st = _cylinder_mesh(sx, 0.0, screw_z_bot, screw_z_top)
            all_tris.append(st + offset)
            all_verts.append(sv)
            offset += len(sv)
    elif box.box_type == "Futures":
        sv, st = _cylinder_mesh(chord_cx, 0.0, screw_z_bot, screw_z_top, radius=2.8)
        all_tris.append(st + offset)
        all_verts.append(sv)

    verts = np.vstack(all_verts).astype(np.float32)
    tris  = np.vstack(all_tris).astype(np.int32)
    norms = _compute_normals(verts, tris)

    return BoardMesh(vertices=verts, normals=norms, triangles=tris)


def build_fin_mesh(
    fin: FinDef,
    n_height: int = 40,
    n_chord: int = 16,
) -> BoardMesh | None:
    """Build a 3-D triangle mesh for *fin*.

    Returns a BoardMesh (vertices XZY mapped to local fin frame) or None if the
    outline is degenerate.

    Local frame returned:
      vertices[:, 0]  = chord  (X in fin frame = board fore-aft)
      vertices[:, 1]  = depth  (Y in fin frame = foil thickness lateral)
      vertices[:, 2]  = height (Z in fin frame = fin height vertical)
    """
    outline = eval_fin_outline(fin, samples_per_segment=32)
    if len(outline) < 4:
        return None

    x_pts = outline[:, 0]
    y_pts = outline[:, 1]  # height along fin

    y_base = float(y_pts.min())
    y_tip  = float(y_pts.max())
    span   = y_tip - y_base
    if span < 1.0:
        return None

    # Build height slices
    heights = np.linspace(y_base, y_tip, n_height)

    # Split outline into LE and TE sides at the tip index
    tip_idx = int(np.argmax(y_pts))

    # LE side: index 0..tip_idx   (base → tip, typically the left edge)
    # TE side: index tip_idx..end (tip → base, typically the right edge)
    le_y = y_pts[:tip_idx + 1]
    le_x = x_pts[:tip_idx + 1]
    te_y = y_pts[tip_idx:]
    te_x = x_pts[tip_idx:]

    if len(le_y) < 2 or len(te_y) < 2:
        return None

    # Interpolation functions: x_le(height), x_te(height)
    # Sort by y (height) for np.interp
    le_order = np.argsort(le_y)
    te_order = np.argsort(te_y)
    le_y_s, le_x_s = le_y[le_order], le_x[le_order]
    te_y_s, te_x_s = te_y[te_order], te_x[te_order]

    x_le = np.interp(heights, le_y_s, le_x_s)
    x_te = np.interp(heights, te_y_s, te_x_s)

    chord = x_te - x_le
    # Taper chord to zero near tip so the mesh closes cleanly
    chord = np.maximum(chord, 0.0)

    thickness_ratio = fin.foil.thickness_ratio
    max_half_t = thickness_ratio / 2.0  # half-thickness as fraction of chord

    # Sharpness per control point: interpolate to each height slice.
    # sharpness controls the EDGE TAPER RATE of the foil cross-section:
    #   sharpness=0 → gradual taper, rounded/soft edges (profile^0.3)
    #   sharpness=1 → abrupt taper, knife-like edges   (profile^1.7)
    # The maximum thickness at the widest point is ALWAYS max_half_t — only
    # how quickly the edges approach zero changes.
    ctrl_ys    = np.array([p.y_mm      for p in fin.points], dtype=np.float64)
    ctrl_sharp = np.array([p.sharpness for p in fin.points], dtype=np.float64)
    order = np.argsort(ctrl_ys)
    sharpness_at_h = np.interp(heights, ctrl_ys[order], ctrl_sharp[order])

    # Build per-slice foil cross-sections
    # Each slice: n_chord+1 points along chord + mirrored for both faces
    # Foil convention: top surface = +Y, bottom = -Y in local frame
    u = np.linspace(0.0, 1.0, n_chord + 1)  # chord parameter

    # Normalised NACA profile (peaks at 1.0 at max-thickness chord location)
    half_t_unit = _naca_half_thickness(u, 1.0)
    unit_peak = half_t_unit.max()
    if unit_peak > 0:
        half_t_norm = half_t_unit / unit_peak   # range [0, 1]; peak stays at 1
    else:
        half_t_norm = half_t_unit

    all_verts: list[np.ndarray] = []
    all_rings: list[np.ndarray] = []
    base_idx = 0

    for i, (h, x_l, chord_i) in enumerate(zip(heights, x_le, chord)):
        if chord_i < 0.5:
            # Degenerate slice near tip — single point
            pt = np.array([[x_l, 0.0, h]], dtype=np.float32)
            all_verts.append(pt)
            all_rings.append(np.array([base_idx], dtype=np.int32))
            base_idx += 1
            continue

        # Power exponent: 0.3 (soft/round edges) → 1.7 (knife edges)
        # Applied to the normalised profile so max thickness stays constant.
        # half_t is a dimensionless fraction [0, max_half_t]; scaled to mm below.
        p = 0.3 + float(sharpness_at_h[i]) * 1.4
        half_t = (half_t_norm ** p) * max_half_t   # fraction of chord
        x_chord = x_l + u * chord_i                # absolute X positions (mm)
        y_top   =  half_t * chord_i                # +depth (mm)
        y_bot   = -half_t * chord_i                # -depth (mm)

        # Top surface: LE → TE  (n_chord+1 points)
        # Bottom surface: TE → LE  (n_chord+1 points, reversed)
        # Together they form a closed ring
        top_pts = np.column_stack([x_chord, y_top, np.full(n_chord + 1, h)])
        bot_pts = np.column_stack([x_chord[::-1], y_bot[::-1], np.full(n_chord + 1, h)])

        # Omit the duplicated LE and TE endpoints
        ring = np.vstack([top_pts, bot_pts[1:-1]])
        n_r = len(ring)
        all_verts.append(ring.astype(np.float32))
        all_rings.append(np.arange(base_idx, base_idx + n_r, dtype=np.int32))
        base_idx += n_r

    if not all_verts:
        return None

    vertices = np.vstack(all_verts).astype(np.float32)

    # ---------------------------------------------------------------------------
    # Triangulate between consecutive rings (strip) + cap tip
    # ---------------------------------------------------------------------------
    tris: list[np.ndarray] = []

    for i in range(len(all_rings) - 1):
        r0 = all_rings[i]
        r1 = all_rings[i + 1]

        if len(r0) == 1:
            # Fan from single tip point to ring below
            apex = r0[0]
            for j in range(len(r1)):
                a, b = r1[j], r1[(j + 1) % len(r1)]
                tris.append(np.array([apex, b, a], dtype=np.int32))
            continue
        if len(r1) == 1:
            # Fan from single tip point above to ring below
            apex = r1[0]
            for j in range(len(r0)):
                a, b = r0[j], r0[(j + 1) % len(r0)]
                tris.append(np.array([apex, a, b], dtype=np.int32))
            continue

        # General case: stitch rings of equal size
        n0 = len(r0)
        n1 = len(r1)
        n_min = min(n0, n1)
        for j in range(n_min):
            a0, a1 = r0[j % n0], r0[(j + 1) % n0]
            b0, b1 = r1[j % n1], r1[(j + 1) % n1]
            tris.append(np.array([a0, b0, b1], dtype=np.int32))
            tris.append(np.array([a0, b1, a1], dtype=np.int32))

    # Cap the base (flat base poly-fan from first vertex of base ring)
    if len(all_rings[0]) >= 3:
        r = all_rings[0]
        for j in range(1, len(r) - 1):
            tris.append(np.array([r[0], r[j + 1], r[j]], dtype=np.int32))

    if not tris:
        return None

    triangles = np.vstack(tris).astype(np.int32)

    # ---------------------------------------------------------------------------
    # Compute per-vertex normals (area-weighted)
    # ---------------------------------------------------------------------------
    normals = np.zeros_like(vertices)
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)  # (M, 3)

    for k in range(3):
        np.add.at(normals, triangles[:, k], face_n)

    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens = np.where(lens < 1e-9, 1.0, lens)
    normals = (normals / lens).astype(np.float32)

    blade = BoardMesh(vertices=vertices, normals=normals, triangles=triangles)
    plug  = _build_fin_box_mesh(fin)
    return merge_meshes(blade, plug) if plug is not None else blade
