from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import PchipInterpolator

from blankforge.data.model import ControlPoint, CurveData, RailProfile, RailStation


def resolve_thickness_curve(curve: CurveData, param_thickness_mm: float) -> CurveData:
    """Return a copy of the thickness curve where ratio-mode points are converted to mm.

    Points with mode='ratio' store a ratio where 1.0 means the global thickness
    parameter; resolved value = ratio * (param_thickness_mm / 2). Fixed-mode points
    are unchanged.
    """
    param_half_t = param_thickness_mm / 2.0
    return CurveData(points=[
        ControlPoint(
            position_mm=cp.position_mm,
            value_mm=cp.value_mm * param_half_t if cp.mode == "ratio" else cp.value_mm,
            mode=cp.mode,
            influence=cp.influence,
        )
        for cp in curve.sorted_points()
    ])


class BoardCurveEvaluator:
    """Evaluates a CurveData model at arbitrary board positions using monotone cubic interpolation."""

    def __init__(self, curve_data: CurveData) -> None:
        pts = curve_data.sorted_points()
        self._constant: float | None = None
        self._interp: PchipInterpolator | None = None
        self._xs: np.ndarray | None = None
        self._ys: np.ndarray | None = None

        if not pts:
            self._constant = 0.0
            return

        # Apply per-point influence: blend each point's value toward the curve value
        # that would result if that point were excluded. influence=1 keeps the point
        # on the curve; influence=0 effectively removes its pull on the curve.
        xs_raw = np.array([p.position_mm for p in pts])
        ys_raw = np.array([p.value_mm for p in pts], dtype=float)
        infl = np.array([getattr(p, "influence", 1.0) for p in pts], dtype=float)

        effective_ys = ys_raw.copy()
        if len(pts) >= 3:
            for i in range(len(pts)):
                if infl[i] < 0.999:
                    other_xs = np.concatenate([xs_raw[:i], xs_raw[i + 1:]])
                    other_ys = np.concatenate([ys_raw[:i], ys_raw[i + 1:]])
                    # Dedup
                    uxs, inv = np.unique(other_xs, return_inverse=True)
                    uys = np.zeros_like(uxs, dtype=float)
                    cnt = np.zeros(len(uxs))
                    for j, k in enumerate(inv):
                        uys[k] += other_ys[j]
                        cnt[k] += 1
                    uys /= cnt
                    if len(uxs) >= 2:
                        excl = PchipInterpolator(uxs, uys, extrapolate=True)
                        excluded_y = float(excl(xs_raw[i]))
                        effective_ys[i] = infl[i] * ys_raw[i] + (1.0 - infl[i]) * excluded_y

        # Deduplicate by position (average values at same position)
        unique_xs, inv = np.unique(xs_raw, return_inverse=True)
        unique_ys = np.zeros_like(unique_xs, dtype=float)
        counts = np.zeros(len(unique_xs))
        for i, idx in enumerate(inv):
            unique_ys[idx] += effective_ys[i]
            counts[idx] += 1
        unique_ys /= counts

        self._xs = unique_xs
        self._ys = unique_ys

        if len(unique_xs) == 1:
            self._constant = float(unique_ys[0])
        else:
            self._interp = PchipInterpolator(unique_xs, unique_ys, extrapolate=False)

    def __call__(self, position_mm: float | np.ndarray) -> float | np.ndarray:
        if self._constant is not None:
            scalar = np.isscalar(position_mm)
            result = np.full(np.shape(position_mm), self._constant) if not scalar else self._constant
            return float(result) if scalar else result

        scalar = np.isscalar(position_mm)
        pos = np.atleast_1d(np.asarray(position_mm, dtype=float))

        result = np.where(
            pos <= self._xs[0],
            self._ys[0],
            np.where(
                pos >= self._xs[-1],
                self._ys[-1],
                self._interp(pos),
            ),
        )
        # Fill NaN from extrapolation with boundary values
        result = np.where(np.isnan(result), np.where(pos < self._xs[0], self._ys[0], self._ys[-1]), result)
        return float(result[0]) if scalar else result


class RailProfileEvaluator:
    """Interpolates rail profile parameters across stations, returns cross-section point arrays."""

    def __init__(self, stations: list[RailStation]) -> None:
        self._stations = sorted(stations, key=lambda s: s.position_mm)

    def at(self, position_mm: float) -> RailProfile:
        if not self._stations:
            return RailProfile()
        if len(self._stations) == 1:
            return self._stations[0].profile

        if position_mm <= self._stations[0].position_mm:
            return self._stations[0].profile
        if position_mm >= self._stations[-1].position_mm:
            return self._stations[-1].profile

        # Find bracketing stations
        for i in range(len(self._stations) - 1):
            a = self._stations[i]
            b = self._stations[i + 1]
            if a.position_mm <= position_mm <= b.position_mm:
                f = (position_mm - a.position_mm) / (b.position_mm - a.position_mm)
                pa, pb = a.profile, b.profile
                return RailProfile(
                    apex_ratio=pa.apex_ratio + f * (pb.apex_ratio - pa.apex_ratio),
                    deck_concave=pa.deck_concave + f * (pb.deck_concave - pa.deck_concave),
                    lower_concave=pa.lower_concave + f * (pb.lower_concave - pa.lower_concave),
                    rail_ratio=pa.rail_ratio + f * (pb.rail_ratio - pa.rail_ratio),
                    softness=pa.softness + f * (pb.softness - pa.softness),
                )
        return self._stations[-1].profile

    def cross_section_points(
        self,
        position_mm: float,
        half_width_mm: float,
        half_thickness_mm: float,
        n_points: int = 64,
        profile: RailProfile | None = None,
    ) -> np.ndarray:
        """
        Returns (n_points, 2) array of (y, z) cross-section outline.
        y = lateral direction (0 = centerline, half_width = rail edge)
        z = vertical direction (0 = bottom, half_thickness*2 = deck)

        The profile is generated for one side only; the caller mirrors for the full board.
        This produces a closed loop starting at the deck centerline.
        """
        if profile is None:
            profile = self.at(position_mm)
        w = half_width_mm
        ht = half_thickness_mm
        t = ht * 2.0  # full thickness

        r = max(1.0, profile.rail_ratio * ht)
        cy = w - r

        # apex_ratio controls height of the widest rail point:
        # 1.0 = apex near deck (high rail), 0.0 = apex at hull level (low rail)
        cz = profile.apex_ratio * max(0.0, t - r)
        arc_top_z = cz + r  # height where arc meets the deck segment

        # Softness controls arc sweep: 0 → 90° arc only, 1 → arc wraps to hull
        end_angle = -(math.pi / 2) * profile.softness
        total_sweep = math.pi / 2 - end_angle

        arc_end_y = cy + r * math.cos(end_angle)
        arc_end_z = cz + r * math.sin(end_angle)

        pts = []

        # Segment 1: Deck — (0, t) → (cy, arc_top_z) with optional concavity.
        # Linear slope from centerline deck height to arc-top height; sin envelope for concavity.
        n_deck = n_points // 4
        for i in range(n_deck + 1):
            u = i / n_deck
            y = cy * u
            z_base = t + (arc_top_z - t) * u
            z = z_base + profile.deck_concave * t * 0.15 * math.sin(u * math.pi)
            pts.append((y, z))

        # Segment 2: Rail arc — π/2 → end_angle
        n_rail = n_points // 4
        for i in range(1, n_rail + 1):
            u = i / n_rail
            angle = math.pi / 2 - total_sweep * u
            y = cy + r * math.cos(angle)
            z = cz + r * math.sin(angle)
            pts.append((y, z))

        # Segment 3: Hull.
        # When the arc ends above the hull plane, extend in the arc's tangent
        # direction (so the side line is C1-continuous with the arc — no kink),
        # then run flat across to the centerline.
        n_hull = n_points // 4
        if arc_end_z > 0.5:
            # Tangent direction at arc end, in the travel direction of the outline.
            # Arc tangent magnitude: (sin(end_angle), -cos(end_angle))
            tx = math.sin(end_angle)
            tz = -math.cos(end_angle)
            # Find where tangent line crosses z = 0
            if tz < -1e-6:
                s_z0 = arc_end_z / (-tz)  # always positive
                y_at_hull = arc_end_y + tx * s_z0
                z_at_hull = 0.0
            else:
                # Tangent is horizontal — fall back to straight drop
                y_at_hull = arc_end_y
                z_at_hull = 0.0
            # Don't let the tangent extension cross the centerline
            if y_at_hull < 0.0:
                if abs(tx) > 1e-6:
                    s_y0 = -arc_end_y / tx
                    y_at_hull = 0.0
                    z_at_hull = max(0.0, arc_end_z + tz * s_y0)
                else:
                    y_at_hull = 0.0

            n_drop = max(2, n_hull // 2)
            for i in range(1, n_drop + 1):
                u = i / n_drop
                y = arc_end_y + (y_at_hull - arc_end_y) * u
                z = arc_end_z + (z_at_hull - arc_end_z) * u
                pts.append((y, z))
            hull_y_start = y_at_hull
            n_flat = max(2, n_hull - n_drop)
        else:
            hull_y_start = arc_end_y
            n_flat = n_hull
        for i in range(1, n_flat + 1):
            u = i / n_flat
            y = hull_y_start * (1.0 - u)
            z = max(0.0, -profile.lower_concave * t * 0.15 * math.sin(u * math.pi))
            pts.append((y, z))

        pts_arr = np.array(pts, dtype=np.float32)
        pts_arr[:, 0] = np.clip(pts_arr[:, 0], 0, w)
        pts_arr[:, 1] = np.clip(pts_arr[:, 1], 0, t + abs(profile.deck_concave) * t * 0.15)
        return pts_arr
