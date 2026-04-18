from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import PchipInterpolator

from blankforge.data.model import CurveData, RailProfile, RailStation


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

        # Deduplicate by position (average values at same position)
        xs_raw = np.array([p.position_mm for p in pts])
        ys_raw = np.array([p.value_mm for p in pts])
        unique_xs, inv = np.unique(xs_raw, return_inverse=True)
        unique_ys = np.zeros_like(unique_xs)
        counts = np.zeros(len(unique_xs))
        for i, idx in enumerate(inv):
            unique_ys[idx] += ys_raw[i]
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
                t = (position_mm - a.position_mm) / (b.position_mm - a.position_mm)
                pa, pb = a.profile, b.profile
                return RailProfile(
                    apex_ratio=pa.apex_ratio + t * (pb.apex_ratio - pa.apex_ratio),
                    upper_concave=pa.upper_concave + t * (pb.upper_concave - pa.upper_concave),
                    lower_rail_angle=pa.lower_rail_angle + t * (pb.lower_rail_angle - pa.lower_rail_angle),
                    softness=pa.softness + t * (pb.softness - pa.softness),
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
        t = half_thickness_mm * 2.0  # full thickness

        apex_z = profile.apex_ratio * t
        softness_r = max(1.0, profile.softness * 20.0)  # rail arc radius mm

        pts = []

        # Segment 1: Upper deck — (0, t) → (w, apex_z)
        n_deck = n_points // 4
        for i in range(n_deck + 1):
            u = i / n_deck
            y = w * u
            z_base = t + (apex_z - t) * u
            concave = profile.upper_concave * t * 0.15 * math.sin(u * math.pi)
            z = z_base + concave
            pts.append((y, z))

        # Segment 2: Rail arc — quarter-circle from (w, apex_z) curving downward
        # Arc center at (w - softness_r, apex_z); sweeps angle 0 → -π/2
        # At angle=0: point = (w, apex_z) — joins deck end exactly (C0 continuous)
        # At angle=-π/2: point = (w - softness_r, apex_z - softness_r)
        n_rail = n_points // 8
        for i in range(1, n_rail + 1):
            u = i / n_rail
            angle = -math.pi / 2 * u
            y = (w - softness_r) + softness_r * math.cos(angle)
            z = apex_z + softness_r * math.sin(angle)
            pts.append((y, z))

        # Segment 3: Lower hull — (w - softness_r, apex_z - softness_r) → (0, 0)
        rail_end_y = w - softness_r
        rail_end_z = apex_z - softness_r
        n_hull = n_points // 4
        for i in range(1, n_hull + 1):
            u = i / n_hull
            y = rail_end_y * (1.0 - u)
            # Smooth convex bottom: slight bow keeps the bottom relatively flat
            z = rail_end_z * (1.0 - u) * (1.0 - 0.3 * u * (1.0 - u))
            pts.append((y, z))

        pts_arr = np.array(pts, dtype=np.float32)
        pts_arr[:, 0] = np.clip(pts_arr[:, 0], 0, w)
        pts_arr[:, 1] = np.clip(pts_arr[:, 1], 0, t + abs(profile.upper_concave) * t * 0.15)
        return pts_arr
