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
        # If arc ends above hull plane (z>0), drop vertically to hull then go horizontal.
        # This preserves full board thickness regardless of softness/apex settings.
        n_hull = n_points // 4
        if arc_end_z > 0.5:
            n_drop = max(2, n_hull // 2)
            for i in range(1, n_drop + 1):
                u = i / n_drop
                pts.append((arc_end_y, arc_end_z * (1.0 - u)))
            n_flat = max(2, n_hull - n_drop)
        else:
            n_flat = n_hull
        for i in range(1, n_flat + 1):
            u = i / n_flat
            y = arc_end_y * (1.0 - u)
            z = max(0.0, -profile.lower_concave * t * 0.15 * math.sin(u * math.pi))
            pts.append((y, z))

        pts_arr = np.array(pts, dtype=np.float32)
        pts_arr[:, 0] = np.clip(pts_arr[:, 0], 0, w)
        pts_arr[:, 1] = np.clip(pts_arr[:, 1], 0, t + abs(profile.deck_concave) * t * 0.15)
        return pts_arr
