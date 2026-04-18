from __future__ import annotations

import math

import numpy as np

from blankforge.data.model import TailConfig


class TailShapeBuilder:
    """Generates 2D tail outline points in the XY plane (top-down view, one side only)."""

    def squaretail(self, half_width_mm: float, n: int = 16) -> np.ndarray:
        pts = []
        for i in range(n + 1):
            t = i / n
            pts.append((half_width_mm, t))
        return np.array(pts, dtype=np.float32)

    def roundtail(self, half_width_mm: float, n: int = 32) -> np.ndarray:
        pts = []
        for i in range(n + 1):
            angle = math.pi / 2 * i / n
            y = half_width_mm * math.cos(angle)
            x_frac = math.sin(angle)
            pts.append((y, x_frac))
        return np.array(pts, dtype=np.float32)

    def swallowtail(self, half_width_mm: float, notch_depth: float = 0.4, notch_width: float = 0.35, n: int = 32) -> np.ndarray:
        pts = []
        mid_y = half_width_mm * notch_width
        for i in range(n // 3 + 1):
            t = i / (n // 3)
            y = half_width_mm * t
            pts.append((y, 0.0))
        for i in range(n // 3 + 1):
            t = i / (n // 3)
            y = half_width_mm - (half_width_mm - mid_y) * t
            z = notch_depth * t
            pts.append((y, z))
        for i in range(n // 3 + 1):
            t = i / (n // 3)
            y = mid_y + (half_width_mm - mid_y) * t
            z = notch_depth * (1 - t)
            pts.append((y, z))
        return np.array(pts, dtype=np.float32)

    def dovetail(self, half_width_mm: float, n: int = 32) -> np.ndarray:
        pts = []
        notch_depth = 0.3
        for i in range(n + 1):
            t = i / n
            y = half_width_mm * t
            z = notch_depth * 4 * t * (1 - t)
            pts.append((y, z))
        return np.array(pts, dtype=np.float32)

    def build(self, tail_config: TailConfig) -> np.ndarray:
        hw = tail_config.width_mm / 2
        if tail_config.shape == "squaretail":
            return self.squaretail(hw)
        elif tail_config.shape == "roundtail":
            return self.roundtail(hw)
        elif tail_config.shape == "swallowtail":
            return self.swallowtail(hw)
        elif tail_config.shape == "dovetail":
            return self.dovetail(hw)
        return self.squaretail(hw)
