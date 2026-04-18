"""
Headless test runner: loads each .surfboard, builds geometry,
renders 2D views (QPainter) and writes statistics.json.

Run with: python3 tests/render_all_samples.py
(or: xvfb-run -a python3 tests/render_all_samples.py for headless GL)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Must create QApplication before any Qt objects
from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath

from blankforge.data.model import BoardModel
from blankforge.data.serializer import SurfboardSerializer
from blankforge.geometry.board import BoardGeometryBuilder
from blankforge.geometry.curves import BoardCurveEvaluator
import numpy as np


def render_top_view(model: BoardModel, path: Path, width: int = 800, height: int = 400) -> None:
    img = QImage(width, height, QImage.Format.Format_RGB32)
    img.fill(QColor(30, 32, 36))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    L = model.parameters.length_mm
    w_eval = BoardCurveEvaluator(model.curves.width)
    xs = np.linspace(0, L, 300)
    ys = w_eval(xs)
    max_hw = max(ys)

    margin = 30
    scale_x = (width - 2 * margin) / L
    scale_y = (height / 2 - margin) / (max_hw + 1)

    cx = width / 2
    cy = height / 2

    right_path = QPainterPath()
    left_path = QPainterPath()
    for i, (x, y) in enumerate(zip(xs, ys)):
        sx = margin + x * scale_x
        sy_r = cy - y * scale_y
        sy_l = cy + y * scale_y
        if i == 0:
            right_path.moveTo(sx, sy_r)
            left_path.moveTo(sx, sy_l)
        else:
            right_path.lineTo(sx, sy_r)
            left_path.lineTo(sx, sy_l)

    from PySide6.QtGui import QPen
    pen = QPen(QColor(80, 160, 220))
    pen.setWidthF(2.0)
    p.setPen(pen)
    p.drawPath(right_path)
    p.drawPath(left_path)
    p.end()
    img.save(str(path))


def render_side_view(model: BoardModel, path: Path, width: int = 800, height: int = 300) -> None:
    img = QImage(width, height, QImage.Format.Format_RGB32)
    img.fill(QColor(30, 32, 36))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    L = model.parameters.length_mm
    r_eval = BoardCurveEvaluator(model.curves.rocker)
    t_eval = BoardCurveEvaluator(model.curves.thickness)
    xs = np.linspace(0, L, 300)
    rockers = r_eval(xs)
    thicks = t_eval(xs)
    max_r = max(rockers)
    max_t = max(thicks)

    margin = 30
    scale_x = (width - 2 * margin) / L
    scale_y = (height - 2 * margin) / (max_r + max_t + 1)

    from PySide6.QtGui import QPen
    for vals, color in [(rockers, QColor(80, 200, 120)), (thicks, QColor(200, 120, 80))]:
        path = QPainterPath()
        for i, (x, y) in enumerate(zip(xs, vals)):
            sx = margin + x * scale_x
            sy = height - margin - y * scale_y
            if i == 0:
                path.moveTo(sx, sy)
            else:
                path.lineTo(sx, sy)
        pen = QPen(color)
        pen.setWidthF(2.0)
        p.setPen(pen)
        p.drawPath(path)
    p.end()
    img.save(str(path))


def render_sample(surfboard_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = SurfboardSerializer.load(surfboard_path)
    builder = BoardGeometryBuilder(use_occt=False)
    mesh, stats = builder.build(model, resolution=40)

    render_top_view(model, output_dir / "top.png")
    render_side_view(model, output_dir / "side.png")

    stats_dict = {
        "volume_cm3": stats.volume_cm3,
        "surface_area_cm2": stats.surface_area_cm2,
        "length_mm": stats.length_mm,
        "width_mm": stats.width_mm,
        "thickness_mm": stats.thickness_mm,
        "nose_width_mm": stats.nose_width_mm,
        "tail_width_mm": stats.tail_width_mm,
    }
    (output_dir / "statistics.json").write_text(json.dumps(stats_dict, indent=2))
    print(f"  volume={stats.volume_cm3/1000:.1f} L ({stats.volume_cm3:.0f} cm³), surface={stats.surface_area_cm2:.0f} cm²")


def main() -> None:
    samples_dir = Path("samples")
    if not samples_dir.exists():
        print("No samples/ directory found. Run from project root.")
        sys.exit(1)

    surfboard_files = sorted(samples_dir.glob("*.surfboard"))
    if not surfboard_files:
        print("No .surfboard files found in samples/")
        sys.exit(1)

    for sf in surfboard_files:
        name = sf.stem
        out_dir = Path("output") / name
        print(f"Rendering {name}…")
        render_sample(sf, out_dir)
        print(f"  → output/{name}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
