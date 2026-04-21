from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from blankforge.data.model import BoardModel


class SurfboardSerializer:
    EXTENSION = ".surfboard"

    @staticmethod
    def save(model: BoardModel, path: Path) -> None:
        path = Path(path)
        if path.suffix != SurfboardSerializer.EXTENSION:
            path = path.with_suffix(SurfboardSerializer.EXTENSION)
        path.write_text(model.model_dump_json(indent=2))

    @staticmethod
    def load(path: Path) -> BoardModel:
        return BoardModel.model_validate_json(Path(path).read_text())

    @staticmethod
    def export_stl(
        model: BoardModel,
        path: Path,
        mesh_resolution: int = 50,
        include_board: bool = True,
        include_fin_left: bool = True,
        include_fin_right: bool = True,
        include_fin_center: bool = True,
    ) -> None:
        mesh = _build_export_mesh(
            model, mesh_resolution,
            include_board, include_fin_left, include_fin_right, include_fin_center,
        )
        if mesh is None:
            raise ValueError("No parts selected for export.")
        _write_stl_binary(mesh.vertices, mesh.triangles, Path(path))

    @staticmethod
    def export_obj(
        model: BoardModel,
        path: Path,
        mesh_resolution: int = 50,
        include_board: bool = True,
        include_fin_left: bool = True,
        include_fin_right: bool = True,
        include_fin_center: bool = True,
    ) -> None:
        mesh = _build_export_mesh(
            model, mesh_resolution,
            include_board, include_fin_left, include_fin_right, include_fin_center,
        )
        if mesh is None:
            raise ValueError("No parts selected for export.")
        _write_obj(mesh.vertices, mesh.normals, mesh.triangles, Path(path))


def _fin_side(fin) -> str:
    """Classify fin by its lateral placement: 'left' | 'right' | 'center'."""
    y = float(fin.placement.y_from_center_mm)
    if y < -1.0:
        return "left"
    if y > 1.0:
        return "right"
    return "center"


def _build_export_mesh(
    model: BoardModel,
    mesh_resolution: int,
    include_board: bool,
    include_fin_left: bool,
    include_fin_right: bool,
    include_fin_center: bool,
):
    from blankforge.geometry.board import BoardGeometryBuilder
    from blankforge.geometry.fin import (
        build_fin_mesh, merge_meshes, transform_fin_to_board,
    )

    parts = []
    if include_board:
        board_mesh, _ = BoardGeometryBuilder(use_occt=False).build(model, resolution=mesh_resolution)
        parts.append(board_mesh)

    if model.fins is not None:
        side_flags = {
            "left":   include_fin_left,
            "right":  include_fin_right,
            "center": include_fin_center,
        }
        for fin in model.fins.fins:
            if not side_flags.get(_fin_side(fin), False):
                continue
            fm = build_fin_mesh(fin, n_height=30, n_chord=12)
            if fm is None:
                continue
            parts.append(transform_fin_to_board(fm, fin, model))

    return merge_meshes(*parts)


def _write_stl_binary(vertices: np.ndarray, triangles: np.ndarray, path: Path) -> None:
    n_tri = len(triangles)
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", n_tri))
        for tri in triangles:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            e1 = v1 - v0
            e2 = v2 - v0
            normal = np.cross(e1, e2)
            n_len = np.linalg.norm(normal)
            if n_len > 0:
                normal /= n_len
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<H", 0))


def _write_obj(vertices: np.ndarray, normals: np.ndarray, triangles: np.ndarray, path: Path) -> None:
    lines = ["# BlankForge OBJ export", ""]
    for v in vertices:
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")
    lines.append("")
    for n in normals:
        lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
    lines.append("")
    for tri in triangles:
        i, j, k = tri[0] + 1, tri[1] + 1, tri[2] + 1
        lines.append(f"f {i}//{i} {j}//{j} {k}//{k}")
    path.write_text("\n".join(lines))
