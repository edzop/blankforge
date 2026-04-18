from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from blankforge.geometry.board import BoardMesh
from blankforge.renderer.base import AbstractRenderer, RendererOutput

_SCRIPT_PATH = Path(__file__).parent / "_blender_script.py"


class BlenderRenderer(AbstractRenderer):
    def name(self) -> str:
        return "Blender"

    def is_available(self) -> bool:
        return shutil.which("blender") is not None

    def render(
        self,
        mesh: BoardMesh,
        output_path: Path,
        width: int = 1920,
        height: int = 1080,
        view: Literal["top", "side", "profile", "perspective"] = "perspective",
        background_color: tuple[float, float, float] = (0.15, 0.15, 0.15),
    ) -> RendererOutput:
        output_path = Path(output_path)
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
            obj_path = Path(tmp.name)

        try:
            _write_obj(mesh, obj_path)
            args = [
                shutil.which("blender"),
                "--background",
                "--python", str(_SCRIPT_PATH),
                "--",
                str(obj_path),
                str(output_path),
                view,
                str(width),
                str(height),
            ]
            result = subprocess.run(args, timeout=120, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"Blender exited {result.returncode}: {result.stderr.decode()[:500]}")
        finally:
            obj_path.unlink(missing_ok=True)

        return RendererOutput(image_path=output_path, width=width, height=height)


def _write_obj(mesh: BoardMesh, path: Path) -> None:
    lines = []
    for v in mesh.vertices:
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")
    for n in mesh.normals:
        lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
    for tri in mesh.triangles:
        i, j, k = tri[0] + 1, tri[1] + 1, tri[2] + 1
        lines.append(f"f {i}//{i} {j}//{j} {k}//{k}")
    path.write_text("\n".join(lines))
