from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, NamedTuple

from blankforge.geometry.board import BoardMesh


class RendererOutput(NamedTuple):
    image_path: Path
    width: int
    height: int


class AbstractRenderer(ABC):
    @abstractmethod
    def render(
        self,
        mesh: BoardMesh,
        output_path: Path,
        width: int = 1920,
        height: int = 1080,
        view: Literal["top", "side", "profile", "perspective"] = "perspective",
        background_color: tuple[float, float, float] = (0.15, 0.15, 0.15),
    ) -> RendererOutput: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def name(self) -> str: ...
