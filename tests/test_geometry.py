import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from blankforge.data.model import BoardModel
from blankforge.geometry.board import BoardGeometryBuilder, _mesh_volume, _mesh_surface_area


def _build(template: str, resolution: int = 30):
    m = BoardModel.from_template(template)
    builder = BoardGeometryBuilder(use_occt=False)
    return builder.build(m, resolution=resolution)


def test_mesh_has_vertices():
    for template in ["longboard", "shortboard", "midlength"]:
        mesh, stats = _build(template)
        assert len(mesh.vertices) > 100, f"{template}: too few vertices"
        assert len(mesh.triangles) > 100, f"{template}: too few triangles"


def test_stats_plausible():
    for template in ["longboard", "shortboard", "midlength"]:
        mesh, stats = _build(template)
        assert stats.volume_cm3 > 0, f"{template}: volume <= 0"
        assert stats.surface_area_cm2 > 0, f"{template}: surface area <= 0"
        # Surfboards: 5–150 liters (5000–150000 cm³)
        assert 5000 < stats.volume_cm3 < 150000, f"{template}: volume {stats.volume_cm3} out of range"
        assert stats.nose_width_1in_mm < stats.width_mm * 1.1
        assert stats.tail_width_1in_mm < stats.width_mm * 1.1


def test_normals_unit_length():
    mesh, _ = _build("shortboard")
    norms = np.linalg.norm(mesh.normals, axis=1)
    assert np.allclose(norms, 1.0, atol=0.01), "Normals not unit length"


def test_vertex_types():
    mesh, _ = _build("shortboard")
    assert mesh.vertices.dtype == np.float32
    assert mesh.normals.dtype == np.float32
    assert mesh.triangles.dtype == np.int32


def test_triangle_indices_in_bounds():
    mesh, _ = _build("shortboard")
    n_verts = len(mesh.vertices)
    assert mesh.triangles.min() >= 0
    assert mesh.triangles.max() < n_verts
