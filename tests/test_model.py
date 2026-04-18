import json
import tempfile
from pathlib import Path

from blankforge.data.model import BoardModel, ControlPoint, CurveData
from blankforge.data.serializer import SurfboardSerializer


def test_template_presets():
    for template in ["longboard", "shortboard", "midlength"]:
        m = BoardModel.from_template(template)
        assert m.meta.template == template
        assert m.parameters.length_mm > 0
        assert len(m.curves.width.points) >= 3
        assert len(m.curves.rocker.points) >= 3
        assert len(m.curves.thickness.points) >= 3
        assert len(m.curves.rail) >= 2


def test_round_trip_json():
    for template in ["longboard", "shortboard", "midlength"]:
        m = BoardModel.from_template(template)
        with tempfile.NamedTemporaryFile(suffix=".surfboard", delete=False) as f:
            path = Path(f.name)
        try:
            SurfboardSerializer.save(m, path)
            loaded = SurfboardSerializer.load(path)
            assert loaded.meta.template == m.meta.template
            assert abs(loaded.parameters.length_mm - m.parameters.length_mm) < 0.01
            assert len(loaded.curves.width.points) == len(m.curves.width.points)
        finally:
            path.unlink(missing_ok=True)


def test_round_trip_samples():
    samples_dir = Path("samples")
    for sf in samples_dir.glob("*.surfboard"):
        m = SurfboardSerializer.load(sf)
        assert m.parameters.length_mm > 0
        assert m.parameters.width_mm > 0


def test_curve_interpolation():
    curve = CurveData(points=[
        ControlPoint(position_mm=0, value_mm=100),
        ControlPoint(position_mm=500, value_mm=200),
        ControlPoint(position_mm=1000, value_mm=150),
    ])
    v_start = curve.interpolated_value(0)
    v_end = curve.interpolated_value(1000)
    v_mid = curve.interpolated_value(500)
    assert abs(v_start - 100) < 1
    assert abs(v_end - 150) < 1
    assert abs(v_mid - 200) < 1


def test_model_validation():
    m = BoardModel.from_template("shortboard")
    m.parameters.length_mm = 2000.0
    assert m.parameters.length_mm == 2000.0
