from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ControlPoint(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    position_mm: float
    value_mm: float


class RailProfile(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    apex_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    upper_concave: float = Field(default=0.0, ge=-1.0, le=1.0)
    lower_rail_angle: float = Field(default=45.0, ge=0.0, le=90.0)
    softness: float = Field(default=0.5, ge=0.0, le=1.0)


class RailStation(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    position_mm: float
    profile: RailProfile = Field(default_factory=RailProfile)


class CurveData(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    points: list[ControlPoint] = Field(default_factory=list)

    def sorted_points(self) -> list[ControlPoint]:
        return sorted(self.points, key=lambda p: p.position_mm)

    def interpolated_value(self, position_mm: float) -> float:
        pts = self.sorted_points()
        if not pts:
            return 0.0
        if len(pts) == 1:
            return pts[0].value_mm
        xs = np.array([p.position_mm for p in pts])
        ys = np.array([p.value_mm for p in pts])
        if position_mm <= xs[0]:
            return float(ys[0])
        if position_mm >= xs[-1]:
            return float(ys[-1])
        idx = int(np.searchsorted(xs, position_mm, side="right")) - 1
        t = (position_mm - xs[idx]) / (xs[idx + 1] - xs[idx])
        return float(ys[idx] + t * (ys[idx + 1] - ys[idx]))


class BoardCurves(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    width: CurveData = Field(default_factory=CurveData)
    rocker: CurveData = Field(default_factory=CurveData)
    thickness: CurveData = Field(default_factory=CurveData)
    rail: list[RailStation] = Field(default_factory=list)


class TailConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    shape: Literal["squaretail", "roundtail", "swallowtail", "dovetail"] = "squaretail"
    width_mm: float = Field(default=300.0, gt=0)
    length_mm: float = Field(default=150.0, gt=0)
    thickness_mm: float = Field(default=45.0, gt=0)


class BoardParameters(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    length_mm: float = Field(default=2000.0, gt=0)
    width_mm: float = Field(default=530.0, gt=0)
    thickness_mm: float = Field(default=65.0, gt=0)
    rocker_mm: float = Field(default=40.0, ge=0)


class BoardMeta(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    name: str = "Untitled Board"
    template: Literal["longboard", "shortboard", "midlength", "custom"] = "shortboard"
    version: str = "1.0"
    created: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    units: str = "mm"


class BoardModel(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    meta: BoardMeta = Field(default_factory=BoardMeta)
    parameters: BoardParameters = Field(default_factory=BoardParameters)
    tail: TailConfig = Field(default_factory=TailConfig)
    curves: BoardCurves = Field(default_factory=BoardCurves)

    @classmethod
    def from_template(cls, template: str) -> "BoardModel":
        template = template.lower()
        presets: dict[str, dict] = {
            "longboard": {
                "length_mm": 2743, "width_mm": 560, "thickness_mm": 70, "rocker_mm": 25,
                "tail_shape": "roundtail", "tail_width": 380, "tail_length": 180, "tail_thickness": 55,
                "nose_width": 420, "wide_point_width": 560, "wide_point_pos": 0.45,
                "rocker_nose": 30, "rocker_tail": 10,
                "thick_center": 70, "thick_nose": 45, "thick_tail": 40,
            },
            "shortboard": {
                "length_mm": 1880, "width_mm": 510, "thickness_mm": 60, "rocker_mm": 50,
                "tail_shape": "squaretail", "tail_width": 300, "tail_length": 120, "tail_thickness": 40,
                "nose_width": 270, "wide_point_width": 510, "wide_point_pos": 0.42,
                "rocker_nose": 55, "rocker_tail": 20,
                "thick_center": 60, "thick_nose": 32, "thick_tail": 30,
            },
            "midlength": {
                "length_mm": 2286, "width_mm": 540, "thickness_mm": 65, "rocker_mm": 35,
                "tail_shape": "roundtail", "tail_width": 350, "tail_length": 160, "tail_thickness": 48,
                "nose_width": 370, "wide_point_width": 540, "wide_point_pos": 0.44,
                "rocker_nose": 38, "rocker_tail": 14,
                "thick_center": 65, "thick_nose": 40, "thick_tail": 38,
            },
        }
        if template not in presets and template != "custom":
            template = "shortboard"

        if template == "custom":
            return cls(meta=BoardMeta(template="custom"), curves=_default_curves(2000, 530, 65, 40, 300, 65))

        p = presets[template]
        L = p["length_mm"]

        parameters = BoardParameters(
            length_mm=L,
            width_mm=p["width_mm"],
            thickness_mm=p["thickness_mm"],
            rocker_mm=p["rocker_mm"],
        )
        tail = TailConfig(
            shape=p["tail_shape"],
            width_mm=p["tail_width"],
            length_mm=p["tail_length"],
            thickness_mm=p["tail_thickness"],
        )
        curves = _build_template_curves(L, p)
        meta = BoardMeta(name=f"New {template.capitalize()}", template=template)
        return cls(meta=meta, parameters=parameters, tail=tail, curves=curves)


def _build_template_curves(length_mm: float, p: dict) -> BoardCurves:
    L = length_mm
    wp = p["wide_point_pos"]

    width_pts = [
        ControlPoint(position_mm=0, value_mm=p["nose_width"] / 2),
        ControlPoint(position_mm=L * wp, value_mm=p["wide_point_width"] / 2),
        ControlPoint(position_mm=L * 0.8, value_mm=p["tail_width"] * 0.75),
        ControlPoint(position_mm=L, value_mm=p["tail_width"] / 2),
    ]

    rocker_pts = [
        ControlPoint(position_mm=0, value_mm=p["rocker_nose"]),
        ControlPoint(position_mm=L * 0.3, value_mm=p["rocker_mm"] * 0.3),
        ControlPoint(position_mm=L * 0.7, value_mm=p["rocker_tail"] * 0.5),
        ControlPoint(position_mm=L, value_mm=p["rocker_tail"]),
    ]

    thickness_pts = [
        ControlPoint(position_mm=0, value_mm=p["thick_nose"] / 2),
        ControlPoint(position_mm=L * 0.35, value_mm=p["thick_center"] / 2),
        ControlPoint(position_mm=L * 0.7, value_mm=p["thick_center"] * 0.45),
        ControlPoint(position_mm=L, value_mm=p["thick_tail"] / 2),
    ]

    rail_stations = [
        RailStation(position_mm=0, profile=RailProfile(apex_ratio=0.55, upper_concave=0.1, lower_rail_angle=55, softness=0.7)),
        RailStation(position_mm=L * 0.3, profile=RailProfile(apex_ratio=0.5, upper_concave=0.0, lower_rail_angle=45, softness=0.6)),
        RailStation(position_mm=L * 0.6, profile=RailProfile(apex_ratio=0.45, upper_concave=-0.1, lower_rail_angle=40, softness=0.4)),
        RailStation(position_mm=L, profile=RailProfile(apex_ratio=0.4, upper_concave=-0.2, lower_rail_angle=35, softness=0.2)),
    ]

    return BoardCurves(
        width=CurveData(points=width_pts),
        rocker=CurveData(points=rocker_pts),
        thickness=CurveData(points=thickness_pts),
        rail=rail_stations,
    )


def _default_curves(L, w, t, r, tw, tt) -> BoardCurves:
    return _build_template_curves(L, {
        "nose_width": w * 0.5, "wide_point_width": w, "wide_point_pos": 0.43,
        "tail_width": tw, "rocker_nose": r * 0.7, "rocker_tail": r * 0.3,
        "rocker_mm": r, "thick_center": t, "thick_nose": t * 0.55, "thick_tail": tt,
    })
