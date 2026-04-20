from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Core fin types
# ---------------------------------------------------------------------------

class FinPoint(BaseModel):
    """A control point on the fin planform outline."""
    model_config = ConfigDict(validate_assignment=True)
    x_mm: float       # horizontal from leading-edge base (0 = LE)
    y_mm: float       # vertical from base (0 = base, positive = tip direction)
    influence: float = Field(default=0.0, ge=0.0, le=1.0)  # spline tension: 0=smooth, 1=sharp corner
    sharpness: float = Field(default=0.0, ge=0.0, le=1.0)  # foil edge thinness: 0=blunt, 1=knife-edge


class FinFoil(BaseModel):
    """Cross-section foil properties for 3D extrusion."""
    model_config = ConfigDict(validate_assignment=True)
    symmetric: bool = True
    thickness_ratio: float = Field(default=0.12, ge=0.02, le=0.50)


# ---------------------------------------------------------------------------
# Box / plug type
# ---------------------------------------------------------------------------

BOX_PRESETS: dict[str, dict] = {
    "FCS":       {"slot_length_mm": 38.0, "slot_width_mm": 14.0},
    "FCS II":    {"slot_length_mm": 40.0, "slot_width_mm": 16.0},
    "Futures":   {"slot_length_mm": 96.0, "slot_width_mm": 10.0},
    "Glassed-in": {"slot_length_mm": 0.0, "slot_width_mm":  0.0},
}

BoxType = Literal["FCS", "FCS II", "Futures", "Glassed-in"]


class FinBox(BaseModel):
    """Box / plug type that attaches the fin to the board."""
    model_config = ConfigDict(validate_assignment=True)
    box_type: BoxType = "FCS II"
    slot_length_mm: float = 40.0
    slot_width_mm: float = 16.0

    @classmethod
    def from_preset(cls, preset: str) -> "FinBox":
        p = BOX_PRESETS.get(preset, BOX_PRESETS["FCS II"])
        return cls(box_type=preset, **p)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Placement on the board
# ---------------------------------------------------------------------------

class FinPlacement(BaseModel):
    """Where this fin sits on the board."""
    model_config = ConfigDict(validate_assignment=True)
    x_from_tail_mm: float = 300.0    # distance from tail end along board length
    y_from_center_mm: float = 0.0    # lateral offset (+ve = toward rail)
    cant_deg: float = Field(default=0.0, ge=-20.0, le=20.0)   # outward tilt
    toe_deg: float  = Field(default=0.0, ge=-10.0, le=10.0)   # toe-in angle


# ---------------------------------------------------------------------------
# Single fin definition
# ---------------------------------------------------------------------------

class FinDef(BaseModel):
    """A single fin: planform outline + foil + placement + box."""
    model_config = ConfigDict(validate_assignment=True)
    name: str = "Fin"
    points: list[FinPoint] = Field(default_factory=list)
    foil: FinFoil = Field(default_factory=FinFoil)
    placement: FinPlacement = Field(default_factory=FinPlacement)
    box: FinBox = Field(default_factory=FinBox)


# ---------------------------------------------------------------------------
# Full fin setup
# ---------------------------------------------------------------------------

SetupType = Literal["single", "twin", "thruster", "quad", "2+1", "5-fin"]


class FinSetup(BaseModel):
    """Complete fin configuration for a board."""
    model_config = ConfigDict(validate_assignment=True)
    setup_type: SetupType = "thruster"
    fins: list[FinDef] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Template outline factories
# ---------------------------------------------------------------------------

def _pts(coords: list[tuple]) -> list[FinPoint]:
    """Build FinPoint list from (x_mm, y_mm, influence=0, sharpness=0) tuples."""
    result = []
    for item in coords:
        x, y = float(item[0]), float(item[1])
        inf   = float(item[2]) if len(item) > 2 else 0.0
        sharp = float(item[3]) if len(item) > 3 else 0.0
        result.append(FinPoint(x_mm=x, y_mm=y, influence=inf, sharpness=sharp))
    return result


def _thruster_outline() -> list[FinPoint]:
    """Standard shortboard rake/thruster side fin (~100×115 mm).
    Tuple format: (x_mm, y_mm, influence, sharpness)
    influence = spline tension (0=smooth curve, 1=sharp corner)
    sharpness = foil edge thinness at this height (0=full foil, 1=knife-edge)
    """
    return _pts([
        (  0,   0, 0.9, 0.0),   # LE base — corner, full foil thickness
        (  5,  42, 0.0, 0.1),   # LE lower
        ( 16,  88, 0.0, 0.45),  # LE upper — foil starts to thin
        ( 38, 113, 0.3, 0.85),  # tip — very thin
        ( 70,  90, 0.0, 0.45),  # TE upper
        ( 88,  45, 0.0, 0.1),   # TE lower
        (100,   0, 0.9, 0.0),   # TE base — corner, full foil thickness
    ])


def _pivot_outline() -> list[FinPoint]:
    """Upright pivot fin — more drive, less sweep (~100×120 mm)."""
    return _pts([
        (  0,   0, 0.9, 0.0),
        (  4,  50, 0.0, 0.1),
        ( 10, 105, 0.0, 0.4),
        ( 30, 122, 0.3, 0.85),
        ( 62, 105, 0.0, 0.4),
        ( 84,  50, 0.0, 0.1),
        (100,   0, 0.9, 0.0),
    ])


def _keel_outline() -> list[FinPoint]:
    """Wide keel fin — classic twin/fish (~140×90 mm)."""
    return _pts([
        (  0,   0, 0.9, 0.0),
        (  0,  38, 0.0, 0.2),
        ( 15,  82, 0.0, 0.6),
        ( 55,  90, 0.2, 0.80),
        ( 95,  75, 0.0, 0.6),
        (128,  38, 0.0, 0.2),
        (140,   0, 0.9, 0.0),
    ])


def _d_fin_outline() -> list[FinPoint]:
    """Longboard D-fin / centre fin (~120×170 mm)."""
    return _pts([
        (  0,   0, 0.9, 0.0),
        (  2,  70, 0.0, 0.2),
        ( 18, 145, 0.0, 0.5),
        ( 45, 168, 0.3, 0.85),
        ( 80, 145, 0.0, 0.5),
        (110,  70, 0.0, 0.2),
        (120,   0, 0.9, 0.0),
    ])


def _center_thruster_outline() -> list[FinPoint]:
    """Smaller upright centre fin for 2+1 setups (~100×145 mm)."""
    return _pts([
        (  0,   0, 0.9, 0.0),
        (  5,  55, 0.0, 0.15),
        ( 15, 125, 0.0, 0.5),
        ( 35, 145, 0.3, 0.85),
        ( 68, 120, 0.0, 0.5),
        ( 88,  55, 0.0, 0.15),
        (100,   0, 0.9, 0.0),
    ])


# Template name → outline factory
FIN_TEMPLATES: dict[str, object] = {
    "Thruster/Rake": _thruster_outline,
    "Pivot":          _pivot_outline,
    "Keel":           _keel_outline,
    "D-Fin":          _d_fin_outline,
    "Center Thruster": _center_thruster_outline,
}


# ---------------------------------------------------------------------------
# Setup factories
# ---------------------------------------------------------------------------

def _thruster_setup(box_type: str = "FCS II") -> FinSetup:
    """Standard 3-fin thruster (left, centre-ish, right)."""
    box = FinBox.from_preset(box_type)
    pts = _thruster_outline()
    # Side fins: ~120 mm from tail, ±115 mm lateral, 3° cant, 3° toe
    left = FinDef(
        name="Left",
        points=pts,
        foil=FinFoil(),
        placement=FinPlacement(x_from_tail_mm=120, y_from_center_mm=115,
                               cant_deg=3.0, toe_deg=3.0),
        box=box,
    )
    right = FinDef(
        name="Right",
        points=pts,
        foil=FinFoil(),
        placement=FinPlacement(x_from_tail_mm=120, y_from_center_mm=-115,
                               cant_deg=-3.0, toe_deg=3.0),
        box=box,
    )
    center = FinDef(
        name="Center",
        points=_thruster_outline(),
        foil=FinFoil(),
        placement=FinPlacement(x_from_tail_mm=300, y_from_center_mm=0,
                               cant_deg=0.0, toe_deg=0.0),
        box=box,
    )
    return FinSetup(setup_type="thruster", fins=[left, center, right])


def _single_setup(box_type: str = "Futures") -> FinSetup:
    """Single longboard centre fin."""
    box = FinBox.from_preset(box_type)
    fin = FinDef(
        name="Center",
        points=_d_fin_outline(),
        foil=FinFoil(thickness_ratio=0.14),
        placement=FinPlacement(x_from_tail_mm=350, y_from_center_mm=0,
                               cant_deg=0.0, toe_deg=0.0),
        box=box,
    )
    return FinSetup(setup_type="single", fins=[fin])


def _two_plus_one_setup(box_type: str = "Futures") -> FinSetup:
    """2+1 longboard setup: centre fin + two smaller side fins."""
    box_center = FinBox.from_preset(box_type)
    box_side = FinBox.from_preset("FCS II")
    center = FinDef(
        name="Center",
        points=_center_thruster_outline(),
        foil=FinFoil(thickness_ratio=0.13),
        placement=FinPlacement(x_from_tail_mm=350, y_from_center_mm=0,
                               cant_deg=0.0, toe_deg=0.0),
        box=box_center,
    )
    left = FinDef(
        name="Left",
        points=_thruster_outline(),
        foil=FinFoil(),
        placement=FinPlacement(x_from_tail_mm=120, y_from_center_mm=100,
                               cant_deg=3.0, toe_deg=3.0),
        box=box_side,
    )
    right = FinDef(
        name="Right",
        points=_thruster_outline(),
        foil=FinFoil(),
        placement=FinPlacement(x_from_tail_mm=120, y_from_center_mm=-100,
                               cant_deg=-3.0, toe_deg=3.0),
        box=box_side,
    )
    return FinSetup(setup_type="2+1", fins=[left, center, right])


def _twin_setup(box_type: str = "FCS II") -> FinSetup:
    """Twin keel or pivot fin setup."""
    box = FinBox.from_preset(box_type)
    pts = _keel_outline()
    left = FinDef(
        name="Left",
        points=pts,
        foil=FinFoil(),
        placement=FinPlacement(x_from_tail_mm=200, y_from_center_mm=120,
                               cant_deg=5.0, toe_deg=2.0),
        box=box,
    )
    right = FinDef(
        name="Right",
        points=pts,
        foil=FinFoil(),
        placement=FinPlacement(x_from_tail_mm=200, y_from_center_mm=-120,
                               cant_deg=-5.0, toe_deg=2.0),
        box=box,
    )
    return FinSetup(setup_type="twin", fins=[left, right])


def default_fins_for_template(template: str) -> FinSetup:
    """Return an appropriate default FinSetup for a board template."""
    if template == "longboard":
        return _single_setup("Futures")
    elif template == "midlength":
        return _two_plus_one_setup("Futures")
    else:  # shortboard, custom
        return _thruster_setup("FCS II")
