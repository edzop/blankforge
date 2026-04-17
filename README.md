# BlankForge

## Parametric Surfboard Generator — Software Design Specification



-----

BlankForge is a cross-platform Python application for designing surfboard shapes parametrically. It provides real-time interactive editing through a PySide6 interface, with OpenCASCADE as the 3D geometry kernel and PyOpenGL for live rendering. All board data is stored as JSON, enabling data-driven variation exploration and AI pipeline integration.

-----

## 1. Technology Stack

|Component            |Technology         |Rationale                                         |
|---------------------|-------------------|--------------------------------------------------|
|GUI Framework        |PySide6            |Official Qt bindings, LGPL license, cross-platform|
|3D Geometry Kernel   |OpenCASCADE (OCCT) |Parametric solid modeling, CAD-grade precision    |
|Real-Time Renderer   |PyOpenGL           |Embedded in PySide window, correct depth testing  |
|High-Quality Renderer|Blender (pluggable)|Photorealistic output via swappable interface     |
|Native File Format   |JSON               |Lightweight, Python-native, AI/ML friendly        |
|Default Unit         |Millimeters (mm)   |Fabrication precision standard                    |

-----

## 2. Board Parameters

### 2.1 Global Dimensions

All measurements default to millimeters.

|Parameter                  |Description                             |
|---------------------------|----------------------------------------|
|Length                     |Overall board length, nose to tail      |
|Width                      |Maximum board width                     |
|Thickness                  |Maximum board thickness                 |
|Rocker                     |Overall nose-to-tail vertical curve     |
|Volume *(calculated)*      |Total board volume derived from geometry|
|Surface Area *(calculated)*|Total surface area derived from geometry|

### 2.2 Tail Configuration

Tail shape is part of the template selection, then refined in the Parameters tab.

|Setting            |Description                                               |
|-------------------|----------------------------------------------------------|
|Tail Shape         |Squaretail / Roundtail / Swallowtail / Dovetail (dropdown)|
|Tail Width (mm)    |Width at the tail end                                     |
|Tail Length (mm)   |Length of the tail section                                |
|Tail Thickness (mm)|Thickness at the tail end                                 |

-----

## 3. Control Point System

BlankForge models the board shape using independent parametric curves — one per dimension. Each curve has its own control points, so adjusting width never affects rocker, and vice versa. Curves are implemented as Bezier or NURBS splines along the board length.

### 3.1 Independent Dimension Curves

|Curve             |Controls                                          |Edited In   |
|------------------|--------------------------------------------------|------------|
|Width Curve       |Local width at each station along the board       |Top View    |
|Rocker Curve      |Vertical height / rocker at each station          |Side View   |
|Thickness Curve   |Local thickness at each station                   |Side View   |
|Rail Profile Curve|Rail sharpness and cross-section shape per station|Profile View|

### 3.2 Control Point Properties

- Default spacing: evenly distributed along board length in millimeters
- Spacing between control points is adjustable
- Control points can be added or removed dynamically
- Each point has two fine-grain sliders: position along length, and curve value

### 3.3 Rail Profile View

In the Profile View, the current station’s rail cross-section is displayed at full detail. Adjacent stations are shown as ghost overlays so the user can visualize transitions. A previous / next control enables stepping through all stations along the board.

Rail sharpness follows a typical surfboard gradient: sharp at the tail, soft and rounded through the mid-section, and slightly sharper again toward the nose.

-----

## 4. User Interface

### 4.1 Tab Layout and Workflow

|Tab|Name         |Purpose                                                       |Phase |
|---|-------------|--------------------------------------------------------------|------|
|1  |Template     |Select board type: Longboard / Shortboard / Midlength / Custom|Setup |
|2  |Parameters   |Adjust global dimensions and configure tail shape             |Setup |
|3  |Top View     |Edit width curve — orthographic top-down view                 |Edit  |
|4  |Side View    |Edit rocker and thickness curves — orthographic side view     |Edit  |
|5  |Profile View |Edit rail profile per station with ghost neighbor context     |Edit  |
|6  |Rendered View|3D perspective preview, free mouse rotation                   |Review|
|7  |Quad View    |All four views in a 2x2 grid simultaneously                   |Review|
|8  |Statistics   |Computed values: volume, surface area, dimensions             |Review|
|9  |Export       |Export to JSON, STL, or other formats                         |Output|

### 4.2 Viewport Controls

|Action      |Orthographic Views (Top/Side/Profile)      |Perspective View |
|------------|-------------------------------------------|-----------------|
|Rotate      |Not available — axis-constrained           |Left mouse drag  |
|Zoom        |Scroll wheel                               |Scroll wheel     |
|Pan         |Middle mouse button drag                   |Middle mouse drag|
|Edit        |Click control point to select; drag to move|View only        |
|Fine control|Two sliders: position + value              |N/A              |

-----

## 5. Rendering

BlankForge uses a pluggable renderer interface. The active renderer can be swapped without modifying the geometry engine or UI.

|Renderer|Use Case                  |Notes                                                               |
|--------|--------------------------|--------------------------------------------------------------------|
|PyOpenGL|Real-time editing viewport|Embedded in PySide window; correct depth testing prevents Z-fighting|
|Blender |Final high-quality renders|Invoked via renderer interface; matches box generator pipeline      |

-----

## 6. Data Model and File Format

The native BlankForge file format is JSON with a `.surfboard` extension. All parameters, curves, control points, and tail settings are stored in a single file. This enables programmatic generation of design variations and direct integration with AI/ML pipelines.

### 6.1 JSON Structure

```
blankforge.surfboard (JSON)
  meta:       { name, template, version, created, units }
  parameters: { length, width, thickness, rocker }
  tail:       { shape, width, length, thickness }
  curves:
    width:     [ { position_mm, value_mm }, ... ]
    rocker:    [ { position_mm, value_mm }, ... ]
    thickness: [ { position_mm, value_mm }, ... ]
    rail:      [ { position_mm, profile_data: {...} }, ... ]
```

### 6.2 Export Formats

- **JSON (.surfboard)** — native BlankForge format, version controlled, AI-ready
- **STL** — 3D printing and CNC manufacturing
- **OBJ** — general 3D interchange

-----

## 7. Project Structure

```
blankforge/
  geometry/
    board.py          # Core board geometry builder (OCCT)
    curves.py         # Bezier/NURBS curve evaluation
    tail.py           # Tail shape constructors
  renderer/
    base.py           # Abstract renderer interface
    opengl.py         # PyOpenGL real-time renderer
    blender.py        # Blender high-quality renderer
  ui/
    main_window.py    # PySide6 main window and tab manager
    tabs/             # One file per tab
    widgets/          # Reusable controls (sliders, viewports)
  data/
    model.py          # Pydantic board data model
    serializer.py     # JSON read/write
  manufacturing/      # Future CAM/export pipeline stub
samples/
  longboard_classic.surfboard
  shortboard_thruster.surfboard
  midlength_egg.surfboard
tests/
  render_all_samples.py   # Renders all samples to output/
output/                   # Per-sample subdirectories (gitignored)
main.py                   # Application entry point
pyproject.toml
```

### 7.1 Sample Test Runner

`render_all_samples.py` iterates through all `.surfboard` files in `samples/`, renders each one in all four views and exports statistics, writing everything to `output/{sample_name}/`.

```
output/longboard_classic/
  top.png
  side.png
  profile.png
  render.png
  statistics.json
```

-----

## 8. Computed Statistics

|Statistic   |Description                                |
|------------|-------------------------------------------|
|Volume      |Total board volume (cm³) from OCCT geometry|
|Surface Area|Total surface area (cm²) from OCCT geometry|
|Length      |Overall board length (mm)                  |
|Width       |Maximum board width (mm)                   |
|Thickness   |Maximum board thickness (mm)               |
|Nose Width  |Width measured at 300mm from nose          |
|Tail Width  |Width measured at 300mm from tail          |

-----

## 9. Future Considerations

- AI integration: feed `.surfboard` JSON to ML models for shape optimization and variation generation
- Manufacturing module: CAM toolpath generation from final OCCT geometry
- Fin system: parametric fin box placement and fin shape editor
- Material density input for estimated board weight calculation
- Batch variation explorer: programmatically generate and compare multiple JSON variants

-----

