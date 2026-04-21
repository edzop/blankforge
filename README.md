# BlankForge

**Parametric Surfboard Designer** — design surfboard shapes interactively, export to STL/OBJ for CNC or 3D printing.

BlankForge is a Python desktop app (PySide6 + OpenGL) for shaping surfboards parametrically. The board is defined by independent dimension curves (width, rocker, thickness, rail profile) edited through dedicated 2D views, with a real-time 3D preview.

---

## Features

### Editing

- **Parameters tab** — set overall length, width, thickness, and rocker. Adjusting a parameter scales the relevant curves proportionally so the board's shape is preserved.
- **Top View** — drag control points on the width curve. Symmetric silhouette, nose on the right.
- **Side View** — edit rocker (centroid) and thickness per station. Each thickness station can be set as **fixed mm** or as a **ratio** of the global thickness parameter — ratio-mode points scale automatically when you change the thickness parameter.
- **Rails** — per-station cross-section editor with apex ratio, deck/lower concave, rail ratio, and softness. Walk through stations with prev/next or copy parameters across stations.
- **Templates** — start from longboard, shortboard, midlength, or custom.

### 3D Preview

- Real-time OpenGL renderer with three-point lighting (warm key, cool fill, cold rim) for clear depth perception.
- **Solid**, **Wireframe**, and **Heatmap** shading modes.
- Heatmap shows mean-curvature variation (blue → red) with an adjustable sensitivity slider.
- Wireframe with toggleable longitudinal / latitudinal / crosshatch lines and density control.
- Orientation gizmo and floating view-control overlay (Reset / Top / Side / Front / Fit) in the upper-right of the viewport.
- Orthographic preset views; perspective for free orbit.
- Optional **Blender** offscreen renderer for high-quality stills.

### Geometry

- Pure-numpy mesh builder: monotone cubic (PCHIP) interpolation along the length, parametric rail cross-sections, strip-triangulated between stations, fan-capped at nose and tail.
- Rail cross-section: smooth deck → tangent arc → vertical drop → flat hull, so thickness is preserved at the rail edge regardless of softness.
- Volume and surface area computed via the divergence theorem.

### Data & Export

- Pydantic-validated data model.
- Native `.surfboard` JSON format.
- Export to **STL** and **OBJ**.

---

## Quickstart

```bash
# Dependencies (Ubuntu/Debian)
sudo apt install -y \
  python3-pyside6.qtcore python3-pyside6.qtgui python3-pyside6.qtwidgets \
  python3-pyside6.qtopengl python3-pyside6.qtopenglwidgets \
  python3-opengl python3-scipy python3-pydantic

# Run
python3 main.py
```

> **Note — PyOpenGL (`python3-opengl` / `PyOpenGL`)** is required for the 3D render view.
> The apt package above installs it for the system Python, but if you are running inside
> a virtual environment or a pyenv/conda Python it will not be visible there.
> In that case install it with pip instead:
>
> ```bash
> python3 -m pip install PyOpenGL
> # or, on an externally-managed system:
> python3 -m pip install --break-system-packages PyOpenGL
> ```
>
> Without PyOpenGL the app launches but the Rendered View stays black.

Optional: install Blender separately to enable the Blender render path from the Rendered View sidebar.

---

## Project Layout

```
blankforge/
  data/
    model.py          # Pydantic board model (parameters, curves, rail stations)
    serializer.py     # JSON load/save, STL/OBJ export
  geometry/
    curves.py         # PCHIP curve evaluator, rail cross-section generator
    board.py          # Mesh builder + volume/surface-area stats
  renderer/
    base.py           # Abstract renderer interface
    opengl.py         # Phong-shaded OpenGL renderer (key/fill/rim)
    blender.py        # Blender subprocess renderer
  ui/
    main_window.py    # Main window, geometry worker thread, tab wiring
    tabs/             # One file per tab
    widgets/          # Reusable widgets (sliders, viewports, control-point editor)
samples/              # Example .surfboard files
tests/                # Geometry tests + headless render runner
main.py
```

---

## Editing notes

- **Length / Width / Thickness / Rocker parameter changes** rescale the corresponding curves so the board's relative shape stays the same. Ratio-mode thickness stations are unaffected by parameter changes.
- **Nose (position 0) and tail (position L)** stations are locked in X — only their value can be edited.
- **Add a station** by double-clicking empty space in the Top or Side view.
- **Delete a station** by right-clicking it (a confirmation menu appears). Endpoints can't be deleted.
- **Pan** with middle mouse, **zoom** with scroll wheel.

---

## Status

Active development. The geometry and editing pipeline are functional; expect ongoing changes to the rail model, lighting, and export options.
