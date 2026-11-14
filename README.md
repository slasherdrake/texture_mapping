# Surface Mapping Project

Mapping 2D images onto curved 3D surfaces using custom parameterization methods.

## Overview

This project implements and evaluates various surface parameterization techniques:
- Analytic sphere projections (equirectangular, Mercator, stereographic)
- Tutte embedding (uniform and mean-value weights)
- Least Squares Conformal Maps (LSCM) with two pinned boundary vertices

The goal is to understand distortion introduced by each method and visualize the results interactively.

## Important Note

**All parameterization algorithms are implemented from scratch.** We do not use library functions like libigl's parameterization methods, as per course requirements. Libraries like NumPy, SciPy, trimesh (mesh I/O only), and polyscope (visualization only) are used for supporting functionality.

## Setup

### Prerequisites
- Python 3.8 or higher
- pip

### Installation

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Run the viewer
```bash
python -m src.viewer
```

This will open an interactive window showing:
- A UV sphere with selectable parameterizations
- Checkerboard texture showing distortion characteristics
- Any `.obj` mesh placed in the `meshes/` directory (displayed alongside the sphere)

Toggle between different projections in the UI to see how distortion varies.

The hemisphere and disk also offer **LSCM**, alongside both Tutte variants,
with angle and area distortion heatmaps. For a method dropdown and side-by-side
checkerboard/heatmap views, run:

```bash
python -m src.viewer_clean
```

Select **Hemisphere** or **Disk**, then choose **LSCM** from the Method dropdown.
Imported meshes also offer LSCM when their topology and winding pass validation.

### LSCM from Python

```python
from src.shapes import disk
from src.param.lscm import lscm_embedding

V, F = disk()
UV = lscm_embedding(V, F)
# Optionally supply two boundary indices and their exact UV coordinates:
# UV = lscm_embedding(V, F, pinned_vertices=[a, b],
#                     pinned_uv=[[0., 0.], [1., 0.]])
```

The implementation assembles area-weighted Cauchy-Riemann equations from scratch
and uses SciPy's sparse least-squares solver. Two well-separated boundary vertices
are pinned automatically; the rest of the boundary remains free. UV coordinates
are not normalized to a square, so custom pins remain exact.

Inputs must be connected, consistently wound triangle disks without degenerate
faces or unused vertices. Closed meshes need a valid cut first. The existing
experimental cut routine does not guarantee a valid disk, and some bundled OBJ
meshes have multiple boundaries or components; the clean viewer reports why LSCM
is unavailable and retains their existing methods. LSCM minimizes conformal
distortion; it does not guarantee area preservation or prevent overlap for every
mesh. See [LSCM_IMPLEMENTATION.md](LSCM_IMPLEMENTATION.md) for the equations.

### Run tests
```bash
pytest
```

Or for verbose output:
```bash
pytest -v
```

Tests verify:
- Mesh vertices lie on the unit sphere
- No degenerate or malformed faces
- Face normals point outward
- UV coordinates are in expected ranges
- Projections handle special cases (poles, equator) correctly

## Project Structure

```
.
├── meshes/              # Place .obj test models here
├── src/
│   ├── shapes.py        # Mesh generators (uv_sphere, hemisphere)
│   ├── param/
│   │   ├── sphere_maps.py  # Analytic projections
│   │   ├── tutte.py        # Fixed-boundary embeddings
│   │   └── lscm.py         # Free-boundary conformal least squares
│   └── viewer.py        # Polyscope viewer
├── tests/
│   └── test_shapes.py   # Mesh validation tests
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## AI Assistance

This repository was scaffolded using Claude Code (Anthropic), as required to disclose by the course. All parameterization algorithms are our own implementations.

## License

Course project for ECE 467/567. Not licensed for external use.
