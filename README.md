# Surface Mapping Project

Mapping 2D images onto curved 3D surfaces using custom parameterization methods.

## Overview

This project implements and evaluates various surface parameterization techniques:
- Analytic sphere projections (equirectangular, Mercator, stereographic)
- Tutte embedding (planned)
- Least Squares Conformal Maps (LSCM) (planned)

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
│   │   └── sphere_maps.py  # Analytic projections
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
