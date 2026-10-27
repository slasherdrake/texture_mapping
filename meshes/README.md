# Test Meshes

This directory contains test meshes for evaluating parameterization methods.

## Downloaded Meshes

- **bunny.obj** - Stanford Bunny (34,834 vertices, 69,451 faces)
  - Classic graphics test model
  - Not watertight (has holes)
  - Complex geometry with varying curvature

- **spot.obj** - Spot the Cow (3,225 vertices, 5,856 faces)
  - Mid-resolution quadruped
  - Good for testing on moderate complexity

- **armadillo.obj** - Armadillo (49,990 vertices, 99,976 faces)
  - High-resolution scan
  - Watertight genus-0 mesh
  - Complex surface with fine details

All meshes sourced from [Alec Jacobson's common 3D test models](https://github.com/alecjacobson/common-3d-test-models).

## Generated Meshes

Meshes can also be generated programmatically in `src/shapes.py`:

- `uv_sphere()` - Full sphere with UV parameterization
- `hemisphere()` - Upper hemisphere (disk topology)
- `torus()` - Donut shape (genus 1)
- `disk()` - Flat disk for Tutte/LSCM testing

## Usage

View all meshes with:
```bash
python src/view_all_meshes.py
```

Or load individual meshes:
```python
import trimesh
mesh = trimesh.load('meshes/bunny.obj')
```
