# Tutte Embedding Implementation

## Overview

This document summarizes the Tutte embedding implementation completed on November 13, 2026.

## What Was Implemented

### Core Algorithm ([src/param/tutte.py](src/param/tutte.py))

Implemented Tutte's barycentric embedding for disk-topology meshes with the following components:

1. **Boundary Detection** (`find_boundary_vertices`)
   - Identifies boundary edges (edges appearing in only one triangle)
   - Constructs ordered boundary loop
   - Returns boundary vertex indices and boolean mask

2. **Adjacency Construction** (`build_adjacency`)
   - Builds vertex-to-vertex adjacency from triangle faces
   - Returns list of neighbor indices for each vertex

3. **Weight Computation**
   - **Uniform weights** (`compute_uniform_weights`): All neighbors get equal weight (1/n)
   - **Mean-value weights** (`compute_mean_value_weights`): Weights based on angles in mesh
     - Formula: w_ij = (tan(α_k/2) + tan(α_k+1/2)) / ||v_i - v_j||
     - Provides better shape preservation than uniform weights

4. **Main Algorithm** (`tutte_embedding`)
   - Fixes boundary vertices to unit circle
   - Solves sparse linear system for interior vertex positions
   - Guarantees no fold-overs (Tutte's theorem)

### Fold-Over Detection Fix ([src/distortion.py](src/distortion.py))

Fixed the fold-over detection logic to correctly identify flipped triangles:
- Previous implementation incorrectly flagged triangles with det(J) < 0 as fold-overs
- New implementation compares 3D triangle orientation vs UV triangle orientation
- Properly handles meshes with mixed winding orders

### Test Suite

Created comprehensive tests:

1. **[test_tutte.py](test_tutte.py)** - Full validation suite
   - Tests on planar disk mesh
   - Tests on curved hemisphere mesh
   - Compares uniform vs mean-value weights
   - Verifies zero fold-overs (Tutte's guarantee)

2. **[debug_tutte.py](debug_tutte.py)** - Orientation debugging
3. **[debug_jacobian.py](debug_jacobian.py)** - Jacobian determinant analysis
4. **[test_viewer_integration.py](test_viewer_integration.py)** - Viewer integration tests

### Viewer Integration ([src/viewer.py](src/viewer.py))

Enhanced the interactive viewer to display:
- UV Sphere (center): Analytic sphere projections
- Hemisphere (left): Tutte embedding of curved surface
- Disk (right): Tutte embedding of planar surface

Each mesh shows both uniform and mean-value weight parameterizations.

## Key Results

### Validation of Tutte's Theorem

All tests confirm **zero fold-overs** for both weight types on all meshes, validating Tutte's theorem that guarantees valid parameterizations.

### Distortion Comparison

From test results on a planar disk (201 vertices, 380 faces):

| Metric | Uniform Weights | Mean-Value Weights |
|--------|----------------|-------------------|
| Mean angle distortion (σ₁/σ₂) | 2.3193 | 1.0000 |
| Max angle distortion | 3.9023 | 1.0000 |
| Mean area distortion (σ₁×σ₂) | 0.7095 | 1.0000 |
| Max area distortion | 2.3003 | 1.0000 |
| Fold-overs | 0 | 0 |

**Key observation**: For the planar disk, mean-value weights achieve **perfect** parameterization (distortion = 1.0) because the mesh is already flat.

### Hemisphere Results

Hemisphere (411 vertices, 760 faces) shows higher distortion due to flattening a curved surface:

| Metric | Uniform Weights | Mean-Value Weights |
|--------|----------------|-------------------|
| Mean angle distortion | 2.6180 | 2.4783 |
| Max angle distortion | 19.1990 | 19.1990 |
| Mean area distortion | 0.5228 | 0.5355 |
| Fold-overs | 0 | 0 |

Mean-value weights show slightly better angle preservation but similar overall performance on curved surfaces.

## Technical Details

### Tutte's Theorem

**Statement**: If you fix the boundary vertices of a disk-topology mesh to a convex polygon and place each interior vertex at the weighted average of its neighbors (with positive weights), the resulting parameterization has no fold-overs.

**Implementation**:
```
For each interior vertex i:
    UV[i] = Σ(w_ij * UV[j]) for all neighbors j

Boundary vertices fixed to:
    UV[boundary[k]] = (cos(2πk/n), sin(2πk/n))
```

This creates a sparse linear system:
```
(I - W_interior) · UV_interior = W_boundary · UV_boundary
```

Solved using `scipy.sparse.linalg.spsolve`.

### Mean-Value Coordinates (Floater 2003)

For vertex i with neighbor j sharing triangles with third vertices k:
```
w_ij = (Σ tan(α_k/2)) / ||v_i - v_j||
```

where α_k is the angle at vertex i in triangle (i,j,k).

Weights are normalized to sum to 1 for each vertex.

## References

- Tutte, W. T. "How to draw a graph." *Proceedings of the London Mathematical Society* 3.1 (1963): 743-767.
- Floater, M. S. "Mean value coordinates." *Computer aided geometric design* 20.1 (2003): 19-27.
- Hormann et al. "Mesh Parameterization: Theory and Practice." *SIGGRAPH Course Notes* (2007).

## Files Modified/Created

**New files:**
- `src/param/tutte.py` - Main Tutte embedding implementation
- `test_tutte.py` - Comprehensive test suite
- `debug_tutte.py` - Debugging script
- `debug_jacobian.py` - Jacobian analysis
- `test_viewer_integration.py` - Integration tests
- `TUTTE_IMPLEMENTATION.md` - This document

**Modified files:**
- `src/distortion.py` - Fixed fold-over detection
- `src/viewer.py` - Added Tutte visualization

## Next Steps

Per [project_tasks.tex](project_tasks.tex), the next milestones are:

1. **D2** (Due Nov 4-11): Theory tests
   - Verify Mercator σ₁/σ₂ ≈ 1 (conformal)
   - Verify Tutte has zero fold-overs ✓ (DONE)

2. **V2** (Due Nov 5-12): Distortion heatmaps and checkerboard texture

3. **M3** (Due Nov 9-21): LSCM implementation with two pinned boundary vertices

4. **K1** (Due Nov 17): Milestone submission
