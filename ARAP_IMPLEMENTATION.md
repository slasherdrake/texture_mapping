# ARAP (As-Rigid-As-Possible) Parameterization

## Overview

ARAP is an iterative optimization method that produces high-quality mesh parameterizations by minimizing both angle and area distortion. It balances the conformal (angle-preserving) properties of LSCM with better area preservation than Tutte.

## Implementation

**File**: `src/param/arap.py`

### Algorithm Structure

ARAP uses a **local-global alternating optimization**:

1. **Initialization**: Start with Tutte embedding (guarantees valid parameterization)
2. **Iterate until convergence**:
   - **LOCAL STEP**: For each triangle, compute optimal 2×2 rotation matrix via SVD
   - **GLOBAL STEP**: Solve sparse linear system for vertex positions with fixed rotations
3. **Return**: Final UV coordinates

### Key Components

**Cotangent Weights**:
```python
w_ij = (cot α + cot β) / 2
```
where α and β are angles opposite edge (i,j).

**Local Step** (Rotation Extraction):
- For each triangle, build covariance matrix: `S = U * V^T`
- Apply SVD: `S = U_svd * Σ * Vt_svd`
- Extract rotation: `R = U_svd * Vt_svd`
- Ensure det(R) = +1 (proper rotation)

**Global Step** (Position Solve):
- Build sparse linear system: `L * u = b`
- Boundary vertices pinned to Tutte positions (hard constraints)
- Solve using adaptive solver:
  - Try `spsolve` (direct, fast, accurate for well-conditioned systems)
  - Fall back to `lsqr` (iterative, robust for ill-conditioned/singular matrices)
- This two-stage approach enables ARAP to handle both small and large boundaries

### Performance Results

ARAP vs. Tutte comparison on all test meshes:

| Mesh | Boundary Size | Tutte Angle | ARAP Angle | Improvement |
|------|---------------|-------------|------------|-------------|
| **Disk** | 40 vertices | 2.278 | 1.000 | 56.1% |
| **Hemisphere** | 120 vertices | 2.678 | 2.284 | 14.7% |
| **Spot** | 122 vertices | 36.555 | 10.985 | 70.0% |

**Key Results**:
- ARAP improves angle distortion over Tutte on **all meshes**
- Works reliably on both small and large boundaries (40-122+ vertices)
- Perfect results on Disk (1.000 angle, 0 fold-overs)
- Spot sees dramatic 70% improvement despite challenging geometry

## Usage

### From Python

```python
from src.param.arap import arap_embedding

# Compute ARAP parameterization
uv = arap_embedding(V, F, max_iterations=10, tolerance=1e-5)

# Or provide custom initialization
from src.param.lscm import lscm_embedding
uv_init = lscm_embedding(V, F)
uv = arap_embedding(V, F, initial_uv=uv_init, max_iterations=10)
```

### In Viewer

ARAP is available in `src.viewer_clean` for all disk-topology meshes:

```bash
python -m src.viewer_clean
```

Select any mesh (Disk, Hemisphere, Spot, Bunny, Armadillo) and choose **"ARAP"** from the method dropdown to see:
- Left: Checkerboard texture showing parameterization
- Right: Distortion heatmap (angle or area)

## Technical Details

### Convergence

- Typically converges in **5-10 iterations**
- Convergence measured by UV coordinate change between iterations
- Default tolerance: 1e-5

### Boundary Handling

ARAP fixes the **entire boundary** (like Tutte) to prevent:
- Drift/translation of parameterization
- Rotation ambiguity
- Numerical instability

This differs from LSCM, which only pins 2 boundary vertices.

### Advantages

1. **Balanced distortion**: Better than Tutte for both angle and area on all test meshes
2. **Robust solving**: Uses direct solver (spsolve) with automatic fallback to iterative solver (LSQR) for ill-conditioned systems
3. **Handles large boundaries**: Works on boundaries from 40 to 122+ vertices
4. **Fast convergence**: Typically 5-10 iterations
5. **Industry standard**: Widely used in games, VFX, and modeling

### Limitations

1. **Requires disk topology**: Must have single boundary loop (like Tutte, unlike LSCM Atlas)
2. **Slower than direct methods**: Iterative (5-10 iterations) vs. single solve (Tutte/LSCM)
3. **Needs good initialization**: Uses Tutte as starting point
4. **Fixed boundary**: Pins all boundary vertices to Tutte positions
5. **May increase fold-overs**: On highly irregular meshes (e.g., Spot), ARAP optimizes for angle preservation which can increase fold-overs compared to Tutte

**Note**: For meshes with severe geometric irregularities, **LSCM Atlas** may provide better overall results through automatic segmentation.

## References

- Sorkine & Alexa, "As-Rigid-As-Possible Surface Modeling" (2007)
- Liu et al., "A Local/Global Approach to Mesh Parameterization" (2008)

## Implementation Notes

All code is implemented from scratch per course requirements:
- Cotangent weight computation
- SVD-based rotation extraction  
- Sparse linear system construction
- Iterative local-global optimization

Only uses: NumPy (arrays), SciPy (sparse matrices, SVD, sparse solver)
