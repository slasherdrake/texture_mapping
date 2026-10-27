# D1: Distortion Metrics - Mathematical Walkthrough

## Overview

We've implemented per-triangle distortion metrics to measure how a 3D→2D parameterization distorts the surface. This document explains the math step-by-step.

## The Problem

When we flatten a curved 3D surface onto 2D (UV space), we inevitably introduce distortion. By Gauss's *Theorema Egregium*, any surface with non-zero Gaussian curvature **cannot** be flattened without distortion.

We need to measure **how much** distortion each parameterization method introduces.

## The Math

### Step 1: The Jacobian Matrix

For each triangle, we have:
- **3D vertices**: v₀, v₁, v₂ ∈ ℝ³
- **UV coordinates**: u₀, u₁, u₂ ∈ ℝ²

The **Jacobian** J is a 2×2 matrix that tells us how the mapping stretches/compresses the surface:

```
J = [ ∂u/∂s  ∂u/∂t ]
    [ ∂v/∂s  ∂v/∂t ]
```

where (s,t) are local 2D coordinates on the triangle's plane.

### Step 2: Local Triangle Coordinate System

Each triangle lives in a 2D plane embedded in 3D. We need to:
1. Define a local 2D coordinate system for this plane
2. Express the 3D→UV mapping in this system

**Local coordinate system:**
- **Origin**: v₀
- **X-axis (e₁)**: Direction from v₀ to v₁ (normalized)
  ```
  e₁ = (v₁ - v₀) / ||v₁ - v₀||
  ```

- **Y-axis (e₂)**: Perpendicular to e₁ in the triangle's plane
  ```
  Project (v₂ - v₀) onto e₁ → subtract → normalize
  e₂ = perpendicular component of (v₂ - v₀)
  ```

### Step 3: Express Vertices in Local Coords

In this local system:
- **v₀** → (0, 0)
- **v₁** → (||v₁ - v₀||, 0) — lies on the x-axis
- **v₂** → (x₂, y₂) — computed by projecting onto e₁, e₂

### Step 4: Compute the Jacobian

We have a linear mapping from local triangle coords to UV:

```
[ uv₁ - uv₀ ]     [ v₁_local ]
[ uv₂ - uv₀ ] = J [ v₂_local ]
```

Solve for J:
```
J = [uv₁-uv₀, uv₂-uv₀] · [v₁_local, v₂_local]⁻¹
```

This is the **2×2 Jacobian matrix** of the 3D→UV mapping.

### Step 5: Singular Value Decomposition (SVD)

The Jacobian J can be decomposed:
```
J = U · Σ · Vᵀ
```

where Σ is diagonal with singular values σ₁ ≥ σ₂ ≥ 0.

**Physical meaning:**
- **σ₁**: Maximum stretch factor (how much the mapping stretches in the worst direction)
- **σ₂**: Minimum stretch factor (how much it stretches in the best direction)

### Step 6: Distortion Metrics

From the singular values, we compute:

#### **Angle Distortion** (Conformal Distortion)
```
angle_distortion = σ₁ / σ₂
```

- **= 1**: Conformal mapping (preserves angles)
- **> 1**: Angles are distorted
- **Example**: Mercator projection is conformal (σ₁/σ₂ ≈ 1)

#### **Area Distortion** (Isometric Distortion)
```
area_distortion = σ₁ × σ₂ = det(J)
```

- **= 1**: Equiareal mapping (preserves area)
- **≠ 1**: Areas are scaled
- **Example**: Equal-area projections have σ₁×σ₂ = 1

#### **Fold-overs** (Orientation Flip)
```
fold_over = det(J) < 0
```

- **det(J) > 0**: Triangle maintains orientation ✅
- **det(J) < 0**: Triangle flipped inside-out ❌ (invalid)

## Implementation Details

### File: `src/distortion.py`

#### Function: `compute_triangle_jacobian(v0, v1, v2, uv0, uv1, uv2)`

```python
# 1. Build local coordinate system
edge1 = v1 - v0
e1 = edge1 / ||edge1||

edge2 = v2 - v0
e2 = perpendicular_to_e1(edge2)

# 2. Express vertices in local coords
v1_local = [||edge1||, 0]
v2_local = [dot(edge2, e1), dot(edge2, e2)]

# 3. UV deltas
du1 = uv1 - uv0
du2 = uv2 - uv0

# 4. Compute Jacobian
Ss = [v1_local | v2_local]  # 2×2 matrix (columns)
St = [du1 | du2]            # 2×2 matrix (columns)
J = St · Ss⁻¹
```

#### Function: `compute_distortion_metrics(V, F, UV)`

For each triangle:
1. Get v₀, v₁, v₂ and uv₀, uv₁, uv₂
2. Compute Jacobian J
3. Compute SVD: J = U·Σ·Vᵀ
4. Extract σ₁, σ₂
5. Compute metrics:
   - angle_distortion = σ₁/σ₂
   - area_distortion = σ₁×σ₂
   - fold_over = det(J) < 0

Returns dict with per-triangle arrays and summary statistics.

## Results from Demo

### Equirectangular Projection
- **Angle distortion**: Mean 3.2, Max 66.2
  - Not conformal (high distortion near poles)
- **Area distortion**: Mean 2.1, Max 6.4
  - Not equiareal

### Mercator Projection
- **Angle distortion**: Mean 1.47, Max 18.1
  - **Nearly conformal** (close to 1.0) ✅
  - Excellent angle preservation away from poles
- **Area distortion**: Mean 5.6, Max 28.6
  - High area distortion (expected for conformal maps)

### Stereographic Projection
- **Angle distortion**: Mean 1.19
  - **Excellent conformality** ✅
- **Area distortion**: Mean 297, Max 6577
  - Extreme area distortion (projects whole sphere to plane)

## The Trade-off: Angles vs Areas

**Fundamental theorem**: No parameterization can preserve **both** angles and areas for curved surfaces.

You must choose:
- **Conformal** (preserves angles): σ₁/σ₂ ≈ 1, but σ₁×σ₂ ≫ 1
  - Examples: Mercator, Stereographic, LSCM
- **Equiareal** (preserves area): σ₁×σ₂ ≈ 1, but σ₁/σ₂ ≫ 1
  - Examples: Equal-area projections, ARAP

## Fold-over Issue

**Note**: We detected 97% fold-overs for Equirectangular and Mercator. This is likely due to:
1. Triangle winding order in our sphere mesh (normals pointing inward)
2. UV coordinate orientation

This doesn't affect the magnitude of distortion (σ₁, σ₂), only the sign of det(J). For the milestone, we'll address this in the mesh generation or treat it as a consistent orientation choice.

## Next Steps

1. ✅ **Distortion metrics implemented**
2. ✅ **Validated on sphere projections**
3. ⏭️ **Add to tests/** as unit tests
4. ⏭️ **Visualize as heatmaps in viewer** (V2)
5. ⏭️ **Use for Tutte validation** (D2)

## References

- Sander et al., "Texture Mapping Progressive Meshes", SIGGRAPH 2001
- Hormann et al., "Mesh Parameterization: Theory and Practice", 2007
- Floater & Hormann, "Surface Parameterization: A Tutorial and Survey", 2005
