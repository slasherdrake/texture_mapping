"""
Debug script to understand the fold-over issue.
"""

import numpy as np
from src.shapes import disk
from src.param.tutte import tutte_embedding


def compute_triangle_area_2d(uv0, uv1, uv2):
    """Compute signed area of a triangle in 2D."""
    # Area = 0.5 * |det([uv1-uv0, uv2-uv0])|
    # Signed area = 0.5 * det([uv1-uv0, uv2-uv0])
    v1 = uv1 - uv0
    v2 = uv2 - uv0
    return 0.5 * (v1[0] * v2[1] - v1[1] * v2[0])


def check_uv_triangles(UV, F):
    """Check if UV triangles have consistent orientation."""
    positive_area = 0
    negative_area = 0
    zero_area = 0

    for face in F:
        uv0 = UV[face[0]]
        uv1 = UV[face[1]]
        uv2 = UV[face[2]]

        area = compute_triangle_area_2d(uv0, uv1, uv2)

        if area > 1e-10:
            positive_area += 1
        elif area < -1e-10:
            negative_area += 1
        else:
            zero_area += 1

    print(f"UV Triangle orientation:")
    print(f"  Positive area (CCW): {positive_area}")
    print(f"  Negative area (CW):  {negative_area}")
    print(f"  Zero area:           {zero_area}")
    print(f"  Total:               {len(F)}")

    return positive_area, negative_area, zero_area


def main():
    print("\n" + "="*70)
    print("DEBUG: Tutte Embedding Fold-Over Detection")
    print("="*70)

    # Generate a simple disk
    V, F = disk(n_radial=5, n_angular=8, radius=1.0)
    print(f"\nGenerated disk: {V.shape[0]} vertices, {F.shape[0]} faces")

    # Check 3D triangle orientation
    print("\n3D mesh (XY plane):")
    pos_3d, neg_3d, zero_3d = check_uv_triangles(V[:, :2], F)

    # Compute Tutte embedding
    print("\nComputing Tutte embedding with uniform weights...")
    UV = tutte_embedding(V, F, weight_type='uniform')

    # Check UV triangle orientation
    print("\nUV parameterization:")
    pos_uv, neg_uv, zero_uv = check_uv_triangles(UV, F)

    # Check a few specific triangles
    print("\n" + "-"*70)
    print("Sample triangles:")
    print("-"*70)
    for i in range(min(5, len(F))):
        face = F[i]
        v0_3d, v1_3d, v2_3d = V[face[0]], V[face[1]], V[face[2]]
        uv0, uv1, uv2 = UV[face[0]], UV[face[1]], UV[face[2]]

        area_3d = compute_triangle_area_2d(v0_3d[:2], v1_3d[:2], v2_3d[:2])
        area_uv = compute_triangle_area_2d(uv0, uv1, uv2)

        print(f"\nTriangle {i}: vertices {face}")
        print(f"  3D area:  {area_3d:>8.4f}")
        print(f"  UV area:  {area_uv:>8.4f}")
        print(f"  Match:    {'✓' if (area_3d > 0 and area_uv > 0) or (area_3d < 0 and area_uv < 0) else '✗'}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    if pos_3d == pos_uv and neg_3d == neg_uv:
        print("✓ Orientation is preserved correctly!")
    else:
        print("✗ Orientation mismatch detected")
        print(f"  This suggests the fold-over detection may be incorrect,")
        print(f"  or the mesh has inconsistent winding order.")


if __name__ == "__main__":
    main()
