"""
Debug the Jacobian computation to understand determinant signs.
"""

import numpy as np
from src.shapes import disk
from src.param.tutte import tutte_embedding
from src.distortion import compute_triangle_jacobian, compute_triangle_area_signed


def main():
    print("\n" + "="*70)
    print("DEBUG: Jacobian Determinant Signs")
    print("="*70)

    # Generate a simple disk
    V, F = disk(n_radial=3, n_angular=6, radius=1.0)
    print(f"\nGenerated disk: {V.shape[0]} vertices, {F.shape[0]} faces")

    # Compute Tutte embedding
    UV = tutte_embedding(V, F, weight_type='uniform')

    # Check first 10 triangles
    print("\n" + "-"*70)
    print("Sample triangle analysis:")
    print("-"*70)
    print(f"{'Tri':<4} {'3D Area':<10} {'UV Area':<10} {'det(J)':<10} {'Match':<6}")
    print("-"*70)

    for i in range(min(10, len(F))):
        face = F[i]
        v0, v1, v2 = V[face[0]], V[face[1]], V[face[2]]
        uv0, uv1, uv2 = UV[face[0]], UV[face[1]], UV[face[2]]

        # 3D area (using XY projection for planar disk)
        area_3d = compute_triangle_area_signed(v0[:2], v1[:2], v2[:2])

        # UV area
        area_uv = compute_triangle_area_signed(uv0, uv1, uv2)

        # Jacobian
        J = compute_triangle_jacobian(v0, v1, v2, uv0, uv1, uv2)
        det_J = np.linalg.det(J)

        # Check if signs match
        sign_3d = np.sign(area_3d)
        sign_uv = np.sign(area_uv)
        sign_det = np.sign(det_J)

        match = "✓" if sign_3d == sign_uv else "✗"

        print(f"{i:<4} {area_3d:>9.4f} {area_uv:>9.4f} {det_J:>9.4f} {match:<6}")

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("If det(J) has opposite sign from both 3D and UV areas,")
    print("then the local coordinate system we built has flipped orientation.")
    print("This doesn't mean there's a fold-over - it just means the Jacobian")
    print("is measuring the inverse transformation.")


if __name__ == "__main__":
    main()
