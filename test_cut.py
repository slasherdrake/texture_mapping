"""
Test mesh cutting on closed meshes.
"""

import numpy as np
import trimesh
from src.param.cut import cut_mesh_to_disk
from src.param.tutte import tutte_embedding
from src.distortion import compute_distortion_metrics


def test_cut_on_mesh(mesh_path):
    """Test cutting and Tutte embedding on a closed mesh."""
    print("\n" + "="*70)
    print(f"Testing: {mesh_path}")
    print("="*70)

    # Load mesh
    mesh = trimesh.load(mesh_path)
    V = np.array(mesh.vertices, dtype=np.float64)
    F = np.array(mesh.faces, dtype=np.int64)

    print(f"\nOriginal mesh: {V.shape[0]} vertices, {F.shape[0]} faces")

    # Center and scale
    center = V.mean(axis=0)
    V -= center
    scale = np.max(np.linalg.norm(V, axis=1))
    if scale > 0:
        V /= scale

    print(f"Centered and scaled to unit size")

    # Cut mesh
    print("\nCutting mesh to disk topology...")
    V_cut, F_cut, boundary = cut_mesh_to_disk(V, F)

    if len(boundary) == 0:
        print("ERROR: No boundary created!")
        return

    print(f"Cut mesh: {V_cut.shape[0]} vertices, {F_cut.shape[0]} faces")
    print(f"Boundary: {len(boundary)} vertices")

    # Apply Tutte embedding
    print("\nApplying Tutte embedding (uniform weights)...")
    try:
        UV = tutte_embedding(V_cut, F_cut, weight_type='uniform')
        print(f"Success! UV coordinates: {UV.shape}")

        # Compute distortion
        print("\nComputing distortion metrics...")
        metrics = compute_distortion_metrics(V_cut, F_cut, UV)

        print(f"\nResults:")
        print(f"  Mean angle distortion: {metrics['mean_angle_distortion']:.3f}")
        print(f"  Max angle distortion: {metrics['max_angle_distortion']:.3f}")
        print(f"  Mean area distortion: {metrics['mean_area_distortion']:.3f}")
        print(f"  Max area distortion: {metrics['max_area_distortion']:.3f}")
        print(f"  Fold-overs: {metrics['num_fold_overs']}")

        if metrics['num_fold_overs'] == 0:
            print(f"\n  ✓ SUCCESS: No fold-overs! Parameterization is valid.")
        else:
            print(f"\n  ✗ WARNING: {metrics['num_fold_overs']} fold-overs detected.")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n" + "="*70)
    print("MESH CUTTING TEST SUITE")
    print("="*70)

    # Test on spot (smallest mesh)
    test_cut_on_mesh("meshes/spot.obj")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
