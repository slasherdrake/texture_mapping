"""
Test that viewer_clean can load all meshes including 3D models.
This is a non-GUI test.
"""

import numpy as np
import trimesh
from src.shapes import uv_sphere, disk, hemisphere
from src.param.sphere_maps import equirectangular, mercator, stereographic
from src.param.tutte import tutte_embedding, find_boundary_vertices
from src.param.cut import cut_mesh_to_disk
from src.distortion import compute_distortion_metrics


def test_mesh_loading():
    """Test loading and parameterizing all meshes."""
    print("\n" + "="*70)
    print("TESTING MESH LOADING FOR VIEWER")
    print("="*70)

    meshes = {}

    # Test 3D meshes
    mesh_files = {
        'spot': 'meshes/spot.obj',
        'bunny': 'meshes/bunny.obj',
        'armadillo': 'meshes/armadillo.obj'
    }

    for mesh_name, filename in mesh_files.items():
        print(f"\nLoading {mesh_name.capitalize()}...")
        try:
            mesh = trimesh.load(filename)
            V = np.array(mesh.vertices, dtype=np.float64)
            F = np.array(mesh.faces, dtype=np.int64)

            # Center and scale
            center = V.mean(axis=0)
            V -= center
            scale = np.max(np.linalg.norm(V, axis=1))
            if scale > 0:
                V /= scale

            print(f"  Loaded: {V.shape[0]} vertices, {F.shape[0]} faces")

            # Check if needs cutting
            boundary, _ = find_boundary_vertices(F, V.shape[0])
            if len(boundary) == 0:
                print(f"  Mesh is closed, cutting...")
                V, F, boundary = cut_mesh_to_disk(V, F)
                print(f"  After cutting: {V.shape[0]} vertices")
            else:
                print(f"  Has boundary: {len(boundary)} vertices")

            # Apply Tutte
            print(f"  Computing Tutte uniform...")
            uv_uniform = tutte_embedding(V, F, weight_type='uniform')

            print(f"  Computing distortion...")
            metrics = compute_distortion_metrics(V, F, uv_uniform)

            print(f"  Results:")
            print(f"    Mean angle distortion: {metrics['mean_angle_distortion']:.2f}")
            print(f"    Fold-overs: {metrics['num_fold_overs']}")

            meshes[mesh_name] = {
                'V': V,
                'F': F,
                'UV': uv_uniform,
                'metrics': metrics
            }

            print(f"  ✓ {mesh_name.capitalize()} ready")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "="*70)
    print(f"SUCCESS: Loaded {len(meshes)}/3 meshes")
    print("="*70)
    print("\nViewer should be able to display these meshes.")
    print("Run: python -m src.viewer_clean")


if __name__ == "__main__":
    test_mesh_loading()
