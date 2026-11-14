"""
Interactive viewer for meshes and parameterizations using polyscope.

Run with: python -m src.viewer
Or: python src/viewer.py
"""

import os
import sys
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

# Add parent directory to path to enable imports when run directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shapes import uv_sphere, disk, hemisphere
from src.param.sphere_maps import equirectangular, mercator, stereographic
from src.param.tutte import tutte_embedding

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def main():
    """Main viewer entry point."""
    # Initialize polyscope
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")

    # Generate and register the UV sphere
    print("Generating UV sphere...")
    V_sphere, F_sphere = uv_sphere(n_lat=40, n_lon=80, radius=1.0)

    ps_mesh_sphere = ps.register_surface_mesh("UV Sphere", V_sphere, F_sphere,
                                               smooth_shade=True)

    # Compute and register sphere parameterizations
    print("Computing sphere parameterizations...")
    uv_equirect = equirectangular(V_sphere)
    uv_mercator = mercator(V_sphere)
    uv_stereo = stereographic(V_sphere)

    # Register sphere parameterizations
    ps_mesh_sphere.add_parameterization_quantity("Equirectangular", uv_equirect,
                                                  defined_on='vertices', enabled=True)
    ps_mesh_sphere.add_parameterization_quantity("Mercator", uv_mercator,
                                                  defined_on='vertices', enabled=False)
    ps_mesh_sphere.add_parameterization_quantity("Stereographic", uv_stereo,
                                                  defined_on='vertices', enabled=False)

    # Generate and register a hemisphere with Tutte embedding
    print("\nGenerating hemisphere for Tutte embedding...")
    V_hemi, F_hemi = hemisphere(n_lat=20, n_lon=40, radius=1.0)

    # Translate hemisphere to the side
    V_hemi_display = V_hemi.copy()
    V_hemi_display[:, 0] -= 2.5

    ps_mesh_hemi = ps.register_surface_mesh("Hemisphere (Tutte)", V_hemi_display, F_hemi,
                                             smooth_shade=True)

    # Compute Tutte embeddings
    print("Computing Tutte embeddings...")
    try:
        uv_tutte_uniform = tutte_embedding(V_hemi, F_hemi, weight_type='uniform')
        uv_tutte_meanvalue = tutte_embedding(V_hemi, F_hemi, weight_type='mean_value')

        ps_mesh_hemi.add_parameterization_quantity("Tutte (Uniform)", uv_tutte_uniform,
                                                    defined_on='vertices', enabled=True)
        ps_mesh_hemi.add_parameterization_quantity("Tutte (Mean-Value)", uv_tutte_meanvalue,
                                                    defined_on='vertices', enabled=False)
        print("Tutte embeddings added successfully!")
    except Exception as e:
        print(f"Error computing Tutte embedding: {e}")

    # Generate and register a planar disk with Tutte embedding
    print("\nGenerating planar disk for Tutte embedding...")
    V_disk, F_disk = disk(n_radial=15, n_angular=30, radius=1.0)

    # Translate disk to the other side
    V_disk_display = V_disk.copy()
    V_disk_display[:, 0] += 2.5

    ps_mesh_disk = ps.register_surface_mesh("Disk (Tutte)", V_disk_display, F_disk,
                                             smooth_shade=True)

    # Compute Tutte embeddings for disk
    print("Computing Tutte embeddings for disk...")
    try:
        uv_disk_uniform = tutte_embedding(V_disk, F_disk, weight_type='uniform')
        uv_disk_meanvalue = tutte_embedding(V_disk, F_disk, weight_type='mean_value')

        ps_mesh_disk.add_parameterization_quantity("Tutte (Uniform)", uv_disk_uniform,
                                                    defined_on='vertices', enabled=True)
        ps_mesh_disk.add_parameterization_quantity("Tutte (Mean-Value)", uv_disk_meanvalue,
                                                    defined_on='vertices', enabled=False)
        print("Disk Tutte embeddings added successfully!")
    except Exception as e:
        print(f"Error computing Tutte embedding for disk: {e}")

    # Try to load a mesh from the meshes/ directory
    meshes_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "meshes")
    obj_files = []
    if os.path.exists(meshes_dir):
        obj_files = [f for f in os.listdir(meshes_dir) if f.endswith('.obj')]

    if obj_files and HAS_TRIMESH:
        obj_path = os.path.join(meshes_dir, obj_files[0])
        print(f"Loading mesh from {obj_path}...")
        try:
            mesh = trimesh.load(obj_path)
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                # Translate mesh to the side so it doesn't overlap the sphere
                V_loaded = np.array(mesh.vertices, dtype=np.float64)
                F_loaded = np.array(mesh.faces, dtype=np.int64)

                # Center and scale the loaded mesh
                center = V_loaded.mean(axis=0)
                V_loaded -= center
                scale = np.max(np.linalg.norm(V_loaded, axis=1))
                if scale > 0:
                    V_loaded /= scale

                # Translate 3 units in x direction
                V_loaded[:, 0] += 3.0

                ps.register_surface_mesh(f"Loaded: {obj_files[0]}", V_loaded, F_loaded,
                                          smooth_shade=True)
                print(f"Loaded mesh with {len(V_loaded)} vertices and {len(F_loaded)} faces")
        except Exception as e:
            print(f"Error loading mesh: {e}")
    elif not obj_files:
        print(f"\nNo .obj files found in {meshes_dir}/")
        print("Place a .obj file there to visualize it alongside the sphere.")
    elif not HAS_TRIMESH:
        print("trimesh not available for loading meshes")

    print("\n" + "="*70)
    print("VIEWER READY!")
    print("="*70)
    print("\nDisplaying:")
    print("  - UV Sphere (center): Analytic sphere projections")
    print("  - Hemisphere (left): Tutte embedding of curved surface")
    print("  - Disk (right): Tutte embedding of planar surface")
    print("\nControls:")
    print("  - Toggle between parameterizations in the UI")
    print("  - Compare distortion between uniform and mean-value weights")
    print("  - Close the window to exit")
    print("="*70)

    # Show the window
    ps.show()


if __name__ == "__main__":
    main()
