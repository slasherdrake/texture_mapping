"""
Interactive viewer for meshes and parameterizations using polyscope.

Run with: python -m src.viewer
"""

import os
import sys
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from src.shapes import uv_sphere
from src.param.sphere_maps import equirectangular, mercator, stereographic

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

    ps_mesh = ps.register_surface_mesh("UV Sphere", V_sphere, F_sphere,
                                        smooth_shade=True)

    # Compute and register parameterizations
    print("Computing parameterizations...")
    uv_equirect = equirectangular(V_sphere)
    uv_mercator = mercator(V_sphere)
    uv_stereo = stereographic(V_sphere)

    # Register parameterizations using the correct polyscope API
    # Note: polyscope 2.x uses add_parameterization_quantity
    ps_mesh.add_parameterization_quantity("Equirectangular", uv_equirect,
                                           defined_on='vertices', enabled=True)
    ps_mesh.add_parameterization_quantity("Mercator", uv_mercator,
                                           defined_on='vertices', enabled=False)
    ps_mesh.add_parameterization_quantity("Stereographic", uv_stereo,
                                           defined_on='vertices', enabled=False)

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

    print("\nViewer ready!")
    print("- Toggle between parameterizations in the UI")
    print("- Equirectangular shows characteristic pole bunching")
    print("- Close the window to exit")

    # Show the window
    ps.show()


if __name__ == "__main__":
    main()
