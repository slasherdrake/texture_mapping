"""
View all available test meshes (both generated and loaded from files).

Run with: python src/view_all_meshes.py
"""

import os
import sys
import numpy as np
import polyscope as ps

# Add parent directory to path to enable imports when run directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shapes import uv_sphere, hemisphere, torus, disk

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def load_and_normalize_mesh(filepath, scale=1.0):
    """
    Load a mesh from file and normalize it to fit in a unit box.

    Returns
    -------
    V : ndarray, vertices
    F : ndarray, faces
    """
    mesh = trimesh.load(filepath)
    V = np.array(mesh.vertices, dtype=np.float64)
    F = np.array(mesh.faces, dtype=np.int64)

    # Center the mesh
    center = V.mean(axis=0)
    V -= center

    # Scale to unit box
    max_extent = np.max(np.abs(V))
    if max_extent > 0:
        V /= max_extent
        V *= scale

    return V, F


def main():
    """Display all available test meshes in a grid layout."""

    # Initialize polyscope
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")

    meshes_to_show = []

    # Generated meshes
    print("Generating analytic meshes...")

    V_sphere, F_sphere = uv_sphere(n_lat=30, n_lon=60, radius=1.0)
    meshes_to_show.append(("UV Sphere", V_sphere, F_sphere, [0, 0, 0]))

    V_hemi, F_hemi = hemisphere(n_lat=30, n_lon=60, radius=1.0)
    meshes_to_show.append(("Hemisphere", V_hemi, F_hemi, [2.5, 0, 0]))

    V_torus, F_torus = torus(n_major=40, n_minor=20, R=1.0, r=0.3)
    meshes_to_show.append(("Torus", V_torus, F_torus, [5, 0, 0]))

    V_disk, F_disk = disk(n_radial=20, n_angular=40, radius=1.0)
    meshes_to_show.append(("Disk", V_disk, F_disk, [7.5, 0, 0]))

    # Load meshes from files
    if HAS_TRIMESH:
        meshes_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "meshes")
        if os.path.exists(meshes_dir):
            obj_files = sorted([f for f in os.listdir(meshes_dir) if f.endswith('.obj')])

            print(f"Loading {len(obj_files)} mesh files...")

            for i, obj_file in enumerate(obj_files):
                obj_path = os.path.join(meshes_dir, obj_file)
                try:
                    V, F = load_and_normalize_mesh(obj_path, scale=1.0)

                    # Place in second row
                    offset = [i * 2.5, -2.5, 0]

                    name = obj_file.replace('.obj', '').capitalize()
                    meshes_to_show.append((name, V, F, offset))

                    print(f"  Loaded {obj_file}: {len(V)} vertices, {len(F)} faces")

                except Exception as e:
                    print(f"  Error loading {obj_file}: {e}")

    # Register all meshes with polyscope
    print(f"\nRegistering {len(meshes_to_show)} meshes with Polyscope...")

    for name, V, F, offset in meshes_to_show:
        V_offset = V + np.array(offset)
        ps.register_surface_mesh(name, V_offset, F, smooth_shade=True)

    print("\n" + "="*60)
    print("MESH GALLERY")
    print("="*60)
    print(f"\nTop row: Generated meshes (sphere, hemisphere, torus, disk)")
    print(f"Bottom row: Loaded meshes (bunny, spot, armadillo)")
    print(f"\nTotal meshes displayed: {len(meshes_to_show)}")
    print("\nControls:")
    print("  - Rotate: Left click + drag")
    print("  - Pan: Right click + drag")
    print("  - Zoom: Scroll wheel")
    print("  - Toggle meshes on/off in the sidebar")
    print("\nClose window to exit.")
    print("="*60)

    # Show the window
    ps.show()


if __name__ == "__main__":
    main()
