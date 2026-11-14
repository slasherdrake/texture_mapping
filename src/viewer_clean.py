"""
Clean viewer with side-by-side mesh and heatmap visualization.

Shows one mesh at a time with:
- Left: Parameterized mesh with checkerboard texture
- Right: Distortion heatmap

Use UI to toggle between meshes and parameterization methods.

Run with: python -m src.viewer_clean
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
from src.param.tutte import tutte_embedding, find_boundary_vertices
from src.param.lscm import lscm_embedding
from src.param.cut import cut_mesh_to_disk
from src.distortion import compute_distortion_metrics

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("Warning: trimesh not available, cannot load .obj files")


# Global state for UI
class ViewerState:
    def __init__(self):
        self.current_mesh = 0  # Index into mesh_names
        self.current_method = 0  # Index into available methods for current mesh
        self.distortion_type = 0  # 0=angle, 1=area

        self.mesh_names = ["UV Sphere", "Hemisphere", "Disk", "Spot", "Bunny", "Armadillo"]
        # Will be populated with mesh data
        self.meshes = {}
        self.current_mesh_left = None
        self.current_mesh_right = None

    def available_methods(self):
        name = self.mesh_names[self.current_mesh]
        key = 'sphere' if name == 'UV Sphere' else name.lower()
        return list(self.meshes[key]['uvs'])


state = ViewerState()


def compute_vertex_distortion(metrics, F, num_vertices):
    """Convert per-face distortion to per-vertex by averaging."""
    angle_distortion = np.zeros(num_vertices)
    area_distortion = np.zeros(num_vertices)
    vertex_count = np.zeros(num_vertices)

    for face_idx, face in enumerate(F):
        angle_dist = metrics['angle_distortion'][face_idx]
        area_dist = metrics['area_distortion'][face_idx]

        if not np.isnan(angle_dist):
            for v_idx in face:
                angle_distortion[v_idx] += angle_dist
                area_distortion[v_idx] += area_dist
                vertex_count[v_idx] += 1

    # Average
    mask = vertex_count > 0
    angle_distortion[mask] /= vertex_count[mask]
    area_distortion[mask] /= vertex_count[mask]

    return angle_distortion, area_distortion


def update_display():
    """Update the displayed meshes based on current state."""
    # Remove old meshes
    if state.current_mesh_left is not None:
        ps.remove_surface_mesh("Mesh (Textured)")
    if state.current_mesh_right is not None:
        ps.remove_surface_mesh("Mesh (Heatmap)")

    # Get current mesh data
    # Map display name to storage key
    mesh_name = state.mesh_names[state.current_mesh]
    mesh_key_map = {
        "UV Sphere": "sphere",
        "Hemisphere": "hemisphere",
        "Disk": "disk",
        "Spot": "spot",
        "Bunny": "bunny",
        "Armadillo": "armadillo"
    }
    mesh_key = mesh_key_map.get(mesh_name, mesh_name.lower())

    if mesh_key not in state.meshes:
        print(f"Error: mesh '{mesh_key}' not loaded (from display name '{mesh_name}')")
        return

    mesh_data = state.meshes[mesh_key]

    methods = state.available_methods()

    # Ensure method index is valid
    if state.current_method >= len(methods):
        state.current_method = 0

    method_name = methods[state.current_method]
    V = mesh_data['V']
    F = mesh_data['F']
    UV = mesh_data['uvs'][method_name]
    metrics = mesh_data['metrics'][method_name]

    # Get distortion values
    angle_dist, area_dist = compute_vertex_distortion(metrics, F, V.shape[0])

    # Choose which distortion to show
    if state.distortion_type == 0:
        distortion_values = angle_dist
        # Use percentile for better visualization (exclude extreme outliers)
        max_val = np.nanpercentile(angle_dist, 95)  # 95th percentile
        min_val = 1.0
        dist_name = "Angle"
    else:
        distortion_values = area_dist
        # Use percentile for better visualization
        max_val = np.nanpercentile(area_dist, 95)  # 95th percentile
        min_val = 0.0
        dist_name = "Area"

    # Clamp distortion values to the range for better visualization
    distortion_values = np.clip(distortion_values, min_val, max_val)

    # Left mesh: with UV parameterization (checkerboard)
    V_left = V.copy()
    V_left[:, 0] -= 1.5  # Translate left

    state.current_mesh_left = ps.register_surface_mesh("Mesh (Textured)", V_left, F, smooth_shade=True)
    state.current_mesh_left.add_parameterization_quantity("UV", UV, defined_on='vertices', enabled=True)

    # Right mesh: with distortion heatmap
    V_right = V.copy()
    V_right[:, 0] += 1.5  # Translate right

    state.current_mesh_right = ps.register_surface_mesh("Mesh (Heatmap)", V_right, F, smooth_shade=True)
    state.current_mesh_right.add_scalar_quantity(
        f"{dist_name} Distortion",
        distortion_values,
        defined_on='vertices',
        enabled=True,
        cmap='turbo',
        vminmax=(min_val, max_val)
    )

    # Update info display
    print(f"\nDisplaying: {state.mesh_names[state.current_mesh]} - {method_name}")
    print(f"  Distortion type: {dist_name}")
    print(f"  Mean {dist_name.lower()}: {metrics[f'mean_{dist_name.lower()}_distortion']:.3f}")
    print(f"  Max {dist_name.lower()}: {metrics[f'max_{dist_name.lower()}_distortion']:.3f}")
    print(f"  Fold-overs: {metrics['num_fold_overs']}")


def ui_callback():
    """ImGui callback for custom UI controls."""
    psim.TextUnformatted("=== Mesh Selection ===")

    # Mesh selector
    changed, new_mesh = psim.Combo("Mesh", state.current_mesh, state.mesh_names)
    if changed:
        state.current_mesh = new_mesh
        state.current_method = 0  # Reset method when changing mesh
        update_display()

    psim.Separator()
    psim.TextUnformatted("=== Parameterization Method ===")

    # Only offer methods successfully computed for the selected mesh.
    methods = state.available_methods()

    changed, new_method = psim.Combo("Method", state.current_method, methods)
    if changed:
        state.current_method = new_method
        update_display()

    psim.Separator()
    psim.TextUnformatted("=== Distortion Type ===")

    # Distortion type selector
    changed, new_dist = psim.RadioButton("Angle Distortion", state.distortion_type, 0)
    if changed:
        state.distortion_type = 0
        update_display()

    changed, new_dist = psim.RadioButton("Area Distortion", state.distortion_type, 1)
    if changed:
        state.distortion_type = 1
        update_display()

    psim.Separator()
    psim.TextUnformatted("=== Info ===")
    psim.TextUnformatted("Left: UV parameterization (checkerboard)")
    psim.TextUnformatted("Right: Distortion heatmap")
    psim.TextUnformatted("")
    psim.TextUnformatted("Blue = low distortion")
    psim.TextUnformatted("Red = high distortion")


def main():
    """Main viewer entry point."""
    # Initialize polyscope
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")
    ps.set_user_callback(ui_callback)

    # Set better initial camera view
    ps.reset_camera_to_home_view()
    ps.look_at((0.0, -5.0, 2.0), (0.0, 0.0, 0.0))

    print("\n" + "="*70)
    print("LOADING MESHES AND COMPUTING PARAMETERIZATIONS")
    print("="*70)

    # ========== SPHERE ==========
    print("\n1. UV Sphere...")
    V_sphere, F_sphere = uv_sphere(n_lat=40, n_lon=80, radius=1.0)
    print(f"   Generated: {V_sphere.shape[0]} vertices, {F_sphere.shape[0]} faces")

    print("   Computing sphere parameterizations...")
    uv_equirect = equirectangular(V_sphere)
    uv_mercator = mercator(V_sphere)
    uv_stereo = stereographic(V_sphere)

    print("   Computing distortion metrics...")
    metrics_equirect = compute_distortion_metrics(V_sphere, F_sphere, uv_equirect)
    metrics_mercator = compute_distortion_metrics(V_sphere, F_sphere, uv_mercator)
    metrics_stereo = compute_distortion_metrics(V_sphere, F_sphere, uv_stereo)

    state.meshes['sphere'] = {
        'V': V_sphere,
        'F': F_sphere,
        'uvs': {
            'Equirectangular': uv_equirect,
            'Mercator': uv_mercator,
            'Stereographic': uv_stereo,
        },
        'metrics': {
            'Equirectangular': metrics_equirect,
            'Mercator': metrics_mercator,
            'Stereographic': metrics_stereo,
        }
    }

    # ========== HEMISPHERE ==========
    print("\n2. Hemisphere...")
    V_hemi, F_hemi = hemisphere(n_lat=20, n_lon=40, radius=1.0)
    print(f"   Generated: {V_hemi.shape[0]} vertices, {F_hemi.shape[0]} faces")

    print("   Computing Tutte embeddings...")
    uv_tutte_uniform = tutte_embedding(V_hemi, F_hemi, weight_type='uniform')
    uv_tutte_meanvalue = tutte_embedding(V_hemi, F_hemi, weight_type='mean_value')

    print("   Computing distortion metrics...")
    metrics_uniform = compute_distortion_metrics(V_hemi, F_hemi, uv_tutte_uniform)
    metrics_meanvalue = compute_distortion_metrics(V_hemi, F_hemi, uv_tutte_meanvalue)

    state.meshes['hemisphere'] = {
        'V': V_hemi,
        'F': F_hemi,
        'uvs': {
            'Tutte Uniform': uv_tutte_uniform,
            'Tutte Mean-Value': uv_tutte_meanvalue,
        },
        'metrics': {
            'Tutte Uniform': metrics_uniform,
            'Tutte Mean-Value': metrics_meanvalue,
        }
    }

    # ========== DISK ==========
    print("\n3. Planar Disk...")
    V_disk, F_disk = disk(n_radial=15, n_angular=30, radius=1.0)
    print(f"   Generated: {V_disk.shape[0]} vertices, {F_disk.shape[0]} faces")

    print("   Computing Tutte embeddings...")
    uv_disk_uniform = tutte_embedding(V_disk, F_disk, weight_type='uniform')
    uv_disk_meanvalue = tutte_embedding(V_disk, F_disk, weight_type='mean_value')

    print("   Computing distortion metrics...")
    metrics_disk_uniform = compute_distortion_metrics(V_disk, F_disk, uv_disk_uniform)
    metrics_disk_meanvalue = compute_distortion_metrics(V_disk, F_disk, uv_disk_meanvalue)

    state.meshes['disk'] = {
        'V': V_disk,
        'F': F_disk,
        'uvs': {
            'Tutte Uniform': uv_disk_uniform,
            'Tutte Mean-Value': uv_disk_meanvalue,
        },
        'metrics': {
            'Tutte Uniform': metrics_disk_uniform,
            'Tutte Mean-Value': metrics_disk_meanvalue,
        }
    }

    # ========== LOAD 3D MESHES ==========
    if HAS_TRIMESH:
        meshes_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "meshes")
        mesh_files = {
            'spot': 'spot.obj',
            'bunny': 'bunny.obj',
            'armadillo': 'armadillo.obj'
        }

        for mesh_name, filename in mesh_files.items():
            mesh_path = os.path.join(meshes_dir, filename)
            if not os.path.exists(mesh_path):
                print(f"\nSkipping {mesh_name}: {filename} not found")
                continue

            print(f"\n{len(state.meshes)+1}. Loading {mesh_name.capitalize()}...")
            try:
                mesh = trimesh.load(mesh_path)
                V_loaded = np.array(mesh.vertices, dtype=np.float64)
                F_loaded = np.array(mesh.faces, dtype=np.int64)

                # Center and scale
                center = V_loaded.mean(axis=0)
                V_loaded -= center
                scale = np.max(np.linalg.norm(V_loaded, axis=1))
                if scale > 0:
                    V_loaded /= scale

                print(f"   Loaded: {V_loaded.shape[0]} vertices, {F_loaded.shape[0]} faces")

                # Check if mesh has boundary or needs cutting
                boundary, _ = find_boundary_vertices(F_loaded, V_loaded.shape[0])

                if len(boundary) == 0:
                    # Closed mesh - need to cut it
                    print("   Mesh is closed, cutting to disk topology...")
                    V_loaded, F_loaded, boundary = cut_mesh_to_disk(V_loaded, F_loaded)
                    print(f"   After cutting: {V_loaded.shape[0]} vertices, {F_loaded.shape[0]} faces")
                else:
                    print(f"   Mesh has boundary: {len(boundary)} vertices")

                # Apply Tutte embedding
                print("   Computing Tutte embeddings...")
                uv_uniform = tutte_embedding(V_loaded, F_loaded, weight_type='uniform')
                uv_meanvalue = tutte_embedding(V_loaded, F_loaded, weight_type='mean_value')

                print("   Computing distortion metrics...")
                metrics_uniform = compute_distortion_metrics(V_loaded, F_loaded, uv_uniform)
                metrics_meanvalue = compute_distortion_metrics(V_loaded, F_loaded, uv_meanvalue)

                state.meshes[mesh_name] = {
                    'V': V_loaded,
                    'F': F_loaded,
                    'uvs': {
                        'Tutte Uniform': uv_uniform,
                        'Tutte Mean-Value': uv_meanvalue,
                    },
                    'metrics': {
                        'Tutte Uniform': metrics_uniform,
                        'Tutte Mean-Value': metrics_meanvalue,
                    }
                }

                print(f"   ✓ {mesh_name.capitalize()} loaded successfully")

            except Exception as e:
                print(f"   ✗ Error loading {mesh_name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("\n⚠️  trimesh not available - 3D meshes will not be loaded")

    # LSCM validates disk topology before solving. The existing cut routine may
    # produce invalid charts; keep their Tutte results available if LSCM rejects one.
    for name, data in state.meshes.items():
        if name == 'sphere':
            continue
        try:
            print(f"Computing LSCM for {name}...")
            uv = lscm_embedding(data['V'], data['F'])
            metrics = compute_distortion_metrics(data['V'], data['F'], uv)
            data['uvs']['LSCM'] = uv
            data['metrics']['LSCM'] = metrics
        except ValueError as error:
            print(f"LSCM unavailable for {name}: {error}")

    state.mesh_names = [name for name in state.mesh_names
                        if ('sphere' if name == 'UV Sphere' else name.lower()) in state.meshes]

    print("\n" + "="*70)
    print("VIEWER READY!")
    print("="*70)
    print(f"\nLoaded {len(state.meshes)} meshes:")
    for mesh_name in state.meshes.keys():
        print(f"  - {mesh_name.capitalize()}")
    print("\nUse the UI panel on the right to:")
    print("  - Select mesh from dropdown")
    print("  - Select parameterization method")
    print("  - Toggle between angle/area distortion")
    print("\nLeft mesh: UV checkerboard texture")
    print("Right mesh: Distortion heatmap")
    print("="*70)

    # Initialize display with first mesh
    update_display()

    # Show the window
    ps.show()


if __name__ == "__main__":
    main()
