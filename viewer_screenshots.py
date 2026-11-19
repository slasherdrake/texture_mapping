"""Generate screenshots from viewer without interactive window."""
import numpy as np
import polyscope as ps
from src.shapes import sphere, hemisphere, disk
from src.param.tutte import tutte_embedding
from src.param.lscm import lscm_embedding
from src.distortion import compute_distortion_metrics
import trimesh

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


print("Loading Disk mesh...")
V, F = disk(resolution=15)

print("Computing Tutte embedding...")
uv = tutte_embedding(V, F, weight_type='mean_value')

print("Computing distortion...")
metrics = compute_distortion_metrics(V, F, uv)

print("Setting up visualization...")
ps.init()
ps.set_ground_plane_mode("none")
ps.set_automatically_compute_scene_extents(False)
ps.set_length_scale(2.0)

# Get vertex distortion
angle_dist, area_dist = compute_vertex_distortion(metrics, F, V.shape[0])
max_val = np.nanpercentile(angle_dist, 95)
distortion_values = np.clip(angle_dist, 1.0, max_val)

# Left mesh: textured
V_left = V.copy()
V_left[:, 0] -= 1.5
mesh_left = ps.register_surface_mesh("Disk (Textured)", V_left, F, smooth_shade=True)
mesh_left.add_parameterization_quantity("UV", uv, defined_on='vertices', enabled=True)

# Right mesh: heatmap
V_right = V.copy()
V_right[:, 0] += 1.5
mesh_right = ps.register_surface_mesh("Disk (Heatmap)", V_right, F, smooth_shade=True)
mesh_right.add_scalar_quantity(
    "Angle Distortion",
    distortion_values,
    defined_on='vertices',
    enabled=True,
    cmap='turbo',
    vminmax=(1.0, max_val)
)

print(f"\nMean angle distortion: {metrics['mean_angle_distortion']:.3f}")
print(f"Max angle distortion: {metrics['max_angle_distortion']:.3f}")
print(f"Fold-overs: {metrics['num_fold_overs']}")

print("\nSaving screenshot...")
ps.screenshot("disk_parameterization.png", transparent_bg=False)

print("\nDone! Screenshot saved to disk_parameterization.png")
print("Note: If you want the interactive viewer, the window should appear")
print("when you run: python -m src.viewer_clean")
print("\nOpening window for 5 seconds...")

# Show window briefly
ps.show(5)
