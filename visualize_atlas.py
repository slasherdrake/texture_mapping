"""
Visualize complete LSCM atlas - shows segmented mesh and parameterized charts.
"""
import numpy as np
import trimesh
import polyscope as ps
import polyscope.imgui as psim
from src.param.atlas import generate_texture_atlas, extract_chart_mesh, split_disconnected_chart

# Configuration
MESH_FILE = 'meshes/spot.obj'  # Change to test different meshes
MAX_CHARTS_TO_SHOW = 10  # Only show first N successful charts (for performance)

class AtlasViewer:
    def __init__(self):
        self.current_chart_idx = 0
        self.chart_meshes = []
        self.chart_uvs = []
        self.show_3d = True
        self.current_chart_name = None

viewer = AtlasViewer()

print("="*70)
print("LSCM TEXTURE ATLAS VISUALIZATION")
print("="*70)

# Load mesh
print(f"\nLoading {MESH_FILE}...")
mesh = trimesh.load(MESH_FILE)
V = np.array(mesh.vertices)
F = np.array(mesh.faces)
print(f"Mesh: {V.shape[0]} vertices, {F.shape[0]} faces")

# Generate atlas
print("\nGenerating texture atlas...")
atlas = generate_texture_atlas(
    V, F,
    percentile=95,
    max_string_length=5,
    min_feature_length=15,
    merge_threshold=0.25,
    max_area_distortion=2.0,
    max_subdivisions=0  # Disable subdivision for visualization
)

successful_charts = atlas['charts'][:MAX_CHARTS_TO_SHOW]
print(f"\n✓ Showing first {len(successful_charts)} successful charts")

# Assign colors to show chart boundaries
print("\nPreparing visualization...")
face_colors = np.zeros((F.shape[0], 3))
np.random.seed(42)

for chart in atlas['charts']:
    color = np.random.rand(3)
    for face_idx in chart['faces']:
        face_colors[face_idx] = color

# Extract chart meshes and UVs for display
for chart in successful_charts:
    chart_id = chart['id']
    if chart_id in atlas['uv_coords']:
        V_chart, F_chart, _ = extract_chart_mesh(V, F, chart)
        uv = atlas['uv_coords'][chart_id]['uv']

        viewer.chart_meshes.append((V_chart, F_chart))
        viewer.chart_uvs.append(uv)

# Initialize Polyscope
ps.init()
ps.set_ground_plane_mode("none")
ps.set_up_dir("z_up")

# Register the full mesh with chart colors
full_mesh = ps.register_surface_mesh("Original Mesh (Segmented)", V, F, smooth_shade=False)
full_mesh.add_color_quantity("Charts", face_colors, defined_on='faces', enabled=True)

def update_chart_display():
    """Update the displayed chart based on current selection."""
    # Remove old chart if exists
    if viewer.current_chart_name is not None:
        try:
            ps.remove_surface_mesh(viewer.current_chart_name)
        except:
            pass  # Mesh doesn't exist, that's ok

    if viewer.current_chart_idx >= len(viewer.chart_meshes):
        return

    V_chart, F_chart = viewer.chart_meshes[viewer.current_chart_idx]
    uv = viewer.chart_uvs[viewer.current_chart_idx]

    if viewer.show_3d:
        # Show 3D chart
        V_display = V_chart.copy()
        V_display[:, 0] += np.max(V[:, 0]) - np.min(V[:, 0]) + 1  # Offset to the right

        chart_name = f"Chart_{viewer.current_chart_idx}_3D"
        viewer.current_chart_name = chart_name

        chart_mesh = ps.register_surface_mesh(chart_name, V_display, F_chart)
        chart_mesh.add_parameterization_quantity("UV", uv, defined_on='vertices', enabled=True)
    else:
        # Show flattened UV chart
        V_flat = np.column_stack([uv, np.zeros(len(uv))])  # UV as XY, Z=0
        V_flat[:, 0] += np.max(V[:, 0]) - np.min(V[:, 0]) + 1  # Offset

        chart_name = f"Chart_{viewer.current_chart_idx}_UV"
        viewer.current_chart_name = chart_name

        chart_mesh = ps.register_surface_mesh(chart_name, V_flat, F_chart)

        # Add checkerboard texture
        chart_mesh.add_parameterization_quantity("Checkerboard", uv, defined_on='vertices', enabled=True)

def ui_callback():
    """ImGui UI for chart selection."""
    psim.TextUnformatted("=== LSCM Atlas Viewer ===")
    psim.Separator()

    psim.TextUnformatted(f"Total successful charts: {len(atlas['charts'])}")
    psim.TextUnformatted(f"Showing: {len(viewer.chart_meshes)} charts")
    psim.Separator()

    # Chart selector
    changed, new_idx = psim.SliderInt("Chart", viewer.current_chart_idx,
                                      v_min=0, v_max=len(viewer.chart_meshes)-1)
    if changed:
        viewer.current_chart_idx = new_idx
        update_chart_display()

    psim.Separator()

    # 3D vs UV toggle
    changed, new_show = psim.Checkbox("Show 3D", viewer.show_3d)
    if changed:
        viewer.show_3d = new_show
        update_chart_display()

    psim.Separator()
    psim.TextUnformatted("=== Current Chart Info ===")
    if viewer.current_chart_idx < len(viewer.chart_meshes):
        chart = successful_charts[viewer.current_chart_idx]
        chart_id = chart['id']

        psim.TextUnformatted(f"Chart ID: {chart_id}")
        psim.TextUnformatted(f"Faces: {chart['n_faces']}")
        psim.TextUnformatted(f"Vertices: {chart['n_vertices']}")

        if chart_id in atlas['metrics']:
            m = atlas['metrics'][chart_id]
            psim.TextUnformatted(f"Angle distortion: {m['mean_angle_distortion']:.3f}")
            psim.TextUnformatted(f"Area distortion: {m['mean_area_distortion']:.3f}")
            psim.TextUnformatted(f"Fold-overs: {m['num_fold_overs']}")

    psim.Separator()
    psim.TextUnformatted("Left: Original mesh (colored by chart)")
    psim.TextUnformatted("Right: Selected chart with UV")

ps.set_user_callback(ui_callback)

# Show first chart
update_chart_display()

print("\n" + "="*70)
print("VISUALIZATION READY!")
print("="*70)
print("\nUI Controls:")
print("  - Use slider to select different charts")
print("  - Toggle '3D' checkbox to see UV flattening")
print("  - Close window when done")
print("="*70)

ps.show()

print("\nVisualization closed.")
