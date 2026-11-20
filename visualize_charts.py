"""
Visualize automatic mesh segmentation - shows each chart in a different color.
"""
import numpy as np
import trimesh
import polyscope as ps
from src.param.segmentation import segment_mesh_for_lscm

# Configuration
MESH_FILE = 'meshes/spot.obj'  # Change this to test different meshes
PERCENTILE = 95  # Keep top 5% sharpest edges

print("="*70)
print("AUTOMATIC MESH SEGMENTATION VISUALIZATION")
print("="*70)

# Load mesh
print(f"\nLoading {MESH_FILE}...")
mesh = trimesh.load(MESH_FILE)
V = np.array(mesh.vertices)
F = np.array(mesh.faces)
print(f"Mesh: {V.shape[0]} vertices, {F.shape[0]} faces")

# Run segmentation
print("\nRunning automatic segmentation...")
charts = segment_mesh_for_lscm(
    V, F,
    percentile=PERCENTILE,
    max_string_length=5,
    min_feature_length=15,
    merge_threshold=0.25
)

print(f"\n✓ Created {len(charts)} charts")

# Assign a color to each chart
print("\nAssigning colors to charts...")
face_colors = np.zeros((F.shape[0], 3))

# Generate distinct colors for each chart
np.random.seed(42)  # For reproducible colors
for i, chart in enumerate(charts):
    # Generate a random color for this chart
    color = np.random.rand(3)

    # Assign this color to all faces in the chart
    for face_idx in chart['faces']:
        face_colors[face_idx] = color

# Initialize Polyscope
print("\nInitializing visualization...")
ps.init()
ps.set_ground_plane_mode("none")
ps.set_up_dir("z_up")

# Register the mesh with chart colors
mesh_ps = ps.register_surface_mesh("Segmented Mesh", V, F, smooth_shade=False)
mesh_ps.add_color_quantity("Chart IDs", face_colors, defined_on='faces', enabled=True)

# Print statistics
print("\n" + "="*70)
print("VISUALIZATION READY!")
print("="*70)
print(f"\nChart statistics:")
print(f"  Total charts: {len(charts)}")
print(f"  Faces per chart (avg): {np.mean([c['n_faces'] for c in charts]):.1f}")
print(f"  Faces per chart (min): {min([c['n_faces'] for c in charts])}")
print(f"  Faces per chart (max): {max([c['n_faces'] for c in charts])}")
print(f"  Vertices per chart (avg): {np.mean([c['n_vertices'] for c in charts]):.1f}")

print("\nChart size distribution:")
small_charts = sum(1 for c in charts if c['n_faces'] < 10)
medium_charts = sum(1 for c in charts if 10 <= c['n_faces'] < 100)
large_charts = sum(1 for c in charts if c['n_faces'] >= 100)
print(f"  Small (<10 faces): {small_charts}")
print(f"  Medium (10-99 faces): {medium_charts}")
print(f"  Large (>=100 faces): {large_charts}")

print("\n" + "="*70)
print("Each chart is shown in a different color.")
print("Close the window when done viewing.")
print("="*70)

# Show the visualization
ps.show()

print("\nVisualization closed.")
