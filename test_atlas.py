"""Test automatic texture atlas generation on Spot mesh."""
import numpy as np
import trimesh
from src.param.atlas import generate_texture_atlas

print("Loading Spot mesh...")
mesh = trimesh.load('meshes/spot.obj')
V = np.array(mesh.vertices)
F = np.array(mesh.faces)

print(f"Mesh: {V.shape[0]} vertices, {F.shape[0]} faces")

# Generate texture atlas
atlas = generate_texture_atlas(
    V, F,
    percentile=95,  # Keep top 5% sharpest edges as features
    max_string_length=5,
    min_feature_length=15,
    merge_threshold=0.25,
    max_area_distortion=2.0
)

# Print results
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

for i, chart in enumerate(atlas['charts']):
    chart_id = chart['id']
    print(f"\nChart {i+1} (ID={chart_id}):")
    print(f"  Faces: {chart['n_faces']}")
    print(f"  Vertices: {chart['n_vertices']}")

    if chart_id in atlas['uv_coords']:
        uv_data = atlas['uv_coords'][chart_id]
        print(f"  UV range: [{np.min(uv_data['uv'], axis=0)}, {np.max(uv_data['uv'], axis=0)}]")

    if chart_id in atlas['metrics']:
        m = atlas['metrics'][chart_id]
        print(f"  Mean angle distortion: {m['mean_angle_distortion']:.3f}")
        print(f"  Mean area distortion: {m['mean_area_distortion']:.3f}")
        print(f"  Fold-overs: {m['num_fold_overs']}")

if atlas['failed_charts']:
    print(f"\n{len(atlas['failed_charts'])} charts failed to parameterize")

print("\nAtlas generation test complete!")
