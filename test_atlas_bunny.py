"""Test automatic texture atlas generation on Bunny mesh."""
import numpy as np
import trimesh
from src.param.atlas import generate_texture_atlas

print("Loading Bunny mesh...")
mesh = trimesh.load('meshes/bunny.obj')
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

# Print summary
print("\n" + "="*70)
print("BUNNY RESULTS SUMMARY")
print("="*70)
print(f"Successful charts: {len(atlas['charts'])}")
print(f"Failed charts: {len(atlas['failed_charts'])}")

# Sample a few successful charts
print("\nSample successful charts:")
for i, chart in enumerate(atlas['charts'][:5]):
    chart_id = chart['id']
    if chart_id in atlas['metrics']:
        m = atlas['metrics'][chart_id]
        print(f"  Chart {chart_id}: {chart['n_faces']} faces, "
              f"angle distortion={m['mean_angle_distortion']:.3f}, "
              f"fold-overs={m['num_fold_overs']}")

print("\nBunny atlas generation complete!")
