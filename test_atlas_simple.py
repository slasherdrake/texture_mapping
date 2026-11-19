"""Test automatic texture atlas generation on simple mesh."""
import numpy as np
from src.shapes import disk
from src.param.atlas import generate_texture_atlas

print("Generating Disk mesh...")
V, F = disk(n_radial=5, n_angular=10)  # Small for testing
print(f"Mesh: {V.shape[0]} vertices, {F.shape[0]} faces")

# Generate texture atlas
print("\nGenerating atlas...")
atlas = generate_texture_atlas(
    V, F,
    percentile=90,  # Less strict threshold
    max_string_length=3,
    min_feature_length=5,
    merge_threshold=0.25,
    max_area_distortion=2.0
)

# Print results
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

print(f"Charts created: {len(atlas['charts'])}")
print(f"Charts failed: {len(atlas['failed_charts'])}")

for i, chart in enumerate(atlas['charts']):
    chart_id = chart['id']
    print(f"\nChart {i+1} (ID={chart_id}):")
    print(f"  Faces: {chart['n_faces']}")
    print(f"  Vertices: {chart['n_vertices']}")

    if chart_id in atlas['metrics']:
        m = atlas['metrics'][chart_id]
        print(f"  Mean angle distortion: {m['mean_angle_distortion']:.3f}")
        print(f"  Fold-overs: {m['num_fold_overs']}")

print("\nTest complete!")
