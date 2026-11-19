"""Test just the feature detection (not full atlas)."""
import numpy as np
from src.shapes import disk
from src.param.segmentation import MeshSegmenter

print("Generating Disk mesh...")
V, F = disk(n_radial=5, n_angular=10)
print(f"Mesh: {V.shape[0]} vertices, {F.shape[0]} faces")

print("\nBuilding segmenter...")
segmenter = MeshSegmenter(V, F)

print("\nDetecting features...")
features = segmenter.detect_features(
    percentile=90,
    max_string_length=3,
    min_feature_length=5
)

print(f"\nFeature detection complete!")
print(f"Found {len(features)} feature curves")

for i, curve in enumerate(features):
    print(f"  Curve {i+1}: {len(curve)} vertices")

print("\nNow testing chart growing...")
charts = segmenter.segment_into_charts(merge_threshold=0.25)

print(f"\nChart segmentation complete!")
print(f"Created {len(charts)} charts")

for i, chart in enumerate(charts):
    print(f"  Chart {i+1}: {chart['n_faces']} faces, {chart['n_vertices']} vertices")

print("\nTest complete!")
