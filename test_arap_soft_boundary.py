import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import trimesh
from src.param.arap import arap_embedding
from src.param.tutte import tutte_embedding
from src.distortion import compute_distortion_metrics

print("="*70)
print("TESTING ARAP WITH SOFT BOUNDARY CONSTRAINTS ON SPOT")
print("="*70)

# Load Spot
mesh = trimesh.load('meshes/spot.obj')
V = np.array(mesh.vertices, dtype=np.float64)
F = np.array(mesh.faces, dtype=np.int64)

# Center and scale
center = V.mean(axis=0)
V -= center
scale = np.max(np.linalg.norm(V, axis=1))
if scale > 0:
    V /= scale

print(f"\nMesh: {V.shape[0]} vertices, {F.shape[0]} faces")
print(f"Boundary: 122 vertices (previously caused singular matrix!)")

print("\nRunning ARAP with SOFT BOUNDARY CONSTRAINTS...")
print("(Using boundary_weight = 1000 × avg_edge_weight)")

try:
    uv = arap_embedding(V, F, max_iterations=10)
    
    # Check for invalid values
    nan_count = np.isnan(uv).sum()
    inf_count = np.isinf(uv).sum()
    
    if nan_count == 0 and inf_count == 0:
        print(f"\n✓✓✓ ARAP SUCCEEDED ON SPOT! ✓✓✓")
        print(f"UV range: [{uv.min(axis=0)}, {uv.max(axis=0)}]")
        
        # Compute distortion
        metrics = compute_distortion_metrics(V, F, uv)
        print(f"\nDistortion Metrics:")
        print(f"  Mean angle: {metrics['mean_angle_distortion']:.3f}")
        print(f"  Max angle: {metrics['max_angle_distortion']:.3f}")
        print(f"  Mean area: {metrics['mean_area_distortion']:.3f}")
        print(f"  Fold-overs: {metrics['num_fold_overs']}")
        
        # Compare with Tutte
        print(f"\nComparison - Tutte on Spot:")
        uv_tutte = tutte_embedding(V, F, weight_type='uniform')
        metrics_tutte = compute_distortion_metrics(V, F, uv_tutte)
        print(f"  Mean angle: {metrics_tutte['mean_angle_distortion']:.3f}")
        print(f"  Mean area: {metrics_tutte['mean_area_distortion']:.3f}")
        
        improvement = (metrics_tutte['mean_angle_distortion'] - metrics['mean_angle_distortion']) / metrics_tutte['mean_angle_distortion'] * 100
        print(f"\n→ ARAP is {improvement:.1f}% better than Tutte on angle distortion!")
    else:
        print(f"\n✗ Still getting NaN/Inf values")
        print(f"NaN: {nan_count}, Inf: {inf_count}")
    
except Exception as e:
    print(f"\n✗ ARAP FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
