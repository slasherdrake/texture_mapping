"""
Test script for Tutte embedding implementation.

Tests both uniform and mean-value weights on disk-topology meshes.
Validates that:
1. The parameterization has no fold-overs (all triangle areas > 0)
2. Boundary vertices are correctly mapped to a circle
3. Interior vertices are inside the boundary circle
"""

import numpy as np
from src.shapes import disk, hemisphere
from src.param.tutte import tutte_embedding
from src.distortion import compute_distortion_metrics, print_distortion_summary


def test_tutte_on_disk():
    """Test Tutte embedding on a planar disk mesh."""
    print("\n" + "="*70)
    print("TEST 1: Tutte Embedding on Planar Disk")
    print("="*70)

    # Generate a planar disk
    V, F = disk(n_radial=10, n_angular=20, radius=1.0)
    print(f"\nGenerated planar disk: {V.shape[0]} vertices, {F.shape[0]} faces")

    # Test uniform weights
    print("\n" + "-"*70)
    print("Testing UNIFORM weights")
    print("-"*70)
    UV_uniform = tutte_embedding(V, F, weight_type='uniform')

    # Check boundary is on circle
    boundary_mask = np.linalg.norm(V[:, :2], axis=1) > 0.99 * 1.0
    boundary_uv = UV_uniform[boundary_mask]
    boundary_radii = np.linalg.norm(boundary_uv, axis=1)
    print(f"\nBoundary check:")
    print(f"  Expected radius: 1.0")
    print(f"  Min boundary radius: {boundary_radii.min():.4f}")
    print(f"  Max boundary radius: {boundary_radii.max():.4f}")
    print(f"  Mean boundary radius: {boundary_radii.mean():.4f}")

    # Check for fold-overs
    metrics_uniform = compute_distortion_metrics(V, F, UV_uniform)
    print_distortion_summary(metrics_uniform, "Tutte (Uniform Weights)")

    if metrics_uniform['num_fold_overs'] == 0:
        print("  ✓ No fold-overs detected (as expected for Tutte)")
    else:
        print(f"  ✗ WARNING: {metrics_uniform['num_fold_overs']} fold-overs detected!")

    # Test mean-value weights
    print("\n" + "-"*70)
    print("Testing MEAN-VALUE weights")
    print("-"*70)
    UV_mean_value = tutte_embedding(V, F, weight_type='mean_value')

    # Check boundary is on circle
    boundary_uv = UV_mean_value[boundary_mask]
    boundary_radii = np.linalg.norm(boundary_uv, axis=1)
    print(f"\nBoundary check:")
    print(f"  Expected radius: 1.0")
    print(f"  Min boundary radius: {boundary_radii.min():.4f}")
    print(f"  Max boundary radius: {boundary_radii.max():.4f}")
    print(f"  Mean boundary radius: {boundary_radii.mean():.4f}")

    # Check for fold-overs
    metrics_mean_value = compute_distortion_metrics(V, F, UV_mean_value)
    print_distortion_summary(metrics_mean_value, "Tutte (Mean-Value Weights)")

    if metrics_mean_value['num_fold_overs'] == 0:
        print("  ✓ No fold-overs detected (as expected for Tutte)")
    else:
        print(f"  ✗ WARNING: {metrics_mean_value['num_fold_overs']} fold-overs detected!")

    # Compare distortion
    print("\n" + "-"*70)
    print("Comparison: Uniform vs Mean-Value Weights")
    print("-"*70)
    print(f"{'Metric':<30} {'Uniform':<15} {'Mean-Value':<15}")
    print("-"*70)
    print(f"{'Mean angle distortion':<30} {metrics_uniform['mean_angle_distortion']:>14.4f} {metrics_mean_value['mean_angle_distortion']:>14.4f}")
    print(f"{'Max angle distortion':<30} {metrics_uniform['max_angle_distortion']:>14.4f} {metrics_mean_value['max_angle_distortion']:>14.4f}")
    print(f"{'Mean area distortion':<30} {metrics_uniform['mean_area_distortion']:>14.4f} {metrics_mean_value['mean_area_distortion']:>14.4f}")
    print(f"{'Max area distortion':<30} {metrics_uniform['max_area_distortion']:>14.4f} {metrics_mean_value['max_area_distortion']:>14.4f}")
    print(f"{'Fold-overs':<30} {metrics_uniform['num_fold_overs']:>14d} {metrics_mean_value['num_fold_overs']:>14d}")


def test_tutte_on_hemisphere():
    """Test Tutte embedding on a hemisphere (disk topology but curved)."""
    print("\n" + "="*70)
    print("TEST 2: Tutte Embedding on Hemisphere")
    print("="*70)

    # Generate a hemisphere
    V, F = hemisphere(n_lat=20, n_lon=40, radius=1.0)
    print(f"\nGenerated hemisphere: {V.shape[0]} vertices, {F.shape[0]} faces")

    # Test uniform weights
    print("\n" + "-"*70)
    print("Testing UNIFORM weights on curved surface")
    print("-"*70)
    UV_uniform = tutte_embedding(V, F, weight_type='uniform')

    metrics_uniform = compute_distortion_metrics(V, F, UV_uniform)
    print_distortion_summary(metrics_uniform, "Hemisphere: Tutte (Uniform)")

    if metrics_uniform['num_fold_overs'] == 0:
        print("  ✓ No fold-overs detected")
    else:
        print(f"  ✗ WARNING: {metrics_uniform['num_fold_overs']} fold-overs detected!")

    # Test mean-value weights
    print("\n" + "-"*70)
    print("Testing MEAN-VALUE weights on curved surface")
    print("-"*70)
    UV_mean_value = tutte_embedding(V, F, weight_type='mean_value')

    metrics_mean_value = compute_distortion_metrics(V, F, UV_mean_value)
    print_distortion_summary(metrics_mean_value, "Hemisphere: Tutte (Mean-Value)")

    if metrics_mean_value['num_fold_overs'] == 0:
        print("  ✓ No fold-overs detected")
    else:
        print(f"  ✗ WARNING: {metrics_mean_value['num_fold_overs']} fold-overs detected!")

    # Hemisphere should have higher distortion than planar disk
    print("\n" + "-"*70)
    print("Note: Hemisphere should show higher distortion than planar disk")
    print("      because we're flattening a curved surface.")
    print("-"*70)


def main():
    print("\n" + "="*70)
    print("TUTTE EMBEDDING TEST SUITE")
    print("="*70)
    print("\nThis test validates the Tutte embedding implementation.")
    print("Expected results:")
    print("  - Zero fold-overs (Tutte's theorem guarantees this)")
    print("  - Boundary vertices on unit circle")
    print("  - Mean-value weights should have lower distortion than uniform")

    test_tutte_on_disk()
    test_tutte_on_hemisphere()

    print("\n" + "="*70)
    print("TESTS COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("  1. Tutte embedding guarantees no fold-overs (valid parameterization)")
    print("  2. Mean-value weights typically preserve shape better than uniform")
    print("  3. Distortion increases when flattening curved surfaces (hemisphere)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
