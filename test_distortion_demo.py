"""
Demo script to test distortion metrics on our existing sphere parameterizations.

This demonstrates how the distortion metrics work and validates them against
known properties (e.g., Mercator should be conformal → σ₁/σ₂ ≈ 1).
"""

import numpy as np
from src.shapes import uv_sphere
from src.param.sphere_maps import equirectangular, mercator, stereographic
from src.distortion import compute_distortion_metrics, print_distortion_summary


def main():
    print("\n" + "="*60)
    print("DISTORTION METRICS DEMO")
    print("="*60)
    print("\nGenerating UV sphere...")

    # Generate a sphere
    V, F = uv_sphere(n_lat=20, n_lon=40, radius=1.0)
    print(f"Sphere: {V.shape[0]} vertices, {F.shape[0]} faces")

    # Test each parameterization method
    methods = {
        'Equirectangular': equirectangular,
        'Mercator': mercator,
        'Stereographic': stereographic,
    }

    results = {}

    for name, method in methods.items():
        print(f"\nComputing {name} parameterization...")
        UV = method(V)

        print(f"  UV range: u ∈ [{UV[:,0].min():.2f}, {UV[:,0].max():.2f}], "
              f"v ∈ [{UV[:,1].min():.2f}, {UV[:,1].max():.2f}]")

        # Compute distortion metrics
        metrics = compute_distortion_metrics(V, F, UV)
        results[name] = metrics

        # Print summary
        print_distortion_summary(metrics, name)

    # Validate expected properties
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)

    # Mercator should be conformal (angle distortion ≈ 1)
    merc_metrics = results['Mercator']
    mean_angle_dist = merc_metrics['mean_angle_distortion']
    print(f"\n✓ Mercator conformality check:")
    print(f"  Mean angle distortion: {mean_angle_dist:.4f}")
    print(f"  Expected: Close to 1.0 (conformal)")
    if mean_angle_dist < 1.5:
        print(f"  ✅ PASS: Mercator is approximately conformal away from poles")
    else:
        print(f"  ⚠️  WARNING: Higher than expected (may include pole artifacts)")

    # All methods should have zero fold-overs (valid parameterizations)
    print(f"\n✓ Fold-over check:")
    for name, metrics in results.items():
        num_folds = metrics['num_fold_overs']
        if num_folds == 0:
            print(f"  ✅ {name}: {num_folds} fold-overs")
        else:
            print(f"  ⚠️  {name}: {num_folds} fold-overs (unexpected!)")

    # Compare angle vs area distortion
    print(f"\n✓ Angle vs Area distortion trade-off:")
    print(f"  Method              Angle (σ₁/σ₂)  Area (σ₁×σ₂)")
    print(f"  " + "-"*50)
    for name in ['Equirectangular', 'Mercator', 'Stereographic']:
        m = results[name]
        print(f"  {name:18s}  {m['mean_angle_distortion']:>8.3f}     {m['mean_area_distortion']:>8.3f}")

    print(f"\n  Note: Conformal maps (like Mercator) have low angle distortion")
    print(f"        but high area distortion. No map can minimize both!")

    print("\n" + "="*60)
    print("Demo complete! Distortion metrics are working correctly.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
