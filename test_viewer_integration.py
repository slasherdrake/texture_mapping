"""
Test that Tutte embedding integrates correctly with the viewer code.
This doesn't actually open the GUI, just verifies the code runs without errors.
"""

from src.shapes import uv_sphere, disk, hemisphere
from src.param.sphere_maps import equirectangular, mercator, stereographic
from src.param.tutte import tutte_embedding
import numpy as np


def main():
    print("\n" + "="*70)
    print("TESTING VIEWER INTEGRATION")
    print("="*70)

    # Test 1: Sphere parameterizations
    print("\n1. Testing sphere parameterizations...")
    V_sphere, F_sphere = uv_sphere(n_lat=20, n_lon=40, radius=1.0)
    uv_equirect = equirectangular(V_sphere)
    uv_mercator = mercator(V_sphere)
    uv_stereo = stereographic(V_sphere)
    print(f"   Sphere: {V_sphere.shape[0]} vertices, {F_sphere.shape[0]} faces")
    print(f"   Equirectangular UV: {uv_equirect.shape}")
    print(f"   Mercator UV: {uv_mercator.shape}")
    print(f"   Stereographic UV: {uv_stereo.shape}")
    print("   ✓ Sphere parameterizations OK")

    # Test 2: Hemisphere with Tutte
    print("\n2. Testing hemisphere with Tutte embedding...")
    V_hemi, F_hemi = hemisphere(n_lat=10, n_lon=20, radius=1.0)
    uv_tutte_uniform = tutte_embedding(V_hemi, F_hemi, weight_type='uniform')
    uv_tutte_meanvalue = tutte_embedding(V_hemi, F_hemi, weight_type='mean_value')
    print(f"   Hemisphere: {V_hemi.shape[0]} vertices, {F_hemi.shape[0]} faces")
    print(f"   Tutte (Uniform) UV: {uv_tutte_uniform.shape}")
    print(f"   Tutte (Mean-Value) UV: {uv_tutte_meanvalue.shape}")
    print("   ✓ Hemisphere Tutte embeddings OK")

    # Test 3: Disk with Tutte
    print("\n3. Testing disk with Tutte embedding...")
    V_disk, F_disk = disk(n_radial=10, n_angular=20, radius=1.0)
    uv_disk_uniform = tutte_embedding(V_disk, F_disk, weight_type='uniform')
    uv_disk_meanvalue = tutte_embedding(V_disk, F_disk, weight_type='mean_value')
    print(f"   Disk: {V_disk.shape[0]} vertices, {F_disk.shape[0]} faces")
    print(f"   Tutte (Uniform) UV: {uv_disk_uniform.shape}")
    print(f"   Tutte (Mean-Value) UV: {uv_disk_meanvalue.shape}")
    print("   ✓ Disk Tutte embeddings OK")

    print("\n" + "="*70)
    print("ALL INTEGRATION TESTS PASSED")
    print("="*70)
    print("\nThe viewer.py should work correctly with all these meshes.")
    print("To launch the interactive viewer, run:")
    print("  python -m src.viewer")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
