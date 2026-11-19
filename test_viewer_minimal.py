"""Minimal viewer test with just the Disk mesh."""
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from src.shapes import disk
from src.param.tutte import tutte_embedding
from src.distortion import compute_distortion_metrics

print("Generating Disk mesh...")
V, F = disk(resolution=15)
print(f"  {V.shape[0]} vertices, {F.shape[0]} faces")

print("Computing Tutte embedding...")
uv = tutte_embedding(V, F, weight_type='uniform')
print("  Done")

print("Computing distortion metrics...")
metrics = compute_distortion_metrics(V, F, uv)
print(f"  Mean angle distortion: {metrics['mean_angle_distortion']:.3f}")

print("\nInitializing Polyscope...")
ps.init()
ps.set_ground_plane_mode("none")

print("Registering mesh...")
mesh = ps.register_surface_mesh("Disk", V, F, smooth_shade=True)
mesh.add_parameterization_quantity("UV", uv, defined_on='vertices', enabled=True)

print("\n" + "="*70)
print("Opening window...")
print("If window doesn't appear:")
print("  - Check if it's behind other windows (Cmd+Tab)")
print("  - Try clicking on Python in the Dock")
print("  - Check Activity Monitor for high CPU")
print("="*70)

ps.show()

print("Window closed.")
