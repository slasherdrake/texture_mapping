import numpy as np
import trimesh
from src.param.lscm import lscm_embedding

# Load bunny
mesh = trimesh.load('meshes/bunny.obj')
V = np.array(mesh.vertices, dtype=np.float64)
F = np.array(mesh.faces, dtype=np.int64)

# Center and scale
center = V.mean(axis=0)
V -= center
scale = np.max(np.linalg.norm(V, axis=1))
if scale > 0:
    V /= scale

print(f"Bunny: {V.shape[0]} vertices, {F.shape[0]} faces")

# Check Euler characteristic: V - E + F
# For disk topology: chi = 1
edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
unique_edges = np.unique(np.sort(edges, axis=1), axis=0)
E = len(unique_edges)
chi = V.shape[0] - E + F.shape[0]
print(f"Euler characteristic: {V.shape[0]} - {E} + {F.shape[0]} = {chi}")
print(f"Expected for disk: 1")
print(f"Expected for sphere: 2")
print(f"Expected for torus: 0")

# Try LSCM
try:
    uv = lscm_embedding(V, F)
    print("✓ LSCM SUCCESS!")
except ValueError as e:
    print(f"✗ LSCM FAILED: {e}")
