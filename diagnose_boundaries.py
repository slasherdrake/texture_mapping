"""
Diagnose boundary topology for meshes failing LSCM.
"""

import numpy as np
import trimesh
from src.param.mesh_cleanup import clean_mesh_for_lscm


def analyze_boundary(V, F):
    """Analyze boundary structure of a mesh."""
    # Find boundary edges (appear in only one face)
    edges = np.concatenate([
        F[:, [0, 1]],
        F[:, [1, 2]],
        F[:, [2, 0]]
    ])

    # Sort edges to make them undirected
    edges_sorted = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)

    boundary_edges = unique_edges[counts == 1]

    if len(boundary_edges) == 0:
        print("  No boundary (closed mesh)")
        return

    # Find boundary vertices
    boundary_verts = np.unique(boundary_edges)
    print(f"  Boundary: {len(boundary_verts)} vertices, {len(boundary_edges)} edges")

    # Count degree of each boundary vertex
    boundary_degree = np.bincount(boundary_edges.ravel(), minlength=V.shape[0])
    boundary_degree = boundary_degree[boundary_verts]

    print(f"  Vertex degrees: min={boundary_degree.min()}, max={boundary_degree.max()}")

    # Count vertices with degree != 2 (indicates multiple loops or branches)
    non_simple = np.sum(boundary_degree != 2)
    if non_simple > 0:
        print(f"  ⚠ {non_simple} boundary vertices have degree != 2 (multiple loops/branches)")
        # Show degree distribution
        unique_degrees, degree_counts = np.unique(boundary_degree, return_counts=True)
        for deg, count in zip(unique_degrees, degree_counts):
            print(f"    Degree {deg}: {count} vertices")
    else:
        print(f"  ✓ All boundary vertices have degree 2 (simple loop)")


def main():
    print("\n" + "="*70)
    print("BOUNDARY TOPOLOGY DIAGNOSIS")
    print("="*70)

    meshes = {
        'spot': 'meshes/spot.obj',
        'bunny': 'meshes/bunny.obj',
        'armadillo': 'meshes/armadillo.obj'
    }

    for name, path in meshes.items():
        print(f"\n{name.upper()}:")
        mesh = trimesh.load(path)
        V = np.array(mesh.vertices, dtype=np.float64)
        F = np.array(mesh.faces, dtype=np.int64)

        # Center and scale
        center = V.mean(axis=0)
        V -= center
        scale = np.max(np.linalg.norm(V, axis=1))
        if scale > 0:
            V /= scale

        print(f"Original mesh: {V.shape[0]} vertices, {F.shape[0]} faces")
        analyze_boundary(V, F)

        # After cleaning
        print(f"\nAfter cleanup:")
        V_clean, F_clean = clean_mesh_for_lscm(V, F)
        analyze_boundary(V_clean, F_clean)


if __name__ == "__main__":
    main()
