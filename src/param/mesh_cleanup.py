"""
Mesh cleanup utilities for preparing meshes for LSCM.

LSCM has strict requirements:
- Single connected component
- Single boundary loop
- No unused vertices
- Consistent winding
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def remove_unreferenced_vertices(V, F):
    """
    Remove vertices that aren't referenced by any face.

    Parameters
    ----------
    V : ndarray, shape (N, 3)
        Vertex positions
    F : ndarray, shape (M, 3)
        Face indices

    Returns
    -------
    V_clean : ndarray
        Vertex positions with unreferenced vertices removed
    F_clean : ndarray
        Face indices remapped to new vertex indices
    """
    # Find which vertices are actually used
    used_vertices = np.unique(F.ravel())

    # Create mapping from old to new indices
    vertex_map = np.full(V.shape[0], -1, dtype=np.int64)
    vertex_map[used_vertices] = np.arange(len(used_vertices))

    # Keep only used vertices
    V_clean = V[used_vertices]

    # Remap face indices
    F_clean = vertex_map[F]

    return V_clean, F_clean


def get_largest_component(V, F):
    """
    Extract the largest connected component from a mesh.

    Parameters
    ----------
    V : ndarray, shape (N, 3)
        Vertex positions
    F : ndarray, shape (M, 3)
        Face indices

    Returns
    -------
    V_component : ndarray
        Vertices of largest component
    F_component : ndarray
        Faces of largest component
    """
    # Build adjacency graph
    edges = np.concatenate([
        F[:, [0, 1]],
        F[:, [1, 2]],
        F[:, [2, 0]]
    ])

    # Create undirected graph
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(V.shape[0], V.shape[0])
    ).tocsr()

    # Find connected components
    n_components, labels = connected_components(graph, directed=False, return_labels=True)

    if n_components == 1:
        # Already connected
        return V, F

    print(f"  Found {n_components} connected components")

    # Find largest component
    component_sizes = np.bincount(labels)
    largest_component = np.argmax(component_sizes)

    print(f"  Keeping largest component: {component_sizes[largest_component]} vertices")

    # Keep only vertices in largest component
    vertex_mask = (labels == largest_component)
    vertices_to_keep = np.where(vertex_mask)[0]

    # Create vertex mapping
    vertex_map = np.full(V.shape[0], -1, dtype=np.int64)
    vertex_map[vertices_to_keep] = np.arange(len(vertices_to_keep))

    # Keep only faces where all vertices are in the largest component
    face_mask = vertex_mask[F].all(axis=1)
    F_component = vertex_map[F[face_mask]]

    V_component = V[vertices_to_keep]

    print(f"  Result: {V_component.shape[0]} vertices, {F_component.shape[0]} faces")

    return V_component, F_component


def clean_mesh_for_lscm(V, F):
    """
    Clean a mesh to meet LSCM requirements.

    Steps:
    1. Remove unreferenced vertices
    2. Extract largest connected component
    3. Remove unreferenced vertices again (after component extraction)

    Parameters
    ----------
    V : ndarray, shape (N, 3)
        Vertex positions
    F : ndarray, shape (M, 3)
        Face indices

    Returns
    -------
    V_clean : ndarray
        Cleaned vertex positions
    F_clean : ndarray
        Cleaned face indices

    Notes
    -----
    This does NOT guarantee the mesh will pass LSCM validation
    (it may still have multiple boundary loops or non-disk topology),
    but it fixes the most common issues.
    """
    print(f"  Cleaning mesh: {V.shape[0]} vertices, {F.shape[0]} faces")

    # Step 1: Remove unreferenced vertices
    V_clean, F_clean = remove_unreferenced_vertices(V, F)
    if V_clean.shape[0] != V.shape[0]:
        print(f"  Removed {V.shape[0] - V_clean.shape[0]} unreferenced vertices")

    # Step 2: Get largest connected component
    V_clean, F_clean = get_largest_component(V_clean, F_clean)

    # Step 3: Remove unreferenced vertices again (some may have been isolated)
    V_clean, F_clean = remove_unreferenced_vertices(V_clean, F_clean)

    print(f"  Final: {V_clean.shape[0]} vertices, {F_clean.shape[0]} faces")

    return V_clean, F_clean
