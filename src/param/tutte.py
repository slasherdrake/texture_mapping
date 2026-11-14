"""
Tutte embedding for disk-topology meshes.

Tutte's embedding theorem (1963) states that if you fix the boundary vertices
of a disk-topology mesh to a convex polygon and place interior vertices at the
weighted average of their neighbors, the resulting parameterization has no
fold-overs (all triangles have positive area).

This implementation supports:
- Uniform weights (all neighbor weights equal)
- Mean-value weights (weights depend on angles in the mesh)

References:
-----------
Tutte, W. T. "How to draw a graph." Proceedings of the London Mathematical
    Society 3.1 (1963): 743-767.
Floater, M. S. "Mean value coordinates." Computer aided geometric design 20.1
    (2003): 19-27.
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def find_boundary_vertices(F, n_vertices):
    """
    Find boundary vertices of a mesh by identifying edges that appear in only
    one triangle.

    Parameters
    ----------
    F : ndarray, shape (M, 3)
        Triangle face indices
    n_vertices : int
        Total number of vertices in the mesh

    Returns
    -------
    boundary : ndarray, shape (B,)
        Indices of boundary vertices in order (counterclockwise)
    is_boundary : ndarray, shape (N,), dtype bool
        Boolean array indicating which vertices are on the boundary

    Notes
    -----
    This assumes the mesh has disk topology (genus 0 with boundary).
    For closed meshes, this will return an empty array.
    """
    # Build edge-to-face map
    # Each edge is represented as a sorted tuple (min_vertex, max_vertex)
    edge_count = {}

    for face in F:
        # Three edges per triangle
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]]))
        ]

        for edge in edges:
            edge_count[edge] = edge_count.get(edge, 0) + 1

    # Boundary edges appear exactly once
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    if len(boundary_edges) == 0:
        # No boundary (closed mesh)
        return np.array([], dtype=np.int64), np.zeros(n_vertices, dtype=bool)

    # Build ordered boundary loop
    # Start with first edge
    boundary = list(boundary_edges[0])
    used_edges = {boundary_edges[0]}

    # Walk around the boundary
    while len(used_edges) < len(boundary_edges):
        current = boundary[-1]

        # Find next edge that shares current vertex
        found = False
        for edge in boundary_edges:
            if edge in used_edges:
                continue

            if edge[0] == current:
                boundary.append(edge[1])
                used_edges.add(edge)
                found = True
                break
            elif edge[1] == current:
                boundary.append(edge[0])
                used_edges.add(edge)
                found = True
                break

        if not found:
            # Shouldn't happen for a well-formed boundary
            break

    # Remove duplicate (last vertex = first vertex)
    if len(boundary) > 0 and boundary[-1] == boundary[0]:
        boundary = boundary[:-1]

    boundary = np.array(boundary, dtype=np.int64)

    # Create boolean mask
    is_boundary = np.zeros(n_vertices, dtype=bool)
    is_boundary[boundary] = True

    # Verify consistency
    if len(boundary) != is_boundary.sum():
        print(f"Warning: boundary has {len(boundary)} vertices but is_boundary has {is_boundary.sum()} True values")
        print(f"This indicates duplicate vertices in boundary list")
        # Remove duplicates
        boundary = np.unique(boundary)
        is_boundary = np.zeros(n_vertices, dtype=bool)
        is_boundary[boundary] = True
        print(f"After deduplication: {len(boundary)} boundary vertices")

    return boundary, is_boundary


def build_adjacency(F, n_vertices):
    """
    Build vertex adjacency list from triangle faces.

    Parameters
    ----------
    F : ndarray, shape (M, 3)
        Triangle face indices
    n_vertices : int
        Total number of vertices

    Returns
    -------
    adjacency : list of lists
        adjacency[i] contains the indices of vertices adjacent to vertex i
    """
    adjacency = [set() for _ in range(n_vertices)]

    for face in F:
        # Each edge creates a bidirectional adjacency
        adjacency[face[0]].add(face[1])
        adjacency[face[0]].add(face[2])
        adjacency[face[1]].add(face[0])
        adjacency[face[1]].add(face[2])
        adjacency[face[2]].add(face[0])
        adjacency[face[2]].add(face[1])

    # Convert sets to sorted lists for consistency
    adjacency = [sorted(list(neighbors)) for neighbors in adjacency]

    return adjacency


def compute_uniform_weights(V, F, adjacency):
    """
    Compute uniform weights (all neighbors have equal weight).

    Parameters
    ----------
    V : ndarray, shape (N, 3)
        Vertex positions (not used for uniform weights, but kept for consistency)
    F : ndarray, shape (M, 3)
        Triangle faces
    adjacency : list of lists
        Vertex adjacency list

    Returns
    -------
    weights : list of dicts
        weights[i][j] = weight from vertex i to neighbor j
    """
    n_vertices = len(adjacency)
    weights = []

    for i in range(n_vertices):
        neighbors = adjacency[i]
        n_neighbors = len(neighbors)

        if n_neighbors == 0:
            weights.append({})
            continue

        # Uniform weight: 1 / num_neighbors (so they sum to 1)
        w = 1.0 / n_neighbors
        weights.append({j: w for j in neighbors})

    return weights


def compute_mean_value_weights(V, F, adjacency):
    """
    Compute mean-value weights based on angles in the mesh.

    Mean-value coordinates were introduced by Floater (2003) and provide
    better shape preservation than uniform weights.

    For vertex i with neighbors j, the weight w_ij is:
        w_ij = (tan(α_k/2) + tan(α_k+1/2)) / ||v_i - v_j||

    where α_k are the angles at vertex i in the triangles sharing edge (i,j).

    Parameters
    ----------
    V : ndarray, shape (N, 3)
        Vertex positions
    F : ndarray, shape (M, 3)
        Triangle faces
    adjacency : list of lists
        Vertex adjacency list

    Returns
    -------
    weights : list of dicts
        weights[i][j] = weight from vertex i to neighbor j
    """
    n_vertices = len(adjacency)
    weights = []

    # Build vertex-to-faces map
    vertex_faces = [[] for _ in range(n_vertices)]
    for face_idx, face in enumerate(F):
        for v in face:
            vertex_faces[v].append(face_idx)

    for i in range(n_vertices):
        neighbors = adjacency[i]
        vertex_weights = {}

        if len(neighbors) == 0:
            weights.append({})
            continue

        vi = V[i]

        for j in neighbors:
            vj = V[j]
            edge_vec = vj - vi
            edge_length = np.linalg.norm(edge_vec)

            if edge_length < 1e-12:
                # Degenerate edge
                vertex_weights[j] = 0.0
                continue

            # Find triangles containing edge (i, j)
            shared_faces = []
            for face_idx in vertex_faces[i]:
                face = F[face_idx]
                if j in face:
                    shared_faces.append(face)

            # Compute angles at vertex i in each triangle
            tan_sum = 0.0

            for face in shared_faces:
                # Find the third vertex (not i or j)
                k = None
                for v in face:
                    if v != i and v != j:
                        k = v
                        break

                if k is None:
                    continue

                vk = V[k]

                # Vectors from i to j and i to k
                vec_j = vj - vi
                vec_k = vk - vi

                # Angle at vertex i
                cos_angle = np.dot(vec_j, vec_k) / (np.linalg.norm(vec_j) * np.linalg.norm(vec_k) + 1e-12)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)

                # tan(angle/2) using half-angle formula: tan(θ/2) = sin(θ)/(1 + cos(θ))
                tan_half = np.sin(angle) / (1.0 + cos_angle + 1e-12)
                tan_sum += tan_half

            # Mean-value weight
            w_ij = tan_sum / edge_length
            vertex_weights[j] = w_ij

        # Normalize so weights sum to 1
        total_weight = sum(vertex_weights.values())
        if total_weight > 1e-12:
            vertex_weights = {j: w / total_weight for j, w in vertex_weights.items()}

        weights.append(vertex_weights)

    return weights


def tutte_embedding(V, F, weight_type='uniform', boundary_shape='circle'):
    """
    Compute Tutte embedding for a disk-topology mesh.

    The boundary vertices are fixed to a convex polygon (circle by default),
    and interior vertices are placed at the weighted average of their neighbors.
    This guarantees a valid parameterization with no fold-overs.

    Parameters
    ----------
    V : ndarray, shape (N, 3)
        Vertex positions
    F : ndarray, shape (M, 3)
        Triangle face indices
    weight_type : {'uniform', 'mean_value'}
        Type of weights to use for interior vertices
    boundary_shape : {'circle'}
        Shape to map boundary vertices to (currently only circle is supported)

    Returns
    -------
    UV : ndarray, shape (N, 2)
        UV coordinates for each vertex

    Raises
    ------
    ValueError
        If the mesh is not disk topology (no boundary found)

    Notes
    -----
    The algorithm solves a sparse linear system:
        UV_interior = W · UV_neighbors
    where W contains the weights and UV_neighbors includes both interior and
    (fixed) boundary vertices.

    This is rearranged to:
        (I - W_interior) · UV_interior = W_boundary · UV_boundary
    which is a sparse linear system we can solve with scipy.
    """
    n_vertices = V.shape[0]

    # Find boundary vertices
    boundary, is_boundary = find_boundary_vertices(F, n_vertices)

    if len(boundary) == 0:
        raise ValueError("Mesh has no boundary (closed mesh). Tutte embedding requires disk topology.")

    n_boundary = len(boundary)
    n_interior = n_vertices - n_boundary

    print(f"Tutte embedding: {n_vertices} vertices ({n_interior} interior, {n_boundary} boundary)")

    # Build adjacency
    adjacency = build_adjacency(F, n_vertices)

    # Compute weights
    if weight_type == 'uniform':
        weights = compute_uniform_weights(V, F, adjacency)
    elif weight_type == 'mean_value':
        weights = compute_mean_value_weights(V, F, adjacency)
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}. Use 'uniform' or 'mean_value'.")

    # Fix boundary to a circle
    if boundary_shape != 'circle':
        raise ValueError(f"Unknown boundary_shape: {boundary_shape}. Currently only 'circle' is supported.")

    UV = np.zeros((n_vertices, 2), dtype=np.float64)

    # Map boundary vertices to circle
    for i, b_idx in enumerate(boundary):
        angle = 2.0 * np.pi * i / n_boundary
        UV[b_idx, 0] = np.cos(angle)  # u
        UV[b_idx, 1] = np.sin(angle)  # v

    # Build linear system for interior vertices
    # We need to map interior vertex indices to rows in our system
    interior_vertices = np.where(~is_boundary)[0]
    interior_to_row = {v: i for i, v in enumerate(interior_vertices)}

    print(f"Debug: n_interior={n_interior}, len(interior_vertices)={len(interior_vertices)}")
    print(f"Debug: n_boundary={n_boundary}, n_vertices={n_vertices}")
    print(f"Debug: is_boundary.sum()={is_boundary.sum()}")

    # Build sparse matrix A and right-hand side b
    # A · UV_interior = b
    # where b comes from the boundary constraints

    A = lil_matrix((n_interior, n_interior), dtype=np.float64)
    b_u = np.zeros(n_interior, dtype=np.float64)
    b_v = np.zeros(n_interior, dtype=np.float64)

    for v in interior_vertices:
        row = interior_to_row[v]

        # Diagonal entry: coefficient of UV[v] = 1
        A[row, row] = 1.0

        # Neighbor contributions
        for neighbor, weight in weights[v].items():
            if is_boundary[neighbor]:
                # Boundary vertex: contributes to right-hand side
                b_u[row] += weight * UV[neighbor, 0]
                b_v[row] += weight * UV[neighbor, 1]
            else:
                # Interior vertex: contributes to matrix
                if neighbor not in interior_to_row:
                    print(f"Warning: neighbor {neighbor} not in interior_to_row (v={v})")
                    continue
                neighbor_row = interior_to_row[neighbor]
                A[row, neighbor_row] -= weight

    # Convert to CSR format for efficient solving
    A = A.tocsr()

    # Solve for interior UV coordinates
    print(f"Solving sparse linear system ({n_interior} × {n_interior})...")
    uv_interior_u = spsolve(A, b_u)
    uv_interior_v = spsolve(A, b_v)

    # Fill in interior vertices
    for i, v in enumerate(interior_vertices):
        UV[v, 0] = uv_interior_u[i]
        UV[v, 1] = uv_interior_v[i]

    print("Tutte embedding complete.")

    return UV
