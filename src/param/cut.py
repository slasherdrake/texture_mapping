"""
Mesh cutting algorithms to convert closed meshes to disk topology.

For parameterization methods like Tutte and LSCM that require disk topology,
we need to cut closed meshes (like bunny, armadillo) along a seam to create
a boundary.

This implements a simple greedy cut that finds a path from a starting vertex
to create a single boundary loop.

References:
-----------
Sheffer et al., "Mesh Parameterization Methods and Their Applications"
Hormann et al., "Mesh Parameterization: Theory and Practice", SIGGRAPH 2007
"""

import numpy as np
from collections import deque
import heapq


def build_edge_map(F, n_vertices):
    """
    Build edge-to-faces map and adjacency information.

    Parameters
    ----------
    F : ndarray, shape (M, 3)
        Triangle faces
    n_vertices : int
        Number of vertices

    Returns
    -------
    edge_to_faces : dict
        Maps (v0, v1) tuple to list of face indices
    vertex_adjacency : list of sets
        vertex_adjacency[i] = set of neighbor vertex indices
    """
    edge_to_faces = {}
    vertex_adjacency = [set() for _ in range(n_vertices)]

    for face_idx, face in enumerate(F):
        # Three edges per triangle
        edges = [
            (min(face[0], face[1]), max(face[0], face[1])),
            (min(face[1], face[2]), max(face[1], face[2])),
            (min(face[2], face[0]), max(face[2], face[0]))
        ]

        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(face_idx)

            # Update adjacency
            vertex_adjacency[edge[0]].add(edge[1])
            vertex_adjacency[edge[1]].add(edge[0])

    return edge_to_faces, vertex_adjacency


def find_boundary_after_cut(F, cut_edges):
    """
    After cutting along edges, find the resulting boundary vertices.

    Parameters
    ----------
    F : ndarray, shape (M, 3)
        Triangle faces
    cut_edges : set of tuples
        Edges to cut (as sorted tuples)

    Returns
    -------
    boundary_vertices : list
        Ordered list of boundary vertex indices
    """
    # Build edge map excluding cut edges
    edge_count = {}

    for face in F:
        edges = [
            (min(face[0], face[1]), max(face[0], face[1])),
            (min(face[1], face[2]), max(face[1], face[2])),
            (min(face[2], face[0]), max(face[2], face[0]))
        ]

        for edge in edges:
            if edge not in cut_edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1

    # Boundary edges appear exactly once (after cutting)
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    # Also include cut edges (they become boundary)
    for edge in cut_edges:
        boundary_edges.append(edge)

    if len(boundary_edges) == 0:
        return []

    # Build ordered boundary loop
    boundary = list(boundary_edges[0])
    used_edges = {boundary_edges[0]}

    while len(used_edges) < len(boundary_edges):
        current = boundary[-1]

        # Find next edge
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
            break

    # Remove duplicate (last = first)
    if len(boundary) > 0 and boundary[-1] == boundary[0]:
        boundary = boundary[:-1]

    return boundary


def shortest_path_cut(V, F, start_vertex=None):
    """
    Cut a closed mesh by finding a shortest path that creates a disk topology.

    This implements a simple strategy:
    1. Pick a starting vertex (or use the provided one)
    2. Find a path that, when cut, opens the mesh into a disk
    3. Use a greedy approach to find a reasonable cut

    Parameters
    ----------
    V : ndarray, shape (N, 3)
        Vertex positions
    F : ndarray, shape (M, 3)
        Triangle faces
    start_vertex : int, optional
        Starting vertex for the cut (default: vertex 0)

    Returns
    -------
    cut_edges : set of tuples
        Edges to cut (as sorted tuples (min, max))
    boundary_vertices : list
        Ordered boundary vertices after cutting

    Notes
    -----
    This is a simplified cutting algorithm. More sophisticated methods use:
    - Dijkstra's algorithm on the dual graph
    - Minimal spanning tree cuts
    - Feature-aware cutting (preserve important features)

    For this implementation, we use a greedy approach that:
    - Starts from a vertex
    - Grows a cut path using BFS
    - Stops when we've created a sufficient boundary
    """
    n_vertices = V.shape[0]

    if start_vertex is None:
        # Pick a vertex on the "edge" of the mesh (max x + y + z)
        sums = V[:, 0] + V[:, 1] + V[:, 2]
        start_vertex = int(np.argmax(sums))

    # Build edge and adjacency maps
    edge_to_faces, vertex_adjacency = build_edge_map(F, n_vertices)

    # Strategy: Find a spanning path from start_vertex that cuts the mesh open
    # We'll use a simple approach: BFS from start, cutting edges as we go

    # Find a target vertex (far from start)
    distances = dijkstra_distances(V, vertex_adjacency, start_vertex)
    target_vertex = int(np.argmax(distances))

    # Find shortest path from start to target
    path = dijkstra_path(vertex_adjacency, start_vertex, target_vertex)

    # Cut along this path
    cut_edges = set()
    for i in range(len(path) - 1):
        edge = (min(path[i], path[i+1]), max(path[i], path[i+1]))
        cut_edges.add(edge)

    # Check if this creates a boundary
    boundary = find_boundary_after_cut(F, cut_edges)

    print(f"Cut: {len(cut_edges)} edges, boundary: {len(boundary)} vertices")

    return cut_edges, boundary


def dijkstra_distances(V, adjacency, start_vertex):
    """
    Compute shortest path distances from start vertex using Dijkstra's algorithm.

    Parameters
    ----------
    V : ndarray, shape (N, 3)
        Vertex positions
    adjacency : list of sets
        Vertex adjacency
    start_vertex : int
        Starting vertex

    Returns
    -------
    distances : ndarray
        Distance from start_vertex to each vertex
    """
    n_vertices = len(adjacency)
    distances = np.full(n_vertices, np.inf)
    distances[start_vertex] = 0.0

    # Priority queue: (distance, vertex)
    pq = [(0.0, start_vertex)]
    visited = set()

    while pq:
        dist, u = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)

        for v in adjacency[u]:
            edge_length = np.linalg.norm(V[u] - V[v])
            new_dist = dist + edge_length

            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(pq, (new_dist, v))

    return distances


def dijkstra_path(adjacency, start_vertex, target_vertex):
    """
    Find shortest path from start to target using Dijkstra's algorithm.

    Parameters
    ----------
    adjacency : list of sets
        Vertex adjacency
    start_vertex : int
        Starting vertex
    target_vertex : int
        Target vertex

    Returns
    -------
    path : list
        Vertex indices along shortest path from start to target
    """
    n_vertices = len(adjacency)
    distances = {start_vertex: 0}
    previous = {}

    # Priority queue: (distance, vertex)
    pq = [(0, start_vertex)]
    visited = set()

    while pq:
        dist, u = heapq.heappop(pq)

        if u == target_vertex:
            # Reconstruct path
            path = []
            current = target_vertex
            while current in previous:
                path.append(current)
                current = previous[current]
            path.append(start_vertex)
            return path[::-1]

        if u in visited:
            continue
        visited.add(u)

        for v in adjacency[u]:
            new_dist = dist + 1  # Unweighted (just hop count)

            if v not in distances or new_dist < distances[v]:
                distances[v] = new_dist
                previous[v] = u
                heapq.heappush(pq, (new_dist, v))

    # No path found
    return [start_vertex]


def cut_mesh_to_disk(V, F):
    """
    Cut a closed mesh to create disk topology for parameterization.

    Parameters
    ----------
    V : ndarray, shape (N, 3)
        Vertex positions
    F : ndarray, shape (M, 3)
        Triangle faces

    Returns
    -------
    V_cut : ndarray, shape (N', 3)
        Vertex positions after cutting (may have duplicates along seam)
    F_cut : ndarray, shape (M, 3)
        Triangle faces with updated vertex indices
    boundary : ndarray
        Ordered boundary vertex indices

    Notes
    -----
    This duplicates vertices along the cut seam so that the mesh has
    a proper boundary for disk-topology parameterization.
    """
    # Find cut
    cut_edges, boundary_initial = shortest_path_cut(V, F)

    if len(cut_edges) == 0:
        print("Warning: No cut needed (mesh already has boundary)")
        return V, F, boundary_initial

    # Duplicate vertices along cut seam
    # Each cut edge will have its vertices duplicated
    cut_vertices = set()
    for edge in cut_edges:
        cut_vertices.add(edge[0])
        cut_vertices.add(edge[1])

    # Create mapping: old vertex -> [list of new vertex indices]
    vertex_mapping = {}
    V_new = []
    next_vertex_id = 0

    for v_idx in range(V.shape[0]):
        if v_idx in cut_vertices:
            # This vertex is on the cut seam - it will be duplicated
            vertex_mapping[v_idx] = []  # Will populate as we process faces
        else:
            # Regular vertex - just copy
            vertex_mapping[v_idx] = [next_vertex_id]
            V_new.append(V[v_idx])
            next_vertex_id += 1

    # Process faces and duplicate cut vertices as needed
    F_new = []

    for face in F:
        new_face = []

        for v_idx in face:
            if v_idx in cut_vertices:
                # Check if we've already created a duplicate for this face
                # For now, create a new one each time (simple approach)
                new_v_idx = next_vertex_id
                V_new.append(V[v_idx])
                vertex_mapping.setdefault(v_idx, []).append(new_v_idx)
                next_vertex_id += 1
                new_face.append(new_v_idx)
            else:
                new_face.append(vertex_mapping[v_idx][0])

        F_new.append(new_face)

    V_cut = np.array(V_new, dtype=np.float64)
    F_cut = np.array(F_new, dtype=np.int64)

    # Find new boundary
    # After cutting, we need to find boundary in the new mesh
    from src.param.tutte import find_boundary_vertices
    boundary, _ = find_boundary_vertices(F_cut, V_cut.shape[0])

    print(f"Cut mesh: {V.shape[0]} -> {V_cut.shape[0]} vertices")
    print(f"Boundary: {len(boundary)} vertices")

    return V_cut, F_cut, boundary
