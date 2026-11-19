"""
Automatic mesh segmentation for LSCM texture atlas generation.

Implements the segmentation algorithm from:
Lévy et al., "Least Squares Conformal Maps for Automatic Texture Atlas Generation",
SIGGRAPH 2002, Section 3.

The algorithm decomposes a mesh into charts homeomorphic to discs by:
1. Detecting sharp feature curves (high curvature zones)
2. Growing charts from seeds, meeting at features
3. Validating charts and subdividing if needed
"""

import numpy as np
from collections import deque
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
import heapq


class MeshSegmenter:
    """Segments a mesh into charts for LSCM parameterization."""

    def __init__(self, V, F):
        """
        Parameters
        ----------
        V : (N, 3) array
            Vertex positions
        F : (M, 3) array
            Triangle faces
        """
        self.V = V
        self.F = F
        self.n_vertices = len(V)
        self.n_faces = len(F)

        # Build data structures
        self._build_topology()
        self._compute_face_normals()

        # Feature detection results
        self.feature_edges = set()
        self.feature_neighbors = set()

        # Segmentation results
        self.face_to_chart = np.full(self.n_faces, -1, dtype=np.int32)
        self.charts = []  # List of dicts with face indices and boundary

    def _build_topology(self):
        """Build edge-face adjacency and halfedge structures."""
        # Edge to faces map
        edges = []
        edge_to_faces = {}

        for fi, face in enumerate(self.F):
            for i in range(3):
                v0, v1 = face[i], face[(i + 1) % 3]
                edge = tuple(sorted([v0, v1]))
                edges.append((v0, v1, fi, i))  # directed edge, face, local edge index

                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(fi)

        self.edges = edges
        self.edge_to_faces = edge_to_faces

        # Build halfedge structure for navigation
        # halfedge = (face_idx, local_edge_idx)
        self.face_halfedges = {}  # face_idx -> [(v0, v1, opposite_face), ...]

        for fi, face in enumerate(self.F):
            halfedges = []
            for i in range(3):
                v0, v1 = face[i], face[(i + 1) % 3]
                edge = tuple(sorted([v0, v1]))

                # Find opposite halfedge
                opposite_fi = None
                for adj_fi in edge_to_faces[edge]:
                    if adj_fi != fi:
                        opposite_fi = adj_fi
                        break

                halfedges.append((v0, v1, opposite_fi))

            self.face_halfedges[fi] = halfedges

    def _compute_face_normals(self):
        """Compute face normals for sharpness computation."""
        v0 = self.V[self.F[:, 0]]
        v1 = self.V[self.F[:, 1]]
        v2 = self.V[self.F[:, 2]]

        normals = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-10)  # Avoid division by zero

        self.face_normals = normals / lengths

    def _compute_edge_sharpness(self):
        """
        Compute edge sharpness using Second Order Differences (SOD).
        Returns angle in degrees between adjacent face normals.
        """
        sharpness = {}

        for edge, faces in self.edge_to_faces.items():
            if len(faces) == 2:
                # Interior edge - compute angle between normals
                n0 = self.face_normals[faces[0]]
                n1 = self.face_normals[faces[1]]

                # Angle between normals (in degrees)
                cos_angle = np.clip(np.dot(n0, n1), -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))

                sharpness[edge] = angle
            else:
                # Boundary edge - mark as sharp
                sharpness[edge] = 180.0

        return sharpness

    def detect_features(self, percentile=95, max_string_length=5,
                       min_feature_length=15, neighborhood_radius=2):
        """
        Detect sharp feature curves (Algorithm 1 from paper).

        Parameters
        ----------
        percentile : float
            Keep edges above this percentile of sharpness (95 = top 5%)
        max_string_length : int
            Maximum length of feature curve to grow in one step
        min_feature_length : int
            Minimum length to keep a feature curve
        neighborhood_radius : int
            Radius for tagging feature neighborhoods
        """
        print(f"Detecting features (keeping top {100-percentile}% sharpest edges)...")

        # Compute sharpness for all edges
        edge_sharpness = self._compute_edge_sharpness()

        # Determine threshold
        sharpness_values = list(edge_sharpness.values())
        threshold = np.percentile(sharpness_values, percentile)

        print(f"  Sharpness threshold: {threshold:.2f} degrees")

        # Filter sharp edges
        sharp_edges = {edge: s for edge, s in edge_sharpness.items()
                      if s >= threshold}

        print(f"  Found {len(sharp_edges)} sharp edges (from {len(edge_sharpness)} total)")

        # Build directed edge graph for feature growing
        self.sharp_edge_graph = {}
        for (v0, v1) in sharp_edges.keys():
            if v0 not in self.sharp_edge_graph:
                self.sharp_edge_graph[v0] = []
            if v1 not in self.sharp_edge_graph:
                self.sharp_edge_graph[v1] = []
            self.sharp_edge_graph[v0].append(v1)
            self.sharp_edge_graph[v1].append(v0)

        # Grow feature curves
        visited_edges = set()
        feature_curves = []

        for edge in sharp_edges.keys():
            if edge in visited_edges:
                continue

            # Grow curve from this edge
            curve = self._expand_feature_curve(
                edge, edge_sharpness, threshold,
                max_string_length, visited_edges
            )

            if len(curve) >= min_feature_length:
                feature_curves.append(curve)
                # Mark edges as visited
                for i in range(len(curve) - 1):
                    e = tuple(sorted([curve[i], curve[i+1]]))
                    visited_edges.add(e)
                    self.feature_edges.add(e)

        print(f"  Detected {len(feature_curves)} feature curves")

        # Tag feature neighborhoods
        self._tag_feature_neighborhoods(neighborhood_radius)

        return feature_curves

    def _expand_feature_curve(self, start_edge, edge_sharpness, threshold,
                             max_string_length, visited):
        """
        Expand a feature curve from a starting edge (Algorithm 1).
        Uses greedy algorithm to find the best path.
        """
        v0, v1 = start_edge
        curve = [v0, v1]

        # Mark starting edge as visited
        visited.add(start_edge)

        # Grow in both directions
        for direction in [v1, v0]:
            current = direction
            prev = v1 if direction == v0 else v0

            while True:
                # Find best string using greedy algorithm
                best_string = self._find_best_string(
                    current, prev, edge_sharpness,
                    max_string_length, visited
                )

                if not best_string or len(best_string) < 2:
                    break

                # Check if string is sharp enough
                string_sharpness = sum(
                    edge_sharpness.get(tuple(sorted([best_string[i], best_string[i+1]])), 0)
                    for i in range(len(best_string) - 1)
                )

                if string_sharpness < threshold * (len(best_string) - 1):
                    break

                # Mark edges in this string as visited to prevent infinite loops
                for i in range(len(best_string) - 1):
                    edge = tuple(sorted([best_string[i], best_string[i+1]]))
                    visited.add(edge)

                # Add to curve (skip first vertex, it's already in curve)
                if direction == v1:
                    curve.extend(best_string[1:])
                else:
                    curve[:0] = best_string[1:][::-1]

                # Update for next iteration
                prev = best_string[-2] if len(best_string) > 1 else current
                current = best_string[-1]

        return curve

    def _find_best_string(self, start, prev, edge_sharpness, max_length, visited):
        """
        Find the best string of vertices using greedy algorithm.
        At each step, follows the sharpest available edge.

        This is O(max_length) instead of O(branches^max_length) for DFS.
        Returns list of vertices forming the string.
        """
        if start not in self.sharp_edge_graph:
            return None

        path = [start]
        visited_in_path = {start}

        # Greedy: at each step, pick the sharpest unvisited edge
        for _ in range(max_length):
            current = path[-1]

            if current not in self.sharp_edge_graph:
                break

            # Find the sharpest available neighbor
            best_next = None
            best_sharpness = -1

            for next_v in self.sharp_edge_graph[current]:
                # Don't go backward
                if len(path) >= 2 and next_v == path[-2]:
                    continue

                # Don't revisit in this path
                if next_v in visited_in_path:
                    continue

                # Check if edge was visited globally
                edge = tuple(sorted([current, next_v]))
                if edge in visited:
                    continue

                # Get sharpness
                s = edge_sharpness.get(edge, 0)

                if s > best_sharpness:
                    best_sharpness = s
                    best_next = next_v

            # If no valid neighbor, stop growing
            if best_next is None:
                break

            # Add to path
            path.append(best_next)
            visited_in_path.add(best_next)

        # Return path only if we grew at least one edge
        return path if len(path) >= 2 else None

    def _tag_feature_neighborhoods(self, radius):
        """Tag faces near feature edges."""
        # Build face adjacency graph
        face_adj = {i: [] for i in range(self.n_faces)}

        for edge, faces in self.edge_to_faces.items():
            if len(faces) == 2:
                face_adj[faces[0]].append(faces[1])
                face_adj[faces[1]].append(faces[0])

        # For each feature edge, mark nearby faces
        feature_faces = set()
        for edge in self.feature_edges:
            if edge in self.edge_to_faces:
                for fi in self.edge_to_faces[edge]:
                    feature_faces.add(fi)

        # BFS from feature faces
        self.feature_neighbors = set()
        queue = deque([(f, 0) for f in feature_faces])
        visited = set(feature_faces)

        while queue:
            face, dist = queue.popleft()
            if dist < radius:
                for adj_face in face_adj[face]:
                    if adj_face not in visited:
                        visited.add(adj_face)
                        self.feature_neighbors.add(adj_face)
                        queue.append((adj_face, dist + 1))

        print(f"  Tagged {len(self.feature_neighbors)} faces in feature neighborhoods")

    def segment_into_charts(self, merge_threshold=0.25):
        """
        Segment mesh into charts using chart growing (Algorithm 2).

        Parameters
        ----------
        merge_threshold : float
            Merge charts if they meet at distance < threshold * max_distance
        """
        print(f"\nSegmenting mesh into charts...")

        # Compute distance to features for all faces
        distance_to_features = self._compute_distance_to_features()

        # Find seeds (local maxima of distance_to_features)
        seeds = self._find_seeds(distance_to_features)

        print(f"  Found {len(seeds)} seed faces")

        # Grow charts using modified Dijkstra
        self._grow_charts(seeds, distance_to_features, merge_threshold)

        # Extract chart information
        self._extract_charts()

        print(f"  Created {len(self.charts)} charts")

        return self.charts

    def _compute_distance_to_features(self):
        """
        Compute distance from each face to nearest feature.
        Uses face adjacency graph with Dijkstra.
        """
        # Build face adjacency with weights (edge lengths)
        n = self.n_faces
        row, col, data = [], [], []

        for fi in range(n):
            for he in self.face_halfedges[fi]:
                v0, v1, opp_fi = he
                if opp_fi is not None:
                    # Edge length as weight
                    edge_len = np.linalg.norm(self.V[v1] - self.V[v0])
                    row.append(fi)
                    col.append(opp_fi)
                    data.append(edge_len)

        graph = coo_matrix((data, (row, col)), shape=(n, n)).tocsr()

        # Feature faces as sources
        sources = []
        for edge in self.feature_edges:
            if edge in self.edge_to_faces:
                sources.extend(self.edge_to_faces[edge])

        # Also include boundary faces if no features
        if not sources:
            # Find boundary faces
            for edge, faces in self.edge_to_faces.items():
                if len(faces) == 1:
                    sources.append(faces[0])

        sources = list(set(sources))

        if not sources:
            # Closed mesh with no features - use two diameter vertices
            print("  Warning: No features or boundary found, using diameter vertices")
            return np.ones(n)

        # Multi-source shortest path
        distances = np.full(n, np.inf)
        distances[sources] = 0

        # Dijkstra from all sources simultaneously
        heap = [(0, s) for s in sources]
        heapq.heapify(heap)

        while heap:
            dist, fi = heapq.heappop(heap)

            if dist > distances[fi]:
                continue

            # Check neighbors
            for he in self.face_halfedges[fi]:
                _, _, opp_fi = he
                if opp_fi is not None:
                    edge_len = graph[fi, opp_fi]
                    new_dist = dist + edge_len

                    if new_dist < distances[opp_fi]:
                        distances[opp_fi] = new_dist
                        heapq.heappush(heap, (new_dist, opp_fi))

        return distances

    def _find_seeds(self, distances):
        """Find local maxima of distance_to_features as seeds."""
        seeds = []

        for fi in range(self.n_faces):
            # Check if local maximum
            is_max = True
            for he in self.face_halfedges[fi]:
                _, _, opp_fi = he
                if opp_fi is not None and distances[opp_fi] > distances[fi]:
                    is_max = False
                    break

            if is_max and distances[fi] > 0:
                seeds.append(fi)

        return seeds

    def _grow_charts(self, seeds, distances, merge_threshold):
        """
        Grow charts from seeds using priority queue.
        Priority = -distance_to_features (grow from far from features first).
        """
        # Initialize charts
        chart_id = 0
        chart_max_dist = {}

        for seed in seeds:
            self.face_to_chart[seed] = chart_id
            chart_max_dist[chart_id] = distances[seed]
            chart_id += 1

        # Priority queue: (-distance, face_idx, chart_id)
        heap = []
        for seed in seeds:
            cid = self.face_to_chart[seed]
            # Add neighbors to queue
            for he in self.face_halfedges[seed]:
                _, _, opp_fi = he
                if opp_fi is not None and self.face_to_chart[opp_fi] == -1:
                    heapq.heappush(heap, (-distances[opp_fi], opp_fi, cid))

        # Grow charts
        while heap:
            neg_dist, fi, cid = heapq.heappop(heap)

            # Already assigned
            if self.face_to_chart[fi] != -1:
                # Check for merge
                other_cid = self.face_to_chart[fi]
                if other_cid != cid:
                    # Check merge condition
                    max_dist = max(chart_max_dist[cid], chart_max_dist[other_cid])
                    threshold = merge_threshold * max_dist

                    if -neg_dist > max_dist - threshold:
                        # Merge charts - reassign all faces from other_cid to cid
                        self.face_to_chart[self.face_to_chart == other_cid] = cid
                        chart_max_dist[cid] = max(chart_max_dist[cid],
                                                   chart_max_dist[other_cid])
                continue

            # Assign face to chart
            self.face_to_chart[fi] = cid

            # Add neighbors
            for he in self.face_halfedges[fi]:
                _, _, opp_fi = he
                if opp_fi is not None and self.face_to_chart[opp_fi] == -1:
                    heapq.heappush(heap, (-distances[opp_fi], opp_fi, cid))

    def _extract_charts(self):
        """Extract chart information from face_to_chart."""
        unique_ids = np.unique(self.face_to_chart[self.face_to_chart >= 0])

        for chart_id in unique_ids:
            # Get faces in this chart
            face_mask = (self.face_to_chart == chart_id)
            chart_faces = np.where(face_mask)[0]

            # Get vertices in this chart
            chart_verts = np.unique(self.F[chart_faces].ravel())

            # Find boundary
            boundary_edges = []
            for fi in chart_faces:
                for he in self.face_halfedges[fi]:
                    v0, v1, opp_fi = he
                    if opp_fi is None or self.face_to_chart[opp_fi] != chart_id:
                        boundary_edges.append((v0, v1))

            self.charts.append({
                'id': int(chart_id),
                'faces': chart_faces,
                'vertices': chart_verts,
                'boundary_edges': boundary_edges,
                'n_faces': len(chart_faces),
                'n_vertices': len(chart_verts)
            })


def segment_mesh_for_lscm(V, F, percentile=95, max_string_length=5,
                          min_feature_length=15, neighborhood_radius=2,
                          merge_threshold=0.25, **kwargs):
    """
    Segment a mesh into charts suitable for LSCM parameterization.

    Parameters
    ----------
    V : (N, 3) array
        Vertex positions
    F : (M, 3) array
        Triangle faces
    percentile : float
        Keep edges above this percentile of sharpness
    max_string_length : int
        Maximum length of feature curve to grow in one step
    min_feature_length : int
        Minimum length to keep a feature curve
    neighborhood_radius : int
        Radius for tagging feature neighborhoods
    merge_threshold : float
        Merge charts if they meet close to their seeds
    **kwargs : optional
        Additional parameters

    Returns
    -------
    charts : list of dict
        Each chart contains 'faces', 'vertices', 'boundary_edges', etc.
    """
    segmenter = MeshSegmenter(V, F)
    segmenter.detect_features(
        percentile=percentile,
        max_string_length=max_string_length,
        min_feature_length=min_feature_length,
        neighborhood_radius=neighborhood_radius
    )
    charts = segmenter.segment_into_charts(merge_threshold=merge_threshold)
    return charts
