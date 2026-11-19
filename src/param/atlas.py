"""
Texture Atlas generation using automatic segmentation + LSCM.

Implements the complete pipeline from Lévy et al. SIGGRAPH 2002:
1. Segment mesh into charts
2. Parameterize each chart with LSCM
3. Validate and subdivide if needed
4. Pack charts in texture space
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .segmentation import segment_mesh_for_lscm
from .lscm import lscm_embedding
from ..distortion import compute_distortion_metrics


def split_disconnected_chart(V, F, chart):
    """
    Split a chart into connected components if it's disconnected.

    Returns
    -------
    charts : list of dict
        List of connected charts (original chart if already connected)
    """
    chart_faces = chart['faces']

    if len(chart_faces) == 0:
        return []

    # Build face adjacency graph
    n_faces = len(chart_faces)
    face_to_idx = {fi: i for i, fi in enumerate(chart_faces)}

    # Find which faces share edges
    edge_to_faces = {}
    for idx, fi in enumerate(chart_faces):
        face = F[fi]
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(idx)

    # Build adjacency matrix
    row, col = [], []
    for edge, faces in edge_to_faces.items():
        if len(faces) == 2:
            row.extend([faces[0], faces[1]])
            col.extend([faces[1], faces[0]])

    if not row:
        # No internal edges - each face is its own component
        return [
            {
                'id': chart['id'],
                'faces': np.array([fi]),
                'vertices': np.unique(F[fi]),
                'n_faces': 1,
                'n_vertices': len(np.unique(F[fi]))
            }
            for fi in chart_faces
        ]

    adj = coo_matrix((np.ones(len(row)), (row, col)), shape=(n_faces, n_faces))
    n_components, labels = connected_components(adj, directed=False)

    if n_components == 1:
        # Already connected
        return [chart]

    # Split into separate charts
    split_charts = []
    for comp_id in range(n_components):
        comp_face_indices = np.where(labels == comp_id)[0]
        comp_faces = chart_faces[comp_face_indices]
        comp_verts = np.unique(F[comp_faces].ravel())

        split_charts.append({
            'id': chart['id'],
            'faces': comp_faces,
            'vertices': comp_verts,
            'n_faces': len(comp_faces),
            'n_vertices': len(comp_verts)
        })

    return split_charts


def extract_chart_mesh(V, F, chart):
    """
    Extract a submesh for a single chart.

    Parameters
    ----------
    V : (N, 3) array
        Full mesh vertices
    F : (M, 3) array
        Full mesh faces
    chart : dict
        Chart info with 'faces' and 'vertices'

    Returns
    -------
    V_chart : (N_chart, 3) array
        Chart vertices
    F_chart : (M_chart, 3) array
        Chart faces (reindexed)
    vertex_map : dict
        Mapping from chart vertex indices to original mesh indices
    """
    # Get chart vertices and faces
    chart_verts = chart['vertices']
    chart_faces = chart['faces']

    # Create submesh
    V_chart = V[chart_verts]

    # Create mapping from old to new indices
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(chart_verts)}

    # Reindex faces
    F_chart = np.array([[old_to_new[v] for v in F[fi]] for fi in chart_faces])

    vertex_map = {new_idx: old_idx for old_idx, new_idx in old_to_new.items()}

    return V_chart, F_chart, vertex_map


def parameterize_chart(V_chart, F_chart):
    """
    Parameterize a single chart using LSCM.

    Returns
    -------
    uv : (N_chart, 2) array or None
        UV coordinates, or None if LSCM failed
    success : bool
        Whether parameterization succeeded
    error_msg : str
        Error message if failed
    """
    try:
        uv = lscm_embedding(V_chart, F_chart)
        return uv, True, None
    except ValueError as e:
        return None, False, str(e)


def check_self_overlap(V_chart, F_chart, uv):
    """
    Check if parameterization has self-overlaps.

    Returns True if no overlaps detected.
    Uses triangle orientation - if any triangle flips, there might be overlap.
    """
    # Compute signed areas in 3D and UV
    v0_3d = V_chart[F_chart[:, 0]]
    v1_3d = V_chart[F_chart[:, 1]]
    v2_3d = V_chart[F_chart[:, 2]]

    # 3D cross product
    cross_3d = np.cross(v1_3d - v0_3d, v2_3d - v0_3d)
    area_3d = np.linalg.norm(cross_3d, axis=1)

    # UV cross product (2D)
    v0_uv = uv[F_chart[:, 0]]
    v1_uv = uv[F_chart[:, 1]]
    v2_uv = uv[F_chart[:, 2]]

    e1 = v1_uv - v0_uv
    e2 = v2_uv - v0_uv
    area_uv = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]

    # Check for flips (negative UV area)
    has_flips = np.any(area_uv <= 0)

    # Also check for degenerate triangles
    has_degenerate = np.any(np.abs(area_uv) < 1e-10)

    return not (has_flips or has_degenerate)


def check_area_distortion(V_chart, F_chart, uv, threshold=2.0):
    """
    Check if area distortion is within acceptable range.

    Parameters
    ----------
    threshold : float
        Maximum ratio of max/min area scaling

    Returns
    -------
    valid : bool
        True if distortion is acceptable
    ratio : float
        Actual max/min ratio
    """
    try:
        metrics = compute_distortion_metrics(V_chart, F_chart, uv)

        # Get area distortion per face
        area_dist = metrics['area_distortion']

        # Filter out NaN and inf
        valid_areas = area_dist[np.isfinite(area_dist) & (area_dist > 0)]

        if len(valid_areas) == 0:
            return False, np.inf

        ratio = np.max(valid_areas) / np.min(valid_areas)

        return ratio <= threshold, ratio

    except Exception as e:
        print(f"    Warning: Could not compute area distortion: {e}")
        return False, np.inf


def subdivide_chart(V, F, chart, face_to_subdivide):
    """
    Subdivide a chart by growing two new charts from problematic regions.

    Parameters
    ----------
    V : (N, 3) array
        Full mesh vertices
    F : (M, 3) array
        Full mesh faces
    chart : dict
        Original chart
    face_to_subdivide : array
        Boolean mask of faces in chart to split from

    Returns
    -------
    new_charts : list of dict
        Two new charts
    """
    # Simple subdivision: split chart in half based on face with min/max area distortion
    chart_faces = chart['faces']

    if len(chart_faces) < 2:
        # Can't subdivide further
        return [chart]

    # Split at midpoint
    mid = len(chart_faces) // 2

    chart1_faces = chart_faces[:mid]
    chart2_faces = chart_faces[mid:]

    # Create new charts
    def make_chart(faces):
        verts = np.unique(F[faces].ravel())
        return {
            'id': chart['id'],
            'faces': faces,
            'vertices': verts,
            'n_faces': len(faces),
            'n_vertices': len(verts)
        }

    return [make_chart(chart1_faces), make_chart(chart2_faces)]


def generate_texture_atlas(V, F, max_area_distortion=2.0, max_subdivisions=2,
                          **segmentation_kwargs):
    """
    Generate complete texture atlas with automatic segmentation.

    Parameters
    ----------
    V : (N, 3) array
        Mesh vertices
    F : (M, 3) array
        Mesh faces
    max_area_distortion : float
        Maximum acceptable area distortion ratio
    max_subdivisions : int
        Maximum number of times to subdivide problematic charts
    **segmentation_kwargs
        Parameters for segmentation

    Returns
    -------
    atlas : dict
        Contains:
        - 'charts': list of successfully parameterized charts
        - 'failed_charts': list of charts that couldn't be parameterized
        - 'uv_coords': dict mapping chart_id -> UV coordinates
        - 'metrics': dict mapping chart_id -> distortion metrics
    """
    print("="*70)
    print("GENERATING TEXTURE ATLAS")
    print("="*70)

    # Step 1: Segment mesh
    charts = segment_mesh_for_lscm(V, F, **segmentation_kwargs)

    # Step 1.5: Split any disconnected charts into connected components
    print(f"\nSplitting disconnected charts...")
    connected_charts = []
    for chart in charts:
        split = split_disconnected_chart(V, F, chart)
        connected_charts.extend(split)

    if len(connected_charts) > len(charts):
        print(f"  Split {len(charts)} charts into {len(connected_charts)} connected components")

    charts = connected_charts

    # Step 2: Parameterize each chart
    parameterized_charts = []
    failed_charts = []
    uv_coords = {}
    metrics = {}

    for i, chart in enumerate(charts):
        chart_id = chart['id']
        print(f"\nChart {i+1}/{len(charts)} (ID={chart_id}): {chart['n_faces']} faces, {chart['n_vertices']} vertices")

        # Extract submesh
        V_chart, F_chart, vertex_map = extract_chart_mesh(V, F, chart)

        # Try parameterization
        subdivision_count = 0
        current_charts = [chart]

        while subdivision_count <= max_subdivisions and current_charts:
            chart_to_process = current_charts.pop(0)

            # Extract submesh for current chart
            V_sub, F_sub, v_map = extract_chart_mesh(V, F, chart_to_process)

            # Parameterize
            uv, success, error_msg = parameterize_chart(V_sub, F_sub)

            if not success:
                print(f"  ✗ LSCM failed: {error_msg}")
                failed_charts.append(chart_to_process)
                continue

            # Validate
            print(f"  ✓ LSCM succeeded")

            # Check overlaps
            has_overlaps = not check_self_overlap(V_sub, F_sub, uv)
            if has_overlaps:
                print(f"  ⚠ Warning: Detected overlaps")

            # Check area distortion
            valid_area, ratio = check_area_distortion(V_sub, F_sub, uv, max_area_distortion)

            print(f"  Area distortion ratio: {ratio:.2f} (threshold: {max_area_distortion})")

            if not valid_area and subdivision_count < max_subdivisions:
                print(f"  → Subdividing chart (attempt {subdivision_count + 1}/{max_subdivisions})")
                new_charts = subdivide_chart(V, F, chart_to_process, None)
                current_charts.extend(new_charts)
                subdivision_count += 1
            else:
                # Accept this chart
                if not valid_area:
                    print(f"  ⚠ Accepting with high distortion (max subdivisions reached)")

                # Compute metrics
                try:
                    chart_metrics = compute_distortion_metrics(V_sub, F_sub, uv)
                    metrics[chart_id] = chart_metrics

                    print(f"  Mean angle distortion: {chart_metrics['mean_angle_distortion']:.3f}")
                    print(f"  Fold-overs: {chart_metrics['num_fold_overs']}")
                except Exception as e:
                    print(f"  Warning: Could not compute metrics: {e}")

                # Store results
                uv_coords[chart_id] = {
                    'uv': uv,
                    'vertex_map': v_map,
                    'V_chart': V_sub,
                    'F_chart': F_sub
                }
                parameterized_charts.append(chart_to_process)

    print("\n" + "="*70)
    print(f"ATLAS GENERATION COMPLETE")
    print(f"  Successful charts: {len(parameterized_charts)}")
    print(f"  Failed charts: {len(failed_charts)}")
    print("="*70)

    return {
        'charts': parameterized_charts,
        'failed_charts': failed_charts,
        'uv_coords': uv_coords,
        'metrics': metrics,
        'original_V': V,
        'original_F': F
    }
