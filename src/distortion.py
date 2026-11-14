"""
Distortion metrics for surface parameterizations.

Given a mesh with vertices V and faces F, and UV coordinates for each vertex,
compute per-triangle distortion metrics based on the Jacobian of the 3D→UV mapping.

Theory:
-------
For each triangle, we compute the Jacobian J of the mapping from the triangle's
local 2D coordinate system (in the 3D surface) to UV space.

The singular values σ₁ ≥ σ₂ of J tell us how the mapping stretches/compresses:
  - σ₁: maximum stretch factor
  - σ₂: minimum stretch factor

Distortion metrics:
  - Angle distortion = σ₁/σ₂  (conformal when = 1, preserves angles)
  - Area distortion = σ₁×σ₂   (equiareal when = 1, preserves area)
  - Fold-over: det(J) < 0     (triangle flips orientation)

References:
-----------
Sander et al., "Texture Mapping Progressive Meshes", SIGGRAPH 2001
Hormann et al., "Mesh Parameterization: Theory and Practice", SIGGRAPH 2007
"""

import numpy as np
from numpy.linalg import svd, det, norm


def compute_triangle_jacobian(v0, v1, v2, uv0, uv1, uv2):
    """
    Compute the 2×2 Jacobian matrix for a single triangle.

    The Jacobian J maps from the triangle's local 2D coordinate system
    (embedded in 3D) to UV space.

    Parameters
    ----------
    v0, v1, v2 : array-like, shape (3,)
        3D vertex positions of the triangle
    uv0, uv1, uv2 : array-like, shape (2,)
        UV coordinates of the triangle vertices

    Returns
    -------
    J : ndarray, shape (2, 2)
        Jacobian matrix of the 3D→UV mapping

    Algorithm
    ---------
    1. Build a local 2D coordinate system for the triangle:
       - Origin at v0
       - e1 = (v1 - v0) / ||v1 - v0||  (first edge, normalized)
       - e2 = perpendicular to e1 in the triangle's plane

    2. Express v1, v2 in this local system:
       - v1_local = (||v1-v0||, 0)
       - v2_local = (v2·e1, v2·e2)  where v2' = v2 - v0

    3. Compute UV deltas:
       - du = uv1 - uv0, uv2 - uv0

    4. The Jacobian maps local triangle coords to UV:
       J = [du₁ du₂] · [v1_local v2_local]⁻¹
    """
    # Convert to numpy arrays
    v0 = np.asarray(v0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    uv0 = np.asarray(uv0, dtype=np.float64)
    uv1 = np.asarray(uv1, dtype=np.float64)
    uv2 = np.asarray(uv2, dtype=np.float64)

    # Step 1: Build local 2D coordinate system in the triangle's plane
    # Edge vectors from v0
    edge1 = v1 - v0  # First edge
    edge2 = v2 - v0  # Second edge

    # First basis vector: normalized first edge
    e1_length = norm(edge1)
    if e1_length < 1e-12:
        # Degenerate triangle (zero area)
        return np.zeros((2, 2))

    e1 = edge1 / e1_length

    # Second basis vector: perpendicular to e1 in the triangle's plane
    # Project edge2 onto e1, then subtract to get perpendicular component
    edge2_along_e1 = np.dot(edge2, e1) * e1
    e2_unnorm = edge2 - edge2_along_e1
    e2_length = norm(e2_unnorm)

    if e2_length < 1e-12:
        # Degenerate triangle (v0, v1, v2 are collinear)
        return np.zeros((2, 2))

    e2 = e2_unnorm / e2_length

    # Step 2: Express v1, v2 in local 2D coordinates
    # v1 in local coords: it lies along e1, so coords are (e1_length, 0)
    v1_local = np.array([e1_length, 0.0])

    # v2 in local coords: project onto e1 and e2
    v2_local = np.array([np.dot(edge2, e1), np.dot(edge2, e2)])

    # Step 3: UV deltas
    du1 = uv1 - uv0  # UV delta for edge 1
    du2 = uv2 - uv0  # UV delta for edge 2

    # Step 4: Compute Jacobian
    # We have:
    #   [du1]   [v1_local]
    #   [du2] = J · [v2_local]
    #
    # So: J = [du1 du2] · [v1_local v2_local]⁻¹

    # Build the 2×2 matrix [v1_local, v2_local] (column vectors)
    Ss = np.column_stack([v1_local, v2_local])

    # Build the 2×2 matrix [du1, du2] (column vectors)
    St = np.column_stack([du1, du2])

    # Check if Ss is invertible (non-degenerate triangle)
    if abs(det(Ss)) < 1e-12:
        # Degenerate triangle
        return np.zeros((2, 2))

    # J = St · Ss⁻¹
    J = St @ np.linalg.inv(Ss)

    return J


def compute_triangle_area_signed(p0, p1, p2):
    """
    Compute signed area of a 2D triangle.

    Parameters
    ----------
    p0, p1, p2 : array-like, shape (2,)
        2D coordinates of triangle vertices

    Returns
    -------
    area : float
        Signed area (positive if CCW, negative if CW)
    """
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    # Signed area = 0.5 * det([p1-p0, p2-p0])
    v1 = p1 - p0
    v2 = p2 - p0
    return 0.5 * (v1[0] * v2[1] - v1[1] * v2[0])


def compute_distortion_metrics(V, F, UV):
    """
    Compute per-triangle distortion metrics for a parameterized mesh.

    Parameters
    ----------
    V : ndarray, shape (N, 3)
        3D vertex positions
    F : ndarray, shape (M, 3)
        Triangle face indices
    UV : ndarray, shape (N, 2)
        UV coordinates for each vertex

    Returns
    -------
    metrics : dict
        Dictionary with per-triangle arrays:
        - 'sigma_max' : Maximum singular value (σ₁) for each triangle
        - 'sigma_min' : Minimum singular value (σ₂) for each triangle
        - 'angle_distortion' : σ₁/σ₂ (conformal when = 1)
        - 'area_distortion' : σ₁×σ₂ (equiareal when = 1)
        - 'determinant' : det(J) for each triangle
        - 'fold_overs' : Boolean array, True where orientation is flipped
        - 'jacobians' : List of 2×2 Jacobian matrices

    Notes
    -----
    - angle_distortion = 1 means angles are preserved (conformal)
    - area_distortion = 1 means area is preserved (equiareal)
    - fold_over = True means the triangle's orientation flipped
    - Degenerate triangles return NaN for distortion metrics
    - Fold-over detection: compares sign of 3D triangle area vs UV triangle area
    """
    n_faces = F.shape[0]

    # Storage for results
    sigma_max = np.zeros(n_faces)
    sigma_min = np.zeros(n_faces)
    angle_distortion = np.zeros(n_faces)
    area_distortion = np.zeros(n_faces)
    determinants = np.zeros(n_faces)
    fold_overs = np.zeros(n_faces, dtype=bool)
    jacobians = []

    # Process each triangle
    for i, face in enumerate(F):
        # Get 3D vertices
        v0, v1, v2 = V[face[0]], V[face[1]], V[face[2]]

        # Get UV coordinates
        uv0, uv1, uv2 = UV[face[0]], UV[face[1]], UV[face[2]]

        # Compute Jacobian for this triangle
        J = compute_triangle_jacobian(v0, v1, v2, uv0, uv1, uv2)
        jacobians.append(J)

        # Compute singular values using SVD: J = U · Σ · V^T
        # Σ contains the singular values σ₁ ≥ σ₂ ≥ 0
        try:
            U, s, Vt = svd(J)
            # s is a 1D array of singular values in descending order
            s1, s2 = s[0], s[1]

            sigma_max[i] = s1
            sigma_min[i] = s2

            # Angle distortion (conformal distortion)
            # σ₁/σ₂ = 1 means conformal (angles preserved)
            if s2 > 1e-12:
                angle_distortion[i] = s1 / s2
            else:
                angle_distortion[i] = np.nan

            # Area distortion (equiareal distortion)
            # σ₁×σ₂ = 1 means equiareal (area preserved)
            area_distortion[i] = s1 * s2

            # Determinant (orientation)
            det_J = det(J)
            determinants[i] = det_J

            # Fold-over check: The correct way to detect fold-overs is to check if
            # the orientation changes between 3D and UV. For a mesh with consistent
            # winding order, we need to verify that triangles maintain their orientation.
            #
            # Compute both 3D and UV triangle areas with their signs
            area_3d_signed = compute_triangle_area_signed(v0[:2], v1[:2], v2[:2])
            area_uv_signed = compute_triangle_area_signed(uv0, uv1, uv2)

            # For 3D meshes not in a plane, use the z-component of the normal
            edge1_3d = v1 - v0
            edge2_3d = v2 - v0
            normal_3d = np.cross(edge1_3d, edge2_3d)

            # Use z-component if mesh is approximately planar, else use full normal
            if np.max(np.abs(V[:, 2])) < 1e-6:  # Planar mesh in XY
                sign_3d = np.sign(area_3d_signed)
            else:
                # For curved meshes, we need to determine orientation differently
                # The Jacobian determinant tells us if orientation is preserved
                # We'll consider the first triangle's det(J) sign as the reference
                sign_3d = np.sign(normal_3d[2]) if abs(normal_3d[2]) > 1e-12 else np.sign(det_J)

            sign_uv = np.sign(area_uv_signed)

            # Fold-over occurs when signs don't match
            # But we need to account for the fact that the entire mesh might have
            # one consistent orientation. Better: just check if UV area is negative
            # when we expect positive (or vice versa) based on det(J)
            fold_overs[i] = (sign_3d * sign_uv < 0) if (abs(sign_3d) > 0 and abs(sign_uv) > 0) else False

        except np.linalg.LinAlgError:
            # Singular matrix (degenerate triangle)
            sigma_max[i] = np.nan
            sigma_min[i] = np.nan
            angle_distortion[i] = np.nan
            area_distortion[i] = np.nan
            determinants[i] = 0.0
            fold_overs[i] = False

    return {
        'sigma_max': sigma_max,
        'sigma_min': sigma_min,
        'angle_distortion': angle_distortion,
        'area_distortion': area_distortion,
        'determinant': determinants,
        'fold_overs': fold_overs,
        'jacobians': jacobians,
        'num_fold_overs': np.sum(fold_overs),
        'mean_angle_distortion': np.nanmean(angle_distortion),
        'mean_area_distortion': np.nanmean(area_distortion),
        'max_angle_distortion': np.nanmax(angle_distortion),
        'max_area_distortion': np.nanmax(area_distortion),
    }


def print_distortion_summary(metrics, method_name="Unknown"):
    """
    Print a human-readable summary of distortion metrics.

    Parameters
    ----------
    metrics : dict
        Output from compute_distortion_metrics
    method_name : str
        Name of the parameterization method (for display)
    """
    print(f"\n{'='*60}")
    print(f"Distortion Analysis: {method_name}")
    print(f"{'='*60}")
    print(f"Triangles analyzed: {len(metrics['angle_distortion'])}")
    print(f"\nAngle Distortion (σ₁/σ₂, conformal when = 1):")
    print(f"  Mean:   {metrics['mean_angle_distortion']:.4f}")
    print(f"  Max:    {metrics['max_angle_distortion']:.4f}")
    print(f"\nArea Distortion (σ₁×σ₂, equiareal when = 1):")
    print(f"  Mean:   {metrics['mean_area_distortion']:.4f}")
    print(f"  Max:    {metrics['max_area_distortion']:.4f}")
    print(f"\nFold-overs (det(J) < 0):")
    print(f"  Count:  {metrics['num_fold_overs']}")
    print(f"  Percent: {100*metrics['num_fold_overs']/len(metrics['fold_overs']):.2f}%")
    print(f"{'='*60}\n")
