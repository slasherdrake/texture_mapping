"""
As-Rigid-As-Possible (ARAP) mesh parameterization.

Implements the local-global algorithm from:
"As-Rigid-As-Possible Surface Modeling" - Sorkine & Alexa (2007)
Adapted for 2D parameterization.

The algorithm alternates between:
1. LOCAL step: Find best-fit rotations for each triangle (closed-form via SVD)
2. GLOBAL step: Find vertex positions that match these rotations (sparse linear solve)
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from scipy.sparse.linalg import lsqr, spsolve
from .tutte import find_boundary_vertices


def compute_cotangent_weights(V, F):
    """
    Compute cotangent weights for all edges in the mesh.
    
    For edge (i,j), the weight is:
        w_ij = (cot α + cot β) / 2
    where α and β are the angles opposite to edge (i,j).
    
    Parameters
    ----------
    V : (N, 3) array
        3D vertex positions
    F : (M, 3) array
        Face indices
        
    Returns
    -------
    weights : dict
        Dictionary mapping edge tuple (i,j) to cotangent weight
    """
    weights = {}
    
    for face in F:
        i, j, k = face
        
        # Get vertex positions
        vi = V[i]
        vj = V[j]
        vk = V[k]
        
        # Compute edges
        e_ij = vj - vi
        e_jk = vk - vj
        e_ki = vi - vk
        
        # Compute cotangents for each angle
        # Angle at vertex i (opposite edge jk)
        cot_i = compute_cotangent(-e_ki, e_ij)
        
        # Angle at vertex j (opposite edge ki)
        cot_j = compute_cotangent(-e_ij, e_jk)
        
        # Angle at vertex k (opposite edge ij)
        cot_k = compute_cotangent(-e_jk, e_ki)
        
        # Add contributions to edge weights
        # Edge (j,k) opposite to vertex i
        edge_jk = tuple(sorted([j, k]))
        weights[edge_jk] = weights.get(edge_jk, 0.0) + cot_i / 2.0
        
        # Edge (k,i) opposite to vertex j
        edge_ki = tuple(sorted([k, i]))
        weights[edge_ki] = weights.get(edge_ki, 0.0) + cot_j / 2.0
        
        # Edge (i,j) opposite to vertex k
        edge_ij = tuple(sorted([i, j]))
        weights[edge_ij] = weights.get(edge_ij, 0.0) + cot_k / 2.0
    
    # Clamp weights to prevent ill-conditioning
    # Negative/tiny weights can happen with obtuse triangles
    # Very large weights can happen with acute triangles
    for edge in weights:
        weights[edge] = np.clip(weights[edge], 1e-6, 1e6)

    return weights


def compute_cotangent(v1, v2):
    """
    Compute cotangent of angle between two vectors.
    cot(θ) = cos(θ) / sin(θ) = (v1 · v2) / ||v1 × v2||
    
    Parameters
    ----------
    v1, v2 : (3,) arrays
        Input vectors
        
    Returns
    -------
    cot : float
        Cotangent of angle between vectors
    """
    dot = np.dot(v1, v2)
    cross = np.cross(v1, v2)
    cross_norm = np.linalg.norm(cross)
    
    if cross_norm < 1e-10:
        return 0.0
    
    return dot / cross_norm


def compute_local_rotations(V, F, uv):
    """
    LOCAL STEP: Compute optimal rotation for each triangle.
    
    For each triangle, find the 2×2 rotation matrix that best maps
    3D edges to current 2D edges via SVD.
    
    Parameters
    ----------
    V : (N, 3) array
        3D vertex positions
    F : (M, 3) array
        Face indices
    uv : (N, 2) array
        Current 2D positions
        
    Returns
    -------
    rotations : (M, 2, 2) array
        Rotation matrix for each triangle
    """
    n_faces = F.shape[0]
    rotations = np.zeros((n_faces, 2, 2))
    
    for t, face in enumerate(F):
        i, j, k = face
        
        # Get 3D positions (take XY projection for 2D parameterization)
        # Actually, we need to use the 3D positions properly
        vi_3d = V[i]
        vj_3d = V[j]
        vk_3d = V[k]
        
        # Get 2D positions
        ui = uv[i]
        uj = uv[j]
        uk = uv[k]
        
        # Build edge matrices
        # 3D edges (project to local 2D coordinate system)
        # Use first two edges of the triangle
        e1_3d = vj_3d - vi_3d
        e2_3d = vk_3d - vi_3d
        
        # Create local 2D coordinate system for 3D triangle
        # X-axis: along e1
        x_axis = e1_3d / (np.linalg.norm(e1_3d) + 1e-10)
        
        # Y-axis: perpendicular to x in the plane
        normal = np.cross(e1_3d, e2_3d)
        y_axis = np.cross(normal, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-10)
        
        # Project 3D edges to this 2D system
        v1_local = np.array([np.dot(e1_3d, x_axis), np.dot(e1_3d, y_axis)])
        v2_local = np.array([np.dot(e2_3d, x_axis), np.dot(e2_3d, y_axis)])
        
        # 2D edges
        u1 = uj - ui
        u2 = uk - ui
        
        # Build matrices (each column is an edge)
        V_mat = np.column_stack([v1_local, v2_local])  # 2×2
        U_mat = np.column_stack([u1, u2])              # 2×2
        
        # Compute covariance matrix
        try:
            S = U_mat @ V_mat.T
            
            # SVD decomposition
            U_svd, sigma, Vt_svd = np.linalg.svd(S)
            
            # Extract rotation
            R = U_svd @ Vt_svd
            
            # Ensure proper rotation (det = +1)
            if np.linalg.det(R) < 0:
                # Flip sign of last column of U
                U_svd[:, -1] *= -1
                R = U_svd @ Vt_svd
            
            rotations[t] = R
            
        except np.linalg.LinAlgError:
            # Degenerate triangle, use identity
            rotations[t] = np.eye(2)
    
    return rotations


def solve_global_step(V, F, uv, rotations, weights, boundary_vertices):
    """
    GLOBAL STEP: Solve for UV positions with fixed rotations.

    Solves the sparse linear system:
        L * u = b
    where the system enforces that each vertex position minimizes
    distance to rotated neighbors.

    Parameters
    ----------
    V : (N, 3) array
        3D vertex positions
    F : (M, 3) array
        Face indices
    uv : (N, 2) array
        Current UV positions (unused, kept for API consistency)
    rotations : (M, 2, 2) array
        Rotation matrix for each face
    weights : dict
        Edge weights
    boundary_vertices : array
        Indices of boundary vertices to pin

    Returns
    -------
    uv_new : (N, 2) array
        Updated UV positions
    """
    n_verts = V.shape[0]

    # Build a mapping from faces to their local 2D coordinates
    # We need this to consistently map 3D edges to 2D
    face_to_local_coords = {}

    for t, face in enumerate(F):
        i, j, k = face

        vi_3d = V[i]
        vj_3d = V[j]
        vk_3d = V[k]

        # Create local 2D coordinate system for this face
        e1_3d = vj_3d - vi_3d
        e2_3d = vk_3d - vi_3d

        x_axis = e1_3d / (np.linalg.norm(e1_3d) + 1e-10)
        normal = np.cross(e1_3d, e2_3d)
        y_axis = np.cross(normal, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-10)

        face_to_local_coords[t] = (x_axis, y_axis)

    # Build edge to faces mapping
    edge_to_faces = {}
    for t, face in enumerate(F):
        i, j, k = face
        for edge in [(i, j), (j, k), (k, i)]:
            edge_key = tuple(sorted(edge))
            if edge_key not in edge_to_faces:
                edge_to_faces[edge_key] = []
            edge_to_faces[edge_key].append((t, edge))

    # Build sparse linear system
    L = lil_matrix((n_verts, n_verts))
    b_u = np.zeros(n_verts)
    b_v = np.zeros(n_verts)

    # For each edge, add contributions from adjacent faces
    for edge_key, face_list in edge_to_faces.items():
        idx1, idx2 = edge_key
        w = weights.get(edge_key, 1.0)

        # Accumulate rotated edge targets from all adjacent faces
        target_sum = np.zeros(2)

        for t, (orig_i, orig_j) in face_list:
            R = rotations[t]
            x_axis, y_axis = face_to_local_coords[t]

            # Get 3D edge (preserve direction)
            if orig_i == idx1 and orig_j == idx2:
                v_edge_3d = V[idx2] - V[idx1]
                sign = 1.0
            else:  # orig_i == idx2 and orig_j == idx1
                v_edge_3d = V[idx1] - V[idx2]
                sign = -1.0

            # Project to local 2D
            v_edge_2d = np.array([np.dot(v_edge_3d, x_axis),
                                  np.dot(v_edge_3d, y_axis)])

            # Apply rotation
            target_sum += sign * (R @ v_edge_2d)

        # Average over adjacent faces
        target = target_sum / len(face_list)

        # Add to system for vertex idx1
        # Equation: w * (u_i - u_j) = w * target
        L[idx1, idx1] += w
        L[idx1, idx2] -= w
        b_u[idx1] += w * target[0]
        b_v[idx1] += w * target[1]

        # Add to system for vertex idx2 (symmetric)
        # Equation: w * (u_j - u_i) = -w * target
        L[idx2, idx2] += w
        L[idx2, idx1] -= w
        b_u[idx2] -= w * target[0]
        b_v[idx2] -= w * target[1]

    # Pin ALL boundary vertices (hard constraints)
    # Clear only rows (not columns) to maintain coupling with interior vertices
    for b_idx in boundary_vertices:
        L[b_idx, :] = 0
        L[b_idx, b_idx] = 1.0
        b_u[b_idx] = uv[b_idx, 0]
        b_v[b_idx] = uv[b_idx, 1]

    # Convert to CSR for efficient solving
    L_csr = L.tocsr()

    # Try direct solver first (faster and more accurate for well-conditioned systems)
    # Fall back to LSQR for singular/ill-conditioned matrices
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=RuntimeWarning)
            u_new = spsolve(L_csr, b_u)
            v_new = spsolve(L_csr, b_v)

            # Check if solution is valid
            if np.isnan(u_new).any() or np.isinf(u_new).any() or \
               np.isnan(v_new).any() or np.isinf(v_new).any():
                raise ValueError("Direct solver produced NaN/Inf")
    except (RuntimeWarning, ValueError, Exception):
        # Direct solver failed, use LSQR (more robust for ill-conditioned systems)
        u_new = lsqr(L_csr, b_u, atol=1e-8, btol=1e-8)[0]
        v_new = lsqr(L_csr, b_v, atol=1e-8, btol=1e-8)[0]

    uv_new = np.column_stack([u_new, v_new])

    # Check for invalid values
    if np.isnan(uv_new).any() or np.isinf(uv_new).any():
        raise ValueError("ARAP produced invalid UV coordinates (NaN/Inf)")

    return uv_new


def arap_embedding(V, F, initial_uv=None, max_iterations=10, tolerance=1e-5):
    """
    Compute As-Rigid-As-Possible (ARAP) parameterization.
    
    Uses an iterative local-global algorithm:
    1. LOCAL:  Given UV positions, compute optimal rotation per triangle
    2. GLOBAL: Given rotations, compute optimal UV positions
    3. Repeat until convergence
    
    Parameters
    ----------
    V : (N, 3) array
        3D vertex positions
    F : (M, 3) array
        Face indices (triangles)
    initial_uv : (N, 2) array, optional
        Initial UV coordinates (uses Tutte if not provided)
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence tolerance (change in UV positions)
        
    Returns
    -------
    uv : (N, 2) array
        ARAP UV coordinates
        
    Raises
    ------
    ValueError
        If mesh is not valid for parameterization
    """
    n_verts = V.shape[0]
    
    # Validate input
    if V.shape[1] != 3:
        raise ValueError(f"Expected 3D vertices, got shape {V.shape}")
    if F.shape[1] != 3:
        raise ValueError(f"Expected triangular faces, got shape {F.shape}")
    
    # Find boundary vertices
    boundary_vertices, is_boundary = find_boundary_vertices(F, n_verts)
    
    if len(boundary_vertices) == 0:
        raise ValueError("ARAP requires a mesh with boundary (disk topology)")
    
    print(f"ARAP embedding: {n_verts} vertices, {F.shape[0]} faces")
    print(f"  Boundary vertices: {len(boundary_vertices)}")
    
    # Initialize UV coordinates
    # Always use Tutte for initialization (simple, robust, works for all disk topologies)
    if initial_uv is None:
        print("  Initializing with Tutte embedding...")
        from .tutte import tutte_embedding
        uv = tutte_embedding(V, F, weight_type='uniform')
    else:
        uv = initial_uv.copy()
    
    # Compute edge weights
    print("  Computing cotangent weights...")
    weights = compute_cotangent_weights(V, F)

    # Iterative optimization
    print("  Starting ARAP iterations...")
    for iteration in range(max_iterations):
        # LOCAL STEP: Compute optimal rotations
        rotations = compute_local_rotations(V, F, uv)
        
        # GLOBAL STEP: Solve for positions
        uv_new = solve_global_step(V, F, uv, rotations, weights, boundary_vertices)
        
        # Check convergence
        change = np.linalg.norm(uv_new - uv)
        print(f"    Iteration {iteration + 1}: change = {change:.6f}")
        
        if change < tolerance:
            print(f"  Converged after {iteration + 1} iterations")
            break
        
        uv = uv_new
    else:
        print(f"  Reached maximum iterations ({max_iterations})")
    
    print("ARAP embedding complete.")
    return uv
