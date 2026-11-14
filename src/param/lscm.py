"""Least Squares Conformal Maps, assembled from triangle gradients.

Reference: Lévy et al., "Least Squares Conformal Maps for Automatic Texture
Atlas Generation", SIGGRAPH 2002, https://brunolevy.github.io/papers/LSCM_SIGGRAPH_2002.pdf
Only the sparse least-squares solve is delegated to SciPy.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr


def _validate_mesh(V, F):
    """Check an oriented, connected triangulated disk; return its boundary."""
    if V.ndim != 2 or V.shape[1] != 3 or len(V) < 3 or not np.isfinite(V).all():
        raise ValueError("V must contain at least three finite 3D vertices.")
    if (F.ndim != 2 or F.shape[1] != 3 or not len(F)
            or not np.issubdtype(F.dtype, np.integer)):
        raise ValueError("F must be a nonempty (M, 3) integer array.")
    if F.min() < 0 or F.max() >= len(V):
        raise ValueError("Face indices are out of range.")
    if len(np.unique(F)) != len(V):
        raise ValueError("Mesh contains unused vertices.")
    if len(np.unique(np.sort(F, axis=1), axis=0)) != len(F):
        raise ValueError("Mesh contains duplicate faces.")

    edges = np.concatenate((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))
    undirected = np.sort(edges, axis=1)
    unique, inverse, counts = np.unique(
        undirected, axis=0, return_inverse=True, return_counts=True)
    if np.any(unique[:, 0] == unique[:, 1]):
        raise ValueError("Mesh contains degenerate faces with repeated vertices.")
    if np.any(counts > 2):
        raise ValueError("Mesh has nonmanifold edges.")
    direction = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    winding = np.bincount(inverse, weights=direction)
    if np.any(winding[counts == 2] != 0):
        raise ValueError("Faces must have consistent winding across shared edges.")

    graph = coo_matrix((np.ones(len(unique)), (unique[:, 0], unique[:, 1])),
                       shape=(len(V), len(V))).tocsr()
    if connected_components(graph, directed=False, return_labels=False) != 1:
        raise ValueError("LSCM requires a connected mesh.")
    boundary_edges = unique[counts == 1]
    if not len(boundary_edges):
        raise ValueError("Mesh has no boundary; cut closed meshes to a disk first.")
    boundary, degree = np.unique(boundary_edges, return_counts=True)
    if np.any(degree != 2) or len(V) - len(unique) + len(F) != 1:
        raise ValueError("LSCM requires disk topology with a single boundary loop.")
    boundary_graph = coo_matrix(
        (np.ones(len(boundary_edges)), (boundary_edges[:, 0], boundary_edges[:, 1])),
        shape=(len(V), len(V))).tocsr()[boundary][:, boundary]
    if connected_components(boundary_graph, directed=False, return_labels=False) != 1:
        raise ValueError("LSCM requires a single boundary loop.")
    return boundary


def lscm_embedding(V, F, pinned_vertices=None, pinned_uv=None):
    """Parameterize an oriented triangle disk with a free boundary.

    Parameters
    ----------
    V : (N, 3) array
        Finite vertex positions; every vertex must belong to the disk.
    F : (M, 3) integer array
        Consistently oriented, nondegenerate triangles.
    pinned_vertices : (2,) integer array, optional
        Two distinct boundary vertices. By default, two Euclidean farthest-point
        sweeps select well-separated boundary vertices in O(boundary size).
    pinned_uv : (2, 2) array, optional
        Distinct fixed coordinates, defaulting to ((0, 0), (1, 0)).

    Returns
    -------
    UV : (N, 2) array
        UV coordinates with the pins preserved exactly. Coordinates are not
        clamped or rescaled to a unit square, which would change the constraints.

    Notes
    -----
    Each triangle contributes area-weighted Cauchy-Riemann residuals
    sqrt(area) * (u_x - v_y, u_y + v_x). The four pinned variables
    are eliminated before solving the rectangular sparse least-squares system.
    LSCM minimizes conformal energy; it does not enforce area preservation or
    an injective map on arbitrary inputs.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F)
    boundary = _validate_mesh(V, F)
    n = len(V)

    # Global scaling cancels from the energy and keeps triangle arithmetic stable.
    geometry_scale = np.max(np.abs(V - V[0]))
    if not np.isfinite(geometry_scale) or geometry_scale == 0:
        raise ValueError("Mesh has degenerate geometry.")
    points = (V - V[0]) / geometry_scale
    if pinned_vertices is None:
        boundary_points = points[boundary]
        first = np.argmax(np.linalg.norm(boundary_points - boundary_points[0], axis=1))
        second = np.argmax(np.linalg.norm(boundary_points - boundary_points[first], axis=1))
        pins = boundary[[first, second]]
    else:
        pins = np.asarray(pinned_vertices)
    if (pins.shape != (2,) or not np.issubdtype(pins.dtype, np.integer)
            or pins[0] == pins[1] or not np.isin(pins, boundary).all()):
        raise ValueError("Choose two distinct boundary vertices as pins.")
    if np.linalg.norm(points[pins[0]] - points[pins[1]]) <= 1e-12:
        raise ValueError("Pinned vertices must be geometrically separated.")
    targets = np.asarray(((0., 0.), (1., 0.)) if pinned_uv is None else pinned_uv,
                         dtype=np.float64)
    if (targets.shape != (2, 2) or not np.isfinite(targets).all()
            or np.linalg.norm(targets[1] - targets[0]) == 0):
        raise ValueError("pinned_uv must contain two distinct finite 2D coordinates.")

    triangles = points[F]
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    length = np.linalg.norm(e1, axis=1)
    double_area = np.linalg.norm(np.cross(e1, e2), axis=1)
    edge_scale = np.maximum(length**2, np.sum(e2**2, axis=1))
    if np.any(double_area <= 1e-14 * edge_scale) or np.any(length == 0):
        raise ValueError("Mesh contains degenerate or numerically singular triangles.")

    # Local triangle coordinates: (0,0), (length,0), (x,y), y > 0.
    x = np.sum(e1 * e2, axis=1) / length
    y = double_area / length
    gx = np.column_stack((-1 / length, 1 / length, np.zeros(len(F))))
    gy = np.column_stack(((x / length - 1) / y, -x / (length * y), 1 / y))
    weight = np.sqrt(double_area / 2)[:, None]
    gx *= weight
    gy *= weight
    rows = np.repeat(np.arange(len(F)), 3)
    cols = F.ravel()
    m = len(F)
    matrix = coo_matrix((
        np.concatenate((gx.ravel(), -gy.ravel(), gy.ravel(), gx.ravel())),
        (np.concatenate((rows, rows, rows + m, rows + m)),
         np.concatenate((cols, cols + n, cols, cols + n)))),
        shape=(2 * m, 2 * n)).tocsr()

    fixed = np.concatenate((pins, pins + n))
    # Solve relative to the first pin to avoid losing precision to UV translation.
    values = (targets - targets[0]).T.ravel()
    free = np.ones(2 * n, dtype=bool)
    free[fixed] = False
    rhs = -matrix[:, fixed] @ values
    result = lsqr(matrix[:, free], rhs, atol=1e-11, btol=1e-11,
                  conlim=1e12, iter_lim=max(1000, 10 * n))
    if result[1] not in (0, 1, 2, 4, 5) or not np.isfinite(result[0]).all():
        raise ValueError("LSCM least-squares solve did not converge; check mesh quality and pins.")
    solution = np.empty(2 * n)
    solution[fixed] = values
    solution[free] = result[0]
    UV = solution.reshape(2, n).T + targets[0]
    UV[pins] = targets
    return UV
