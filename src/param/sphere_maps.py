"""
Analytic parameterization methods for spherical surfaces.

All functions assume input vertices V lie on a unit sphere (or will be
normalized to the unit sphere). They return UV coordinates of shape (N, 2).
"""

import numpy as np


def equirectangular(V):
    """
    Equirectangular (plate carree) projection: (longitude, latitude).

    Maps longitude to u in [-pi, pi] and latitude to v in [-pi/2, pi/2].
    This is the simplest spherical projection but introduces severe distortion
    near the poles where meridians converge.

    Parameters
    ----------
    V : ndarray, shape (N, 3)
        Vertex positions, assumed to lie on or near the unit sphere

    Returns
    -------
    UV : ndarray, shape (N, 2)
        UV coordinates with u = longitude, v = latitude
    """
    # Normalize to unit sphere
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)  # Avoid division by zero
    V_normalized = V / norms

    x, y, z = V_normalized[:, 0], V_normalized[:, 1], V_normalized[:, 2]

    # Longitude: arctan2(y, x) gives angle in [-pi, pi]
    u = np.arctan2(y, x)

    # Latitude: arcsin(z) gives angle in [-pi/2, pi/2]
    # Clamp z to [-1, 1] to handle numerical errors
    z_clamped = np.clip(z, -1.0, 1.0)
    v = np.arcsin(z_clamped)

    UV = np.column_stack([u, v])
    return UV


def mercator(V):
    """
    Mercator projection: (longitude, ln(tan(pi/4 + latitude/2))).

    Preserves angles (conformal) but severely distorts area near poles.
    The poles themselves map to +/- infinity, so we clamp latitude to avoid
    numerical overflow. Latitudes beyond +/- 85 degrees are clamped.

    Parameters
    ----------
    V : ndarray, shape (N, 3)
        Vertex positions, assumed to lie on or near the unit sphere

    Returns
    -------
    UV : ndarray, shape (N, 2)
        UV coordinates with u = longitude, v = mercator(latitude)
    """
    # Normalize to unit sphere
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    V_normalized = V / norms

    x, y, z = V_normalized[:, 0], V_normalized[:, 1], V_normalized[:, 2]

    # Longitude: arctan2(y, x) in [-pi, pi]
    u = np.arctan2(y, x)

    # Latitude: arcsin(z) in [-pi/2, pi/2]
    z_clamped = np.clip(z, -1.0, 1.0)
    lat = np.arcsin(z_clamped)

    # Clamp latitude to avoid poles (tan(pi/2) = infinity)
    # +/- 85 degrees = +/- 1.48353 radians (out of pi/2 = 1.5708)
    max_lat = np.radians(85.0)
    lat = np.clip(lat, -max_lat, max_lat)

    # Mercator formula: ln(tan(pi/4 + lat/2))
    v = np.log(np.tan(np.pi / 4.0 + lat / 2.0))

    UV = np.column_stack([u, v])
    return UV


def stereographic(V):
    """
    Stereographic projection from the north pole onto the plane z = 0.

    Projects each point on the sphere from the north pole (0, 0, 1) through
    the point onto the equatorial plane z = 0. The north pole itself is
    handled explicitly by mapping it to the origin (though this creates a
    discontinuity).

    This projection is conformal (preserves angles) and maps the sphere minus
    the north pole onto the entire plane.

    Parameters
    ----------
    V : ndarray, shape (N, 3)
        Vertex positions, assumed to lie on or near the unit sphere

    Returns
    -------
    UV : ndarray, shape (N, 2)
        UV coordinates (u, v) on the equatorial plane
    """
    # Normalize to unit sphere
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    V_normalized = V / norms

    x, y, z = V_normalized[:, 0], V_normalized[:, 1], V_normalized[:, 2]

    # Stereographic projection formula from north pole:
    # (u, v) = (x, y) / (1 - z)
    # When z is close to 1 (near north pole), this approaches infinity.
    # We handle this by clamping the denominator.

    denom = 1.0 - z
    # Clamp denominator to avoid division by zero at north pole
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)

    u = x / denom
    v = y / denom

    UV = np.column_stack([u, v])
    return UV
