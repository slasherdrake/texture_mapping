"""
Analytic mesh generators for parameterization testing.
"""

import numpy as np


def uv_sphere(n_lat=40, n_lon=80, radius=1.0):
    """
    Generate a UV sphere mesh with latitude/longitude parameterization.

    The sphere uses duplicated vertices along the longitude seam (-pi and +pi)
    so that UV coordinates can be discontinuous across the seam. This is necessary
    for texture mapping, where u=0 and u=1 should map to different UV coordinates
    even though they represent the same physical location on the sphere.

    Poles are collapsed to a single vertex each to avoid degenerate triangles.

    Parameters
    ----------
    n_lat : int
        Number of latitude divisions (pole to pole)
    n_lon : int
        Number of longitude divisions (around the equator)
    radius : float
        Sphere radius

    Returns
    -------
    V : ndarray, shape (N, 3), dtype float64
        Vertex positions
    F : ndarray, shape (M, 3), dtype int64
        Triangle face indices
    """
    vertices = []
    faces = []

    # North pole - single vertex at the top
    vertices.append([0.0, 0.0, radius])

    # Generate intermediate latitude rings
    # Each ring has n_lon+1 vertices to allow seam duplication
    for i_lat in range(1, n_lat):
        # Latitude angle: 0 at north pole, pi at south pole
        theta = np.pi * i_lat / n_lat
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        z = radius * cos_theta

        for i_lon in range(n_lon + 1):  # +1 for seam duplication
            # Longitude angle: -pi to +pi
            phi = -np.pi + (2.0 * np.pi * i_lon / n_lon)
            x = radius * sin_theta * np.cos(phi)
            y = radius * sin_theta * np.sin(phi)
            vertices.append([x, y, z])

    # South pole - single vertex at the bottom
    south_pole_idx = len(vertices)
    vertices.append([0.0, 0.0, -radius])

    # Build faces
    # North pole cap (counter-clockwise from outside)
    for i_lon in range(n_lon):
        v0 = 0  # North pole
        v1 = 1 + i_lon
        v2 = 1 + (i_lon + 1)
        faces.append([v0, v1, v2])

    # Intermediate quads (split into two triangles each)
    for i_lat in range(n_lat - 2):
        for i_lon in range(n_lon):
            # Current ring starts at vertex (1 + i_lat * (n_lon + 1))
            ring_start = 1 + i_lat * (n_lon + 1)
            next_ring_start = 1 + (i_lat + 1) * (n_lon + 1)

            v0 = ring_start + i_lon
            v1 = ring_start + (i_lon + 1)
            v2 = next_ring_start + (i_lon + 1)
            v3 = next_ring_start + i_lon

            # Two triangles per quad
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    # South pole cap
    last_ring_start = 1 + (n_lat - 2) * (n_lon + 1)
    for i_lon in range(n_lon):
        v0 = last_ring_start + i_lon
        v1 = last_ring_start + (i_lon + 1)
        v2 = south_pole_idx
        faces.append([v0, v1, v2])

    V = np.array(vertices, dtype=np.float64)
    F = np.array(faces, dtype=np.int64)

    return V, F


def hemisphere(n_lat=40, n_lon=80, radius=1.0):
    """
    Generate the upper hemisphere (z >= 0) as an open disk-like surface.

    This produces a mesh suitable for testing planar parameterization methods
    like Tutte embedding and LSCM, which require a disk topology.

    Parameters
    ----------
    n_lat : int
        Number of latitude divisions (pole to equator)
    n_lon : int
        Number of longitude divisions (around the equator)
    radius : float
        Hemisphere radius

    Returns
    -------
    V : ndarray, shape (N, 3), dtype float64
        Vertex positions
    F : ndarray, shape (M, 3), dtype int64
        Triangle face indices
    """
    # Use half the latitude divisions (plus one for equator)
    n_lat_hemi = (n_lat // 2) + 1

    vertices = []
    faces = []

    # North pole - single vertex at the top
    vertices.append([0.0, 0.0, radius])

    # Generate latitude rings down to equator (inclusive)
    for i_lat in range(1, n_lat_hemi):
        # Latitude angle: 0 at north pole, pi/2 at equator
        theta = (np.pi / 2.0) * i_lat / (n_lat_hemi - 1)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        z = radius * cos_theta

        for i_lon in range(n_lon + 1):  # +1 for seam duplication
            # Longitude angle: -pi to +pi
            phi = -np.pi + (2.0 * np.pi * i_lon / n_lon)
            x = radius * sin_theta * np.cos(phi)
            y = radius * sin_theta * np.sin(phi)
            vertices.append([x, y, z])

    # Build faces
    # North pole cap
    for i_lon in range(n_lon):
        v0 = 0  # North pole
        v1 = 1 + i_lon
        v2 = 1 + (i_lon + 1)
        faces.append([v0, v1, v2])

    # Intermediate quads (split into two triangles each)
    for i_lat in range(n_lat_hemi - 2):
        for i_lon in range(n_lon):
            ring_start = 1 + i_lat * (n_lon + 1)
            next_ring_start = 1 + (i_lat + 1) * (n_lon + 1)

            v0 = ring_start + i_lon
            v1 = ring_start + (i_lon + 1)
            v2 = next_ring_start + (i_lon + 1)
            v3 = next_ring_start + i_lon

            # Two triangles per quad
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    V = np.array(vertices, dtype=np.float64)
    F = np.array(faces, dtype=np.int64)

    return V, F


def torus(n_major=40, n_minor=20, R=1.0, r=0.3):
    """
    Generate a torus mesh (donut shape).

    The torus is parameterized by major radius R and minor radius r.
    Major angle (around the hole) goes 0 to 2π, minor angle (around
    the tube) goes 0 to 2π.

    Parameters
    ----------
    n_major : int
        Number of divisions around the major circle (hole)
    n_minor : int
        Number of divisions around the minor circle (tube)
    R : float
        Major radius (distance from center to tube center)
    r : float
        Minor radius (tube thickness)

    Returns
    -------
    V : ndarray, shape (N, 3), dtype float64
        Vertex positions
    F : ndarray, shape (M, 3), dtype int64
        Triangle face indices
    """
    vertices = []
    faces = []

    # Generate vertices
    for i in range(n_major):
        theta = 2.0 * np.pi * i / n_major  # Major angle

        for j in range(n_minor):
            phi = 2.0 * np.pi * j / n_minor  # Minor angle

            # Torus equation
            x = (R + r * np.cos(phi)) * np.cos(theta)
            y = (R + r * np.cos(phi)) * np.sin(theta)
            z = r * np.sin(phi)

            vertices.append([x, y, z])

    # Generate faces (quads split into triangles)
    for i in range(n_major):
        i_next = (i + 1) % n_major  # Wrap around

        for j in range(n_minor):
            j_next = (j + 1) % n_minor  # Wrap around

            # Vertex indices for the quad
            v0 = i * n_minor + j
            v1 = i * n_minor + j_next
            v2 = i_next * n_minor + j_next
            v3 = i_next * n_minor + j

            # Two triangles per quad
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    V = np.array(vertices, dtype=np.float64)
    F = np.array(faces, dtype=np.int64)

    return V, F


def disk(n_radial=20, n_angular=40, radius=1.0):
    """
    Generate a planar disk mesh (disk topology, good for Tutte/LSCM testing).

    The disk lies in the XY plane (z=0) and is tessellated with triangles
    radiating from the center.

    Parameters
    ----------
    n_radial : int
        Number of radial divisions (center to edge)
    n_angular : int
        Number of angular divisions (around the circle)
    radius : float
        Disk radius

    Returns
    -------
    V : ndarray, shape (N, 3), dtype float64
        Vertex positions (z=0 for all vertices)
    F : ndarray, shape (M, 3), dtype int64
        Triangle face indices
    """
    vertices = []
    faces = []

    # Center vertex
    vertices.append([0.0, 0.0, 0.0])

    # Generate concentric rings
    for i in range(1, n_radial + 1):
        r = radius * i / n_radial

        for j in range(n_angular):
            theta = 2.0 * np.pi * j / n_angular
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            vertices.append([x, y, 0.0])

    # Generate faces
    # Center triangles (from center to first ring)
    for j in range(n_angular):
        j_next = (j + 1) % n_angular
        v0 = 0  # Center
        v1 = 1 + j
        v2 = 1 + j_next
        faces.append([v0, v1, v2])

    # Ring-to-ring quads (split into triangles)
    for i in range(n_radial - 1):
        ring_start = 1 + i * n_angular
        next_ring_start = 1 + (i + 1) * n_angular

        for j in range(n_angular):
            j_next = (j + 1) % n_angular

            v0 = ring_start + j
            v1 = ring_start + j_next
            v2 = next_ring_start + j_next
            v3 = next_ring_start + j

            # Two triangles per quad
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    V = np.array(vertices, dtype=np.float64)
    F = np.array(faces, dtype=np.int64)

    return V, F
