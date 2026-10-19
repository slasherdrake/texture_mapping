"""
Tests for mesh generation functions in src/shapes.py
"""

import numpy as np
import pytest

from src.shapes import uv_sphere, hemisphere
from src.param.sphere_maps import equirectangular, mercator, stereographic


class TestUVSphere:
    """Tests for uv_sphere function"""

    def test_vertices_on_unit_sphere(self):
        """All vertices should have norm 1 (within tolerance)"""
        V, F = uv_sphere(n_lat=20, n_lon=40, radius=1.0)
        norms = np.linalg.norm(V, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-9), \
            f"Vertices not on unit sphere: min={norms.min()}, max={norms.max()}"

    def test_scaled_sphere(self):
        """Vertices should scale correctly with radius"""
        radius = 2.5
        V, F = uv_sphere(n_lat=20, n_lon=40, radius=radius)
        norms = np.linalg.norm(V, axis=1)
        assert np.allclose(norms, radius, atol=1e-9), \
            f"Vertices not on sphere of radius {radius}"

    def test_no_out_of_range_indices(self):
        """All face indices should reference valid vertices"""
        V, F = uv_sphere(n_lat=20, n_lon=40, radius=1.0)
        n_verts = V.shape[0]
        assert F.min() >= 0, "Negative vertex index found"
        assert F.max() < n_verts, \
            f"Face references vertex {F.max()} but only {n_verts} vertices exist"

    def test_no_degenerate_faces(self):
        """No face should have zero area (degenerate triangle)"""
        V, F = uv_sphere(n_lat=20, n_lon=40, radius=1.0)

        for i, face in enumerate(F):
            v0, v1, v2 = V[face[0]], V[face[1]], V[face[2]]
            # Compute face area using cross product
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(cross)
            assert area > 1e-10, \
                f"Degenerate face {i} with area {area}: vertices {face}"

    def test_correct_vertex_count(self):
        """Check that vertex count matches expected formula"""
        n_lat, n_lon = 20, 40
        V, F = uv_sphere(n_lat=n_lat, n_lon=n_lon, radius=1.0)

        # Expected: 2 poles + (n_lat - 1) rings * (n_lon + 1) vertices per ring
        expected_verts = 2 + (n_lat - 1) * (n_lon + 1)
        assert V.shape[0] == expected_verts, \
            f"Expected {expected_verts} vertices, got {V.shape[0]}"


class TestHemisphere:
    """Tests for hemisphere function"""

    def test_vertices_on_unit_sphere(self):
        """All vertices should have norm 1 (within tolerance)"""
        V, F = hemisphere(n_lat=20, n_lon=40, radius=1.0)
        norms = np.linalg.norm(V, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-9), \
            f"Vertices not on unit sphere: min={norms.min()}, max={norms.max()}"

    def test_all_vertices_in_upper_hemisphere(self):
        """All vertices should have z >= 0"""
        V, F = hemisphere(n_lat=20, n_lon=40, radius=1.0)
        assert np.all(V[:, 2] >= -1e-9), \
            f"Found vertices below equator: min z = {V[:, 2].min()}"

    def test_no_degenerate_faces(self):
        """No face should have zero area"""
        V, F = hemisphere(n_lat=20, n_lon=40, radius=1.0)

        for i, face in enumerate(F):
            v0, v1, v2 = V[face[0]], V[face[1]], V[face[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(cross)
            assert area > 1e-10, \
                f"Degenerate face {i} with area {area}"


class TestEquirectangular:
    """Tests for equirectangular projection"""

    def test_uv_ranges(self):
        """UV coordinates should be in expected ranges"""
        V, _ = uv_sphere(n_lat=20, n_lon=40, radius=1.0)
        UV = equirectangular(V)

        # u (longitude) should be in [-pi, pi]
        assert np.all(UV[:, 0] >= -np.pi - 1e-6), \
            f"u values below -pi: min={UV[:, 0].min()}"
        assert np.all(UV[:, 0] <= np.pi + 1e-6), \
            f"u values above pi: max={UV[:, 0].max()}"

        # v (latitude) should be in [-pi/2, pi/2]
        assert np.all(UV[:, 1] >= -np.pi/2 - 1e-6), \
            f"v values below -pi/2: min={UV[:, 1].min()}"
        assert np.all(UV[:, 1] <= np.pi/2 + 1e-6), \
            f"v values above pi/2: max={UV[:, 1].max()}"

    def test_north_pole(self):
        """North pole should map to v = pi/2"""
        V = np.array([[0.0, 0.0, 1.0]])
        UV = equirectangular(V)
        assert np.isclose(UV[0, 1], np.pi/2, atol=1e-6), \
            f"North pole latitude should be pi/2, got {UV[0, 1]}"

    def test_south_pole(self):
        """South pole should map to v = -pi/2"""
        V = np.array([[0.0, 0.0, -1.0]])
        UV = equirectangular(V)
        assert np.isclose(UV[0, 1], -np.pi/2, atol=1e-6), \
            f"South pole latitude should be -pi/2, got {UV[0, 1]}"


class TestMercator:
    """Tests for Mercator projection"""

    def test_output_shape(self):
        """Output should have correct shape"""
        V, _ = uv_sphere(n_lat=20, n_lon=40, radius=1.0)
        UV = mercator(V)
        assert UV.shape == (V.shape[0], 2), \
            f"Expected shape {(V.shape[0], 2)}, got {UV.shape}"

    def test_no_infinities(self):
        """Mercator should not produce infinite values (poles clamped)"""
        V, _ = uv_sphere(n_lat=20, n_lon=40, radius=1.0)
        UV = mercator(V)
        assert np.all(np.isfinite(UV)), \
            "Mercator produced non-finite values"


class TestStereographic:
    """Tests for stereographic projection"""

    def test_output_shape(self):
        """Output should have correct shape"""
        V, _ = uv_sphere(n_lat=20, n_lon=40, radius=1.0)
        UV = stereographic(V)
        assert UV.shape == (V.shape[0], 2), \
            f"Expected shape {(V.shape[0], 2)}, got {UV.shape}"

    def test_no_infinities(self):
        """Stereographic should not produce infinite values"""
        V, _ = uv_sphere(n_lat=20, n_lon=40, radius=1.0)
        UV = stereographic(V)
        assert np.all(np.isfinite(UV)), \
            "Stereographic produced non-finite values"

    def test_equator_mapping(self):
        """Points on equator should map reasonably"""
        V = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        UV = stereographic(V)
        # Equator points (z=0) should map to finite values
        assert np.all(np.isfinite(UV)), \
            "Equator points should map to finite values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])