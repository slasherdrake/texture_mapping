"""Analytic, numerical, and viewer integration checks for LSCM."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.distortion import compute_distortion_metrics, compute_triangle_jacobian
from src.param.lscm import lscm_embedding
from src.param.tutte import tutte_embedding
from src.shapes import disk, hemisphere


def rectangle():
    # Irregular triangles; the interior point must also be solved for.
    return (np.array([[0., 0., 0.], [3., 0., 0.], [3., 1., 0.],
                      [0., 1., 0.], [.7, .4, 0.]]),
            np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]))


def conformal_energy(V, F, UV):
    energy = 0.
    for face in F:
        a, b, c = V[face]
        J = compute_triangle_jacobian(a, b, c, *UV[face])
        area = np.linalg.norm(np.cross(b - a, c - a)) / 2
        energy += area * ((J[0, 0] - J[1, 1])**2 + (J[0, 1] + J[1, 0])**2)
    return energy


def test_planar_patch_recovers_similarity_and_free_boundary():
    V, F = rectangle()
    UV = lscm_embedding(V, F, pinned_vertices=[0, 1])
    np.testing.assert_allclose(UV, V[:, :2] / 3, atol=1e-9)
    np.testing.assert_array_equal(UV[[0, 1]], [[0., 0.], [1., 0.]])


def test_custom_pins_rotation_translation_and_geometry_scale():
    V, F = rectangle()
    targets = np.array([[7., -2.], [7., 4.]])
    expected = np.column_stack((7 - 2 * V[:, 1], -2 + 2 * V[:, 0]))
    for scale in (1e-9, 1., 1e9):
        # Rotate the 3D surface into the YZ plane.
        rotated = V[:, [2, 0, 1]] * scale
        UV = lscm_embedding(rotated, F, [0, 1], targets)
        np.testing.assert_allclose(UV, expected, atol=1e-8)
        np.testing.assert_array_equal(UV[[0, 1]], targets)


def test_single_triangle():
    V = np.array([[0., 0., 0.], [2., 0., 0.], [.5, 1., 0.]])
    UV = lscm_embedding(V, [[0, 1, 2]], [0, 1])
    np.testing.assert_allclose(UV, V[:, :2] / 2, atol=1e-9)


def test_automatic_pins_on_planar_disk():
    V, F = disk(n_radial=5, n_angular=16)
    UV = lscm_embedding(V, F)
    metrics = compute_distortion_metrics(V, F, UV)
    np.testing.assert_allclose(metrics['angle_distortion'], 1., atol=1e-7)
    assert metrics['num_fold_overs'] == 0
    assert np.isfinite(UV).all()


def test_curved_patch_minimizes_energy_with_same_pins_as_tutte():
    V, F = hemisphere(n_lat=12, n_lon=20)
    tutte = tutte_embedding(V, F)
    # Two well-separated equator vertices (off the duplicated seam).
    pins = [len(V) - 18, len(V) - 8]
    UV = lscm_embedding(V, F, pins, tutte[pins])
    np.testing.assert_array_equal(UV[pins], tutte[pins])
    assert conformal_energy(V, F, UV) < conformal_energy(V, F, tutte)
    assert np.all(compute_distortion_metrics(V, F, UV)['determinant'] > 0)
    # A perturbation of the free variables cannot improve the optimum.
    perturbation = np.random.default_rng(0).normal(size=UV.shape) * .001
    perturbation[pins] = 0
    assert conformal_energy(V, F, UV + perturbation) > conformal_energy(V, F, UV)


@pytest.mark.parametrize('pins, targets', [
    ([0, 0], None), ([0, 4], None), ([-1, 1], None), ([0., 1.], None),
    ([0, 1, 2], None), ([0, 1], [[0, 0], [0, 0]]),
    ([0, 1], [[0, 0], [np.nan, 0]]), ([0, 1], [0, 1]),
])
def test_invalid_pin_constraints(pins, targets):
    V, F = rectangle()
    with pytest.raises(ValueError):
        lscm_embedding(V, F, pins, targets)


@pytest.mark.parametrize('kind', [
    'closed', 'disconnected', 'degenerate', 'winding', 'unused',
    'bad_index', 'float_faces', 'nonfinite', 'duplicate', 'empty',
])
def test_invalid_meshes(kind):
    V, F = rectangle()
    if kind == 'closed':
        V = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        F = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    elif kind == 'disconnected':
        V = np.vstack((V, V + [5, 0, 0]))
        F = np.vstack((F, F + 5))
    elif kind == 'degenerate':
        V[4] = V[0]
    elif kind == 'winding':
        F[0] = F[0, ::-1]
    elif kind == 'unused':
        V = np.vstack((V, [[9, 9, 9]]))
    elif kind == 'bad_index':
        F[0, 0] = len(V)
    elif kind == 'float_faces':
        F = F.astype(float)
    elif kind == 'nonfinite':
        V[0, 0] = np.nan
    elif kind == 'duplicate':
        F = np.vstack((F, F[0]))
    elif kind == 'empty':
        F = np.empty((0, 3), dtype=int)
    with pytest.raises(ValueError):
        lscm_embedding(V, F)


def test_multiple_boundary_loops_are_rejected():
    V, F = disk(n_radial=3, n_angular=12)
    # Remove the center fan and its now-unused center vertex, leaving an annulus.
    with pytest.raises(ValueError, match='disk topology'):
        lscm_embedding(V[1:], F[12:] - 1)


def test_nonmanifold_edge_is_rejected():
    V, F = rectangle()
    V = np.vstack((V, [[0., -1., 0.]]))
    F = np.vstack((F, [[0, 4, 5]]))
    with pytest.raises(ValueError, match='nonmanifold'):
        lscm_embedding(V, F)


def test_failed_solver_is_reported(monkeypatch):
    import src.param.lscm as module
    monkeypatch.setattr(module, 'lsqr', lambda *args, **kwargs: (np.zeros(6), 7))
    V, F = rectangle()
    with pytest.raises(ValueError, match='did not converge'):
        lscm_embedding(V, F, [0, 1])


@pytest.mark.parametrize('module_name', ['src.viewer', 'src.viewer_clean'])
def test_viewers_register_lscm_without_a_display(monkeypatch, module_name):
    import importlib
    module = importlib.import_module(module_name)
    ps = MagicMock()
    monkeypatch.setattr(module, 'ps', ps)
    monkeypatch.setattr(module, 'HAS_TRIMESH', False)
    # Exercise the real setup, parameterization, and metric paths on small meshes.
    monkeypatch.setattr(module, 'disk', lambda **kw: disk(n_radial=3, n_angular=12))
    monkeypatch.setattr(module, 'hemisphere', lambda **kw: hemisphere(n_lat=8, n_lon=12))
    from src.shapes import uv_sphere
    monkeypatch.setattr(module, 'uv_sphere', lambda **kw: uv_sphere(n_lat=6, n_lon=12))
    if module_name.endswith('viewer_clean'):
        monkeypatch.setattr(module, 'state', module.ViewerState())
    module.main()
    ps.show.assert_called_once()
    if module_name.endswith('viewer_clean'):
        for key in ('disk', 'hemisphere'):
            assert 'LSCM' in module.state.meshes[key]['uvs']
            assert 'LSCM' in module.state.meshes[key]['metrics']
        module.state.current_mesh = module.state.mesh_names.index('Disk')
        assert 'LSCM' in module.state.available_methods()
        module.state.current_method = module.state.available_methods().index('LSCM')
        module.update_display()
        actual = ps.register_surface_mesh.return_value.add_parameterization_quantity.call_args.args[1]
        np.testing.assert_array_equal(actual, module.state.meshes['disk']['uvs']['LSCM'])
    else:
        calls = ps.register_surface_mesh.return_value.add_parameterization_quantity.call_args_list
        assert sum(call.args[0] == 'LSCM' for call in calls) == 2
