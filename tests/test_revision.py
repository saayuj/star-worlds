"""Tests for the components added in the journal revision."""

import numpy as np
import pytest

from starworlds.shapes3d import Sphere3D, Box3D, Cylinder3D
from starworlds.integration import StarWorldUpdater
from starworlds.reactive_controller import ReactiveController
from starworlds.kernel_selection import select_kernel_3d_admissible
from starworlds.baselines import per_obstacle_proxies, any_overlap, CBFQPController
from starworlds.star_world import ObstacleProxy


@pytest.mark.parametrize("shape", [
    Sphere3D([0.1, 0.2, 0.3], 0.1),
    Box3D([0.0, 0.0, 0.0], [0.05, 0.1, 0.2]),
    Cylinder3D([0.0, 0.0, 0.0], 0.06, 0.12),
])
def test_support_points_on_boundary(shape):
    pts = shape.polygon_approximation(64)
    sd = shape.signed_distance(pts)
    assert np.all(np.abs(sd) < 1e-9)
    infl = shape.inflated(0.05)
    assert np.all(np.abs(infl.signed_distance(infl.polygon_approximation(64))) < 1e-9)


def test_moved_copy_drops_cache():
    s = Sphere3D([0, 0, 0], 0.1)
    s._hull_eq_cache = "stale"
    assert not hasattr(s.moved([1, 0, 0]), "_hull_eq_cache")
    assert not hasattr(s.inflated(0.1), "_hull_eq_cache")


def test_ray_radius_exact_for_sphere():
    s = Sphere3D([0.5, 0.0, 0.4], 0.12)
    pr = per_obstacle_proxies([s], n=400)[0]
    for d in np.random.default_rng(0).normal(size=(20, 3)):
        x = s.center + 0.3 * d / np.linalg.norm(d)
        r = ReactiveController.ray_radius(pr, x)
        assert abs(r - 0.12) < 0.004          # polytope approximation error


def test_admissible_kernel_excludes_robot_and_goal():
    obs = [Sphere3D([0.55, -0.06, 0.45], 0.14), Sphere3D([0.55, 0.06, 0.45], 0.14)]
    x, xg = np.array([0.30, 0.0, 0.45]), np.array([0.72, 0.0, 0.45])
    res = StarWorldUpdater(n_approx_3d=48).update(obs, x, xg)
    assert len(res.proxies) == 1 and not res.proxies[0].is_fallback
    from scipy.spatial import ConvexHull
    for piece in res.proxies[0].pieces:
        h = ConvexHull(piece)
        for e in (x, xg):
            assert np.any(h.equations[:, :3] @ e + h.equations[:, 3] > 0)


def test_kernel_selection_respects_forbid_dir():
    obs = [Sphere3D([0.5, 0.0, 0.45], 0.12)]
    x, xg = np.array([0.3, 0.0, 0.45]), np.array([0.7, 0.0, 0.45])
    K, kc = select_kernel_3d_admissible(obs, x, xg)
    off = kc - np.array([kc[0], 0.0, 0.45])
    K2, kc2 = select_kernel_3d_admissible(obs, x, xg, forbid_dir=off)
    off2 = kc2 - np.array([kc2[0], 0.0, 0.45])
    assert float(off @ off2) < 0


def test_modulation_does_not_penetrate_box_proxy():
    box = Box3D([0.5, 0.0, 0.45], [0.03, 0.15, 0.10]).inflated(0.05)
    x0, xg = np.array([0.30, 0.0, 0.45]), np.array([0.70, 0.0, 0.45])
    res = StarWorldUpdater(n_approx_3d=48).update([box], x0, xg)
    ctrl = ReactiveController(v_max=0.3, safety_radius=0.0, goal_tolerance=0.02)
    # simulate the point robot; it must never enter the inflated box
    x = x0.copy()
    for _ in range(600):
        v = ctrl.compute_velocity(x, xg, res.proxies)
        x = x + 0.005 * v
        assert box.signed_distance(x)[0] > -0.004
    assert np.linalg.norm(x - xg) < 0.05


def test_overlap_detection():
    a = Sphere3D([0, 0, 0], 0.1)
    assert any_overlap([a, Sphere3D([0.15, 0, 0], 0.1)])
    assert not any_overlap([a, Sphere3D([0.25, 0, 0], 0.1)])


def test_cbf_keeps_safe_set():
    obs = [Sphere3D([0.5, 0.0, 0.45], 0.1)]
    c = CBFQPController(v_max=0.3, alpha=4.0)
    x = np.array([0.3, 0.001, 0.45])
    for _ in range(400):
        x = x + 0.01 * c.compute_velocity(x, np.array([0.7, 0, 0.45]), obs)
        assert obs[0].signed_distance(x)[0] > -1e-3
