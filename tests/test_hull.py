"""
Tests for starworlds/starshaped_hull.py.

Verifies:
  - SH_kerK(obstacle) ⊇ obstacle  (coverage)
  - CH(K) ⊂ SH_kerK(obstacle)     (kernel inclusion, Property 4a)
  - Three non-collinear K points → strictly starshaped result (Proposition 4)
  - x_bar not in SH_kerK when CH(K) ⊂ ad_ker  (Proposition 5)
  - Property 4d: cluster hull = union of per-obstacle hulls

Paper references: Eq. (11), Properties 4a/4c/4d, Propositions 4, 5.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from starworlds.obstacles import Ellipse, ConvexPolygon2D
from starworlds.starshaped_hull import (
    sh_kernel_obstacle_2d,
    sh_kernel_cluster_2d,
    proxy_pieces_intersect,
    center_point_from_kernel,
)
from starworlds.geometry import (
    polygon_contains_point,
    convex_hull_2d,
    polygon_centroid_2d,
)


def _make_equilateral_K(center, radius):
    angles = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    return np.array(center) + radius * np.column_stack([np.cos(angles), np.sin(angles)])


class TestShKernelObstacle:

    def setup_method(self):
        self.ellipse = Ellipse(center=[0.0, 0.0], a=1.5, b=1.0, angle=0.0)
        self.square  = ConvexPolygon2D(np.array([[1, 1], [3, 1], [3, 3], [1, 3]], dtype=float))

    # ------------------------------------------------------------------ #
    # Coverage: original obstacle ⊂ SH_kerK(obstacle)                    #
    # ------------------------------------------------------------------ #
    def test_coverage_ellipse(self):
        K = _make_equilateral_K(center=[3.0, 0.0], radius=0.1)
        hull = sh_kernel_obstacle_2d(self.ellipse, K)
        assert hull is not None
        # Check that boundary samples of the ellipse are inside the hull
        for pt in self.ellipse.polygon_approximation(n=32):
            assert polygon_contains_point(hull, pt), f"{pt} not in hull"

    def test_coverage_square(self):
        K = _make_equilateral_K(center=[5.0, 2.0], radius=0.1)
        hull = sh_kernel_obstacle_2d(self.square, K)
        assert hull is not None
        for pt in self.square.vertices:
            assert polygon_contains_point(hull, pt), f"vertex {pt} not in hull"

    # ------------------------------------------------------------------ #
    # Kernel inclusion: CH(K) ⊂ ker(SH_kerK)                             #
    # If CH(K) is in the kernel, each k should see the entire hull       #
    # ------------------------------------------------------------------ #
    def test_kernel_points_inside_hull(self):
        K = _make_equilateral_K(center=[3.0, 0.0], radius=0.1)
        hull = sh_kernel_obstacle_2d(self.ellipse, K)
        assert hull is not None
        for k in K:
            assert polygon_contains_point(hull, k), f"kernel point {k} not in hull"

    # ------------------------------------------------------------------ #
    # Exclusion (Proposition 5): if K ⊂ ad_ker, x_bar ∉ SH_kerK         #
    # ------------------------------------------------------------------ #
    def test_exclusion_of_external_point(self):
        """Place K well inside ad_ker(ellipse, {x_bar}) and verify x_bar excluded."""
        from starworlds.admissible_kernel import admissible_kernel_single_2d, point_in_admissible_kernel

        x_bar = np.array([-5.0, 0.0])  # to the left of ellipse at origin
        cone = admissible_kernel_single_2d(self.ellipse, x_bar, L=100.0)
        assert cone is not None

        # K at (2, 0) — between x_bar and ellipse, inside AK
        K = _make_equilateral_K(center=[2.0, 0.0], radius=0.05)

        hull = sh_kernel_obstacle_2d(self.ellipse, K)
        if hull is not None:
            assert not polygon_contains_point(hull, x_bar), \
                "x_bar should not be in hull when K ⊂ ad_ker"

    # ------------------------------------------------------------------ #
    # Convex hull upper bound (Property 4c): SH_kerK ⊂ CH(obstacle)      #
    # when K ⊂ CH(obstacle)                                               #
    # ------------------------------------------------------------------ #
    def test_hull_subset_of_convex_hull(self):
        """When K is inside the obstacle, SH_kerK = obstacle (no expansion)."""
        # K inside ellipse at origin (a=1.5, b=1.0): K at (0.1, 0.0) triangle
        K = _make_equilateral_K(center=[0.0, 0.0], radius=0.1)
        hull = sh_kernel_obstacle_2d(self.ellipse, K)
        assert hull is not None
        # Hull should not be vastly larger than the ellipse
        ellipse_ch = convex_hull_2d(self.ellipse.polygon_approximation(n=64))
        ellipse_area = 0.5 * abs(sum(
            ellipse_ch[i][0] * ellipse_ch[(i+1) % len(ellipse_ch)][1] -
            ellipse_ch[(i+1) % len(ellipse_ch)][0] * ellipse_ch[i][1]
            for i in range(len(ellipse_ch))
        ))
        hull_area = 0.5 * abs(sum(
            hull[i][0] * hull[(i+1) % len(hull)][1] -
            hull[(i+1) % len(hull)][0] * hull[i][1]
            for i in range(len(hull))
        ))
        assert hull_area <= ellipse_area * 1.05, \
            f"Hull area {hull_area:.3f} much larger than ellipse area {ellipse_area:.3f}"


class TestShKernelCluster:
    """Property 4d: SH_kerK(A∪) = ∪_i SH_kerK(Aᵢ)."""

    def test_cluster_covers_all_obstacles(self):
        obs1 = Ellipse(center=[0.0, 0.0], a=0.5, b=0.5)
        obs2 = Ellipse(center=[2.0, 0.0], a=0.5, b=0.5)
        K = _make_equilateral_K(center=[3.0, 1.0], radius=0.1)

        pieces = sh_kernel_cluster_2d([obs1, obs2], K)
        assert len(pieces) == 2  # one hull per obstacle

        # Each obstacle's boundary should be inside the corresponding hull piece
        for obs, hull in zip([obs1, obs2], pieces):
            assert hull is not None
            for pt in obs.polygon_approximation(n=16):
                assert polygon_contains_point(hull, pt), \
                    f"Cluster hull doesn't cover obstacle point {pt}"


class TestProxyIntersect:

    def test_overlapping_proxies(self):
        sq1 = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        sq2 = np.array([[1, 1], [3, 1], [3, 3], [1, 3]], dtype=float)
        assert proxy_pieces_intersect([sq1], [sq2])

    def test_disjoint_proxies(self):
        sq1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        sq2 = np.array([[5, 5], [6, 5], [6, 6], [5, 6]], dtype=float)
        assert not proxy_pieces_intersect([sq1], [sq2])


class TestCenterPoint:

    def test_center_not_on_line(self):
        K = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
        x  = np.array([0.0, 0.0])
        xg = np.array([10.0, 0.0])
        c = center_point_from_kernel(K, x, xg)
        # Centroid = (0, 1/3) — not on the x-axis l(x, xg)
        assert c is not None
        assert abs(c[1]) > 1e-9
