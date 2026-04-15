"""
Tests for starworlds/admissible_kernel.py.

Verifies:
  - Admissible kernel is non-empty for a free exterior point.
  - Admissible kernel is None (empty) when x_bar is inside the obstacle.
  - Points in the admissible kernel produce starshaped hulls that EXCLUDE x_bar.
  - Cluster admissible kernel is the intersection of per-obstacle kernels.

Paper references: §IV-A, Eq. (6)–(9), Properties 2–3, Proposition 3.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from starworlds.obstacles import Ellipse, ConvexPolygon2D
from starworlds.admissible_kernel import (
    admissible_kernel_single_2d,
    build_ak_cache,
    cluster_admissible_kernel_2d,
    point_in_admissible_kernel,
)
from starworlds.geometry import polygon_contains_point, tangent_points_2d


# ---------------------------------------------------------------------------
# Single obstacle AK
# ---------------------------------------------------------------------------

class TestAdmissibleKernelSingle:

    def setup_method(self):
        # Ellipse at (3, 0); external point at origin
        self.obs = Ellipse(center=[3.0, 0.0], a=1.0, b=0.5, angle=0.0)
        self.x_bar_ext = np.array([0.0, 0.0])   # free exterior
        self.x_bar_int = np.array([3.0, 0.0])   # inside

    def test_exterior_returns_polygon(self):
        cone = admissible_kernel_single_2d(self.obs, self.x_bar_ext, L=100.0)
        assert cone is not None
        assert cone.shape[1] == 2

    def test_interior_returns_none(self):
        cone = admissible_kernel_single_2d(self.obs, self.x_bar_int, L=100.0)
        assert cone is None

    def test_point_toward_obstacle_in_ak(self):
        """A point between x_bar and the obstacle should be in the AK."""
        cone = admissible_kernel_single_2d(self.obs, self.x_bar_ext, L=100.0)
        # Point at (1.5, 0) — between x_bar=(0,0) and ellipse center=(3,0)
        assert point_in_admissible_kernel(cone, np.array([1.5, 0.0]))

    def test_point_behind_x_bar_not_in_ak(self):
        """A point directly behind x_bar (opposite from obstacle) is excluded."""
        cone = admissible_kernel_single_2d(self.obs, self.x_bar_ext, L=100.0)
        # Point at (-2, 0) — same side as x_bar, further from obstacle
        assert not point_in_admissible_kernel(cone, np.array([-2.0, 0.0]))


class TestAdmissibleKernelExclusion:
    """
    Core invariant (Definition 3 / Eq. 6):
    For any kernel point k in ad_ker(A, {x_bar}), the starshaped hull
    SH_k(A) must NOT contain x_bar.

    We verify this by checking that x_bar is not in the convex hull of
    {obs_boundary_samples} ∪ {tangent_pts_from_k} (which defines SH_k for
    a convex obstacle via Eq. 4).
    """

    def setup_method(self):
        self.obs = Ellipse(center=[3.0, 0.0], a=1.0, b=0.5, angle=0.0)
        self.x_bar = np.array([0.0, 0.0])

    def test_hull_from_ak_kernel_excludes_x_bar(self):
        from starworlds.geometry import convex_hull_2d
        from starworlds.starshaped_hull import sh_kernel_obstacle_2d

        cone = admissible_kernel_single_2d(self.obs, self.x_bar, L=100.0)
        assert cone is not None

        # Sample several candidate kernel points from inside the cone
        candidates = [np.array([1.0, 0.5]), np.array([1.5, -0.3]),
                      np.array([2.0, 0.2]), np.array([1.0, -0.5])]

        for k in candidates:
            if not point_in_admissible_kernel(cone, k):
                continue
            K = np.array([k, k + np.array([0.01, 0.0]), k + np.array([0.0, 0.01])])
            hull = sh_kernel_obstacle_2d(self.obs, K, n_approx=64)
            if hull is not None:
                in_hull = polygon_contains_point(hull, self.x_bar)
                assert not in_hull, f"x_bar in hull for k={k}"


# ---------------------------------------------------------------------------
# Cluster AK (Property 2, Property 3, Eq. 7)
# ---------------------------------------------------------------------------

class TestClusterAdmissibleKernel:

    def setup_method(self):
        self.obs1 = Ellipse(center=[2.0, 0.0], a=0.5, b=0.5)
        self.obs2 = Ellipse(center=[2.0, 2.0], a=0.5, b=0.5)
        self.x    = np.array([0.0, 0.0])
        self.xg   = np.array([5.0, 5.0])
        self.obstacles = [self.obs1, self.obs2]

    def test_cluster_ak_is_subset_of_individual(self):
        """Cluster AK ⊆ individual AK for each obstacle (Property 2)."""
        excluded = [self.x, self.xg]
        cache = build_ak_cache(self.obstacles, excluded, L=100.0)

        ak_cluster = cluster_admissible_kernel_2d(cache, [0, 1], [0, 1])
        ak_obs0    = cluster_admissible_kernel_2d(cache, [0],    [0, 1])
        ak_obs1    = cluster_admissible_kernel_2d(cache, [1],    [0, 1])

        assert ak_cluster is not None

        # Sample points in cluster AK and verify they're in each individual AK
        from starworlds.geometry import polygon_centroid_2d
        centroid = polygon_centroid_2d(ak_cluster)
        assert point_in_admissible_kernel(ak_obs0, centroid), \
            "Cluster AK centroid not in obs0 AK"
        assert point_in_admissible_kernel(ak_obs1, centroid), \
            "Cluster AK centroid not in obs1 AK"

    def test_ak_none_when_obstacle_contains_x(self):
        """If robot position is inside an obstacle, AK should be None."""
        x_inside = np.array([2.0, 0.0])  # inside obs1
        cache = build_ak_cache(self.obstacles, [x_inside, self.xg], L=100.0)
        ak = cluster_admissible_kernel_2d(cache, [0], [0, 1])
        assert ak is None
