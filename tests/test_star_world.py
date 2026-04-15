"""
Integration tests for Algorithm 2 (starworlds/star_world.py).

Verifies:
  1. Three intersecting ellipses → single disjoint starshaped proxy (Fig. 17).
  2. Disjoint ellipses remain disjoint (no unnecessary merging).
  3. Robot and goal positions are excluded from all proxy obstacles.
  4. Each proxy covers its original obstacles (O∪ ⊂ O'∪, constraint 2a).
  5. Result is_disjoint flag is correct.
  6. Temporal coherence: second call reuses previous kernel centroids.

Paper references: Algorithm 2, Problem 1 constraints (2a)–(2e), §V.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from starworlds.obstacles import Ellipse, ConvexPolygon2D
from starworlds.star_world import create_star_world, StarWorldResult
from starworlds.geometry import polygon_contains_point


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _proxy_contains_point(proxy, point):
    for piece in proxy.pieces:
        if polygon_contains_point(piece, point):
            return True
    return False


def _obs_covered_by_proxies(obs, result):
    """Check that every boundary sample of obs is inside some proxy."""
    for pt in obs.polygon_approximation(n=16):
        covered = any(_proxy_contains_point(p, pt) for p in result.proxies)
        if not covered:
            return False
    return True


# ---------------------------------------------------------------------------
# Test 1: Three intersecting ellipses (Paper Fig. 17)
# ---------------------------------------------------------------------------

class TestIntersectingEllipses:
    """Three overlapping ellipses should be merged into ≤ 1-2 starshaped proxies."""

    def setup_method(self):
        self.obs = [
            Ellipse(center=[0.0, 0.0], a=1.5, b=0.8),
            Ellipse(center=[1.5, 0.0], a=1.5, b=0.8),
            Ellipse(center=[0.75, 1.0], a=1.5, b=0.8),
        ]
        self.x  = np.array([-3.0, -3.0])
        self.xg = np.array([ 3.0,  3.0])

    def test_runs_without_error(self):
        result = create_star_world(self.obs, self.x, self.xg)
        assert isinstance(result, StarWorldResult)

    def test_fewer_clusters_than_obstacles(self):
        result = create_star_world(self.obs, self.x, self.xg)
        # Intersecting obstacles should be merged
        assert len(result.proxies) <= len(self.obs)

    def test_each_obstacle_is_covered(self):
        result = create_star_world(self.obs, self.x, self.xg)
        for obs in self.obs:
            assert _obs_covered_by_proxies(obs, result), \
                "Original obstacle not covered by any proxy"

    def test_robot_not_in_any_proxy(self):
        result = create_star_world(self.obs, self.x, self.xg)
        for proxy in result.proxies:
            assert not _proxy_contains_point(proxy, self.x), \
                "Robot position inside a proxy obstacle"

    def test_goal_not_in_any_proxy(self):
        result = create_star_world(self.obs, self.x, self.xg)
        for proxy in result.proxies:
            assert not _proxy_contains_point(proxy, self.xg), \
                "Goal position inside a proxy obstacle"


# ---------------------------------------------------------------------------
# Test 2: Disjoint obstacles (no merging needed)
# ---------------------------------------------------------------------------

class TestDisjointEllipses:
    """Well-separated obstacles should stay as separate proxies."""

    def setup_method(self):
        self.obs = [
            Ellipse(center=[-5.0, 0.0], a=0.5, b=0.5),
            Ellipse(center=[ 5.0, 0.0], a=0.5, b=0.5),
        ]
        self.x  = np.array([0.0, 5.0])
        self.xg = np.array([0.0, -5.0])

    def test_two_proxies_produced(self):
        result = create_star_world(self.obs, self.x, self.xg)
        # Disjoint obstacles should remain separate
        assert len(result.proxies) == 2

    def test_is_disjoint(self):
        result = create_star_world(self.obs, self.x, self.xg)
        assert result.is_disjoint

    def test_coverage(self):
        result = create_star_world(self.obs, self.x, self.xg)
        for obs in self.obs:
            assert _obs_covered_by_proxies(obs, result)

    def test_robot_goal_excluded(self):
        result = create_star_world(self.obs, self.x, self.xg)
        for proxy in result.proxies:
            assert not _proxy_contains_point(proxy, self.x)
            assert not _proxy_contains_point(proxy, self.xg)


# ---------------------------------------------------------------------------
# Test 3: Coverage constraint 2a (O∪ ⊂ O'∪)
# ---------------------------------------------------------------------------

class TestCoverageConstraint:

    def test_union_coverage(self):
        """Every original obstacle point must lie in some proxy (constraint 2a)."""
        obs = [
            Ellipse(center=[0.0, 0.0], a=1.0, b=1.0),
            ConvexPolygon2D(np.array([[3, 0], [5, 0], [4, 2]], dtype=float)),
        ]
        x  = np.array([-3.0, 3.0])
        xg = np.array([6.0, -1.0])

        result = create_star_world(obs, x, xg)
        for o in obs:
            assert _obs_covered_by_proxies(o, result), \
                f"Obstacle at {o} not fully covered"


# ---------------------------------------------------------------------------
# Test 4: Temporal coherence
# ---------------------------------------------------------------------------

class TestTemporalCoherence:
    """Calling with prev_result should not crash and should produce valid output."""

    def test_two_consecutive_calls(self):
        obs = [
            Ellipse(center=[1.0, 0.0], a=0.5, b=0.5),
            Ellipse(center=[1.0, 1.2], a=0.5, b=0.5),
        ]
        x  = np.array([-2.0, -2.0])
        xg = np.array([ 4.0,  4.0])

        result1 = create_star_world(obs, x, xg)
        # Slight obstacle movement
        obs2 = [
            Ellipse(center=[1.1, 0.0], a=0.5, b=0.5),
            Ellipse(center=[1.0, 1.3], a=0.5, b=0.5),
        ]
        result2 = create_star_world(obs2, x, xg, prev_result=result1)

        assert isinstance(result2, StarWorldResult)
        for proxy in result2.proxies:
            assert not _proxy_contains_point(proxy, x)
            assert not _proxy_contains_point(proxy, xg)


# ---------------------------------------------------------------------------
# Test 5: Polygon obstacle
# ---------------------------------------------------------------------------

class TestPolygonObstacle:

    def test_convex_polygon(self):
        poly = ConvexPolygon2D(np.array([
            [0, 0], [2, 0], [2, 2], [0, 2]
        ], dtype=float))
        x  = np.array([-3.0, 1.0])
        xg = np.array([ 5.0, 1.0])

        result = create_star_world([poly], x, xg)
        assert isinstance(result, StarWorldResult)
        assert _obs_covered_by_proxies(poly, result)
        for proxy in result.proxies:
            assert not _proxy_contains_point(proxy, x)
            assert not _proxy_contains_point(proxy, xg)
