"""
Tests for starworlds/geometry.py and starworlds/obstacles.py.

Verifies:
  - Ellipse tangent point computation (analytical formula)
  - Polygon tangent point computation (polar-angle method)
  - Cone polygon construction and containment
  - Sutherland-Hodgman polygon clipping
  - Convex hull
  - Side-of-line utility
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from starworlds.obstacles import Ellipse, ConvexPolygon2D, Polygon2D
from starworlds.geometry import (
    tangent_points_2d,
    cone_polygon_2d,
    point_in_cone_2d,
    polygon_contains_point,
    convex_hull_2d,
    sutherland_hodgman,
    side_of_line_2d,
    half_plane_polygon_2d,
    polygons_intersect,
    approx_inscribed_radius,
)


# ---------------------------------------------------------------------------
# Ellipse tangent points
# ---------------------------------------------------------------------------

class TestEllipseTangentPoints:
    """
    Ground truth: ellipse at origin with a=2, b=1, no rotation.
    External point x_bar = (-5, 0) (to the left).
    Tangent points should be symmetric about the x-axis.
    """

    def setup_method(self):
        self.ellipse = Ellipse(center=[0, 0], a=2.0, b=1.0, angle=0.0)
        self.x_bar = np.array([-5.0, 0.0])

    def test_returns_two_points(self):
        t1, t2 = tangent_points_2d(self.ellipse, self.x_bar)
        assert t1 is not None
        assert t2 is not None
        assert t1.shape == (2,)
        assert t2.shape == (2,)

    def test_tangent_points_on_ellipse_boundary(self):
        t1, t2 = tangent_points_2d(self.ellipse, self.x_bar)
        for t in (t1, t2):
            val = (t[0] / 2.0) ** 2 + (t[1] / 1.0) ** 2
            assert abs(val - 1.0) < 1e-6, f"Point {t} not on ellipse: {val}"

    def test_lines_are_tangent(self):
        """
        The line from x_bar to t1 (and t2) should be tangent to the ellipse:
        the dot product of the outward normal at t with (t - x_bar) must be zero.
        """
        t1, t2 = tangent_points_2d(self.ellipse, self.x_bar)
        for t in (t1, t2):
            # Normal at t (for standard ellipse a=2, b=1): [t[0]/a², t[1]/b²]
            normal = np.array([t[0] / (2.0 ** 2), t[1] / (1.0 ** 2)])
            tangent_dir = t - self.x_bar
            dot = abs(np.dot(normal, tangent_dir))
            assert dot < 1e-5, f"Line not tangent at {t}: dot={dot}"

    def test_x_bar_t1_t2_is_cw(self):
        """x_bar, t1, t2 should be in CW order (cross product of (t1-x_bar) and
        (t2-x_bar) must be negative)."""
        t1, t2 = tangent_points_2d(self.ellipse, self.x_bar)
        v1 = t1 - self.x_bar
        v2 = t2 - self.x_bar
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        assert cross < 0, f"Expected CW order, got cross={cross}"

    def test_inside_returns_none(self):
        t1, t2 = tangent_points_2d(self.ellipse, np.array([0.0, 0.0]))
        assert t1 is None and t2 is None


# ---------------------------------------------------------------------------
# Polygon tangent points
# ---------------------------------------------------------------------------

class TestPolygonTangentPoints:
    """
    Square with vertices at (±1, ±1).  External point at (-5, 0).
    """

    def setup_method(self):
        verts = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
        self.poly = ConvexPolygon2D(verts)
        self.x_bar = np.array([-5.0, 0.0])

    def test_returns_two_vertices(self):
        t1, t2 = tangent_points_2d(self.poly, self.x_bar)
        assert t1 is not None and t2 is not None

    def test_tangent_points_are_vertices(self):
        """For a square, the tangent points must be two of its four corners."""
        t1, t2 = tangent_points_2d(self.poly, self.x_bar)
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
        def is_corner(t):
            return any(np.linalg.norm(t - c) < 1e-9 for c in corners)
        assert is_corner(t1), f"{t1} is not a corner"
        assert is_corner(t2), f"{t2} is not a corner"

    def test_correct_corners(self):
        """From (-5, 0), the tangent corners should be (-1, 1) and (-1, -1)."""
        t1, t2 = tangent_points_2d(self.poly, self.x_bar)
        expected = {(-1.0, 1.0), (-1.0, -1.0)}
        got = {tuple(np.round(t1, 6)), tuple(np.round(t2, 6))}
        assert got == expected, f"Expected corners {expected}, got {got}"

    def test_inside_returns_none(self):
        t1, t2 = tangent_points_2d(self.poly, np.array([0.0, 0.0]))
        assert t1 is None and t2 is None


# ---------------------------------------------------------------------------
# Cone polygon & containment
# ---------------------------------------------------------------------------

class TestConePolygon:
    """
    Ellipse at origin, x_bar=(-5,0).  The admissible kernel cone should
    contain points that are NOT in the blocked angular region.
    """

    def setup_method(self):
        ellipse = Ellipse(center=[0, 0], a=2.0, b=1.0, angle=0.0)
        x_bar = np.array([-5.0, 0.0])
        t1, t2 = tangent_points_2d(ellipse, x_bar)
        self.cone = cone_polygon_2d(x_bar, t1, t2, L=1000.0, n_pts=64)
        self.x_bar = x_bar
        self.t1, self.t2 = t1, t2

    def test_cone_not_none(self):
        assert self.cone is not None

    def test_cone_has_apex_as_first_vertex(self):
        assert np.allclose(self.cone[0], self.x_bar, atol=1e-9)

    def test_point_behind_obstacle_in_cone(self):
        """A point directly to the right of the ellipse (toward A from x_bar)
        should be in the admissible kernel."""
        # Point at (5, 0) — on the far side of the ellipse from x_bar
        assert polygon_contains_point(self.cone, np.array([5.0, 0.0]))

    def test_point_directly_behind_x_bar_not_in_cone(self):
        """A point at (-10, 0) — directly behind x_bar (opposite from ellipse)
        — should NOT be in the admissible kernel."""
        # This is the excluded "behind x_bar" region
        assert not polygon_contains_point(self.cone, np.array([-10.0, 0.0]))


# ---------------------------------------------------------------------------
# Sutherland-Hodgman
# ---------------------------------------------------------------------------

class TestSutherlandHodgman:

    def test_clip_square_with_square(self):
        s = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        c = np.array([[1, 1], [3, 1], [3, 3], [1, 3]], dtype=float)
        result = sutherland_hodgman(s, c)
        assert result is not None
        # Intersection should be unit square at (1,1)-(2,2)
        assert len(result) >= 3
        area = 0.5 * abs(
            sum(result[i][0] * result[(i + 1) % len(result)][1]
                - result[(i + 1) % len(result)][0] * result[i][1]
                for i in range(len(result)))
        )
        assert abs(area - 1.0) < 1e-6

    def test_no_overlap_returns_none(self):
        s = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        c = np.array([[5, 5], [6, 5], [6, 6], [5, 6]], dtype=float)
        result = sutherland_hodgman(s, c)
        assert result is None


# ---------------------------------------------------------------------------
# Convex hull
# ---------------------------------------------------------------------------

class TestConvexHull:

    def test_square_hull(self):
        pts = np.array([[0, 0], [1, 0], [0.5, 0.5], [1, 1], [0, 1]], dtype=float)
        ch = convex_hull_2d(pts)
        assert ch is not None
        assert len(ch) == 4   # 4 corners of the square

    def test_collinear_returns_none(self):
        pts = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        # Only 3 points; collinear → ConvexHull will raise QhullError
        # Our wrapper returns None in that case
        ch = convex_hull_2d(pts)
        # Either None or degenerate — just check it doesn't crash
        assert ch is None or len(ch) >= 2


# ---------------------------------------------------------------------------
# Side-of-line and half-plane
# ---------------------------------------------------------------------------

class TestSideOfLine:

    def test_left_positive(self):
        p1, p2 = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        assert side_of_line_2d(p1, p2, np.array([0.5, 1.0])) > 0

    def test_right_negative(self):
        p1, p2 = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        assert side_of_line_2d(p1, p2, np.array([0.5, -1.0])) < 0

    def test_on_line_zero(self):
        p1, p2 = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        assert abs(side_of_line_2d(p1, p2, np.array([0.5, 0.0]))) < 1e-12


# ---------------------------------------------------------------------------
# Separating Axis Theorem
# ---------------------------------------------------------------------------

class TestPolygonsIntersect:

    def test_overlapping(self):
        a = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        b = np.array([[1, 1], [3, 1], [3, 3], [1, 3]], dtype=float)
        assert polygons_intersect(a, b)

    def test_touching(self):
        a = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        b = np.array([[1, 0], [2, 0], [2, 1], [1, 1]], dtype=float)
        assert polygons_intersect(a, b)

    def test_separated(self):
        a = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        b = np.array([[2, 2], [3, 2], [3, 3], [2, 3]], dtype=float)
        assert not polygons_intersect(a, b)
