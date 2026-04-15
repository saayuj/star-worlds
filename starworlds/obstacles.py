"""
Obstacle representations for the Star Worlds algorithm.

Paper §III, Problem 1:
  - W = R² : obstacles are polygons or convex sets (Ellipse, ConvexPolygon2D, Polygon2D)
  - W = R³ : obstacles are convex sets (Ellipsoid)

All classes expose:
  contains(point)             — True if point is strictly inside or on boundary
  polygon_approximation(n)    — (n, dim) array of boundary vertices (CCW in 2D)
  is_convex                   — bool property
  dim                         — spatial dimension property
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


class Obstacle(ABC):
    """Abstract base for all obstacle shapes."""

    @abstractmethod
    def contains(self, point: np.ndarray) -> bool: ...

    @abstractmethod
    def polygon_approximation(self, n: int = 64) -> np.ndarray:
        """Return boundary vertices as (n, dim) array in CCW order."""
        ...

    @property
    @abstractmethod
    def is_convex(self) -> bool: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...


# ---------------------------------------------------------------------------
# 2-D obstacles
# ---------------------------------------------------------------------------

class Ellipse(Obstacle):
    """
    2-D ellipse: { x : ((R^T(x-c))^T diag(1/a², 1/b²) R^T(x-c)) ≤ 1 }

    Parameters
    ----------
    center : array_like, shape (2,)
    a, b   : semi-axis lengths along the local x- and y-axes
    angle  : rotation of the ellipse's major axis from the world x-axis [radians]
    """

    def __init__(self, center, a: float, b: float, angle: float = 0.0):
        self.center = np.asarray(center, dtype=float)
        self.a = float(a)
        self.b = float(b)
        self.angle = float(angle)
        c, s = np.cos(angle), np.sin(angle)
        self._R = np.array([[c, -s], [s, c]])   # rotation matrix (local→world)

    @property
    def is_convex(self) -> bool:
        return True

    @property
    def dim(self) -> int:
        return 2

    def contains(self, point) -> bool:
        p = np.asarray(point, dtype=float)
        local = self._R.T @ (p - self.center)
        return (local[0] / self.a) ** 2 + (local[1] / self.b) ** 2 <= 1.0

    def polygon_approximation(self, n: int = 64) -> np.ndarray:
        """Return n-gon approximation of the ellipse boundary (CCW, closed)."""
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        local = np.column_stack([self.a * np.cos(theta), self.b * np.sin(theta)])
        return (self._R @ local.T).T + self.center

    def tangent_angles(self, x_bar: np.ndarray):
        """
        Analytically compute the two tangent points of the ellipse from an
        exterior point x_bar.  Returns (t1, t2) with x_bar, t1, t2 in CW order.
        Returns (None, None) if x_bar is inside or on the boundary.

        Derivation (local frame):
          Boundary: (u[0]/a)² + (u[1]/b)² = 1,  u = R^T(t - c)
          Tangent condition: ∇b · (t - x_bar) = 0
            → (cos θ / a) * (a cos θ - x̄_loc[0]) + (sin θ / b) * (b sin θ - x̄_loc[1]) = 0
            → 1 - (x̄_loc[0]/a) cos θ - (x̄_loc[1]/b) sin θ = 0
          With A = x̄_loc[0]/a, B = x̄_loc[1]/b:
            A cos θ + B sin θ = 1
            → θ = atan2(B, A) ± acos(1/√(A²+B²))
        """
        x_bar = np.asarray(x_bar, dtype=float)
        local = self._R.T @ (x_bar - self.center)
        A_coeff = local[0] / self.a
        B_coeff = local[1] / self.b
        r2 = A_coeff ** 2 + B_coeff ** 2
        if r2 < 1.0 + 1e-10:    # inside or on boundary
            return None, None
        delta = np.arccos(np.clip(1.0 / np.sqrt(r2), -1.0, 1.0))
        theta0 = np.arctan2(B_coeff, A_coeff)
        th1, th2 = theta0 + delta, theta0 - delta
        t1_loc = np.array([self.a * np.cos(th1), self.b * np.sin(th1)])
        t2_loc = np.array([self.a * np.cos(th2), self.b * np.sin(th2)])
        t1 = self._R @ t1_loc + self.center
        t2 = self._R @ t2_loc + self.center
        # Ensure x_bar, t1, t2 is CW: t1 should have larger polar angle from x_bar
        ang1 = np.arctan2(t1[1] - x_bar[1], t1[0] - x_bar[0])
        ang2 = np.arctan2(t2[1] - x_bar[1], t2[0] - x_bar[0])
        if ang1 < ang2:
            t1, t2 = t2, t1
        return t1, t2


class ConvexPolygon2D(Obstacle):
    """
    2-D strictly convex polygon with vertices given in CCW order.

    Vertices should be a convex polygon.  Use Polygon2D for the non-convex case.
    """

    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=float)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError("vertices must be (n, 2)")

    @property
    def is_convex(self) -> bool:
        return True

    @property
    def dim(self) -> int:
        return 2

    def contains(self, point) -> bool:
        """Point-in-convex-polygon via half-plane tests."""
        p = np.asarray(point, dtype=float)
        verts = self.vertices
        n = len(verts)
        for i in range(n):
            a, b = verts[i], verts[(i + 1) % n]
            edge = b - a
            normal = np.array([-edge[1], edge[0]])   # inward-pointing for CCW
            if np.dot(p - a, normal) < -1e-10:
                return False
        return True

    def polygon_approximation(self, n: int = None) -> np.ndarray:
        return self.vertices.copy()


class Polygon2D(Obstacle):
    """
    General (possibly non-convex) 2-D polygon with vertices in CCW order.
    For Algorithm 2, non-convex polygons should be convex-decomposed first
    (see geometry.convex_decompose_2d).
    """

    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=float)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError("vertices must be (n, 2)")
        self._convex_flag = self._check_convex()

    def _check_convex(self) -> bool:
        v = self.vertices
        n = len(v)
        sign = None
        for i in range(n):
            a, b, c = v[i], v[(i + 1) % n], v[(i + 2) % n]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if abs(cross) < 1e-12:
                continue
            s = np.sign(cross)
            if sign is None:
                sign = s
            elif s != sign:
                return False
        return True

    @property
    def is_convex(self) -> bool:
        return self._convex_flag

    @property
    def dim(self) -> int:
        return 2

    def contains(self, point) -> bool:
        """Ray-casting point-in-polygon test."""
        p = np.asarray(point, dtype=float)
        x, y = p
        verts = self.vertices
        n = len(verts)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = verts[i]
            xj, yj = verts[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def polygon_approximation(self, n: int = None) -> np.ndarray:
        return self.vertices.copy()


# ---------------------------------------------------------------------------
# 3-D obstacles
# ---------------------------------------------------------------------------

class Ellipsoid(Obstacle):
    """
    3-D ellipsoid: { x : (x-c)^T Q^{-1} (x-c) ≤ 1 }

    Parameters
    ----------
    center : array_like, shape (3,)
    Q      : (3, 3) positive-definite shape matrix (Q = R diag(a², b², c²) R^T)
    """

    def __init__(self, center, Q):
        self.center = np.asarray(center, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self._Qinv = np.linalg.inv(self.Q)

    @property
    def is_convex(self) -> bool:
        return True

    @property
    def dim(self) -> int:
        return 3

    def contains(self, point) -> bool:
        d = np.asarray(point, dtype=float) - self.center
        return float(d @ self._Qinv @ d) <= 1.0

    def polygon_approximation(self, n: int = 64) -> np.ndarray:
        """
        Returns points on the ellipsoid surface (for tangent computations).
        Uses Fibonacci sphere sampling to give roughly uniform coverage.
        """
        golden = np.pi * (3.0 - np.sqrt(5.0))
        idx = np.arange(n)
        y = 1.0 - (idx / (n - 1.0)) * 2.0
        r = np.sqrt(np.clip(1.0 - y * y, 0.0, None))
        theta = golden * idx
        unit = np.column_stack([r * np.cos(theta), y, r * np.sin(theta)])
        # Transform unit sphere to ellipsoid: x = c + L @ u, where Q = L @ L^T
        L = np.linalg.cholesky(self.Q)
        return (L @ unit.T).T + self.center
