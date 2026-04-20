"""
3-D geometric primitives for the Star Worlds algorithm.

Provides the 3-D analogues of the 2-D routines in geometry.py:

  convex_hull_3d                — 3-D convex hull vertices (scipy wrapper)
  convex_hull_with_equations_3d — hull + half-space equations for fast containment
  point_in_convex_hull_3d       — O(n_faces) containment test via hull equations
  polygon_centroid_3d           — centroid of a 3-D point cloud
  regular_tetrahedron_3d        — 4-vertex kernel shape (analogue of equilateral triangle)
  polygons_intersect_3d         — separating-hyperplane intersection test for two hulls

Used by star_world.py (create_star_world_3d) and rrt_planner.py for 3-D planning.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Convex hull
# ---------------------------------------------------------------------------

def convex_hull_3d(points: np.ndarray) -> Optional[np.ndarray]:
    """
    Return the vertices of the 3-D convex hull of *points* as a (m, 3) array.

    Deduplicates points before calling scipy.  Returns None if fewer than
    4 non-coplanar points are available.
    """
    pts = np.unique(np.asarray(points, dtype=float).reshape(-1, 3), axis=0)
    if len(pts) < 4:
        return None
    try:
        hull = ConvexHull(pts)
        return pts[np.unique(hull.simplices.flatten())]
    except Exception:
        return None


def convex_hull_with_equations_3d(
    points: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Return (hull_vertices, equations) for fast half-space containment testing.

    *equations* is scipy's ConvexHull.equations — shape (n_faces, 4), where
    row i encodes  equations[i, :3] @ p + equations[i, 3] <= 0  for all
    interior points p.

    Returns (None, None) on failure.
    """
    pts = np.unique(np.asarray(points, dtype=float).reshape(-1, 3), axis=0)
    if len(pts) < 4:
        return None, None
    try:
        hull = ConvexHull(pts)
        verts = pts[np.unique(hull.simplices.flatten())]
        return verts, hull.equations
    except Exception:
        return None, None


def point_in_convex_hull_3d(
    points: np.ndarray,
    point: np.ndarray,
    equations: Optional[np.ndarray] = None,
    tol: float = 1e-9,
) -> bool:
    """
    Return True if *point* lies inside (or on) the 3-D convex hull of *points*.

    If pre-computed *equations* (ConvexHull.equations) are supplied the test
    is O(n_faces).  Otherwise the hull is recomputed — prefer pre-computation
    when calling inside a planning loop.
    """
    p = np.asarray(point, dtype=float)
    if equations is not None:
        return bool(np.all(equations[:, :3] @ p + equations[:, 3] <= tol))
    pts = np.unique(np.asarray(points, dtype=float).reshape(-1, 3), axis=0)
    if len(pts) < 4:
        return False
    try:
        hull = ConvexHull(pts)
        return bool(np.all(hull.equations[:, :3] @ p + hull.equations[:, 3] <= tol))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Centroid
# ---------------------------------------------------------------------------

def polygon_centroid_3d(points: np.ndarray) -> np.ndarray:
    """Centroid (mean) of a 3-D point cloud."""
    return np.mean(np.asarray(points, dtype=float), axis=0)


# ---------------------------------------------------------------------------
# Kernel shape — regular tetrahedron  (3-D analogue of equilateral triangle)
# ---------------------------------------------------------------------------

def regular_tetrahedron_3d(center: np.ndarray, radius: float) -> np.ndarray:
    """
    Return 4 vertices of a regular tetrahedron with circumradius *radius*
    centred at *center*.  Returns a (4, 3) array.
    """
    center = np.asarray(center, dtype=float)
    verts = np.array([
        [ 0.0,                   0.0,                  1.0      ],
        [ 2.0 * np.sqrt(2) / 3,  0.0,                 -1.0 / 3  ],
        [-      np.sqrt(2) / 3,  np.sqrt(6) / 3,      -1.0 / 3  ],
        [-      np.sqrt(2) / 3, -np.sqrt(6) / 3,      -1.0 / 3  ],
    ], dtype=float)
    return center + radius * verts


# ---------------------------------------------------------------------------
# Convex-hull intersection test  (used for proxy re-clustering)
# ---------------------------------------------------------------------------

def polygons_intersect_3d(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
) -> bool:
    """
    Return True if the 3-D convex hulls of *pts_a* and *pts_b* intersect.

    Uses a vertex-membership test:
      • Any vertex of hull A inside hull B  →  intersecting
      • Any vertex of hull B inside hull A  →  intersecting

    Sufficient for the re-clustering step where overlapping proxies share
    interior points.
    """
    pts_a = np.asarray(pts_a, dtype=float)
    pts_b = np.asarray(pts_b, dtype=float)

    _, eq_a = convex_hull_with_equations_3d(pts_a)
    _, eq_b = convex_hull_with_equations_3d(pts_b)

    if eq_a is None or eq_b is None:
        return False

    for v in pts_a:
        if point_in_convex_hull_3d(pts_b, v, equations=eq_b):
            return True
    for v in pts_b:
        if point_in_convex_hull_3d(pts_a, v, equations=eq_a):
            return True
    return False
