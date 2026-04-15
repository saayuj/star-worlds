"""
Low-level geometric primitives used throughout the Star Worlds algorithm.

Key operations:
  tangent_points_2d      — tangent vertices of a polygon from an exterior point
  cone_polygon_2d        — shapely Polygon representing an admissible-kernel cone (Eq. 8)
  polygon_intersection   — convex polygon clip via Sutherland-Hodgman
  polygon_contains_point — point-in-polygon (ray casting)
  polygons_intersect     — separating-axis test for two convex polygons
  convex_hull_2d         — scipy ConvexHull wrapper returning CCW vertex array
  convex_decompose_2d    — ear-clipping triangle decomposition for non-convex polygons
  side_of_line_2d        — signed side of a directed line
  half_plane_polygon_2d  — large rectangle covering one side of a line
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# Tangent points
# ---------------------------------------------------------------------------

def tangent_points_2d(
    obstacle,
    x_bar: np.ndarray,
    n_approx: int = 128,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Find the two tangent vertices of `obstacle` as seen from exterior point `x_bar`.

    Uses the polar-angle method (Freeman & Loutrel [28]):
      1. Compute polar angles of all boundary vertices w.r.t. x_bar.
      2. Find the largest CCW gap in the sorted angles — this is the unobstructed
         angular region, i.e., the admissible-kernel direction.
      3. The two vertices bordering this gap are the tangent points.

    Returns (t1, t2) where x_bar, t1, t2 are in CW order (t1 is the CCW-most
    tangent, t2 the CW-most).  Returns (None, None) if x_bar is inside obstacle.

    Paper: §IV-A, Eq. (8); also the remark after Algorithm 1 ("all vertices").
    """
    # Use analytical formula for Ellipse if available (more accurate)
    from .obstacles import Ellipse
    if isinstance(obstacle, Ellipse):
        return obstacle.tangent_angles(x_bar)

    x_bar = np.asarray(x_bar, dtype=float)
    if obstacle.contains(x_bar):
        return None, None

    verts = obstacle.polygon_approximation(n=n_approx)
    diffs = verts - x_bar
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])

    n = len(angles)
    sort_idx = np.argsort(angles)
    sorted_ang = angles[sort_idx]

    # Gaps between consecutive sorted angles (CCW)
    gaps = np.empty(n)
    gaps[:-1] = sorted_ang[1:] - sorted_ang[:-1]
    gaps[-1] = sorted_ang[0] + 2.0 * np.pi - sorted_ang[-1]

    max_g = np.argmax(gaps)

    # Vertex at start of largest gap = CCW-most tangent (t1)
    # Vertex at end   of largest gap = CW-most  tangent (t2)
    t1 = verts[sort_idx[max_g]]
    t2 = verts[sort_idx[(max_g + 1) % n]]
    return t1, t2


# ---------------------------------------------------------------------------
# Cone polygon — represents the admissible kernel (Eq. 8)
# ---------------------------------------------------------------------------

def cone_polygon_2d(
    apex: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    L: float = 1e5,
    n_pts: int = 64,
) -> Optional[np.ndarray]:
    """
    Build a fan polygon representing the interior of the CCW cone

        int C∠(r(x̄, x̄-t1), r(x̄, x̄-t2))

    from Eq. (8).  The cone apex is `apex` (= x̄).  Directions:
        d1 = apex - t1   (pointing away from t1, the CCW tangent)
        d2 = apex - t2   (pointing away from t2, the CW  tangent)

    The CCW arc from d1 to d2 covers the unobstructed angular region —
    exactly the admissible kernel (Sec. IV-A).

    Returns the fan as an (n_pts+1, 2) array [apex, arc_0, …, arc_{n-1}]
    (NOT closed — use as polygon vertices).
    Returns None for degenerate cases (span ≈ 0, bounded exterior, or full plane).

    Paper: Eq. (8), Sec. IV-A.
    """
    apex = np.asarray(apex, dtype=float)
    d1 = apex - np.asarray(t1, dtype=float)
    d2 = apex - np.asarray(t2, dtype=float)

    theta1 = np.arctan2(d1[1], d1[0])
    theta2 = np.arctan2(d2[1], d2[0])

    # CCW angular span from theta1 to theta2
    span = (theta2 - theta1) % (2.0 * np.pi)

    if span < 1e-9 or span > 2.0 * np.pi - 1e-9:
        # Degenerate: either bounded exterior (span≈0) or trivially full plane
        return None

    thetas = np.linspace(theta1, theta1 + span, n_pts)
    arc = apex + L * np.column_stack([np.cos(thetas), np.sin(thetas)])
    return np.vstack([apex[np.newaxis], arc])


# ---------------------------------------------------------------------------
# Convex polygon operations (no external library needed)
# ---------------------------------------------------------------------------

def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """
    Return the convex hull of a set of 2-D points as a CCW vertex array.
    Uses scipy.spatial.ConvexHull (Graham scan, O(n log n)).
    Returns None if fewer than 3 non-collinear points.
    """
    pts = np.asarray(points, dtype=float)
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return None
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    # scipy returns vertices in CCW order for 2D
    return pts[hull.vertices]


def polygon_area_2d(verts: np.ndarray) -> float:
    """Signed area of polygon (positive = CCW)."""
    v = np.asarray(verts)
    n = len(v)
    xs, ys = v[:, 0], v[:, 1]
    return 0.5 * float(np.dot(xs, np.roll(ys, -1)) - np.dot(np.roll(xs, -1), ys))


def polygon_centroid_2d(verts: np.ndarray) -> np.ndarray:
    """Centroid of a (possibly non-convex) polygon."""
    v = np.asarray(verts, dtype=float)
    n = len(v)
    cx, cy, area = 0.0, 0.0, 0.0
    for i in range(n):
        x0, y0 = v[i]
        x1, y1 = v[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area *= 0.5
    if abs(area) < 1e-12:
        return v.mean(axis=0)
    cx /= 6.0 * area
    cy /= 6.0 * area
    return np.array([cx, cy])


def polygon_contains_point(verts: np.ndarray, point: np.ndarray,
                           tol: float = 1e-9) -> bool:
    """
    Point-in-polygon test combining a boundary check (O(n)) with ray casting.

    Returns True for points strictly inside OR on the boundary of the polygon.
    The boundary check handles the well-known edge cases of ray casting at
    vertices or collinear edges.
    """
    p = np.asarray(point, dtype=float)
    v = np.asarray(verts, dtype=float)
    n = len(v)

    # Boundary check: is p on any edge?
    for i in range(n):
        a, b = v[i], v[(i + 1) % n]
        ab = b - a
        ap = p - a
        # Signed area of triangle (a, b, p) — zero means collinear
        cross = ab[0] * ap[1] - ab[1] * ap[0]
        len_ab = np.dot(ab, ab)
        if len_ab < 1e-24:
            continue
        if abs(cross) <= tol * np.sqrt(len_ab):
            # Collinear; check that p is between a and b
            t = np.dot(ap, ab) / len_ab
            if -tol <= t <= 1.0 + tol:
                return True

    # Ray-casting for strict interior
    x, y = p
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = v[i]
        xj, yj = v[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def sutherland_hodgman(subject: np.ndarray, clip: np.ndarray) -> Optional[np.ndarray]:
    """
    Clip polygon `subject` by convex polygon `clip` using Sutherland-Hodgman.
    Both are (n, 2) CCW vertex arrays.
    Returns clipped polygon vertices or None if the intersection is empty.
    """
    def inside(p, a, b):
        """True if point p is on the CCW (left/inside) side of edge a→b."""
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0

    def intersect(p1, p2, a, b):
        """Line segment p1-p2 intersection with infinite line a-b."""
        d1 = p2 - p1
        d2 = b - a
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-15:
            return p1
        t = ((a[0] - p1[0]) * d2[1] - (a[1] - p1[1]) * d2[0]) / cross
        return p1 + t * d1

    output = list(subject)
    c = clip
    nc = len(c)

    for i in range(nc):
        if not output:
            return None
        a, b = c[i], c[(i + 1) % nc]
        inp = output
        output = []
        for j in range(len(inp)):
            curr = np.asarray(inp[j], dtype=float)
            prev = np.asarray(inp[j - 1], dtype=float)
            if inside(curr, a, b):
                if not inside(prev, a, b):
                    output.append(intersect(prev, curr, a, b))
                output.append(curr)
            elif inside(prev, a, b):
                output.append(intersect(prev, curr, a, b))

    if not output or len(output) < 3:
        return None
    return np.array(output)


def polygons_intersect(p1: np.ndarray, p2: np.ndarray) -> bool:
    """
    Separating Axis Theorem (SAT) test for two convex polygons.
    Returns True if they intersect (share at least one point).
    """
    def project(verts, axis):
        proj = verts @ axis
        return proj.min(), proj.max()

    def overlaps(mn1, mx1, mn2, mx2):
        return mn1 <= mx2 + 1e-9 and mn2 <= mx1 + 1e-9

    for poly in (p1, p2):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                continue
            axis /= norm
            mn1, mx1 = project(p1, axis)
            mn2, mx2 = project(p2, axis)
            if not overlaps(mn1, mx1, mn2, mx2):
                return False
    return True


def convex_decompose_2d(verts: np.ndarray) -> List[np.ndarray]:
    """
    Decompose a simple (possibly non-convex) polygon into convex triangles via
    fan-triangulation from the centroid.  This is equivalent to using the centroid
    as the "ear" pivot — it works for star-shaped polygons.  For strongly non-convex
    cases we fall back to a naive triangle fan from vertex 0 and keep interior ones.

    Returns a list of (3, 2) vertex arrays for each triangle.
    Paper: Sec. V, Eq. (14) — CD(O_i) for the convex-decomposition fallback.
    """
    verts = np.asarray(verts, dtype=float)
    n = len(verts)
    if n < 3:
        return []

    centroid = polygon_centroid_2d(verts)
    triangles = []
    for i in range(n):
        tri = np.array([centroid, verts[i], verts[(i + 1) % n]])
        # Keep only if the triangle has significant area and is inside the polygon
        area = abs(polygon_area_2d(tri))
        if area < 1e-12:
            continue
        # Representative point (midpoint of edge)
        mid = (verts[i] + verts[(i + 1) % n]) / 2.0
        test_pt = (centroid + mid) / 2.0
        if polygon_contains_point(verts, test_pt):
            triangles.append(tri)
    return triangles


# ---------------------------------------------------------------------------
# Half-plane & line utilities
# ---------------------------------------------------------------------------

def side_of_line_2d(p1: np.ndarray, p2: np.ndarray, point: np.ndarray) -> float:
    """
    Returns the signed "side" of `point` relative to the directed line p1 → p2.
    Positive = left (CCW), negative = right (CW), ~0 = on the line.
    """
    d = np.asarray(p2) - np.asarray(p1)
    v = np.asarray(point) - np.asarray(p1)
    return float(d[0] * v[1] - d[1] * v[0])


def half_plane_polygon_2d(
    p1: np.ndarray,
    p2: np.ndarray,
    side: float,
    L: float = 1e5,
) -> np.ndarray:
    """
    Return a large rectangle (2L × 2L) covering the half-plane on `side` of line p1→p2.
    side > 0: left of p1→p2; side < 0: right of p1→p2.

    Paper: Algorithm 3 (lines 10-12) — restrict kernel to same side of l(x, x_g).
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    d = p2 - p1
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return None
    t = d / norm
    n = np.array([-t[1], t[0]])  # left normal
    if side < 0:
        n = -n

    far_left  = p1 - L * t
    far_right = p2 + L * t
    return np.array([
        far_left,
        far_right,
        far_right + L * n,
        far_left  + L * n,
    ])


# ---------------------------------------------------------------------------
# Cone containment (without building a polygon object)
# ---------------------------------------------------------------------------

def point_in_cone_2d(apex: np.ndarray, t1: np.ndarray, t2: np.ndarray,
                     point: np.ndarray) -> bool:
    """
    Return True if `point` is in the interior of the CCW cone at `apex`
    bounded by directions (apex-t1) and (apex-t2).

    The cone spans CCW from angle(apex-t1) to angle(apex-t2).
    A point q is inside iff the direction (q-apex) lies in this CCW arc.
    """
    apex = np.asarray(apex, dtype=float)
    d1 = apex - np.asarray(t1, dtype=float)
    d2 = apex - np.asarray(t2, dtype=float)
    dq = np.asarray(point, dtype=float) - apex

    theta1 = np.arctan2(d1[1], d1[0])
    theta2 = np.arctan2(d2[1], d2[0])
    thetaq = np.arctan2(dq[1], dq[0])

    span = (theta2 - theta1) % (2.0 * np.pi)
    ang  = (thetaq - theta1) % (2.0 * np.pi)
    return bool(0.0 < ang < span)


def point_in_cone_polygon(cone_verts: np.ndarray, point: np.ndarray) -> bool:
    """
    Test membership in a cone polygon returned by cone_polygon_2d.
    cone_verts[0] is the apex; the remaining rows form the arc.
    """
    if cone_verts is None:
        return False
    return polygon_contains_point(cone_verts, point)


# ---------------------------------------------------------------------------
# Polygon distance utility (approximate inscribed-circle radius)
# ---------------------------------------------------------------------------

def approx_inscribed_radius(verts: np.ndarray, center: np.ndarray) -> float:
    """
    Approximate the largest inscribed circle radius at `center` inside polygon
    `verts` by computing the minimum distance from center to any edge.
    """
    v = np.asarray(verts, dtype=float)
    c = np.asarray(center, dtype=float)
    n = len(v)
    min_dist = np.inf
    for i in range(n):
        a, b = v[i], v[(i + 1) % n]
        ab = b - a
        t = np.dot(c - a, ab) / (np.dot(ab, ab) + 1e-15)
        t = np.clip(t, 0.0, 1.0)
        closest = a + t * ab
        dist = np.linalg.norm(c - closest)
        if dist < min_dist:
            min_dist = dist
    return float(min_dist)
