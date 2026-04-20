"""
Starshaped hull construction (Paper §IV-B and §V-D).

Implements:

  sh_kernel_obstacle_2d(obstacle, K)
      Eq. (11): SH_kerK(A_conv) = A_conv ∪ CH(∪_{k∈K} T_k(A_conv) ∪ {k})
      Valid for any convex 2-D obstacle (Ellipse, ConvexPolygon2D, or a single
      convex Polygon2D).

  sh_kernel_cluster_2d(obstacles, K)
      Property 4d: SH_kerK(A∪) = ∪_i SH_kerK(A_i)
      Returns a list of convex polygon arrays (one per obstacle in the cluster),
      each representing SH_kerK(A_i) = A_i ∪ CH(tangent_pts_i ∪ K).

  is_strictly_starshaped_2d(proxy_pieces, kernel_K)
      Sanity check: verifies that all kernel points see the entire proxy set
      (Proposition 4 — satisfied by construction when |K| = 3 with n+1=3
      affinely independent points in R²).

Notes on the "all vertices" remark (Algorithm 1, Paper §IV-B):
  When computing SH_k(P) for a single kernel point k and a polygon P, the
  standard approach (Arkin et al. [27]) considers only *convex* vertices.
  Algorithm 1 extends this to ALL vertices to handle the specified-kernel case.
  Our implementation processes all polygon vertices (via tangent_points_2d which
  uses all boundary samples), so the "all vertices" requirement is satisfied.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from .geometry import (
    tangent_points_2d,
    convex_hull_2d,
    polygon_contains_point,
    polygons_intersect,
)
from .geometry_3d import (
    convex_hull_3d,
    polygon_centroid_3d,
    polygons_intersect_3d,
)


# ---------------------------------------------------------------------------
# Single obstacle — Eq. (11)
# ---------------------------------------------------------------------------

def sh_kernel_obstacle_2d(
    obstacle,
    K: np.ndarray,
    n_approx: int = 64,
) -> Optional[np.ndarray]:
    """
    Compute SH_kerK(obstacle) for a single convex 2-D obstacle.

    Formula (Eq. 11):
        SH_kerK(A_conv) = A_conv ∪ CH( ∪_{k∈K} T_k(A_conv) ∪ {k} )

    The result is the convex hull of:
        all obstacle boundary vertices
        +  all tangent points of the obstacle from each kernel point k_i
        +  all kernel points k_i

    Per §V-D: "each original obstacle, in addition to the obstacle shape,
    contributes with the convex hull of the three kernel points and their
    respective tangent points" — 6 tangent points + 3 kernel points = 9 points
    → convex hull computed in O(1).

    Returns
    -------
    np.ndarray  shape (m, 2) — CCW vertices of SH_kerK(obstacle), or None.

    Paper: Eq. (11), §IV-B, §V-D.
    """
    K = np.asarray(K, dtype=float)
    obs_verts = obstacle.polygon_approximation(n=n_approx)

    # Start with the obstacle boundary points
    all_points = [obs_verts]

    for k in K:
        k = np.asarray(k, dtype=float)
        if obstacle.contains(k):
            # k is inside A_conv: SH_k(A_conv) = A_conv (Property 1a)
            # No extension needed; just include k itself for CH completeness
            all_points.append(k[np.newaxis])
        else:
            t1, t2 = tangent_points_2d(obstacle, k, n_approx=n_approx)
            if t1 is not None:
                all_points.append(np.array([k, t1, t2]))
            else:
                all_points.append(k[np.newaxis])

    pts = np.vstack(all_points)
    hull = convex_hull_2d(pts)
    return hull


# ---------------------------------------------------------------------------
# Cluster — Property 4d
# ---------------------------------------------------------------------------

def sh_kernel_cluster_2d(
    obstacles: list,
    K: np.ndarray,
    n_approx: int = 64,
) -> List[Optional[np.ndarray]]:
    """
    Compute SH_kerK(cl∪) decomposed as one convex piece per obstacle.

    Property 4d:
        SH_kerK(∪_i A_i) = ∪_i SH_kerK(A_i)

    Returns a list of convex polygon arrays (one per obstacle in the cluster).
    The full proxy obstacle is the union of these pieces.

    Paper: Property 4d, §V-D.
    """
    pieces = []
    for obs in obstacles:
        hull = sh_kernel_obstacle_2d(obs, K, n_approx=n_approx)
        if hull is not None:
            pieces.append(hull)
    return pieces


# ---------------------------------------------------------------------------
# Convex fallback — CD(O_i) — Eq. (14)
# ---------------------------------------------------------------------------

def convex_decompose_obstacle(obstacle, n_approx: int = 64) -> List[np.ndarray]:
    """
    Convex decomposition of a single obstacle (Eq. 14, fallback path in Alg. 2).

    For convex obstacles: returns [obstacle_vertices].
    For non-convex polygons: returns triangle fan from geometry.convex_decompose_2d.

    Paper: Eq. (14), §V.
    """
    from .geometry import convex_decompose_2d

    if obstacle.is_convex:
        verts = obstacle.polygon_approximation(n_approx)
        return [verts]
    else:
        verts = obstacle.polygon_approximation()
        return convex_decompose_2d(verts)


# ---------------------------------------------------------------------------
# Proxy intersection detection (for re-clustering)
# ---------------------------------------------------------------------------

def proxy_pieces_intersect(
    pieces_a: List[np.ndarray],
    pieces_b: List[np.ndarray],
) -> bool:
    """
    Check whether the union of convex pieces_a intersects the union of convex pieces_b.

    Tests all pairs of convex sub-pieces using the Separating Axis Theorem.
    Returns True if ANY pair intersects.

    Paper: Algorithm 2 lines 10-13 — re-clustering step.
    """
    for pa in pieces_a:
        for pb in pieces_b:
            if pa is None or pb is None:
                continue
            if polygons_intersect(pa, pb):
                return True
    return False


# ---------------------------------------------------------------------------
# 3-D starshaped hull  (Eq. 11 lifted to R³)
# ---------------------------------------------------------------------------

def sh_kernel_obstacle_3d(
    obstacle,
    K: np.ndarray,
    n_approx: int = 128,
) -> Optional[np.ndarray]:
    """
    3-D analogue of Eq. (11):  SH_kerK(A) = CH(A ∪ K).

    For a convex obstacle the starshaped hull is the convex hull of the
    obstacle surface samples together with the kernel points K.

    Returns a (m, 3) array of hull vertices, or None.
    """
    K = np.asarray(K, dtype=float).reshape(-1, 3)
    obs_pts = obstacle.polygon_approximation(n=n_approx)   # (n, 3)
    all_pts = np.vstack([obs_pts, K])
    return convex_hull_3d(all_pts)


def sh_kernel_cluster_3d(
    obstacles: list,
    K: np.ndarray,
    n_approx: int = 128,
) -> List[Optional[np.ndarray]]:
    """
    3-D Property 4d:  SH_kerK(∪_i A_i) = ∪_i SH_kerK(A_i).

    Returns one (m_i, 3) hull-vertex array per obstacle.
    """
    return [sh_kernel_obstacle_3d(obs, K, n_approx) for obs in obstacles]


def proxy_pieces_intersect_3d(
    pieces_a: List[np.ndarray],
    pieces_b: List[np.ndarray],
) -> bool:
    """Check whether two 3-D proxy unions overlap (re-clustering step)."""
    for pa in pieces_a:
        for pb in pieces_b:
            if pa is None or pb is None:
                continue
            if polygons_intersect_3d(pa, pb):
                return True
    return False


def center_point_from_kernel_3d(K: np.ndarray) -> np.ndarray:
    """
    Centroid of the 3-D kernel tetrahedron K (shape (4, 3)).

    The centroid lies in the interior of CH(K) and serves as the reference
    point x_{c,i} for the motion planner.
    """
    return np.mean(np.asarray(K, dtype=float), axis=0)


# ---------------------------------------------------------------------------
# Center point extraction for the motion planner (2-D)
# ---------------------------------------------------------------------------

def center_point_from_kernel(
    K: np.ndarray,
    x: np.ndarray,
    xg: np.ndarray,
) -> np.ndarray:
    """
    Select a valid center point x_{c,i} ∈ int ker(O'_i) \\ l(x, x_g).

    From §VI-A:
        CH(K_i) ⊂ ker(O'_i)  (Property 4a)
        x_{c,i} = centroid(K_i)  unless it lies on l(x, x_g)

    If centroid(K) is exactly on line l(x, x_g), perturb slightly.

    Paper: §VI-A.
    """
    from .geometry import side_of_line_2d

    centroid = K.mean(axis=0)
    side = side_of_line_2d(x, xg, centroid)

    if abs(side) > 1e-8:
        return centroid

    # Centroid is on the line — perturb perpendicular to l(x, x_g)
    d = np.asarray(xg) - np.asarray(x)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return centroid
    perp = np.array([-d[1], d[0]]) / norm
    # Pick the perturbation direction that keeps us in int(CH(K))
    eps = 1e-3 * np.linalg.norm(K[0] - K[1])
    for sign in (1.0, -1.0):
        candidate = centroid + sign * eps * perp
        if polygon_contains_point(K, candidate):
            return candidate
    return centroid
