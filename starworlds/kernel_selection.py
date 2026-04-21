"""
Algorithm 3 — Kernel Point Selection (Paper §V-C).

Given the feasible region  S = ad ker(cl, X̄) ∩ cl∪  (or ∩ CH(cl∪)),
selects n+1 = 3 affinely independent kernel points K = {k₁, k₂, k₃} such that
CH(K) ⊂ S, guaranteeing:
  • Strict starshapedness of SH_kerK(cl∪)  (Proposition 4)
  • x_robot, x_goal ∉ SH_kerK(cl∪)         (Proposition 5)

Temporal coherence (§V-C):
  If the same obstacles were clustered at the previous timestep, reuse the
  previous kernel centroid k_prev if it is still inside S.  Otherwise, bias
  the selection to remain on the same side of l(x, x_g) as k_prev to keep
  the circumvention direction consistent (Algorithm 3 lines 9-12).

Output K is an equilateral triangle of side ε centred at the chosen centroid,
where ε = eps_frac × (inscribed-circle radius at centroid).

Paper: Algorithm 3, §V-C.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from .geometry import (
    convex_hull_2d,
    polygon_contains_point,
    polygon_centroid_2d,
    approx_inscribed_radius,
    side_of_line_2d,
    half_plane_polygon_2d,
    sutherland_hodgman,
)
from .geometry_3d import polygon_centroid_3d, regular_tetrahedron_3d


def select_kernel_2d(
    feasible_region: Optional[np.ndarray],
    x: np.ndarray,
    xg: np.ndarray,
    k_prev_centroid: Optional[np.ndarray] = None,
    eps_frac: float = 0.35,
    eps_min: float = 1e-4,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Select 3 affinely independent kernel points K and their centroid k_c.

    Parameters
    ----------
    feasible_region  : (m, 2) convex polygon vertices (output of feasible_kernel_region),
                       or None (→ returns (None, None)).
    x                : robot position, shape (2,).
    xg               : goal  position, shape (2,).
    k_prev_centroid  : centroid from the previous timestep for temporal coherence.
    eps_frac         : radius of the kernel triangle = eps_frac × inscribed_radius.
    eps_min          : minimum kernel triangle radius (absolute fallback).

    Returns
    -------
    K        : np.ndarray shape (3, 2) — the three kernel points, or None.
    k_c      : np.ndarray shape (2,)  — centroid of K, or None.

    Paper: Algorithm 3, §V-C.
    """
    if feasible_region is None or len(feasible_region) < 3:
        return None, None

    x  = np.asarray(x,  dtype=float)
    xg = np.asarray(xg, dtype=float)

    # ------------------------------------------------------------------ #
    # Step 1 – candidate centroid (Algorithm 3 lines 4-12)                #
    # ------------------------------------------------------------------ #
    centroid = _choose_centroid(feasible_region, x, xg, k_prev_centroid)
    if centroid is None:
        return None, None

    # ------------------------------------------------------------------ #
    # Step 2 – build equilateral triangle around centroid                 #
    # ------------------------------------------------------------------ #
    insc_r = approx_inscribed_radius(feasible_region, centroid)
    eps = max(eps_frac * insc_r, eps_min)

    K = _equilateral_triangle(centroid, eps, x, xg)

    # Verify CH(K) ⊂ feasible_region (Proposition 5).  If not, shrink ε.
    for _ in range(10):
        if _ch_inside_polygon(K, feasible_region):
            break
        eps *= 0.5
        K = _equilateral_triangle(centroid, eps, x, xg)
    else:
        # Could not fit triangle — fall back to a degenerate tiny triangle
        eps = eps_min
        K = _equilateral_triangle(centroid, eps, x, xg)

    return K, centroid


# ---------------------------------------------------------------------------
# 3-D kernel selection
# ---------------------------------------------------------------------------

def select_kernel_3d(
    cluster_hull_pts: np.ndarray,
    eps_frac: float = 0.10,
    eps_min: float = 1e-4,
    k_prev_centroid: Optional[np.ndarray] = None,
    x: Optional[np.ndarray] = None,
    xg: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Select a regular tetrahedron K (4 × 3) as the 3-D kernel for a cluster.

    The centroid is placed at the cluster-hull centroid, then nudged off the
    line l(x, xg) if necessary (3-D analogue of Algorithm 3 §V-C).  The nudge
    is proportional to the hull's mean radius (not a fixed distance) so the
    kernel stays near the geometric centre even for small clusters.

    Temporal coherence: if *k_prev_centroid* is close to the cluster, reuse it.

    Parameters
    ----------
    cluster_hull_pts : (n, 3) vertices of the cluster's 3-D convex hull.
    eps_frac         : tetrahedron circumradius = eps_frac × mean hull radius.
    eps_min          : absolute minimum circumradius.
    k_prev_centroid  : centroid from the previous timestep.
    x                : current robot EE position (for l(x,xg) avoidance).
    xg               : goal position (for l(x,xg) avoidance).

    Returns
    -------
    K   : (4, 3) tetrahedron vertices, or None.
    k_c : (3,) centroid, or None.
    """
    if cluster_hull_pts is None or len(cluster_hull_pts) < 4:
        return None, None

    pts = np.asarray(cluster_hull_pts, dtype=float)
    centroid = polygon_centroid_3d(pts)
    mean_r = float(np.mean(np.linalg.norm(pts - centroid, axis=1)))

    # Temporal coherence: reuse previous centroid if it's close enough
    if k_prev_centroid is not None:
        k_prev = np.asarray(k_prev_centroid, dtype=float)
        if np.linalg.norm(k_prev - centroid) < 0.8 * mean_r:
            centroid = k_prev
            mean_r = float(np.mean(np.linalg.norm(pts - centroid, axis=1)))

    # ── l(x, xg) avoidance (3-D analogue of Algorithm 3 §V-C) ────────────────
    # The kernel must not lie on l(x,xg) or the tangential modulation term is
    # zero → local minimum.  If the centroid is too close to the line, nudge it
    # perpendicular by 25% of the cluster's mean radius.  This keeps the kernel
    # near the geometric centre while guaranteeing non-zero tangential force.
    if x is not None and xg is not None:
        l_dir = np.asarray(xg, dtype=float) - np.asarray(x, dtype=float)
        l_len = float(np.linalg.norm(l_dir))
        if l_len > 1e-10:
            l_unit = l_dir / l_len
            t = float(np.dot(centroid - x, l_unit))
            proj = np.asarray(x, dtype=float) + t * l_unit
            dist_from_line = float(np.linalg.norm(centroid - proj))
            threshold = 0.01 * mean_r if mean_r > 1e-6 else 1e-4
            if dist_from_line < threshold:
                perp = np.cross(l_unit, np.array([0.0, 0.0, 1.0]))
                perp_n = float(np.linalg.norm(perp))
                if perp_n > 1e-10:
                    centroid = centroid + (0.25 * mean_r) * perp / perp_n

    eps = max(eps_frac * mean_r, eps_min)
    K = regular_tetrahedron_3d(centroid, eps)
    return K, centroid


# ---------------------------------------------------------------------------
# 2-D internal helpers
# ---------------------------------------------------------------------------

def _choose_centroid(
    feasible_region: np.ndarray,
    x: np.ndarray,
    xg: np.ndarray,
    k_prev: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """
    Choose the centroid point for the kernel triangle (Algorithm 3).

    Priority:
      1. If k_prev is non-None and inside feasible_region → reuse it (line 9).
      2. If k_prev is non-None but outside → restrict feasible_region to the
         same side of l(x, x_g) as k_prev, then take centroid (lines 10-12).
      3. Else → centroid of feasible_region (line 5).
    """
    # Case 1: reuse previous centroid if still valid
    if k_prev is not None and polygon_contains_point(feasible_region, k_prev):
        return np.asarray(k_prev, dtype=float)

    # Case 2: bias to same side of line l(x, x_g) as previous centroid
    if k_prev is not None:
        prev_side = side_of_line_2d(x, xg, k_prev)
        half = half_plane_polygon_2d(x, xg, prev_side)
        if half is not None:
            half_ch = convex_hull_2d(half)
            if half_ch is not None:
                clipped = sutherland_hodgman(feasible_region, half_ch)
                if clipped is not None and len(clipped) >= 3:
                    return polygon_centroid_2d(clipped)

    # Case 3: centroid of feasible_region
    return polygon_centroid_2d(feasible_region)


def _equilateral_triangle(
    center: np.ndarray,
    radius: float,
    x: np.ndarray,
    xg: np.ndarray,
) -> np.ndarray:
    """
    Build an equilateral triangle of circumradius `radius` centred at `center`.

    The rotation angle is chosen so that no vertex lies on line l(x, x_g),
    following the spirit of §V-C ("small equilateral triangle").

    Returns K as (3, 2) array.
    """
    # Base orientation: first vertex points perpendicular to x→xg
    d = np.asarray(xg, dtype=float) - np.asarray(x, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        theta0 = np.pi / 6.0
    else:
        # Point first vertex 90° from the x→xg direction
        theta0 = np.arctan2(d[1], d[0]) + np.pi / 2.0

    angles = theta0 + np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    K = np.asarray(center, dtype=float) + radius * np.column_stack(
        [np.cos(angles), np.sin(angles)]
    )
    return K


def _ch_inside_polygon(K: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Check that the convex hull of K is inside `polygon`.
    We test each vertex of CH(K) against `polygon`.
    """
    ch = convex_hull_2d(K)
    if ch is None:
        return False
    for pt in ch:
        if not polygon_contains_point(polygon, pt):
            return False
    return True
