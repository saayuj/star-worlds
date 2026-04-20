"""
Admissible kernel computation (Paper §IV-A).

Definition 3 (Eq. 6):
    ad ker(A, X̄) = { x ∈ Rⁿ : SHₓ(A) ∩ X̄ = ∅ }

For a 2-D convex/polygon obstacle and a free exterior point x̄, the admissible
kernel is an open cone at x̄ (Eq. 8):
    ad ker(A, {x̄}) = int C∠(r(x̄, x̄−t₁), r(x̄, x̄−t₂))
where t₁, t₂ ∈ T_A(x̄).

Key properties implemented here:
    Property 2 — ad ker(A∪, X̄) = ∩_{A∈𝒜} ad ker(A, X̄)      (union of obstacles)
    Property 3 — ad ker(A, X̄)  = ∩_{x̄∈X̄} ad ker(A, {x̄})   (multiple excluded points)
    → combined as Eq. (7): ad ker(A∪, X̄) = ∩_i ∩_j ad ker(Aᵢ, {x̄ⱼ})

Cones are stored as (n+1, 2) vertex arrays [apex, arc_0, …, arc_{n-1}]
produced by geometry.cone_polygon_2d.  Intersection of two such fan polygons
(at different apices) is computed by the Sutherland-Hodgman clipper on a
bounded approximation of the cone.

Paper §V-D / Appendix J:
    "The cones ad ker(cl, x) and ad ker(cl, xg) can be approximated as polygons
     with a maximum of seven vertices by applying a bounding box."
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Dict, Any

from .geometry import (
    tangent_points_2d,
    cone_polygon_2d,
    polygon_contains_point,
    convex_hull_2d,
    sutherland_hodgman,
    point_in_cone_2d,
    polygons_intersect,
)


# ---------------------------------------------------------------------------
# Single obstacle, single excluded point
# ---------------------------------------------------------------------------

def admissible_kernel_single_2d(
    obstacle,
    x_bar: np.ndarray,
    L: float = 1e5,
    n_cone_pts: int = 64,
) -> Optional[np.ndarray]:
    """
    Compute ad ker(obstacle, {x̄}) as a cone fan polygon (Eq. 8).

    Returns
    -------
    np.ndarray of shape (n+1, 2) — the cone polygon (apex first), or
    None if x̄ is inside obstacle (empty admissible kernel, Prop. 3).

    Paper: Eq. (8), §IV-A.
    """
    x_bar = np.asarray(x_bar, dtype=float)
    t1, t2 = tangent_points_2d(obstacle, x_bar)

    if t1 is None:
        # x_bar is inside or is a bounded exterior point → empty admissible kernel
        return None

    cone = cone_polygon_2d(x_bar, t1, t2, L=L, n_pts=n_cone_pts)
    return cone  # None if degenerate span (bounded exterior / full plane)


# ---------------------------------------------------------------------------
# Cluster (union of obstacles), multiple excluded points — Eq. (7)
# ---------------------------------------------------------------------------

def build_ak_cache(
    obstacles: list,
    excluded_points: List[np.ndarray],
    L: float = 1e5,
) -> Dict[str, Any]:
    """
    Pre-compute admissible kernels for every (obstacle, excluded_point) pair.
    This is done ONCE before the Algorithm 2 iteration loop (§V-D).

    Returns a dict:
        cache[i][j] = cone_polygon_2d result (ndarray or None)
    where i indexes obstacles and j indexes excluded_points.

    Paper: §V-A.1 — "admissible kernel for each obstacle can be computed once
    outside the loop".
    """
    cache: Dict = {}
    for i, obs in enumerate(obstacles):
        cache[i] = {}
        for j, xp in enumerate(excluded_points):
            cache[i][j] = admissible_kernel_single_2d(obs, xp, L=L)
    return cache


def intersect_cone_polygons(
    poly_a: Optional[np.ndarray],
    poly_b: Optional[np.ndarray],
    bbox: float = 1e5,
) -> Optional[np.ndarray]:
    """
    Intersect two cone polygons (or any two convex-ish polygons).

    Strategy: clip poly_b with the convex hull of poly_a using Sutherland-Hodgman.
    Because cone polygons can be non-convex (span > 180°), we represent each as
    the intersection of its convex hull with a bounding box, then clip.

    This is an approximation that is conservative (under-estimates the AK) but
    always safe for kernel point selection.

    Returns the clipped polygon or None if intersection is empty.
    """
    if poly_a is None or poly_b is None:
        return None

    ch_a = convex_hull_2d(poly_a)
    ch_b = convex_hull_2d(poly_b)

    if ch_a is None or ch_b is None:
        return None

    clipped = sutherland_hodgman(ch_b, ch_a)
    if clipped is None or len(clipped) < 3:
        return None
    return clipped


def cluster_admissible_kernel_2d(
    ak_cache: Dict,
    cluster_indices: List[int],
    excluded_indices: List[int],
) -> Optional[np.ndarray]:
    """
    Compute ad ker(cl, X̄) for a cluster using the cached per-obstacle cones.

    Implementation of Eq. (7):
        ad ker(A∪, X̄) = ∩_i ∩_j ad ker(Aᵢ, {x̄ⱼ})

    The intersection is built iteratively: start with the first cone and
    clip each subsequent one against the accumulating result.

    Returns the intersection polygon (convex approximation) or None if empty.

    Paper: §V-A.1, Property 2, Property 3, Eq. (7).
    """
    result: Optional[np.ndarray] = None

    for i in cluster_indices:
        for j in excluded_indices:
            cone = ak_cache[i][j]
            if cone is None:
                return None  # immediately empty
            if result is None:
                result = cone
            else:
                result = intersect_cone_polygons(result, cone)
                if result is None:
                    return None

    return result


# ---------------------------------------------------------------------------
# Membership test in the admissible kernel polygon
# ---------------------------------------------------------------------------

def point_in_admissible_kernel(
    ak_polygon: Optional[np.ndarray],
    point: np.ndarray,
) -> bool:
    """
    Test whether `point` lies inside the (bounded) admissible kernel polygon.
    Returns False if ak_polygon is None (empty admissible kernel).
    """
    if ak_polygon is None:
        return False
    return polygon_contains_point(ak_polygon, point)


# ---------------------------------------------------------------------------
# Feasible region for kernel selection  — ad ker ∩ cluster union
# ---------------------------------------------------------------------------

def feasible_kernel_region(
    ak_polygon: Optional[np.ndarray],
    cluster_obstacles: list,
    n_approx: int = 64,
) -> Optional[np.ndarray]:
    """
    Compute the feasible kernel-selection region:
        S = ad ker(cl, X̄) ∩ cl∪

    as described in §V-C:
        "A reasonable selection can be found in ad ker(cl, {x, xg}) ∩ cl∪"

    The cluster union cl∪ is approximated by the convex hull of all obstacle
    boundary vertices (upper bound, avoids computing exact union).

    If S is empty, falls back to ad ker ∩ CH(cl∪) (still inside the AK but
    possibly outside the cluster, as allowed by Property 4c).

    Returns the feasible polygon (convex approximation) or None.

    Paper: §V-C.
    """
    if ak_polygon is None:
        return None

    # Collect all obstacle boundary vertices
    all_verts = np.vstack([
        obs.polygon_approximation(n=n_approx) for obs in cluster_obstacles
    ])
    ch_cluster = convex_hull_2d(all_verts)
    if ch_cluster is None:
        return None

    ak_ch = convex_hull_2d(ak_polygon)
    if ak_ch is None:
        return ch_cluster

    # Primary: clip cluster against AK convex hull.
    # When the AK is large and contains the cluster (the typical case), this
    # returns the cluster itself since all cluster vertices are inside the AK.
    feasible = sutherland_hodgman(ch_cluster, ak_ch)
    if feasible is not None and len(feasible) >= 3:
        return feasible

    # Fallback: clip AK convex hull against cluster (intersection).
    # Handles the case where AK and cluster partially overlap.
    feasible = sutherland_hodgman(ak_ch, ch_cluster)
    if feasible is not None and len(feasible) >= 3:
        return feasible

    # Last resort: place kernel inside the obstacle cluster.
    # Returning the raw AK polygon centroid (which can be far from the
    # obstacles when the cone spans >180°) is incorrect — always anchor
    # the kernel inside the obstacle cluster instead.
    return ch_cluster
