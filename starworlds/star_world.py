"""
Algorithm 2 — Forming Disjoint Star Worlds (Paper §V).

Entry point:
    create_star_world(obstacles, x, xg, prev_result, L) → StarWorldResult

The algorithm iterates three steps until cluster count stabilises:
  1. Admissible kernel per cluster         (§V-A.1)
  2. Kernel selection + starshaped hull    (§V-A.2, §V-C)
  3. Re-clustering from proxy intersections (§V-A.3)

Early exit (line 7 in Algorithm 2):
  If any cluster's admissible kernel is empty, all obstacles are returned as
  their convex decompositions — an intersecting star world.

Data structures
---------------
clusters : list of frozenset{int}  — original obstacle indices per cluster
AK cache : dict[obs_idx][excl_idx] → cone polygon (ndarray or None)
  Computed once before the loop with obstacles indexed by their position in
  the input list, and excluded_points = [x, xg].

Paper: Algorithm 2, §V.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, FrozenSet, Dict, Tuple, Any


@dataclass
class ObstacleProxy:
    """
    One starshaped proxy obstacle O'_i produced by Algorithm 2.

    Attributes
    ----------
    pieces          : list of convex polygon vertex arrays (union = O'_i)
    kernel_K        : (3, 2) ndarray — the specified kernel points K_i
    center_point    : (2,) ndarray  — x_{c,i} ∈ int ker(O'_i) \\ l(x, x_g)
    original_indices: sorted list of original obstacle indices in this cluster
    is_fallback     : True if this proxy comes from convex decomposition fallback
    """
    pieces: List[np.ndarray]
    kernel_K: Optional[np.ndarray]
    center_point: np.ndarray
    original_indices: List[int]
    is_fallback: bool = False


@dataclass
class StarWorldResult:
    """
    Full output of Algorithm 2.

    Attributes
    ----------
    proxies    : list of ObstacleProxy — one per cluster (the O' collection)
    is_disjoint: True if the result is a disjoint star world (Problem 1 solution)
    clusters   : cluster assignment (for temporal coherence at next timestep)
    """
    proxies: List[ObstacleProxy]
    is_disjoint: bool
    clusters: List[FrozenSet[int]]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_star_world(
    obstacles: list,
    x: np.ndarray,
    xg: np.ndarray,
    prev_result: Optional[StarWorldResult] = None,
    L: float = 1e5,
    n_approx: int = 64,
    max_iter: int = 20,
) -> StarWorldResult:
    """
    Reshape a workspace of (possibly intersecting) obstacles into a disjoint
    star world using Algorithm 2.

    Parameters
    ----------
    obstacles   : list of Obstacle objects (Ellipse, ConvexPolygon2D, Polygon2D).
                  Non-convex Polygon2D objects are internally convex-decomposed.
    x           : robot position, shape (2,).
    xg          : goal position, shape (2,).
    prev_result : output from the previous timestep (for temporal coherence).
    L           : bounding distance for cone polygons (workspace scale × 10).
    n_approx    : polygon approximation resolution for ellipses.
    max_iter    : safety cap on the clustering loop (should be ≤ 5 in practice).

    Returns
    -------
    StarWorldResult with proxies, is_disjoint flag, and cluster assignments.

    Paper: Algorithm 2, §V.
    """
    from .admissible_kernel import build_ak_cache, cluster_admissible_kernel_2d, feasible_kernel_region
    from .starshaped_hull import (
        sh_kernel_cluster_2d,
        convex_decompose_obstacle,
        proxy_pieces_intersect,
        center_point_from_kernel,
    )
    from .kernel_selection import select_kernel_2d

    x  = np.asarray(x,  dtype=float)
    xg = np.asarray(xg, dtype=float)
    n_obs = len(obstacles)

    # Flatten non-convex obstacles into convex pieces, tracking origin index
    flat_obstacles, flat_origins = _flatten_obstacles(obstacles, n_approx)

    # ------------------------------------------------------------------ #
    # Pre-compute AK for each (flat obstacle, excluded point) pair        #
    # Paper: §V-A.1 "computed once outside the loop"                     #
    # ------------------------------------------------------------------ #
    excluded_points = [x, xg]
    ak_cache = build_ak_cache(flat_obstacles, excluded_points, L=L)

    # Build previous centroid map: frozenset(orig_indices) → centroid
    prev_centroids: Dict[FrozenSet[int], np.ndarray] = {}
    if prev_result is not None:
        for proxy in prev_result.proxies:
            key = frozenset(proxy.original_indices)
            prev_centroids[key] = proxy.center_point

    # ------------------------------------------------------------------ #
    # Initialise: each flat obstacle is its own cluster                   #
    # ------------------------------------------------------------------ #
    clusters: List[FrozenSet[int]] = [frozenset([i]) for i in range(len(flat_obstacles))]

    for iteration in range(max_iter):
        proxies: List[Tuple[List[np.ndarray], Optional[np.ndarray], List[int]]] = []
        early_exit = False

        for cl in clusters:
            cl_list = sorted(cl)
            cl_obstacles = [flat_obstacles[i] for i in cl_list]

            # ---------------------------------------------------------- #
            # Step 1: Admissible kernel for this cluster (Eq. 7)          #
            # ---------------------------------------------------------- #
            ak = cluster_admissible_kernel_2d(ak_cache, cl_list, [0, 1])

            if ak is None:
                # Line 7: empty AK → early exit with convex decomposition
                early_exit = True
                break

            # ---------------------------------------------------------- #
            # Step 2a: Feasible kernel region S = AK ∩ cl∪               #
            # ---------------------------------------------------------- #
            feasible = feasible_kernel_region(ak, cl_obstacles, n_approx=n_approx)

            if feasible is None:
                early_exit = True
                break

            # ---------------------------------------------------------- #
            # Step 2b: Kernel point selection (Algorithm 3)               #
            # ---------------------------------------------------------- #
            # Build original-obstacle key for temporal coherence
            orig_indices = sorted({flat_origins[i] for i in cl_list})
            prev_k = prev_centroids.get(frozenset(orig_indices))

            K, kc = select_kernel_2d(feasible, x, xg, k_prev_centroid=prev_k)

            if K is None:
                early_exit = True
                break

            # ---------------------------------------------------------- #
            # Step 2c: Starshaped hull (Eq. 11, Property 4d)             #
            # ---------------------------------------------------------- #
            pieces = sh_kernel_cluster_2d(cl_obstacles, K, n_approx=n_approx)
            proxies.append((pieces, K, orig_indices))

        if early_exit:
            break

        # ------------------------------------------------------------------ #
        # Step 3: Re-clustering (Algorithm 2 lines 10-13)                    #
        # ------------------------------------------------------------------ #
        new_clusters = _recluster(proxies, flat_origins, len(flat_obstacles))

        if len(new_clusters) == len(clusters):
            # Cluster count unchanged → algorithm converged
            break

        clusters = new_clusters

    if early_exit:
        return _fallback_result(obstacles, x, xg, n_approx)

    # Build result
    result_proxies = []
    for (pieces, K, orig_idx) in proxies:
        if not pieces or K is None:
            continue
        xc = center_point_from_kernel(K, x, xg)
        result_proxies.append(ObstacleProxy(
            pieces=pieces,
            kernel_K=K,
            center_point=xc,
            original_indices=orig_idx,
            is_fallback=False,
        ))

    # Check disjointness: no pair of proxies should share interior points
    is_disjoint = _check_disjoint(result_proxies)
    result_clusters = [frozenset(p.original_indices) for p in result_proxies]

    return StarWorldResult(
        proxies=result_proxies,
        is_disjoint=is_disjoint,
        clusters=result_clusters,
    )


# ---------------------------------------------------------------------------
# Re-clustering (Algorithm 2 lines 10-13)
# ---------------------------------------------------------------------------

def _recluster(
    proxies: list,
    flat_origins: List[int],
    n_flat: int,
) -> List[FrozenSet[int]]:
    """
    Merge flat-obstacle clusters whose proxy pieces intersect.

    Clustering is driven by the CURRENT proxy geometry (O') but applied to the
    ORIGINAL flat-obstacle indices (line 13: "clustering applied on O but
    determined by intersection of O'").

    Uses union-find for O(n α(n)) merging.

    Paper: Algorithm 2 line 13, §V-A.3.
    """
    n = len(proxies)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    from .starshaped_hull import proxy_pieces_intersect
    for i in range(n):
        for j in range(i + 1, n):
            if proxy_pieces_intersect(proxies[i][0], proxies[j][0]):
                union(i, j)

    # Group by root
    groups: Dict[int, FrozenSet[int]] = {}
    for i, (pieces, K, orig_idx) in enumerate(proxies):
        root = find(i)
        if root not in groups:
            groups[root] = frozenset()
        # Map original indices back to flat indices
        flat_idx = frozenset(
            j for j in range(n_flat) if flat_origins[j] in orig_idx
        )
        groups[root] = groups[root] | flat_idx

    return list(groups.values())


# ---------------------------------------------------------------------------
# Fallback: convex decomposition (Eq. 14)
# ---------------------------------------------------------------------------

def _fallback_result(
    obstacles: list,
    x: np.ndarray,
    xg: np.ndarray,
    n_approx: int,
) -> StarWorldResult:
    """
    Algorithm 2 line 7: return O' = {CD(O_i) : O_i ∈ O}.
    Each obstacle is replaced by its convex decomposition pieces.
    Result is an intersecting star world (is_disjoint=False).

    Paper: Eq. (14), §V-A.1.
    """
    from .starshaped_hull import convex_decompose_obstacle, center_point_from_kernel

    proxies = []
    for i, obs in enumerate(obstacles):
        pieces = convex_decompose_obstacle(obs, n_approx)
        if not pieces:
            continue
        # For fallback, no kernel triangle — use obstacle centroid as center
        centroid = np.vstack(pieces).mean(axis=0)
        proxies.append(ObstacleProxy(
            pieces=pieces,
            kernel_K=None,
            center_point=centroid,
            original_indices=[i],
            is_fallback=True,
        ))

    clusters = [frozenset(p.original_indices) for p in proxies]
    return StarWorldResult(proxies=proxies, is_disjoint=False, clusters=clusters)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_obstacles(
    obstacles: list,
    n_approx: int,
) -> Tuple[list, List[int]]:
    """
    Convex-decompose any non-convex Polygon2D into convex pieces.
    Returns (flat_obstacles, flat_origins) where flat_origins[i] is the
    index in `obstacles` that flat_obstacles[i] came from.

    Paper: §III, Problem 1 — non-convex obstacles modelled as union of convex parts.
    """
    from .obstacles import ConvexPolygon2D
    from .geometry import convex_decompose_2d

    flat_obstacles = []
    flat_origins = []

    for idx, obs in enumerate(obstacles):
        if obs.is_convex:
            flat_obstacles.append(obs)
            flat_origins.append(idx)
        else:
            verts = obs.polygon_approximation()
            tri_list = convex_decompose_2d(verts)
            if not tri_list:
                flat_obstacles.append(obs)
                flat_origins.append(idx)
            else:
                for tri_verts in tri_list:
                    flat_obstacles.append(ConvexPolygon2D(tri_verts))
                    flat_origins.append(idx)

    return flat_obstacles, flat_origins


def _check_disjoint(proxies: List[ObstacleProxy]) -> bool:
    """
    Verify that no two proxy obstacles share interior points.
    Returns True if all proxies are pairwise disjoint.
    """
    from .starshaped_hull import proxy_pieces_intersect
    n = len(proxies)
    for i in range(n):
        for j in range(i + 1, n):
            if proxy_pieces_intersect(proxies[i].pieces, proxies[j].pieces):
                return False
    return True
