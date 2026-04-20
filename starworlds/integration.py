"""
Integration interface between the Star Worlds algorithm and your motion planner / simulator.

Usage — 2-D (Ellipse / ConvexPolygon2D obstacles):
    result = updater.update(obstacles=[obs1, obs2, ...],
                            robot_pos=np.array([x, y]),
                            goal_pos=np.array([gx, gy]))

Usage — 3-D (Ellipsoid obstacles) — auto-detected from obstacle.dim == 3:
    result = updater.update(obstacles=[ell1, ell2, ...],
                            robot_pos=np.array([x, y, z]),
                            goal_pos=np.array([gx, gy, gz]))

The updater maintains temporal coherence across calls (Algorithm 3).
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from .star_world import create_star_world, create_star_world_3d, StarWorldResult


class StarWorldUpdater:
    """
    Stateful wrapper around Algorithm 2 that preserves temporal coherence.

    Parameters
    ----------
    L         : bounding radius for cone polygons (set ≥ 10 × workspace radius).
    n_approx  : boundary sample count for ellipse polygon approximation.
    max_iter  : maximum Algorithm 2 iterations (≤ 5 in all practical cases, §V-D).
    """

    def __init__(
        self,
        L: float = 1e4,
        n_approx: int = 64,
        n_approx_3d: int = 128,
        max_iter: int = 20,
    ):
        self.L = L
        self.n_approx = n_approx
        self.n_approx_3d = n_approx_3d
        self.max_iter = max_iter
        self._prev_result: Optional[StarWorldResult] = None

    # ---------------------------------------------------------------------- #
    # Primary interface                                                        #
    # ---------------------------------------------------------------------- #

    def update(
        self,
        obstacles: list,
        robot_pos,
        goal_pos,
    ) -> StarWorldResult:
        """
        Compute (or update) the disjoint star world for the current timestep.

        Parameters
        ----------
        obstacles : list of Obstacle — current workspace obstacles (already
                    inflated by robot radius + safety margin, as in §III /
                    Problem 1).
        robot_pos : array_like shape (2,) or (3,).
        goal_pos  : array_like shape (2,) or (3,).

        Returns
        -------
        StarWorldResult
            .proxies      — list of ObstacleProxy, one per cluster.
            .is_disjoint  — True if a valid disjoint star world was found.
            .clusters     — cluster assignments for next call's temporal coherence.

        Each ObstacleProxy exposes:
            .center_point    — x_{c,i} for the harmonic/contraction motion planner.
            .pieces          — list of convex polygon vertex arrays forming O'_i.
            .kernel_K        — (3, 2) kernel triangle (subset of ker(O'_i)).
        """
        x  = np.asarray(robot_pos, dtype=float)
        xg = np.asarray(goal_pos,  dtype=float)

        # Auto-detect 3-D from the first obstacle's .dim attribute
        is_3d = (
            len(obstacles) > 0
            and hasattr(obstacles[0], 'dim')
            and obstacles[0].dim == 3
        )

        if is_3d:
            result = create_star_world_3d(
                obstacles   = obstacles,
                x           = x,
                xg          = xg,
                prev_result = self._prev_result,
                n_approx    = self.n_approx_3d,
                max_iter    = self.max_iter,
            )
        else:
            result = create_star_world(
                obstacles   = obstacles,
                x           = x,
                xg          = xg,
                prev_result = self._prev_result,
                L           = self.L,
                n_approx    = self.n_approx,
                max_iter    = self.max_iter,
            )

        self._prev_result = result
        return result

    def reset(self):
        """Clear temporal-coherence state (call when obstacles change topology)."""
        self._prev_result = None

    # ---------------------------------------------------------------------- #
    # Convenience: extract per-obstacle info for a motion planner             #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def center_points(result: StarWorldResult) -> List[np.ndarray]:
        """
        Return list of center points x_{c,i} for each proxy obstacle.
        These satisfy x_{c,i} ∈ int ker(O'_i) \\ l(x, x_g) and are
        ready for direct use in Huber et al. [20,21] style planners.
        """
        return [p.center_point for p in result.proxies]

    @staticmethod
    def summary(result: StarWorldResult) -> str:
        """Human-readable one-line summary of the star world result."""
        kind = "disjoint" if result.is_disjoint else "intersecting (fallback)"
        n = len(result.proxies)
        fallback_count = sum(1 for p in result.proxies if p.is_fallback)
        return (
            f"Star world: {n} proxy obstacle(s), {kind}"
            + (f", {fallback_count} from convex-decomposition fallback" if fallback_count else "")
        )
