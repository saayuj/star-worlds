"""
Modulation-based reactive obstacle avoidance controller (3-D).

Based on:
  Huber, Billard, Slotine. "Avoidance of Convex and Concave Obstacles with
  Convergence Ensured Through Contraction." IEEE RA-L 4(2), 2019.

  Huber, Slotine, Billard. "Avoiding Dense and Dynamic Obstacles in Enclosed
  Spaces: Application to Moving in Crowds." IEEE Trans. Robot. 38(5), 2022.

Theory
------
Each star-world proxy O'_i (from Algorithm 2) comes with a guaranteed kernel
point x_{c,i} ∈ int ker(O'_i).  The controller models O'_i as a convex
obstacle whose "center" (for the purpose of the Gamma function and the normal
direction) is x_{c,i}.

For a point x outside all proxies the Gamma function is

    Γ_i(x) = ||x - x_{c,i}|| / r_i(x)

where r_i(x) is the distance from x_{c,i} to the proxy boundary *in the
direction of (x - x_{c,i})*.  For sphere proxies (our common case) this
reduces to the sphere radius, which makes Γ_i exact.

The modulation matrix for proxy i is

    M_i(x) = E_i(x) · diag(λ_n, λ_t, λ_t) · E_i(x)^T

where E_i = [n_i | e1_i | e2_i] is an orthonormal frame with n_i the outward
normal (= (x - x_{c,i}) / ||…||).

    λ_n = 1 − η / Γ_i   → 0 at boundary (blocks penetration)
    λ_t = 1 + η / Γ_i   → amplified (slides robot around)

The combined modulation uses proximity weighting (w_i ∝ 1/Γ_i^p, normalised):

    ẋ = M(x) · f₀(x) / ||M(x) f₀(x)|| · v_max

where f₀(x) = (x_goal − x) / ||x_goal − x|| is the nominal unit velocity.
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np


class ReactiveController:
    """
    Modulation-based 3-D reactive controller for star-world obstacle avoidance.

    Parameters
    ----------
    v_max : float
        Maximum EE speed (m/s).
    eta : float
        Modulation gain.  η=1 gives λ_n→0 exactly at the boundary.
    weight_power : float
        Exponent p for proximity weighting (higher = steeper fall-off).
    safety_radius : float
        Extra safety margin added to each proxy's bounding radius when
        computing Γ.  Keeps the EE a little further from boundaries.
    goal_tolerance : float
        Distance below which the goal is considered reached.
    """

    def __init__(
        self,
        v_max: float = 0.10,
        eta: float = 1.0,
        weight_power: float = 2.0,
        safety_radius: float = 0.02,
        goal_tolerance: float = 0.05,
    ) -> None:
        self.v_max = v_max
        self.eta = eta
        self.weight_power = weight_power
        self.safety_radius = safety_radius
        self.goal_tolerance = goal_tolerance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_velocity(
        self,
        x: np.ndarray,
        xg: np.ndarray,
        proxies: list,
    ) -> np.ndarray:
        """
        Compute a modulated 3-D EE velocity that steers x toward xg while
        avoiding the given star-world proxies.

        Parameters
        ----------
        x       : (3,) current EE position.
        xg      : (3,) goal EE position.
        proxies : list of ObstacleProxy objects from StarWorldUpdater.

        Returns
        -------
        v : (3,) velocity vector (m/s), ||v|| ≤ v_max.
        """
        x = np.asarray(x, dtype=float)
        xg = np.asarray(xg, dtype=float)

        goal_vec = xg - x
        dist = float(np.linalg.norm(goal_vec))
        if dist < self.goal_tolerance:
            return np.zeros(3)

        f0 = goal_vec / dist  # nominal unit velocity toward goal

        if not proxies:
            return self.v_max * f0

        # Γ for each proxy (clamp so Γ ≥ 1 — we are always "outside")
        gammas = [max(self._gamma(x, proxy), 1.001) for proxy in proxies]

        weights = self._weights(gammas)

        # Weighted sum of modulation matrices
        M = np.zeros((3, 3))
        for w, proxy, g in zip(weights, proxies, gammas):
            M += w * self._modulation_matrix(x, proxy, g)

        v = M @ f0

        v_norm = float(np.linalg.norm(v))
        if v_norm < 1e-10:
            # Degenerate: push tangentially using cross product heuristic
            v = np.cross(f0, np.array([0.0, 0.0, 1.0]))
            v_norm = float(np.linalg.norm(v))
            if v_norm < 1e-10:
                return np.zeros(3)

        return self.v_max * v / v_norm

    def goal_reached(self, x: np.ndarray, xg: np.ndarray) -> bool:
        return float(np.linalg.norm(np.asarray(x) - np.asarray(xg))) < self.goal_tolerance

    def arm_clearances(
        self,
        arm_points: list,
        proxies: list,
    ) -> list:
        """
        Batch Gamma query for a list of arm sample points.

        Parameters
        ----------
        arm_points : list of (3,) world-space positions along the arm.
        proxies    : list of ObstacleProxy from StarWorldUpdater.

        Returns
        -------
        List of (gamma_min, closest_proxy_idx, unit_repulsion_vec) —
        one entry per arm point.  gamma_min = inf and repulsion = zeros
        when there are no proxies.
        """
        results = []
        for xp in arm_points:
            xp = np.asarray(xp, dtype=float)
            if not proxies:
                results.append((float('inf'), -1, np.zeros(3)))
                continue
            gammas = [self._gamma(xp, pr) for pr in proxies]
            idx_min = int(np.argmin(gammas))
            g_min   = gammas[idx_min]
            c = np.asarray(proxies[idx_min].center_point, dtype=float)
            d = xp - c
            d_norm = float(np.linalg.norm(d))
            rep = (d / d_norm) if d_norm > 1e-8 else np.zeros(3)
            results.append((g_min, idx_min, rep))
        return results

    # ------------------------------------------------------------------
    # Γ function
    # ------------------------------------------------------------------

    def _gamma(self, x: np.ndarray, proxy) -> float:
        """
        Normalised distance measure: Γ = 1 on proxy boundary, > 1 outside.

        Uses the DIRECTION-DEPENDENT radius r(d) = support function of the
        proxy hull in the direction d = (x - c)/||x - c||.  This is the
        distance from the kernel centre c to the proxy boundary *in the
        direction of x*, making Gamma exact for any convex proxy shape —
        not just spheres.

        Contrast with the naive bounding-sphere radius (max over all vertices)
        which always overestimates and makes the robot "feel" the proxy from
        too far away.
        """
        c = proxy.center_point
        r = self._radius_in_direction(proxy, x) + self.safety_radius
        if r < 1e-10:
            return 10.0
        d = float(np.linalg.norm(x - c))
        return max(d / r, 1e-3)

    def _radius_in_direction(self, proxy, x: np.ndarray) -> float:
        """
        Support function of the proxy hull in the direction (x - c).

        For a convex set, the boundary in direction d̂ lies at distance
        max_v { (v - c) · d̂ } from c.  This gives a direction-dependent
        radius that is always ≤ the naive bounding radius.

        For a union of convex pieces (the star-world case) we take the max
        over all pieces — the union's boundary in direction d̂ is the
        outermost piece boundary in that direction.
        """
        c = proxy.center_point
        d = x - c
        d_norm = float(np.linalg.norm(d))
        if d_norm < 1e-10:
            return self._bounding_radius(proxy)
        d_unit = d / d_norm
        if not proxy.pieces:
            return 0.1
        all_pts = np.vstack(proxy.pieces)
        # Support function: project all hull vertices onto d̂, take max
        projections = (all_pts - c) @ d_unit
        return float(max(float(np.max(projections)), 0.05))

    def _bounding_radius(self, proxy) -> float:
        """Max distance from proxy center to any hull vertex (for visualisation)."""
        if not proxy.pieces:
            return 0.1
        all_pts = np.vstack(proxy.pieces)
        return float(np.max(np.linalg.norm(all_pts - proxy.center_point, axis=1)))

    # ------------------------------------------------------------------
    # Modulation matrix
    # ------------------------------------------------------------------

    def _modulation_matrix(
        self,
        x: np.ndarray,
        proxy,
        gamma: float,
    ) -> np.ndarray:
        """
        Build 3×3 modulation matrix M_i = E D Eᵀ.

        λ_n = 1 − η/Γ   (outward-normal eigenvalue)
        λ_t = 1 + η/Γ   (tangent-space eigenvalue)
        """
        c = proxy.center_point
        d_vec = x - c
        d_norm = float(np.linalg.norm(d_vec))

        if d_norm < 1e-10:
            return np.eye(3)

        n = d_vec / d_norm  # outward normal from proxy center
        e1, e2 = _tangent_basis(n)

        lam_n = 1.0 - self.eta / max(gamma, 1e-6)
        lam_t = 1.0 + self.eta / max(gamma, 1e-6)

        E = np.column_stack([n, e1, e2])  # orthonormal, so E^{-1} = Eᵀ
        D = np.diag([lam_n, lam_t, lam_t])

        return E @ D @ E.T

    # ------------------------------------------------------------------
    # Weight function
    # ------------------------------------------------------------------

    def _weights(self, gammas: List[float]) -> np.ndarray:
        """
        Proximity weights: w_i ∝ (1/Γ_i)^p, normalised to sum to 1.

        Obstacles with smaller Γ (closer to the proxy boundary) receive
        higher weight, making the controller focus on the nearest threat.
        """
        inv_g = np.array([1.0 / max(g, 1e-6) ** self.weight_power for g in gammas])
        total = float(inv_g.sum())
        if total < 1e-12:
            return np.ones(len(gammas)) / len(gammas)
        return inv_g / total


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _tangent_basis(n: np.ndarray):
    """Return (e1, e2) orthogonal to unit vector n and to each other."""
    # Pick a vector not collinear with n
    if abs(n[0]) < 0.9:
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(n, v)
    e1 = e1 / float(np.linalg.norm(e1))
    e2 = np.cross(n, e1)
    return e1, e2
