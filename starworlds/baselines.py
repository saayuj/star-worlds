"""
Baseline obstacle representations and reactive controllers used in the
revised evaluation.

Representations (consumed by ReactiveController, i.e. DS modulation):
  per_obstacle_proxies  — one proxy per inflated obstacle, reference point at
                          the obstacle centre (the "raw" modulation baseline).
  convex_hull_proxies   — overlapping obstacles are clustered and replaced by
                          the convex hull of the cluster (naive merging, the
                          alternative discussed in Dahlin & Karayiannidis).

Controllers (operate directly on the inflated obstacle shapes):
  CBFQPController — control-barrier-function quadratic program (Ames et al.)
  APFController   — artificial potential field (Khatib 1986)

Utilities:
  any_overlap     — overlap detector used by adaptive star-world activation.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .star_world import ObstacleProxy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _samples(obs, n):
    return np.asarray(obs.polygon_approximation(n=n), dtype=float)


def _center(obs):
    return np.asarray(obs.center, dtype=float)


def _bound_r(obs):
    if hasattr(obs, "bounding_radius"):
        return float(obs.bounding_radius())
    return float(np.sqrt(np.max(np.linalg.eigvalsh(obs.Q))))


def signed_distance(obs, p) -> float:
    if hasattr(obs, "signed_distance"):
        return float(obs.signed_distance(p)[0])
    d = np.asarray(p, dtype=float) - obs.center
    # sphere-like Ellipsoid (all experiments use isotropic Q)
    r = float(np.sqrt(np.max(np.linalg.eigvalsh(obs.Q))))
    return float(np.linalg.norm(d) - r)


def sd_gradient(obs, p, h: float = 1e-4) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    g = np.zeros(3)
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        g[k] = (signed_distance(obs, p + e) - signed_distance(obs, p - e)) / (2 * h)
    n = float(np.linalg.norm(g))
    return g / n if n > 1e-12 else g


def pair_overlap(a, b, n: int = 64) -> bool:
    ca, cb = _center(a), _center(b)
    if np.linalg.norm(ca - cb) > _bound_r(a) + _bound_r(b):
        return False
    sa = _samples(a, n)
    if any(signed_distance(b, s) <= 0.0 for s in sa):
        return True
    sb = _samples(b, n)
    if any(signed_distance(a, s) <= 0.0 for s in sb):
        return True
    # one center inside the other (containment)
    return signed_distance(b, ca) <= 0.0 or signed_distance(a, cb) <= 0.0


def overlap_clusters(obstacles: list, n: int = 64) -> List[List[int]]:
    m = len(obstacles)
    parent = list(range(m))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(m):
        for j in range(i + 1, m):
            if pair_overlap(obstacles[i], obstacles[j], n):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(m):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def any_overlap(obstacles: list, n: int = 64) -> bool:
    return any(len(c) > 1 for c in overlap_clusters(obstacles, n))


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

def per_obstacle_proxies(obstacles: list, n: int = 64) -> List[ObstacleProxy]:
    out = []
    for i, obs in enumerate(obstacles):
        out.append(ObstacleProxy(pieces=[_samples(obs, n)], kernel_K=None,
                                 center_point=_center(obs),
                                 original_indices=[i], is_fallback=False))
    return out


def convex_hull_proxies(obstacles: list, n: int = 64) -> List[ObstacleProxy]:
    out = []
    for cl in overlap_clusters(obstacles, n):
        pts = np.vstack([_samples(obstacles[i], n) for i in cl])
        center = np.mean([_center(obstacles[i]) for i in cl], axis=0)
        out.append(ObstacleProxy(pieces=[pts], kernel_K=None,
                                 center_point=center,
                                 original_indices=sorted(cl), is_fallback=False))
    return out


# ---------------------------------------------------------------------------
# Controllers acting on inflated shapes
# ---------------------------------------------------------------------------

class CBFQPController:
    """
    Safety filter  min ||v − v_nom||²  s.t.  ∇h_i·(v − v_i) ≥ −α h_i
    with h_i(x) = sd_i(x) the signed distance to inflated obstacle i and
    v_i its (estimated) velocity.  The QP is solved exactly with a small
    active-set / dual coordinate-ascent (Hildreth) iteration.
    """

    def __init__(self, v_max=0.30, alpha=4.0, goal_tolerance=0.02,
                 slowdown_radius=0.05, activation=0.25):
        self.v_max = v_max
        self.alpha = alpha
        self.goal_tolerance = goal_tolerance
        self.slowdown_radius = slowdown_radius
        self.activation = activation

    def compute_velocity(self, x, xg, obstacles, velocities=None):
        x = np.asarray(x, float)
        e = np.asarray(xg, float) - x
        d = float(np.linalg.norm(e))
        if d < self.goal_tolerance:
            return np.zeros(3)
        speed = self.v_max * min(1.0, max(d / self.slowdown_radius, 0.15))
        v_nom = speed * e / d
        A, b = [], []
        for i, obs in enumerate(obstacles):
            h = signed_distance(obs, x)
            if h > self.activation:
                continue
            g = sd_gradient(obs, x)
            vo = np.zeros(3) if velocities is None else np.asarray(velocities[i])
            A.append(g)
            b.append(-self.alpha * h + float(g @ vo))
        if not A:
            return v_nom
        A = np.asarray(A)
        b = np.asarray(b)
        # Hildreth's dual coordinate ascent for  min ½||v − v_nom||²,  A v ≥ b
        lam = np.zeros(len(b))
        G = A @ A.T
        for _ in range(200):
            lam_old = lam.copy()
            for k in range(len(b)):
                v = v_nom + A.T @ lam
                w = (b[k] - A[k] @ v) / max(G[k, k], 1e-12)
                lam[k] = max(0.0, lam[k] + w)
            if np.max(np.abs(lam - lam_old)) < 1e-9:
                break
        v = v_nom + A.T @ lam
        n = float(np.linalg.norm(v))
        if n > self.v_max:
            v *= self.v_max / n
        return v


class APFController:
    """Khatib-style attractive/repulsive potential field (velocity form)."""

    def __init__(self, v_max=0.30, k_rep=0.004, rho0=0.10,
                 goal_tolerance=0.02, slowdown_radius=0.05):
        self.v_max = v_max
        self.k_rep = k_rep
        self.rho0 = rho0
        self.goal_tolerance = goal_tolerance
        self.slowdown_radius = slowdown_radius

    def compute_velocity(self, x, xg, obstacles, velocities=None):
        x = np.asarray(x, float)
        e = np.asarray(xg, float) - x
        d = float(np.linalg.norm(e))
        if d < self.goal_tolerance:
            return np.zeros(3)
        f = e / d
        for obs in obstacles:
            rho = max(signed_distance(obs, x), 1e-3)
            if rho < self.rho0:
                f = f + self.k_rep * (1.0 / rho - 1.0 / self.rho0) / rho ** 2 \
                    * sd_gradient(obs, x)
        n = float(np.linalg.norm(f))
        if n < 1e-9:
            return np.zeros(3)
        speed = self.v_max * min(1.0, max(d / self.slowdown_radius, 0.15))
        return speed * f / n
