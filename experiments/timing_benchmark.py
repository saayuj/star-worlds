"""
Single-thread computation-time benchmark (E8).

Random sphere clutter with N = 1 … 30 obstacles; for each N, 30 random
scenes.  Measures the star-world update (cold call and warm call with
temporal coherence), per-obstacle proxy construction (baseline), the
modulated-velocity evaluation and the number of resulting clusters.

  python -m experiments.timing_benchmark      → results/timing.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from starworlds.integration import StarWorldUpdater            # noqa: E402
from starworlds.reactive_controller import ReactiveController  # noqa: E402
from starworlds.shapes3d import Sphere3D                        # noqa: E402
from starworlds import baselines as B                          # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'timing.json')


def scene(rng, n, dr=0.05):
    x = np.array([0.25, -0.35, 0.42])
    xg = np.array([0.65, 0.35, 0.42])
    obs = []
    while len(obs) < n:
        c = rng.uniform([0.30, -0.30, 0.22], [0.70, 0.30, 0.62])
        r = rng.uniform(0.04, 0.08)
        s = Sphere3D(c, r).inflated(dr)
        if s.signed_distance(x)[0] > 0.01 and s.signed_distance(xg)[0] > 0.01:
            obs.append(s)
    return obs, x, xg


def main():
    rng = np.random.default_rng(7)
    ctrl = ReactiveController(v_max=0.3, safety_radius=0.01, goal_tolerance=0.02,
                              slowdown_radius=0.05)
    rows = []
    for n in (1, 2, 4, 6, 8, 10, 15, 20, 25, 30):
        for k in range(30):
            obs, x, xg = scene(rng, n)
            up = StarWorldUpdater(n_approx_3d=48,
                                  kernel_kwargs=dict(reach_center=np.array([0, 0, 0.333])))
            t0 = time.perf_counter()
            res = up.update(obs, x, xg)
            t_cold = time.perf_counter() - t0
            t0 = time.perf_counter()
            res = up.update(obs, x + 0.005, xg)
            t_warm = time.perf_counter() - t0
            t0 = time.perf_counter()
            raw = B.per_obstacle_proxies(obs, 48)
            t_raw = time.perf_counter() - t0
            t0 = time.perf_counter()
            for _ in range(5):
                ctrl.compute_velocity(x + 0.01, xg, res.proxies)
            t_vel = (time.perf_counter() - t0) / 5
            t0 = time.perf_counter()
            for _ in range(5):
                ctrl.compute_velocity(x + 0.01, xg, raw)
            t_vel_raw = (time.perf_counter() - t0) / 5
            rows.append(dict(n=n, rep=k, t_cold_ms=1e3 * t_cold, t_warm_ms=1e3 * t_warm,
                             t_raw_ms=1e3 * t_raw, t_vel_ms=1e3 * t_vel,
                             t_vel_raw_ms=1e3 * t_vel_raw,
                             n_clusters=len(res.proxies), disjoint=bool(res.is_disjoint),
                             fallback=any(pp.is_fallback for pp in res.proxies)))
        r = [q for q in rows if q['n'] == n]
        print(f"N={n:2d}  SW {np.median([q['t_warm_ms'] for q in r]):6.1f} ms  "
              f"clusters {np.mean([q['n_clusters'] for q in r]):4.1f}  "
              f"fallback {np.mean([q['fallback'] for q in r]):.2f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rows, open(OUT, 'w'))


if __name__ == '__main__':
    main()
