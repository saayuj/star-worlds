"""
Path-length decomposition (E9) for the static scenarios.

For every randomised trial (same seeds as the main experiment) we compute
  L_s    straight-line distance start → goal
  L_obs  shortest path that respects the inflated obstacles
  L_prx  shortest path that respects the star-world proxies
with a 26-connected grid Dijkstra (1.25 cm), and join them with the executed
path lengths of the main experiment.  Grid lengths are normalised by the
grid length of the unobstructed segment to remove the metric bias.

    L_exec / L_s = (L_obs/L_s) · (L_prx/L_obs) · (L_exec/L_prx)
                    inherent      reshaping      modulation
                    detour        (geometric)    (controller)

  python -m experiments.path_analysis     → results/path_analysis.json
"""

from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.sim_core import Scene, TrialConfig            # noqa: E402
from starworlds.integration import StarWorldUpdater           # noqa: E402
from starworlds.reactive_controller import proxy_equations    # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'path_analysis.json')
STATIC = ['single', 'overlap', 'wall', 'boxes', 'cylinders', 'irregular']
H = 0.0125


def grid_geodesic(occ_fn, a, b, lo, hi, h=H):
    nx, ny, nz = np.ceil((hi - lo) / h).astype(int) + 1
    gx, gy, gz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    P = lo + h * np.stack([gx, gy, gz], -1).reshape(-1, 3)
    free = ~occ_fn(P)
    idx = -np.ones(P.shape[0], int)
    idx[free] = np.arange(free.sum())
    ijk = np.stack([gx, gy, gz], -1).reshape(-1, 3)[free]
    rows, cols, w = [], [], []
    for d in [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
              if (i, j, k) > (0, 0, 0)]:
        nb = ijk + np.array(d)
        ok = np.all((nb >= 0) & (nb < [nx, ny, nz]), axis=1)
        flat = nb[ok, 0] * ny * nz + nb[ok, 1] * nz + nb[ok, 2]
        tgt = idx[flat]
        src = np.where(ok)[0]
        good = tgt >= 0
        rows.append(src[good])
        cols.append(tgt[good])
        w.append(np.full(good.sum(), h * np.linalg.norm(d)))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    w = np.concatenate(w)
    n = free.sum()
    G = coo_matrix((np.r_[w, w], (np.r_[rows, cols], np.r_[cols, rows])), shape=(n, n)).tocsr()
    Pf = P[free]
    ia = int(np.argmin(np.linalg.norm(Pf - a, axis=1)))
    ib = int(np.argmin(np.linalg.norm(Pf - b, axis=1)))
    dist = dijkstra(G, indices=ia)[ib]
    return float(dist + np.linalg.norm(Pf[ia] - a) + np.linalg.norm(Pf[ib] - b))


def analyse(args):
    scen, seed = args
    cfg = TrialConfig(scenario=scen, method='sw', seed=seed)
    sc = Scene(cfg)
    infl = [s.inflated(cfg.dr) for s in sc.true_shapes(0.0)]
    a, b = sc.start, sc.goal
    up = StarWorldUpdater(n_approx_3d=cfg.n_approx,
                          kernel_kwargs=dict(reach_center=np.array([0, 0, 0.333])))
    res = up.update(infl, a, b)
    eqs = [eq for pr in res.proxies for eq in proxy_equations(pr)]
    allpts = np.vstack([s.polygon_approximation(48) for s in infl] + [a[None], b[None]])
    lo = np.maximum(allpts.min(0) - 0.25, [0.0, -0.9, 0.02])
    hi = allpts.max(0) + 0.25

    def occ_obs(P):
        o = np.zeros(len(P), bool)
        for s in infl:
            o |= s.signed_distance(P) <= 0
        return o

    def occ_prx(P):
        o = np.zeros(len(P), bool)
        for A, bb in eqs:
            o |= np.all(P @ A.T + bb <= 0, axis=1)
        return o | occ_obs(P)

    Ls = float(np.linalg.norm(b - a))
    Ls_g = grid_geodesic(lambda P: np.zeros(len(P), bool), a, b, lo, hi)
    Lo_g = grid_geodesic(occ_obs, a, b, lo, hi)
    Lp_g = grid_geodesic(occ_prx, a, b, lo, hi)
    return dict(scenario=scen, seed=seed, L_s=Ls, L_obs=Lo_g * Ls / Ls_g,
                L_prx=Lp_g * Ls / Ls_g, n_proxies=len(res.proxies))


if __name__ == '__main__':
    jobs = [(s, k) for s in STATIC for k in range(20)]
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    with Pool(procs) as pool:
        rows = pool.map(analyse, jobs)
    json.dump(rows, open(OUT, 'w'))
    for s in STATIC:
        r = [q for q in rows if q['scenario'] == s]
        print(s, 'L_obs/L_s %.3f' % np.median([q['L_obs'] / q['L_s'] for q in r]),
              'L_prx/L_obs %.3f' % np.median([q['L_prx'] / q['L_obs'] for q in r]))
