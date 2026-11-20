"""
Batch runner for the revised evaluation (headless, multi-process).

  python -m experiments.run_experiments --suite main      # E1 main comparison
  python -m experiments.run_experiments --suite all

Each suite writes one JSON-lines file in results/ with one row per trial.
Already-finished trials (same config key) are skipped, so runs can resume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.sim_core import TrialConfig, run_trial    # noqa: E402
from experiments.scenarios import CORE, SHAPES              # noqa: E402

RESULTS = os.environ.get('RESULTS_DIR', os.path.join(os.path.dirname(__file__), '..', 'results'))
N_SEEDS = 20
MAIN_METHODS = ['sw', 'sw_adaptive', 'mod', 'ch', 'cbf', 'apf', 'sw_legacy', 'mod_legacy']


def suite_configs(name, n_seeds=N_SEEDS):
    seeds = range(n_seeds)
    C = []
    if name == 'main':            # E1: all methods, core + shape scenarios, 180 s
        for sc in CORE + SHAPES:
            for m in MAIN_METHODS:
                for s in seeds:
                    C.append(TrialConfig(scenario=sc, method=m, seed=s, timeout=180.0))
    elif name == 'margin':        # E2: reduced safety margin baseline
        for sc in CORE + SHAPES:
            for dr in (0.0, 0.01, 0.02, 0.03, 0.05):
                for m in ('mod', 'sw'):
                    for s in seeds:
                        C.append(TrialConfig(scenario=sc, method=m, seed=s, dr=dr,
                                             timeout=180.0))
    elif name == 'noise':         # E3: perception noise
        for sc in CORE + SHAPES:
            for sd in (0.0, 0.005, 0.010):
                for m in ('sw', 'mod'):
                    for s in seeds:
                        C.append(TrialConfig(scenario=sc, method=m, seed=s,
                                             noise_std=sd, timeout=180.0))
    elif name == 'noise2':        # E3b: arm clearance sized to the noise level
        for sc in CORE + SHAPES:
            for sd in (0.005, 0.010):
                for m in ('sw', 'mod'):
                    for s in seeds:
                        C.append(TrialConfig(scenario=sc, method=m, seed=s, noise_std=sd,
                                             qp_dsafe=0.015 + 2 * sd, timeout=180.0))
    elif name == 'latency':       # E10: execution model / computation latency
        for sc in ('dynamic', 'moderate', 'corridor', 'simple'):
            for lat in (0.0, -1.0, 0.1, 0.25, 0.5):
                for m in ('sw', 'mod'):
                    for s in seeds:
                        C.append(TrialConfig(scenario=sc, method=m, seed=s, latency=lat,
                                             timeout=180.0))
    elif name == 'rates':         # E4: update-rate sensitivity
        for sc in ('moderate', 'corridor', 'dynamic', 'simple'):
            for hz in (1.0, 2.0, 4.0, 10.0, 20.0, 40.0):
                for s in seeds:
                    C.append(TrialConfig(scenario=sc, method='sw', seed=s, sw_hz=hz,
                                         timeout=180.0))
            for chz in (10.0, 20.0, 80.0):
                for s in seeds:
                    C.append(TrialConfig(scenario=sc, method='sw', seed=s, ctrl_hz=chz,
                                         timeout=180.0))
    elif name == 'ablation':      # E6: contribution of each revision
        for sc in CORE + SHAPES:
            for abl in ('kernel', 'ctrl', 'arm', 'ws', 'flip', 'ws,flip'):
                for s in seeds:
                    C.append(TrialConfig(scenario=sc, method='sw', seed=s, ablate=abl,
                                         timeout=180.0))
    elif name == 'speed':         # E7: obstacle speed (truly dynamic scenes)
        for sc in ('dynamic', 'moderate', 'corridor'):
            for sp in (0.5, 1.0, 1.5, 2.0, 3.0):
                for m in ('sw', 'mod', 'cbf'):
                    for s in seeds:
                        C.append(TrialConfig(scenario=sc, method=m, seed=s, speed_scale=sp,
                                             timeout=180.0))
    elif name == 'timeout':       # E5: convergence-time check, very long horizon
        for sc in CORE + SHAPES:
            for m in ('sw_legacy', 'mod_legacy'):
                for s in seeds:
                    C.append(TrialConfig(scenario=sc, method=m, seed=s, timeout=600.0))
    else:
        raise ValueError(name)
    return C


def key(cfg: TrialConfig):
    d = asdict(cfg)
    d.pop('record', None)
    return json.dumps(d, sort_keys=True)


def _work(cfg):
    try:
        r = run_trial(cfg)
    except Exception as e:  # keep the batch alive; failures are recorded
        r = dict(asdict(cfg))
        r['error'] = repr(e)
    r['key'] = key(cfg)
    return r


def _clean(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


def run_suite(name, procs, n_seeds=N_SEEDS):
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f'{name}.jsonl')
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['key'])
                except Exception:
                    pass
    todo = [c for c in suite_configs(name, n_seeds) if key(c) not in done]
    print(f'[{name}] {len(todo)} trials to run ({len(done)} done)', flush=True)
    with Pool(procs) as pool, open(out, 'a') as f:
        for i, r in enumerate(pool.imap_unordered(_work, todo, chunksize=2)):
            f.write(json.dumps({k: _clean(v) for k, v in r.items()}) + '\n')
            f.flush()
            if (i + 1) % 100 == 0:
                print(f'[{name}] {i + 1}/{len(todo)}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='main')
    ap.add_argument('--procs', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--seeds', type=int, default=N_SEEDS)
    a = ap.parse_args()
    suites = ['main', 'margin', 'noise', 'noise2', 'rates', 'ablation', 'speed', 'timeout',
                  'latency'] if a.suite == 'all' else [a.suite]
    for s in suites:
        run_suite(s, a.procs, a.seeds)
