"""
Interactive / headless demo: Franka Panda + Star-Worlds in PyBullet.

Thin wrapper around experiments/sim_core.py, so the demo runs exactly the
code used for the experiments in the paper.

Examples
--------
    python examples/franka_pybullet_demo.py --scenario overlap            # GUI, star worlds
    python examples/franka_pybullet_demo.py --scenario overlap --method mod
    python examples/franka_pybullet_demo.py --scenario dynamic --headless # no window

Methods: sw (default), sw_adaptive, mod, ch, cbf, apf, sw_legacy, mod_legacy.
GUI overlays: green wireframe = star-world proxy pieces, green cross = kernel
centre x_c, white trail = EE path, green / red dots = start / goal.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.sim_core import TrialConfig, run_trial      # noqa: E402
from experiments.scenarios import SCENARIOS                  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', default='overlap', choices=sorted(SCENARIOS))
    ap.add_argument('--method', default='sw',
                    choices=['sw', 'sw_adaptive', 'mod', 'ch', 'cbf', 'apf',
                             'sw_legacy', 'mod_legacy'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--randomize', action='store_true',
                    help='randomise start/goal/obstacles/initial posture as in the paper')
    ap.add_argument('--timeout', type=float, default=60.0)
    ap.add_argument('--headless', action='store_true', help='no GUI window')
    ap.add_argument('--no-wait', action='store_true', help='do not wait for Enter')
    # kept for backward compatibility with the original CLI
    ap.add_argument('--raw', action='store_true', help='alias for --method mod')
    ap.add_argument('--no-gui', action='store_true', help='alias for --headless')
    a = ap.parse_args()
    method = 'mod' if a.raw else a.method
    gui = not (a.headless or a.no_gui)
    r = run_trial(TrialConfig(scenario=a.scenario, method=method, seed=a.seed,
                              randomize=a.randomize, timeout=a.timeout, gui=gui,
                              realtime=gui, wait_enter=gui and not a.no_wait))
    print(f"success={r['success']}  reached={r['reached']}  collided={r['collided']}  "
          f"t={r['t_end']:.2f}s  path ratio={r['ratio']:.2f}  "
          f"min clearance={1000 * r['min_clearance']:.1f} mm  "
          f"world update={r['rep_ms_mean']:.1f} ms")


if __name__ == '__main__':
    main()
