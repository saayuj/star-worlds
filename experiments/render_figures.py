"""
Record selected trials and render full-arm figures offscreen (no GUI).

  python -m experiments.render_figures record   → results/logs/*.pkl
  python -m experiments.render_figures render   → paper figures (PNG)

Rendering uses PyBullet's TinyRenderer through getCameraImage in DIRECT
mode.  Multi-exposure posture overlays are built from the segmentation
mask: robot pixels of earlier frames are alpha-blended onto the final frame.
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pybullet as p
import pybullet_data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments.sim_core import TrialConfig, run_trial, Scene, _collision_shape, _matrix_to_quat  # noqa
from experiments import scenarios as S  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
LOGS = os.path.join(ROOT, 'results', 'logs')
FIGS = os.environ.get('FIG_DIR', os.path.join(ROOT, '..', 'revision', 'figures'))

RECORD = [
    # name, config
    ('diag_single_legacy', TrialConfig(scenario='single', method='sw_legacy', randomize=False, timeout=60)),
    ('diag_single_sw', TrialConfig(scenario='single', method='sw', randomize=False, timeout=60)),
    ('diag_overlap_legacy', TrialConfig(scenario='overlap', method='sw_legacy', randomize=False, timeout=60)),
    ('diag_overlap_sw', TrialConfig(scenario='overlap', method='sw', randomize=False, timeout=60)),
    ('diag_overlap_seed1_noflip', TrialConfig(scenario='overlap', method='sw', seed=1, ablate='ws,flip', timeout=60)),
    ('diag_overlap_seed1_sw', TrialConfig(scenario='overlap', method='sw', seed=1, timeout=60)),
    ('wall_sw', TrialConfig(scenario='wall', method='sw', randomize=False, timeout=60)),
    ('wall_mod_legacy', TrialConfig(scenario='wall', method='mod_legacy', randomize=False, timeout=60)),
    ('wall_mod', TrialConfig(scenario='wall', method='mod', randomize=False, timeout=60)),
    ('overlap_mod', TrialConfig(scenario='overlap', method='mod', randomize=False, timeout=60)),
    ('dynamic_sw', TrialConfig(scenario='dynamic', method='sw', randomize=False, timeout=60)),
    ('irregular_sw', TrialConfig(scenario='irregular', method='sw', randomize=False, timeout=60)),
    ('cylinders_sw', TrialConfig(scenario='cylinders', method='sw', randomize=False, timeout=60)),
    ('boxes_sw', TrialConfig(scenario='boxes', method='sw', randomize=False, timeout=60)),
]


def record():
    os.makedirs(LOGS, exist_ok=True)
    for name, cfg in RECORD:
        cfg.record = True
        r = run_trial(cfg)
        pickle.dump(r, open(os.path.join(LOGS, name + '.pkl'), 'wb'))
        print(name, r['success'], round(r['t_end'], 2), round(r['ratio'], 2), flush=True)


class Renderer:
    def __init__(self, scenario, W=1100, Hh=820):
        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.loadURDF('plane.urdf', physicsClientId=self.cid)
        self.robot = p.loadURDF('franka_panda/panda.urdf', [0, 0, 0], useFixedBase=True,
                                physicsClientId=self.cid)
        spec = S.get(scenario)
        self.obs = []
        for o in spec['obstacles']:
            sh = o['shape']
            if sh.kind == 'cylinder':
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=sh.radius, length=2 * sh.hh,
                                          rgbaColor=o['color'], physicsClientId=self.cid)
            else:
                vis = p.createVisualShape(rgbaColor=o['color'], physicsClientId=self.cid,
                                          **_collision_shape(sh))
            b = p.createMultiBody(0, -1, vis, sh.center.tolist(), _matrix_to_quat(sh.R),
                                  physicsClientId=self.cid)
            self.obs.append((b, sh))
        self.W, self.H = W, Hh
        self.markers = []

    def camera(self, target, dist, yaw, pitch):
        self.view = p.computeViewMatrixFromYawPitchRoll(target, dist, yaw, pitch, 0, 2,
                                                        physicsClientId=self.cid)
        self.proj = p.computeProjectionMatrixFOV(40, self.W / self.H, 0.05, 5.0,
                                                 physicsClientId=self.cid)

    def set_state(self, q, obs_centers):
        for j in range(7):
            p.resetJointState(self.robot, j, float(q[j]), physicsClientId=self.cid)
        for (b, sh), c in zip(self.obs, obs_centers):
            p.resetBasePositionAndOrientation(b, list(c), _matrix_to_quat(sh.R),
                                              physicsClientId=self.cid)

    def path(self, X, color, r=0.004, every=2):
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=color,
                                  physicsClientId=self.cid)
        for x in X[::every]:
            self.markers.append(p.createMultiBody(0, -1, vis, list(x), physicsClientId=self.cid))

    def point(self, x, color, r=0.012):
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=color,
                                  physicsClientId=self.cid)
        self.markers.append(p.createMultiBody(0, -1, vis, list(x), physicsClientId=self.cid))

    def render(self):
        w, h, rgb, depth, seg = p.getCameraImage(
            self.W, self.H, self.view, self.proj, renderer=p.ER_TINY_RENDERER,
            lightDirection=[1, 1, 2], shadow=0, physicsClientId=self.cid)
        rgb = np.reshape(rgb, (h, w, 4))[:, :, :3].astype(float) / 255.0
        seg = np.reshape(seg, (h, w))
        obj = np.where(seg >= 0, seg & ((1 << 24) - 1), -1)
        return rgb, obj

    def close(self):
        p.disconnect(self.cid)


def overlay(name, cam, n_ghosts=5, out=None, draw_path=True, ghost_alpha=0.42, crop=None):
    import matplotlib.pyplot as plt
    r = pickle.load(open(os.path.join(LOGS, name + '.pkl'), 'rb'))
    L = r['log']
    X = np.array(L['x'])
    R = Renderer(r['scenario'])
    R.camera(*cam)
    if draw_path:
        R.path(X, [0.12, 0.12, 0.12, 1.0], r=0.005)
    R.point(r['start_pos'], [0.1, 0.8, 0.1, 1])
    R.point(r['goal_pos'], [0.9, 0.1, 0.1, 1])
    idx = np.linspace(0, len(X) - 1, n_ghosts + 1).astype(int)
    R.set_state(L['q'][idx[-1]], L['obs'][idx[-1]])
    base, _ = R.render()
    comp = base.copy()
    for k in idx[:-1]:
        R.set_state(L['q'][k], L['obs'][idx[-1]])
        img, obj = R.render()
        m = obj == R.robot
        comp[m] = (1 - ghost_alpha) * comp[m] + ghost_alpha * img[m]
    R.set_state(L['q'][idx[-1]], L['obs'][idx[-1]])
    img, obj = R.render()
    m = obj == R.robot
    comp[m] = img[m]
    R.close()
    if crop:
        comp = comp[crop[0]:crop[1], crop[2]:crop[3]]
    plt.imsave(out or os.path.join(FIGS, name + '_overlay.png'), np.clip(comp, 0, 1))


def strip(name, cam, n=4, out=None, crop=None):
    """Side-by-side frames (for moving obstacles)."""
    import matplotlib.pyplot as plt
    r = pickle.load(open(os.path.join(LOGS, name + '.pkl'), 'rb'))
    L = r['log']
    X = np.array(L['x'])
    R = Renderer(r['scenario'], W=700, Hh=620)
    R.camera(*cam)
    R.point(r['start_pos'], [0.1, 0.8, 0.1, 1])
    R.point(r['goal_pos'], [0.9, 0.1, 0.1, 1])
    idx = np.linspace(0, len(X) - 1, n).astype(int)
    frames = []
    for j, k in enumerate(idx):
        # EE trace up to this frame
        for mk in R.markers[2:]:
            p.removeBody(mk, physicsClientId=R.cid)
        R.markers = R.markers[:2]
        R.path(X[:k + 1], [0.12, 0.12, 0.12, 1], r=0.005)
        R.set_state(L['q'][k], L['obs'][k])
        img, _ = R.render()
        if crop:
            img = img[crop[0]:crop[1], crop[2]:crop[3]]
        frames.append((L['t'][k], img))
    R.close()
    fig, axs = plt.subplots(1, n, figsize=(3.0 * n, 2.9))
    fig.subplots_adjust(top=0.9)
    for ax, (t, img) in zip(axs, frames):
        ax.imshow(img)
        ax.set_title(f't = {t:.1f} s', fontsize=10)
        ax.axis('off')
    plt.tight_layout(pad=0.6)
    plt.savefig(out or os.path.join(FIGS, name + '_strip.png'), dpi=200)
    plt.close(fig)


def render_all():
    os.makedirs(FIGS, exist_ok=True)
    cw = ([0.35, 0.0, 0.45], 1.6, 60, -20)
    crop = (60, 640, 220, 950)
    for n in ('wall_sw', 'wall_mod_legacy', 'wall_mod'):
        overlay(n, cw, n_ghosts=0 if 'legacy' in n else 4, crop=crop)
    co = ([0.40, 0.0, 0.45], 1.6, 60, -20)
    for n in ('overlap_mod', 'diag_overlap_sw'):
        overlay(n, co, n_ghosts=4, crop=crop)
    for n in ('boxes_sw', 'cylinders_sw', 'irregular_sw'):
        overlay(n, co, n_ghosts=4, crop=crop)
    strip('dynamic_sw', ([0.40, 0.0, 0.42], 1.45, 55, -32), n=4, crop=(40, 560, 60, 660))


if __name__ == '__main__':
    if sys.argv[1] == 'record':
        record()
    else:
        render_all()
