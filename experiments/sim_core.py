"""
Headless PyBullet simulation core for the revised evaluation.

One call to ``run_trial(TrialConfig(...))`` executes a single randomised
trial and returns a flat metrics dictionary (plus an optional trajectory
log).  The same core drives the interactive GUI demo
(examples/franka_pybullet_demo.py) when ``gui=True``.

Timing model
------------
Physics runs at 240 Hz.  Obstacles are moved at EVERY physics step (they
never pause); the planner perceives them at the star-world rate (default
4 Hz) and the velocity command is recomputed at the control rate (default
40 Hz).  In headless mode the simulation is not throttled to wall-clock
time, so the measured planner compute time does not stall the world.

Methods
-------
  sw           Star-World reshaping (revised: 3-D admissible kernel,
               exact star-shaped Γ, Huber weights, tail-effect removal)
  sw_adaptive  Star-World only when inflated obstacles overlap, otherwise mod
  sw_legacy    Star-World pipeline exactly as in the submitted manuscript
  mod          DS modulation against each inflated obstacle (no reshaping)
  mod_legacy   'raw' baseline exactly as in the submitted manuscript
  ch           DS modulation against convex hulls of overlapping clusters
  cbf          Control-barrier-function QP safety filter
  apf          Artificial potential field
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pybullet as p
import pybullet_data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from starworlds.integration import StarWorldUpdater                    # noqa: E402
from starworlds.reactive_controller import ReactiveController          # noqa: E402
from starworlds.shapes3d import quat_to_matrix                          # noqa: E402
from starworlds import baselines as B                                  # noqa: E402
from experiments import scenarios as S                                 # noqa: E402

PHYSICS_HZ = 240
Q_NOMINAL = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.8, 0.785])
LL = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
UL = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
WS_LO = np.array([0.10, -0.75, 0.10])
WS_HI = np.array([0.95, 0.75, 0.90])
EE_LINK = 11
ARM_LINKS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
ARM_FRACS = np.array([0.0, 0.5, 1.0])
ARM_RADII = {0: 0.06, 1: 0.06, 2: 0.06, 3: 0.06, 4: 0.05, 5: 0.05, 6: 0.04,
             7: 0.03, 8: 0.04}
ARM_GAMMA_TH = 1.6
ARM_GAIN = 0.30
ARM_POSTURE_GAIN = 0.08
ARM_EMA = 0.25
ARM_MAX_NS = 0.12
POSTURE_ACTIVE_SCALE = 1.0
# Whole-body QP arm layer
QP_ALPHA = 6.0          # CBF gain on link-obstacle distance [1/s]
QP_DSAFE = 0.015        # extra clearance required between link surface and obstacle [m]
QP_ACT = 0.15           # constraints activated below this distance [m]
QP_REG = 0.02           # joint-velocity regularisation weight
QP_POSTURE_K = 0.6      # posture-restoring joint velocity gain [1/s]
QP_JL_ALPHA = 4.0       # joint-limit barrier gain [1/s]
QP_QDMAX = 1.5          # joint speed limit [rad/s]


@dataclass
class TrialConfig:
    scenario: str = 'overlap'
    method: str = 'sw'
    seed: int = 0
    randomize: bool = True
    dr: float = 0.05                 # safety inflation Δr [m]
    timeout: float = 180.0           # [s] simulated time
    sw_hz: float = 4.0               # perception / star-world rate
    ctrl_hz: float = 40.0            # velocity-command rate
    noise_std: float = 0.0           # perception noise on obstacle centres [m]
    n_approx: int = 48               # surface samples per obstacle
    v_max: float = 0.30
    goal_tol: float = 0.02
    perturb_start: float = 0.03
    perturb_obs: float = 0.015
    perturb_q: float = 0.25
    arm_avoid: bool = True
    arm_mode: str = 'qp'             # 'qp' whole-body QP | 'ns' IK + null-space APF
    gui: bool = False
    realtime: bool = False
    wait_enter: bool = False
    record: bool = False
    tag: str = ''
    speed_scale: float = 1.0         # time scaling of obstacle motion
    # comma-separated components reverted to the submitted version
    # (ablation): kernel, ctrl, arm, ws, flip
    ablate: str = ''
    qp_dsafe: float = -1.0           # arm clearance [m]; <0 → QP_DSAFE


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def _collision_shape(shape):
    k = shape.kind
    if k == 'sphere':
        return dict(shapeType=p.GEOM_SPHERE, radius=shape.radius)
    if k == 'box':
        return dict(shapeType=p.GEOM_BOX, halfExtents=shape.h.tolist())
    if k == 'cylinder':
        return dict(shapeType=p.GEOM_CYLINDER, radius=shape.radius, height=2 * shape.hh)
    raise ValueError(k)


def _matrix_to_quat(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat().tolist()


class Scene:
    def __init__(self, cfg: TrialConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        spec = S.get(cfg.scenario)
        self.spec = spec
        rnd = cfg.randomize
        u = lambda a, n=3: self.rng.uniform(-a, a, n) if rnd else np.zeros(n)
        self.t_phase = self.rng.uniform(0.0, 2 * np.pi / 0.6) if rnd else 0.0
        self.q0 = Q_NOMINAL + (self.rng.normal(0, cfg.perturb_q, 7) if rnd else 0.0)
        self.q0 = np.clip(self.q0, LL + 0.05, UL - 0.05)
        self.obs = []
        for o in spec['obstacles']:
            self.obs.append(dict(shape=o['shape'], traj=o['traj'], color=o['color'],
                                 offset=u(cfg.perturb_obs)))
        # Start/goal perturbation by rejection sampling: both must lie outside
        # every inflated obstacle (by 1 cm) at t = 0, otherwise the task is
        # infeasible by construction.
        for _ in range(200):
            self.start = np.asarray(spec['start'], float) + u(cfg.perturb_start)
            self.goal = np.asarray(spec['goal'], float) + u(cfg.perturb_start)
            infl = [sh.inflated(cfg.dr + 0.01) for sh in self.true_shapes(0.0)]
            if all(float(sh.signed_distance(self.start)[0]) > 0 and
                   float(sh.signed_distance(self.goal)[0]) > 0 for sh in infl):
                break
        self.dynamic = any(
            np.linalg.norm(o['traj'](0.0) - o['traj'](1.3)) > 1e-9 for o in self.obs)

    def center(self, i, t):
        o = self.obs[i]
        return o['traj'](self.cfg.speed_scale * t + self.t_phase) + o['offset']

    def true_shapes(self, t):
        return [o['shape'].moved(self.center(i, t)) for i, o in enumerate(self.obs)]


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class FrankaSim:
    def __init__(self, cfg: TrialConfig):
        self.cfg = cfg
        self.scene = Scene(cfg)
        self.cid = p.connect(p.GUI if cfg.gui else p.DIRECT)
        if cfg.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.cid)
        self.robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True,
                                physicsClientId=self.cid)
        nj = p.getNumJoints(self.robot, physicsClientId=self.cid)
        self.movable = [j for j in range(nj) if p.getJointInfo(
            self.robot, j, physicsClientId=self.cid)[2] != p.JOINT_FIXED]
        for j in range(nj):
            p.resetJointState(self.robot, j, 0.0, physicsClientId=self.cid)
        for j in range(7):
            p.resetJointState(self.robot, j, float(self.scene.q0[j]), physicsClientId=self.cid)
        # Obstacles: visual + collision geometry, but with contact disabled so
        # the arm is never pushed physically; getClosestPoints still reports
        # exact distances between the Franka collision meshes and obstacles.
        self.obs_ids = []
        for i, o in enumerate(self.scene.obs):
            sh = o['shape']
            cs = _collision_shape(sh)
            col = p.createCollisionShape(physicsClientId=self.cid, **cs)
            vkw = dict(cs)
            if sh.kind == 'cylinder':
                vkw = dict(shapeType=p.GEOM_CYLINDER, radius=sh.radius, length=2 * sh.hh)
            vis = p.createVisualShape(rgbaColor=o['color'], physicsClientId=self.cid, **vkw)
            body = p.createMultiBody(0, col, vis, self.scene.center(i, 0.0).tolist(),
                                     _matrix_to_quat(sh.R), physicsClientId=self.cid)
            p.setCollisionFilterGroupMask(body, -1, 0, 0, physicsClientId=self.cid)
            self.obs_ids.append(body)
        # Invisible "ghost" copies placed at the PERCEIVED obstacle poses
        # (perception rate, with noise); the arm layer queries these, so it
        # never uses ground-truth information.
        self.ghost_ids = []
        for i, o in enumerate(self.scene.obs):
            col = p.createCollisionShape(physicsClientId=self.cid,
                                         **_collision_shape(o['shape']))
            g = p.createMultiBody(0, col, -1, self.scene.center(i, 0.0).tolist(),
                                  _matrix_to_quat(o['shape'].R), physicsClientId=self.cid)
            p.setCollisionFilterGroupMask(g, -1, 0, 0, physicsClientId=self.cid)
            self.ghost_ids.append(g)

        m = cfg.method
        legacy = m in ('sw_legacy', 'mod_legacy')
        # Legacy methods reproduce the submitted pipeline: IK + null-space APF
        # arm layer with the submitted parameters (EMA 0.10, ||Δq|| ≤ 0.07).
        abl = set(a for a in cfg.ablate.split(',') if a)
        if legacy:
            cfg.arm_mode = 'ns'
            abl |= {'kernel', 'ctrl', 'arm', 'ws', 'flip'}
        if 'arm' in abl:
            cfg.arm_mode = 'ns'
        self.abl = abl
        self.arm_ema = 0.10 if legacy else ARM_EMA
        self.arm_max_ns = 0.07 if legacy else ARM_MAX_NS
        self.legacy = legacy
        self.dsafe = QP_DSAFE if cfg.qp_dsafe < 0 else cfg.qp_dsafe
        self.n_approx = 16 if legacy else cfg.n_approx
        ctrl_legacy = 'ctrl' in abl
        self.ctrl = ReactiveController(v_max=cfg.v_max, eta=1.0, weight_power=2.0,
                                       safety_radius=0.01, goal_tolerance=cfg.goal_tol,
                                       legacy=ctrl_legacy,
                                       slowdown_radius=0.0 if ctrl_legacy else 0.05)
        k_legacy = 'kernel' in abl
        self.updater = StarWorldUpdater(
            n_approx_3d=self.n_approx,
            kernel_mode='legacy' if k_legacy else 'admissible',
            kernel_kwargs=None if (k_legacy or 'ws' in abl)
            else dict(reach_center=np.array([0.0, 0.0, 0.333])))
        self.cbf = B.CBFQPController(v_max=cfg.v_max, goal_tolerance=cfg.goal_tol)
        self.apf = B.APFController(v_max=cfg.v_max, goal_tolerance=cfg.goal_tol)
        self.q_corr_smooth = np.zeros(7)
        self._viz_ids = []
        self._trace_prev = None

    # ---------------------------------------------------------------- utils
    def step_obstacles(self, t):
        for i, b in enumerate(self.obs_ids):
            sh = self.scene.obs[i]['shape']
            p.resetBasePositionAndOrientation(b, self.scene.center(i, t).tolist(),
                                              _matrix_to_quat(sh.R), physicsClientId=self.cid)

    def ee(self):
        return np.array(p.getLinkState(self.robot, EE_LINK, physicsClientId=self.cid)[0])

    def q_all(self):
        return [p.getJointState(self.robot, j, physicsClientId=self.cid)[0] for j in self.movable]

    def jac(self, link, local, q_list):
        z = [0.0] * len(self.movable)
        Jv, _ = p.calculateJacobian(self.robot, link, local, q_list, z, z,
                                    physicsClientId=self.cid)
        return np.array(Jv)[:, :7]

    def min_clearance(self):
        """Min signed distance between any robot link and any true obstacle."""
        dmin = np.inf
        for b in self.obs_ids:
            cps = p.getClosestPoints(self.robot, b, 0.25, physicsClientId=self.cid)
            for c in cps:
                dmin = min(dmin, c[8])
        return dmin

    # ----------------------------------------------------------- perception
    def perceive(self, t):
        cfg = self.cfg
        shapes = []
        for i, o in enumerate(self.scene.obs):
            c = self.scene.center(i, t)
            if cfg.noise_std > 0:
                c = c + self.scene.rng.normal(0.0, cfg.noise_std, 3)
            shapes.append(o['shape'].moved(c).inflated(cfg.dr))
        # un-inflated shapes at the perceived poses (used by the arm layer,
        # which accounts for the link radii explicitly)
        self.perceived_true = [sh.inflated(-cfg.dr) for sh in shapes]
        if hasattr(self, 'ghost_ids'):
            for g, sh in zip(self.ghost_ids, self.perceived_true):
                p.resetBasePositionAndOrientation(g, sh.center.tolist(), _matrix_to_quat(sh.R),
                                                  physicsClientId=self.cid)
        return shapes

    def representation(self, shapes, x, xg):
        """Return (proxies_for_modulation, info) for the configured method."""
        m = self.cfg.method
        info = {'sw_used': False, 'clusters': None}
        if m in ('sw', 'sw_legacy') or (m == 'sw_adaptive' and B.any_overlap(shapes, 32)):
            res = self.updater.update(shapes, x, xg)
            info['sw_used'] = True
            info['clusters'] = tuple(sorted(tuple(sorted(c)) for c in res.clusters))
            info['disjoint'] = res.is_disjoint
            return res.proxies, info
        if m == 'ch':
            pr = B.convex_hull_proxies(shapes, self.n_approx)
            info['clusters'] = tuple(sorted(tuple(pp.original_indices) for pp in pr))
            return pr, info
        if m == 'sw_adaptive':
            self.updater.reset()
        return B.per_obstacle_proxies(shapes, self.n_approx), info

    # ------------------------------------------------------------- control
    def velocity(self, x, xg, proxies, shapes, vels):
        m = self.cfg.method
        if m == 'cbf':
            return self.cbf.compute_velocity(x, xg, shapes, vels)
        if m == 'apf':
            return self.apf.compute_velocity(x, xg, shapes, vels)
        return self.ctrl.compute_velocity(x, xg, proxies)

    def _radius(self, proxy, pt):
        if self.ctrl.legacy:
            return self.ctrl._radius_in_direction(proxy, pt)
        return self.ctrl.ray_radius(proxy, pt)

    def arm_points(self):
        """
        Capsule samples along every arm link plus points on the hand body.

        Returns [(world_point, link_index, local_offset_in_COM_frame, radius)].
        The hand (link 8) is 0.2 m wide across the finger axis, so in
        addition to the centre-line samples it is covered by five points
        across its width (radius 3.5 cm); rotating the last joint moves these
        points without moving the EE, which the null-space layer exploits.
        """
        states = p.getLinkStates(self.robot, ARM_LINKS, computeForwardKinematics=True,
                                 physicsClientId=self.cid)
        org = np.array([s[4] for s in states])
        Rf = [quat_to_matrix(s[5]) for s in states]
        com = np.array([s[0] for s in states])
        Rc = [quat_to_matrix(s[1]) for s in states]
        segs = np.empty_like(org)
        segs[:-1] = org[1:] - org[:-1]
        segs[-1] = Rf[-1] @ np.array([0, 0, 0.05])
        pts = []
        for k, li in enumerate(ARM_LINKS):
            wps = [org[k] + f * segs[k] for f in ARM_FRACS]
            rad = [ARM_RADII[li]] * len(wps)
            if li == 8:
                for y in (-0.085, -0.045, 0.045, 0.085):
                    wps.append(org[k] + Rf[k] @ np.array([0.0, y, 0.045]))
                    rad.append(0.035)
            for wp, rr in zip(wps, rad):
                pts.append((wp, li, (Rc[k].T @ (wp - com[k])).tolist(), rr))
        return pts

    def nullspace_correction(self, proxies, q_list):
        q = np.array(q_list[:7])
        corr = np.zeros(7)
        active = False
        if proxies:
            centers = np.array([pp.center_point for pp in proxies])
            bound = np.array([self.ctrl._bounding_radius(pp) for pp in proxies])
            denom = ARM_GAMMA_TH - 1.0
            jcache = {}
            for (wp, li, off, lr) in self.arm_points():
                dv = wp[None, :] - centers
                dn = np.linalg.norm(dv, axis=1)
                near = np.where(dn < ARM_GAMMA_TH * (bound + lr))[0]
                for pi in near:
                    r = self._radius(proxies[pi], wp) + lr
                    g = max(dn[pi] / max(r, 1e-9), 1e-3)
                    if g >= ARM_GAMMA_TH or dn[pi] < 1e-8:
                        continue
                    active = True
                    key = (li, tuple(off))
                    if key not in jcache:
                        jcache[key] = self.jac(li, off, q_list)
                    rho = max(g - 1.0, 1e-4)
                    w = min(max((1 / rho - 1 / denom) / (1 / denom), 0.0), 5.0)
                    corr += ARM_GAIN * w * (jcache[key].T @ (dv[pi] / dn[pi]))
        pe = Q_NOMINAL - q
        if np.linalg.norm(pe) > 0.05:
            # posture attraction is weakened while links are being repelled so
            # that it cannot pull a link back towards the obstacle
            corr += (ARM_POSTURE_GAIN * (POSTURE_ACTIVE_SCALE if active else 1.0)) * pe
            active = True
        return corr if active else None

    def qp_velocity(self, v_des, ql, shapes, vels=None):
        """
        Whole-body velocity QP (arm layer of the revised pipeline):

          min_q̇  ||J_ee q̇ − v_des||² + ε ||q̇ − k_p (q_nom − q)||²
          s.t.   n_jᵀ J_j q̇ ≥ −α (d_j − r_j − d_safe) + n_jᵀ v_obs   (every
                 link / hand sample point j close to an obstacle)
                 joint-limit barriers and |q̇_i| ≤ q̇_max.

        d_j and n_j are the signed distance and outward normal from point j
        to the (perceived) obstacle surface, r_j the link radius.  The EE
        velocity is tracked exactly whenever the constraints allow it;
        otherwise the QP returns the closest safe motion.
        """
        import quadprog
        q = np.array(ql[:7])
        J = self.jac(EE_LINK, [0, 0, 0], ql)
        G = J.T @ J + QP_REG * np.eye(7)
        a = J.T @ np.asarray(v_des, float) + QP_REG * QP_POSTURE_K * (Q_NOMINAL - q)
        C, b = [], []
        if self.cfg.arm_avoid and shapes:
            # exact closest points between every robot link mesh and every
            # perceived obstacle (PyBullet narrow phase)
            states = {}
            for oi, g in enumerate(self.ghost_ids):
                vo = np.zeros(3) if vels is None else np.asarray(vels[oi])
                cps = p.getClosestPoints(self.robot, g, QP_ACT + self.dsafe,
                                         physicsClientId=self.cid)
                for c in cps:
                    li = c[3]
                    if li < 0 or li == EE_LINK:
                        continue
                    if li not in states:
                        ls = p.getLinkState(self.robot, li, computeForwardKinematics=True,
                                            physicsClientId=self.cid)
                        states[li] = (np.array(ls[0]), quat_to_matrix(ls[1]))
                    com, Rc = states[li]
                    pa = np.array(c[5])
                    n = np.array(c[7])          # normal on obstacle, towards robot
                    off = (Rc.T @ (pa - com)).tolist()
                    Jp = self.jac(li, off, ql)
                    C.append(n @ Jp)
                    b.append(-QP_ALPHA * (c[8] - self.dsafe) + float(n @ vo))
        n_obs = len(b)
        m = 0.05
        for i in range(7):
            e = np.zeros(7)
            e[i] = 1.0
            C += [e, -e, e, -e]
            b += [-QP_JL_ALPHA * (q[i] - (LL[i] + m)), -QP_JL_ALPHA * ((UL[i] - m) - q[i]),
                  -QP_QDMAX, -QP_QDMAX]
        Cm = np.array(C).T
        bm = np.array(b, dtype=float)
        for relax in (0.0, 0.05, 0.2, 1.0, None):
            try:
                if relax is None:
                    qd = np.linalg.solve(G, a)
                else:
                    bb = bm.copy()
                    bb[:n_obs] -= relax
                    qd = quadprog.solve_qp(G, a, Cm, bb, 0)[0]
                self.qp_relaxed = relax
                break
            except ValueError:
                continue
        return np.clip(qd, -QP_QDMAX, QP_QDMAX)

    def command_joint_velocity(self, qd, ql, dt):
        q = np.array(ql[:7])
        q_cmd = np.clip(q + qd * dt, LL, UL)
        for j in range(7):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL,
                                    targetPosition=float(q_cmd[j]),
                                    targetVelocity=float(qd[j]), force=1000,
                                    physicsClientId=self.cid)

    def apply_ik(self, target, q_list=None):
        q_cur = np.array([p.getJointState(self.robot, j, physicsClientId=self.cid)[0]
                          for j in range(7)])
        q_ik = np.array(p.calculateInverseKinematics(
            self.robot, EE_LINK, target.tolist(), lowerLimits=LL.tolist(),
            upperLimits=UL.tolist(), jointRanges=(UL - LL).tolist(),
            restPoses=q_cur.tolist(), physicsClientId=self.cid)[:7])
        dq = np.zeros(7)
        if np.any(self.q_corr_smooth != 0):
            ql = q_list if q_list is not None else self.q_all()
            J = self.jac(EE_LINK, [0, 0, 0], ql)
            N = np.eye(7) - np.linalg.pinv(J) @ J
            dq = N @ self.q_corr_smooth
            nn = float(np.linalg.norm(dq))
            if nn > self.arm_max_ns:
                dq *= self.arm_max_ns / nn
        q_cmd = np.clip(q_ik + dq, LL, UL)
        for j in range(7):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL,
                                    targetPosition=float(q_cmd[j]), force=1000,
                                    physicsClientId=self.cid)

    # ------------------------------------------------------------ GUI viz
    def draw(self, proxies):
        from scipy.spatial import ConvexHull
        for i in self._viz_ids:
            p.removeUserDebugItem(i, physicsClientId=self.cid)
        self._viz_ids = []
        for pp in proxies:
            for pts in pp.pieces:
                try:
                    h = ConvexHull(pts)
                except Exception:
                    continue
                edges = set()
                for s in h.simplices:
                    for a in range(3):
                        edges.add(tuple(sorted((s[a], s[(a + 1) % 3]))))
                for a, b in list(edges)[::2]:
                    self._viz_ids.append(p.addUserDebugLine(
                        pts[a].tolist(), pts[b].tolist(), [0, 0.85, 0.25], 1.0,
                        physicsClientId=self.cid))
            c = np.asarray(pp.center_point)
            for k in range(3):
                e = np.zeros(3)
                e[k] = 0.02
                self._viz_ids.append(p.addUserDebugLine(
                    (c - e).tolist(), (c + e).tolist(), [0, 1, 0], 3,
                    physicsClientId=self.cid))

    # -------------------------------------------------------------- trial
    def move_to_start(self, max_steps=2400):
        x0 = self.scene.start
        if self.cfg.arm_mode == 'qp':
            for k in range(max_steps):
                if k % 6 == 0:
                    ql = self.q_all()
                    e = x0 - self.ee()
                    v = 3.0 * e
                    n = float(np.linalg.norm(v))
                    if n > 0.3:
                        v *= 0.3 / n
                    qd = self.qp_velocity(v, ql, self.perceived_true)
                    self.command_joint_velocity(qd, ql, 6.0 / PHYSICS_HZ)
                p.stepSimulation(physicsClientId=self.cid)
                if self.cfg.realtime:
                    time.sleep(1.0 / PHYSICS_HZ)
                if k > 240 and k % 6 == 0 and np.linalg.norm(self.ee() - x0) < 0.005:
                    break
            return
        for k in range(max_steps):
            if k % 6 == 0:
                ql = self.q_all()
                c = self.nullspace_correction(self._start_proxies, ql)
                raw = c if c is not None else np.zeros(7)
                self.q_corr_smooth = self.arm_ema * raw + (1 - self.arm_ema) * self.q_corr_smooth
            self.apply_ik(x0)
            p.stepSimulation(physicsClientId=self.cid)
            if self.cfg.realtime:
                time.sleep(1.0 / PHYSICS_HZ)
            if k > 240 and k % 6 == 0 and np.linalg.norm(self.ee() - x0) < 0.01:
                break
        self.q_corr_smooth[:] = 0.0

    def run(self):
        cfg = self.cfg
        sc = self.scene
        self.step_obstacles(0.0)
        self._start_proxies = B.per_obstacle_proxies(self.perceive(0.0), self.n_approx)
        self.move_to_start()
        xg = sc.goal
        x = self.ee()
        start_err = float(np.linalg.norm(x - sc.start))
        if cfg.gui:
            p.addUserDebugPoints([sc.start.tolist()], [[0, 1, 0]], 12, physicsClientId=self.cid)
            p.addUserDebugPoints([xg.tolist()], [[1, 0, 0]], 12, physicsClientId=self.cid)
            if cfg.wait_enter:
                input("Arm at start. Press Enter to begin...")
        ctrl_div = max(1, int(round(PHYSICS_HZ / cfg.ctrl_hz)))
        sw_div = max(1, int(round(PHYSICS_HZ / cfg.sw_hz)))
        dt_ctrl = ctrl_div / PHYSICS_HZ

        t = 0.0
        shapes = self.perceive(0.0)
        prev_c = np.array([s.center for s in shapes])
        vels = np.zeros_like(prev_c)
        t0 = time.perf_counter()
        proxies, info = self.representation(shapes, x, xg)
        rep_times = [time.perf_counter() - t0]
        arm_proxies = proxies if cfg.method not in ('cbf', 'apf') else \
            B.per_obstacle_proxies(shapes, self.n_approx)
        n_prox = [len(proxies)]
        sw_used = [info['sw_used']]
        cluster_seq = [info['clusters']]
        x_target = x.copy()
        path = 0.0
        x_prev = x.copy()
        min_clear = np.inf
        min_ee_true = np.inf
        margin_viol = 0
        n_ctrl = 0
        vel_times = []
        collided = False
        reached = False
        t_reach = np.nan
        stall = 0
        stall_run = 0
        n_flips = 0
        clip_n = 0
        manip_min = np.inf
        jl_margin_min = np.inf
        log = {'t': [], 'x': [], 'q': [], 'obs': [], 'v': [], 'proxies': [], 'manip': [],
               'jl': [], 'clip': [], 'gamma_min': []} if cfg.record else None
        step = 0
        while True:
            if cfg.arm_mode == 'ns':
                self.apply_ik(x_target, None)
            p.stepSimulation(physicsClientId=self.cid)
            step += 1
            t += 1.0 / PHYSICS_HZ
            self.step_obstacles(t)
            if cfg.realtime:
                time.sleep(1.0 / PHYSICS_HZ)

            if step % sw_div == 0:
                shapes = self.perceive(t)
                cc = np.array([s.center for s in shapes])
                vels = (cc - prev_c) / (sw_div / PHYSICS_HZ)
                prev_c = cc
                t0 = time.perf_counter()
                proxies, info = self.representation(shapes, self.ee(), xg)
                rep_times.append(time.perf_counter() - t0)
                arm_proxies = proxies if cfg.method not in ('cbf', 'apf') else \
                    B.per_obstacle_proxies(shapes, self.n_approx)
                n_prox.append(len(proxies))
                sw_used.append(info['sw_used'])
                cluster_seq.append(info['clusters'])
                if cfg.gui and step % (sw_div * 2) == 0:
                    self.draw(proxies)
                if cfg.record:
                    log['proxies'].append((t, [(np.asarray(pp.center_point),
                                                [np.asarray(pc) for pc in pp.pieces])
                                               for pp in proxies]))

            if step % ctrl_div == 0:
                n_ctrl += 1
                x = self.ee()
                path += float(np.linalg.norm(x - x_prev))
                if cfg.gui and self._trace_prev is not None:
                    p.addUserDebugLine(self._trace_prev.tolist(), x.tolist(), [1, 1, 1], 2,
                                       physicsClientId=self.cid)
                self._trace_prev = x.copy()
                ee_speed = float(np.linalg.norm(x - x_prev)) / dt_ctrl
                x_prev = x.copy()
                # safety metrics on TRUE geometry
                clr = self.min_clearance()
                min_clear = min(min_clear, clr)
                if clr < -0.005:
                    collided = True
                true_now = sc.true_shapes(t)
                sd_true = min(float(s.signed_distance(x)[0]) for s in true_now)
                min_ee_true = min(min_ee_true, sd_true)
                sd_infl = min(float(s.signed_distance(x)[0]) for s in shapes)
                if sd_infl < -0.01:
                    margin_viol += 1
                # control
                t0 = time.perf_counter()
                v = self.velocity(x, xg, proxies, shapes, vels)
                vel_times.append(time.perf_counter() - t0)
                xt = x + v * dt_ctrl
                xc = np.clip(xt, WS_LO, WS_HI)
                if np.any(np.abs(xc - xt) > 1e-9):
                    clip_n += 1
                x_target = xc
                ql = self.q_all()
                if cfg.arm_mode == 'qp':
                    qd = self.qp_velocity((xc - x) / dt_ctrl, ql, self.perceived_true, vels)
                    self.command_joint_velocity(qd, ql, dt_ctrl)
                elif cfg.arm_avoid:
                    c = self.nullspace_correction(arm_proxies, ql)
                    raw = c if c is not None else np.zeros(7)
                    self.q_corr_smooth = self.arm_ema * raw + (1 - self.arm_ema) * self.q_corr_smooth
                # diagnostics
                J = self.jac(EE_LINK, [0, 0, 0], ql)
                manip = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
                manip_min = min(manip_min, manip)
                q7 = np.array(ql[:7])
                jl = float(np.min(np.minimum(q7 - LL, UL - q7)))
                jl_margin_min = min(jl_margin_min, jl)
                dist = float(np.linalg.norm(x - xg))
                if ee_speed < 0.01 and dist > 0.05:
                    stall += 1
                # deadlock detection → kernel flip (star-world methods only)
                if ee_speed < 0.02 and dist > 0.05 and np.linalg.norm(v) > 0.1:
                    stall_run += 1
                else:
                    stall_run = 0
                if (stall_run * dt_ctrl >= 1.0 and 'flip' not in self.abl
                        and cfg.method in ('sw', 'sw_adaptive') and info.get('sw_used')):
                    if self.updater.flip(x, xg):
                        n_flips += 1
                        proxies, info = self.representation(shapes, x, xg)
                    stall_run = 0
                if cfg.record:
                    log['t'].append(t)
                    log['x'].append(x.copy())
                    log['q'].append(q7.copy())
                    log['obs'].append(np.array([sc.center(i, t) for i in range(len(sc.obs))]))
                    log['v'].append(np.asarray(v).copy())
                    log['manip'].append(manip)
                    log['jl'].append(jl)
                    log['clip'].append(bool(np.any(np.abs(xc - xt) > 1e-9)))
                    log['gamma_min'].append(min([self.ctrl._gamma(x, pp) for pp in proxies],
                                                default=np.inf))
                if dist < cfg.goal_tol:
                    reached = True
                    t_reach = t
                    break
                if t >= cfg.timeout:
                    break
        final_dist = float(np.linalg.norm(self.ee() - xg))
        straight = float(np.linalg.norm(xg - sc.start))
        flips = sum(1 for a, b in zip(cluster_seq[:-1], cluster_seq[1:])
                    if a is not None and b is not None and a != b)
        res = dict(asdict(cfg))
        res.pop('record', None)
        res.update(
            reached=reached, collided=collided, success=bool(reached and not collided),
            t_reach=t_reach, t_end=t, path=path, straight=straight,
            ratio=path / straight if straight > 0 else np.nan,
            final_dist=final_dist, start_err=start_err,
            min_clearance=float(min_clear), min_ee_true=float(min_ee_true),
            margin_viol_frac=margin_viol / max(n_ctrl, 1),
            rep_ms_mean=1e3 * float(np.mean(rep_times)),
            rep_ms_max=1e3 * float(np.max(rep_times)),
            vel_ms_mean=1e3 * float(np.mean(vel_times)) if vel_times else np.nan,
            n_proxies_mean=float(np.mean(n_prox)), n_obstacles=len(sc.obs),
            sw_used_frac=float(np.mean(sw_used)), cluster_flips=flips,
            n_updates=len(cluster_seq),
            stall_frac=stall / max(n_ctrl, 1), clip_frac=clip_n / max(n_ctrl, 1),
            n_flips=n_flips,
            manip_min=manip_min, jl_margin_min=jl_margin_min,
            dynamic=sc.dynamic,
        )
        if cfg.record:
            res['log'] = log
            res['start_pos'] = sc.start
            res['goal_pos'] = xg
        return res

    def close(self):
        try:
            p.disconnect(self.cid)
        except Exception:
            pass


def run_trial(cfg: TrialConfig):
    sim = FrankaSim(cfg)
    try:
        return sim.run()
    finally:
        sim.close()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='overlap')
    ap.add_argument('--method', default='sw')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--timeout', type=float, default=60)
    ap.add_argument('--no-rand', action='store_true')
    a = ap.parse_args()
    r = run_trial(TrialConfig(scenario=a.scenario, method=a.method, seed=a.seed,
                              timeout=a.timeout, randomize=not a.no_rand))
    for k in ['reached', 'collided', 'success', 't_reach', 'ratio', 'min_clearance',
              'min_ee_true', 'margin_viol_frac', 'rep_ms_mean', 'stall_frac', 'clip_frac',
              'manip_min', 'jl_margin_min', 'n_proxies_mean', 'start_err']:
        print(f"{k:18s} {r[k]}")
