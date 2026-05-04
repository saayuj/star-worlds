"""
PyBullet + Franka Panda + Star-Worlds + Reactive Controller demo.

Two modes
---------
Default (star-worlds ON):
    Star-worlds algorithm (Algorithm 2, 3-D) reshapes sphere obstacles into
    disjoint star-shaped proxies.  The Huber modulation controller uses the
    proxy kernel centres x_{c,i} to compute a collision-free EE velocity.

Raw mode (--raw, star-worlds OFF):
    Algorithm 2 is skipped entirely.  The Huber controller modulates directly
    against the raw sphere obstacles (one proxy per sphere, centre = sphere
    centre).  This is the naive baseline that gets stuck in local minima when
    obstacles overlap.
"""

import argparse
import sys
import os
import numpy as np
import pybullet as p
import pybullet_data
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from starworlds import StarWorldUpdater, Ellipsoid, ObstacleProxy
from starworlds.reactive_controller import ReactiveController
from examples.obstacle_scenarios import get_scenario, SCENARIO_WAYPOINTS


# ──────────────────────────────────────────────────────────────────────────────
# Demo class
# ──────────────────────────────────────────────────────────────────────────────

class FrankaStarWorldsDemo:
    """
    Franka arm + Star-Worlds + modulation-based reactive controller.
    """

    # Physics rate (Hz) and per-step divisors
    PHYSICS_HZ = 240
    CONTROL_DIVIDER = 6           # velocity update:    240/6  = 40 Hz
    SW_UPDATE_DIVIDER = 60        # star-world rebuild:  240/60 =  4 Hz
    VIZ_UPDATE_DIVIDER = 240      # wireframe redraw:   240/240 =  1 Hz

    # ── Full-arm collision avoidance (APF + null-space IK biasing) ──────────
    # ALL 9 Franka links are monitored (joints 0–8).
    # Each link is approximated as a capsule: we sample it at fractions
    # [0.0, 0.5, 1.0] along the segment between consecutive joint origins.
    # Fraction 0.0 = this joint's origin; 1.0 = next joint's origin (owned
    # by the child link but attributed to the parent for the Jacobian).
    # Link 8 (panda_hand) uses [0.0, 0.5] only — EE covers the tip.
    #   joint  0 → panda_link1   (shoulder)
    #   joint  1 → panda_link2   (upper-arm)
    #   joint  2 → panda_link3   (elbow-upper)
    #   joint  3 → panda_link4   (elbow-lower)
    #   joint  4 → panda_link5   (forearm)
    #   joint  5 → panda_link6   (pre-wrist)
    #   joint  6 → panda_link7   (wrist)
    #   joint  7 → panda_link8   (fixed — hand base)
    #   joint  8 → panda_hand    (fixed — palm)
    ARM_MONITOR_LINK_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    ARM_CAPSULE_FRACTIONS    = [0.0, 0.5, 1.0]   # fractions along each link segment
    # Physical radius of each monitored link [m] — used to compute Gamma relative
    # to the link's actual surface, not its centre-line.
    # Franka Panda approximate collision radii from URDF:
    ARM_LINK_RADII = {
        0: 0.06, 1: 0.06, 2: 0.06, 3: 0.06,
        4: 0.05, 5: 0.05, 6: 0.04, 7: 0.03, 8: 0.04,
    }
    ARM_GAMMA_THRESHOLD = 1.6   # Gamma threshold relative to link surface
                                # (Gamma=1 → link surface at proxy boundary)
    ARM_CORRECTION_GAIN = 0.30  # Jacobian-transpose APF gain
    ARM_POSTURE_GAIN    = 0.08  # null-space posture attraction gain
    ARM_EMA_ALPHA       = 0.10  # EMA smoothing: smaller = smoother/slower
    ARM_MAX_NS_NORM     = 0.07  # max L2 norm of null-space step [rad]
    # Nominal Franka posture: elbow-up, reach-forward — prevents weird flips
    Q_NOMINAL = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.8, 0.785])

    def __init__(self, use_gui: bool = True, scenario: str = 'simple',
                 use_raw: bool = False) -> None:
        self.use_raw = use_raw          # True → bypass Algorithm 2 (naive baseline)
        # ── PyBullet setup ────────────────────────────────────────────────────
        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")

        # ── Robot ─────────────────────────────────────────────────────────────
        self.franka_id = self._load_franka()
        self.ee_link_index = 11          # panda_hand link

        # Pre-compute movable-joint list once (FIXED joints excluded from
        # PyBullet Jacobian DOF count; invariant across the simulation).
        nj_total = p.getNumJoints(self.franka_id)
        self._movable_joints: list = [
            ji for ji in range(nj_total)
            if p.getJointInfo(self.franka_id, ji)[2] != p.JOINT_FIXED
        ]
        self._movable_dof: int = len(self._movable_joints)

        # ── Scenario ──────────────────────────────────────────────────────────
        self.scenario_name = scenario
        print(f"Loading scenario: {scenario}")
        self.obstacles = self._create_obstacles_from_scenario(scenario)
        print(f"Created {len(self.obstacles)} obstacles")

        # ── Star-Worlds updater ───────────────────────────────────────────────
        # n_approx_3d=16 keeps each update under ~5 ms so the physics loop
        # runs at near real-time even during star-world rebuilds.
        self.star_updater = StarWorldUpdater(n_approx_3d=16)

        # ── Reactive controller ───────────────────────────────────────────────
        self.controller = ReactiveController(
            v_max=0.30,
            eta=1.0,
            weight_power=2.0,
            safety_radius=0.01,   # small; obstacles already inflated +5 cm in Alg.2
            goal_tolerance=0.02,  # 2 cm — tight; checked at 40 Hz so not missed
        )

        # ── Workspace bounds (used to clamp IK targets) ───────────────────────
        self.ws_lower = np.array([0.10, -0.75, 0.10])
        self.ws_upper = np.array([0.95,  0.75, 0.90])

        # ── Visualisation book-keeping ────────────────────────────────────────
        self.proxy_visual_ids: list = []   # createMultiBody IDs (kernel dots)
        self.proxy_line_ids:   list = []   # addUserDebugLine IDs (hull wireframe)
        self.trace_line_ids:   list = []
        self._vel_arrow_id:    int  = -1
        self._prev_ee: np.ndarray | None = None
        # Raw APF joint-space correction Δq and its EMA-smoothed version.
        # _nullspace_correction writes _arm_q_corr; _apply_ik reads
        # _arm_q_corr_smooth (updated via EMA each control step).
        self._arm_q_corr:        np.ndarray | None = None
        self._arm_q_corr_smooth: np.ndarray        = np.zeros(7)
        # Cached joint-state list (all movable joints) written once per control
        # step in _nullspace_correction and reused in _apply_ik to avoid a
        # second round of getJointState calls.
        self._last_q_list: list | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Robot loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_franka(self) -> int:
        try:
            robot_id = p.loadURDF(
                "franka_panda/panda.urdf",
                basePosition=[0, 0, 0],
                useFixedBase=True,
            )
        except Exception:
            print("Warning: franka_panda/panda.urdf not found — trying ur5.urdf")
            robot_id = p.loadURDF("ur5.urdf", basePosition=[0, 0, 0],
                                  useFixedBase=True)
        num_joints = p.getNumJoints(robot_id)
        for ji in range(num_joints):
            p.resetJointState(robot_id, ji, 0.0)
        return robot_id

    # ──────────────────────────────────────────────────────────────────────────
    # Obstacle / scenario helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _create_obstacles_from_scenario(self, scenario_name: str) -> list:
        """
        Create obstacles as VISUAL-ONLY bodies (no collision geometry).

        Passing baseCollisionShapeIndex=-1 tells PyBullet to render the sphere
        but not register it in the physics engine.  The robot arm can therefore
        pass through it physically, eliminating spurious body-link collisions
        that previously corrupted the EE trajectory.

        All star-worlds and reactive-controller computations are unaffected:
        obstacle positions are still queried via getBasePositionAndOrientation
        and passed to Algorithm 2 / the modulation controller exactly as before.
        """
        obs_list = []
        for cfg in get_scenario(scenario_name):
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=cfg['radius'],
                                      rgbaColor=cfg['color'])
            # baseCollisionShapeIndex=-1  →  visual only, no physics collision
            body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                     baseVisualShapeIndex=vis,
                                     basePosition=cfg['position'])
            obs_list.append({'id': body, 'radius': cfg['radius'],
                              'trajectory': cfg['trajectory']})
        return obs_list

    def update_obstacle_positions(self, t: float) -> None:
        for obs in self.obstacles:
            pos = obs['trajectory'](t)
            p.resetBasePositionAndOrientation(obs['id'], pos.tolist(), [0,0,0,1])

    def get_starworld_obstacles_3d(self) -> list:
        """Return current obstacle positions as Ellipsoid objects (3-D)."""
        ellipsoids = []
        for obs in self.obstacles:
            pos, _ = p.getBasePositionAndOrientation(obs['id'])
            r = obs['radius'] + 0.05          # 5 cm safety margin
            ellipsoids.append(Ellipsoid(center=np.array(pos), Q=(r**2)*np.eye(3)))
        return ellipsoids

    # ──────────────────────────────────────────────────────────────────────────
    # EE helpers
    # ──────────────────────────────────────────────────────────────────────────

    def get_ee_position(self) -> np.ndarray:
        state = p.getLinkState(self.franka_id, self.ee_link_index)
        return np.array(state[0])

    def _get_arm_collision_points(self) -> list:
        """
        Return [(world_pos, link_idx, local_offset), ...] for all arm sample
        points.  Each link is modelled as a capsule sampled at
        ARM_CAPSULE_FRACTIONS along the segment from this joint's world origin
        to the next joint's world origin.

        local_offset is expressed in the link's own frame so it can be passed
        directly to p.calculateJacobian.  For fraction 0.0 it is [0,0,0];
        for fractions > 0 it is R_i^T (frac*(p_{i+1} - p_i)).

        Perf: origins/quats fetched via a single getLinkStates call (one
        C-extension round-trip instead of N); quat→matrix fully vectorised
        with numpy broadcasting; inner fraction loop uses pre-computed segs.
        """
        indices = self.ARM_MONITOR_LINK_INDICES
        n_links = len(indices)

        # ── Single batch FK query ──────────────────────────────────────────
        states = p.getLinkStates(self.franka_id, indices,
                                  computeForwardKinematics=True)
        # state[4] = worldLinkFramePosition, state[5] = orientation (xyzw)
        origins = np.array([s[4] for s in states], dtype=float)  # (N,3)
        quats   = np.array([s[5] for s in states], dtype=float)  # (N,4) xyzw

        # ── Vectorised quat → rotation matrices ───────────────────────────
        # quats columns: x y z w
        x, y, z, w = quats[:,0], quats[:,1], quats[:,2], quats[:,3]
        # Build (N,3,3) rotation matrices without any Python loops
        Rs = np.empty((n_links, 3, 3), dtype=float)
        Rs[:,0,0] = 1 - 2*(y*y + z*z);  Rs[:,0,1] = 2*(x*y - z*w);  Rs[:,0,2] = 2*(x*z + y*w)
        Rs[:,1,0] = 2*(x*y + z*w);       Rs[:,1,1] = 1 - 2*(x*x + z*z);  Rs[:,1,2] = 2*(y*z - x*w)
        Rs[:,2,0] = 2*(x*z - y*w);       Rs[:,2,1] = 2*(y*z + x*w);       Rs[:,2,2] = 1 - 2*(x*x + y*y)

        # ── Segment vectors (p_{i+1} − p_i) ──────────────────────────────
        segs = np.empty_like(origins)               # (N,3)
        segs[:-1] = origins[1:] - origins[:-1]
        segs[-1]  = np.array([0.0, 0.0, 0.05])      # extend last link toward EE

        # ── Build sample points ───────────────────────────────────────────
        fracs = np.asarray(self.ARM_CAPSULE_FRACTIONS, dtype=float)  # (F,)
        # world_pos[k,f] = origins[k] + fracs[f]*segs[k]  →  (N,F,3)
        world_pts = origins[:, None, :] + fracs[None, :, None] * segs[:, None, :]  # (N,F,3)
        # local_off[k,f] = Rs[k].T @ (fracs[f]*segs[k])  →  (N,F,3)
        frac_segs  = fracs[None, :, None] * segs[:, None, :]          # (N,F,3)
        local_offs = np.einsum('nij,nfj->nfi', Rs, frac_segs)         # (N,F,3) via Rᵀ@v = einsum

        points = []
        for k, li in enumerate(indices):
            for fi, _ in enumerate(fracs):
                points.append((
                    world_pts[k, fi],
                    li,
                    local_offs[k, fi].tolist(),
                ))
        return points

    def _link_gamma(self, world_pos: np.ndarray, proxy, link_idx: int) -> float:
        """
        Compute Gamma for an arm link sample point, accounting for the link's
        physical radius.

        Standard Gamma = d / r_proxy(dir), where d is distance from the proxy
        kernel centre and r_proxy is the direction-dependent proxy radius.
        This treats the link as a zero-radius point: Gamma=1 means the link
        CENTRE is at the proxy boundary.

        Arm links have physical radius ~4–6 cm.  We extend the effective proxy
        radius by ARM_LINK_RADII[link_idx] so that:
            Gamma = 1  ⇒  link SURFACE touching the proxy boundary
            Gamma > 1  ⇒  link surface outside the proxy
            Gamma < 1  ⇒  link surface penetrating the proxy

        This makes the threshold ARM_GAMMA_THRESHOLD physically meaningful
        with respect to the actual collision geometry.
        """
        c       = np.asarray(proxy.center_point, dtype=float)
        # Direction-dependent proxy radius + link physical radius
        r_proxy = self.controller._radius_in_direction(proxy, world_pos)
        link_r  = self.ARM_LINK_RADII.get(link_idx, 0.05)
        r_eff   = r_proxy + link_r
        if r_eff < 1e-10:
            return 10.0
        d = float(np.linalg.norm(world_pos - c))
        return max(d / r_eff, 1e-3)

    def _nullspace_correction(
        self,
        arm_points: list,
        proxies: list,
    ) -> np.ndarray | None:
        """
        Compute the raw APF joint-space correction Δq for null-space projection.

        Two terms are combined:

        1. APF obstacle repulsion
           For each capsule sample x_j with Γ_j < ARM_GAMMA_THRESHOLD
           (where Γ accounts for the link's physical radius via _link_gamma):
             • Repulsion direction from the star-world proxy kernel centre c_i*
               (NOT the raw obstacle centre — this respects the proxy geometry).
             • APF weight: w = (1/(Γ-1) - 1/(Γ_th-1)) / (1/(Γ_th-1)), capped at 5.
             • Δq += gain_apf · w · J_jᵀ r̂

        2. Posture attraction (null-space damping)
           Always-on term that gently draws joints toward Q_NOMINAL.
           Δq += gain_posture · (Q_NOMINAL - q_cur)
           This prevents the arm from drifting into mechanically disadvantaged
           configurations (wrist singularities, elbow flips) when repeated
           APF pushes accumulate.

        Returns the combined Δq (shape (7,)), or None if no link is close to
        any proxy AND posture error is negligible.
        """
        nj_arm   = 7
        q_list   = [p.getJointState(self.franka_id, ji)[0]
                    for ji in self._movable_joints]
        zero_vel = [0.0] * self._movable_dof
        q_cur    = np.array(q_list[:nj_arm], dtype=float)
        # Share with _apply_ik to avoid a second getJointState round-trip.
        self._last_q_list = q_list

        q_corr    = np.zeros(nj_arm)
        any_close = False
        apf_denom = max(self.ARM_GAMMA_THRESHOLD - 1.0, 1e-6)

        # ── Precompute proxy geometry (done once, reused for all arm points) ─
        n_prx = len(proxies)
        if n_prx == 0:
            pass  # fall through to posture term
        else:
            # proxy_centers: (P,3),  proxy_r_dirs computed per point below
            proxy_centers = np.array(
                [proxy.center_point for proxy in proxies], dtype=float
            )  # (P,3)
            # Per-proxy link radii contribution (we add it after r_proxy lookup)
            # link_radii indexed by link_idx, fetched inside the arm_points loop.

            # ── Term 1: APF obstacle repulsion ─────────────────────────────
            # Jacobians cached by link_idx (same J for all fractions of same link
            # that share the same local_off are cached by (link_idx, frac_tuple)).
            _jac_cache: dict = {}

            for (world_pos, link_idx, local_off) in arm_points:
                link_r = self.ARM_LINK_RADII.get(link_idx, 0.05)

                # Vectorised gamma for ALL proxies at once ──────────────────
                # d_vecs: (P,3),  d_norms: (P,)
                d_vecs  = world_pos[None, :] - proxy_centers        # (P,3)
                d_norms = np.linalg.norm(d_vecs, axis=1)            # (P,)

                # Direction-dependent proxy radius for each proxy
                # (calls into Python but unavoidable — one call per proxy)
                r_proxies = np.array([
                    self.controller._radius_in_direction(proxy, world_pos)
                    for proxy in proxies
                ], dtype=float)                                       # (P,)
                r_effs = r_proxies + link_r                          # (P,)
                r_effs = np.maximum(r_effs, 1e-10)

                gammas = np.maximum(d_norms / r_effs, 1e-3)         # (P,)

                # Only process proxies below threshold
                active = np.where(gammas < self.ARM_GAMMA_THRESHOLD)[0]
                if active.size == 0:
                    continue
                any_close = True

                # Fetch Jacobian once per (link, local_off) combination
                cache_key = (link_idx, tuple(local_off))
                if cache_key not in _jac_cache:
                    Jv, _ = p.calculateJacobian(
                        self.franka_id, link_idx, local_off,
                        q_list, zero_vel, zero_vel,
                    )
                    _jac_cache[cache_key] = np.array(Jv)[:, :nj_arm]  # (3,7)
                Jv_arm = _jac_cache[cache_key]   # (3,7)

                for pi in active:
                    dn = d_norms[pi]
                    if dn < 1e-8:
                        continue
                    repulsion = d_vecs[pi] / dn
                    g   = gammas[pi]
                    rho = max(g - 1.0, 1e-4)
                    weight = min((1.0/rho - 1.0/apf_denom) / (1.0/apf_denom), 5.0)
                    weight = max(weight, 0.0)
                    q_corr += self.ARM_CORRECTION_GAIN * weight * (Jv_arm.T @ repulsion)

        # ── Term 2: posture attraction ───────────────────────────────────
        # Always-on: pulls toward nominal posture in the null space.
        posture_err = self.Q_NOMINAL - q_cur
        posture_norm = float(np.linalg.norm(posture_err))
        if posture_norm > 0.05:   # only apply if noticeably off-nominal
            q_corr += self.ARM_POSTURE_GAIN * posture_err
            any_close = True

        return q_corr if any_close else None

    def _apply_ik(self, target: np.ndarray) -> None:
        """
        Command the arm toward `target` EE position with null-space arm avoidance.

        Flow
        ----
        1. Read current joint positions q_cur.
        2. Run IK seeded from q_cur (restPoses=q_cur) so the solver returns
           the minimal-joint-motion solution — smooth, no discontinuous jumps.
        3. If avoidance correction is active:
               J    = EE position Jacobian at q_cur  (3×7)
               N    = I - J†J   (null-space projector, rank 4)
               dq_ns = N @ corr_smooth
               scale dq_ns to norm ≤ ARM_MAX_NS_NORM WITHOUT per-component clip
               (clipping per-component breaks the null-space property and moves the EE).
        4. q_cmd = q_ik + dq_ns, clipped to joint limits.
        """
        nj_arm = 7
        ll = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        ul = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
        jr = (ul - ll).tolist()

        # Current joint positions
        q_cur = np.array([
            p.getJointState(self.franka_id, ji)[0]
            for ji in range(nj_arm)
        ], dtype=float)

        # IK seeded from current joints — minimises joint motion, avoids solution jumps
        q_ik_all = p.calculateInverseKinematics(
            self.franka_id, self.ee_link_index, target.tolist(),
            lowerLimits=ll.tolist(), upperLimits=ul.tolist(),
            jointRanges=jr, restPoses=q_cur.tolist(),
        )
        q_ik = np.array(q_ik_all[:nj_arm], dtype=float)

        # Null-space avoidance projection
        dq_ns = np.zeros(nj_arm)
        if np.any(self._arm_q_corr_smooth != 0):
            # Reuse joint state already fetched this step by _nullspace_correction
            q_list   = self._last_q_list if self._last_q_list is not None else [
                p.getJointState(self.franka_id, ji)[0] for ji in self._movable_joints
            ]
            zero_vel = [0.0] * self._movable_dof
            Jv_ee, _ = p.calculateJacobian(
                self.franka_id, self.ee_link_index, [0.0, 0.0, 0.0],
                q_list, zero_vel, zero_vel,
            )
            J = np.array(Jv_ee)[:, :nj_arm]   # (3, 7)
            J_pinv = np.linalg.pinv(J)
            N = np.eye(nj_arm) - J_pinv @ J
            raw_ns = N @ self._arm_q_corr_smooth

            # Scale the whole vector so its norm ≤ ARM_MAX_NS_NORM.
            # This preserves the null-space direction (J @ raw_ns ≈ 0 still holds)
            # unlike per-component clipping which breaks the null-space property
            # and causes the EE to drift.
            ns_norm = float(np.linalg.norm(raw_ns))
            if ns_norm > self.ARM_MAX_NS_NORM:
                raw_ns = raw_ns * (self.ARM_MAX_NS_NORM / ns_norm)
            dq_ns = raw_ns

        q_cmd = np.clip(q_ik + dq_ns, ll, ul)

        for ji in range(nj_arm):
            p.setJointMotorControl2(
                self.franka_id, ji,
                p.POSITION_CONTROL,
                targetPosition=q_cmd[ji],
                force=1000,
            )

    def _get_raw_proxies(self) -> list:
        """
        Build one fake ObstacleProxy per sphere using the sphere's *actual*
        centre as the repulsion centre — no star-world reshaping.

        This is the naive baseline: overlapping spheres each push outward from
        their own centres.  When two spheres overlap the +y and -y normals
        cancel while the goal-directed x component is suppressed → local minimum.
        """
        proxies = []
        for i, obs in enumerate(self.obstacles):
            pos, _ = p.getBasePositionAndOrientation(obs['id'])
            c = np.array(pos)
            r = obs['radius'] + 0.05

            # Fibonacci sphere surface (same resolution as n_approx_3d=16)
            n   = 16
            idx = np.arange(n)
            phi   = np.arccos(1 - 2 * (idx + 0.5) / n)
            theta = np.pi * (1 + np.sqrt(5)) * idx
            pts   = c + r * np.column_stack([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ])

            proxies.append(ObstacleProxy(
                pieces=[pts],
                kernel_K=None,
                center_point=c,      # raw sphere centre, NOT an Algorithm-2 kernel
                original_indices=[i],
                is_fallback=False,
            ))
        return proxies

    def _move_to_start(self, target: np.ndarray, max_steps: int = 1200,
                        tol: float = 0.03) -> None:
        """Drive the arm to `target` and wait for arrival (blocking)."""
        for step in range(max_steps):
            self._apply_ik(target)
            p.stepSimulation()
            time.sleep(1.0 / self.PHYSICS_HZ)
            if step % self.CONTROL_DIVIDER == 0:
                if np.linalg.norm(self.get_ee_position() - target) < tol:
                    return

    # ──────────────────────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────────────────────

    def _visualize_proxies(self, proxies: list) -> None:
        """
        Re-draw proxy visualisation using the ACTUAL convex hull wireframe.

        For each proxy:
          • Small solid green dot at the kernel centre (center_point).
          • Green wireframe of the true 3-D convex hull of proxy.pieces —
            one line per unique hull edge via scipy.spatial.ConvexHull.

        This shows the genuine star-shaped proxy boundary (a capsule/lens
        for two merged sphere obstacles), not a sphere approximation.
        """
        from scipy.spatial import ConvexHull

        for vid in self.proxy_visual_ids:
            try:
                p.removeBody(vid)
            except Exception:
                pass
        self.proxy_visual_ids.clear()

        for lid in self.proxy_line_ids:
            try:
                p.removeUserDebugItem(lid)
            except Exception:
                pass
        self.proxy_line_ids.clear()

        for proxy in proxies:
            xc = np.array(proxy.center_point)

            # ── Kernel centre: small solid bright-green dot ───────────────────
            vis_dot = p.createVisualShape(p.GEOM_SPHERE, radius=0.025,
                                          rgbaColor=[0, 1.0, 0, 1])
            self.proxy_visual_ids.append(
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_dot,
                                  basePosition=xc.tolist()))

            if not proxy.pieces:
                continue

            all_pts = np.vstack(proxy.pieces)
            if len(all_pts) < 4:
                continue

            # ── Convex hull wireframe ─────────────────────────────────────────
            try:
                hull = ConvexHull(all_pts)
            except Exception:
                continue

            # Each simplex is a triangle; collect unique edges to avoid
            # drawing every shared edge twice.
            edges: set = set()
            for simplex in hull.simplices:
                for i in range(3):
                    edges.add(tuple(sorted((simplex[i], simplex[(i + 1) % 3]))))

            for i, j in edges:
                lid = p.addUserDebugLine(
                    all_pts[i].tolist(),
                    all_pts[j].tolist(),
                    lineColorRGB=[0, 0.85, 0.25],
                    lineWidth=1.5,
                )
                self.proxy_line_ids.append(lid)

    def _draw_velocity_arrow(self, x: np.ndarray, v: np.ndarray) -> None:
        """Draw a short line in the direction of the current EE velocity."""
        if not p.isConnected():
            return
        if self._vel_arrow_id >= 0:
            try:
                p.removeUserDebugItem(self._vel_arrow_id)
            except Exception:
                pass
        speed = float(np.linalg.norm(v))
        if speed < 1e-6:
            self._vel_arrow_id = -1
            return
        tip = x + v / speed * 0.12   # 12 cm arrow head
        self._vel_arrow_id = p.addUserDebugLine(
            x.tolist(), tip.tolist(),
            lineColorRGB=[1, 0.5, 0], lineWidth=3,
        )

    def _trace_ee(self, x: np.ndarray) -> None:
        """Append a white trace segment from previous EE position to current."""
        if self._prev_ee is not None and p.isConnected():
            lid = p.addUserDebugLine(
                self._prev_ee.tolist(), x.tolist(),
                lineColorRGB=[1, 1, 1], lineWidth=1,
            )
            self.trace_line_ids.append(lid)
        self._prev_ee = x.copy()

    # ──────────────────────────────────────────────────────────────────────────
    # Main reactive control loop
    # ──────────────────────────────────────────────────────────────────────────

    def run_reactive_demo(self) -> None:
        """
        Reactive control loop:
          - Every physics step: apply IK to current x_target.
          - Every CONTROL_DIVIDER steps: update star-world, compute new velocity,
            advance x_target by one velocity step.
        """
        print("=" * 60)
        print("Franka + Star-Worlds + Reactive Controller")
        print("=" * 60)

        # Stabilise
        for _ in range(100):
            p.stepSimulation()

        # Waypoints from scenario
        wp = SCENARIO_WAYPOINTS.get(self.scenario_name,
                                    {'start': [0.3, -0.1, 0.4],
                                     'goal':  [0.6, -0.3, 0.4]})
        start = np.array(wp['start'], dtype=float)
        goal  = np.array(wp['goal'],  dtype=float)

        # Initialise obstacles at t=0
        t = 0.0
        self.update_obstacle_positions(t)

        # Wait for user — PyBullet window is already open so they can inspect
        # the scene before anything starts moving.
        print(f"\nStart : {np.round(start, 3)}")
        print(f"Goal  : {np.round(goal, 3)}")
        # Move arm to start
        print(f"Moving to start position...")
        self._move_to_start(start)
        print("At start position.")

        # Draw start (green) and goal (red) markers
        p.addUserDebugPoints([start.tolist()], [[0, 1, 0]], pointSize=15)
        p.addUserDebugPoints([goal.tolist()],  [[1, 0, 0]], pointSize=15)
        # Draw a line showing the direct (unobstructed) route
        p.addUserDebugLine(start.tolist(), goal.tolist(),
                           lineColorRGB=[0.5, 0.5, 0.5], lineWidth=1)

        print(f"Goal: {np.round(goal, 3)}")
        print(f"dist_to_goal = {np.linalg.norm(goal - start):.3f} m")
        input("\nArm is at start position.  Frame your recording, then press Enter to begin...\n")

        # ── Control state ──────────────────────────────────────────────────────
        x_target = self.get_ee_position().copy()
        dt_control = self.CONTROL_DIVIDER / self.PHYSICS_HZ   # s per planning step
        max_sim_time = 60.0    # give up after 60 s
        report_interval = 1.0  # terminal print every 1 s
        last_report = -report_interval

        step = 0
        goal_reached = False
        current_proxies: list = []
        current_velocity = np.zeros(3)

        # ── Initial proxy computation ──────────────────────────────────────────
        mode_label = "RAW (no star-worlds)" if self.use_raw else "Star-Worlds (Algorithm 2)"
        print(f"\nMode: {mode_label}")
        x_now = self.get_ee_position()
        self.update_obstacle_positions(0.0)
        if self.use_raw:
            current_proxies = self._get_raw_proxies()
            # Raw mode: no star-worlds → no proxy visualization
        else:
            obs_3d = self.get_starworld_obstacles_3d()
            result = self.star_updater.update(obs_3d, x_now, goal)
            current_proxies = result.proxies
            self._visualize_proxies(current_proxies)

        while p.isConnected():
            # ── Physics step (every iteration, 240 Hz) ─────────────────────────
            self._apply_ik(x_target)
            p.stepSimulation()
            time.sleep(1.0 / self.PHYSICS_HZ)
            step += 1
            t += 1.0 / self.PHYSICS_HZ

            # ── Velocity update (every CONTROL_DIVIDER steps, 40 Hz) ───────────
            # Recomputes the velocity using the CURRENT EE position and the
            # most-recently-built set of proxies (no blocking call here).
            if step % self.CONTROL_DIVIDER == 0:
                x_now = self.get_ee_position()

                current_velocity = self.controller.compute_velocity(
                    x_now, goal, current_proxies
                )
                x_target = x_now + current_velocity * dt_control
                x_target = np.clip(x_target, self.ws_lower, self.ws_upper)

                # ── Null-space arm-link correction ────────────────────────────
                # Sample world positions of monitored arm links and compute a
                # Jacobian-transpose joint correction that pushes any link
                # with Γ < ARM_GAMMA_THRESHOLD away from obstacle proxies.
                # The correction is stored and used by every _apply_ik call
                # until the next control step (null-space IK biasing).
                if current_proxies:
                    arm_pts = self._get_arm_collision_points()
                    self._arm_q_corr = self._nullspace_correction(
                        arm_pts, current_proxies
                    )
                else:
                    self._arm_q_corr = None

                # EMA smoothing: blend new raw correction into the smooth signal.
                # When no links are close (None), decay the smooth signal to zero.
                raw = self._arm_q_corr if self._arm_q_corr is not None else np.zeros(7)
                alpha = self.ARM_EMA_ALPHA
                self._arm_q_corr_smooth = (
                    alpha * raw + (1.0 - alpha) * self._arm_q_corr_smooth
                )

                self._draw_velocity_arrow(x_now, current_velocity)
                self._trace_ee(x_now)

                # ── Goal check at control rate (40 Hz) so we don't overshoot ──
                dist_to_goal = float(np.linalg.norm(x_now - goal))
                if dist_to_goal < self.controller.goal_tolerance:
                    goal_reached = True
                    print(f"\n✓ Goal reached at t={t:.2f}s  "
                          f"(dist={dist_to_goal:.3f} m)")
                    break

                # ── Timeout ────────────────────────────────────────────────────
                if t > max_sim_time:
                    dist = float(np.linalg.norm(x_now - goal))
                    print(f"\n⚠  Time limit reached ({max_sim_time:.0f}s).  "
                          f"dist_to_goal={dist:.3f} m")
                    break

            # ── Proxy rebuild (every SW_UPDATE_DIVIDER steps, 4 Hz) ───────────
            if step % self.SW_UPDATE_DIVIDER == 0:
                self.update_obstacle_positions(t)
                if self.use_raw:
                    current_proxies = self._get_raw_proxies()
                else:
                    obs_3d = self.get_starworld_obstacles_3d()
                    result = self.star_updater.update(obs_3d, x_now, goal)
                    current_proxies = result.proxies
                    # Wireframe redrawn at VIZ_UPDATE_DIVIDER (1 Hz), not every
                    # SW rebuild (4 Hz), to keep the physics loop responsive.
                    if step % self.VIZ_UPDATE_DIVIDER == 0:
                        self._visualize_proxies(current_proxies)

                # ── Terminal report ────────────────────────────────────────────
                if t - last_report >= report_interval:
                    dist = float(np.linalg.norm(x_now - goal))
                    gammas_str = ""
                    arm_str = ""
                    if current_proxies and not self.use_raw:
                        gs = [self.controller._gamma(x_now, px)
                              for px in current_proxies]
                        gammas_str = " | Γ_ee=[" + " ".join(f"{g:.2f}" for g in gs) + "]"
                        # Per-link min Gamma across all capsule samples for each link
                        arm_pts = self._get_arm_collision_points()
                        link_labels = [f"L{li}" for li in self.ARM_MONITOR_LINK_INDICES]
                        # Collect min-gamma per link (each link has len(ARM_CAPSULE_FRACTIONS) samples)
                        n_fracs = len(self.ARM_CAPSULE_FRACTIONS)
                        arm_gs = []
                        for k in range(len(self.ARM_MONITOR_LINK_INDICES)):
                            samples = arm_pts[k * n_fracs : (k + 1) * n_fracs]
                            g_link = min(
                                min(self.controller._gamma(wp, px) for px in current_proxies)
                                for wp, _, _ in samples
                            )
                            arm_gs.append(g_link)
                        arm_str = " | Γ_arm=[" + " ".join(
                            f"{lbl}:{g:.1f}" for lbl, g in zip(link_labels, arm_gs)
                        ) + "]"
                    speed = float(np.linalg.norm(current_velocity))
                    print(f"t={t:5.1f}s | EE={np.round(x_now,3)} | "
                          f"dist={dist:.3f} m | v={speed:.3f} m/s | "
                          f"proxies={len(current_proxies)}{gammas_str}{arm_str}")
                    last_report = t

        if not goal_reached and p.isConnected():
            x_final = self.get_ee_position()
            print(f"\nFinal EE: {np.round(x_final, 3)}  "
                  f"dist_to_goal={np.linalg.norm(x_final - goal):.3f} m")

        print("\nDemo complete.  Close the window or press Ctrl+C to exit.")
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1.0 / self.PHYSICS_HZ)

    # ──────────────────────────────────────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────────────────────────────────────

    def run_demo(self) -> None:
        """Alias for run_reactive_demo (kept for backward compatibility)."""
        self.run_reactive_demo()

    def cleanup(self) -> None:
        if p.isConnected():
            p.disconnect()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Franka + Star-Worlds + Reactive Controller Demo'
    )
    parser.add_argument('--scenario', type=str, default='test',
                        choices=['overlap', 'test', 'simple', 'moderate', 'corridor'],
                        help='Obstacle scenario to use')
    parser.add_argument('--no-gui', action='store_true',
                        help='Run without GUI (headless)')
    parser.add_argument('--raw', action='store_true',
                        help='Naive baseline: skip star-worlds, modulate against '
                             'raw sphere centres (shows local-minimum failure)')
    args = parser.parse_args()

    demo = None
    try:
        demo = FrankaStarWorldsDemo(use_gui=not args.no_gui,
                                    scenario=args.scenario,
                                    use_raw=args.raw)
        demo.run_demo()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if demo is not None:
            demo.cleanup()
