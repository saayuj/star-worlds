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

Usage
-----
    python examples/franka_pybullet_demo.py --scenario overlap          # star-worlds ON
    python examples/franka_pybullet_demo.py --scenario overlap --raw    # naive baseline
    python examples/franka_pybullet_demo.py --scenario simple
    python examples/franka_pybullet_demo.py --no-gui --scenario test
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
    # Proxy wireframe uses addUserDebugLine which is a synchronous PyBullet
    # call.  A hull with ~60 edges redrawn at 4 Hz noticeably slows the loop.
    # Dropping viz to 1 Hz is invisible to the eye but cuts the overhead 4×.

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

    def _apply_ik(self, target: np.ndarray) -> None:
        """Send one IK position-control command."""
        joint_poses = p.calculateInverseKinematics(
            self.franka_id, self.ee_link_index, target.tolist()
        )
        nj = p.getNumJoints(self.franka_id)
        for ji in range(min(7, nj)):
            p.setJointMotorControl2(
                self.franka_id, ji,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[ji],
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
        print("\nPyBullet window is open.  Press Enter to start the simulation...")
        input()

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
        print("\nStarting reactive control loop.  Close window or Ctrl+C to stop.\n")

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
                    if current_proxies and not self.use_raw:
                        gs = [self.controller._gamma(x_now, px)
                              for px in current_proxies]
                        gammas_str = " | Γ=[" + " ".join(f"{g:.2f}" for g in gs) + "]"
                    speed = float(np.linalg.norm(current_velocity))
                    print(f"t={t:5.1f}s | EE={np.round(x_now,3)} | "
                          f"dist={dist:.3f} m | v={speed:.3f} m/s | "
                          f"proxies={len(current_proxies)}{gammas_str}")
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
