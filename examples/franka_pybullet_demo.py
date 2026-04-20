"""
PyBullet + Franka Panda + Star-Worlds + RRT integration demo.

Demonstrates:
1. Loading Franka Panda arm in PyBullet
2. Creating dynamic obstacles (moving spheres)
3. Using StarWorldUpdater to reshape obstacles into disjoint star-shaped proxies
4. Planning collision-free paths with RRT
5. Executing trajectories with inverse kinematics

Usage:
    python examples/franka_pybullet_demo.py
"""

import argparse
import sys
import os
import numpy as np
import pybullet as p
import pybullet_data
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from starworlds import StarWorldUpdater, Ellipsoid
from starworlds.rrt_planner import RRTPlanner
from examples.obstacle_scenarios import get_scenario, SCENARIO_WAYPOINTS


class FrankaStarWorldsDemo:
    """Integration of Franka arm with star-worlds obstacle reshaping and RRT planning."""
    
    def __init__(self, use_gui=True, scenario='simple'):
        # PyBullet setup
        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load environment
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load Franka Panda
        self.franka_id = self._load_franka()
        
        # Get end-effector link index
        self.ee_link_index = 11  # Franka panda_hand link
        
        # Load scenario
        self.scenario_name = scenario
        print(f"Loading scenario: {scenario}")
        self.obstacles = self._create_obstacles_from_scenario(scenario)
        print(f"Created {len(self.obstacles)} obstacles")
        
        # Star-worlds updater — n_approx_3d controls sphere surface resolution
        self.star_updater = StarWorldUpdater(n_approx_3d=64)

        # RRT planner — 3-D EE workspace bounds for the Franka arm
        workspace_lower_3d = np.array([0.15, -0.70, 0.15])
        workspace_upper_3d = np.array([0.90,  0.70, 0.85])

        self.rrt = RRTPlanner(
            step_size=0.04,
            goal_bias=0.20,
            max_iterations=15000,
            goal_tolerance=0.05,
            bounds=(workspace_lower_3d, workspace_upper_3d),
        )

        # Default Z for waypoints specified in obstacle_scenarios
        self.default_z = 0.4
        
        # Visualization
        self.path_line_ids = []
        self.proxy_visual_ids = []
        
    def _load_franka(self):
        """Load Franka Panda robot."""
        # Try to load from pybullet_data
        try:
            franka_id = p.loadURDF(
                "franka_panda/panda.urdf",
                basePosition=[0, 0, 0],
                useFixedBase=True
            )
        except:
            print("Warning: Could not load franka_panda/panda.urdf from pybullet_data")
            print("Attempting to use local URDF or falling back to UR5...")
            # Fallback to UR5 if Franka not available
            try:
                franka_id = p.loadURDF("ur5.urdf", basePosition=[0, 0, 0], useFixedBase=True)
                print("Loaded UR5 as fallback")
            except:
                print("Error: Could not load robot URDF")
                raise
                
        # Set initial joint positions (home configuration)
        num_joints = p.getNumJoints(franka_id)
        for joint_idx in range(num_joints):
            p.resetJointState(franka_id, joint_idx, 0.0)
            
        return franka_id
    
    def _create_obstacles_from_scenario(self, scenario_name):
        """Create obstacles from a predefined scenario."""
        scenario_config = get_scenario(scenario_name)
        obstacles = []
        
        for obs_config in scenario_config:
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=obs_config['radius'],
                rgbaColor=obs_config['color']
            )
            collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=obs_config['radius']
            )
            obs_body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=obs_config['position']
            )
            
            obstacles.append({
                'id': obs_body,
                'radius': obs_config['radius'],
                'trajectory': obs_config['trajectory']
            })
        
        return obstacles
    
    def get_ee_position(self):
        """Get current end-effector position in task space (3D)."""
        ee_state = p.getLinkState(self.franka_id, self.ee_link_index)
        return np.array(ee_state[0])
    
    
    def update_obstacle_positions(self, t):
        """Update dynamic obstacle positions based on time."""
        for obs in self.obstacles:
            new_pos = obs['trajectory'](t)
            p.resetBasePositionAndOrientation(
                obs['id'],
                new_pos.tolist(),
                [0, 0, 0, 1]
            )
    
    def get_starworld_obstacles_3d(self):
        """
        Read obstacle positions from PyBullet and return 3-D Ellipsoid objects.
        Each sphere (radius r) is modelled as Ellipsoid(center_3d, Q=r^2*I3).
        A 5 cm safety margin inflates each sphere.
        """
        ellipsoids = []
        for obs in self.obstacles:
            pos, _ = p.getBasePositionAndOrientation(obs['id'])
            r = obs['radius'] + 0.05      # 5 cm safety margin
            Q = (r ** 2) * np.eye(3)
            ellipsoids.append(Ellipsoid(center=np.array(pos), Q=Q))
        return ellipsoids
    
    def plan_path(self, start_3d, goal_3d):
        """
        Plan a 3-D collision-free path using star-worlds (3-D) + RRT.
        No 2-D projection — star-world and RRT both operate in (x,y,z) EE space.
        Returns a (N, 3) waypoint array or None.
        """
        start = np.asarray(start_3d, dtype=float)
        goal  = np.asarray(goal_3d,  dtype=float)

        print(f"  Start: {np.round(start, 3)}")
        print(f"  Goal:  {np.round(goal, 3)}")

        obstacles_3d = self.get_starworld_obstacles_3d()
        print(f"  Obstacles: {len(obstacles_3d)}")
        for i, obs in enumerate(obstacles_3d):
            r = float(np.sqrt(obs.Q[0, 0]))
            print(f"    Obs {i}: center={np.round(obs.center, 3)}, r={r:.3f}")

        # Star-world update (Ellipsoid.dim==3 -> dispatched to 3-D algorithm)
        result = self.star_updater.update(
            obstacles=obstacles_3d,
            robot_pos=start,
            goal_pos=goal,
        )

        print(f"  Star-world: {len(result.proxies)} proxies, "
              f"disjoint={result.is_disjoint}")
        obs_centers = [obs.center for obs in obstacles_3d]
        for i, proxy in enumerate(result.proxies):
            xc = proxy.center_point
            dists = [np.linalg.norm(xc - oc) for oc in obs_centers]
            md = min(dists)
            tag = "OK" if md < 0.5 else "FAR - possible bug"
            print(f"    Proxy {i}: center={np.round(xc, 3)}  "
                  f"(dist {md:.3f} to nearest obs) [{tag}]")

        self._visualize_proxies_3d(result.proxies)

        print(f"  Running 3-D RRT...")
        path = self.rrt.plan(
            start=start,
            goal=goal,
            obstacle_proxies=result.proxies,
        )

        if path is None:
            print(f"  RRT failed to find a path")
            return None

        length = sum(
            np.linalg.norm(path[k + 1] - path[k]) for k in range(len(path) - 1)
        )
        print(f"  Path: {len(path)} waypoints, length={length:.3f} m")
        return path

    def _move_ee_to_position(
        self,
        target_pos: np.ndarray,
        max_steps: int = 1200,
        tolerance: float = 0.03,
    ):
        """
        Move the EE to target_pos using IK, waiting until arrival or timeout.
        Replaces the old fixed-duration loop with an arrival-verified loop.
        """
        target = np.asarray(target_pos, dtype=float)
        num_joints = p.getNumJoints(self.franka_id)
        for step in range(max_steps):
            joint_poses = p.calculateInverseKinematics(
                self.franka_id,
                self.ee_link_index,
                target.tolist(),
            )
            for ji in range(min(7, num_joints)):
                p.setJointMotorControl2(
                    self.franka_id, ji,
                    p.POSITION_CONTROL,
                    targetPosition=joint_poses[ji],
                    force=500,
                )
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
            if step % 12 == 0:
                ee_pos = np.array(
                    p.getLinkState(self.franka_id, self.ee_link_index)[0]
                )
                if np.linalg.norm(ee_pos - target) < tolerance:
                    return

    def _visualize_proxies_3d(self, proxies):
        """Show each proxy centre (green sphere) and bounding sphere (transparent)."""
        for vid in self.proxy_visual_ids:
            p.removeBody(vid)
        self.proxy_visual_ids.clear()
        for proxy in proxies:
            xc = proxy.center_point.tolist()
            cvis = p.createVisualShape(p.GEOM_SPHERE, radius=0.02,
                                       rgbaColor=[0, 1, 0, 1])
            self.proxy_visual_ids.append(
                p.createMultiBody(baseMass=0,
                                  baseVisualShapeIndex=cvis,
                                  basePosition=xc)
            )
            if proxy.pieces:
                all_pts = np.vstack(proxy.pieces)
                max_r = float(np.max(
                    np.linalg.norm(all_pts - proxy.center_point, axis=1)
                ))
                if max_r > 0.01:
                    svis = p.createVisualShape(p.GEOM_SPHERE, radius=max_r,
                                               rgbaColor=[0, 1, 0, 0.15])
                    self.proxy_visual_ids.append(
                        p.createMultiBody(baseMass=0,
                                          baseVisualShapeIndex=svis,
                                          basePosition=xc)
                    )

    def visualize_path(self, path):
        """Draw the planned path as yellow line segments in PyBullet."""
        for line_id in self.path_line_ids:
            p.removeUserDebugItem(line_id)
        self.path_line_ids.clear()
        if path is None or len(path) < 2:
            return
        for i in range(len(path) - 1):
            lid = p.addUserDebugLine(
                path[i].tolist(),
                path[i + 1].tolist(),
                lineColorRGB=[1, 1, 0],
                lineWidth=3,
            )
            self.path_line_ids.append(lid)

    def execute_path(self, path, waypoint_tolerance: float = 0.04):
        """
        Execute a 3-D EE path with arrival-verified IK control.
        For each waypoint the arm drives via IK until within waypoint_tolerance
        or until a per-waypoint step budget is exhausted.
        """
        if path is None:
            print("No path to execute!")
            return
        print(f"Executing path with {len(path)} waypoints...")
        num_joints = p.getNumJoints(self.franka_id)
        for i, waypoint in enumerate(path):
            target = np.asarray(waypoint, dtype=float)
            for step in range(480):   # up to 2 s per waypoint at 240 Hz
                joint_poses = p.calculateInverseKinematics(
                    self.franka_id,
                    self.ee_link_index,
                    target.tolist(),
                )
                for ji in range(min(7, num_joints)):
                    p.setJointMotorControl2(
                        self.franka_id, ji,
                        p.POSITION_CONTROL,
                        targetPosition=joint_poses[ji],
                        force=500,
                    )
                p.stepSimulation()
                time.sleep(1.0 / 240.0)
                if step % 12 == 0:
                    ee_pos = np.array(
                        p.getLinkState(self.franka_id, self.ee_link_index)[0]
                    )
                    if np.linalg.norm(ee_pos - target) < waypoint_tolerance:
                        break

    def run_demo(self):
        """Run the complete demo."""
        print("="*60)
        print("Franka + Star-Worlds + RRT Demo")
        print("="*60)
        
        # Wait for simulation to stabilize
        for _ in range(100):
            p.stepSimulation()
        
        # Get waypoints from scenario
        waypoints = SCENARIO_WAYPOINTS.get(self.scenario_name, {
            'start': [0.3, -0.1, self.default_z],
            'goal': [0.6, -0.3, self.default_z]
        })
        start = np.array(waypoints['start'])
        goal = np.array(waypoints['goal'])
        
        # Move arm to start position first
        print(f"Moving arm to start position {start}...")
        self._move_ee_to_position(start)
        print("Arm at start position.")
        
        # Visualize start and goal
        p.addUserDebugPoints([start.tolist()], [[0, 1, 0]], pointSize=15)
        p.addUserDebugPoints([goal.tolist()], [[1, 0, 0]], pointSize=15)
        
        # Plan and execute path
        t = 0
        
        # Update obstacle positions at t=0
        self.update_obstacle_positions(t)

        # Use actual EE position as planning start
        current_pos = self.get_ee_position()
        print(f"\nPlanning 3-D path from {np.round(current_pos, 3)} to {np.round(goal, 3)}...")

        current_path = self.plan_path(current_pos, goal)

        if current_path is not None:
            print(f"✓ Path found: {len(current_path)} waypoints")
            self.visualize_path(current_path)
            print(f"Executing path...")
            self.execute_path(current_path)

            final_pos = self.get_ee_position()
            dist_to_goal = float(np.linalg.norm(final_pos - goal))
            if dist_to_goal < 0.08:
                print(f"✓ Goal reached! (dist={dist_to_goal:.3f} m)")
            else:
                print(f"⚠ Arm stopped {dist_to_goal:.3f} m from goal "
                      f"(EE @ {np.round(final_pos, 3)})")
        else:
            print(f"✗ No path found!")

        print("\nDemo complete!")
        print("Press Ctrl+C or close window to exit.")

        for _ in range(500):
            p.stepSimulation()
            time.sleep(0.01)
    
    def cleanup(self):
        """Clean up PyBullet connection."""
        p.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Franka + Star-Worlds + RRT Demo')
    parser.add_argument('--scenario', type=str, default='simple',
                        choices=['simple', 'moderate', 'corridor'],
                        help='Obstacle scenario to use')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI')
    args = parser.parse_args()
    
    try:
        demo = FrankaStarWorldsDemo(
            use_gui=not args.no_gui,
            scenario=args.scenario
        )
        demo.run_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        if 'demo' in locals():
            demo.cleanup()