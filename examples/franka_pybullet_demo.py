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

from starworlds import StarWorldUpdater, Ellipse
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
        
        # Star-worlds updater (L ~ 10× workspace size)
        self.star_updater = StarWorldUpdater(L=10.0)
        
        # RRT planner configuration
        # Workspace bounds: approximate Franka reachable space (2D X-Y plane)
        workspace_lower = np.array([0.2, -0.6])  # x, y
        workspace_upper = np.array([0.8, 0.6])
        
        self.rrt = RRTPlanner(
            step_size=0.03,           # Smaller step for finer resolution
            goal_bias=0.2,            # Higher bias toward goal
            max_iterations=10000,     # More iterations
            goal_tolerance=0.05,
            bounds=(workspace_lower, workspace_upper)
        )
        
        # Default Z height for path execution
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
    
    def get_ee_position_2d(self):
        """Get current end-effector position projected to 2D (X-Y plane)."""
        pos_3d = self.get_ee_position()
        return pos_3d[:2]  # Return only X, Y
    
    def update_obstacle_positions(self, t):
        """Update dynamic obstacle positions based on time."""
        for obs in self.obstacles:
            new_pos = obs['trajectory'](t)
            p.resetBasePositionAndOrientation(
                obs['id'],
                new_pos.tolist(),
                [0, 0, 0, 1]
            )
    
    def get_starworld_obstacles(self):
        """Extract obstacles for star-world processing (2D projection to X-Y plane)."""
        starworld_obstacles = []
        
        for obs in self.obstacles:
            pos, _ = p.getBasePositionAndOrientation(obs['id'])
            
            # Add safety margin (5cm)
            safety_margin = 0.05
            radius = obs['radius'] + safety_margin
            
            # Project to 2D (X-Y plane) - star-worlds works in 2D
            center_2d = np.array([pos[0], pos[1]])
            
            # Create 2D ellipse obstacle
            starworld_obstacles.append(
                Ellipse(center=center_2d, a=radius, b=radius)
            )
            
        return starworld_obstacles
    
    def plan_path(self, start_3d, goal_3d):
        """
        Plan collision-free path using star-worlds + RRT.
        
        Parameters
        ----------
        start_3d : np.ndarray
            Start position in task space (3,)
        goal_3d : np.ndarray
            Goal position in task space (3,)
            
        Returns
        -------
        path : np.ndarray or None
            Planned path as (N, 3) array, or None if planning fails
        """
        # Project to 2D for star-worlds (X-Y plane)
        start_2d = start_3d[:2]
        goal_2d = goal_3d[:2]
        
        print(f"  Start (2D): {start_2d}")
        print(f"  Goal (2D): {goal_2d}")
        
        # Get current obstacles (already in 2D)
        obstacles = self.get_starworld_obstacles()
        print(f"  Obstacles: {len(obstacles)}")
        for i, obs in enumerate(obstacles):
            print(f"    Obs {i}: center={obs.center}, radius={obs.a:.3f}")
        
        # Reshape obstacles into disjoint star-shaped proxies
        result = self.star_updater.update(
            obstacles=obstacles,
            robot_pos=start_2d,
            goal_pos=goal_2d
        )
        
        print(f"  Star-world result: {len(result.proxies)} proxies, disjoint={result.is_disjoint}")
        obs_centers_2d = [obs.center for obs in obstacles]
        for i, proxy in enumerate(result.proxies):
            xc = proxy.center_point
            # Find the closest original obstacle centre (diagnostic)
            if obs_centers_2d:
                dists = [np.linalg.norm(xc - oc) for oc in obs_centers_2d]
                min_d = min(dists)
                closest = obs_centers_2d[dists.index(min_d)]
                status = "OK" if min_d < 0.5 else "FAR FROM OBSTACLES - may be a bug"
                print(f"    Proxy {i}: center={xc}  (dist {min_d:.3f} from obs @ {closest}) [{status}]")
            else:
                print(f"    Proxy {i}: center={xc}")

        # Visualize proxy obstacles
        self._visualize_proxies(result.proxies, z_height=self.default_z)

        # Plan with RRT in 2D
        print(f"  Running RRT planner...")
        path_2d = self.rrt.plan(
            start=start_2d,
            goal=goal_2d,
            obstacle_proxies=result.proxies
        )

        if path_2d is None:
            print(f"  RRT failed to find path — check proxy coverage and start/goal positions")
            return None

        print(f"  Path found: {len(path_2d)} waypoints, "
              f"length={sum(np.linalg.norm(path_2d[k+1]-path_2d[k]) for k in range(len(path_2d)-1)):.3f} m")
        
        # Convert 2D path back to 3D by interpolating Z between start and goal
        path_3d = self._lift_path_to_3d(path_2d, start_3d[2], goal_3d[2])
        
        return path_3d
    
    def _lift_path_to_3d(self, path_2d, start_z, goal_z):
        """Convert 2D path to 3D by linearly interpolating Z coordinate."""
        n_points = len(path_2d)
        z_values = np.linspace(start_z, goal_z, n_points)
        path_3d = np.column_stack([path_2d, z_values])
        return path_3d
    
    def _move_ee_to_position(self, target_pos, duration=2.0):
        """Move end-effector to target position using IK."""
        num_steps = int(240 * duration)  # 240 Hz
        
        for _ in range(num_steps):
            # IK to get joint positions
            joint_poses = p.calculateInverseKinematics(
                self.franka_id,
                self.ee_link_index,
                target_pos.tolist()
            )
            
            # Set joint positions
            num_joints = p.getNumJoints(self.franka_id)
            for joint_idx in range(min(len(joint_poses), num_joints)):
                p.setJointMotorControl2(
                    self.franka_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    joint_poses[joint_idx]
                )
            
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    
    def _visualize_proxies(self, proxies, z_height=0.4):
        """Visualize star-shaped proxy obstacles (2D proxies shown at fixed Z height)."""
        # Remove old visualizations
        for vid in self.proxy_visual_ids:
            p.removeBody(vid)
        self.proxy_visual_ids.clear()
        
        # Draw new proxies
        for proxy in proxies:
            # Convert 2D center to 3D
            center_3d = [proxy.center_point[0], proxy.center_point[1], z_height]
            
            # Draw center point
            center_visual = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.02,
                rgbaColor=[0, 1, 0, 1]
            )
            center_body = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=center_visual,
                basePosition=center_3d
            )
            self.proxy_visual_ids.append(center_body)
            
            # Draw bounding sphere (approximate visualization)
            if hasattr(proxy, 'pieces') and len(proxy.pieces) > 0:
                # Compute approximate radius from pieces
                max_dist = 0
                for piece in proxy.pieces:
                    if len(piece) > 0:
                        dists = np.linalg.norm(piece - proxy.center_point, axis=1)
                        max_dist = max(max_dist, np.max(dists))
                
                if max_dist > 0:
                    boundary_visual = p.createVisualShape(
                        p.GEOM_SPHERE,
                        radius=max_dist,
                        rgbaColor=[0, 1, 0, 0.2]
                    )
                    boundary_body = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=boundary_visual,
                        basePosition=center_3d
                    )
                    self.proxy_visual_ids.append(boundary_body)
    
    def visualize_path(self, path):
        """Draw path in PyBullet."""
        # Remove old path lines
        for line_id in self.path_line_ids:
            p.removeUserDebugItem(line_id)
        self.path_line_ids.clear()
        
        if path is None or len(path) < 2:
            return
            
        # Draw line segments
        for i in range(len(path) - 1):
            line_id = p.addUserDebugLine(
                path[i].tolist(),
                path[i + 1].tolist(),
                lineColorRGB=[1, 1, 0],
                lineWidth=3
            )
            self.path_line_ids.append(line_id)
    
    def execute_path(self, path, duration=5.0):
        """
        Execute path using simple end-effector position control.
        
        Note: This is a simplified version. For production, use proper
        inverse kinematics and joint-space trajectory tracking.
        """
        if path is None:
            print("No path to execute!")
            return
            
        print(f"Executing path with {len(path)} waypoints...")
        
        num_steps = len(path)
        time_per_step = duration / num_steps
        
        for i, waypoint in enumerate(path):
            # Simple IK: use PyBullet's built-in IK solver
            joint_poses = p.calculateInverseKinematics(
                self.franka_id,
                self.ee_link_index,
                waypoint.tolist()
            )
            
            # Set joint positions
            num_joints = p.getNumJoints(self.franka_id)
            for joint_idx in range(min(len(joint_poses), num_joints)):
                p.setJointMotorControl2(
                    self.franka_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    joint_poses[joint_idx]
                )
            
            # Step simulation
            for _ in range(int(240 * time_per_step)):  # 240 Hz
                p.stepSimulation()
                time.sleep(1.0 / 240.0)
    
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
        current_pos = self.get_ee_position()
        
        print(f"\nPlanning path from {start[:2]} to {goal[:2]}...")
        
        # Update obstacles
        self.update_obstacle_positions(t)
        
        current_path = self.plan_path(current_pos, goal)
        
        if current_path is not None:
            print(f"✓ Path found with {len(current_path)} waypoints")
            self.visualize_path(current_path)
            
            # Execute path
            print(f"Executing path...")
            self.execute_path(current_path, duration=3.0)
            print(f"✓ Path execution complete!")
        else:
            print(f"✗ No path found!")
        
        print("\nDemo complete!")
        print("Press Ctrl+C or close window to exit.")
        
        # Keep window open for a bit longer
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