"""
RRT (Rapidly-exploring Random Tree) motion planner for star-shaped workspaces.

Integrates with star-worlds to plan collision-free paths around disjoint
star-shaped obstacle proxies.
"""

import numpy as np
from scipy.spatial import ConvexHull
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RRTNode:
    """Node in the RRT tree."""
    position: np.ndarray  # (2,) or (3,) array
    parent: Optional['RRTNode'] = None
    cost: float = 0.0


class RRTPlanner:
    """
    RRT planner for navigation in star-world obstacle environments.
    
    Parameters
    ----------
    step_size : float
        Maximum distance to extend the tree in each iteration
    goal_bias : float
        Probability [0, 1] of sampling the goal instead of random point
    max_iterations : int
        Maximum number of tree expansion attempts
    goal_tolerance : float
        Distance within which a node is considered to reach the goal
    bounds : tuple of array-like
        Workspace bounds as (lower, upper) where each is shape (2,) or (3,)
    """
    
    def __init__(
        self,
        step_size: float = 0.1,
        goal_bias: float = 0.1,
        max_iterations: int = 5000,
        goal_tolerance: float = 0.05,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ):
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.max_iterations = max_iterations
        self.goal_tolerance = goal_tolerance
        self.bounds = bounds
        
    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacle_proxies: List = None
    ) -> Optional[np.ndarray]:
        """
        Plan a path from start to goal avoiding obstacle proxies.
        
        Parameters
        ----------
        start : np.ndarray
            Start position, shape (2,) or (3,)
        goal : np.ndarray
            Goal position, shape (2,) or (3,)
        obstacle_proxies : list, optional
            List of star-world proxy obstacles from StarWorldUpdater.
            Each proxy should have .pieces (list of convex polygon vertices)
            and .center_point attributes.
            
        Returns
        -------
        path : np.ndarray or None
            Path from start to goal as (N, 2) or (N, 3) array.
            Returns None if no path found within max_iterations.
        """
        start = np.asarray(start, dtype=float)
        goal  = np.asarray(goal,  dtype=float)

        if obstacle_proxies is None:
            obstacle_proxies = []

        # Pre-compute hull equations for 3-D proxies (O(n_faces) containment)
        self._proxy_equations: Dict[int, List] = {}
        for pid, proxy in enumerate(obstacle_proxies):
            self._proxy_equations[pid] = self._precompute_proxy_equations(proxy)

        # Initialize tree with start node
        tree = [RRTNode(position=start.copy())]

        # Collision check for start / goal
        start_collision = any(
            self._point_in_proxy(start, proxy, pid)
            for pid, proxy in enumerate(obstacle_proxies)
        )
        goal_collision = any(
            self._point_in_proxy(goal, proxy, pid)
            for pid, proxy in enumerate(obstacle_proxies)
        )
        if start_collision:
            print(f"    WARNING: Start position is in collision!")
        if goal_collision:
            print(f"    WARNING: Goal position is in collision!")
        
        for iteration in range(self.max_iterations):
            # Sample random point (with goal bias)
            if np.random.random() < self.goal_bias:
                random_point = goal
            else:
                random_point = self._sample_random_point(start.shape[0])
                
            # Find nearest node in tree
            nearest_node = self._get_nearest_node(tree, random_point)
            
            # Steer toward random point
            new_position = self._steer(nearest_node.position, random_point)
            
            # Check collision along path
            if not self._is_collision_free(nearest_node.position, new_position, obstacle_proxies):
                continue
                
            # Add new node to tree
            new_cost = nearest_node.cost + np.linalg.norm(new_position - nearest_node.position)
            new_node = RRTNode(position=new_position, parent=nearest_node, cost=new_cost)
            tree.append(new_node)
            
            # Check if goal reached
            if np.linalg.norm(new_position - goal) < self.goal_tolerance:
                # Construct path
                path = self._extract_path(new_node)
                print(f"    RRT succeeded after {iteration+1} iterations, tree size: {len(tree)}")
                return path
        
        # No path found
        print(f"    RRT failed after {self.max_iterations} iterations, tree size: {len(tree)}")
        return None
    
    def _sample_random_point(self, dim: int) -> np.ndarray:
        """Sample a random point in the workspace."""
        if self.bounds is not None:
            lower, upper = self.bounds
            return np.random.uniform(lower, upper, size=dim)
        else:
            # Default: sample in [-1, 1]^dim
            return np.random.uniform(-1, 1, size=dim)
    
    def _get_nearest_node(self, tree: List[RRTNode], point: np.ndarray) -> RRTNode:
        """Find the nearest node in the tree to the given point."""
        distances = [np.linalg.norm(node.position - point) for node in tree]
        nearest_idx = np.argmin(distances)
        return tree[nearest_idx]
    
    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """
        Steer from from_pos toward to_pos, limited by step_size.
        
        Returns new position at most step_size away from from_pos.
        """
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return to_pos.copy()
        else:
            return from_pos + (direction / distance) * self.step_size
    
    def _is_collision_free(
        self,
        from_pos: np.ndarray,
        to_pos: np.ndarray,
        obstacle_proxies: List
    ) -> bool:
        """
        Check if the line segment from from_pos to to_pos is collision-free.
        
        Uses simple point-in-polygon test at multiple waypoints along the segment.
        """
        # Sample points along the segment
        num_checks = int(np.ceil(np.linalg.norm(to_pos - from_pos) / (self.step_size * 0.1))) + 1
        alphas = np.linspace(0, 1, num_checks)
        
        for alpha in alphas:
            point = from_pos + alpha * (to_pos - from_pos)
            for pid, proxy in enumerate(obstacle_proxies):
                if self._point_in_proxy(point, proxy, pid):
                    return False
        return True

    # ---------------------------------------------------------------------- #
    # Proxy containment helpers                                                #
    # ---------------------------------------------------------------------- #

    def _precompute_proxy_equations(self, proxy) -> List:
        """
        Pre-compute scipy ConvexHull.equations for each 3-D piece so that
        containment tests inside the RRT loop are O(n_faces).
        Returns a list with one entry per piece: equations array or None.
        2-D pieces (shape (n, 2)) get None — they use the ray-cast test.
        """
        eqs = []
        if not hasattr(proxy, 'pieces'):
            return eqs
        for piece in proxy.pieces:
            piece = np.asarray(piece, dtype=float)
            if piece.ndim == 2 and piece.shape[1] == 3 and len(piece) >= 4:
                try:
                    hull = ConvexHull(piece)
                    eqs.append(hull.equations)
                except Exception:
                    eqs.append(None)
            else:
                eqs.append(None)
        return eqs

    def _point_in_proxy(self, point: np.ndarray, proxy, proxy_id: int = -1) -> bool:
        """
        Check if *point* is inside any convex piece of the obstacle proxy.

        2-D pieces (n, 2) — ray-casting algorithm.
        3-D pieces (n, 3) — pre-computed half-space equations (exact test).
        """
        if not hasattr(proxy, 'pieces'):
            return False

        eqs_list = self._proxy_equations.get(proxy_id, [None] * len(proxy.pieces))

        for piece, eq in zip(proxy.pieces, eqs_list):
            piece = np.asarray(piece, dtype=float)
            if piece.ndim != 2:
                continue
            if piece.shape[1] == 2:
                if self._point_in_polygon_2d(point[:2], piece):
                    return True
            elif piece.shape[1] == 3:
                if self._point_in_convex_hull_3d(point, piece, eq):
                    return True
        return False

    def _point_in_polygon_2d(self, point: np.ndarray, vertices: np.ndarray) -> bool:
        """Ray-casting point-in-polygon test (2-D)."""
        x, y = point
        n = len(vertices)
        inside = False
        p1x, p1y = vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _point_in_convex_hull_3d(
        self,
        point: np.ndarray,
        vertices: np.ndarray,
        equations: Optional[np.ndarray] = None,
        tol: float = 1e-9,
    ) -> bool:
        """
        Exact 3-D point-in-convex-hull test via half-space representation.

        Uses pre-computed *equations* when available (O(n_faces)).
        Falls back to recomputing the hull from *vertices* otherwise.
        """
        p = np.asarray(point, dtype=float)
        if equations is not None:
            return bool(np.all(equations[:, :3] @ p + equations[:, 3] <= tol))
        pts = np.asarray(vertices, dtype=float)
        if len(pts) < 4:
            return False
        try:
            hull = ConvexHull(pts)
            return bool(np.all(hull.equations[:, :3] @ p + hull.equations[:, 3] <= tol))
        except Exception:
            return False
    
    def _extract_path(self, goal_node: RRTNode) -> np.ndarray:
        """
        Extract path from start to goal by backtracking through tree.
        
        Returns path in forward order (start -> goal).
        """
        path = []
        current = goal_node
        
        while current is not None:
            path.append(current.position)
            current = current.parent
            
        # Reverse to get start -> goal order
        path = path[::-1]
        return np.array(path)


class RRTStarPlanner(RRTPlanner):
    """
    RRT* planner with path optimization via rewiring.
    
    Asymptotically optimal variant of RRT that improves path quality
    by rewiring the tree to minimize cost.
    
    Additional Parameters
    ---------------------
    rewire_radius : float
        Radius within which to search for rewiring opportunities
    """
    
    def __init__(
        self,
        step_size: float = 0.1,
        goal_bias: float = 0.1,
        max_iterations: int = 5000,
        goal_tolerance: float = 0.05,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        rewire_radius: float = 0.3
    ):
        super().__init__(step_size, goal_bias, max_iterations, goal_tolerance, bounds)
        self.rewire_radius = rewire_radius
        
    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacle_proxies: List = None
    ) -> Optional[np.ndarray]:
        """
        Plan an optimized path using RRT*.
        
        See RRTPlanner.plan() for parameter documentation.
        """
        start = np.asarray(start, dtype=float)
        goal  = np.asarray(goal,  dtype=float)

        if obstacle_proxies is None:
            obstacle_proxies = []

        # Pre-compute hull equations for 3-D proxies
        self._proxy_equations = {}
        for pid, proxy in enumerate(obstacle_proxies):
            self._proxy_equations[pid] = self._precompute_proxy_equations(proxy)

        tree = [RRTNode(position=start.copy())]
        best_goal_node = None
        
        for iteration in range(self.max_iterations):
            # Sample
            if np.random.random() < self.goal_bias:
                random_point = goal
            else:
                random_point = self._sample_random_point(start.shape[0])
                
            # Nearest neighbor
            nearest_node = self._get_nearest_node(tree, random_point)
            
            # Steer
            new_position = self._steer(nearest_node.position, random_point)
            
            # Collision check
            if not self._is_collision_free(nearest_node.position, new_position, obstacle_proxies):
                continue
                
            # Find near nodes for rewiring
            near_nodes = self._get_near_nodes(tree, new_position)
            
            # Choose best parent (minimize cost)
            best_parent = nearest_node
            best_cost = nearest_node.cost + np.linalg.norm(new_position - nearest_node.position)
            
            for near_node in near_nodes:
                potential_cost = near_node.cost + np.linalg.norm(new_position - near_node.position)
                if potential_cost < best_cost:
                    if self._is_collision_free(near_node.position, new_position, obstacle_proxies):
                        best_parent = near_node
                        best_cost = potential_cost
                        
            # Add new node
            new_node = RRTNode(position=new_position, parent=best_parent, cost=best_cost)
            tree.append(new_node)
            
            # Rewire tree
            for near_node in near_nodes:
                potential_cost = best_cost + np.linalg.norm(near_node.position - new_position)
                if potential_cost < near_node.cost:
                    if self._is_collision_free(new_position, near_node.position, obstacle_proxies):
                        near_node.parent = new_node
                        near_node.cost = potential_cost
                        
            # Check goal
            if np.linalg.norm(new_position - goal) < self.goal_tolerance:
                if best_goal_node is None or best_cost < best_goal_node.cost:
                    best_goal_node = new_node
                    
        if best_goal_node is not None:
            return self._extract_path(best_goal_node)
        return None
    
    def _get_near_nodes(self, tree: List[RRTNode], point: np.ndarray) -> List[RRTNode]:
        """Find all nodes within rewire_radius of point."""
        near_nodes = []
        for node in tree:
            if np.linalg.norm(node.position - point) < self.rewire_radius:
                near_nodes.append(node)
        return near_nodes
