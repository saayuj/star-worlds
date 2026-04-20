# star-worlds

Implementation of **"Creating Star Worlds: Reshaping the Robot Workspace for Online Motion Planning"** — Dahlin & Karayiannidis, *IEEE Transactions on Robotics* 39(5), 2023. [DOI: 10.1109/TRO.2023.3279029](https://doi.org/10.1109/TRO.2023.3279029)

MEAM 6230 Final Project — University of Pennsylvania, Spring 2026.

---

## What it does

Given a set of (possibly intersecting) obstacles and robot/goal positions, Algorithm 2 produces a **disjoint star world**: a set of proxy obstacles that are strictly starshaped, pairwise disjoint, covering of the original obstacles, and excluding of the robot and goal — properties required for convergence guarantees of harmonic/contraction-based motion planners.

This is integrated with a Franka Panda arm in PyBullet: sphere obstacles are modelled as 3-D `Ellipsoid` objects, a disjoint star world is computed in full 3-D EE space, RRT plans a collision-free 3-D path through the proxy obstacles, and the arm executes it via IK with arrival verification.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Requires Python ≥ 3.9.

---

## Directory structure

```
star-worlds/
├── starworlds/
│   ├── obstacles.py          # Ellipse, ConvexPolygon2D, Polygon2D, Ellipsoid
│   ├── geometry.py           # 2-D: tangent points, cone polygons, polygon clipping (SAT)
│   ├── geometry_3d.py        # 3-D: convex hull, point-in-hull, tetrahedron kernel
│   ├── admissible_kernel.py  # ad ker(A, X̄)  — Eq. (6)–(9), §IV-A
│   ├── starshaped_hull.py    # SH_kerK(A)    — Eq. (11), Property 4d  (2-D and 3-D)
│   ├── kernel_selection.py   # Algorithm 3   — kernel point selection (2-D and 3-D)
│   ├── star_world.py         # Algorithm 2   — main pipeline (2-D and 3-D)
│   ├── integration.py        # StarWorldUpdater — stateful wrapper, auto-detects 2-D/3-D
│   └── rrt_planner.py        # RRT / RRT*    — works in both 2-D and 3-D
├── examples/
│   ├── 2d_static.py          # Star-world construction visualisation (3 ellipses)
│   ├── 2d_dynamic.py         # Animated moving obstacles (2-D)
│   ├── franka_pybullet_demo.py  # Full 3-D Franka + PyBullet demo
│   └── obstacle_scenarios.py    # Scenario definitions (simple / moderate / corridor)
├── tests/
│   ├── test_geometry.py
│   ├── test_admissible_kernel.py
│   ├── test_hull.py
│   └── test_star_world.py
└── requirements.txt
```

---

## Running the demos

### Franka arm — 3-D PyBullet demo

```bash
python examples/franka_pybullet_demo.py --scenario simple
python examples/franka_pybullet_demo.py --scenario moderate
python examples/franka_pybullet_demo.py --scenario corridor
python examples/franka_pybullet_demo.py --no-gui --scenario simple   # headless
```

Each run:
1. Loads the Franka Panda in PyBullet with sphere obstacles
2. Computes a 3-D disjoint star world (one `Ellipsoid` per sphere)
3. Plans a collision-free 3-D path with RRT
4. Executes it via IK with per-waypoint arrival verification
5. Reports whether the goal was reached and stays open until you close it or press Ctrl+C

| Scenario | Obstacles | Expected |
|---|---|---|
| `simple` | 3 | 3 disjoint proxies, goal reached |
| `moderate` | 6 | 4 proxies (some clusters merged), goal reached |
| `corridor` | 18 | 1 merged proxy, path threads the corridor gap |

### 2-D visualisation examples

```bash
python examples/2d_static.py          # before/after star-world construction
python examples/2d_dynamic.py         # animated moving obstacles
python examples/2d_dynamic.py --save  # save as dynamic_result.gif
```

---

## Tests

```bash
pytest tests/ -v        # 51 tests, ~2 s
```

---

## API quick-start

### 3-D (Ellipsoid obstacles — Franka / PyBullet)

```python
from starworlds import StarWorldUpdater, Ellipsoid
from starworlds.rrt_planner import RRTPlanner
import numpy as np

updater = StarWorldUpdater(n_approx_3d=64)

# Each sphere → Ellipsoid(center, Q=r²·I₃)
obstacles = [Ellipsoid(center=pos, Q=(r**2)*np.eye(3)) for pos, r in sphere_data]

result = updater.update(obstacles=obstacles,
                        robot_pos=start_3d, goal_pos=goal_3d)

rrt = RRTPlanner(step_size=0.04, goal_bias=0.2, max_iterations=15000,
                 goal_tolerance=0.05,
                 bounds=(np.array([0.15,-0.7,0.15]), np.array([0.9,0.7,0.85])))

path = rrt.plan(start=start_3d, goal=goal_3d, obstacle_proxies=result.proxies)
```

### 2-D (Ellipse / ConvexPolygon2D obstacles)

```python
from starworlds import StarWorldUpdater, Ellipse

updater = StarWorldUpdater()
result = updater.update(obstacles=[Ellipse(center, a, b), ...],
                        robot_pos=start_2d, goal_pos=goal_2d)
# result.proxies → list of ObstacleProxy
# result.is_disjoint → True if convergence is guaranteed
```

`StarWorldUpdater` auto-detects dimensionality from `obstacle.dim` and routes to the 2-D or 3-D algorithm automatically.

---

## Algorithm overview

| Step | Description | Paper ref |
|------|-------------|-----------|
| Admissible kernel | Cone polygon (2-D) or cluster hull (3-D) per obstacle | §IV-A, Eq. (8) |
| Kernel selection | Equilateral triangle K (2-D) or tetrahedron K (3-D) at cluster centroid | Algorithm 3, §V-C |
| Starshaped hull | `SH_kerK(cluster) = CH(obstacle_surface ∪ K)` | Eq. (11), §IV-B |
| Re-clustering | Union-find merge of intersecting proxy pairs | Algorithm 2, §V-A.3 |
| Fallback | Convex decomposition if AK is empty | Eq. (14) |

Typical convergence: **1–2 iterations** for well-separated obstacles.
