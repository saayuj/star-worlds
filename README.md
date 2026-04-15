# star-worlds

Implementation of **"Creating Star Worlds: Reshaping the Robot Workspace for Online Motion Planning"** — Dahlin & Karayiannidis, *IEEE Transactions on Robotics* 39(5), 2023. [DOI: 10.1109/TRO.2023.3279029](https://doi.org/10.1109/TRO.2023.3279029)

MEAM 6230 Final Project — University of Pennsylvania, Spring 2026.

---

## What this does

Given a set of **possibly intersecting** obstacle regions and a robot/goal position, Algorithm 2 outputs a **disjoint star world**: a collection of proxy obstacles that are

- **strictly starshaped** — required for convergence guarantees of harmonic/contraction-based planners  
- **pairwise disjoint** — required for (almost) global convergence proofs  
- **covering** the original obstacles — collision-free in the original scene  
- **excluding** the robot and goal positions — motion planning remains feasible  

The output is consumed by a motion planner (e.g. Huber et al. [20, 21]) that operates in star worlds. Our teammate connects this to the Franka arm simulation (PyBullet / Gazebo).

---

## Repository layout

```
star-worlds/
├── starworlds/
│   ├── obstacles.py          # Ellipse, ConvexPolygon2D, Polygon2D, Ellipsoid
│   ├── geometry.py           # Tangent points, cones, polygon clipping, SAT
│   ├── admissible_kernel.py  # §IV-A — ad ker(A, X̄), Eq. (6)–(9)
│   ├── starshaped_hull.py    # §IV-B — SH_kerK(A), Eq. (11), Alg. 1, Property 4d
│   ├── kernel_selection.py   # Algorithm 3 — kernel point selection (§V-C)
│   ├── star_world.py         # Algorithm 2 — main pipeline (§V)
│   └── integration.py        # StarWorldUpdater — stateful interface for simulators
├── tests/
│   ├── test_geometry.py
│   ├── test_admissible_kernel.py
│   ├── test_hull.py
│   └── test_star_world.py
├── examples/
│   ├── visualize.py          # shared plot helpers
│   ├── 2d_static.py          # Fig. 17 reproduction: three intersecting ellipses
│   └── 2d_dynamic.py         # Fig. 18 reproduction: moving obstacles (animated)
└── requirements.txt
```

---

## Setup

Use a virtual environment to keep dependencies isolated:

```bash
python3 -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

Requires Python ≥ 3.9, numpy ≥ 1.24, scipy ≥ 1.10, shapely ≥ 2.0, matplotlib ≥ 3.7.

> **Note:** if you installed without a venv and pytest isn't found on `$PATH`, run it via  
> `python3 -m pytest tests/ -v`

---

## Running the examples

### Static: three intersecting ellipses (reproduces Fig. 17)

```bash
python examples/2d_static.py
```

Shows a before/after pair: the original intersecting ellipses (which create spurious local minima for the motion planner) and the single disjoint starshaped proxy produced by Algorithm 2. Saves `examples/static_result.png`.

### Dynamic: moving obstacles (reproduces Fig. 18 qualitatively)

```bash
python examples/2d_dynamic.py          # interactive animation
python examples/2d_dynamic.py --save   # save as examples/dynamic_result.gif
```

Two circular obstacles (modelling humans) move sinusoidally around a static wall. Algorithm 2 is called at every timestep via `StarWorldUpdater`, which maintains temporal coherence by reusing kernel centroids across frames.

---

## Running the tests

```bash
pytest tests/ -v
```

Test coverage:
- `test_geometry.py` — tangent points, cone construction, Sutherland-Hodgman, SAT
- `test_admissible_kernel.py` — core invariant (no excluded point in hull from AK kernel)
- `test_hull.py` — coverage, kernel inclusion, exclusion guarantee (Prop. 5)
- `test_star_world.py` — Algorithm 2 end-to-end: disjointness, coverage, robot/goal exclusion

---

## Integration with PyBullet and RRT motion planning

### Quick start

Run a demo scenario with the Franka Panda arm:

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Run a scenario (simple, moderate, or corridor)
python examples/franka_pybullet_demo.py --scenario simple
python examples/franka_pybullet_demo.py --scenario moderate
python examples/franka_pybullet_demo.py --scenario corridor
```

This launches a PyBullet simulation that:
1. Creates obstacles in the workspace
2. Reshapes them into disjoint star-shaped proxies using Algorithm 2
3. Plans a collision-free path using RRT
4. Executes the path on the Franka Panda arm

### How the integration works

The integration follows a **2D planning + 3D execution** approach:

1. **Obstacle modeling** — Obstacles are represented as ellipses in the X-Y plane (star-worlds is 2D-only). Each obstacle includes a safety margin for collision avoidance.

2. **Star-world reshaping** — `StarWorldUpdater` reshapes overlapping obstacles into disjoint star-shaped proxies that preserve collision-free space while enabling convergence guarantees for motion planners.

3. **2D path planning** — RRT searches for a path in the 2D workspace (X-Y projection) using the proxy obstacles for collision checking. This produces waypoints `[(x₁, y₁), (x₂, y₂), ...]`.

4. **3D trajectory execution** — The 2D waypoints are lifted to 3D by interpolating Z-heights between start and goal. Inverse kinematics maps each 3D waypoint to joint angles for the Franka arm.

### Basic API usage

```python
from starworlds import StarWorldUpdater, Ellipse
from starworlds.rrt_planner import RRTPlanner

# Initialize star-world updater (L should be ~10× workspace radius)
updater = StarWorldUpdater(L=10.0)

# Create obstacles with safety margin
obstacles = [
    Ellipse(center=[x, y], a=radius + 0.05, b=radius + 0.05)
    for x, y, radius in obstacle_data
]

# Reshape obstacles into star-world proxies
result = updater.update(
    obstacles=obstacles,
    robot_pos=start_pos[:2],  # 2D position [x, y]
    goal_pos=goal_pos[:2]
)

# Plan path using RRT with proxy obstacles
planner = RRTPlanner(
    workspace_bounds=[[0.2, -0.6], [0.8, 0.6]],
    step_size=0.03,
    goal_bias=0.2,
    max_iterations=10000
)

path_2d = planner.plan(
    start=start_pos[:2],
    goal=goal_pos[:2],
    proxy_obstacles=result.proxies
)

# Lift to 3D and execute (PyBullet IK + joint control)
# See examples/franka_pybullet_demo.py for full implementation
```

**Key points:**
- `result.proxies` contains the disjoint star-shaped obstacles for collision checking
- `result.is_disjoint` indicates whether convergence is guaranteed
- The 2D planner uses `_point_in_polygon_2d()` to check collisions against proxy polygons
- Scenarios are defined in [examples/obstacle_scenarios.py](examples/obstacle_scenarios.py) for easy configuration

Non-convex obstacles can be modelled as a **union of convex pieces** (e.g. an L-shaped wall as two `ConvexPolygon2D`). Non-convex `Polygon2D` objects are internally decomposed via triangle fan.

---

## Algorithm summary

| Step | What | Paper ref |
|------|------|-----------|
| Precompute | `ad ker(O_i, {x, x_g})` as cone polygons for each obstacle | Eq. (8), §IV-A |
| Loop | For each cluster: intersect per-obstacle cones → cluster AK | Eq. (7), Property 2-3 |
| | Select 3 affinely independent kernel points K ⊂ AK | Algorithm 3, §V-C |
| | Build proxy O'_i = SH_kerK(cluster) via Eq. (11) + Property 4d | §IV-B, §V-D |
| | Re-cluster by proxy intersection (union-find) | Algorithm 2, §V-A.3 |
| Exit | If AK empty: return convex decomposition (intersecting star world) | Eq. (14), §V-A.1 |

Typical convergence: **2 iterations** (verified empirically in §V-D, 972/1000 random scenes).
