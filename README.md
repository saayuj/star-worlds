# star-worlds

Implementation of **"Creating Star Worlds: Reshaping the Robot Workspace for Online Motion Planning"** — Dahlin & Karayiannidis, *IEEE Transactions on Robotics* 39(5), 2023. [DOI: 10.1109/TRO.2023.3279029](https://doi.org/10.1109/TRO.2023.3279029)

MEAM 6230 Final Project — University of Pennsylvania, Spring 2026.

---

## What it does

Given a set of (possibly intersecting) obstacles and robot/goal positions, Algorithm 2 produces a **disjoint star world**: a set of proxy obstacles that are strictly starshaped, pairwise disjoint, covering the original obstacles, and excluding the robot and goal — properties required for convergence guarantees of reactive motion planners.

This is integrated with a Franka Panda arm in PyBullet using a **modulation-based reactive controller** (Huber et al. 2019/2022): sphere obstacles are modelled as 3-D `Ellipsoid` objects, a disjoint star world is computed in full 3-D EE space at 4 Hz, and the controller continuously modulates a goal-directed velocity to deflect the EE around proxy obstacles at 40 Hz. The arm tracks the resulting velocity commands via IK position control.

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
│   ├── obstacles.py             # Ellipse, ConvexPolygon2D, Polygon2D, Ellipsoid
│   ├── geometry.py              # 2-D: tangent points, cone polygons, polygon clipping (SAT)
│   ├── geometry_3d.py           # 3-D: convex hull, point-in-hull, tetrahedron kernel
│   ├── admissible_kernel.py     # ad ker(A, X̄)  — Eq. (6)–(9), §IV-A
│   ├── starshaped_hull.py       # SH_kerK(A)    — Eq. (11), Property 4d  (2-D and 3-D)
│   ├── kernel_selection.py      # Algorithm 3   — kernel point selection (2-D and 3-D)
│   ├── star_world.py            # Algorithm 2   — main pipeline (2-D and 3-D)
│   ├── integration.py           # StarWorldUpdater — stateful wrapper, auto-detects 2-D/3-D
│   ├── reactive_controller.py   # Modulation-based reactive controller (Huber et al.)
│   └── rrt_planner.py           # RRT / RRT*    — works in both 2-D and 3-D (kept for reference)
├── examples/
│   ├── 2d_static.py             # Star-world construction visualisation (3 ellipses)
│   ├── 2d_dynamic.py            # Animated moving obstacles (2-D)
│   ├── franka_pybullet_demo.py  # Full 3-D Franka + PyBullet reactive control demo
│   └── obstacle_scenarios.py    # Scenario definitions (test / simple / moderate / corridor)
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
python examples/franka_pybullet_demo.py                       # default: test scenario
python examples/franka_pybullet_demo.py --scenario simple
python examples/franka_pybullet_demo.py --scenario moderate
python examples/franka_pybullet_demo.py --scenario corridor
python examples/franka_pybullet_demo.py --no-gui --scenario simple   # headless
```

Each run:
1. Opens the PyBullet GUI and loads the Franka Panda with sphere obstacles
2. Pauses — **press Enter in the terminal** to start the simulation
3. Moves the arm to the start position via IK
4. Enters the reactive control loop:
   - **40 Hz**: reads current EE position, computes modulated velocity from star-world proxies, advances IK target
   - **4 Hz**: rebuilds the disjoint star world from current obstacle positions (supports dynamic obstacles)
5. Reports goal distance in the terminal every second; stays open until goal is reached, then until you close the window or press Ctrl+C

GUI overlays:
- **Green dot**: kernel center x_{c,i} of each proxy (the interior point the modulation uses)
- **Transparent green sphere**: proxy bounding volume
- **Orange arrow**: current EE velocity direction
- **White trail**: EE path traced so far
- **Green/red dots**: start and goal markers

| Scenario | Obstacles | Notes |
|---|---|---|
| `test` | 1 static | Minimal sanity check — single sphere on direct path |
| `simple` | 3 (2 moving, 1 static) | Easy — 3 separate proxies |
| `moderate` | 6 (5 moving, 1 static) | Medium — some clusters may merge |
| `corridor` | 18 (16 static walls + 2 moving) | Hard — large merged proxy |

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

### 3-D reactive control (Ellipsoid obstacles — Franka / PyBullet)

```python
from starworlds import StarWorldUpdater, Ellipsoid, ReactiveController
import numpy as np

updater    = StarWorldUpdater(n_approx_3d=16)
controller = ReactiveController(v_max=0.30, eta=1.0, safety_radius=0.03)

# Each sphere → Ellipsoid(center, Q=r²·I₃)
obstacles = [Ellipsoid(center=pos, Q=(r**2)*np.eye(3)) for pos, r in sphere_data]

# Rebuild star world (call at ~4 Hz)
result = updater.update(obstacles=obstacles,
                        robot_pos=ee_pos, goal_pos=goal_pos)

# Compute velocity (call at ~40 Hz, using cached result.proxies between rebuilds)
velocity = controller.compute_velocity(x=ee_pos, xg=goal_pos,
                                       proxies=result.proxies)
# velocity is a (3,) vector, ||velocity|| ≤ v_max
```

### 2-D (Ellipse / ConvexPolygon2D obstacles)

```python
from starworlds import StarWorldUpdater, Ellipse

updater = StarWorldUpdater()
result = updater.update(obstacles=[Ellipse(center, a, b), ...],
                        robot_pos=start_2d, goal_pos=goal_2d)
# result.proxies    → list of ObstacleProxy
# result.is_disjoint → True if convergence is guaranteed
```

`StarWorldUpdater` auto-detects dimensionality from `obstacle.dim` and routes to the 2-D or 3-D algorithm automatically.

---

## Reactive controller

`ReactiveController` implements the modulation-based obstacle avoidance from Huber, Billard & Slotine (IEEE RA-L 2019) and Huber, Slotine & Billard (IEEE Trans. Robot. 2022), connected to the star-world proxy shapes.

For each proxy O'_i with kernel center x_{c,i}:

| Quantity | Formula |
|---|---|
| Gamma (normalised distance) | Γ_i = ‖x − x_{c,i}‖ / r_i  (= 1 on boundary, > 1 outside) |
| Outward normal | n_i = (x − x_{c,i}) / ‖…‖ |
| Normal eigenvalue | λ_n = 1 − η/Γ_i  →  0 at boundary (blocks penetration) |
| Tangent eigenvalue | λ_t = 1 + η/Γ_i  →  amplified (slides around obstacle) |
| Modulation matrix | M_i = E_i · diag(λ_n, λ_t, λ_t) · E_iᵀ |
| Combined velocity | ẋ = M(x) f₀ / ‖M(x) f₀‖ · v_max |

The connection to star-worlds: x_{c,i} ∈ int ker(O'_i) is the kernel center guaranteed by Algorithm 2, which is exactly the repulsion center the modulation needs.

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
