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

## Integration with the motion planner / simulator

```python
from starworlds import StarWorldUpdater, Ellipse, ConvexPolygon2D
import numpy as np

updater = StarWorldUpdater(L=1e4)   # L should be ~10× your workspace radius

# --- call this at every simulator timestep ---
obstacles = [
    Ellipse(center=obs_pos, a=radius + safety_margin, b=radius + safety_margin)
    for obs_pos in current_obstacle_positions
]

result = updater.update(
    obstacles = obstacles,
    robot_pos = robot_workspace_pos,   # shape (2,) for 2D, (3,) for 3D
    goal_pos  = goal_workspace_pos,
)

# result.proxies[i].center_point  → x_{c,i} for the motion planner
# result.proxies[i].pieces        → list of convex polygon vertex arrays
# result.is_disjoint              → True if convergence is guaranteed
```

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
