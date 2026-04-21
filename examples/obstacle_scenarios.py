"""
Obstacle scenario configurations for PyBullet demos.

Each scenario defines obstacle configurations with:
- position: [x, y, z]
- radius: sphere radius
- color: [r, g, b, a]
- trajectory: function(t) -> np.array([x, y, z]) for dynamic obstacles
"""

import numpy as np


def get_scenario(name='simple'):
    """
    Get obstacle scenario by name.

    Available scenarios:
    - 'test'    : 1 static obstacle, short path — minimal sanity check
    - 'overlap' : 2 heavily overlapping spheres — before/after comparison demo
    - 'simple'  : 3 obstacles, easy
    - 'moderate': 6 obstacles, medium difficulty
    - 'corridor': narrow passage between walls
    """
    scenarios = {
        'test':     test_scenario,
        'overlap':  overlap_scenario,
        'simple':   simple_scenario,
        'moderate': moderate_scenario,
        'corridor': corridor_scenario,
    }

    if name not in scenarios:
        raise ValueError(f"Unknown scenario '{name}'. Available: {list(scenarios.keys())}")

    return scenarios[name]()


def overlap_scenario():
    """
    Two large, heavily overlapping spheres forming an impassable wall.

    Geometry
    --------
    Two spheres at x=0.50, z=0.45, offset ±0.06 m in y (at y=−0.06 and y=+0.06).
    Radius 0.09 m each.  Surfaces overlap (2×0.06=0.12 < 2×0.09=0.18).
    With 5 cm safety margin inflated radii = 0.14 m → always merged by Algorithm 2.

    Approach direction: x-axis (left → right).  Both obstacles are at x=0.50,
    directly across the straight-line path from start (x=0.30) to goal (x=0.70).

    Why the naive controller fails
    ------------------------------
    Normal from obs-below (y=−0.06) pushes in +y; normal from obs-above (y=+0.06)
    pushes in −y.  These y-components cancel exactly at y=0.  Combined modulation
    drives EE straight forward in +x (into the wall).
    (26 penetration steps, only 1.5 cm y-drift — essentially a straight line.)

    Why star-worlds succeeds
    ------------------------
    Algorithm 2 merges both spheres into one proxy.  The 3-D l(x,xg)-avoidance
    shifts the kernel 4 cm in −y off the approach axis so the tangential force
    is non-zero.  EE arcs 15 cm in +y around the merged proxy with 0 penetrations.

    Start : [0.30, 0.0, 0.45]  |base| ≈ 0.54 m  (Γ≈1.26)
    Goal  : [0.70, 0.0, 0.45]  |base| ≈ 0.83 m  (Γ≈1.31)
    """
    return [
        {
            'radius': 0.09,
            'color':  [1.0, 0.35, 0.0, 0.90],   # deep orange
            'position': [0.55, -0.06, 0.45],
            'trajectory': lambda t: np.array([0.55, -0.06, 0.45]),
        },
        {
            'radius': 0.09,
            'color':  [1.0, 0.60, 0.0, 0.90],   # amber
            'position': [0.55,  0.06, 0.45],
            'trajectory': lambda t: np.array([0.55,  0.06, 0.45]),
        },
    ]


def test_scenario():
    """
    Minimal sanity-check: 1 static sphere sits exactly on the straight-line
    path from start to goal.

    Start : [0.38, -0.15, 0.50]
    Goal  : [0.62,  0.15, 0.50]
    Obstacle centre: [0.50, 0.0, 0.50], radius 0.10

    Both start/goal are well within the Franka's comfortable reach envelope
    (~0.5 m from base), so _move_to_start completes quickly.
    The reactive controller must deflect left or right to pass the sphere.
    """
    return [
        {
            'radius': 0.10,
            'color': [1, 0.4, 0, 0.8],   # orange
            'position': [0.50, 0.0, 0.50],
            'trajectory': lambda t: np.array([0.50, 0.0, 0.50])  # static
        },
    ]


def simple_scenario():
    """4 obstacles — easy navigation with a star-world merge demo.

    - Obs 0 (red,   dynamic): oscillates in y — independent proxy
    - Obs 1 (blue,  dynamic): oscillates in x — independent proxy
    - Obs 2 (green, static):  small sphere at [0.58,  0.06, 0.40]  ┐ inflated radii
    - Obs 3 (lime,  static):  small sphere at [0.64, -0.04, 0.40]  ┘ overlap → merged
      center-to-center ≈ 0.11 m  <  2×(0.055+0.05) = 0.21 m  → Algorithm 2 merges
      them into one star-shaped proxy, shown as a single wireframe cage.
    """
    return [
        {
            'radius': 0.08,
            'color': [1, 0, 0, 0.7],
            'position': [0.35, 0.30, 0.4],
            'trajectory': lambda t: np.array([0.35, 0.30 + 0.04 * np.sin(t), 0.4])
        },
        {
            'radius': 0.07,
            'color': [0, 0, 1, 0.7],
            'position': [0.42, -0.30, 0.3],
            'trajectory': lambda t: np.array([0.42 + 0.04 * np.cos(t * 1.5), -0.30, 0.3])
        },
        {
            'radius': 0.055,
            'color': [0.1, 0.9, 0.1, 0.85],   # bright green
            'position': [0.58, 0.06, 0.40],
            'trajectory': lambda t: np.array([0.58, 0.06, 0.40])   # static
        },
        {
            'radius': 0.055,
            'color': [0.6, 1.0, 0.1, 0.85],   # lime / yellow-green
            'position': [0.64, -0.04, 0.40],
            'trajectory': lambda t: np.array([0.64, -0.04, 0.40])  # static
        },
    ]


def moderate_scenario():
    """6 obstacles - moderate difficulty."""
    return [
        # Upper region obstacles
        {
            'radius': 0.09,
            'color': [1, 0, 0, 0.7],
            'position': [0.3, 0.4, 0.4],
            'trajectory': lambda t: np.array([0.3 + 0.05 * np.sin(t * 0.8), 0.4, 0.4])
        },
        {
            'radius': 0.08,
            'color': [1, 0.5, 0, 0.7],
            'position': [0.5, 0.35, 0.4],
            'trajectory': lambda t: np.array([0.5, 0.35 + 0.03 * np.cos(t * 1.2), 0.4])
        },
        # Middle region obstacles
        {
            'radius': 0.07,
            'color': [0, 1, 0, 0.7],
            'position': [0.4, 0.1, 0.35],
            'trajectory': lambda t: np.array([0.4 + 0.04 * np.sin(t), 0.1, 0.35])
        },
        {
            'radius': 0.08,
            'color': [0, 1, 1, 0.7],
            'position': [0.6, 0.0, 0.4],
            'trajectory': lambda t: np.array([0.6, 0.0 + 0.05 * np.sin(t * 1.5), 0.4])
        },
        # Lower region obstacles
        {
            'radius': 0.09,
            'color': [0, 0, 1, 0.7],
            'position': [0.35, -0.35, 0.3],
            'trajectory': lambda t: np.array([0.35, -0.35, 0.3])  # Static
        },
        {
            'radius': 0.07,
            'color': [1, 0, 1, 0.7],
            'position': [0.55, -0.25, 0.35],
            'trajectory': lambda t: np.array([0.55 + 0.03 * np.cos(t * 2), -0.25, 0.35])
        },
    ]


def corridor_scenario():
    """
    Narrow corridor: two side walls (3 spheres each) + one moving blocker.

    Key design constraint: obstacle spacing > 2 × inflated_radius so that
    wall spheres do NOT overlap → each stays its own proxy → no giant merged
    proxy.  Inflated radius = obs_radius + 0.05 = 0.07 + 0.05 = 0.12 m.
    Wall sphere spacing = 0.28 m  >  2 × 0.12 = 0.24 m  →  disjoint. ✓

    Corridor geometry (top view, z omitted):
        Left wall  x = 0.30  y = {-0.28, 0.00, +0.28}
        Right wall x = 0.60  y = {-0.28, 0.00, +0.28}
        Passage width = 0.60 − 0.30 − 2×0.07 = 0.16 m (tight but passable)

    Start : [0.45, -0.52, 0.40]  (below corridor)
    Goal  : [0.45, +0.52, 0.40]  (above corridor)
    """
    obstacles = []

    wall_ys = [-0.28, 0.0, 0.28]

    # Left wall
    for y in wall_ys:
        obstacles.append({
            'radius': 0.07,
            'color': [0.45, 0.45, 0.55, 0.85],
            'position': [0.30, y, 0.40],
            'trajectory': lambda t, y_pos=y: np.array([0.30, y_pos, 0.40]),
        })

    # Right wall
    for y in wall_ys:
        obstacles.append({
            'radius': 0.07,
            'color': [0.55, 0.45, 0.45, 0.85],
            'position': [0.60, y, 0.40],
            'trajectory': lambda t, y_pos=y: np.array([0.60, y_pos, 0.40]),
        })

    # Moving blocker in the passage (oscillates left-right)
    obstacles.append({
        'radius': 0.06,
        'color': [1.0, 0.3, 0.1, 0.90],
        'position': [0.45, 0.0, 0.40],
        'trajectory': lambda t: np.array([0.45 + 0.08 * np.sin(t * 0.7), 0.0, 0.40]),
    })

    return obstacles


# Predefined start/goal positions for each scenario.
#
# All positions are chosen so that ||pos|| < 0.82 m from the robot base,
# which keeps them comfortably inside the Franka Panda's reachable workspace
# (~0.855 m max reach).  Positions that were previously outside this range
# caused the IK to stall at the arm's extension limit, producing the "trailing
# off near the end" symptom.
SCENARIO_WAYPOINTS = {
    'overlap': {
        # Approach along x-axis.  Obstacles at x=0.55, y=±0.06.
        # Start at x=0.30 (|base|≈0.54 m) — clear of proxy.
        # Goal  at x=0.72 (|base|≈0.85 m) — clear of proxy on the far side.
        # Raw: ±y normals cancel → arm drives into wall.
        # SW:  merged proxy + kernel offset → arm arcs around.
        'start': [0.30, 0.0, 0.45],   # |base| ≈ 0.54 m
        'goal':  [0.72, 0.0, 0.45],   # |base| ≈ 0.85 m
    },
    'test': {
        'start': [0.38, -0.15, 0.50],   # |base| ≈ 0.57 m
        'goal':  [0.62,  0.15, 0.50],   # |base| ≈ 0.67 m
    },
    'simple': {
        'start': [0.28, -0.40, 0.40],   # |base| ≈ 0.60 m — clear of all obstacles
        'goal':  [0.60,  0.38, 0.40],   # |base| ≈ 0.77 m — clear of all obstacles
    },
    'moderate': {
        'start': [0.25, -0.40, 0.40],   # |base| ≈ 0.58 m
        'goal':  [0.60,  0.44, 0.40],   # |base| ≈ 0.81 m — clear of all proxies
    },
    'corridor': {
        'start': [0.45, -0.52, 0.40],   # below corridor entrance, |base| ≈ 0.77 m
        'goal':  [0.45,  0.52, 0.40],   # above corridor exit,    |base| ≈ 0.77 m
    },
}
