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
    - 'simple'  : 3 obstacles, easy
    - 'moderate': 6 obstacles, medium difficulty
    - 'corridor': narrow passage between walls
    """
    scenarios = {
        'test':     test_scenario,
        'simple':   simple_scenario,
        'moderate': moderate_scenario,
        'corridor': corridor_scenario,
    }

    if name not in scenarios:
        raise ValueError(f"Unknown scenario '{name}'. Available: {list(scenarios.keys())}")

    return scenarios[name]()


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
    """3 obstacles - easy navigation.

    Path from start [0.22, -0.45] to goal [0.75, 0.45]:
      - Obs 0 (red, moving): upper-left region, well clear of goal
      - Obs 1 (blue, moving): lower region, well clear of start
      - Obs 2 (green, static): mid-right region, forces path to curve around it
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
            'radius': 0.08,
            'color': [0, 1, 0, 0.7],
            'position': [0.60, 0.05, 0.4],
            'trajectory': lambda t: np.array([0.60, 0.05, 0.4])  # Static
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
    """Narrow corridor with walls on sides."""
    obstacles = []
    
    # Left wall (series of obstacles)
    for y in np.linspace(-0.5, 0.5, 8):
        obstacles.append({
            'radius': 0.08,
            'color': [0.5, 0.5, 0.5, 0.7],
            'position': [0.25, y, 0.4],
            'trajectory': lambda t, y_pos=y: np.array([0.25, y_pos, 0.4])
        })
    
    # Right wall (series of obstacles)
    for y in np.linspace(-0.5, 0.5, 8):
        obstacles.append({
            'radius': 0.08,
            'color': [0.6, 0.6, 0.6, 0.7],
            'position': [0.65, y, 0.4],
            'trajectory': lambda t, y_pos=y: np.array([0.65, y_pos, 0.4])
        })
    
    # Moving obstacles in corridor
    obstacles.extend([
        {
            'radius': 0.07,
            'color': [1, 0, 0, 0.7],
            'position': [0.45, 0.3, 0.4],
            'trajectory': lambda t: np.array([0.45, 0.3 + 0.15 * np.sin(t * 0.8), 0.4])
        },
        {
            'radius': 0.07,
            'color': [0, 0, 1, 0.7],
            'position': [0.45, -0.2, 0.4],
            'trajectory': lambda t: np.array([0.45, -0.2 + 0.15 * np.cos(t * 0.6), 0.4])
        },
    ])
    
    return obstacles


# Predefined start/goal positions for each scenario.
#
# All positions are chosen so that ||pos|| < 0.82 m from the robot base,
# which keeps them comfortably inside the Franka Panda's reachable workspace
# (~0.855 m max reach).  Positions that were previously outside this range
# caused the IK to stall at the arm's extension limit, producing the "trailing
# off near the end" symptom.
SCENARIO_WAYPOINTS = {
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
        'start': [0.45, -0.55, 0.40],   # |base| ≈ 0.78 m — outside corridor
        'goal':  [0.45,  0.55, 0.40],   # |base| ≈ 0.78 m — outside corridor exit
    },
}
