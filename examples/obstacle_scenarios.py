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
    - 'simple': 3 obstacles, easy
    - 'moderate': 6 obstacles, medium difficulty
    - 'corridor': narrow passage between walls
    """
    scenarios = {
        'simple': simple_scenario,
        'moderate': moderate_scenario,
        'corridor': corridor_scenario,
    }
    
    if name not in scenarios:
        raise ValueError(f"Unknown scenario '{name}'. Available: {list(scenarios.keys())}")
    
    return scenarios[name]()


def simple_scenario():
    """3 obstacles - easy navigation."""
    return [
        {
            'radius': 0.10,
            'color': [1, 0, 0, 0.7],
            'position': [0.35, 0.45, 0.4],
            'trajectory': lambda t: np.array([0.35, 0.45 + 0.05 * np.sin(t), 0.4])
        },
        {
            'radius': 0.08,
            'color': [0, 0, 1, 0.7],
            'position': [0.35, -0.45, 0.3],
            'trajectory': lambda t: np.array([0.35 + 0.05 * np.cos(t * 1.5), -0.45, 0.3])
        },
        {
            'radius': 0.09,
            'color': [0, 1, 0, 0.7],
            'position': [0.7, 0.1, 0.4],
            'trajectory': lambda t: np.array([0.7, 0.1, 0.4])  # Static
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


# Predefined start/goal positions for each scenario
SCENARIO_WAYPOINTS = {
    'simple': {
        'start': [0.22, -0.45, 0.4],  # Further bottom-left
        'goal': [0.75, 0.45, 0.4]     # Further top-right
    },
    'moderate': {
        'start': [0.22, -0.48, 0.4],  # Bottom-left, clear of obstacles
        'goal': [0.75, 0.48, 0.4]     # Top-right, clear of obstacles (not too far)
    },
    'corridor': {
        'start': [0.45, -0.65, 0.4],  # Outside corridor, clear of walls
        'goal': [0.45, 0.65, 0.4]     # Outside corridor exit, clear of walls
    },
}
