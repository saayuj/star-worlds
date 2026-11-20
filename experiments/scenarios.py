"""
Benchmark scenarios for the revised evaluation.

Every scenario returns a dict with
  obstacles : list of {'shape': ConvexShape3D (true, un-inflated geometry,
              centred at its nominal position), 'traj': f(t) -> centre,
              'color': rgba}
  start, goal : nominal EE start / goal positions [m]
  dynamic   : True if any obstacle moves
  desc      : one-line description (used for the scenario table)

Randomisation (applied by the simulator, see sim_core.randomize):
  start/goal perturbation, per-obstacle position offset, phase of moving
  obstacles and the initial joint configuration of the arm.
"""

from __future__ import annotations

import numpy as np

from starworlds.shapes3d import Sphere3D, Box3D, Cylinder3D, quat_to_matrix


def _static(c):
    c = np.asarray(c, float)
    return lambda t: c.copy()


def _rotz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def _rotx(a):
    return np.array([[1.0, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


C_ORANGE = [1.0, 0.45, 0.0, 0.9]
C_AMBER = [1.0, 0.65, 0.0, 0.9]
C_RED = [0.85, 0.1, 0.1, 0.9]
C_BLUE = [0.15, 0.25, 0.9, 0.9]
C_GREEN = [0.1, 0.75, 0.2, 0.9]
C_GREY = [0.5, 0.5, 0.58, 0.9]
C_PURPLE = [0.6, 0.2, 0.75, 0.9]


def sphere(c, r, color, traj=None):
    return {'shape': Sphere3D(c, r), 'traj': traj or _static(c), 'color': color}


def single():
    return dict(
        desc="One sphere centred on the straight start-goal segment",
        obstacles=[sphere([0.50, 0.0, 0.50], 0.10, C_ORANGE)],
        start=[0.38, -0.15, 0.50], goal=[0.62, 0.15, 0.50])


def overlap():
    return dict(
        desc="Two strongly overlapping spheres forming a wall across the path",
        obstacles=[sphere([0.55, -0.06, 0.45], 0.09, C_ORANGE),
                   sphere([0.55, 0.06, 0.45], 0.09, C_AMBER)],
        start=[0.30, 0.0, 0.45], goal=[0.72, 0.0, 0.45])


def wall():
    return dict(
        desc="Three overlapping spheres stacked into a vertical wall; the path crosses a junction",
        obstacles=[sphere([0.45, 0.0, 0.30], 0.075, C_RED),
                   sphere([0.45, 0.0, 0.42], 0.075, C_ORANGE),
                   sphere([0.45, 0.0, 0.54], 0.075, C_BLUE)],
        start=[0.35, -0.25, 0.48], goal=[0.55, 0.25, 0.48])


def simple():
    return dict(
        desc="Two slowly moving spheres and a static overlapping pair",
        obstacles=[
            sphere([0.35, 0.30, 0.40], 0.08, C_RED,
                   lambda t: np.array([0.35, 0.30 + 0.04 * np.sin(t), 0.40])),
            sphere([0.42, -0.30, 0.30], 0.07, C_BLUE,
                   lambda t: np.array([0.42 + 0.04 * np.cos(1.5 * t), -0.30, 0.30])),
            sphere([0.58, 0.06, 0.40], 0.055, C_GREEN),
            sphere([0.64, -0.04, 0.40], 0.055, [0.6, 1.0, 0.1, 0.9]),
        ],
        start=[0.28, -0.40, 0.40], goal=[0.60, 0.38, 0.40])


def moderate():
    return dict(
        desc="Six spheres, five oscillating, medium density",
        obstacles=[
            sphere([0.30, 0.40, 0.40], 0.09, C_RED,
                   lambda t: np.array([0.30 + 0.05 * np.sin(0.8 * t), 0.40, 0.40])),
            sphere([0.50, 0.35, 0.40], 0.08, C_ORANGE,
                   lambda t: np.array([0.50, 0.35 + 0.03 * np.cos(1.2 * t), 0.40])),
            sphere([0.40, 0.10, 0.35], 0.07, C_GREEN,
                   lambda t: np.array([0.40 + 0.04 * np.sin(t), 0.10, 0.35])),
            sphere([0.60, 0.00, 0.40], 0.08, [0, 0.8, 0.8, 0.9],
                   lambda t: np.array([0.60, 0.05 * np.sin(1.5 * t), 0.40])),
            sphere([0.35, -0.35, 0.30], 0.09, C_BLUE),
            sphere([0.55, -0.25, 0.35], 0.07, C_PURPLE,
                   lambda t: np.array([0.55 + 0.03 * np.cos(2 * t), -0.25, 0.35])),
        ],
        start=[0.25, -0.40, 0.40], goal=[0.60, 0.44, 0.40])


def corridor():
    """
    Revised corridor.  In the submitted version the inflated blocker
    intersected both inflated walls at every phase of its motion, so no
    path respecting the safety margin existed.  Here the walls are 0.42 m
    apart and the blocker periodically opens a margin-respecting gap of up
    to 9 cm on either side.
    """
    obs = []
    for y in (-0.28, 0.0, 0.28):
        obs.append(sphere([0.24, y, 0.40], 0.07, C_GREY))
        obs.append(sphere([0.66, y, 0.40], 0.07, [0.58, 0.48, 0.48, 0.9]))
    obs.append(sphere([0.45, 0.0, 0.40], 0.05, C_ORANGE,
                      lambda t: np.array([0.45 + 0.10 * np.sin(0.7 * t), 0.0, 0.40])))
    return dict(desc="Narrow corridor of six spheres with an oscillating blocker",
                obstacles=obs, start=[0.45, -0.52, 0.40], goal=[0.45, 0.52, 0.40])


def dynamic():
    """Continuously moving obstacles crossing the path (up to 0.15 m/s)."""
    def pair(dz):
        return lambda t: np.array([0.47, 0.22 * np.sin(0.6 * t), 0.42 + dz])
    return dict(
        desc="Overlapping sphere pair sweeping across the path plus a crossing sphere",
        obstacles=[
            sphere([0.47, 0.0, 0.37], 0.07, C_ORANGE, pair(-0.05)),
            sphere([0.47, 0.0, 0.47], 0.07, C_AMBER, pair(0.05)),
            sphere([0.40, 0.18, 0.40], 0.06, C_BLUE,
                   lambda t: np.array([0.40 + 0.12 * np.sin(0.9 * t + 1.0), 0.18, 0.40])),
        ],
        start=[0.30, -0.36, 0.42], goal=[0.56, 0.36, 0.42])


def boxes():
    return dict(
        desc="Two overlapping boxes forming a T-shaped (non-convex) barrier",
        obstacles=[
            {'shape': Box3D([0.50, 0.0, 0.40], [0.03, 0.14, 0.05]),
             'traj': _static([0.50, 0.0, 0.40]), 'color': C_BLUE},
            {'shape': Box3D([0.50, 0.0, 0.50], [0.03, 0.05, 0.12]),
             'traj': _static([0.50, 0.0, 0.50]), 'color': C_GREEN},
        ],
        start=[0.32, -0.04, 0.44], goal=[0.68, 0.04, 0.44])


def cylinders():
    return dict(
        desc="Two overlapping vertical cylinders and a tilted cylinder",
        obstacles=[
            {'shape': Cylinder3D([0.50, -0.05, 0.42], 0.06, 0.14),
             'traj': _static([0.50, -0.05, 0.42]), 'color': C_ORANGE},
            {'shape': Cylinder3D([0.50, 0.06, 0.42], 0.06, 0.14),
             'traj': _static([0.50, 0.06, 0.42]), 'color': C_AMBER},
            {'shape': Cylinder3D([0.40, -0.30, 0.40], 0.05, 0.12, R=_rotx(0.6)),
             'traj': _static([0.40, -0.30, 0.40]), 'color': C_GREY},
        ],
        start=[0.32, 0.0, 0.42], goal=[0.68, 0.02, 0.42])


def irregular():
    """Non-convex L-shaped object = union of a box, a cylinder and a sphere."""
    return dict(
        desc="Irregular non-convex object (box + cylinder + sphere union)",
        obstacles=[
            {'shape': Box3D([0.50, -0.02, 0.36], [0.04, 0.12, 0.04], R=_rotz(0.3)),
             'traj': _static([0.50, -0.02, 0.36]), 'color': C_PURPLE},
            {'shape': Cylinder3D([0.50, 0.08, 0.46], 0.045, 0.11),
             'traj': _static([0.50, 0.08, 0.46]), 'color': C_GREEN},
            sphere([0.50, -0.06, 0.45], 0.07, C_RED),
        ],
        start=[0.32, 0.0, 0.42], goal=[0.68, 0.0, 0.44])


SCENARIOS = {
    'single': single, 'overlap': overlap, 'wall': wall, 'simple': simple,
    'moderate': moderate, 'corridor': corridor, 'dynamic': dynamic,
    'boxes': boxes, 'cylinders': cylinders, 'irregular': irregular,
}
CORE = ['single', 'overlap', 'wall', 'simple', 'moderate', 'corridor', 'dynamic']
SHAPES = ['boxes', 'cylinders', 'irregular']


def get(name):
    return SCENARIOS[name]()
