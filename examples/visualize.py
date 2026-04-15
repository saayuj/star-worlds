"""
Shared visualization utilities for Star Worlds examples.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from typing import List, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from starworlds.obstacles import Ellipse, ConvexPolygon2D, Polygon2D
from starworlds.star_world import StarWorldResult, ObstacleProxy
from starworlds.geometry import polygon_centroid_2d


# Colour palette matching paper's figure style
ORIG_COLOR    = "#555555"    # grey  — original obstacles
PROXY_COLORS  = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0",
                 "#00BCD4", "#FF5722", "#CDDC39", "#E91E63"]
KERNEL_COLOR  = "#1565C0"    # dark blue — kernel triangle CH(K)
CENTER_COLOR  = "#00C853"    # green diamond — center point x_c
ROBOT_COLOR   = "#212121"    # black circle — robot x
GOAL_COLOR    = "#212121"    # black star — goal xg


def draw_obstacle(ax, obstacle, color=ORIG_COLOR, alpha=0.5, label=None):
    """Draw an original obstacle (filled polygon)."""
    verts = obstacle.polygon_approximation(n=64)
    poly = plt.Polygon(verts, closed=True, facecolor=color, edgecolor="k",
                       alpha=alpha, linewidth=1.2, label=label)
    ax.add_patch(poly)


def draw_proxy(ax, proxy: ObstacleProxy, color: str, alpha=0.35):
    """Draw a starshaped proxy obstacle (union of convex pieces)."""
    for i, piece in enumerate(proxy.pieces):
        if piece is None or len(piece) < 3:
            continue
        poly = plt.Polygon(piece, closed=True,
                           facecolor=color, edgecolor=color,
                           alpha=alpha, linewidth=1.5)
        ax.add_patch(poly)
        # Thin outline in darker shade
        outline = plt.Polygon(piece, closed=True, fill=False,
                              edgecolor=color, linewidth=1.5)
        ax.add_patch(outline)


def draw_kernel_triangle(ax, K: np.ndarray):
    """Draw the kernel triangle CH(K) in blue."""
    if K is None:
        return
    tri = plt.Polygon(K, closed=True, facecolor=KERNEL_COLOR,
                      edgecolor=KERNEL_COLOR, alpha=0.4, linewidth=1.0)
    ax.add_patch(tri)


def draw_robot_goal(ax, x: np.ndarray, xg: np.ndarray):
    ax.plot(*x,  'o', color=ROBOT_COLOR, markersize=8, zorder=10, label="Robot $x$")
    ax.plot(*xg, '*', color=GOAL_COLOR,  markersize=12, zorder=10, label="Goal $x_g$")


def draw_center_points(ax, result: StarWorldResult):
    for proxy in result.proxies:
        ax.plot(*proxy.center_point, 'D', color=CENTER_COLOR,
                markersize=8, zorder=9)


def setup_axes(ax, xlim, ylim, title=""):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.grid(True, linewidth=0.4, alpha=0.5)


def plot_star_world(
    obstacles,
    result: StarWorldResult,
    x: np.ndarray,
    xg: np.ndarray,
    title: str = "",
    xlim=(-6, 6),
    ylim=(-6, 6),
    ax=None,
    show=True,
):
    """
    Full star-world visualisation: original obstacles (grey) + proxy obstacles
    (coloured, translucent) + kernel triangles (blue) + robot/goal markers.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Original obstacles
    for i, obs in enumerate(obstacles):
        draw_obstacle(ax, obs, color=ORIG_COLOR, alpha=0.6,
                      label="Original" if i == 0 else None)

    # Proxy obstacles
    for i, proxy in enumerate(result.proxies):
        color = PROXY_COLORS[i % len(PROXY_COLORS)]
        draw_proxy(ax, proxy, color=color)
        draw_kernel_triangle(ax, proxy.kernel_K)

    draw_center_points(ax, result)
    draw_robot_goal(ax, x, xg)
    setup_axes(ax, xlim, ylim, title)

    kind = "Disjoint ★ world" if result.is_disjoint else "Intersecting ★ world (fallback)"
    ax.set_xlabel(f"{kind} — {len(result.proxies)} proxy obstacle(s)")

    if show:
        plt.tight_layout()
        plt.show()

    return ax
