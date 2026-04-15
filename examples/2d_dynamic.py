"""
Dynamic 2-D example: moving obstacles with online Algorithm 2.

Reproduces the spirit of Paper Fig. 18:
  - Two circular obstacles (modelling humans) moving on sinusoidal paths
  - One L-shaped wall modelled as two convex polygon pieces
  - All obstacles inflated by a robot radius margin
  - Algorithm 2 called at each timestep via StarWorldUpdater
  - Temporal coherence keeps kernel centroids stable across frames

Run (interactive animation):
    python examples/2d_dynamic.py

Run (save to gif — requires pillow):
    python examples/2d_dynamic.py --save
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from starworlds import Ellipse, ConvexPolygon2D, StarWorldUpdater
from examples.visualize import (
    draw_obstacle, draw_proxy, draw_kernel_triangle,
    draw_robot_goal, draw_center_points, setup_axes,
)

# ---------------------------------------------------------------------------
# Scene parameters
# ---------------------------------------------------------------------------

ROBOT_RADIUS = 0.25   # inflation margin added to each obstacle
DT           = 0.05   # timestep [s]
T_MAX        = 8.0    # simulation duration [s]
FRAMES       = int(T_MAX / DT)

X_ROBOT = np.array([-4.0, -1.0])
X_GOAL  = np.array([ 4.0,  1.0])

XLIM = (-6, 6)
YLIM = (-5, 5)

PROXY_COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0",
                "#00BCD4", "#FF5722"]


# ---------------------------------------------------------------------------
# Obstacle generators
# ---------------------------------------------------------------------------

def make_scene(t: float):
    """Return list of obstacles at time t (all inflated by ROBOT_RADIUS)."""
    r = ROBOT_RADIUS

    # Human 1: circle of radius 0.4+r, moves right-then-left
    c1 = np.array([-1.0 + 1.5 * np.sin(0.8 * t), -1.5])
    h1 = Ellipse(center=c1, a=0.4 + r, b=0.4 + r)

    # Human 2: circle of radius 0.4+r, moves in a figure-of-eight
    c2 = np.array([1.5 * np.cos(0.6 * t), 1.0 + 0.6 * np.sin(1.2 * t)])
    h2 = Ellipse(center=c2, a=0.4 + r, b=0.4 + r)

    # Wall 1: horizontal bar (static)
    w1 = ConvexPolygon2D(np.array([
        [-3.0, -3.0], [3.0, -3.0], [3.0, -2.5 + r], [-3.0, -2.5 + r]
    ]))

    # Wall 2: vertical segment (static)
    w2 = ConvexPolygon2D(np.array([
        [-0.5 - r, -2.5], [0.5 + r, -2.5], [0.5 + r, 0.5], [-0.5 - r, 0.5]
    ]))

    return [h1, h2, w1, w2]


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def run_animation(save: bool = False):
    updater = StarWorldUpdater(L=300.0, n_approx=32)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(
        "Star Worlds — online Algorithm 2 (dynamic obstacles)\n"
        "Dahlin & Karayiannidis, IEEE TRO 2023",
        fontsize=11,
    )

    frame_patches = []  # track artists to clear each frame

    def init():
        setup_axes(ax, XLIM, YLIM)
        ax.plot(*X_ROBOT, 'o', color='k', markersize=8, zorder=10)
        ax.plot(*X_GOAL,  '*', color='k', markersize=12, zorder=10)
        return []

    def update(frame_idx):
        # Clear dynamic artists from previous frame
        for p in frame_patches:
            p.remove()
        frame_patches.clear()

        t = frame_idx * DT
        obstacles = make_scene(t)

        # Run Algorithm 2
        result = updater.update(obstacles, X_ROBOT, X_GOAL)

        # Draw original obstacles (grey)
        for obs in obstacles:
            verts = obs.polygon_approximation(n=32)
            patch = plt.Polygon(verts, closed=True, facecolor="#555555",
                                edgecolor="k", alpha=0.4, linewidth=1.0, zorder=2)
            ax.add_patch(patch)
            frame_patches.append(patch)

        # Draw proxy obstacles (coloured)
        for i, proxy in enumerate(result.proxies):
            color = PROXY_COLORS[i % len(PROXY_COLORS)]
            for piece in proxy.pieces:
                if piece is None or len(piece) < 3:
                    continue
                patch = plt.Polygon(piece, closed=True,
                                    facecolor=color, edgecolor=color,
                                    alpha=0.25, linewidth=1.5, zorder=3)
                ax.add_patch(patch)
                frame_patches.append(patch)
                outline = plt.Polygon(piece, closed=True, fill=False,
                                      edgecolor=color, linewidth=1.5, zorder=4)
                ax.add_patch(outline)
                frame_patches.append(outline)

            # Kernel triangle
            if proxy.kernel_K is not None:
                tri = plt.Polygon(proxy.kernel_K, closed=True,
                                  facecolor="#1565C0", edgecolor="#1565C0",
                                  alpha=0.4, linewidth=0.8, zorder=5)
                ax.add_patch(tri)
                frame_patches.append(tri)

            # Center point
            dot, = ax.plot(*proxy.center_point, 'D', color="#00C853",
                           markersize=7, zorder=6)
            frame_patches.append(dot)

        # Status text
        kind = "disjoint" if result.is_disjoint else "intersecting"
        txt = ax.text(0.02, 0.97,
                      f"t = {t:.2f}s | {len(result.proxies)} proxy(ies) | {kind}",
                      transform=ax.transAxes, fontsize=8, va="top",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        frame_patches.append(txt)

        setup_axes(ax, XLIM, YLIM, title="")
        return frame_patches

    anim = animation.FuncAnimation(
        fig, update, frames=FRAMES, init_func=init,
        interval=int(DT * 1000), blit=False,
    )

    if save:
        out = os.path.join(os.path.dirname(__file__), "dynamic_result.gif")
        print(f"Saving animation → {out}  (this may take a moment)…")
        anim.save(out, writer="pillow", fps=int(1 / DT))
        print("Done.")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Save animation as dynamic_result.gif")
    args = parser.parse_args()
    run_animation(save=args.save)
