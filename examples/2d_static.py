"""
Static 2-D example: three intersecting ellipses.

Reproduces the scenario from Paper Fig. 17:
  - Three overlapping ellipses in R²
  - Algorithm 2 merges them into a single disjoint starshaped proxy
  - Left panel: raw intersecting obstacles (local minima at intersections)
  - Right panel: star-world after Algorithm 2 (convergence restored)

Run:
    python examples/2d_static.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from starworlds import Ellipse, ConvexPolygon2D, create_star_world
from starworlds.star_world import StarWorldResult
from examples.visualize import (
    draw_obstacle, draw_proxy, draw_kernel_triangle,
    draw_robot_goal, draw_center_points, setup_axes,
)


# ---------------------------------------------------------------------------
# Scene definition (matches Paper Fig. 17 qualitatively)
# ---------------------------------------------------------------------------

OBSTACLES = [
    Ellipse(center=[-0.5, 0.0], a=1.4, b=0.7, angle=0.0),
    Ellipse(center=[ 0.5, 0.0], a=1.4, b=0.7, angle=0.0),
    Ellipse(center=[ 0.0, 1.0], a=1.4, b=0.7, angle=np.pi / 2),
]

X_ROBOT = np.array([-3.5, -3.0])
X_GOAL  = np.array([ 3.5,  3.0])

XLIM = (-5, 5)
YLIM = (-5, 5)


def main():
    # ------------------------------------------------------------------ #
    # Apply Algorithm 2                                                    #
    # ------------------------------------------------------------------ #
    result = create_star_world(OBSTACLES, X_ROBOT, X_GOAL, L=200.0)

    print("=" * 60)
    print("Static 2-D example: three intersecting ellipses")
    print("=" * 60)
    print(f"  Number of original obstacles : {len(OBSTACLES)}")
    print(f"  Number of proxy obstacles    : {len(result.proxies)}")
    print(f"  Result type                  : {'Disjoint' if result.is_disjoint else 'Intersecting (fallback)'}")
    for i, proxy in enumerate(result.proxies):
        print(f"  Proxy {i}: covers obstacles {proxy.original_indices}, "
              f"center = {np.round(proxy.center_point, 3)}")
    print()

    # ------------------------------------------------------------------ #
    # Plot                                                                  #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        "Star Worlds — Dahlin & Karayiannidis (2023)\n"
        "Three intersecting ellipses: before and after Algorithm 2",
        fontsize=12,
    )

    # Left: original (intersecting) obstacles
    ax_l = axes[0]
    for i, obs in enumerate(OBSTACLES):
        draw_obstacle(ax_l, obs, alpha=0.55, label=f"$O_{i+1}$" if i < 3 else None)
    draw_robot_goal(ax_l, X_ROBOT, X_GOAL)
    setup_axes(ax_l, XLIM, YLIM,
               title="Before: intersecting obstacles\n"
                     "(local minima exist at intersection points)")
    ax_l.legend(loc="lower right", fontsize=8)

    # Right: star world after Algorithm 2
    ax_r = axes[1]
    PROXY_COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
    for i, obs in enumerate(OBSTACLES):
        draw_obstacle(ax_r, obs, alpha=0.3)
    for i, proxy in enumerate(result.proxies):
        color = PROXY_COLORS[i % len(PROXY_COLORS)]
        draw_proxy(ax_r, proxy, color=color)
        draw_kernel_triangle(ax_r, proxy.kernel_K)
    draw_center_points(ax_r, result)
    draw_robot_goal(ax_r, X_ROBOT, X_GOAL)
    kind = "Disjoint" if result.is_disjoint else "Intersecting (fallback)"
    setup_axes(ax_r, XLIM, YLIM,
               title=f"After: Algorithm 2 ({kind} ★ world)\n"
                     f"{len(result.proxies)} proxy obstacle(s), convergence guaranteed")
    # Legend handles
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#555555", alpha=0.4, label="Original obstacles"),
        Patch(facecolor=PROXY_COLORS[0], alpha=0.5, label="Proxy obstacle (hull)"),
        Patch(facecolor="#1565C0", alpha=0.5, label="Kernel triangle CH(K)"),
    ]
    ax_r.legend(handles=handles, loc="lower right", fontsize=8)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "static_result.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved figure → {out}")
    plt.show()


if __name__ == "__main__":
    main()
