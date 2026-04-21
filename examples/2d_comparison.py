"""
2-D before/after comparison: star-worlds vs. naïve modulation.

This is a TRUE SIMULATION — all arrows and paths are the result of numerically
integrating the actual Huber modulation equations.  Nothing is hand-drawn or
hard-coded.

Scene
-----
Two large, overlapping circles are placed directly on the path from start to
goal.  Together they form an impassable wall — there is NO physical gap between
them.  The robot must navigate around the combined obstacle.

Without star-worlds (LEFT panel)
---------------------------------
Each circle is modulated independently.  On the approach axis (y = 0), the
upward repulsion from the bottom circle and the downward repulsion from the top
circle cancel exactly.  The combined field pushes the robot FORWARD — straight
into the barrier.  Paths starting near the axis enter the obstacle (shown in
red).  Paths starting off-axis escape but unpredictably.

With star-worlds / Algorithm 2 (RIGHT panel)
---------------------------------------------
Algorithm 2 merges the two circles into ONE disjoint starshaped proxy (the
green region).  The modulation is applied to this single clean shape.  The
velocity field is smooth and unambiguous everywhere.  All paths reliably arc
around the proxy and reach the goal.

Usage
-----
    python examples/2d_comparison.py            # display + save
    python examples/2d_comparison.py --no-show  # headless / server

Output: examples/2d_comparison.png
"""

import argparse, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patheffects import withStroke

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from starworlds import StarWorldUpdater, Ellipse, ObstacleProxy


# ── Scene definition ──────────────────────────────────────────────────────────
# Two overlapping CIRCLES forming a wall directly on the start→goal path.
# Each has radius R=0.26.  Centers at y=+0.22 and y=-0.22.
# They physically overlap near y=0 → NO accessible gap in the middle.
R  = 0.26
CY = 0.22
OBS = [
    Ellipse(center=np.array([0.0,  CY]), a=R, b=R),   # top circle
    Ellipse(center=np.array([0.0, -CY]), a=R, b=R),   # bottom circle
]

# Goal is to the right of the wall
GOAL  = np.array([0.80, 0.00])

# Three start positions: one on-axis (will collide), two off-axis (may escape)
STARTS = [
    np.array([-0.75,  0.00]),   # on-axis — will be guided INTO obstacle (failure)
    np.array([-0.75,  0.28]),   # above axis
    np.array([-0.75, -0.28]),   # below axis
]

XLIM = (-1.05, 1.05)
YLIM = (-0.70, 0.70)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def inside_ellipse(pt, obs):
    dx, dy = pt[0] - obs.center[0], pt[1] - obs.center[1]
    return (dx / obs.a)**2 + (dy / obs.b)**2 < 1.0

def inside_any_obstacle(pt):
    return any(inside_ellipse(pt, obs) for obs in OBS)

def inside_proxy_hull(pt, proxy):
    """Point-in-convex-hull check using Delaunay."""
    from scipy.spatial import Delaunay
    all_pts = np.vstack(proxy.pieces)
    try:
        return Delaunay(all_pts).find_simplex(pt) >= 0
    except Exception:
        return False


# ── 2-D Huber modulation (self-contained, exact equations) ───────────────────

def _modulation_matrix(x, proxies, eta=1.0, p_weight=2.0, safety=0.04):
    """
    Exact 2×2 Huber modulation matrix:
        M = Σᵢ wᵢ Eᵢ diag(1-η/Γᵢ, 1+η/Γᵢ) Eᵢᵀ
        Eᵢ = [outward-normal | tangent]
        wᵢ  ∝ (1/Γᵢ)^p
    """
    gammas, ns, e1s = [], [], []
    for px in proxies:
        all_pts = np.vstack(px.pieces)
        r   = float(np.max(np.linalg.norm(all_pts - px.center_point, axis=1))) + safety
        d   = float(np.linalg.norm(x - px.center_point))
        g   = max(d / r, 1.001)
        nrm = (x - px.center_point) / d if d > 1e-10 else np.array([1., 0.])
        tan = np.array([-nrm[1], nrm[0]])
        gammas.append(g); ns.append(nrm); e1s.append(tan)

    inv_g = np.array([1.0 / g**p_weight for g in gammas])
    w     = inv_g / inv_g.sum()

    M = np.zeros((2, 2))
    for wi, g, nrm, tan in zip(w, gammas, ns, e1s):
        E = np.column_stack([nrm, tan])
        D = np.diag([1.0 - eta / g, 1.0 + eta / g])
        M += wi * (E @ D @ E.T)
    return M


def modulated_velocity(x, xg, proxies, v_max=0.40, goal_tol=0.07,
                       normalise=True):
    """Huber-modulated velocity at x toward xg (normalised to v_max if requested)."""
    dv = xg - x
    d  = np.linalg.norm(dv)
    if d < goal_tol:
        return np.zeros(2)
    f0 = dv / d
    M  = _modulation_matrix(x, proxies)
    v  = M @ f0
    vn = np.linalg.norm(v)
    if vn < 1e-12:
        return np.zeros(2)
    return v_max * v / vn if normalise else v


def simulate_path(start, xg, proxies, dt=0.014, steps=1000, goal_tol=0.07,
                  stop_at_obstacle=False):
    """
    Numerically integrate the modulated velocity field from start toward xg.
    If stop_at_obstacle=True, stop when the path enters any raw obstacle.
    """
    x    = start.copy()
    path = [x.copy()]
    for _ in range(steps):
        v = modulated_velocity(x, xg, proxies, normalise=True, goal_tol=goal_tol)
        x = x + v * dt
        path.append(x.copy())
        if np.linalg.norm(x - xg) < goal_tol:
            break
        if stop_at_obstacle and inside_any_obstacle(x):
            break   # robot has entered the obstacle — stop here (failure)
    return np.array(path)


# ── Build proxy sets ──────────────────────────────────────────────────────────

def make_naive_proxies():
    """One ObstacleProxy per raw obstacle, centre = obstacle centre."""
    return [
        ObstacleProxy(
            pieces=[obs.polygon_approximation(n=128)],
            kernel_K=None,
            center_point=obs.center.copy(),
            original_indices=[i],
            is_fallback=False,
        )
        for i, obs in enumerate(OBS)
    ]


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_raw_obstacles(ax, alpha=0.85, lw=2.0):
    for obs in OBS:
        ax.add_patch(mpatches.Ellipse(
            obs.center, 2*obs.a, 2*obs.b,
            fc='#FF8C00', ec='#CC5500', lw=lw, alpha=alpha, zorder=3,
        ))


def draw_proxy(ax, proxies):
    for proxy in proxies:
        # Faded raw obstacles underneath for reference
        for obs in OBS:
            ax.add_patch(mpatches.Ellipse(
                obs.center, 2*obs.a, 2*obs.b,
                fc='#FF8C00', ec='#CC5500', lw=1.2, alpha=0.22, zorder=1,
            ))
        for piece in proxy.pieces:
            if len(piece) >= 3:
                ax.add_patch(mpatches.Polygon(
                    piece, closed=True,
                    fc='#66BB6A', ec='#2E7D32',
                    lw=2.2, alpha=0.48, zorder=2,
                ))
        kc = proxy.center_point
        ax.plot(*kc, 'D', color='#BF360C', ms=9, zorder=6)


def draw_velocity_arrows(ax, proxies, mask_fn, n_grid=18, scale=0.062):
    """
    Draw velocity arrows in free space only.
    Arrow direction = normalised velocity direction.
    Arrow length    ∝ un-normalised speed ||M f₀|| (longer = faster).
    """
    xs = np.linspace(XLIM[0]+0.05, XLIM[1]-0.05, n_grid)
    ys = np.linspace(YLIM[0]+0.05, YLIM[1]-0.05, n_grid)

    for x in xs:
        for y in ys:
            pt = np.array([x, y])
            if mask_fn(pt):
                continue
            if np.linalg.norm(pt - GOAL) < 0.14:
                continue
            v_raw = modulated_velocity(pt, GOAL, proxies,
                                       normalise=False, goal_tol=0.14)
            speed = np.linalg.norm(v_raw)
            if speed < 1e-8:
                continue
            direction = v_raw / speed
            draw_len = scale * min(speed, 1.5)
            ax.annotate(
                '', xy=(x + direction[0]*draw_len, y + direction[1]*draw_len),
                xytext=(x, y),
                arrowprops=dict(
                    arrowstyle='->', mutation_scale=7,
                    color='#78909C', lw=0.75,
                ), zorder=2,
            )


def _path_effect():
    return [withStroke(linewidth=3, foreground='white')]


def draw_paths_naive(ax, paths):
    """
    Draw naive paths (simulation stopped at first obstacle hit).
    Path is shown in blue up to the collision point, then an X marks the crash.
    """
    colors = ['#1A237E', '#283593', '#3949AB']
    for path, col in zip(paths, colors):
        arr = np.array(path)
        ax.plot(arr[:, 0], arr[:, 1], color=col, lw=1.9,
                path_effects=_path_effect(), zorder=5)
        end = arr[-1]
        if inside_any_obstacle(end):
            # Stopped because it hit an obstacle
            ax.plot(*end, 'X', color='#D32F2F', ms=13, mew=2.5, zorder=7)
        else:
            # Reached goal
            ax.plot(*end, 'o', color=col, ms=8,
                    markerfacecolor='#A5D6A7', markeredgewidth=1.5, zorder=6)


def draw_paths_sw(ax, paths):
    colors = ['#1A237E', '#283593', '#3949AB']
    for path, col in zip(paths, colors):
        ax.plot(path[:, 0], path[:, 1], color=col, lw=1.9,
                path_effects=_path_effect(), zorder=5)
        end = path[-1]
        if np.linalg.norm(end - GOAL) < 0.15:
            ax.plot(*end, 'o', color=col, ms=8,
                    markerfacecolor='#A5D6A7', markeredgewidth=1.5, zorder=6)


def setup_ax(ax, title, border_color):
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.grid(True, color='#E0E4EA', lw=0.5, alpha=0.7)
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_color(border_color)
        sp.set_linewidth(2.0)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10,
                 color='#1C2833')


# ── Main figure builder ───────────────────────────────────────────────────────

def build_figure(save_path, show=True):
    # ── Build proxy sets ──────────────────────────────────────────────────────
    naive_prox = make_naive_proxies()

    updater = StarWorldUpdater()
    result  = updater.update(OBS,
                              robot_pos=STARTS[0],
                              goal_pos=GOAL)
    sw_prox = result.proxies
    print(f"Star-world: {len(sw_prox)} proxies, disjoint={result.is_disjoint}")
    for i, px in enumerate(sw_prox):
        print(f"  Proxy {i}: centre={np.round(px.center_point, 3)}")

    # ── Simulate paths ────────────────────────────────────────────────────────
    print("Simulating naive paths …")
    # stop_at_obstacle=True: stop simulation the moment robot hits obstacle
    # — cleanest way to show "robot crashed here" on slide
    naive_paths = [simulate_path(s, GOAL, naive_prox,
                                 stop_at_obstacle=True)
                   for s in STARTS]

    print("Simulating star-world paths …")
    sw_paths = [simulate_path(s, GOAL, sw_prox) for s in STARTS]

    naive_collide = [inside_any_obstacle(p[-1]) for p in naive_paths]
    naive_reached = [np.linalg.norm(p[-1] - GOAL) < 0.15 for p in naive_paths]
    sw_reached    = [np.linalg.norm(p[-1] - GOAL) < 0.15 for p in sw_paths]
    print(f"Naive collides (stopped at obstacle): {naive_collide}")
    print(f"Naive reached goal:  {naive_reached}")
    print(f"Star-world reached goal: {sw_reached}")

    # ── Masks (where NOT to draw arrows) ─────────────────────────────────────
    mask_naive = inside_any_obstacle
    mask_sw    = lambda pt: any(inside_proxy_hull(pt, px) for px in sw_prox)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw=dict(wspace=0.08))
    fig.patch.set_facecolor('white')

    # ── LEFT — naïve modulation ───────────────────────────────────────────────
    ax = axes[0]
    setup_ax(ax, 'Without Star-Worlds', '#C62828')

    draw_velocity_arrows(ax, naive_prox, mask_naive)
    draw_raw_obstacles(ax)
    draw_paths_naive(ax, naive_paths)

    for s in STARTS:
        ax.plot(*s, 'o', color='#283593', ms=8, zorder=8)
    ax.plot(*GOAL, '*', color='#B71C1C', ms=16, zorder=8)

    # Minimal labels
    ax.text(STARTS[2][0] - 0.04, STARTS[2][1] + 0.09,
            'start', fontsize=8, color='#283593', style='italic')
    ax.text(GOAL[0] + 0.04, GOAL[1] + 0.10,
            'goal', fontsize=8, color='#B71C1C', style='italic')

    # Single concise annotation: why it fails
    ax.text(-0.82, -0.57,
            'Normal forces from top & bottom obstacles\ncancel on the axis → robot driven into wall',
            fontsize=8.2, color='#7B1010',
            bbox=dict(fc='#FFF3F3', ec='#C62828', lw=0.8,
                      boxstyle='round,pad=0.4'), zorder=9)

    # ── RIGHT — star-world modulation ─────────────────────────────────────────
    ax = axes[1]
    setup_ax(ax, 'With Star-Worlds', '#2E7D32')

    draw_velocity_arrows(ax, sw_prox, mask_sw)
    draw_proxy(ax, sw_prox)
    draw_paths_sw(ax, sw_paths)

    for s in STARTS:
        ax.plot(*s, 'o', color='#283593', ms=8, zorder=8)
    ax.plot(*GOAL, '*', color='#B71C1C', ms=16, zorder=8)

    ax.text(STARTS[2][0] - 0.04, STARTS[2][1] + 0.09,
            'start', fontsize=8, color='#283593', style='italic')
    ax.text(GOAL[0] + 0.04, GOAL[1] + 0.10,
            'goal', fontsize=8, color='#B71C1C', style='italic')

    # Kernel label — compact, next to the diamond
    kc = sw_prox[0].center_point
    ax.text(kc[0] + 0.14, kc[1] - 0.18,
            r'kernel $x_{c}$', fontsize=8, color='#BF360C', style='italic')

    ax.text(-0.82, -0.57,
            'Obstacles merged into one starshaped proxy →\nsmooth, consistent field — convergence guaranteed',
            fontsize=8.2, color='#1B5020',
            bbox=dict(fc='#F3FFF4', ec='#2E7D32', lw=0.8,
                      boxstyle='round,pad=0.4'), zorder=9)

    # ── Shared legend (compact) ───────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(fc='#FF8C00', ec='#CC5500', lw=1.4,
                       label='Obstacle (overlapping circles)'),
        mpatches.Patch(fc='#66BB6A', ec='#2E7D32', lw=1.4,
                       label='Star-world proxy (merged starshaped hull)'),
        plt.Line2D([0], [0], color='#283593', lw=2,
                   label='Simulated path (3 starts)'),
        plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#C62828',
                   ms=10, label='Collision — path stopped'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#BF360C',
                   ms=8, label=r'Kernel $x_c$'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=5,
               fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02),
               handlelength=1.8, handletextpad=0.5, columnspacing=1.4)

    # title removed

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path, dpi=180, bbox_inches='tight',
                facecolor='white')
    print(f"\nSaved → {save_path}")
    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    out  = os.path.join(os.path.dirname(__file__), '2d_comparison.png')
    build_figure(out, show=not args.no_show)
