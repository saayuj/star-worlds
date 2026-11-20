"""
Generate every results figure and LaTeX table of the revised manuscript.

  python -m experiments.make_figures [scenarios|results|all]

Outputs go to ../revision/figures and ../revision/tables.
"""

from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from experiments import scenarios as S              # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, '..', 'revision', 'figures')
TAB = os.path.join(ROOT, '..', 'revision', 'tables')

# Fixed entity colours (validated categorical palette, fixed order)
COL = {'sw': '#2a78d6', 'mod': '#eb6834', 'ch': '#1baf7a', 'cbf': '#eda100',
       'apf': '#e87ba4', 'sw_legacy': '#6250d6', 'mod_legacy': '#8a8984',
       'sw_adaptive': '#2a78d6'}
MARK = {'sw': 'o', 'mod': 's', 'ch': '^', 'cbf': 'D', 'apf': 'v', 'sw_legacy': 'P',
        'mod_legacy': 'X', 'sw_adaptive': 'o'}
LABEL = {'sw': 'SW (ours)', 'sw_adaptive': 'SW-adaptive', 'mod': 'MOD', 'ch': 'CH-merge',
         'cbf': 'CBF-QP', 'apf': 'APF', 'sw_legacy': 'SW-v1', 'mod_legacy': 'MOD-v1'}
ORDER = ['sw', 'sw_adaptive', 'mod', 'ch', 'cbf', 'apf', 'sw_legacy', 'mod_legacy']
SC_ORDER = S.CORE + S.SHAPES
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.edgecolor': '#9a9993',
    'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.5,
    'font.family': 'serif', 'mathtext.fontset': 'cm', 'savefig.dpi': 300,
})


def load(name):
    import pandas as pd
    f = os.path.join(RES, f'{name}.jsonl')
    if not os.path.exists(f):
        return None
    df = pd.DataFrame([json.loads(l) for l in open(f)])
    if 'error' in df:
        bad = df['error'].notna()
        if bad.any():
            print(f'[{name}] {bad.sum()} errored trials', df.loc[bad, 'error'].unique()[:3])
        df = df[~bad]
    return df


def mcnemar(df, a, b):
    """Exact two-sided McNemar test on trials paired by (scenario, seed)."""
    from scipy.stats import binomtest
    if a == b:
        return None
    A = df[df.method == a].set_index(['scenario', 'seed']).success
    Bm = df[df.method == b].set_index(['scenario', 'seed']).success
    j = A.to_frame('a').join(Bm.to_frame('b'), how='inner')
    n10 = int((j.a & ~j.b).sum())
    n01 = int((~j.a & j.b).sum())
    if n10 + n01 == 0:
        return 1.0
    return float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ---------------------------------------------------------------------------
# Scenario schematic + table
# ---------------------------------------------------------------------------

def _draw_shape(ax, sh, color, alpha=0.55):
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 16)
    if sh.kind == 'sphere':
        x = sh.radius * np.outer(np.cos(u), np.sin(v))
        y = sh.radius * np.outer(np.sin(u), np.sin(v))
        z = sh.radius * np.outer(np.ones_like(u), np.cos(v))
        P = np.stack([x, y, z], -1)
    elif sh.kind == 'cylinder':
        zz = np.linspace(-sh.hh, sh.hh, 8)
        U, Z = np.meshgrid(u, zz)
        P = np.stack([sh.radius * np.cos(U), sh.radius * np.sin(U), Z], -1)
    else:
        h = sh.h
        faces = []
        for ax_i in range(3):
            for sgn in (-1, 1):
                a, b = [k for k in range(3) if k != ax_i]
                s1, s2 = np.meshgrid(np.linspace(-1, 1, 2), np.linspace(-1, 1, 2))
                F = np.zeros(s1.shape + (3,))
                F[..., ax_i] = sgn * h[ax_i]
                F[..., a] = s1 * h[a]
                F[..., b] = s2 * h[b]
                faces.append(F)
        for F in faces:
            W = F @ sh.R.T + sh.center
            ax.plot_surface(W[..., 0], W[..., 1], W[..., 2], color=color, alpha=alpha,
                            linewidth=0, shade=True)
        return
    W = P @ sh.R.T + sh.center
    ax.plot_surface(W[..., 0], W[..., 1], W[..., 2], color=color, alpha=alpha, linewidth=0,
                    shade=True)


def fig_scenarios():
    fig = plt.figure(figsize=(7.2, 3.3))
    for k, name in enumerate(SC_ORDER):
        spec = S.get(name)
        ax = fig.add_subplot(2, 5, k + 1, projection='3d')
        for o in spec['obstacles']:
            c0 = o['traj'](0.0)
            _draw_shape(ax, o['shape'].moved(c0), o['color'][:3])
            ts = np.linspace(0, 12, 200)
            tr = np.array([o['traj'](t) for t in ts])
            if np.ptp(tr, axis=0).max() > 1e-6:
                ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=INK2, lw=0.8, ls='--')
        s, g = np.array(spec['start']), np.array(spec['goal'])
        ax.plot(*np.c_[s, g], color='#9a9993', lw=0.8, ls=':')
        ax.scatter(*s, color='#008300', s=14, depthshade=False)
        ax.scatter(*g, color='#d03b3b', marker='*', s=30, depthshade=False)
        ax.set_xlim(0.2, 0.75)
        ax.set_ylim(-0.55, 0.55)
        ax.set_zlim(0.15, 0.7)
        ax.set_box_aspect((0.55, 1.1, 0.55))
        ax.view_init(elev=22, azim=-60)
        ax.set_title(name, pad=-2)
        ax.tick_params(pad=-4, labelsize=5)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.set_alpha(0.0)
            a._axinfo['grid']['color'] = GRID
        ax.set_xticks([0.3, 0.7])
        ax.set_yticks([-0.4, 0, 0.4])
        ax.set_zticks([0.2, 0.6])
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.95, wspace=0.0, hspace=0.12)
    fig.savefig(os.path.join(FIG, 'scenarios.pdf'))
    fig.savefig(os.path.join(FIG, 'scenarios.png'))
    plt.close(fig)


def _fmt(v):
    return '(' + ', '.join(f'{x:.2f}' for x in v) + ')'


def tab_scenarios():
    lines = [r'\begin{table*}[t]', r'\centering',
             r'\caption{Benchmark scenarios. Positions in metres in the robot base frame; '
             r'radii and half-extents of the true (un-inflated) obstacles. Every trial adds '
             r'$\Delta r=0.05$\,m, perturbs start and goal by $\mathcal{U}(\pm0.03)$\,m per axis '
             r'(rejection-sampled outside the inflated obstacles), obstacle positions by '
             r'$\mathcal{U}(\pm0.015)$\,m, the phase of moving obstacles, and the initial joint '
             r'configuration by $\mathcal{N}(0,0.25^2)$\,rad around '
             r'$q_\mathrm{nom}=(0,-0.5,0,-2.0,0,1.8,0.785)$\,rad. Boxes: half-extents $h$; '
             r'cylinders: radius and half-height $hh$ (axis along local $z$); $T$: period of '
             r'the oscillation. The exact definitions are in \texttt{experiments/scenarios.py} '
             r'of the public repository (tag \texttt{robomech-revision}).}',
             r'\label{tab:scenarios}', r'\scriptsize',
             r'\begin{tabular}{@{}l p{8.2cm} l l@{}}', r'\toprule',
             r'Scenario & Obstacles (shape, centre, size; motion) & Start & Goal \\', r'\midrule']
    for name in SC_ORDER:
        spec = S.get(name)
        parts = []
        for o in spec['obstacles']:
            sh = o['shape']
            c = o['traj'](0.0)
            if sh.kind == 'sphere':
                sz = f'r={sh.radius:.3f}'
            elif sh.kind == 'box':
                sz = 'h=' + _fmt(sh.h)
            else:
                sz = f'r={sh.radius:.3f}, hh={sh.hh:.2f}'
            mv = ''
            ts = np.linspace(0, 400, 40001)
            tr = np.array([o['traj'](t) for t in ts])
            if np.ptp(tr, axis=0).max() > 1e-6:
                amp = np.ptp(tr, axis=0) / 2
                k = int(np.argmax(amp))
                sig = tr[:, k] - tr[:, k].mean()
                F = np.abs(np.fft.rfft(sig))
                fr = np.fft.rfftfreq(len(sig), ts[1] - ts[0])
                per = 1.0 / fr[1 + int(np.argmax(F[1:]))]
                axn = 'xyz'[k]
                mv = rf', osc.\ ${axn}\pm{amp.max():.2f}$, $T={per:.1f}$\,s'
            if not np.allclose(sh.R, np.eye(3)):
                from scipy.spatial.transform import Rotation
                rv = Rotation.from_matrix(sh.R).as_rotvec()
                ax_ = 'xyz'[int(np.argmax(np.abs(rv)))]
                sz += rf', rot.\ {np.linalg.norm(rv):.1f}\,rad about ${ax_}$'
            parts.append(f'{sh.kind[0].upper()}{_fmt(c)} {sz}{mv}')
        lines.append(rf'\texttt{{{name}}} & ' + '; '.join(parts) + rf' & {_fmt(spec["start"])} & {_fmt(spec["goal"])} \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    open(os.path.join(TAB, 'scenarios.tex'), 'w').write('\n'.join(lines))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def tab_main(df):
    """Success rate (%) per scenario × method, with collision rate marked."""
    meths = ['sw', 'sw_adaptive', 'mod', 'ch', 'cbf', 'apf', 'sw_legacy', 'mod_legacy']
    hdr = ' & '.join(LABEL[m] for m in meths)
    L = [r'\begin{table*}[t]', r'\centering',
         r'\caption{Main comparison: success rate (\%) over 20 randomised trials per cell '
         r'(180\,s limit). Success requires reaching the goal (2\,cm) with no robot--obstacle '
         r'penetration deeper than 5\,mm measured on the Franka collision meshes. Superscripts '
         r'give the number of trials with a collision. Bold: best per row. Last rows: overall '
         r'success with 95\,\% Wilson interval, median time to goal and median path-length ratio '
         r'of successful trials.}',
         r'\label{tab:main}', r'\small', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{@{}l' + 'c' * len(meths) + '@{}}',
         r'\toprule', r'Scenario & ' + hdr + r' \\', r'\midrule']
    for sc in SC_ORDER:
        d = df[df.scenario == sc]
        vals = {m: d[d.method == m] for m in meths}
        rates = {m: 100 * v.success.mean() if len(v) else np.nan for m, v in vals.items()}
        best = np.nanmax(list(rates.values()))
        cells = []
        for m in meths:
            v = vals[m]
            if not len(v):
                cells.append('--')
                continue
            s = f'{rates[m]:.0f}'
            if abs(rates[m] - best) < 1e-9:
                s = r'\textbf{' + s + '}'
            nc = int(v.collided.sum())
            if nc:
                s += rf'$^{{{nc}}}$'
            cells.append(s)
        L.append(rf'\texttt{{{sc}}} & ' + ' & '.join(cells) + r' \\')
        if sc == S.CORE[-1]:
            L.append(r'\midrule')
    L.append(r'\midrule')
    tot, tim, rat = [], [], []
    for m in meths:
        v = df[df.method == m]
        k, n = int(v.success.sum()), len(v)
        lo, hi = wilson(k, n)
        tot.append(rf'{100 * k / n:.1f} [{100 * lo:.0f},{100 * hi:.0f}]')
        sv = v[v.success]
        tim.append(f'{sv.t_reach.median():.2f}' if len(sv) else '--')
        rat.append(f'{sv.ratio.median():.2f}' if len(sv) else '--')
    L.append(r'Overall (\%) & ' + ' & '.join(r'\scriptsize ' + t for t in tot) + r' \\')
    pv = []
    for m in meths:
        p = mcnemar(df, 'sw', m)
        pv.append('--' if p is None else (r'$<$0.001' if p < 1e-3 else f'{p:.3f}'))
    L.append(r'$p$ vs.\ SW & ' + ' & '.join(pv) + r' \\')
    L.append(r'Median $T$ (s) & ' + ' & '.join(tim) + r' \\')
    L.append(r'Median $L/L_s$ & ' + ' & '.join(rat) + r' \\')
    if 'track_rms' in df:
        trk = [f"{df[df.method == m].track_rms.median():.3f}" for m in meths]
        L.append(r'Track.\ RMS (m/s) & ' + ' & '.join(trk) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table*}']
    open(os.path.join(TAB, 'main.tex'), 'w').write('\n'.join(L))


def fig_main(df):
    """Success rate per scenario: dot plot with Wilson intervals."""
    meths = ['sw', 'mod', 'ch', 'cbf', 'apf', 'sw_legacy']
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    xs = np.arange(len(SC_ORDER))
    w = 0.12
    for j, m in enumerate(meths):
        ys, lo, hi = [], [], []
        for sc in SC_ORDER:
            v = df[(df.scenario == sc) & (df.method == m)]
            k, n = int(v.success.sum()), len(v)
            if n == 0:
                ys.append(np.nan); lo.append(0); hi.append(0)
                continue
            a, b = wilson(k, n)
            ys.append(100 * k / n)
            lo.append(100 * (k / n - a))
            hi.append(100 * (b - k / n))
        off = (j - (len(meths) - 1) / 2) * w
        ax.errorbar(xs + off, ys, yerr=[lo, hi], fmt=MARK[m], ms=4.2, color=COL[m],
                    ecolor=COL[m], elinewidth=0.8, capsize=0, label=LABEL[m],
                    markeredgecolor='white', markeredgewidth=0.5)
    for x in xs[:-1]:
        ax.axvline(x + 0.5, color=GRID, lw=0.6, zorder=0)
    ax.axvline(len(S.CORE) - 0.5, color='#9a9993', lw=0.9, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(SC_ORDER)
    ax.set_ylabel('success rate (%)')
    ax.set_ylim(-5, 105)
    ax.set_xlim(-0.5, len(SC_ORDER) - 0.5)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=6, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False,
              handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, 'success.pdf'))
    plt.close(fig)


def fig_ratio(df):
    """Path-length ratio and time of successful trials (box plots)."""
    meths = ['sw', 'mod', 'ch', 'cbf']
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2), sharey=False)
    for ax, col, lab in ((axs[0], 'ratio', r'path-length ratio $L/L_s$'),
                         (axs[1], 't_reach', 'time to goal (s)')):
        xs = np.arange(len(SC_ORDER))
        w = 0.19
        for j, m in enumerate(meths):
            data = [df[(df.scenario == sc) & (df.method == m) & df.success][col].values
                    for sc in SC_ORDER]
            pos = xs + (j - 1.5) * w
            keep = [i for i, d in enumerate(data) if len(d)]
            bp = ax.boxplot([data[i] for i in keep], positions=pos[keep], widths=w * 0.85,
                            patch_artist=True, showfliers=False, whis=(5, 95),
                            medianprops=dict(color='white', lw=1.0),
                            boxprops=dict(lw=0), whiskerprops=dict(color=COL[m], lw=0.8),
                            capprops=dict(lw=0))
            for b in bp['boxes']:
                b.set_facecolor(COL[m])
            ax.plot([], [], 's', color=COL[m], label=LABEL[m], ms=5)
        ax.set_xticks(xs)
        ax.set_xticklabels(SC_ORDER, rotation=40, ha='right')
        ax.set_ylabel(lab)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.axvline(len(S.CORE) - 0.5, color='#9a9993', lw=0.9)
    axs[0].legend(ncol=4, loc='lower left', bbox_to_anchor=(0.0, 1.0), frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, 'ratio_time.pdf'))
    plt.close(fig)


def fig_margin(df):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.1))
    drs = sorted(df.dr.unique())
    for m in ('sw', 'mod'):
        d = df[df.method == m]
        succ = [100 * d[d.dr == r].success.mean() for r in drs]
        clr = [1000 * d[d.dr == r].min_ee_true.quantile(0.05) for r in drs]
        axs[0].plot(drs, succ, marker=MARK[m], color=COL[m], label=LABEL[m], ms=4.5)
        axs[1].plot(drs, clr, marker=MARK[m], color=COL[m], label=LABEL[m], ms=4.5)
    axs[0].set_ylabel('success rate (%)')
    axs[1].set_ylabel('5th pct. min. EE clearance (mm)')
    for ax in axs:
        ax.set_xlabel(r'safety margin $\Delta r$ (m)')
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_xticks(drs)
    axs[1].axhline(0, color='#9a9993', lw=0.8)
    axs[0].legend(frameon=False, loc='lower left')
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, 'margin.pdf'))
    plt.close(fig)


def fig_noise_rates_speed(noise, rates, speed, noise2=None):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.1))
    # (a) noise: default arm clearance (dashed) vs clearance sized to noise (solid)
    sds = sorted(noise.noise_std.unique())
    for m in ('sw', 'mod'):
        d = noise[noise.method == m]
        axs[0].plot([1000 * s for s in sds], [100 * d[d.noise_std == s].success.mean() for s in sds],
                    marker=MARK[m], color=COL[m], ms=4.0, ls='--', mfc='white',
                    label=LABEL[m] + r', $d_\mathrm{safe}$=1.5 cm')
        if noise2 is not None:
            d2 = noise2[noise2.method == m]
            xs = [0.0] + sorted(d2.noise_std.unique())
            ys = [100 * d[d.noise_std == 0].success.mean()] + \
                [100 * d2[d2.noise_std == s].success.mean() for s in xs[1:]]
            axs[0].plot([1000 * s for s in xs], ys, marker=MARK[m], color=COL[m], ms=4.5,
                        label=LABEL[m] + r', $d_\mathrm{safe}$=1.5 cm+2$\sigma$')
    axs[0].set_xlabel(r'perception noise $\sigma$ (mm)')
    axs[0].set_ylabel('success rate (%)')
    axs[0].set_title('(a) perception noise')
    axs[0].legend(frameon=False, loc='lower left', fontsize=5.5)
    axs[0].set_xticks([0, 5, 10])
    # (b) star-world rate
    rr = rates[rates.ctrl_hz == 40.0]
    hz = sorted(rr.sw_hz.unique())
    axs[1].plot(hz, [100 * rr[rr.sw_hz == h].success.mean() for h in hz], marker='o',
                color=COL['sw'], ms=4.5, label='success')
    axs[1].set_xscale('log')
    axs[1].set_xticks(hz)
    axs[1].set_xticklabels([f'{h:g}' for h in hz])
    axs[1].set_xlabel('star-world update rate (Hz)')
    axs[1].set_title('(b) update rate (SW, 4 scenes)')
    # (c) speed
    sp = sorted(speed.speed_scale.unique())
    for m in ('sw', 'mod', 'cbf'):
        d = speed[speed.method == m]
        axs[2].plot(sp, [100 * d[d.speed_scale == s].success.mean() for s in sp],
                    marker=MARK[m], color=COL[m], label=LABEL[m], ms=4.5)
    axs[2].set_xlabel('obstacle speed scale')
    axs[2].set_title('(c) obstacle speed')
    axs[2].legend(frameon=False, loc='lower left')
    for ax in axs:
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_ylim(-3, 103)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, 'robustness.pdf'))
    plt.close(fig)


def fig_timing(tim):
    import pandas as pd
    t = pd.DataFrame(tim)
    g = t.groupby('n')
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    n = g.size().index.values
    for col, lab, c, mk in (('t_warm_ms', 'SW update (Alg. 2)', COL['sw'], 'o'),
                            ('t_raw_ms', 'per-obstacle proxies', COL['mod'], 's'),
                            ('t_vel_ms', 'modulated velocity (SW)', COL['ch'], '^')):
        med = g[col].median().values
        q1 = g[col].quantile(0.1).values
        q9 = g[col].quantile(0.9).values
        ax.fill_between(n, q1, q9, color=c, alpha=0.18, lw=0)
        ax.plot(n, med, marker=mk, color=c, ms=3.8, label=lab)
    ax.set_yscale('log')
    ax.set_xlabel('number of obstacles $N$')
    ax.set_ylabel('time (ms)')
    ax.axhline(250, color='#9a9993', lw=0.8, ls='--')
    ax.text(n[0], 290, '4 Hz world-update budget', ha='left', fontsize=6, color=INK2)
    ax.axhline(25, color='#9a9993', lw=0.8, ls=':')
    ax.text(n[-1], 29, '40 Hz control budget', ha='right', fontsize=6, color=INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.6, which='major')
    ax.legend(frameon=False, loc='lower right', fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, 'timing.pdf'))
    plt.close(fig)


def fig_diag():
    """Near-equilibrium diagnosis: single (v1 vs revised) and overlap seed 1
    (revised without vs with workspace-aware kernel + flip)."""
    rows = [
        ('single, nominal', [('diag_single_legacy', COL['sw_legacy'], 'SW-v1 (submitted)'),
                             ('diag_single_sw', COL['sw'], 'SW (revised)')]),
        ('overlap, seed 1', [('diag_overlap_seed1_noflip', COL['mod'],
                              'SW without ws-aware side / flip'),
                             ('diag_overlap_seed1_sw', COL['sw'], 'SW (revised)')]),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.7), sharex='row')
    for r_i, (title, runs) in enumerate(rows):
        for name, c, lab in runs:
            f = os.path.join(RES, 'logs', name + '.pkl')
            if not os.path.exists(f):
                continue
            r = pickle.load(open(f, 'rb'))
            L = r['log']
            t = np.array(L['t'])
            X = np.array(L['x'])
            axs[r_i, 0].plot(t, np.linalg.norm(X - r['goal_pos'], axis=1), color=c, label=lab)
            axs[r_i, 1].plot(t, L['jl'], color=c)
            axs[r_i, 2].plot(t, L['manip'], color=c)
        axs[r_i, 0].set_ylabel('distance to goal (m)')
        axs[r_i, 1].set_ylabel('min. joint-limit\nmargin (rad)')
        axs[r_i, 2].set_ylabel('manipulability')
        axs[r_i, 0].legend(frameon=False, fontsize=6, title=title, title_fontsize=6.5,
                           loc='upper right')
        axs[r_i, 1].axhline(0.05, color='#9a9993', lw=0.8, ls=':')
        axs[r_i, 1].axhline(0, color='#9a9993', lw=0.8)
    for ax in axs.flat:
        ax.yaxis.grid(True, color=GRID, lw=0.6)
    for ax in axs[1]:
        ax.set_xlabel('time (s)')
    for ax in axs[0]:
        ax.set_xlabel('time (s)')
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, 'diagnosis.pdf'))
    plt.close(fig)


def tab_extra(abl, main, tout, path):
    import pandas as pd
    out = {}
    # ablation
    if abl is not None:
        base = main[main.method == 'sw']
        rows = [('full revised pipeline', base)]
        names = {'kernel': 'v1 kernel (centroid)', 'ctrl': 'v1 modulation',
                 'arm': 'v1 arm layer (IK + null-space APF)', 'ws': 'no workspace-aware side',
                 'flip': 'no deadlock flip', 'ws,flip': 'neither ws-aware nor flip'}
        for k, lab in names.items():
            rows.append((lab, abl[abl.ablate == k]))
        L = [r'\begin{table}[t]', r'\centering',
             r'\caption{Ablation of the revisions (SW, 10 scenarios $\times$ 20 trials). '
             r'Each row reverts one component to the submitted version.}',
             r'\label{tab:ablation}', r'\small', r'\resizebox{\columnwidth}{!}{%', r'\begin{tabular}{@{}lccc@{}}', r'\toprule',
             r'Variant & Success (\%) & Coll.\ (\%) & Med.\ $L/L_s$ \\', r'\midrule']
        for lab, d in rows:
            s = d[d.success]
            L.append(rf'{lab} & {100 * d.success.mean():.1f} & {100 * d.collided.mean():.1f} & '
                     rf'{s.ratio.median():.2f} \\')
        L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
        open(os.path.join(TAB, 'ablation.tex'), 'w').write('\n'.join(L))
    # long-horizon convergence of the submitted pipeline
    if tout is not None:
        L = [r'\begin{table}[t]', r'\centering',
             r'\caption{Submitted (v1) pipeline with a 600\,s limit (20 trials per cell): '
             r'goal reached (\%), collisions (\%), and median / maximum time to goal (s).}',
             r'\label{tab:timeout}', r'\small', r'\resizebox{\columnwidth}{!}{%', r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
             r'Scenario & \multicolumn{2}{c}{SW-v1} & \multicolumn{2}{c}{MOD-v1} \\',
             r' & reach / coll. & $T$ med / max & reach / coll. & $T$ med / max \\', r'\midrule']
        for sc in SC_ORDER:
            cells = []
            for m in ('sw_legacy', 'mod_legacy'):
                d = tout[(tout.scenario == sc) & (tout.method == m)]
                rr = d[d.reached]
                cells.append(f'{100 * d.reached.mean():.0f} / {100 * d.collided.mean():.0f}')
                cells.append(f'{rr.t_reach.median():.1f} / {rr.t_reach.max():.1f}' if len(rr) else '--')
            L.append(rf'\texttt{{{sc}}} & ' + ' & '.join(cells) + r' \\')
        L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
        open(os.path.join(TAB, 'timeout.tex'), 'w').write('\n'.join(L))
    # path decomposition
    if path is not None:
        P = pd.DataFrame(path)
        L = [r'\begin{table}[t]', r'\centering',
             r'\caption{Path-length decomposition for the static scenarios (medians over 20 '
             r'trials). $L_\mathrm{obs}$: shortest path respecting the inflated obstacles; '
             r'$L_\mathrm{prx}$: shortest path respecting the star-world proxies; '
             r'$L_\mathrm{SW}$, $L_\mathrm{MOD}$: executed paths (successful trials).}',
             r'\label{tab:path}', r'\small', r'\resizebox{\columnwidth}{!}{%', r'\begin{tabular}{@{}lccccc@{}}', r'\toprule',
             r'Scenario & $\frac{L_\mathrm{obs}}{L_s}$ & $\frac{L_\mathrm{prx}}{L_\mathrm{obs}}$ & '
             r'$\frac{L_\mathrm{SW}}{L_\mathrm{prx}}$ & $\frac{L_\mathrm{SW}}{L_\mathrm{obs}}$ & '
             r'$\frac{L_\mathrm{MOD}}{L_\mathrm{obs}}$ \\', r'\midrule']
        for sc in P.scenario.unique():
            q = P[P.scenario == sc].set_index('seed')
            sw = main[(main.scenario == sc) & (main.method == 'sw') & main.success].set_index('seed')
            md = main[(main.scenario == sc) & (main.method == 'mod') & main.success].set_index('seed')
            j1 = q.join(sw[['path']], how='inner')
            j2 = q.join(md[['path']], how='inner')
            L.append(rf'\texttt{{{sc}}} & {np.median(q.L_obs / q.L_s):.2f} & '
                     rf'{np.median(q.L_prx / q.L_obs):.2f} & '
                     rf'{np.median(j1.path / j1.L_prx):.2f} & {np.median(j1.path / j1.L_obs):.2f} & '
                     + (rf'{np.median(j2.path / j2.L_obs):.2f}' if len(j2) else '--') + r' \\')
        L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
        open(os.path.join(TAB, 'path.tex'), 'w').write('\n'.join(L))


def tab_latency(df):
    if df is None:
        return
    conds = [(0.0, 'synchronous'), (-1.0, 'async., measured'), (0.1, 'fixed 0.10\\,s'),
             (0.25, 'fixed 0.25\\,s'), (0.5, 'fixed 0.50\\,s')]
    L = [r'\begin{table}[t]', r'\centering',
         r'\caption{Execution model and computation latency (E10; \texttt{dynamic}, '
         r'\texttt{moderate}, \texttt{corridor}, \texttt{simple}; 80 trials per cell). '
         r'Success / collision rate (\%), 5th percentile of the minimum robot--obstacle '
         r'clearance (mm) and median EE velocity tracking error (RMS, m/s).}',
         r'\label{tab:latency}', r'\resizebox{\columnwidth}{!}{%',
         r'\begin{tabular}{@{}lcccccc@{}}', r'\toprule',
         r'Execution & \multicolumn{3}{c}{SW} & \multicolumn{3}{c}{MOD} \\',
         r' & succ./coll. & clear. & track. & succ./coll. & clear. & track. \\', r'\midrule']
    for lat, lab in conds:
        cells = []
        for m in ('sw', 'mod'):
            d = df[(df.method == m) & (df.latency == lat)]
            cells += [f'{100 * d.success.mean():.0f} / {100 * d.collided.mean():.0f}',
                      f'{1000 * d.min_clearance.quantile(0.05):.0f}',
                      f'{d.track_rms.median():.3f}']
        L.append(lab + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    open(os.path.join(TAB, 'latency.tex'), 'w').write('\n'.join(L))


def numbers(dfs):
    """Key numbers quoted in the text → tables/numbers.json"""
    out = {}
    main = dfs['main']
    for m in main.method.unique():
        v = main[main.method == m]
        out[f'succ_{m}'] = 100 * v.success.mean()
        out[f'coll_{m}'] = 100 * v.collided.mean()
        out[f'reach_{m}'] = 100 * v.reached.mean()
        out[f'rep_ms_{m}'] = v.rep_ms_mean.mean()
        out[f'rep_ms_max_{m}'] = v.rep_ms_max.max()
        out[f'vel_ms_{m}'] = v.vel_ms_mean.mean()
        out[f'flips_{m}'] = v.n_flips.sum() if 'n_flips' in v else 0
        out[f'swfrac_{m}'] = v.sw_used_frac.mean()
        out[f'clr_min_{m}'] = v[v.success].min_clearance.min() if v.success.any() else None
    json.dump({k: (float(v) if v is not None else None) for k, v in out.items()},
              open(os.path.join(TAB, 'numbers.json'), 'w'), indent=1)


def main(which):
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    if which in ('scenarios', 'all'):
        fig_scenarios()
        tab_scenarios()
    if which in ('results', 'all'):
        dfs = {k: load(k) for k in ('main', 'margin', 'noise', 'noise2', 'rates', 'ablation', 'speed',
                                    'timeout', 'latency')}
        main_ = dfs['main']
        tab_main(main_)
        fig_main(main_)
        fig_ratio(main_)
        if dfs['margin'] is not None:
            fig_margin(dfs['margin'])
        if all(dfs[k] is not None for k in ('noise', 'rates', 'speed')):
            fig_noise_rates_speed(dfs['noise'], dfs['rates'], dfs['speed'], dfs.get('noise2'))
        tf = os.path.join(RES, 'timing.json')
        if os.path.exists(tf):
            fig_timing(json.load(open(tf)))
        fig_diag()
        pf = os.path.join(RES, 'path_analysis.json')
        tab_extra(dfs['ablation'], main_, dfs['timeout'],
                  json.load(open(pf)) if os.path.exists(pf) else None)
        tab_latency(dfs['latency'])
        numbers(dfs)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'all')
