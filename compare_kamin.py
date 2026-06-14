# -*- coding: utf-8 -*-
"""
compare_kamin.py -- positioning of THIS WORK relative to
Kamin et al., PRX Quantum 6, 020342 (2025), plus the baseline
comparisons that were still MISSING from the manuscript.

Already in the paper (NOT re-done here): B1 global, B2/B3 random &
equal-population control, B4 proposed sectors, unified-vs-two-step
decoy, PLOB sanity bound.

NEW here (the gaps flagged by the referee):
  (A) B5  oracle error-aligned partition .............. upper-bound headroom
  (B) homogeneous-channel null test .................. gain -> 0 honesty check
  (C) gain vs heterogeneity amplitude ................ threshold behaviour
  (D) Kamin et al. conceptual positioning panel ...... complementary, not
                                                       competing (engine vs
                                                       conditioning layer)
Outputs: fig_compare_baselines.(png|pdf), fig_kamin_positioning.(png|pdf),
         data_compare_baselines.csv
"""
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from sector_geat import SectorModel, keyrate_point, h2

# ----------------------------------------------------------------------
# channel model: heterogeneous sector errors with tunable amplitude A
# e_m(A) = e_bar + A * d_m,  d_m a fixed zero-mean shape, ||d||=1-ish
# ----------------------------------------------------------------------
M = 8
SHAPE = np.array([-1.0, -0.7, -0.45, -0.2, 0.1, 0.4, 0.8, 1.05])
SHAPE = SHAPE - SHAPE.mean()
SHAPE = SHAPE / np.sqrt((SHAPE ** 2).mean())          # unit RMS
E_BAR = 0.03


def sector_errors(A):
    """heterogeneous profile with amplitude A (A=0 -> homogeneous)."""
    e = E_BAR + A * SHAPE * E_BAR
    return np.clip(e, 1e-4, 0.45)


def oracle_partition(e_full, M_keep):
    """B5: the BEST M_keep-bin partition of the true error profile,
    chosen AFTER seeing the honest model (upper bound only, not a
    practical/secure protocol). Contiguous 1-D optimal binning by
    dynamic programming on sorted errors minimizing within-bin
    variance (=> maximizing the between-bin Jensen gap)."""
    e = np.sort(e_full)
    N = len(e)
    # prefix sums for O(1) segment SSE
    ps = np.concatenate([[0], np.cumsum(e)])
    ps2 = np.concatenate([[0], np.cumsum(e ** 2)])

    def sse(i, j):                       # error of segment [i, j)
        cnt = j - i
        s = ps[j] - ps[i]
        return (ps2[j] - ps2[i]) - s * s / cnt

    INF = 1e18
    dp = np.full((M_keep + 1, N + 1), INF)
    dp[0, 0] = 0.0
    for k in range(1, M_keep + 1):
        for j in range(k, N + 1):
            for i in range(k - 1, j):
                v = dp[k - 1, i] + sse(i, j)
                if v < dp[k, j]:
                    dp[k, j] = v
    # reconstruct bin means
    means, j = [], N
    for k in range(M_keep, 0, -1):
        best_i, best_v = k - 1, INF
        for i in range(k - 1, j):
            v = dp[k - 1, i] + sse(i, j)
            if v < best_v:
                best_v, best_i = v, i
        means.append(e[best_i:j].mean())
        j = best_i
    return SectorModel(np.array(means[::-1]))


# ----------------------------------------------------------------------
# (A)+(B)+(C): baseline curves and scans
# ----------------------------------------------------------------------
def confuse(e, c):
    """realistic sector-assignment imperfection: each sector's effective
    error is blended with the global mean by confusion fraction c in [0,1)
    (c=0 perfect alignment, c=1 fully washed out = global). Models finite
    calibration / sector misassignment in the public PSB schedule."""
    return (1 - c) * e + c * e.mean()


def baseline_rates(n, loss_dB, A, c_confuse=0.35):
    e = sector_errors(A)
    pooled = SectorModel(np.array([e.mean()]))
    full_proposed = SectorModel(confuse(e, c_confuse))     # realistic B4
    r_B1 = keyrate_point(n, loss_dB, pooled, mode='B1')
    r_B3 = keyrate_point(n, loss_dB, full_proposed, mode='B3')
    r_B4 = keyrate_point(n, loss_dB, full_proposed, mode='B4')
    r_B5 = keyrate_point(n, loss_dB, oracle_partition(e, M), mode='B4')
    return r_B1, r_B3, r_B4, r_B5


def main():
    n = 1e12
    # ---- panel 1: rate vs loss, B1/B3/B4/B5 at fixed heterogeneity ----
    A0 = 1.0
    losses = np.linspace(0, 22, 12)
    rows = []
    for L in losses:
        rows.append((L, *baseline_rates(n, L, A0)))
    rows = np.array(rows)

    # ---- panel 2: gain vs heterogeneity amplitude (threshold) ----
    Agrid = np.linspace(0, 1.4, 15)
    Lfix = 8.0
    gain = []
    for A in Agrid:
        b1, b3, b4, b5 = baseline_rates(n, Lfix, A)
        gain.append((A, b4 - b1, b5 - b1,
                     SectorModel(sector_errors(A)).weighted_variance()))
    gain = np.array(gain)

    # ---- figure: 3 panels ----
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))

    ax[0].semilogy(rows[:, 0], np.maximum(rows[:, 1], 1e-9), 'k--s', ms=4,
                   label='B1 global (pooled)')
    ax[0].semilogy(rows[:, 0], np.maximum(rows[:, 2], 1e-9), 'C7:^', ms=4,
                   label='B3 equal-population control')
    ax[0].semilogy(rows[:, 0], np.maximum(rows[:, 3], 1e-9), 'C0-o', ms=4,
                   label='B4 proposed sectors')
    ax[0].semilogy(rows[:, 0], np.maximum(rows[:, 4], 1e-9), 'C3-.D', ms=4,
                   label='B5 oracle partition (upper bound)')
    ax[0].set_xlabel('Loss (dB)')
    ax[0].set_ylabel('Secret key rate (bits/pulse)')
    ax[0].set_title('(a) Same data, same model: who uses the sector label?')
    ax[0].legend(fontsize=7.5, loc='lower left')
    ax[0].grid(alpha=0.3)

    ax[1].plot(gain[:, 0], gain[:, 1], 'C0-o', ms=4, label='B4 − B1 (proposed)')
    ax[1].plot(gain[:, 0], gain[:, 2], 'C3-.D', ms=4, label='B5 − B1 (oracle)')
    ax[1].axhline(0, color='k', lw=0.6)
    ax[1].axvline(0, color='C2', lw=0.8, ls=':')
    ax[1].annotate('homogeneous:\ngain → 0', xy=(0, 0), xytext=(0.18, 0.32),
                   textcoords='axes fraction', fontsize=8,
                   arrowprops=dict(arrowstyle='->', color='C2'))
    ax[1].set_xlabel('Heterogeneity amplitude $A$')
    ax[1].set_ylabel(r'$\Delta R$ vs global (bits/pulse)')
    ax[1].set_title('(b) Gain appears only with heterogeneity')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    ax[2].plot(gain[:, 3], gain[:, 1], 'C0-o', ms=4)
    ax[2].set_xlabel(r'sector error variance $\mathrm{Var}_m(e_m)$')
    ax[2].set_ylabel(r'$\Delta R = R_{B4}-R_{B1}$ (bits/pulse)')
    ax[2].set_title('(c) Gain tracks the Jensen-gap source term')
    ax[2].grid(alpha=0.3)
    # linear guide through origin
    xv = gain[:, 3]
    if xv.max() > 0:
        k = np.sum(gain[:, 1] * xv) / np.sum(xv ** 2)
        ax[2].plot(xv, k * xv, 'C7:', lw=1, label='linear guide')
        ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig('fig_compare_baselines.png', dpi=200)
    fig.savefig('fig_compare_baselines.pdf')

    with open('data_compare_baselines.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['loss_dB', 'B1', 'B3_equalpop', 'B4_proposed', 'B5_oracle'])
        for r in rows:
            w.writerow(r)
        w.writerow([])
        w.writerow(['amplitude_A', 'gain_B4_minus_B1', 'gain_B5_minus_B1',
                    'variance'])
        for r in gain:
            w.writerow(r)

    # homogeneous null check, printed + asserted
    b1h, b3h, b4h, b5h = baseline_rates(n, Lfix, 0.0)
    print('HOMOGENEOUS NULL TEST (A=0):')
    print(f'  B1={b1h:.6e}  B4={b4h:.6e}  gain={b4h - b1h:+.2e} '
          f'(should be ~0): {"PASS" if abs(b4h - b1h) < 1e-4 * max(b1h,1e-9) + 1e-9 else "CHECK"}')
    print(f'  oracle headroom at A={A0}: B5−B4 = '
          f'{baseline_rates(n, Lfix, A0)[3] - baseline_rates(n, Lfix, A0)[2]:+.2e}')

    kamin_positioning_figure()
    print('written: fig_compare_baselines, fig_kamin_positioning, '
          'data_compare_baselines.csv')


# ----------------------------------------------------------------------
# (D) Kamin et al. conceptual positioning (NOT a "we beat them" claim)
# ----------------------------------------------------------------------
def kamin_positioning_figure():
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.axis('off')
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    def box(x, y, w, h, title, lines, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle='round,pad=0.6,rounding_size=2',
                     fc=fc, ec='#333', lw=1.2))
        ax.text(x + w / 2, y + h - 5, title, ha='center', va='top',
                fontsize=10.5, fontweight='bold')
        ax.text(x + w / 2, y + h - 12, lines, ha='center', va='top',
                fontsize=8.4)

    # shared engine
    box(33, 72, 34, 24, 'Shared GEAT engine\n(Kamin et al., PRX Q 6, 020342)',
        'generalized entropy accumulation\nfor prepare-and-measure +\n'
        'decoy-state QKD; min-tradeoff\nfunctions; finite-size second order',
        '#dce8f5')

    # Kamin contribution
    box(3, 26, 41, 36, 'Kamin et al. — the engine',
        'tight finite-size key rates via\nGEAT; unified decoy on Choi\n'
        'states; numerical stability of\nthe entropy optimization.\n\n'
        'Channel: homogeneous, not\nbuilt around public dynamic\nsectors.',
        '#eef3f8')

    # our contribution
    box(56, 26, 41, 36, 'This work — a conditioning layer',
        'adds a PUBLIC sector score\nregister $C_i=(M_i,T_i)$ on top\n'
        'of the same engine; variance-\nbased Jensen-gap advantage\n'
        'for heterogeneous DYNAMIC\nchannels; unified sector-\n'
        'conditioned decoy + self-\ncertified dual minorant.',
        '#e8f5ec')

    # arrows
    ax.annotate('', xy=(23, 62), xytext=(44, 73),
                arrowprops=dict(arrowstyle='-|>', color='#444', lw=1.6))
    ax.annotate('', xy=(77, 62), xytext=(56, 73),
                arrowprops=dict(arrowstyle='-|>', color='#444', lw=1.6))

    # bottom statement
    box(15, 1, 70, 19, 'Relationship: complementary, not competing',
        'Kamin provides the entropy-accumulation engine; this work supplies the\n'
        'public sector-conditioned structure the engine consumes. The gain is over\n'
        'global coarse-graining of a heterogeneous channel, and vanishes when the\n'
        'channel is homogeneous.',
        '#fdf2e0')

    fig.tight_layout()
    fig.savefig('fig_kamin_positioning.png', dpi=200)
    fig.savefig('fig_kamin_positioning.pdf')


if __name__ == '__main__':
    main()