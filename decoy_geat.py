# -*- coding: utf-8 -*-
"""
decoy_geat.py -- THE CLOSING STEP: full GEAT layer on dual-extracted
sector-conditioned decoy min-trade-off functions.

Pipeline per distance:
  1. per sector m: unified_sector_dual() -> certified affine crossover
     min-trade-off function f_m on the (mu, input, outcome) test cells
     (validated minorant; numerical-tolerance margin tau subtracted);
  2. joint sector-conditioned function f_sc: gradient component on the
     joint score symbol (m, mu, input, outcome) is
        lambda_m[mu][cell] / (p_m * p_mu * 1/2),
     offsets fold into the honest value (Proposition: affineness);
  3. GEAT finite-size layer (same secure machinery as the qubit case):
     range over components / gamma, K(alpha), eps split; gamma and
     alpha optimized per point;
  4. baselines: B4 = per-sector duals; B1 = single dual at pooled error.

Output: fig_decoy_geat.(png|pdf), data_decoy_geat.csv
"""
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from unified_decoy import kappa_family, sector_observed
from sector_geat import key_length, h2, ALPHA_GRID

TAU = 2e-5                      # numerical-tolerance margin (solver accuracy)
E8 = np.array([0.005, 0.010, 0.015, 0.022, 0.032, 0.045, 0.060, 0.080])
INTENS = (0.9, 2e-2, 1e-3)
P_MU = np.array([0.8, 0.15, 0.05])
ETA_DET, ALPHA_FIB, P_DARK = 0.6, 0.2, 1e-7
F_EC = 1.16
GAMMAS = np.logspace(-3, np.log10(0.4), 14)


def build_sector_functions(eta, n, e_list, gamma0=0.1):
    """width-regularized certified family f_m(kappa) per sector."""
    return [kappa_family(eta, P_DARK, e, fw_iters=28) for e in e_list]


def geat_rate_from_duals(n, fams, p_m, eta):
    """optimize (kappa, gamma, alpha); fams: per-sector kappa families."""
    M = len(fams)
    nk = min(len(f) for f in fams)
    best = 0.0
    for ki in range(nk):
        dds = [f[ki] for f in fams]
        p1 = dds[0]['p1']
        for gamma in GAMMAS:
            pref = (1 - gamma) ** 2 * p1
            comps, g_hon = [], 0.0
            for m, dd in enumerate(dds):
                if dd['value_hon'] <= 0:
                    g_hon = -1; break
                g_hon += p_m[m] * pref * dd['value_hon']
                for j, mu in enumerate(INTENS):
                    scale = pref / (p_m[m] * P_MU[j] * 0.5)
                    comps.extend((dd['lam'][mu] * scale).tolist())
            if g_hon <= 0:
                continue
            comps.append(0.0)
            rng = float(np.max(comps) - np.min(comps))
            lEC = 0.0
            for m, dd in enumerate(dds):
                o = dd['qhat'][INTENS[0]][:3]
                Q = o[0] + o[1]
                E = o[0] / max(Q, 1e-12)
                lEC += p_m[m] * Q * h2(min(max(E, 1e-6), 0.5))
            lEC = F_EC * n * (1 - gamma) ** 2 * lEC
            ls = key_length(n, g_hon, rng, gamma, ALPHA_GRID, lEC)
            r = float(np.max(ls)) / n
            if r > best:
                best = r
    return best


def main():
    n = 1e12
    dists = np.array([0, 10, 20, 30, 40, 50, 60, 70])
    M = len(E8)
    p_m = np.full(M, 1 / M)
    r4, r1 = [], []
    for L in dists:
        eta = ETA_DET * 10 ** (-ALPHA_FIB * L / 10)
        ebar = float(np.mean(E8))
        fam1 = build_sector_functions(eta, n, [ebar])
        rB1 = geat_rate_from_duals(n, fam1, np.array([1.0]), eta)
        best4 = rB1                      # M'=1 coarsening = B1 candidate
        for Mt in (2, 4):
            e_c = E8.reshape(Mt, -1).mean(axis=1)
            famc = build_sector_functions(eta, n, list(e_c))
            r = geat_rate_from_duals(n, famc, np.full(Mt, 1 / Mt), eta)
            best4 = max(best4, r)
        r4.append(best4); r1.append(rB1)
        print(f'L={L:>3.0f} km: B4*(decoy-GEAT, adaptive)={r4[-1]:.6f}  '
              f'B1={r1[-1]:.6f}  gain={r4[-1] - r1[-1]:+.2e}')
    r4 = np.array(r4); r1 = np.array(r1)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].semilogy(dists, np.maximum(r4, 1e-9), 'b-o', ms=4,
                   label=r'B4 sector-conditioned (full decoy-GEAT), $n=10^{12}$')
    ax[0].semilogy(dists, np.maximum(r1, 1e-9), 'r--s', ms=4,
                   label=r'B1 global (same pipeline)')
    ax[0].set_xlabel('Distance (km)')
    ax[0].set_ylabel('Key rate (bits per emitted pulse)')
    ax[0].set_title('Full GEAT layer on dual-extracted decoy functions')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(dists, np.maximum(r4 - r1, 0), 'g-^', ms=4)
    ax[1].set_xlabel('Distance (km)')
    ax[1].set_ylabel(r'$\Delta R$ (bits/pulse)')
    ax[1].set_title('Sector gain under the full decoy-GEAT pipeline')
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('fig_decoy_geat.png', dpi=200)
    fig.savefig('fig_decoy_geat.pdf')

    with open('data_decoy_geat.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['km', 'B4_decoy_geat', 'B1_decoy_geat'])
        for d, a, b in zip(dists, r4, r1):
            w.writerow([d, a, b])
    print('written: fig_decoy_geat, data_decoy_geat.csv')


if __name__ == '__main__':
    main()
