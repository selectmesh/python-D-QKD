# -*- coding: utf-8 -*-
"""
run_selection.py -- adds SECTOR SELECTION to the dynamic scenario:
sectors whose certified error is too high are publicly sifted out
(their rounds carry no key and no error-correction cost; their test
statistics are still collected). The kept set, like the partition, is a
public post-processing choice fixed from the calibrated honest model.

B4** := max over (coarsening, kept-prefix, gamma, alpha).

This is the strongest honest form of the manuscript's claim: the global
analysis is poisoned by the bad sectors of the dynamic channel, while
selection keeps only the good ones. We report whatever extension the
declared model actually gives.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

from sector_geat import (SectorModel, keyrate_point, key_length,
                         h2, h2prime, E_FLOOR, E_CEIL,
                         ALPHA_GRID, GAMMA_GRID)

alpha_fib = 0.2
e0 = np.array([0.004, 0.006, 0.008, 0.010, 0.014, 0.018, 0.024, 0.030])
kappa = np.array([0.00005, 0.0001, 0.0002, 0.0005,
                  0.002, 0.004, 0.007, 0.010])


def keyrate_selected(n, loss_dB, e_m, p_m=None):
    """max over kept-prefix (e ascending) of the sector-selected rate."""
    M = len(e_m)
    p_m = np.full(M, 1.0 / M) if p_m is None else p_m
    order = np.argsort(e_m)
    e_s, p_s = e_m[order], p_m[order]
    p_det = 10.0 ** (-loss_dB / 10.0)
    best = 0.0
    for keep in range(1, M + 1):
        ek = np.clip(e_s[:keep], E_FLOOR, E_CEIL)
        pk = p_s[:keep]
        for gamma in GAMMA_GRID:
            pref = (1 - gamma) ** 2
            # perspective tangents on kept sectors only
            g_err = pref * (1 - h2(ek) - (1 - ek) * h2prime(ek))
            g_ok = pref * (1 - h2(ek) + ek * h2prime(ek))
            comps = np.concatenate([g_err, g_ok, [0.0]])
            rng = float(np.max(comps) - np.min(comps))
            g_hon = pref * p_det * float(pk @ (1 - h2(ek)))
            if g_hon <= 0:
                continue
            lEC = 1.16 * n * pref * p_det * float(pk @ h2(ek))
            ls = key_length(n, g_hon, rng, gamma, ALPHA_GRID, lEC)
            m = float(np.max(ls)) / n
            if m > best:
                best = m
    return best


dists = np.linspace(0, 220, 56)
n_dyn = 1e12
rB1, rB4sel = [], []
for L in dists:
    e_L = np.clip(e0 + kappa * L, 1e-4, 0.499)
    rB1.append(keyrate_point(n_dyn, alpha_fib * L, SectorModel(e_L), 'B1'))
    rB4sel.append(keyrate_selected(n_dyn, alpha_fib * L, e_L))
rB1 = np.array(rB1); rB4sel = np.array(rB4sel)

L1 = dists[rB1 > 0][-1] if (rB1 > 0).any() else 0.0
L4 = dists[rB4sel > 0][-1] if (rB4sel > 0).any() else 0.0

eta = 10 ** (-alpha_fib * dists / 10.0)
plob = np.where(eta < 1, -np.log2(np.maximum(1 - eta, 1e-300)), np.inf)

fig, ax = plt.subplots(figsize=(6.6, 4.6))
ax.semilogy(dists, np.maximum(rB4sel, 1e-10), 'b-', lw=1.8,
            label=r'B4$^{**}$ sector-conditioned + selection, $n=10^{12}$')
ax.semilogy(dists, np.maximum(rB1, 1e-10), 'r--', lw=1.6,
            label=r'B1 global, $n=10^{12}$')
ax.semilogy(dists, plob, 'k:', label='PLOB bound')
ax.axvline(L1, color='r', ls=':', alpha=0.5)
ax.axvline(L4, color='b', ls=':', alpha=0.5)
ax.annotate('global abort\n%.0f km' % L1, (L1, 3e-9), color='r',
            fontsize=8, ha='right')
ax.annotate('sector abort\n%.0f km' % L4, (L4, 3e-9), color='b',
            fontsize=8, ha='left')
ax.set_ylim(1e-9, 10)
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Key rate (bits per emitted pulse)')
ax.set_title('Dynamic channel: sector selection vs global analysis')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('fig_dynamic_selection.png', dpi=200)
fig.savefig('fig_dynamic_selection.pdf')
plt.close(fig)

with open('data_dynamic_selection.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['distance_km', 'B4_selected', 'B1_global', 'PLOB'])
    for i, d in enumerate(dists):
        w.writerow([d, rB4sel[i], rB1[i],
                    plob[i] if np.isfinite(plob[i]) else ''])

viol = bool(np.any(np.maximum(rB4sel, 0) > plob + 1e-15))
out = f"""SECTOR-SELECTION DYNAMIC RESULT (n=1e12, declared kappa model)
  B1 global abort distance        : {L1:.0f} km
  B4** selection abort distance   : {L4:.0f} km
  certified extension             : {L4 - L1:.0f} km
  PLOB violated                   : {viol}
"""
print(out)
open('summary_selection.txt', 'w').write(out)
