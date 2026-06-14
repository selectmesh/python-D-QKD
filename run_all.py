# -*- coding: utf-8 -*-
"""
run_all.py -- executes Steps 1-2 end to end and produces:
  fig_qubit_B1_B3_B4.(png|pdf)   key rate vs loss, n = 1e8/1e10/1e12
  fig_gain_vs_M.(png|pdf)        finite-size sector gain vs sector count M
  fig_decoy_sectorized.(png|pdf) WCP decoy-state sectorized rates vs distance
  data_qubit.csv, data_decoy.csv numerical data behind the figures
  console report: Jensen gap vs variance lower bound check
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

from sector_geat import (SectorModel, keyrate_point, asymptotic_rate,
                         h2, key_length, crossover_tradeoff_sectorized,
                         lambda_EC_sectorized, ALPHA_GRID, GAMMA_GRID)
from decoy_sector_lp import decoy_sector_inputs
from sector_geat import h2prime, E_FLOOR, E_CEIL

# ----------------------------------------------------------------------
# Heterogeneous sector profile (M = 8), calibrated phenomenological model
# as declared in the manuscript's Numerical-status statement:
# misalignment angles spread so that e_m in [0.5%, 8%], mean ~ 3.2%
# ----------------------------------------------------------------------
M = 8
e_m = np.array([0.005, 0.010, 0.015, 0.022, 0.032, 0.045, 0.060, 0.080])
model = SectorModel(e_m)

print('=' * 64)
print('SECTOR PROFILE  (M = %d)' % M)
print('e_m       :', np.array2string(model.e_m, precision=3))
print('e_bar     : %.4f' % model.e_bar)
gap = model.jensen_gap()
var = model.weighted_variance()
lb = (2.0 / np.log(2)) * var
print('Jensen gap         h2(ebar)-sum p h2(e) = %.5f bits' % gap)
print('Variance lower bd  (2/ln2) Var_m(e_m)   = %.5f bits' % lb)
print('Strong-concavity check: gap >= bound ->', gap >= lb)
print('=' * 64)

# ----------------------------------------------------------------------
# FIGURE 1: qubit sectorized BB84, B1 vs B3 vs B4, three block sizes
# ----------------------------------------------------------------------
losses = np.linspace(0, 40, 41)
ns = [1e8, 1e10, 1e12]
colors = {1e8: 'tab:red', 1e10: 'tab:orange', 1e12: 'tab:blue'}

results = {}
for n in ns:
    for mode in ['B1', 'B4']:
        results[(n, mode)] = np.array(
            [keyrate_point(n, L, model, mode) for L in losses])
    print('n = %.0e done' % n)
asym_B4 = np.array([asymptotic_rate(L, model, 'B4') for L in losses])
asym_B1 = np.array([asymptotic_rate(L, model, 'B1') for L in losses])

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
for n in ns:
    ax[0].semilogy(losses, np.maximum(results[(n, 'B4')], 1e-9), '-',
                   color=colors[n], label=r'B4 sector, $n=10^{%d}$'
                   % int(np.log10(n)))
    ax[0].semilogy(losses, np.maximum(results[(n, 'B1')], 1e-9), '--',
                   color=colors[n], alpha=0.65,
                   label=r'B1 global, $n=10^{%d}$' % int(np.log10(n)))
ax[0].semilogy(losses, np.maximum(asym_B4, 1e-9), 'k-', lw=1,
               label='B4 asymptotic')
ax[0].semilogy(losses, np.maximum(asym_B1, 1e-9), 'k--', lw=1,
               label='B1 asymptotic')
ax[0].set_ylim(1e-7, 1)
ax[0].set_xlabel('Loss (dB)')
ax[0].set_ylabel('Key rate (bits per emitted pulse)')
ax[0].set_title('Sector-conditioned GEAT vs global analysis (qubit BB84)')
ax[0].legend(fontsize=7, ncol=2)
ax[0].grid(alpha=0.3)

# gain panel
for n in ns:
    gain = results[(n, 'B4')] - results[(n, 'B1')]
    ax[1].plot(losses, np.maximum(gain, 0), color=colors[n],
               label=r'$n=10^{%d}$' % int(np.log10(n)))
ax[1].plot(losses, asym_B4 - asym_B1, 'k-', lw=1, label='asymptotic')
ax[1].set_xlabel('Loss (dB)')
ax[1].set_ylabel(r'$\Delta R = R_{B4} - R_{B1}$')
ax[1].set_title('Finite-size sector-conditioning gain')
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig('fig_qubit_B1_B3_B4.png', dpi=200)
fig.savefig('fig_qubit_B1_B3_B4.pdf')
plt.close(fig)

with open('data_qubit.csv', 'w', newline='') as f:
    w = csv.writer(f)
    head = ['loss_dB']
    for n in ns:
        head += ['B4_n1e%d' % int(np.log10(n)), 'B1_n1e%d' % int(np.log10(n))]
    head += ['B4_asym', 'B1_asym']
    w.writerow(head)
    for i, L in enumerate(losses):
        row = [L]
        for n in ns:
            row += [results[(n, 'B4')][i], results[(n, 'B1')][i]]
        row += [asym_B4[i], asym_B1[i]]
        w.writerow(row)

# ----------------------------------------------------------------------
# FIGURE 2: gain vs sector count M at fixed loss (the finite-key
# optimum of M -- 'Limitations' item three, computed not asserted)
# ----------------------------------------------------------------------
loss0 = 10.0
Ms = [1, 2, 4, 8, 16, 32]
n0 = 1e10
gains, ratesB4 = [], []
# continuous error landscape e(x) on [0,1] interpolating the M=8 profile;
# bin-AVERAGING preserves the mean exactly, so e_bar is identical for all M
xs_fine = np.linspace(0, 1, 8000)
e_fine = np.interp(xs_fine, (np.arange(8) + 0.5) / 8.0, e_m)
for Mtest in Ms:
    e_test = e_fine.reshape(Mtest, -1).mean(axis=1)
    mtest = SectorModel(e_test)
    rB4 = keyrate_point(n0, loss0, mtest, 'B4')
    rB1 = keyrate_point(n0, loss0, mtest, 'B1')
    ratesB4.append(rB4)
    gains.append(rB4 - rB1)
    print('M = %2d  R_B4 = %.4e  gain = %.4e' % (Mtest, rB4, rB4 - rB1))

fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.plot(Ms, gains, 'o-', color='tab:green')
ax.set_xscale('log', base=2)
ax.set_xlabel('Number of sectors $M$')
ax.set_ylabel(r'Finite-size gain $\Delta R$ (bits/pulse)')
ax.set_title(r'Sector-count optimum, $n=10^{10}$, loss $=10$ dB')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('fig_gain_vs_M.png', dpi=200)
fig.savefig('fig_gain_vs_M.pdf')
plt.close(fig)

# ----------------------------------------------------------------------
# FIGURE 3: WCP decoy-state sectorized rates (sector-conditioned LPs
# feeding the same GEAT layer) -- Step 1 decoy branch
# ----------------------------------------------------------------------
intensities = (0.9, 2e-2, 1e-3)            # Kamin et al. choice
p_int = {0.9: 0.5, 2e-2: 0.25, 1e-3: 0.25}
alpha_fib = 0.2                            # dB/km
eta_det = 0.6
p_dark = 1e-7

dists = np.linspace(0, 150, 31)
n_dec = 1e12
gamma_dec_grid = np.logspace(-3, np.log10(0.5), 12)

rate_dec_B4, rate_dec_B1 = [], []
for Lkm in dists:
    eta = eta_det * 10 ** (-alpha_fib * Lkm / 10.0)
    best4 = best1 = 0.0
    for gamma in gamma_dec_grid:
        w1, e1, ok = decoy_sector_inputs(n_dec, gamma, model, eta,
                                         p_dark, intensities, p_int)
        if not ok.all():
            continue
        sift = (1 - gamma) ** 2
        # --- B4: sector-conditioned tangents on single-photon entropy
        e1c = np.clip(e1, E_FLOOR, E_CEIL)
        g_hon4 = sift * float(np.sum(w1 * (1 - h2(e1c))))
        rng4 = sift * float(np.max(h2prime(e1c)))
        # EC on the full sifted signal gain (signal intensity, all photons)
        from decoy_sector_lp import channel_observables
        Qs = np.array([channel_observables(0.9, eta, p_dark, em)
                       for em in model.e_m])
        Qsig = float(model.p_m @ Qs[:, 0])
        Esig = float(model.p_m @ Qs[:, 1]) / max(Qsig, 1e-15)
        lEC = 1.16 * n_dec * sift * Qsig * h2(min(Esig, 0.5))
        if g_hon4 > 0:
            l4 = float(np.max(key_length(n_dec, g_hon4, rng4, gamma,
                                         ALPHA_GRID, lEC)))
            best4 = max(best4, l4 / n_dec)
        # --- B1: pooled single-photon bound (one LP on pooled stats)
        w1p = float(np.sum(w1))
        e1p = float(np.clip(np.sum(w1 * e1) / max(w1p, 1e-15),
                            E_FLOOR, E_CEIL))
        g_hon1 = sift * w1p * (1 - h2(e1p))
        rng1 = sift * h2prime(e1p)
        if g_hon1 > 0:
            l1 = float(np.max(key_length(n_dec, g_hon1, rng1, gamma,
                                         ALPHA_GRID, lEC)))
            best1 = max(best1, l1 / n_dec)
    rate_dec_B4.append(best4)
    rate_dec_B1.append(best1)
rate_dec_B4 = np.array(rate_dec_B4)
rate_dec_B1 = np.array(rate_dec_B1)
print('decoy branch done')

# PLOB overlay (bits per pulse, point-to-point)
eta_ch = eta_det * 10 ** (-alpha_fib * dists / 10.0)
plob = -np.log2(1 - eta_ch)

fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.semilogy(dists, np.maximum(rate_dec_B4, 1e-10), 'b-',
            label=r'B4 sector-conditioned, $n=10^{12}$')
ax.semilogy(dists, np.maximum(rate_dec_B1, 1e-10), 'b--', alpha=0.7,
            label=r'B1 global, $n=10^{12}$')
ax.semilogy(dists, plob, 'k:', label='PLOB bound')
ax.set_ylim(1e-9, 10)
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Key rate (bits per emitted pulse)')
ax.set_title('WCP decoy-state D-QKD, sector-conditioned LPs + GEAT layer')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('fig_decoy_sectorized.png', dpi=200)
fig.savefig('fig_decoy_sectorized.pdf')
plt.close(fig)

with open('data_decoy.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['distance_km', 'B4', 'B1', 'PLOB'])
    for i, d in enumerate(dists):
        w.writerow([d, rate_dec_B4[i], rate_dec_B1[i], plob[i]])

print('ALL FIGURES AND DATA WRITTEN')
