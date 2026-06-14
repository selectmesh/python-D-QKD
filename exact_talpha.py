# -*- coding: utf-8 -*-
"""
exact_talpha.py -- ITEM 2: verification of the simplified second-order bound.

The production curves use the SECURE simplified bound [Kamin Eq. (43)].
Here we evaluate the EXACT first+second-order quantity [Kamin Eq. (41)]:

   f(p_hon) + T_alpha(f)
   = inf_{behaviors}  [ W(q) + g.(q_hon - q)
                        - (a-1)/(2-a) * ln2/2 * Vtilde(q, g) ]      (*)

with Vtilde from Kamin Eq. (39).  For the qubit sectorized protocol the
single-round behaviors are exactly parametrized by per-sector detection
probabilities d_m in [0,1] and error rates e_m in [0,1/2] (every such
pair is attainable, and the entropy of the worst state with those
statistics is the perspective value), so the inf reduces to a smooth
2M-dimensional minimization solved by multi-start L-BFGS-B.

Checks performed:
  (A) M-scan at 10 dB, n=1e10: is M* = 4 robust under the exact bound?
  (B) dynamic endpoints: do the 28 km (global abort) and 168 km
      (selected-sector survival) conclusions hold?
Exact >= simplified must hold at every point (the simplified bound is a
rigorous lower bound on (*)); any violation flags a bug.
"""
import numpy as np
from scipy.optimize import minimize
from sector_geat import (SectorModel, h2, h2prime, E_FLOOR, E_CEIL,
                         K_alpha, LN2, keyrate_point, GAMMA_GRID, ALPHA_GRID,
                         lambda_EC_sectorized, lambda_EC_pooled,
                         crossover_tradeoff_sectorized,
                         crossover_tradeoff_pooled, key_length)
from run_scenarios import coarsen, E8


def tangent_components(e_anchor, gamma):
    t = np.clip(e_anchor, E_FLOOR, E_CEIL)
    pref = (1 - gamma) ** 2
    g_err = pref * (1 - h2(t) - (1 - t) * h2prime(t))
    g_ok = pref * (1 - h2(t) + t * h2prime(t))
    g_nd = np.zeros_like(g_err)
    return g_err, g_ok, g_nd


def exact_first_second(model, p_det, gamma, alpha, n_starts=8, seed=0):
    """exact value of (*) for the sectorized tangent g at the honest point."""
    M = model.M
    pm = model.p_m
    g_err, g_ok, g_nd = tangent_components(model.e_m, gamma)
    gmax = max(float(np.max(g_ok)), 0.0)
    pref = (1 - gamma) ** 2
    a1 = (alpha - 1) / (2 - alpha)
    dA, kappa = 2, 1
    # honest conditional-on-test distribution and g.q_hon
    q_hon = np.concatenate([pm * p_det * model.e_m,
                            pm * p_det * (1 - model.e_m),
                            pm * (1 - p_det)])
    gvec = np.concatenate([g_err, g_ok, g_nd])

    def Vtilde(q):
        s = float(np.sum(q / gamma * (gmax - gvec) ** 2)
                  - (gmax - float(gvec @ q)) ** 2)
        return (np.log2(1 + 2 * dA ** kappa) + np.sqrt(max(2 + s, 0.0))) ** 2

    def F(x):
        d = 1 / (1 + np.exp(-x[:M]))                  # (0,1)
        e = 0.5 / (1 + np.exp(-x[M:]))                # (0,0.5)
        e = np.clip(e, E_FLOOR, E_CEIL)
        q = np.concatenate([pm * d * e, pm * d * (1 - e), pm * (1 - d)])
        W = pref * float(pm @ (d * (1 - h2(e))))
        return (W + float(gvec @ (q_hon - q))
                - a1 * (LN2 / 2) * Vtilde(q))

    rng = np.random.default_rng(seed)
    best = np.inf
    # start at honest + random starts
    x0h = np.concatenate([np.log(p_det / (1 - p_det)) * np.ones(M),
                          np.log(model.e_m / (0.5 - model.e_m))])
    starts = [x0h] + [rng.normal(0, 2, 2 * M) for _ in range(n_starts - 1)]
    for x0 in starts:
        r = minimize(F, x0, method='L-BFGS-B')
        if r.fun < best:
            best = float(r.fun)
    return best


def exact_keyrate(n, loss_dB, model, mode='B4', span=2):
    """key rate with EXACT (*) replacing the simplified bound; alpha,gamma
    re-optimized on a local grid around the simplified-bound optimum."""
    p_det = 10 ** (-loss_dB / 10)
    # locate simplified optimum (gamma index, alpha index)
    best_simpl, gi0, ai0 = -1, 0, 0
    for gi, gamma in enumerate(GAMMA_GRID):
        if mode == 'B4':
            g_hon, rng_ = crossover_tradeoff_sectorized(model, p_det, gamma)
            lEC = lambda_EC_sectorized(n, model, p_det, gamma)
        else:
            g_hon, rng_ = crossover_tradeoff_pooled(model, p_det, gamma)
            lEC = lambda_EC_pooled(n, model, p_det, gamma)
        ls = key_length(n, g_hon, rng_, gamma, ALPHA_GRID, lEC)
        ai = int(np.argmax(ls))
        if ls[ai] / n > best_simpl:
            best_simpl, gi0, ai0 = ls[ai] / n, gi, ai
    # exact evaluation on local grid
    best_exact = 0.0
    for gi in range(max(0, gi0 - span), min(len(GAMMA_GRID), gi0 + span + 1)):
        gamma = GAMMA_GRID[gi]
        if mode == 'B4':
            lEC = lambda_EC_sectorized(n, model, p_det, gamma)
            mdl = model
        else:
            lEC = lambda_EC_pooled(n, model, p_det, gamma)
            mdl = SectorModel(np.array([model.e_bar]))
        for ai in range(max(0, ai0 - span), min(len(ALPHA_GRID), ai0 + span + 1)):
            alpha = ALPHA_GRID[ai]
            fs = exact_first_second(mdl, p_det, gamma, alpha)
            g_err, g_ok, _ = tangent_components(mdl.e_m, gamma)
            comps = np.concatenate([g_err, g_ok, [0.0]])
            rng_ = float(np.max(comps) - np.min(comps))
            x = 1 + rng_ / gamma
            Ka = K_alpha(alpha, x)
            a1 = (alpha - 1) / (2 - alpha)
            eps = 1e-8
            eps_PA = alpha / (2 * alpha - 1) * eps
            eps_EV = (alpha - 1) / (2 * alpha - 1) * eps
            l = (n * fs - n * a1 ** 2 * Ka - lEC
                 - np.log2(1 / eps_EV) - alpha / (alpha - 1) * np.log2(1 / eps_PA) + 2)
            if l / n > best_exact:
                best_exact = l / n
    return best_simpl, best_exact


if __name__ == '__main__':
    print('=' * 66)
    print('(A) M-scan verification, 10 dB, n = 1e10  (simplified vs exact)')
    print('=' * 66)
    rows = []
    for Mt in [1, 2, 4, 8]:
        mdl = SectorModel(coarsen(E8, Mt))
        s, x = exact_keyrate(1e10, 10.0, mdl, 'B4')
        rows.append((Mt, s, x))
        print(f'  M={Mt}:  simplified={s:.6e}   exact={x:.6e}   '
              f'(exact-simpl={x - s:+.2e})')
    order_s = max(rows, key=lambda r: r[1])[0]
    order_x = max(rows, key=lambda r: r[2])[0]
    print(f'  M* (simplified) = {order_s},  M* (exact) = {order_x}')

    print('=' * 66)
    print('(B) dynamic-endpoint verification, n = 1e12')
    print('=' * 66)
    e0 = np.array([0.004, 0.006, 0.008, 0.010, 0.014, 0.018, 0.024, 0.030])
    kap = np.array([5e-5, 1e-4, 2e-4, 5e-4, 2e-3, 4e-3, 7e-3, 1e-2])
    for L, keep, label in [(30.0, 8, 'just past global abort (B1)'),
                           (160.0, 3, 'near selected-sector limit (B4**)')]:
        e_L = np.clip(e0 + kap * L, 1e-4, 0.499)
        order = np.argsort(e_L)
        ek = e_L[order][:keep]
        mdl = SectorModel(ek)            # kept sectors, equal weights
        # weight correction: kept fraction
        frac = keep / 8.0
        s, x = exact_keyrate(1e12, 0.2 * L, mdl, 'B4')
        print(f'  L={L:.0f} km ({label}): simplified={frac * s:.3e}  '
              f'exact={frac * x:.3e}  positive(exact)={frac * x > 0}')
    # B1 at 30 km must be non-positive
    e_L = np.clip(e0 + kap * 30.0, 1e-4, 0.499)
    s1, x1 = exact_keyrate(1e12, 0.2 * 30, SectorModel(e_L), 'B1')
    print(f'  L=30 km global B1: simplified={s1:.3e}  exact={x1:.3e}  '
          f'(abort confirmed: {x1 <= 1e-12})')
