# -*- coding: utf-8 -*-
"""
decoy_sector_lp.py
==================
Sector-conditioned decoy-state estimation (Subsec. `sector_decoy_constraints`
of the revised manuscript), implemented as the standard decoy linear
programs, SOLVED PER SECTOR with Hoeffding confidence intervals built from
the SECTOR populations N_m (this is precisely the 'hidden cost of geometric
resolution' the manuscript discusses).

For each sector m and each intensity mu in {mu_sig, mu_2, mu_3}:
    observed gain      Qhat_{m,mu}    (channel model below)
    observed error     Ehat_{m,mu}
    Hoeffding radius   delta = sqrt( ln(2/eps_PE_cell) / (2 N_{m,mu}) )

LP1 (yield):     minimize Y_1
                 s.t.  sum_n P_mu(n) Y_n  in [Qhat - delta, Qhat + delta]
                       0 <= Y_n <= 1,  n <= n_cut  (+ tail slack)
LP2 (error):     maximize b_1   (b_n := e_n Y_n)
                 s.t.  sum_n P_mu(n) b_n  in [EQhat - delta, EQhat + delta]
                       0 <= b_n <= Y_n^upper
Then  e1_upper = b1_upper / Y1_lower.

Channel model per sector (standard WCP/BB84, Wang-Lutkenhaus style):
    eta      = eta_det * 10^(-alpha_fib * L / 10)
    Y0       = 2 * p_dark
    Q_mu     = 1 - (1 - Y0) * exp(-eta * mu)
    E_mu Q_mu= e0 * Y0 + e_mis^(m) * (1 - exp(-eta * mu)),  e0 = 1/2.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.stats import poisson


def channel_observables(mu, eta, p_dark, e_mis):
    Y0 = 2.0 * p_dark
    Q = 1.0 - (1.0 - Y0) * np.exp(-eta * mu)
    EQ = 0.5 * Y0 + e_mis * (1.0 - np.exp(-eta * mu))
    return Q, min(EQ, Q)


def sector_decoy_bounds(eta, p_dark, e_mis, N_mu, intensities,
                        eps_cell=1e-10, n_cut=10):
    """
    Returns (Y1_lower, e1_upper, feasible) for ONE sector.
    N_mu: dict {mu: number of test rounds of this sector at intensity mu}.
    """
    mus = list(intensities)
    nph = np.arange(n_cut + 1)

    # confidence intervals from SECTOR statistics
    rows_lo, rows_hi = [], []
    P = {}
    for mu in mus:
        Q, EQ = channel_observables(mu, eta, p_dark, e_mis)
        d = np.sqrt(np.log(2.0 / eps_cell) / (2.0 * max(N_mu[mu], 1.0)))
        pn = poisson.pmf(nph, mu)
        tail = 1.0 - pn.sum()                    # photon-number tail
        P[mu] = (pn, tail, max(Q - d, 0.0), min(Q + d, 1.0),
                 max(EQ - d, 0.0), min(EQ + d, 1.0))

    # ---------- LP1: minimize Y1 ----------
    c = np.zeros(n_cut + 1); c[1] = 1.0
    A_ub, b_ub = [], []
    for mu in mus:
        pn, tail, Qlo, Qhi, _, _ = P[mu]
        A_ub.append(pn);  b_ub.append(Qhi)            # sum <= Qhi
        A_ub.append(-pn); b_ub.append(-(Qlo - tail))  # sum >= Qlo - tail
    res1 = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                   bounds=[(0, 1)] * (n_cut + 1), method='highs')
    if not res1.success:
        return 0.0, 0.5, False
    Y1_lo = max(res1.fun, 0.0)

    # upper bounds on yields (for LP2 box): maximize each Y_n is <=1; use 1.
    # ---------- LP2: maximize b1 ----------
    c2 = np.zeros(n_cut + 1); c2[1] = -1.0            # maximize -> minimize -b1
    A2, b2 = [], []
    for mu in mus:
        pn, tail, _, _, Elo, Ehi = P[mu]
        A2.append(pn);  b2.append(Ehi)
        A2.append(-pn); b2.append(-(Elo - tail))
    res2 = linprog(c2, A_ub=np.array(A2), b_ub=np.array(b2),
                   bounds=[(0, 1)] * (n_cut + 1), method='highs')
    if not res2.success:
        return Y1_lo, 0.5, False
    b1_hi = -res2.fun
    e1_hi = min(b1_hi / Y1_lo, 0.5) if Y1_lo > 1e-12 else 0.5
    return Y1_lo, e1_hi, Y1_lo > 1e-12


def decoy_sector_inputs(n, gamma, model, eta, p_dark, intensities, p_int,
                        eps_cell=1e-10):
    """
    Build the per-sector single-photon inputs (s1 fraction, e1_ph upper)
    that feed the sector-conditioned min-trade-off functions.
    Returns arrays (w1_m, e1_m, ok_m):
       w1_m = p_m * P_{mu_sig}(1) * Y1_lower_m  (single-photon sifted weight,
              before the (1-gamma)^2 sifting factor),
       e1_m = phase-error upper bound of sector m.
    """
    M = model.M
    w1 = np.zeros(M); e1 = np.zeros(M); ok = np.zeros(M, bool)
    mu_sig = intensities[0]
    P1_sig = poisson.pmf(1, mu_sig)
    for m in range(M):
        N_mu = {mu: n * gamma * model.p_m[m] * p_int[mu]
                for mu in intensities}
        Y1, e1m, good = sector_decoy_bounds(eta, p_dark, model.e_m[m],
                                            N_mu, intensities, eps_cell)
        w1[m] = model.p_m[m] * P1_sig * Y1
        e1[m] = e1m
        ok[m] = good
    return w1, e1, ok
