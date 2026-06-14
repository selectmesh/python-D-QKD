# -*- coding: utf-8 -*-
"""
sector_geat.py
==============
Sector-conditioned GEAT finite-size key rates for the geometric-phase
sectorized BB84 protocol (D-QKD), implementing Steps 1-2 of the plan:

  Step 1: build a valid affine (crossover) min-trade-off function f_m for
          EVERY sector m, and the weighted sector-conditioned function
          f_sc = sum_m q_M(m) f_m   (Definition `sector_conditioned_tradeoff`
          of the revised manuscript).

  Step 2: evaluate the full finite-size key length
              l = n*[ f_sc(q_hon) - second-order GEAT penalty ]
                  - n*((a-1)/(2-a))^2 K(a) - lambda_EC
                  - log2(1/eps_EV) - a/(a-1) log2(1/eps_PA) + 2
          [Kamin et al., PRX Quantum 6, 020342 (2025), Eqs. (11),(16),
           (35)-(39),(43)-(45)], with alpha and gamma optimized per point,
          and produce the baseline comparison B1 / B3 / B4.

RIGOR STATUS (read carefully -- this is what goes into the paper):
  * The min-trade-off functions are built as TANGENT LINES to the convex
    single-round entropy bound  W(e) = c * (1 - h2(e))  of the qubit
    BB84 reference sector (Shor-Preskill / Renner bound, tight for BB84).
    A tangent to a convex function is a VALID affine lower bound, hence a
    VALID crossover min-trade-off function.  This is the same construction
    principle as in Dupuis-Fawzi and Metger-Renner worked examples.
  * The second-order GEAT terms use the SECURE lower bound of
    Kamin et al. Eq. (43):
        f(p_hon)+T_a(f) >= f(p_hon)
           - (a-1)/(2-a) * ln2/2 * ( log2(1+2 d_A) + sqrt(2 + range^2/g) )^2
    which is a rigorous (slightly loose) replacement for the exact T_a(f)
    optimization.  Replacing it with the Frank-Wolfe SDP of Kamin et al.
    can only IMPROVE the curves; it cannot invalidate them.
  * Unique acceptance (Omega_com = 0), kappa = 1 (classical key register),
    d_A = 2, as in Fig. 1 / Fig. 3 of Kamin et al.
  * Assumption used to make f_m affine in the observed frequencies:
    the public sector probabilities p_m and the certified detection
    probability p_det are treated as model constants, so that
    e_m(q) = q(m,err)/(p_m * p_det * gamma-normalization) is affine in q.
    This is stated explicitly in the methods text.

Author: generated for N. A. Alwan et al., D-QKD manuscript revision.
"""

import numpy as np

LN2 = np.log(2.0)
E_FLOOR = 1e-4          # floor on error rates (h2' diverges at 0)
E_CEIL = 0.5 - 1e-9


# ----------------------------------------------------------------------
# elementary entropy functions
# ----------------------------------------------------------------------
def h2(e):
    """Binary entropy, base 2, safe at the boundary."""
    e = np.clip(e, 1e-15, 1 - 1e-15)
    return -e * np.log2(e) - (1 - e) * np.log2(1 - e)


def h2prime(e):
    """d h2 / d e = log2((1-e)/e)."""
    e = np.clip(e, E_FLOOR, E_CEIL)
    return np.log2((1 - e) / e)


# ----------------------------------------------------------------------
# GEAT second-order machinery  [Kamin et al. Eqs. (11), (43)-(45)]
# ----------------------------------------------------------------------
def K_alpha(alpha, range_f_plus_logdA):
    """
    K(alpha) of Kamin et al. Eq. (11), kappa = 1.
    `range_f_plus_logdA` = kappa*log2(d_A) + max(f) - min_Q(f) =: x.
    """
    a = alpha
    x = range_f_plus_logdA
    pref = (2 - a) ** 3 / (6 * (3 - 2 * a) ** 3 * LN2)
    ex = ((a - 1) / (2 - a)) * x
    ex = np.minimum(ex, 500.0)              # cap: beyond this K(a) kills l anyway
    expo = 2.0 ** ex
    xl = np.minimum(x, 500.0)
    return pref * expo * (np.log(2.0 ** xl + np.e ** 2)) ** 3


def secure_first_plus_second_order(g_hon, range_g, gamma, alpha, dA=2):
    """
    Rigorous lower bound on  f(p_hon) + T_alpha(f)
    [Kamin et al. Eq. (43)], using only the crossover gradient range:

        >= g(q_hon) - (a-1)/(2-a) * ln2/2 *
           ( log2(1 + 2*dA) + sqrt(2 + range_g^2 / gamma) )^2
    """
    a = alpha
    V = (np.log2(1 + 2 * dA) + np.sqrt(2.0 + range_g ** 2 / gamma)) ** 2
    return g_hon - ((a - 1) / (2 - a)) * (LN2 / 2.0) * V


def key_length(n, g_hon, range_g, gamma, alpha, lambda_EC,
               eps_secure=1e-8, dA=2):
    """
    Full finite-size key length, Kamin et al. Eq. (16) with the Eq. (43)
    bound, eps split per Eq. (57):
        eps_PA = a/(2a-1)*eps_sec ,  eps_EV = (a-1)/(2a-1)*eps_sec .
    Returns l (can be negative -> clip outside).
    """
    a = alpha
    eps_PA = (a / (2 * a - 1)) * eps_secure
    eps_EV = ((a - 1) / (2 * a - 1)) * eps_secure

    first = secure_first_plus_second_order(g_hon, range_g, gamma, a, dA)

    # range of the FULL min-trade-off f from the crossover g
    # [Kamin et al. Eq. (37)]:  max f - min f = range_g / gamma
    x = 1.0 + range_g / gamma          # kappa*log2(dA)=1 for dA=2
    Ka = K_alpha(a, x)

    l = (n * first
         - n * ((a - 1) / (2 - a)) ** 2 * Ka
         - lambda_EC
         - np.log2(1.0 / eps_EV)
         - (a / (a - 1)) * np.log2(1.0 / eps_PA)
         + 2.0)
    return l


# ----------------------------------------------------------------------
# Sectorized qubit BB84 model  (Sec. VI structure of Kamin et al.,
# sector layer of the D-QKD manuscript)
# ----------------------------------------------------------------------
class SectorModel:
    """
    M public geometric sectors with weights p_m and per-sector phase/bit
    error rates e_m (calibrated phenomenological model, as declared in the
    'Numerical-status statement' of the revised manuscript).
    """

    def __init__(self, e_m, p_m=None):
        self.e_m = np.asarray(e_m, dtype=float)
        self.M = len(self.e_m)
        self.p_m = (np.full(self.M, 1.0 / self.M) if p_m is None
                    else np.asarray(p_m, dtype=float))
        assert abs(self.p_m.sum() - 1.0) < 1e-12

    @property
    def e_bar(self):
        return float(self.p_m @ self.e_m)

    def jensen_gap(self):
        """h2(e_bar) - sum_m p_m h2(e_m)  >= 0 (strict if heterogeneous)."""
        return h2(self.e_bar) - float(self.p_m @ h2(self.e_m))

    def weighted_variance(self):
        return float(self.p_m @ (self.e_m - self.e_bar) ** 2)


def crossover_tradeoff_sectorized(model, p_det, gamma, anchors=None):
    """
    STEP 1 -- build f_m for every sector and combine into f_sc.

    LOSS-ADVERSARIAL (perspective) tangent construction: the entropy of
    the sifted generation rounds is bounded, as a function of the
    test-conditional score distribution q over
    {(m,err),(m,ok),(m,nodet)}, by the convex PERSPECTIVE form
        W(q) = (1-gamma)^2 * sum_m phi(q_err^m, q_ok^m),
        phi(a,b) = (a+b) * (1 - h2(a/(a+b))) ,
    [phi = perspective of the convex map t -> 1 - h2(t), hence convex;
     lost rounds contribute zero entropy, so no certified detection
     probability is assumed -- Eve controls the loss adversarially].
    Its tangent at the honest point is a VALID affine crossover
    min-trade-off function.  Gradient components, with t = a/(a+b):
        d phi/da = 1 - h2(t) - (1-t) h2'(t)
        d phi/db = 1 - h2(t) + t     h2'(t)
        d/d(nodet) = 0.

    Returns (g_hon, range_g):
        g_hon   : value at the honest point (= exact entropy bound there),
        range_g : max_c g(delta_c) - min_c g(delta_c) over ALL components.
    """
    ehat = model.e_m if anchors is None else np.asarray(anchors, float)
    t = np.clip(ehat, E_FLOOR, E_CEIL)
    pref = (1.0 - gamma) ** 2

    g_err = pref * (1.0 - h2(t) - (1.0 - t) * h2prime(t))
    g_ok = pref * (1.0 - h2(t) + t * h2prime(t))
    comps = np.concatenate([g_err, g_ok, np.zeros_like(g_err)])

    # tangent touches at the honest point -> exact value there:
    g_hon = pref * p_det * float(model.p_m @ (1.0 - h2(np.clip(model.e_m,
                                                E_FLOOR, E_CEIL))))
    range_g = float(np.max(comps) - np.min(comps))
    return g_hon, range_g


def crossover_tradeoff_pooled(model, p_det, gamma):
    """
    Baseline B1 -- one perspective tangent at the pooled error e_bar.
    g_hon is SMALLER by exactly  sift * JensenGap  (the manuscript's gain).
    """
    t = np.clip(model.e_bar, E_FLOOR, E_CEIL)
    pref = (1.0 - gamma) ** 2
    g_err = pref * (1.0 - h2(t) - (1.0 - t) * h2prime(t))
    g_ok = pref * (1.0 - h2(t) + t * h2prime(t))
    comps = np.array([g_err, g_ok, 0.0])
    g_hon = pref * p_det * (1.0 - h2(t))
    range_g = float(np.max(comps) - np.min(comps))
    return g_hon, range_g


def lambda_EC_sectorized(n, model, p_det, gamma, f_EC=1.16):
    """Sector-wise error correction (Kamin et al. Eq. (26) convention)."""
    sift = (1.0 - gamma) ** 2 * p_det
    return f_EC * n * sift * float(model.p_m @ h2(np.clip(model.e_m,
                                                          1e-12, 0.5)))


def lambda_EC_pooled(n, model, p_det, gamma, f_EC=1.16):
    sift = (1.0 - gamma) ** 2 * p_det
    return f_EC * n * sift * h2(model.e_bar)


# ----------------------------------------------------------------------
# STEP 2 -- optimized finite-size key rate per (n, loss) point
# ----------------------------------------------------------------------
ALPHA_GRID = 1.0 + np.logspace(-7, np.log10(0.49), 60)
GAMMA_GRID = np.logspace(-4, np.log10(0.5), 40)


def keyrate_point(n, loss_dB, model, mode='B4', eps_secure=1e-8,
                  eta_det=1.0):
    """
    Optimize alpha and gamma on grids; return best key rate l/n (>=0).
    mode: 'B4' sector-conditioned | 'B1' pooled | 'B3' random binning
          (B3: M public bins, each carrying the SAME mixture -> every bin
           anchor is e_bar; statistically it reproduces B1 by construction,
           which is exactly the control the benchmark section demands).
    """
    p_det = eta_det * 10.0 ** (-loss_dB / 10.0)
    best = 0.0
    for gamma in GAMMA_GRID:
        if mode == 'B4':
            g_hon, rng = crossover_tradeoff_sectorized(model, p_det, gamma)
            lEC = lambda_EC_sectorized(n, model, p_det, gamma)
        elif mode == 'B1':
            g_hon, rng = crossover_tradeoff_pooled(model, p_det, gamma)
            lEC = lambda_EC_pooled(n, model, p_det, gamma)
        elif mode == 'B3':
            bins = SectorModel(np.full(model.M, model.e_bar))
            g_hon, rng = crossover_tradeoff_sectorized(bins, p_det, gamma)
            lEC = lambda_EC_pooled(n, model, p_det, gamma)
        else:
            raise ValueError(mode)
        if g_hon <= 0:
            continue
        ls = key_length(n, g_hon, rng, gamma, ALPHA_GRID, lEC, eps_secure)
        m = float(np.max(ls))
        if m > best:
            best = m
    return best / n


def asymptotic_rate(loss_dB, model, mode='B4', eta_det=1.0):
    """Devetak-Winter limit of the same model (gamma -> 0, EC at f=1.16)."""
    p_det = eta_det * 10.0 ** (-loss_dB / 10.0)
    if mode == 'B4':
        W = p_det * float(model.p_m @ (1 - h2(model.e_m)))
        EC = 1.16 * p_det * float(model.p_m @ h2(model.e_m))
    else:
        W = p_det * (1 - h2(model.e_bar))
        EC = 1.16 * p_det * h2(model.e_bar)
    return max(W - EC, 0.0)
