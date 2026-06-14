# -*- coding: utf-8 -*-
"""
unified_decoy.py -- ITEM 1 (the decisive step): the UNIFIED sector-conditioned
decoy-state construction, merging the decoy linear constraints and the
single-photon entropy objective into ONE convex optimization over the
single-photon Choi state, in the style of Kamin et al. (PRX Quantum 6,
020342, 2025), Eq. (79):

    min_{J1, Y0..YNph, delta}   p_mu_sig(1) * W(rho^g_{J1})
    s.t.  sum_n P_mu(n) Yvec_n + delta^mu  in  [Qhat - dH, Qhat + dH]   (per mu, per outcome)
          0 <= delta^mu <= 1 - sum_{n<=Nph} P_mu(n)
          Yvec_1 = Phi[rho^t_{J1}]        <-- the unification: single-photon
                                              yields are TIED to the Choi state
          0 <= Y_n(outcome),  sum over (err,ok,nd) outcomes = 1
          J1 >= 0,  Tr_B J1 = I_A

Solved per sector (sectors decouple) by Frank-Wolfe with CVXPY SDP
subproblems; a RELIABLE lower bound is returned via the standard
first-order (Winick-style) linearization SDP at the final iterate.

Spaces (qubit BB84 after squashing):
  A (flying qubit, dim 2), B (squashed: qubit + no-detection flag, dim 3).
  J1 in Pos(A x B), Tr_B J1 = I_A.
  rho^{g/t}_J = Tr_A[(I_{A'} (x) J)(|xi><xi|^{T_A} (x) I_B)],
  |xi> = (|00>+|11>)/sqrt(2)  (source replacement; bases live in the POVMs).

Entropy map (Kamin Eqs. (60),(63),(64)):
  W(rho) = D( G(rho) || Z(G(rho)) )   in bits,
  G Kraus: K = sum_s |s>_S (x) |s><s|_{A} (x) Pdet_B,  Pdet = diag(1,1,0),
  Z: pinching on S.
(The sifting prefactor (1-gamma)^2 and p_mu_sig(1) are applied outside.)
"""
import numpy as np
import cvxpy as cp
from scipy.stats import poisson
from sector_geat import h2

# ---------------- spaces and fixed operators ----------------
dA, dB = 2, 3
I2, I3 = np.eye(2), np.eye(3)
ket = lambda d, i: np.eye(d)[:, i:i+1]
# maximally entangled |xi> on A' (x) A  (both dim 2)
xi = (np.kron(ket(2, 0), ket(2, 0)) + np.kron(ket(2, 1), ket(2, 1))) / np.sqrt(2)
XI = xi @ xi.T                                    # on A' (x) A  (4x4)
# partial transpose over A (source-replacement convention rho_J uses XI^{T_A})
XI_TA = XI.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)

# Bob squashed POVMs on B: X-basis outcomes + no-detect flag
plus = (ket(3, 0) + ket(3, 1)) / np.sqrt(2)
minus = (ket(3, 0) - ket(3, 1)) / np.sqrt(2)
MB_plus, MB_minus = plus @ plus.T, minus @ minus.T
MB_nd = np.diag([0., 0., 1.])
# Alice test POVMs on A' (X basis, source-replacement convention)
pA = (ket(2, 0) + ket(2, 1)) / np.sqrt(2)
mA = (ket(2, 0) - ket(2, 1)) / np.sqrt(2)
MA_plus, MA_minus = pA @ pA.T, mA @ mA.T

# definitive clean versions:
def CHI(J):
    """rho_{A'B}[i b, j d] = sum_{a,c} XI[(i,a),(j,c)] * J[(a,b),(c,d)]"""
    Jr = J.reshape(2, 3, 2, 3)
    XId = XI_TA.reshape(2, 2, 2, 2)
    rho = np.einsum('icja,abcd->ibjd', XId, Jr)
    return rho.reshape(6, 6)

def CHI_adj(X):
    """chi†(X)[(a,b),(c,d)] = sum_{i,j} XI[(i,a),(j,c)] X[(i,b),(j,d)]"""
    Xr = X.reshape(2, 3, 2, 3)
    XId = XI_TA.reshape(2, 2, 2, 2)
    out = np.einsum('icja,ibjd->abcd', XId, Xr)
    return out.reshape(6, 6)

# G map Kraus (S (x) A' (x) B), S dim 2
Pdet = np.diag([1., 1., 0.])
K_G = (np.kron(np.kron(ket(2, 0), np.diag([1., 0.])), Pdet)
       + np.kron(np.kron(ket(2, 1), np.diag([0., 1.])), Pdet))   # 12x6

def Gmap(rho):    return K_G @ rho @ K_G.T
def Gadj(X):      return K_G.T @ X @ K_G
def Zmap(X):
    X = X.reshape(2, 6, 2, 6).copy()
    X[0, :, 1, :] = 0; X[1, :, 0, :] = 0
    return X.reshape(12, 12)

def safe_log(M, eps=1e-10):
    w, V = np.linalg.eigh((M + M.T) / 2)
    w = np.clip(w, eps, None)
    return (V * np.log2(w)) @ V.T

def W_and_grad(J, pert=1e-10):
    """W(rho_J)=D(G||Z G) in bits and gradient wrt J (6x6 symmetric).
    Perturbation applied AFTER the G map (Gr -> Gr + pert*I) so that the
    supports of Gr and Z(Gr) stay numerically consistent (Z(I)=I)."""
    rho = CHI(J)
    Gr = Gmap(rho)
    Gr = Gr + pert * np.eye(12)
    ZGr = Zmap(Gr)
    lG, lZ = safe_log(Gr, eps=1e-14), safe_log(ZGr, eps=1e-14)
    W = float(np.trace(Gr @ (lG - lZ)))
    grad_rho = Gadj(lG - lZ)            # d/d rho Tr[G(rho)(logG - logZG)] (bits)
    return max(W, 0.0), CHI_adj(grad_rho)

# single-photon test observables tied to J (the unification constraint)
def Y1_operators():
    """Linear maps J -> Y1^{(a,b)} = Tr[(MA_a (x) MB_b)^{via XI} J]."""
    ops = {}
    for an, MA in [('+', MA_plus), ('-', MA_minus)]:
        for bn, MB in [('+', MB_plus), ('-', MB_minus), ('nd', MB_nd)]:
            X = np.kron(MA, MB)                     # on A' (x) B
            ops[(an, bn)] = CHI_adj(X)              # Tr[X CHI(J)] = Tr[ops J]
    return ops
Y1OPS = Y1_operators()

# ---------------- channel model (honest statistics) ----------------
def sector_observed(mu, eta, p_dark, e_mis):
    """X-basis test observables: prob of (err, ok, nodet) per pulse."""
    Y0 = 2 * p_dark
    Q = 1 - (1 - Y0) * np.exp(-eta * mu)
    EQ = 0.5 * Y0 + e_mis * (1 - np.exp(-eta * mu))
    EQ = min(EQ, Q)
    return np.array([EQ, Q - EQ, 1 - Q])           # err, ok, nd

# ---------------- the unified optimization (one sector) ----------------
def unified_sector(eta, p_dark, e_mis, N_mu, intensities,
                   eps_cell=1e-10, Nph=6, fw_iters=40, verbose=False):
    """
    Returns (W_lower, w1, info): reliable lower bound on
       p_sig(1) * W(rho^g_{J1})   over the unified feasible set,
    plus the single-photon weight implied at the optimum.
    """
    mus = list(intensities)
    nvec = np.arange(Nph + 1)
    Pmu = {mu: poisson.pmf(nvec, mu) for mu in mus}
    obs = {mu: sector_observed(mu, eta, p_dark, e_mis) for mu in mus}
    dH = {mu: np.sqrt(np.log(2 / eps_cell) / (2 * max(N_mu[mu], 1.0)))
          for mu in mus}
    # BOTH Alice test inputs are constrained (the protocol sends |+> and |->
    # uniformly); outcome layout per photon number:
    #   [ (+,err),(+,ok),(+,nd), (-,err),(-,ok),(-,nd) ]  -- 6 components,
    # each input triple summing to 1 (conditional yields per input).
    Jv = cp.Variable((6, 6), symmetric=True)
    Y = {n: cp.Variable(6, nonneg=True) for n in nvec}
    dlt = {mu: cp.Variable(6) for mu in mus}
    cons = [Jv >> 0,
            cp.partial_trace(Jv, [2, 3], 1) == I2]
    for n in nvec:
        cons += [cp.sum(Y[n][0:3]) == 1, cp.sum(Y[n][3:6]) == 1, Y[n] <= 1]
    for mu in mus:
        tail = 1 - Pmu[mu].sum()
        model = sum(Pmu[mu][n] * Y[n] for n in nvec) + dlt[mu]
        ob6 = np.concatenate([obs[mu], obs[mu]])     # symmetric honest model
        lo = np.maximum(ob6 - dH[mu], 0.0)
        hi = np.minimum(ob6 + dH[mu], 1.0)
        cons += [model >= lo, model <= hi,
                 dlt[mu] >= 0, dlt[mu] <= tail + 1e-12]
    # unification: Y_1 tied to the Choi state on BOTH input rows
    # (factor 2 = 1/p(input) under uniform source replacement)
    cons += [Y[1][0] == 2 * cp.trace(Y1OPS[('+', '-')] @ Jv),
             Y[1][1] == 2 * cp.trace(Y1OPS[('+', '+')] @ Jv),
             Y[1][2] == 2 * cp.trace(Y1OPS[('+', 'nd')] @ Jv),
             Y[1][3] == 2 * cp.trace(Y1OPS[('-', '+')] @ Jv),
             Y[1][4] == 2 * cp.trace(Y1OPS[('-', '-')] @ Jv),
             Y[1][5] == 2 * cp.trace(Y1OPS[('-', 'nd')] @ Jv)]

    grad_param = cp.Parameter((6, 6), symmetric=True)
    prob = cp.Problem(cp.Minimize(cp.trace(grad_param @ Jv)), cons)

    # Frank-Wolfe
    J = np.eye(6) / 3                               # Tr_B = I_A feasible start
    # project start into feasible set once (solve with zero gradient)
    grad_param.value = np.zeros((6, 6))
    prob.solve(solver=cp.CLARABEL)
    J = Jv.value.copy()
    for k in range(fw_iters):
        W, g = W_and_grad(J)
        grad_param.value = (g + g.T) / 2
        prob.solve(solver=cp.CLARABEL, warm_start=True)
        S = Jv.value
        step = 2.0 / (k + 3)
        J = (1 - step) * J + step * S
        if verbose and k % 10 == 0:
            print(f'  FW {k}: W={W:.6f}')
    # reliable lower bound: min over feasible of linearization at J
    W, g = W_and_grad(J)
    grad_param.value = (g + g.T) / 2
    prob.solve(solver=cp.CLARABEL, warm_start=True)
    lin_min = float(np.trace(((g + g.T) / 2) @ Jv.value))
    lin_at_J = float(np.trace(((g + g.T) / 2) @ J))
    W_lower = max(W + lin_min - lin_at_J, 0.0)
    mu_sig = mus[0]
    p1 = poisson.pmf(1, mu_sig)
    w1 = p1 * float(Y[1].value[0] + Y[1].value[1]) if Y[1].value is not None else p1
    return p1 * W_lower, w1, dict(W_fw=W, W_lower=W_lower)

if __name__ == '__main__':
    # sanity: honest point, one sector, moderate loss
    eta_det, alpha_fib, p_dark = 0.6, 0.2, 1e-7
    L = 20.0
    eta = eta_det * 10 ** (-alpha_fib * L / 10)
    intensities = (0.9, 2e-2, 1e-3)
    n, gamma, p_m = 1e12, 0.05, 1 / 8
    N_mu = {mu: n * gamma * p_m / 3 for mu in intensities}
    e_mis = 0.02
    val, w1, info = unified_sector(eta, p_dark, e_mis, N_mu, intensities,
                                   fw_iters=30, verbose=True)
    print('unified  p1*W lower bound =', val)
    # analytic expectation at honest point: p1*Y1*(1-h2(e1)) with Y1~eta
    Y1h = 1 - (1 - 2 * p_dark) * (1 - eta)
    e1h = (eta * e_mis + p_dark) / Y1h
    print('honest analytic reference =', poisson.pmf(1, 0.9) * Y1h * (1 - h2(e1h)))


# ======================================================================
# DUAL EXTRACTION: from the unified SDP to a certified affine
# (crossover) min-trade-off function  f(q) = V(qhat) + lam.(q - qhat)
# ======================================================================
def unified_sector_dual(eta, p_dark, e_mis, N_mu, intensities,
                        Nph=6, fw_iters=35):
    """
    Two-stage construction (Kamin-style):
      stage 1: Frank-Wolfe on the interval-constrained unified problem
               -> linearization point Jbar (best tangent of W);
      stage 2: EQUALITY-constrained linear SDP at the honest cell
               probabilities qhat:
                   V(qhat) = min <gradW(Jbar), J> + const
                   s.t.  sum_n P_mu(n) Y_n + delta_mu = qhat_mu  (all mu)
                         (J,Y,delta) in the structural set;
               LP/SDP duality: V is CONVEX in qhat and the equality
               duals lam give the GLOBAL affine minorant
                   f(q) = V(qhat) + lam.(q - qhat)  <=  V(q)  for all q,
               and V(q) lower-bounds the true single-photon entropy of
               any attack with conditional test statistics q.
    Returns: dict with value (p1*V), lam (per (mu,outcome) cell, scaled
             by p1), qhat cells, and closures for validation.
    """
    mus = list(intensities)
    nvec = np.arange(Nph + 1)
    Pmu = {mu: poisson.pmf(nvec, mu) for mu in mus}
    obs = {mu: sector_observed(mu, eta, p_dark, e_mis) for mu in mus}
    dH = {mu: np.sqrt(np.log(2 / 1e-10) / (2 * max(N_mu[mu], 1.0)))
          for mu in mus}
    p1 = poisson.pmf(1, mus[0])

    # ---------- stage 1: FW with interval constraints ----------
    Jv = cp.Variable((6, 6), symmetric=True)
    Y = {n: cp.Variable(6, nonneg=True) for n in nvec}
    dlt = {mu: cp.Variable(6) for mu in mus}
    cons = [Jv >> 0, cp.partial_trace(Jv, [2, 3], 1) == I2]
    for n in nvec:
        cons += [cp.sum(Y[n][0:3]) == 1, cp.sum(Y[n][3:6]) == 1, Y[n] <= 1]
    for mu in mus:
        tail = 1 - Pmu[mu].sum()
        model = sum(Pmu[mu][n] * Y[n] for n in nvec) + dlt[mu]
        ob6 = np.concatenate([obs[mu], obs[mu]])
        cons += [model >= np.maximum(ob6 - dH[mu], 0),
                 model <= np.minimum(ob6 + dH[mu], 1),
                 dlt[mu] >= 0, dlt[mu] <= tail + 1e-12]
    cons += [Y[1][0] == 2 * cp.trace(Y1OPS[('+', '-')] @ Jv),
             Y[1][1] == 2 * cp.trace(Y1OPS[('+', '+')] @ Jv),
             Y[1][2] == 2 * cp.trace(Y1OPS[('+', 'nd')] @ Jv),
             Y[1][3] == 2 * cp.trace(Y1OPS[('-', '+')] @ Jv),
             Y[1][4] == 2 * cp.trace(Y1OPS[('-', '-')] @ Jv),
             Y[1][5] == 2 * cp.trace(Y1OPS[('-', 'nd')] @ Jv)]
    gpar = cp.Parameter((6, 6), symmetric=True)
    prob = cp.Problem(cp.Minimize(cp.trace(gpar @ Jv)), cons)
    gpar.value = np.zeros((6, 6))
    prob.solve(solver=cp.CLARABEL)
    J = Jv.value.copy()
    for k in range(fw_iters):
        _, g = W_and_grad(J)
        gpar.value = (g + g.T) / 2
        prob.solve(solver=cp.CLARABEL, warm_start=True)
        J = (1 - 2 / (k + 3)) * J + 2 / (k + 3) * Jv.value
    Wbar, gbar = W_and_grad(J)
    gbar = (gbar + gbar.T) / 2
    const0 = Wbar - float(np.trace(gbar @ J))     # W(J) >= <gbar,J> + const0

    # ---------- stage 2: equality-constrained dual problem ----------
    J2 = cp.Variable((6, 6), symmetric=True)
    Y2 = {n: cp.Variable(6, nonneg=True) for n in nvec}
    d2 = {mu: cp.Variable(6) for mu in mus}
    qpar = {mu: cp.Parameter(6) for mu in mus}
    cons2 = [J2 >> 0, cp.partial_trace(J2, [2, 3], 1) == I2]
    for n in nvec:
        cons2 += [cp.sum(Y2[n][0:3]) == 1, cp.sum(Y2[n][3:6]) == 1, Y2[n] <= 1]
    eqcons = {}
    for mu in mus:
        tail = 1 - Pmu[mu].sum()
        model = sum(Pmu[mu][n] * Y2[n] for n in nvec) + d2[mu]
        eqcons[mu] = (model == qpar[mu])
        cons2 += [eqcons[mu], d2[mu] >= 0, d2[mu] <= tail + 1e-12]
    cons2 += [Y2[1][0] == 2 * cp.trace(Y1OPS[('+', '-')] @ J2),
              Y2[1][1] == 2 * cp.trace(Y1OPS[('+', '+')] @ J2),
              Y2[1][2] == 2 * cp.trace(Y1OPS[('+', 'nd')] @ J2),
              Y2[1][3] == 2 * cp.trace(Y1OPS[('-', '+')] @ J2),
              Y2[1][4] == 2 * cp.trace(Y1OPS[('-', '-')] @ J2),
              Y2[1][5] == 2 * cp.trace(Y1OPS[('-', 'nd')] @ J2)]
    prob2 = cp.Problem(cp.Minimize(cp.trace(gbar @ J2)), cons2)

    def V_of(qcells):                # qcells: dict mu -> 6-vector
        for mu in mus:
            qpar[mu].value = qcells[mu]
        prob2.solve(solver=cp.CLARABEL)
        if prob2.status not in ('optimal', 'optimal_inaccurate'):
            return None, None
        # cvxpy equality-dual sign convention: subgradient of V wrt rhs = -dual
        lam = {mu: -np.array(eqcons[mu].dual_value).flatten() for mu in mus}
        return float(prob2.value) + const0, lam

    qhat = {mu: np.concatenate([obs[mu], obs[mu]]) for mu in mus}
    V0, lam = V_of(qhat)
    return dict(p1=p1, V0=V0, lam=lam, qhat=qhat, V_of=V_of,
                value=p1 * max(V0, 0.0), mus=mus, gbar=gbar, const0=const0)


def validate_minorant(dd, eta, p_dark, n_tests=6, seed=1):
    """certificate: f(q') <= V(q') for perturbed channels q'."""
    rng = np.random.default_rng(seed)
    ok = True
    for t in range(n_tests):
        em = float(np.clip(0.02 + rng.normal(0, 0.015), 0.001, 0.12))
        sc = float(np.clip(1 + rng.normal(0, 0.15), 0.5, 1.2))
        qp = {}
        for mu in dd['mus']:
            o = sector_observed(mu, eta * sc, p_dark, em)
            qp[mu] = np.concatenate([o, o])
        Vp, _ = dd['V_of'](qp)
        if Vp is None:
            continue
        f = dd['V0'] + sum(float(dd['lam'][mu] @ (qp[mu] - dd['qhat'][mu]))
                           for mu in dd['mus'])
        gap = Vp - f
        ok = ok and (gap >= -1e-6)
        print(f'   test {t}: e={em:.3f} eta x {sc:.2f}: '
              f'f={f:+.5f}  V={Vp:+.5f}  minorant gap={gap:+.2e} '
              f'{"OK" if gap >= -1e-6 else "VIOLATION"}')
    return ok


def lagrangian_family(eta, p_dark, e_mis, N_mu, intensities,
                      s_grid=(1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01),
                      Nph=6, fw_iters=30):
    """
    OPTIMIZED min-trade-off family: for any multiplier vector lam,
        f_lam(q) = g(lam) + lam.q,
        g(lam) = min_{structural set} [ W_lin(J) - lam . model(Y,delta) ],
    is a VALID affine minorant (weak duality; no statistical constraints
    inside g).  We evaluate the family lam = s*lam_star (shrunk exact
    tangent): by the envelope theorem the honest value is flat near s=1
    while the gradient spread shrinks linearly -- the practical handle
    that tames the GEAT spread penalty (Kamin's 'optimized
    min-trade-off' in its simplest one-parameter form).
    Returns list of dicts: {s, value_hon, lam (dict mu->6), p1}.
    """
    dd = unified_sector_dual(eta, p_dark, e_mis, N_mu, intensities,
                             Nph=Nph, fw_iters=fw_iters)
    mus = list(intensities)
    nvec = np.arange(Nph + 1)
    Pmu = {mu: poisson.pmf(nvec, mu) for mu in mus}
    # rebuild structural-set SDP with parametric linear objective
    J2 = cp.Variable((6, 6), symmetric=True)
    Y2 = {n: cp.Variable(6, nonneg=True) for n in nvec}
    d2 = {mu: cp.Variable(6) for mu in mus}
    cons2 = [J2 >> 0, cp.partial_trace(J2, [2, 3], 1) == I2]
    for n in nvec:
        cons2 += [cp.sum(Y2[n][0:3]) == 1, cp.sum(Y2[n][3:6]) == 1, Y2[n] <= 1]
    models = {}
    for mu in mus:
        tail = 1 - Pmu[mu].sum()
        models[mu] = sum(Pmu[mu][n] * Y2[n] for n in nvec) + d2[mu]
        cons2 += [d2[mu] >= 0, d2[mu] <= tail + 1e-12]
    cons2 += [Y2[1][0] == 2 * cp.trace(Y1OPS[('+', '-')] @ J2),
              Y2[1][1] == 2 * cp.trace(Y1OPS[('+', '+')] @ J2),
              Y2[1][2] == 2 * cp.trace(Y1OPS[('+', 'nd')] @ J2),
              Y2[1][3] == 2 * cp.trace(Y1OPS[('-', '+')] @ J2),
              Y2[1][4] == 2 * cp.trace(Y1OPS[('-', '-')] @ J2),
              Y2[1][5] == 2 * cp.trace(Y1OPS[('-', 'nd')] @ J2)]
    lpar = {mu: cp.Parameter(6) for mu in mus}
    # gbar/const0 recovered from dd by re-deriving at its tangent: use the
    # equality-dual machinery's objective pieces via a fresh FW pass is
    # wasteful; instead reconstruct from dd: V0 = g(lam*) + lam*.qhat
    # => g(lam*) known; for general s we must solve the SDP, so we need
    # gbar: stash it on dd in unified_sector_dual (patched below).
    gbar, const0 = dd['gbar'], dd['const0']
    obj = cp.Minimize(cp.trace(gbar @ J2)
                      - sum(lpar[mu] @ models[mu] for mu in mus))
    prob2 = cp.Problem(obj, cons2)
    fam = []
    for s in s_grid:
        for mu in mus:
            lpar[mu].value = s * dd['lam'][mu]
        prob2.solve(solver=cp.CLARABEL)
        if prob2.status not in ('optimal', 'optimal_inaccurate'):
            continue
        gval = float(prob2.value) + const0
        val = gval + s * sum(float(dd['lam'][mu] @ dd['qhat'][mu])
                             for mu in mus)
        fam.append(dict(s=s, value_hon=val,
                        lam={mu: s * dd['lam'][mu] for mu in mus},
                        p1=dd['p1'], qhat=dd['qhat']))
    return fam


def kappa_family(eta, p_dark, e_mis, kappa_grid=None, Nph=6, fw_iters=30):
    """
    Width-regularized optimized min-trade-off family.

    For each interval half-width kappa: solve the linearized problem
        V_k = min <gbar,J>+c0  s.t. |model - qhat| <= kappa (+ structure);
    its interval duals provide a candidate multiplier vector lam_k whose
    l1-norm is explicitly penalized by kappa (LP duality:
    V_k = max_lam [g(lam) - kappa*||lam||_1]).  Each candidate is then
    SELF-CERTIFIED by one structural SDP:
        g(lam) = min_{structure} [ <gbar,J> + c0 - lam.(model - qhat) ],
    which is a valid honest value for the affine minorant
        f(q) = g(lam) + lam.(q - qhat)
    for ANY lam (weak duality) -- no dual sign/scale convention enters
    the certificate.
    Returns list of {kappa, value_hon, lam, p1, qhat}.
    """
    if kappa_grid is None:
        kappa_grid = [1e-4, 3e-4, 1e-3, 3e-3, 6e-3, 1e-2, 2e-2]
    mus = list((0.9, 2e-2, 1e-3))
    nvec = np.arange(Nph + 1)
    Pmu = {mu: poisson.pmf(nvec, mu) for mu in mus}
    obs = {mu: sector_observed(mu, eta, p_dark, e_mis) for mu in mus}
    qhat = {mu: np.concatenate([obs[mu], obs[mu]]) for mu in mus}
    p1 = poisson.pmf(1, mus[0])

    def structural(Jv, Yv, dv):
        cons = [Jv >> 0, cp.partial_trace(Jv, [2, 3], 1) == I2]
        for n in nvec:
            cons += [cp.sum(Yv[n][0:3]) == 1, cp.sum(Yv[n][3:6]) == 1,
                     Yv[n] <= 1]
        for mu in mus:
            tail = 1 - Pmu[mu].sum()
            cons += [dv[mu] >= 0, dv[mu] <= tail + 1e-12]
        cons += [Yv[1][0] == 2 * cp.trace(Y1OPS[('+', '-')] @ Jv),
                 Yv[1][1] == 2 * cp.trace(Y1OPS[('+', '+')] @ Jv),
                 Yv[1][2] == 2 * cp.trace(Y1OPS[('+', 'nd')] @ Jv),
                 Yv[1][3] == 2 * cp.trace(Y1OPS[('-', '+')] @ Jv),
                 Yv[1][4] == 2 * cp.trace(Y1OPS[('-', '-')] @ Jv),
                 Yv[1][5] == 2 * cp.trace(Y1OPS[('-', 'nd')] @ Jv)]
        return cons

    # ---- stage 1: FW at moderate Hoeffding widths to get gbar ----
    J = None
    Jv = cp.Variable((6, 6), symmetric=True)
    Yv = {n: cp.Variable(6, nonneg=True) for n in nvec}
    dv = {mu: cp.Variable(6) for mu in mus}
    cons = structural(Jv, Yv, dv)
    for mu in mus:
        model = sum(Pmu[mu][n] * Yv[n] for n in nvec) + dv[mu]
        cons += [model >= np.maximum(qhat[mu] - 1e-4, 0),
                 model <= np.minimum(qhat[mu] + 1e-4, 1)]
    gpar = cp.Parameter((6, 6), symmetric=True)
    prob = cp.Problem(cp.Minimize(cp.trace(gpar @ Jv)), cons)
    gpar.value = np.zeros((6, 6))
    prob.solve(solver=cp.CLARABEL)
    J = Jv.value.copy()
    for k in range(fw_iters):
        _, g = W_and_grad(J)
        gpar.value = (g + g.T) / 2
        prob.solve(solver=cp.CLARABEL, warm_start=True)
        J = (1 - 2 / (k + 3)) * J + 2 / (k + 3) * Jv.value
    Wb, gb = W_and_grad(J)
    gbar = (gb + gb.T) / 2
    c0 = Wb - float(np.trace(gbar @ J))

    # ---- per kappa: interval duals -> candidate lam -> certify g(lam) ----
    # interval problem with parametric width
    Jk = cp.Variable((6, 6), symmetric=True)
    Yk = {n: cp.Variable(6, nonneg=True) for n in nvec}
    dk = {mu: cp.Variable(6) for mu in mus}
    consk = structural(Jk, Yk, dk)
    lo_c, hi_c = {}, {}
    kap = cp.Parameter(nonneg=True)
    PMU_PROT = {mus[0]: 0.8, mus[1]: 0.15, mus[2]: 0.05}
    for mu in mus:
        model = sum(Pmu[mu][n] * Yk[n] for n in nvec) + dk[mu]
        w = PMU_PROT[mu] * 0.5
        lo_c[mu] = (model >= qhat[mu] - kap * w)
        hi_c[mu] = (model <= qhat[mu] + kap * w)
        consk += [lo_c[mu], hi_c[mu]]
    probk = cp.Problem(cp.Minimize(cp.trace(gbar @ Jk)), consk)

    # certification problem with parametric lam
    Jc = cp.Variable((6, 6), symmetric=True)
    Yc = {n: cp.Variable(6, nonneg=True) for n in nvec}
    dc = {mu: cp.Variable(6) for mu in mus}
    consc = structural(Jc, Yc, dc)
    lpar = {mu: cp.Parameter(6) for mu in mus}
    objc = cp.Minimize(cp.trace(gbar @ Jc)
                       - sum(lpar[mu] @ (sum(Pmu[mu][n] * Yc[n] for n in nvec)
                                         + dc[mu] - qhat[mu]) for mu in mus))
    probc = cp.Problem(objc, consc)

    fam = []
    for kappa in kappa_grid:
        kap.value = kappa
        probk.solve(solver=cp.CLARABEL, warm_start=True)
        if probk.status not in ('optimal', 'optimal_inaccurate'):
            continue
        lam = {}
        for mu in mus:
            nu_lo = np.array(lo_c[mu].dual_value).flatten()
            nu_hi = np.array(hi_c[mu].dual_value).flatten()
            lam[mu] = nu_lo - nu_hi
        # try both sign conventions; the certificate picks the truth
        best = None
        for sgn in (+1.0, -1.0):
            for mu in mus:
                lpar[mu].value = sgn * lam[mu]
            probc.solve(solver=cp.CLARABEL, warm_start=True)
            if probc.status not in ('optimal', 'optimal_inaccurate'):
                continue
            val = float(probc.value) + c0
            if best is None or val > best[0]:
                best = (val, {mu: sgn * lam[mu] for mu in mus})
        if best is None:
            continue
        fam.append(dict(kappa=kappa, value_hon=best[0], lam=best[1],
                        p1=p1, qhat=qhat))
    return fam
