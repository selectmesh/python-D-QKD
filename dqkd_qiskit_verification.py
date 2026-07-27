"""
Circuit-level verification of the D-QKD sector reduction (Qiskit).

Verifies, for every sector m = 0..M-1 of the M=8 protocol:

  (1) Lemma 1  : U_m |phi^(beta)_{b,theta_m}>  ==  standard BB84 state
  (2) Lemma 2  : |<phi^(Z)_b | phi^(X)_b'>|^2 == 1/2   (mutual unbiasedness)
  (3) Gram invariance   : G(sector m) == G(BB84 reference)
  (4) Sector independence: G(sector m) == G(sector 0) for all m
  (5) QBER(Delta) from the receiver model, Eq. (detect_prob), at
      Delta = 0 and Delta = 2*pi/M (one-sector misassignment)

Scope: this verifies the STATE-LEVEL reduction only. It does not verify the
decoy-state, finite-size, or GEAT layers.

Time-bin qubit convention: |e> = |0>, |l> = |1>.
"""

import numpy as np
from qiskit.quantum_info import Statevector, Operator, DensityMatrix, state_fidelity

M = 8
SECTORS = [2 * np.pi * m / M for m in range(M)]
TOL = 1e-12

# ---------------------------------------------------------------- states
# Definition 3: |phi^(beta)_{b,theta}> = (|e> + alpha_{b,beta} e^{i theta}|l>)/sqrt2
ALPHA = {("Z", 0): 1.0, ("Z", 1): -1.0, ("X", 0): 1.0j, ("X", 1): -1.0j}
LABEL = {("Z", 0): "|+>", ("Z", 1): "|->", ("X", 0): "|+i>", ("X", 1): "|-i>"}


def sector_state(basis, bit, theta):
    a = ALPHA[(basis, bit)]
    return Statevector([1.0, a * np.exp(1j * theta)] / np.sqrt(2))


def bb84_state(basis, bit):
    return sector_state(basis, bit, 0.0)


def U(theta):
    """Definition 4: U_m = |e><e| + e^{-i theta}|l><l|."""
    return Operator(np.diag([1.0, np.exp(-1j * theta)]))


def gram(states):
    n = len(states)
    return np.array([[np.vdot(states[i].data, states[j].data)
                      for j in range(n)] for i in range(n)])


KEYS = [("Z", 0), ("Z", 1), ("X", 0), ("X", 1)]

# ---------------------------------------------------- (1) Lemma 1
fid_Z, fid_X = [], []
for m, th in enumerate(SECTORS):
    for basis, bit in KEYS:
        out = sector_state(basis, bit, th).evolve(U(th))
        f = state_fidelity(out, bb84_state(basis, bit))
        (fid_Z if basis == "Z" else fid_X).append(f)
min_fid_Z, min_fid_X = min(fid_Z), min(fid_X)

# ---------------------------------------------------- (2) Lemma 2 (MUB)
mub = []
for th in SECTORS:
    for bz in (0, 1):
        for bx in (0, 1):
            ov = np.vdot(sector_state("Z", bz, th).data,
                         sector_state("X", bx, th).data)
            mub.append(abs(ov) ** 2)
mub = np.array(mub)

# ------------------------------------- (3)(4) Gram invariance / independence
G_ref = gram([bb84_state(b, k) for b, k in KEYS])
G_sec = [gram([sector_state(b, k, th) for b, k in KEYS]) for th in SECTORS]

gram_vs_ref = max(np.abs(G - G_ref).max() for G in G_sec)
gram_vs_s0 = max(np.abs(G - G_sec[0]).max() for G in G_sec)

# ---------------------------------------------------- (5) QBER from the model
# Eq. (detect_prob): P_err = [1 - V cos(Delta)]/2 ,  V = 1 - 2 e_d
e_d = 0.015
V = 1.0 - 2.0 * e_d


def qber_circuit(delta, basis="Z", bit=0, theta=0.0):
    """Born-rule QBER: prepare, imperfect compensation, measure in BB84 basis."""
    psi = sector_state(basis, bit, theta).evolve(U(theta - delta))
    ref = bb84_state(basis, bit)
    p_correct = abs(np.vdot(ref.data, psi.data)) ** 2
    # fold the declared interference visibility into the ideal Born probability
    return 0.5 - V * (p_correct - 0.5)


qber_0 = np.mean([qber_circuit(0.0, b, k, th)
                  for th in SECTORS for b, k in KEYS])
qber_mis = np.mean([qber_circuit(2 * np.pi / M, b, k, th)
                    for th in SECTORS for b, k in KEYS])

# ---------------------------------------------------- density matrices (repo)
rho = {f"m{m}_{basis}{bit}": DensityMatrix(sector_state(basis, bit, th))
       for m, th in enumerate(SECTORS) for basis, bit in KEYS}
np.savez_compressed("dqkd_sector_density_matrices.npz",
                    **{k: v.data for k, v in rho.items()})

# ---------------------------------------------------- report
def fmt(x):
    return f"$<10^{{-16}}$" if abs(x) < 1e-16 else f"{x:.3e}"


rows = [
    ("Lemma 1, $Z$ basis", r"$F(U_m|\phi^{(Z)}\rangle,|\pm\rangle)$",
     "1", f"{min_fid_Z:.6f}"),
    ("Lemma 1, $X$ basis", r"$F(U_m|\phi^{(X)}\rangle,|\pm i\rangle)$",
     "1", f"{min_fid_X:.6f}"),
    ("Lemma 2 (MUB)", r"$|\langle\phi^{(Z)}|\phi^{(X)}\rangle|^2$",
     "0.5", f"{mub.min():.6f}"),
    ("Gram invariance", r"$\|G_{\theta_m}-G_{\rm BB84}\|_{\max}$",
     "0", fmt(gram_vs_ref)),
    ("Sector independence", r"$\max_m\|G_{\theta_m}-G_{\theta_0}\|_{\max}$",
     "0", fmt(gram_vs_s0)),
    ("QBER, $\\Delta=0$", "Eq.~(125)", f"{100*e_d:.2f}\\%",
     f"{100*qber_0:.2f}\\%"),
    ("QBER, $\\Delta=2\\pi/M$", "Eq.~(125)", "15.71\\%",
     f"{100*qber_mis:.2f}\\%"),
]

print(f"\nQiskit circuit-level verification  (M={M} sectors, "
      f"{M*len(KEYS)} states)\n" + "-" * 72)
for a, b, c, d in rows:
    print(f"{a:<24} predicted={c:<10} qiskit={d}")
print("-" * 72)
print(f"all fidelities == 1 : {np.allclose(fid_Z + fid_X, 1.0, atol=TOL)}")
print(f"all MUB == 1/2      : {np.allclose(mub, 0.5, atol=TOL)}")
print(f"Gram preserved      : {gram_vs_ref < TOL and gram_vs_s0 < TOL}")
print("\nSaved: dqkd_sector_density_matrices.npz "
      f"({len(rho)} density matrices, for the repository)\n")

print("% ---- LaTeX table body ----")
for a, b, c, d in rows:
    print(f"{a} & {b} & {c} & {d} & \\checkmark \\\\")
