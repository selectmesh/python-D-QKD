#!/usr/bin/env python3
"""
Cross-method finite-key comparison for sector-conditioned D-QKD
================================================================
Eight published finite-key treatments are applied to the SAME simulated
detection record, at identical N, security parameter eps_sec, error-correction
efficiency and (where applicable) number of groups M.

What differs between rows is ONLY:
  (i)  the finite-statistics concentration bound used for the phase error,
  (ii) the entropy-layer penalty family (EUR-type vs entropy-accumulation),
  (iii) the PARTITION of the data (pooled / contiguous blocks / phase sectors).

Rows 1-6 are pooled analyses; row 7 (Bae et al.) is the only prior *structured*
analysis and uses contiguous temporal blocks; row 8 is this work.

IMPORTANT (honesty statement reproduced in the caption): absolute values depend
on our declared channel model, so only the RELATIVE ordering is meaningful. We
implement the characteristic statistical treatment of each reference, not a
bit-exact reproduction of its full framework.

Outputs: methods_results.csv, methods_table.tex, fig_methods.pdf/.png
"""

import numpy as np, csv
from math import log2, log, sqrt

# ----------------------------------------------------------------- constants
ALPHA_DB, ETA_DET, Q_SIFT, F_EC, Y1 = 0.2, 0.85, 0.5, 1.16, 0.5
EPS_SEC = 1e-8
EPS_PE, EPS_PA, EPS_COR, EPS_S = EPS_SEC/2, EPS_SEC/4, EPS_SEC/8, EPS_SEC/8
N_SLOTS, M_GROUPS, N_EMIT = 20000, 8, 1e10


def h2(x):
    return 0.0 if x <= 0 or x >= 1 else -x*log2(x)-(1-x)*log2(1-x)


# ----------------------------------------------------------- channel scenarios
def channel(scn, n_cycles=50, amp=0.30):
    t = (np.arange(N_SLOTS)+0.5)/N_SLOTS
    if scn == "monotone":
        e_lo, e_hi = 0.005, 0.16
        return t.copy(), e_lo+(e_hi-e_lo)*t, 1.0+amp*(1-2*t)
    lo, hi = (0.004, 0.055) if scn == "quasi-mild" else (0.005, 0.16)
    ph = (n_cycles*t) % 1.0
    c = np.cos(2*np.pi*ph)
    return ph, 0.5*(lo+hi) - 0.5*(hi-lo)*c, 1.0+amp*c


# -------------------------------------------------- finite-statistics families
def d_hoeffding(n, eps, p):                      # generic bounded-variable
    return sqrt(log(2/eps)/(2*n))

def d_chernoff(n, eps, p):                       # Lim et al., PRA 89, 022307
    return sqrt(2*p*log(1/eps)/n) + 2*log(1/eps)/(3*n)

def d_sharp(n, eps, p):                          # Mannalath et al., PRL 135, 020803
    return sqrt(2*p*(1-p)*log(1/eps)/n) + log(1/eps)/(3*n)


# ------------------------------------------------------ entropy-layer penalties
def pen_eur(n, k):                               # EUR-type smoothing penalty
    return k*sqrt(n*log2(2/EPS_S))

def pen_eat(n, scale):                           # entropy-accumulation-type
    return scale*(log2(2/EPS_COR) + 2*log2(1/(2*EPS_PA)) + sqrt(n))


# --------------------------------------------------------------- the 8 methods
#  name, reference key, partition, delta-fn, penalty-fn
METHODS = [
    ("Tomamichel \\etal (EUR)",      "Tomamichel",       "pooled",  d_hoeffding, lambda n: pen_eur(n, 7.0)),
    ("Lim \\etal (Chernoff)",         "lim2014",          "pooled",  d_chernoff,  lambda n: pen_eur(n, 6.0)),
    ("Wiesemann \\etal (consol.)",    "Wiesemann2026",    "pooled",  d_chernoff,  lambda n: pen_eur(n, 5.0)),
    ("Kamin \\etal (GEAT)",           "KaminGEAT",        "pooled",  d_hoeffding, lambda n: pen_eat(n, 1.0)),
    ("Arqand \\etal (R\\'enyi GEAT)", "ArqandGREAT",      "pooled",  d_hoeffding, lambda n: pen_eat(n, 0.7)),
    ("Mannalath \\etal (sharp)",      "Mannalath2025",    "pooled",  d_sharp,     lambda n: pen_eur(n, 5.0)),
    ("Bae \\etal (blockwise)",        "Bae2025Blockwise", "blocks",  d_chernoff,  lambda n: pen_eur(n, 5.0)),
    ("\\textbf{This work}",           None,               "sectors", d_hoeffding, lambda n: pen_eat(n, 0.7)),
]


def partition(kind, phase):
    n = len(phase)
    if kind == "pooled":
        return np.zeros(n, dtype=int)
    if kind == "blocks":                                   # contiguous in time
        return np.minimum((np.arange(n)*M_GROUPS)//n, M_GROUPS-1)
    if kind == "sectors":                                  # phase-aligned
        return np.minimum((phase*M_GROUPS).astype(int), M_GROUPS-1)
    raise ValueError(kind)


# Distance-dependent error growth, following the dynamic-channel model of the
# manuscript, e_m(L) = e_m^{(0)} + kappa_m L, with kappa_m largest in the phases
# that are already noisiest, so that the profile polarizes with distance.
KAPPA_MAX = 1.1e-3          # per km, applied to the dirtiest phase


def grow(e0, L):
    span = e0.max() - e0.min()
    w = (e0 - e0.min()) / span if span > 0 else np.zeros_like(e0)
    return np.minimum(0.5, e0 + KAPPA_MAX * w * L)


def evaluate(lab, e_t, eta_rel, L, dfun, pfun):
    eta = 10.0**(-ALPHA_DB*L/10.0)
    pdet = eta*ETA_DET*eta_rel
    e_t = grow(e_t, L)
    per = N_EMIT/len(e_t)
    gids = np.unique(lab)
    eps_g = EPS_PE/len(gids)
    ell = 0.0
    for g in gids:
        s = (lab == g)
        n = float(np.sum(per*pdet[s])*Q_SIFT)
        if n <= 0:
            continue
        e = float(np.sum(per*pdet[s]*e_t[s])*Q_SIFT)/n
        ep = min(0.5, e + dfun(n, eps_g, max(e, 1e-9)))
        contrib = n*Y1*max(0.0, 1-h2(ep)) - F_EC*n*h2(min(0.5, e)) - pfun(n)
        if contrib > 0:
            ell += contrib
    return max(0.0, ell)/N_EMIT


# ------------------------------------------------------------------------ main
if __name__ == "__main__":
    SCEN = [("monotone", "I. Monotone drift"),
            ("quasi-mild", "II. Quasi-periodic (mild)"),
            ("quasi-strong", "III. Quasi-periodic (strong)")]
    L_REF, rows = 50, []

    for scn, pretty in SCEN:
        ph, e_t, eta = channel(scn)
        for name, ref, part, dfun, pfun in METHODS:
            lab = partition(part, ph)
            R = evaluate(lab, e_t, eta, L_REF, dfun, pfun)
            rows.append(dict(scenario=scn, pretty=pretty, method=name, ref=ref,
                             partition=part, L_km=L_REF, R=R))

    # ---- console
    print(f"{'method':<32}{'partition':<10}" + "".join(f"{p:>14}" for _, p in SCEN))
    print("-"*82)
    for name, ref, part, _, _ in METHODS:
        line = f"{name.replace(chr(92)+'textbf','').replace('{','').replace('}',''):<32}{part:<10}"
        for scn, _ in SCEN:
            R = [r for r in rows if r["method"] == name and r["scenario"] == scn][0]["R"]
            line += f"{R:>14.3e}"
        print(line)

    with open("methods_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- LaTeX
    def sci(x):
        if x <= 0:
            return r"---"
        n = int(np.floor(np.log10(x)))
        return r"$%.2f\times10^{%d}$" % (x/10.0**n, n)

    partname = {"pooled": "pooled", "blocks": "time blocks", "sectors": "phase sectors"}
    with open("methods_table.tex", "w") as f:
        f.write("% auto-generated by compare_methods.py\n")
        for name, ref, part, _, _ in METHODS:
            cite = "" if ref is None else r"~\cite{%s}" % ref
            cells = "".join(" & " + sci([r for r in rows if r["method"] == name
                                         and r["scenario"] == s][0]["R"]) for s, _ in SCEN)
            f.write(f"{name}{cite} & {partname[part]}{cells} \\\\\n")

    # figure is produced at the end of the file, once make_figure is defined
def make_figure(rows, SCEN, L_REF):
    """
    Labelled-curve figure in the style of a Nature-family comparison plot:
    every analysis is a curve terminating where it stops certifying a key, and
    each curve carries a coloured label box at its endpoint.  No in-figure
    title; all description belongs to the LaTeX caption.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.linewidth": 0.8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    SHORT = ["Tomamichel", "Lim", "Wiesemann", "Kamin",
             "Arqand", "Mannalath", "Bae", "This work"]
    COL = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B3", "#937860", "#D9A130", "#0E7C6B"]
    MK = ["o", "s", "^", "v", "D", "P", "X", "o"]
    LW = [1.4]*7 + [2.6]
    MS = [4]*7 + [5.5]

    scn = "quasi-mild"
    ph, e0, eta = channel(scn)
    Ls = np.arange(0, 251, 2.0)

    YFLOOR = 1.6e-6
    fig, ax = plt.subplots(figsize=(7.1, 4.3))
    ends = []
    for i, (name, ref, part, dfun, pfun) in enumerate(METHODS):
        lab = partition(part, ph)
        R = np.array([evaluate(lab, e0, eta, L, dfun, pfun) for L in Ls])
        ok = R > YFLOOR          # keep curves and end labels inside the axes
        if not ok.any():
            continue
        x, yv = Ls[ok], R[ok]
        ax.plot(x, yv, color=COL[i], lw=LW[i], marker=MK[i], ms=MS[i],
                markevery=max(1, len(x)//9), markeredgecolor="white",
                markeredgewidth=0.5, zorder=3 + (i == 7)*5, clip_on=True)
        ends.append([x[-1], yv[-1], i])

    # --- place the end labels ----------------------------------------------
    # The analyses that abort early terminate on top of one another, so their
    # labels are stacked downward in the empty region beneath the cluster;
    # the surviving analysis is labelled directly at its own endpoint.
    early = sorted([e for e in ends if e[0] < 150], key=lambda t: -t[1])
    late = [e for e in ends if e[0] >= 150]

    XCOL, ytop, SEP = 74.0, 4.0e-5, 0.295            # decades between labels
    for k, (xe, ye, i) in enumerate(early):
        yl = 10 ** (np.log10(ytop) - k * SEP)
        ax.annotate(SHORT[i], xy=(xe, ye), xytext=(XCOL, yl),
                    fontsize=7.6, fontweight="bold", color="#15181C",
                    va="center", ha="left", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.30", fc=COL[i],
                              ec="#3A3F45", lw=0.55, alpha=0.92),
                    arrowprops=dict(arrowstyle="-", color=COL[i], lw=0.7,
                                    shrinkA=1, shrinkB=2, alpha=0.8))
    for xe, ye, i in late:
        ax.annotate(SHORT[i], xy=(xe, ye), xytext=(xe + 8, ye * 2.6),
                    fontsize=8.2, fontweight="bold", color="white",
                    va="center", ha="left", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.34", fc=COL[i],
                              ec="#0A5A4E", lw=0.7),
                    arrowprops=dict(arrowstyle="-", color=COL[i], lw=0.9,
                                    shrinkA=1, shrinkB=2))

    ax.set_yscale("log")
    ax.set_xlim(0, 320)
    ax.set_ylim(1e-6, 4e-1)
    ax.set_xlabel("Transmission distance, $L$ (km)")
    ax.set_ylabel("Secret key rate (bits per emitted pulse)")
    ax.grid(True, which="major", ls="--", lw=0.5, color="#C8CDD2", alpha=0.75)
    ax.grid(True, which="minor", axis="y", ls=":", lw=0.35,
            color="#DDE1E5", alpha=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#5A6068")

    handles = [plt.Line2D([], [], color=COL[i], lw=LW[i], marker=MK[i],
                          ms=MS[i], markeredgecolor="white",
                          markeredgewidth=0.5, label=SHORT[i])
               for i in range(8)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.155),
              ncol=4, frameon=True, fancybox=False, edgecolor="#A9AEB4",
              fontsize=7.6, handlelength=2.2, columnspacing=1.6,
              borderpad=0.6, handletextpad=0.6)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"fig_methods.{ext}", dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    make_figure(rows, SCEN, L_REF)
    print("\nwrote methods_results.csv, methods_table.tex, fig_methods.pdf/.png")
