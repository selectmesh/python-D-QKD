# -*- coding: utf-8 -*-
r"""
MECHANISM FIGURE - two panels replacing four.

Panel (a) merges the old fig2 panels.  Those two panels plotted the SAME gain
against two x-variables that are deterministically related: the sector profile is
e_m = e_base + (beta A/2)(1 - cos 2 pi m/M), so

        Var_w(e) = beta^2 A^2 / 8 ,

i.e. the "gain vs variance" panel was the "gain vs A" panel with a reparametrised
axis.  One axes with a secondary top scale carries both, and removes a duplication.

Panel (b) merges the old ablation panels on their common distance axis: the left
(log) scale carries the certified key rate, the right (linear) scale carries the
necessity gain R_B4/R_B5 - 1.  The three analyses B1, B3, B5 coincide at the
abort floor under strong drift, so they are drawn once and labelled once.

No in-figure titles.  Panels are labelled (a), (b) for the caption.
Output: fig_mechanism.pdf / .png
"""
import csv
import os

import numpy as np

import min_entropy_secondorder as so
import stage4_corrected as s4c

HERE = os.path.dirname(os.path.abspath(__file__))
BETA = 0.5                     # profile amplitude
M = 8
L0 = 50.0                      # evaluation distance
E_BASE = 0.015                 # intrinsic error floor
N_ASYM = 1e30                  # drives the decoy-LP confidence radius to zero


def set_profile(A, n):
    """Sector offsets e_m = e_base + (beta A/2)(1 - cos 2 pi m/M); kappa = 0,
    so the heterogeneity is the sector-dependent compensation residual alone."""
    m = np.arange(M)
    s4c.M, s4c.p, s4c.N = M, 1.0 / M, n
    s4c.e0m = E_BASE + BETA * A / 2.0 * (1 - np.cos(2 * np.pi * m / M))
    s4c.kapm = np.zeros(M)


def _obs(L, n):
    o = [s4c.sector_decoy(L, m, n / M) for m in range(M)]
    return (np.array([x[2] for x in o]), np.array([x[1] for x in o]),
            np.array([x[3] for x in o]), np.array([x[4] for x in o]))


def resolved_all(L, form, n):
    """Sector-resolved rate keeping ALL M sectors (what the theorem bounds)."""
    Q1, e1, Qs, Es = _obs(L, n)
    h = float(np.sum(s4c.p * (
        Q1 * (1 - np.array([s4c.h2(min(.5, x)) for x in e1]))
        - s4c.F_EC * Qs * np.array([s4c.h2(min(.5, x)) for x in Es]))))
    if form is None:
        return h
    g, q = [], []
    for k in range(M):
        t = min(max(float(e1[k]), 1e-9), .5)
        hp = np.log2((1 - t) / t)
        b = (1 - so.GAMMA) ** 2 * (1 - s4c.h2(t))
        g += [b + (1 - so.GAMMA) ** 2 * t * hp,
              b - (1 - so.GAMMA) ** 2 * (1 - t) * hp]
        w = s4c.p * float(Qs[k])
        q += [w * (1 - t), w * t]
    return so.second_order(h, g, q, n, form)


def pooled_first_order(L, n):
    """Pooled first-order rate, mean single-photon gain, and event-weighted Var."""
    Q1, e1, Qs, Es = _obs(L, n)
    Q1b, Qsb = Q1.mean(), Qs.mean()
    e1b = float((e1 * Q1).sum() / Q1.sum())
    Esb = float((Es * Qs).sum() / Qs.sum())
    h = Q1b * (1 - s4c.h2(min(.5, e1b))) - s4c.F_EC * Qsb * s4c.h2(min(.5, Esb))
    w = Q1 / Q1.sum()
    return h, Q1b, float((w * (e1 - (w * e1).sum()) ** 2).sum())


def load_ablation():
    path = os.path.join(HERE, "ablation_results.csv")
    if not os.path.exists(path):
        raise SystemExit("ablation_results.csv not found - run ablation_test.py first")
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            if k != "scenario":
                r[k] = float(v)
    return rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.0, "axes.labelsize": 10, "axes.linewidth": 0.9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8.0, "pdf.fonttype": 42, "ps.fonttype": 42})

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.15))

    # ---------------------------------------------------------------- panel (a)
    A = np.linspace(0.0, 0.8, 33)
    S = 1e6                                   # display in units of 1e-6 per pulse
    asy, fin12, fin14, bnd, varw = [], [], [], [], []
    for a in A:
        set_profile(a, N_ASYM)
        h_res = resolved_all(L0, None, N_ASYM)
        h_pool, Q1b, vw = pooled_first_order(L0, N_ASYM)
        asy.append(S * (h_res - h_pool))
        bnd.append(S * (2 / np.log(2)) * Q1b * vw)
        varw.append(vw)
        for n, store in ((1e12, fin12), (1e14, fin14)):
            set_profile(a, n)
            store.append(S * (resolved_all(L0, "simplified", n)
                              - s4c.rate_pooled(L0, "simplified")))
    asy = np.array(asy); fin12 = np.array(fin12); fin14 = np.array(fin14)
    bnd = np.array(bnd); varw = np.array(varw)

    axA.axhline(0.0, color="#5A6068", lw=0.9)
    axA.plot(A, asy, color="#0E7C6B", lw=2.4, label=r"asymptotic ($n\to\infty$)")
    axA.plot(A, fin14, color="#8172B3", lw=1.7, ls=(0, (5, 2)),
             label=r"finite key, $n=10^{14}$")
    axA.plot(A, fin12, color="#4C72B0", lw=1.7, ls=(0, (2, 1.6)),
             label=r"finite key, $n=10^{12}$")
    axA.plot(A, bnd, color="#C0392B", lw=1.6, ls=(0, (1, 1.2)),
             label="strong-concavity bound")

    # where the n = 1e12 curve turns positive: the operating threshold
    pos = np.where(fin12 > 0)[0]
    if len(pos) and pos[0] > 0:
        i = pos[0]
        Acr = np.interp(0.0, [fin12[i - 1], fin12[i]], [A[i - 1], A[i]])
        axA.axvline(Acr, color="#8C9196", lw=0.8, ls=(0, (2, 2)))
        axA.annotate(r"$A_{\mathrm{crit}}$", xy=(Acr, 0),
                     xytext=(Acr + 0.02, 0.70), textcoords=("data", "axes fraction"),
                     fontsize=8, color="#5A6068",
                     arrowprops=dict(arrowstyle="-", color="#8C9196", lw=0.7,
                                     shrinkA=2, shrinkB=2))
        print("  A_crit at n=1e12 : %.3f   (Var_w = %.3e)"
              % (Acr, np.interp(Acr, A, varw)))

    axA.set_xlabel(r"heterogeneity amplitude $A$")
    axA.set_ylabel(r"key-rate gain $\Delta R$  [$10^{-6}$ per pulse]")
    axA.set_xlim(0, 0.8)
    axA.grid(True, ls="-", lw=0.5, color="#E7EAED")
    axA.set_axisbelow(True)
    axA.spines[["top", "right"]].set_visible(False)
    axA.legend(loc="upper left", framealpha=0.95, edgecolor="#A9AEB4",
               fontsize=7.2, handlelength=1.9, borderpad=0.35, labelspacing=0.3)

    # secondary top axis: the MEASURED Var_w.  The analytic beta^2 A^2/8 does
    # not survive the decoy LP, which amplifies e1, so the map is interpolated.
    v4 = varw * 1e4
    sec = axA.secondary_xaxis(
        "top", functions=(lambda x: np.interp(x, A, v4),
                          lambda v: np.interp(v, v4, A)))
    sec.set_xlabel(r"measured $\mathrm{Var}_w(e_{1,\mathrm{ph}})$  [$10^{-4}$]",
                   fontsize=9)
    sec.set_xticks([0, 50, 100, 150, 200, 250])
    sec.tick_params(labelsize=8)

    # ---------------------------------------------------------------- panel (b)
    rows = load_ablation()
    strong = sorted([r for r in rows if r["scenario"] == "quasiperiodic"],
                    key=lambda r: r["L_km"])
    L = np.array([r["L_km"] for r in strong])
    R_B4 = np.array([r["R_B4"] for r in strong])
    R_other = np.array([max(r["R_B1"], r["R_B3"], r["R_B5"]) for r in strong])
    FLOOR = 2e-6
    R_other = np.where(R_other > 0, R_other, FLOOR)

    axB.set_yscale("log")
    axB.fill_between(L, 1e-7, FLOOR, color="#DDE1E6", alpha=0.55, lw=0)
    axB.text(0.98, 0.035, "abort region", transform=axB.transAxes,
             ha="right", fontsize=7.2, color="#5A6068")
    axB.plot(L, R_B4, color="#0E7C6B", lw=2.4, marker="D", ms=4.5,
             markeredgecolor="white", markeredgewidth=0.6,
             label="B4 geometric sectors")
    axB.plot(L, R_other, color="#C0392B", lw=1.6, ls=(0, (4, 2)), marker="o",
             ms=3.6, markerfacecolor="white",
             label="B1, B3, B5 (coincide, abort)")
    axB.set_xlabel("distance (km)")
    axB.set_ylabel("secret key rate (bits per emitted pulse)")
    axB.set_ylim(1e-7, 1e-1)
    axB.grid(True, ls="-", lw=0.5, color="#E7EAED")
    axB.set_axisbelow(True)
    axB.spines[["top"]].set_visible(False)

    axC = axB.twinx()
    for sc, col, mk, lab in (("monotone", "#4C72B0", "s", "monotone drift"),
                             ("quasiperiodic-mild", "#8172B3", "^",
                              "quasi-periodic (mild)")):
        s = sorted([r for r in rows if r["scenario"] == sc], key=lambda r: r["L_km"])
        axC.plot([r["L_km"] for r in s],
                 [100.0 * r["dR_necessity"] / r["R_B5"] if r["R_B5"] > 0 else 0.0
                  for r in s],
                 color=col, lw=1.5, marker=mk, ms=4, markeredgecolor="white",
                 markeredgewidth=0.5, label=lab)
    axC.axhline(0.0, color="#8C9196", lw=0.7)
    axC.set_ylabel(r"necessity gain $R_{B4}/R_{B5}-1$  (%)", fontsize=9)
    axC.spines[["top"]].set_visible(False)
    axC.set_ylim(-1.0, 9.0)

    # two small legends instead of one large block: rates on the left axis,
    # necessity gain on the right axis, each next to the curves it describes
    axB.legend(loc="center left", framealpha=0.92, edgecolor="#A9AEB4",
               fontsize=7.2, handlelength=1.8, borderpad=0.4,
               title="key rate (left axis)", title_fontsize=7.2,
               bbox_to_anchor=(0.0, 0.45))
    axC.legend(loc="upper right", framealpha=0.92, edgecolor="#A9AEB4",
               fontsize=7.2, handlelength=1.8, borderpad=0.4,
               title="necessity gain (right axis)", title_fontsize=7.2)

    # ---------------------------------------------------------------- labels
    for ax, tag in ((axA, "a"), (axB, "b")):
        ax.text(-0.16, 1.14, "(%s)" % tag, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left")

    fig.tight_layout()

    def save_with_fallback(ext):
        target = os.path.join(HERE, f"fig_mechanism.{ext}")
        fallback = os.path.join(HERE, f"fig_mechanism_{ext}_fallback.{ext}")
        try:
            if os.path.exists(target):
                os.remove(target)
            fig.savefig(target, dpi=400, bbox_inches="tight")
            print(f"wrote {target}")
        except PermissionError:
            fig.savefig(fallback, dpi=400, bbox_inches="tight")
            print(f"wrote {fallback} (fallback due to permission issue)")

    for ext in ("pdf", "png"):
        save_with_fallback(ext)

    print("wrote fig_mechanism.pdf / .png")
    print("  panel (a): gain vs A from the CORRECTED decoy pipeline "
          "(stage4_corrected), measured Var_w on the top axis")
    print("  panel (b): rate (left, log) and necessity gain (right, %) vs distance")


if __name__ == "__main__":
    main()
