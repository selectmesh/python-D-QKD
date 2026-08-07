# -*- coding: utf-8 -*-
r"""
METHODS FIGURE - two panels replacing three.

Panel (a) is the cross-method comparison (formerly fig_merged_comparison):
both source models on one distance axis, against the published pooled proofs.

Panel (b) is the distinctive content of the old fig_qubit_adaptive: the
certified finite-size sector gain  dR = R_B4* - R_B1  as a function of loss, for
several block lengths.  Its other panel plotted rate versus loss, which panel (a)
already carries, so it is dropped rather than duplicated.

The two panels share an abscissa up to the fixed fibre attenuation,
loss(dB) = alpha * L with alpha = 0.2 dB/km, so panel (b) carries distance on a
secondary axis and the reader can move between the panels directly.

No in-figure titles; panels labelled (a), (b).
Output: fig_methods.pdf / .png
"""
import os

import numpy as np

import min_entropy_secondorder as so
import stage4_corrected as s4c

HERE = os.path.dirname(os.path.abspath(__file__))
N = 1e12
FLOOR = 1e-9
ALPHA_DB = so.ALPHA_DB


def curve(fn, Lv, cap=None):
    R = np.array([max(fn(L), 0.0) if (cap is None or L <= cap) else 0.0 for L in Lv])
    return np.where(R > FLOOR, R, np.nan)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.0, "axes.labelsize": 10, "axes.linewidth": 0.9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 7.0, "pdf.fonttype": 42, "ps.fonttype": 42})

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.35))

    TEAL, TEAL2 = "#0E7C6B", "#5B8C7B"
    PURP, PURP2 = "#6A3D9A", "#9E7BB5"
    RED, GREY = "#C0392B", "#8C9196"

    # ------------------------------------------------------------- panel (a)
    Lv = np.arange(1.0, 210.0, 1.0)
    R_pool = curve(lambda L: so.rate_pooled(L, N, "simplified"), Lv)
    R_simp = curve(lambda L: so.rate_at(L, N, "simplified", True), Lv)
    R_var = curve(lambda L: so.rate_at(L, N, "variance", True), Lv)
    D_pool = curve(lambda L: s4c.rate_pooled(L, "variance"), Lv, cap=130)
    D_rig = curve(lambda L: s4c.rate_select(L, "simplified"), Lv, cap=130)
    D_var = curve(lambda L: s4c.rate_select(L, "variance"), Lv, cap=130)
    plob = -np.log2(1 - 10 ** (-ALPHA_DB * Lv / 10.0))

    axA.semilogy(Lv, plob, color=GREY, lw=1.0, ls=(0, (1.2, 2.2)))
    axA.text(150, 1.9 * np.interp(150, Lv, plob), "PLOB", fontsize=7.5,
             color="#666C73", rotation=-20, rotation_mode="anchor")
    axA.fill_between(Lv, FLOOR, np.nan_to_num(R_var), where=(Lv >= 61),
                     color=TEAL, alpha=0.06, lw=0)

    axA.semilogy(Lv, R_pool, color=RED, lw=1.8, ls=(0, (5, 2)),
                 label="7 published proofs, single-photon")
    axA.semilogy(Lv, D_pool, color=RED, lw=1.3, ls=(0, (1.4, 1.6)),
                 label="pooled, decoy (WCP)")
    axA.semilogy(Lv, R_simp, color=TEAL2, lw=1.4, ls=(0, (1, 1.1)),
                 label="this work, single-photon (simplified)")
    axA.semilogy(Lv, R_var, color=TEAL, lw=2.4, marker="o", ms=4.2, markevery=16,
                 markeredgecolor="white", markeredgewidth=0.5,
                 label="this work, single-photon (item 5b)")
    axA.semilogy(Lv, D_rig, color=PURP2, lw=1.4, ls=(0, (1, 1.1)),
                 label="this work, decoy (rigorous)")
    axA.semilogy(Lv, D_var, color=PURP, lw=2.4, marker="s", ms=3.8, markevery=16,
                 markeredgecolor="white", markeredgewidth=0.5,
                 label="this work, decoy (item 5b)")

    for x, c in ((18, RED), (61, RED), (113, PURP2), (122, PURP),
                 (158, TEAL2), (188, TEAL)):
        axA.axvline(x, color=c, lw=0.7, ls=(0, (1.5, 2.5)), alpha=0.7)
    for x, c, dy in ((18, RED, 1.3e-9), (61, RED, 1.3e-9), (113, PURP2, 1.3e-9),
                     (122, PURP, 4.0e-9), (158, TEAL2, 1.3e-9), (188, TEAL, 4.0e-9)):
        axA.annotate("%d" % x, xy=(x, dy), xytext=(x + 1.5, dy), fontsize=6.8,
                     color=c, ha="left", va="bottom", fontweight="bold")

    axA.set_xlim(0, 210)
    axA.set_ylim(FLOOR, 1)
    axA.set_xlabel("transmission distance $L$ (km)")
    axA.set_ylabel("certified secret key rate (bits per pulse)")
    axA.grid(True, which="major", ls="-", lw=0.5, color="#E7EAED")
    axA.set_axisbelow(True)
    axA.spines[["top", "right"]].set_visible(False)
    axA.legend(loc="upper right", bbox_to_anchor=(1.2, 1.13), framealpha=0.95,
               edgecolor="#A9AEB4", handlelength=2.0, borderpad=0.35,
               labelspacing=0.3)

    # ------------------------------------------------------------- panel (b)
    # When does splitting the data actually pay?  The first-order gain does not
    # depend on the block length, but the cost of estimating each sector from
    # n/M rounds does, so the net gain crosses zero at an M-dependent block
    # length and then saturates at the asymptotic ceiling.  Evaluated on a
    # conservative fibre drift (kappa_max = 1e-4/km), not on the declared model
    # of panel (a).
    KMAX, KRATIO, L0 = 1.0e-4, 200.0, 50.0
    ns = np.logspace(10, 17, 22)

    def configure(M, n):
        mid = (np.arange(M) + 0.5) / M
        s4c.M, s4c.p, s4c.N = M, 1.0 / M, n
        s4c.e0m = 0.004 + (0.030 - 0.004) * mid
        kmin = KMAX / KRATIO
        s4c.kapm = kmin * (KMAX / kmin) ** mid

    keep = (s4c.M, s4c.p, s4c.N, s4c.e0m.copy(), s4c.kapm.copy())
    for M, col, mk in ((2, "#4C72B0", "s"), (4, TEAL, "o"),
                       (8, PURP, "^"), (16, "#B8860B", "D")):
        g = []
        for n in ns:
            configure(M, n)
            a = s4c.rate_select(L0, "simplified")
            b = s4c.rate_pooled(L0, "simplified")
            g.append(100.0 * (a / b - 1.0) if b > 0 else np.nan)
        axB.semilogx(ns, g, color=col, lw=1.8, marker=mk, ms=3.6, markevery=3,
                     markeredgecolor="white", markeredgewidth=0.5,
                     label=r"$M=%d$" % M)
    s4c.M, s4c.p, s4c.N, s4c.e0m, s4c.kapm = keep

    axB.axhline(0.0, color="#5A6068", lw=1.0)
    axB.fill_between([ns[0], ns[-1]], -45, 0, color="#C0392B", alpha=0.05, lw=0)
    axB.text(0.04, 0.30, "global pooling favoured", transform=axB.transAxes,
             fontsize=7.4, color="#9A4038")
    axB.text(0.04, 1, "sector-conditioned analysis favoured", transform=axB.transAxes,
             fontsize=7.4, color="#0A5A4E", va="top")

    axB.set_xlabel(r"block length $N$ (emitted pulses)")
    axB.set_ylabel(r"sector gain $R_{\mathrm{sec}}/R_{\mathrm{pool}}-1$ (%)")
    axB.set_xlim(ns[0], ns[-1])
    axB.set_ylim(-42, 6)
    axB.grid(True, which="major", ls="-", lw=0.5, color="#E7EAED")
    axB.grid(True, which="minor", axis="x", ls="-", lw=0.35, color="#F3F5F7")
    axB.set_axisbelow(True)
    axB.spines[["top", "right"]].set_visible(False)
    axB.legend(loc="center right", framealpha=0.95, edgecolor="#A9AEB4",
               ncol=1, handlelength=1.8, borderpad=0.4,
               title=r"conservative drift, $L=50$ km", title_fontsize=6.8)

    for ax, tag in ((axA, "a"), (axB, "b")):
        ax.text(-0.17, 1.13, "(%s)" % tag, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "fig_methods." + ext), dpi=400,
                    bbox_inches="tight")
    print("wrote fig_methods.pdf / .png")
    print("  panel (a): cross-method comparison, both source models")
    print("  panel (b): certified sector gain vs loss, with distance on top")


if __name__ == "__main__":
    main()
