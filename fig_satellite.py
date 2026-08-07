# -*- coding: utf-8 -*-
r"""
SATELLITE FIGURE - two panels replacing four.

The old fig4_satellite carried three panels (overpass profile, per-sector phase
error, rate comparison) and fig_freespace_satellite a fourth.  Their message is
a single negative control, so two panels suffice:

  (a) where the heterogeneity comes from: the elevation track of the pass, the
      channel efficiency it produces, and the resulting per-sector phase error;
  (b) why it is not enough: the sector-resolved gain is negative at every M and
      becomes more negative as M grows, because the per-sector confidence radius
      widens faster than the variance term pays.

All numbers come from satellite_sector_gain.py, the model of the manuscript's
satellite column (1 GHz clock, 300 s pass, 10-70 deg elevation).
No in-figure titles; panels labelled (a), (b).
Output: fig_satellite.pdf / .png
"""
import os

import numpy as np

import satellite_sector_gain as S

HERE = os.path.dirname(os.path.abspath(__file__))
M_LIST = (2, 4, 8, 16)
M_SHOW = 8


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.0, "axes.labelsize": 10, "axes.linewidth": 0.9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 7.6, "pdf.fonttype": 42, "ps.fonttype": 42})

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.15))

    C_ELEV, C_ETA, C_ERR = "#0E7C6B", "#4C72B0", "#C0392B"

    # ------------------------------------------------------------- panel (a)
    n = M_SHOW * 400
    track = S.elevation_track(n)
    t = np.linspace(0.0, 1.0, len(track))
    axA.plot(t, track, color=C_ELEV, lw=2.0, label="elevation")
    axA.set_xlabel("fraction of the pass")
    axA.set_ylabel(r"elevation $\theta$ (deg)", color=C_ELEV)
    axA.tick_params(axis="y", labelcolor=C_ELEV)
    axA.set_xlim(0, 1)
    axA.set_ylim(0, 78)
    axA.grid(True, ls="-", lw=0.5, color="#E7EAED")
    axA.set_axisbelow(True)
    axA.spines[["top"]].set_visible(False)

    # sector bands
    for m in range(M_SHOW):
        if m % 2 == 0:
            axA.axvspan(m / M_SHOW, (m + 1) / M_SHOW, color="#8C9196", alpha=0.07, lw=0)
    axA.text(0.5, 0.045, r"$M=%d$ equal-duration sectors" % M_SHOW,
             transform=axA.transAxes, ha="center", fontsize=7.4, color="#5A6068")

    axA2 = axA.twinx()
    axA2.semilogy(t, S.eta(track), color=C_ETA, lw=1.6, ls=(0, (5, 2)),
                  label=r"channel efficiency $\eta$")
    axA2.set_ylabel(r"channel efficiency $\eta$", color=C_ETA)
    axA2.tick_params(axis="y", labelcolor=C_ETA)
    axA2.spines[["top"]].set_visible(False)

    # per-sector phase error, drawn as a step over the sector bands
    p, Q1, e1, Qm, Em = S.sector_observables(M_SHOW)
    axA3 = axA.twinx()
    axA3.spines["right"].set_position(("axes", 1.28))
    edges = np.arange(M_SHOW + 1) / M_SHOW
    axA3.stairs(100 * e1, edges, color=C_ERR, lw=1.8, baseline=None)
    axA3.set_ylabel(r"$e_{1,\mathrm{ph}}^{(m)}$ (%)", color=C_ERR)
    axA3.tick_params(axis="y", labelcolor=C_ERR)
    axA3.set_ylim(0, 4.2)
    axA3.spines[["top"]].set_visible(False)

    # ------------------------------------------------------------- panel (b)
    gains_asy, gains_fin = [], []
    for M in M_LIST:
        gains_asy.append(100 * S.analyse(M)["gain"])
        gains_fin.append(100 * S.analyse_finite(M)["gain"])

    x = np.arange(len(M_LIST))
    w = 0.36
    axB.axhline(0.0, color="#5A6068", lw=0.9)
    axB.bar(x - w / 2, gains_asy, w, color="#0E7C6B", alpha=0.85,
            label="asymptotic (first order)")
    axB.bar(x + w / 2, gains_fin, w, color="#C0392B", alpha=0.85,
            label=r"finite key, $n=3\times10^{11}$")
    for xi, g in zip(x + w / 2, gains_fin):
        axB.annotate("%.2f%%" % g, xy=(xi, g), xytext=(0, -11),
                     textcoords="offset points", ha="center", fontsize=7.4,
                     color="#7A2018")
    axB.set_xticks(x)
    axB.set_xticklabels([str(m) for m in M_LIST])
    axB.set_xlabel(r"number of sectors $M$")
    axB.set_ylabel(r"sector-resolved gain  $R_{\mathrm{sec}}/R_{\mathrm{pool}}-1$  (%)")
    axB.set_ylim(-3.1, 0.9)
    axB.grid(True, axis="y", ls="-", lw=0.5, color="#E7EAED")
    axB.set_axisbelow(True)
    axB.spines[["top", "right"]].set_visible(False)
    axB.legend(loc="lower left", framealpha=0.95, edgecolor="#A9AEB4")

    for ax, tag in ((axA, "a"), (axB, "b")):
        ax.text(-0.17, 1.10, "(%s)" % tag, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "fig_satellite." + ext), dpi=400,
                    bbox_inches="tight")
    print("wrote fig_satellite.pdf / .png")
    print("  M   asymptotic %   finite-key %")
    for M, a, f in zip(M_LIST, gains_asy, gains_fin):
        print("  %-3d %11.2f %14.2f" % (M, a, f))


if __name__ == "__main__":
    main()
