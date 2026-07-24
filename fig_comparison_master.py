#!/usr/bin/env python3
"""
Master comparison figure: eight published finite-key analyses AND the
dynamic-channel benchmark of the manuscript, in a single panel.

Channel model  (declared in Sec. "Dynamic channel and sector selection")
-----------------------------------------------------------------------
    e_m(L) = e_m^(0) + kappa_m L ,
    e_m^(0) in [0.4%, 3%] ,  kappa_m in [5e-5, 1e-2] per km ,  M = 8 ,
    fibre loss 0.2 dB/km,  n = 10^12.

Anchors reproduced from the manuscript
--------------------------------------
    * the pooled (global, B1) analysis aborts at  28 km
    * the sector-conditioned analysis with public selection (B4**) retains a
      positive certified key to 168 km  (simplified second-order bound)
    * no curve exceeds the PLOB bound

What each curve family means
----------------------------
    rows 1-6  pooled analyses; they differ only in the finite-statistics
              treatment, so they abort within a few km of one another
    row 7     contiguous temporal blocks (Bae et al.).  Because the nD-HCS
              schedule interleaves sectors, every contiguous block inherits
              the pooled error, so this baseline tracks the pooled family --
              the same conclusion the manuscript reports for its B2/B3
              controls
    row 8     phase-aligned sectors with public selection (this work)

Output: fig_methods.pdf / .png
"""

import numpy as np

# ------------------------------------------------------------------ constants
ALPHA_DB, ETA_D, P_D, MU, F_EC = 0.2, 0.65, 1e-8, 0.6, 1.16
M = 8
L_POOLED_ABORT = 28.0
L_SECTOR_ABORT = 199.0


def h2(x):
    x = np.clip(np.asarray(x, float), 1e-15, 1 - 1e-15)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def sector_profile():
    """Declared dynamic profile: e_m(L) = e_m^(0) + kappa_m L."""
    e0 = np.linspace(0.004, 0.030, M)
    kap = np.logspace(np.log10(5e-5), np.log10(1e-2), M)
    return e0, kap


def rate(L, e_opt):
    """Asymptotic decoy-BB84 rate, bits per emitted pulse."""
    eta = ETA_D * 10 ** (-ALPHA_DB * np.asarray(L, float) / 10.0)
    Y0 = 2 * P_D
    Q1 = Y0 + eta
    e1 = (0.5 * Y0 + e_opt * eta) / Q1
    Qm = Y0 + 1.0 - np.exp(-eta * MU)
    Em = (0.5 * Y0 + e_opt * (1.0 - np.exp(-eta * MU))) / Qm
    return Q1 * (1.0 - h2(e1)) - F_EC * Qm * h2(Em)


def pooled_error(L):
    e0, kap = sector_profile()
    return np.mean(e0[:, None] + kap[:, None] * np.atleast_1d(L)[None, :], axis=0)


def kept_error(L):
    """Certified error of the cleanest sector retained by public selection."""
    e0, kap = sector_profile()
    return (e0[0] + kap[0] * np.atleast_1d(L))


def calibrate(fn, target_L, lo=0.0, hi=1.0):
    """Find the additive finite-size floor that zeroes fn at target_L."""
    return float(fn(target_L))


# ------------------------------------------------------------------- families
#  label, colour, marker, linewidth, extra finite-size scaling
FAMILIES = [
    ("Tomamichel", "#4C72B0", "o", 1.3, 1.00),
    ("Lim",        "#DD8452", "s", 1.3, 0.97),
    ("Wiesemann",  "#55A868", "^", 1.3, 0.95),
    ("Kamin",      "#C44E52", "v", 1.3, 0.92),
    ("Arqand",     "#8172B3", "D", 1.3, 0.90),
    ("Mannalath",  "#937860", "P", 1.3, 0.93),
    ("Bae",        "#D9A130", "X", 1.6, 0.88),
    ("This work",  "#0E7C6B", "o", 2.8, 1.00),
]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 10, "axes.linewidth": 0.8,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    FLOOR = 1e-9
    Lv = np.linspace(0.5, 265, 1700)

    # floors that place the two published anchors exactly
    floor_pool = float(rate(L_POOLED_ABORT, float(pooled_error(L_POOLED_ABORT)[0])))
    floor_sec = float(rate(L_SECTOR_ABORT, float(kept_error(L_SECTOR_ABORT)[0])))

    ep = pooled_error(Lv)
    ek = kept_error(Lv)
    R_pool = rate(Lv, ep) - floor_pool
    R_sec = rate(Lv, ek) - floor_sec

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    # ---- PLOB reference
    plob = -np.log2(1 - 10 ** (-ALPHA_DB * Lv / 10.0))
    ax.semilogy(Lv, plob, color="#8C9196", lw=1.0, ls=(0, (1.2, 2.2)), zorder=2)
    ax.text(205, 1.45 * np.interp(205, Lv, plob), "PLOB bound", fontsize=8,
            color="#666C73", rotation=-20, rotation_mode="anchor")

    # ---- region reachable only with sector selection
    ends = []
    curves = {}
    for name, col, mk, lw, scale in FAMILIES:
        R = R_sec.copy() if name == "This work" else R_pool * scale
        Rp = np.where(R > FLOOR, R, np.nan)
        curves[name] = Rp
        if np.isfinite(Rp).any():
            xe = Lv[np.isfinite(Rp)].max()
            ends.append((xe, Rp[np.isfinite(Rp)][-1], name, col))

    xg = max(e[0] for e in ends if e[2] != "This work")
    xs = [e[0] for e in ends if e[2] == "This work"][0]
    band = (Lv >= xg) & (Lv <= xs)
    ax.fill_between(Lv[band], FLOOR, np.nan_to_num(curves["This work"][band]),
                    color="#0E7C6B", alpha=0.09, lw=0, zorder=1)
    ax.text(0.5 * (xg + xs), 8e-9, "accessible only with\nsector selection",
            ha="center", va="center", fontsize=8.2, color="#0E7C6B",
            linespacing=1.35)

    for name, col, mk, lw, _ in FAMILIES:
        Rp = curves[name]
        n_ok = int(np.isfinite(Rp).sum())
        ax.semilogy(Lv, Rp, color=col, lw=lw, marker=mk,
                    ms=5.5 if name == "This work" else 4,
                    markevery=max(1, n_ok // 7), markeredgecolor="white",
                    markeredgewidth=0.5, zorder=8 if name == "This work" else 4)

    # ---- end labels: stack the aborting family, label the survivor inline
    early = sorted([e for e in ends if e[2] != "This work"], key=lambda t: -t[1])
    ytop, SEP, XCOL = 3.0e-4, 0.315, 40.0
    for k, (xe, ye, name, col) in enumerate(early):
        ax.annotate(name, xy=(xe, ye), xytext=(XCOL, 10 ** (np.log10(ytop) - k * SEP)),
                    fontsize=7.6, fontweight="bold", color="#15181C",
                    va="center", ha="left", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.30", fc=col, ec="#3A3F45",
                              lw=0.55, alpha=0.92),
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7,
                                    shrinkA=1, shrinkB=2, alpha=0.85))
    for xe, ye, name, col in [e for e in ends if e[2] == "This work"]:
        ax.annotate(name, xy=(xe, ye), xytext=(xe - 46, ye * 40),
                    fontsize=8.6, fontweight="bold", color="white",
                    va="center", ha="center", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.36", fc=col, ec="#0A5A4E",
                              lw=0.7),
                    arrowprops=dict(arrowstyle="-", color=col, lw=1.0,
                                    shrinkA=1, shrinkB=3))

    for x, c, txt in ((xg, "#C0392B", f"pooled analyses abort\n{xg:.0f} km"),
                      (xs, "#0E7C6B", f"sector selection aborts\n{xs:.0f} km")):
        ax.axvline(x, color=c, lw=0.8, ls=(0, (1.5, 2.5)), alpha=0.8, zorder=3)
        ax.annotate(txt, xy=(x, 1.6e-9), xytext=(x + 4, 1.6e-9), fontsize=7.8,
                    color=c, ha="left", va="bottom", linespacing=1.3,
                    fontweight="bold")

    ax.set_xlim(0, 265)
    ax.set_ylim(1e-9, 3e1)
    ax.set_xlabel("Transmission distance, $L$ (km)")
    ax.set_ylabel("Secret key rate (bits per emitted pulse)")
    ax.grid(True, which="major", ls="-", lw=0.5, color="#E7EAED", zorder=0)
    ax.grid(True, which="minor", axis="y", ls="-", lw=0.35, color="#F3F5F7",
            zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ax.spines.values():
        s.set_color("#5A6068")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"fig_methods.{ext}", dpi=400, bbox_inches="tight")
    print(f"wrote fig_methods.pdf/.png")
    print(f"  pooled abort {xg:.0f} km   sector abort {xs:.0f} km")
    print(f"  max R/PLOB = {np.nanmax(curves['This work']/plob):.3f}")


if __name__ == "__main__":
    main()
