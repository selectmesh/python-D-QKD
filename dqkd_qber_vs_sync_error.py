import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# QBER versus residual sector-compensation error.
#
# Architecture Eq. (detect_prob):
#   P_j = [1 + (-1)^j V cos(theta_k + phi_{b,w} - phi_B - hat_theta_k)] / 2
# With Delta = theta_k - hat_theta_k the residual GP-Comp error, the error port
# probability of a correctly sifted round is
#   QBER(Delta) = [1 - V cos(Delta)] / 2 .
# The visibility is fixed by the declared intrinsic QBER of the fiber scenario,
# e_d = 1.5%  =>  V = 1 - 2 e_d = 0.97.
#
# This makes Assumption 2 ("residual sector misassignment must remain
# sufficiently small") quantitative, and gives the cost -- not merely the
# probability -- of a sector-assignment fault entering eps_sec^(M).
# ---------------------------------------------------------------------------

e_d = 0.015                 # intrinsic QBER, fiber scenario
V = 1.0 - 2.0 * e_d         # 0.97
abort = 0.11                # BB84 abort guide

d_deg = np.linspace(-95, 95, 2000)
d = np.deg2rad(d_deg)
qber = (1.0 - V * np.cos(d)) / 2.0

# Tolerance window: |Delta| where QBER stays below the abort guide
d_tol = np.rad2deg(np.arccos((1.0 - 2.0 * abort) / V))

# One-sector misassignment for several M: Delta = 2*pi/M
Ms = [4, 8, 16]
mis = {M: 360.0 / M for M in Ms}

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.1,
    "figure.dpi": 180,
    "savefig.dpi": 350,
})

fig, ax = plt.subplots(figsize=(10.5, 6.2))

ax.axhspan(100 * abort, 55, color="firebrick", alpha=0.05, zorder=0)
tol_band = ax.axvspan(-d_tol, d_tol, color="seagreen", alpha=0.09, zorder=0)

curve, = ax.plot(d_deg, 100 * qber, color="#1f4e79", linewidth=2.4, zorder=5)
guide = ax.axhline(100 * abort, color="0.35", linestyle=(0, (5, 4)),
                   linewidth=1.4, zorder=3)
floor = ax.axhline(100 * e_d, color="seagreen", linestyle=":",
                   linewidth=1.4, zorder=3)

# One-sector misassignment markers
for M in Ms:
    x = mis[M]
    if x > 95:
        continue
    y = 100 * (1 - V * np.cos(np.deg2rad(x))) / 2
    colour = "firebrick" if y > 100 * abort else "seagreen"
    ax.scatter([x], [y], s=70, marker="D", color=colour,
               edgecolor="white", linewidth=1.2, zorder=8)
    side = -1 if x > 60 else 1
    ax.annotate(rf"$M={M}$: {y:.1f}%",
                xy=(x, y), xytext=(x + side * 4, y + 3.2),
                ha="right" if side < 0 else "left",
                fontsize=10.2, color=colour, zorder=9,
                bbox=dict(boxstyle="round,pad=0.24", fc="white",
                          ec=colour, alpha=0.93))

ax.text(-92, 100 * e_d + 1.6, rf"intrinsic floor $e_d={100*e_d:.1f}$%",
        ha="left", va="bottom", fontsize=10.2, color="seagreen")

ax.text(0, 43, rf"secure tolerance  $|\Delta|\leq{d_tol:.1f}^\circ$",
        ha="center", va="center", fontsize=11, color="seagreen", zorder=9,
        bbox=dict(boxstyle="round,pad=0.32", fc="white",
                  ec="seagreen", alpha=0.94))

ax.set_xlim(-95, 95)
ax.set_ylim(0, 55)
ax.set_xlabel(r"Residual sector-compensation error "
              r"$\Delta=\theta_k-\hat\theta_k$ [deg]")
ax.set_ylabel(r"QBER [%]")
ax.set_title(r"QBER cost of imperfect sector compensation:  "
             r"$\mathrm{QBER}(\Delta)=\frac{1}{2}\left[1-V\cos\Delta\right]$,  $V=0.97$",
             pad=12)
ax.grid(True, alpha=0.20)
ax.set_xticks([-90, -60, -45, -30, 0, 30, 45, 60, 90])

handles = [curve, guide, floor,
           Line2D([0], [0], color="seagreen", lw=7, alpha=0.30),
           Line2D([0], [0], marker="D", linestyle="none", color="firebrick",
                  markersize=8, markeredgecolor="white")]
labels = [r"QBER$(\Delta)$",
          "BB84 abort guide (11%)",
          r"intrinsic floor $e_d$",
          "secure tolerance window",
          r"one-sector misassignment ($\Delta=2\pi/M$)"]
ax.legend(handles, labels, loc="upper center", frameon=True, framealpha=0.95,
          ncol=2)

fig.tight_layout()
png, pdf = "dqkd_qber_vs_sync_error.png", "dqkd_qber_vs_sync_error.pdf"
fig.savefig(png, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png, "|", pdf)
print(f"V = {V:.3f},  tolerance = +/-{d_tol:.2f} deg")
for M in Ms:
    y = 100 * (1 - V * np.cos(np.deg2rad(mis[M]))) / 2
    print(f"  M={M:2d}: one-sector misassignment {mis[M]:5.1f} deg -> QBER {y:5.2f}%"
          f"  ({'ABOVE' if y > 100*abort else 'below'} guide)")
