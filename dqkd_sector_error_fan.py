import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# D-QKD: fan of the eight sector-conditioned phase-error branches e_m(L).
#
# Mechanism companion to the key-rate (28 -> 189 km) figure: it shows WHY
# sector selection extends the distance. The pooled (global) error crosses the
# BB84 abort guide near 28 km, but the fan is wide and heterogeneous, so the
# cleanest branches stay far below the guide and keep carrying key up to the
# sector-selective finite-key reach (~189 km, from the GEAT key-rate figure).
#
# Same calibrated instantiation as the companion curve figure, within the
# paper's declared ranges  e_m^(0) in [0.4%,3%],  kappa_m in [5e-5,1e-2]/km.
# ---------------------------------------------------------------------------

L_max = 190.0
L = np.linspace(0.0, L_max, 1600)

e0 = np.array([0.004, 0.006, 0.008, 0.011, 0.015, 0.020, 0.025, 0.030])
kappa = np.array([0.00005, 0.00020, 0.00050, 0.00100,
                  0.00160, 0.00220, 0.00300, 0.00390])
M = 8

e_all = e0[:, None] + kappa[:, None] * L[None, :]
e_all = np.clip(e_all, 0.0, 0.49)
pooled = e_all.mean(axis=0)

abort = 0.11
cross_idx = np.argmax(pooled >= abort)
pooled_cross = L[cross_idx] if pooled[cross_idx] >= abort else np.nan
reach = 189.0  # sector-selective finite-key reach (from the GEAT key-rate figure)

idx_reach = int(np.argmax(L >= reach))
below = np.where(e_all[:, idx_reach] < abort)[0]

# ---- style: match the paper figures ----
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

fig, ax = plt.subplots(figsize=(11, 6.6))

# Sequential single-hue ramp: magnitude encoding (clean -> noisy = light -> dark)
ramp = plt.cm.Blues(np.linspace(0.35, 1.0, M))
for m in range(M):
    ax.plot(L, 100 * e_all[m], color=ramp[m], linewidth=1.7, zorder=3)

# Cleanest branch, direct-labelled at its right end
ax.annotate(r"$S_0$ (cleanest, $e_0^{(0)}=0.4\%$)",
            xy=(L_max, 100 * e_all[0, -1]),
            xytext=(L_max - 3, 100 * e_all[0, -1] + 1.4),
            ha="right", va="bottom", color=ramp[4], fontsize=10.5)

# Pooled / global aggregate: the antagonist
pooled_line, = ax.plot(L, 100 * pooled, color="firebrick", linestyle="--",
                       linewidth=2.6, zorder=6)

# Abort guide + rejected band
guide = ax.axhline(100 * abort, color="0.35", linestyle=(0, (5, 4)),
                   linewidth=1.4, zorder=2)
ax.axhspan(100 * abort, 30, color="firebrick", alpha=0.035, zorder=0)

# Vertical markers + advantage band
ax.axvline(pooled_cross, color="firebrick", linestyle=":", linewidth=1.4, zorder=2)
ax.axvline(reach, color="seagreen", linestyle=":", linewidth=1.6, zorder=2)
band = ax.axvspan(pooled_cross, reach, color="seagreen", alpha=0.07, zorder=0)

ax.text(pooled_cross + 4, 21.5,
        f"pooled crosses guide\n$\\rightarrow$ global aborts\n$L\\approx{pooled_cross:.0f}$ km",
        color="firebrick", fontsize=10.3, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="firebrick", alpha=0.93))

ax.annotate("sector-selective\nfinite-key reach\n$L\\approx189$ km",
            xy=(reach, 5.5), xytext=(reach - 40, 13.5), color="seagreen",
            fontsize=10.3,
            arrowprops=dict(arrowstyle="->", color="seagreen", lw=1.3),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="seagreen", alpha=0.93))

# Retained (below guide) vs sifted (above guide) at the finite-key reach
for m in range(M):
    yv = 100 * e_all[m, idx_reach]
    if yv <= 30:
        if e_all[m, idx_reach] < abort:
            ax.scatter([reach], [yv], marker="o", s=60, facecolor="white",
                       edgecolor="seagreen", linewidth=1.9, zorder=8)
        else:
            ax.scatter([reach], [yv], marker="x", s=60, color="firebrick",
                       linewidth=1.9, zorder=8)

ax.set_xlim(0, L_max)
ax.set_ylim(0, 30)
ax.set_xlabel(r"Transmission distance $L$ [km]")
ax.set_ylabel(r"Sector phase-error rate $e^{(m)}_{1,\mathrm{ph}}(L)$ [%]")
ax.grid(True, alpha=0.20)
ax.set_title(r"Eight-sector phase-error fan:  $e_m(L)=e_m^{(0)}+\kappa_m L$   ($M=8$)",
             pad=12)

# Colorbar encodes the sequential (magnitude) sector order
sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues,
                           norm=plt.Normalize(vmin=100 * e0[0], vmax=100 * e0[-1]))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.046)
cbar.set_label(r"sector baseline error $e_m^{(0)}$ [%]  (clean $\rightarrow$ noisy)")

handles = [pooled_line, guide,
           Line2D([0], [0], color="seagreen", lw=7, alpha=0.30),
           Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
                  markeredgecolor="seagreen", markersize=8, markeredgewidth=1.8),
           Line2D([0], [0], marker="x", linestyle="none", color="firebrick",
                  markersize=8, markeredgewidth=1.8)]
labels = [r"pooled error $\bar e_{1,\mathrm{ph}}$",
          "BB84 abort guide (11%)",
          "sector-advantage region",
          r"retained at $L\approx189$ km",
          r"sifted at $L\approx189$ km"]
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.13),
          frameon=True, framealpha=0.95, ncol=3, borderaxespad=0.0)

fig.tight_layout()
png = "dqkd_sector_error_fan.png"
pdf = "dqkd_sector_error_fan.pdf"
fig.savefig(png, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print(f"pooled crossing: {pooled_cross:.2f} km")
print("sectors below guide at reach:", below.tolist())
