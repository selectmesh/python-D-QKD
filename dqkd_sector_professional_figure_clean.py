import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =============================================================================
# D-QKD: photon transmission and sector-conditioned phase-error evolution
# =============================================================================
# Paper-grounded equations and ranges:
#   eta_ch(L) = 10^(-alpha L / 10), alpha = 0.2 dB/km
#   e_m(L) = e_m^(0) + kappa_m L
#   e_m^(0) in [0.4%, 3%]
#   kappa_m in [5e-5, 1e-2] per km
#   M = 8 sectors
#
# The paper provides ranges, not the explicit 8-element arrays. The arrays below
# are one admissible calibrated instantiation chosen so that the pooled error
# reaches the BB84 abort region near 28 km, consistent with the paper's dynamic
# benchmark. The active public sector schedule is deliberately interleaved to
# reflect the nD-HCS/PSB scheduling idea rather than contiguous time bins.
# =============================================================================

# Distance grid
L_max = 190.0
L = np.linspace(0.0, L_max, 1600)
alpha = 0.2
eta_det = 0.85
p_dark = 1e-8

# Photon transmission / single-photon detection proxy
eta_ch = 10.0 ** (-alpha * L / 10.0)
Q1 = eta_det * eta_ch + p_dark

# One explicit admissible eight-sector dynamic-error instantiation
e0 = np.array([0.004, 0.006, 0.008, 0.011, 0.015, 0.020, 0.025, 0.030])
kappa = np.array([0.00005, 0.00030, 0.00080, 0.00180,
                  0.00350, 0.00500, 0.00650, 0.00890])
M = 8

# Sector-resolved phase-error branches e_m(L)
e_all = e0[:, None] + kappa[:, None] * L[None, :]
e_all = np.clip(e_all, 0.0, 0.49)

# Pooled error for equal sector weights
pooled_error = e_all.mean(axis=0)

# Deterministic interleaved public sector schedule across the distance axis.
# Every sector appears twice; the order is non-contiguous by design.
sector_sequence = np.array([0, 4, 1, 6, 2, 7, 3, 5,
                            0, 2, 6, 1, 7, 4, 3, 5])
num_windows = len(sector_sequence)
edges = np.linspace(0.0, L_max, num_windows + 1)

active_sector = np.zeros_like(L, dtype=int)
for j, m in enumerate(sector_sequence):
    if j < num_windows - 1:
        mask = (L >= edges[j]) & (L < edges[j + 1])
    else:
        mask = (L >= edges[j]) & (L <= edges[j + 1])
    active_sector[mask] = m

active_error = e_all[active_sector, np.arange(L.size)]

# BB84 asymptotic phase-error threshold, used here only as a visual guide.
abort_threshold = 0.11
cross_idx = np.argmax(pooled_error >= abort_threshold)
pooled_crossing = L[cross_idx] if pooled_error[cross_idx] >= abort_threshold else np.nan

# -------------------------
# Publication-style layout
# -------------------------
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.1,
    "figure.dpi": 180,
    "savefig.dpi": 350,
})

fig, ax_photon = plt.subplots(figsize=(15, 8.5))
ax_error = ax_photon.twinx()

# Muted sector colours for the public-sector schedule bands.
sector_colours = plt.cm.tab20(np.linspace(0.02, 0.72, M))
for j, m in enumerate(sector_sequence):
    ax_photon.axvspan(edges[j], edges[j + 1],
                      color=sector_colours[m], alpha=0.085, linewidth=0)
    x_mid = 0.5 * (edges[j] + edges[j + 1])
    ax_error.text(x_mid, 47.5, rf"$S_{m}$",
                  ha="center", va="center", fontsize=10.5,
                  color=sector_colours[m], fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.18",
                            facecolor="white", edgecolor=sector_colours[m],
                            linewidth=0.8, alpha=0.92))

# Photon movement / survival curve
photon_line, = ax_photon.semilogy(
    L, Q1,
    linewidth=3.0,
    label=r"Detected single-photon fraction $Q_1(L)$",
    zorder=4,
)

# Photon markers along the propagation curve
marker_distances = np.arange(0, 191, 15)
marker_idx = np.searchsorted(L, marker_distances)
marker_idx = np.clip(marker_idx, 0, L.size - 1)
ax_photon.scatter(L[marker_idx], Q1[marker_idx], s=42,
                  facecolor="white", edgecolor=photon_line.get_color(),
                  linewidth=1.6, zorder=5)

# Sector-conditioned error curve: red and dynamically switches sector branch
sector_error_line, = ax_error.plot(
    L, 100.0 * active_error,
    color="crimson", linewidth=2.7,
    label=r"Active sector error $e^{(m(L))}_{1,\mathrm{ph}}(L)$",
    zorder=7,
)

# Pooled/global phase-error curve
pooled_line, = ax_error.plot(
    L, 100.0 * pooled_error,
    color="firebrick", linestyle="--", linewidth=1.8,
    label=r"Pooled error $\bar e_{1,\mathrm{ph}}(L)$",
    zorder=6,
)

# Abort threshold and rejected region
threshold_line = ax_error.axhline(
    100.0 * abort_threshold,
    color="0.25", linestyle=(0, (5, 4)), linewidth=1.5,
    label="BB84 abort guide (11%)",
    zorder=3,
)
ax_error.axhspan(100.0 * abort_threshold, 50.0,
                 color="crimson", alpha=0.035, zorder=1)
ax_error.text(187, 47.8, "high-error sectors are publicly sifted out",
              ha="right", va="top", color="crimson", fontsize=11,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor="crimson", alpha=0.90))

# Mark the pooled crossing near 28 km
if np.isfinite(pooled_crossing):
    ax_error.axvline(pooled_crossing, color="firebrick", linestyle=":",
                     linewidth=1.4, alpha=0.9)
    ax_error.annotate(
        "Pooled error enters abort region\n" + rf"$L \approx {pooled_crossing:.1f}$ km",
        xy=(pooled_crossing, 100.0 * abort_threshold),
        xytext=(pooled_crossing + 15, 17.5),
        color="firebrick",
        arrowprops=dict(arrowstyle="->", color="firebrick", lw=1.4),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="firebrick", alpha=0.94),
        zorder=10,
    )

# Highlight representative sector points at the middle of each window
for j, m in enumerate(sector_sequence):
    x_mid = 0.5 * (edges[j] + edges[j + 1])
    idx = np.searchsorted(L, x_mid)
    y_mid = 100.0 * active_error[idx]
    marker = "o" if active_error[idx] < abort_threshold else "x"
    ax_error.scatter([x_mid], [y_mid], s=44,
                     marker=marker, color="crimson",
                     linewidth=1.5, zorder=8)

# Axes
ax_photon.set_xlim(0, L_max)
ax_photon.set_ylim(1e-4, 1.05)
ax_error.set_ylim(0, 50)
ax_photon.set_xlabel(r"Transmission distance $L$ [km]")
ax_photon.set_ylabel(r"Detected single-photon fraction $Q_1(L)$")
ax_error.set_ylabel(r"Sector phase-error rate $e^{(m)}_{1,\mathrm{ph}}(L)$ [\%]",
                    color="crimson")
ax_error.tick_params(axis="y", colors="crimson")
ax_error.spines["right"].set_color("crimson")
ax_photon.grid(True, which="both", alpha=0.22)

# Title and explanatory subtitle
fig.suptitle(
    "Photon transmission and sector-conditioned phase-error evolution",
    y=0.975, fontsize=18, fontweight="semibold"
)
ax_photon.set_title(
    r"Public PSB sector schedule selects the active branch "
    r"$e_m(L)=e_m^{(0)}+\kappa_mL$ while photon transmission follows "
    r"$\eta_{\mathrm{ch}}(L)=10^{-\alpha L/10}$",
    fontsize=12.5, pad=18
)

# Unified legend
handles = [photon_line, sector_error_line, pooled_line, threshold_line]
labels = [h.get_label() for h in handles]
ax_photon.legend(handles, labels, loc="lower left", frameon=True,
                 framealpha=0.96, ncol=2)

# Compact interpretation and scientific note below the axis
fig.text(
    0.08, 0.055,
    "Blue: detected single-photon fraction decreases with distance.  "
    "Red: the public PSB label switches the active sector-error branch.  "
    "○ retained sector; × rejected sector.",
    ha="left", va="bottom", fontsize=10.2
)
fig.text(
    0.08, 0.025,
    "The paper specifies ranges rather than explicit per-sector arrays. The plotted "
    "$e_m^{(0)}$ and $\\kappa_m$ values are one admissible calibrated instantiation "
    "within those ranges, selected to reproduce the reported pooled-error crossing near 28 km.",
    ha="left", va="bottom", fontsize=9.5
)

fig.subplots_adjust(left=0.08, right=0.90, top=0.88, bottom=0.16)

png_path = "dqkd_photon_sector_error_curve.png"
pdf_path = "dqkd_photon_sector_error_curve.pdf"
fig.savefig(png_path, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
print(f"Pooled crossing: {pooled_crossing:.3f} km")