#!/usr/bin/env python3
"""
Nature-style figure: sector-resolved vs global coarse-grained analysis.
ALL curves computed from the authors' own model (Fiber_channel.py):
7D hyperchaotic driver -> sector QBER profile -> decoy-BB84 rate model.

Panels
  a  order-of-operations pipeline: average-then-entropy vs entropy-then-average
  b  the data both analyses receive: bimodal sector QBER, 11% line, pooled bar
  c  strong concavity on h2 with the model's real numbers (gap = 0.093 bits)
  d  HERO: key rate vs distance -- 14-20x rate gap + 49 km distance extension
  e  falsifiability: gain vs sector variance, through the origin, above bound
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------- style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 6.2,
    "axes.linewidth": 0.5,
    "axes.labelsize": 6.6,
    "axes.titlesize": 6.8,
    "xtick.labelsize": 5.8, "ytick.labelsize": 5.8,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "legend.fontsize": 5.6,
    "mathtext.fontset": "dejavusans",
    "figure.dpi": 200,
})
# Okabe-Ito
C_GLB = "#0072B2"   # global coarse-grained
C_SEC = "#D55E00"   # sector-resolved
C_OK  = "#009E73"   # pure sectors / gain
C_EXT = "#E69F00"   # distance-extension band
C_TOX = "#CC3311"   # toxic sectors
C_GRY = "#555555"

# ================================================================= model
def nD_HC(t, X, p):
    n = len(X); dX = np.zeros(n)
    for i in range(n):
        dX[i] = (p[i % len(p)] * X[(i + 1) % n]
                 - p[(i + 1) % len(p)] * X[(i - 2) % n]) * X[(i - 1) % n]
    return dX

def gen_HC():
    p = [40., 60., 8., 20., 0.1, 77., 10.]
    X0 = [1., 1., 2., 2., 1., 1., 0.1]
    t = np.linspace(0, 10., 10000)
    sol = solve_ivp(nD_HC, (0, 10.), X0, args=(p,), t_eval=t,
                    method="RK45", rtol=1e-10)
    return sol.y[:, 2000:]

ALPHA, ETA_D, P_D = 0.2, 0.65, 1e-8
F_EC, MU, M, qs = 1.16, 0.6, 8, 1.0

def h2(x):
    x = np.clip(np.asarray(x, float), 1e-15, 1 - 1e-15)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

def channel_yields(L, e_opt):
    eta = ETA_D * 10 ** (-ALPHA * L / 10.0)
    Y0, e0 = 2 * P_D, 0.5
    Qmu = Y0 + 1.0 - np.exp(-eta * MU)
    Emu = np.clip((e0 * Y0 + e_opt * (1 - np.exp(-eta * MU))) / max(Qmu, 1e-20),
                  1e-4, 0.499)
    Q1 = Y0 + eta
    e1 = np.clip((e0 * Y0 + e_opt * eta) / max(Q1, 1e-20), 1e-4, 0.499)
    return Q1, e1, Qmu, Emu

def sector_errors(hm, intensity=1.0):
    raw = np.zeros(M)
    for m in range(M):
        s = np.sign(hm[m % len(hm)] - np.mean(hm))
        raw[m] = s * np.abs(hm[m % len(hm)]) ** 0.2
    raw = (raw - raw.min()) / (raw.max() - raw.min())
    return 0.005 + intensity * raw * (0.260 - 0.005)

def single_rate(L, e):
    Q1, e1, Qmu, Emu = channel_yields(L, e)
    return max(0.0, qs * (Q1 * (1 - h2(e1)) - F_EC * Qmu * h2(Emu)))

hc = gen_HC(); hm = hc[1, :M]
e_m = sector_errors(hm)
p_m = np.full(M, 1 / M)
e_bar = float(e_m.mean())
Var = float(np.sum(p_m * (e_m - e_bar) ** 2))
gap = float(h2(e_bar) - np.sum(p_m * h2(e_m)))
bound = 2 / np.log(2) * Var

Lv = np.linspace(0.1, 320, 900)
Rg = np.array([single_rate(L, e_bar) for L in Lv])
Rs = np.array([np.mean([single_rate(L, e) for e in e_m]) for L in Lv])
L_cut = Lv[np.where(Rg > 1e-15)[0][-1]]

Iv = np.linspace(0.02, 1.0, 40)
g_var, g_gain = [], []
for I in Iv:
    e2 = sector_errors(hm, I)
    rg = single_rate(50.0, float(e2.mean()))
    rs = np.mean([single_rate(50.0, e) for e in e2])
    g_var.append(np.var(e2)); g_gain.append(max(rs - rg, 0.0))
g_var, g_gain = np.array(g_var), np.array(g_gain)
Q1_50 = channel_yields(50.0, e_bar)[0]

# ================================================================= helpers
def box(ax, x, y, w, h, fc, ec, txt, fs=5.6, tc="k", lw=0.6, weight="normal",
        r=0.015):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={r}",
                 lw=lw, ec=ec, fc=fc, zorder=3))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs,
            color=tc, zorder=4, linespacing=1.35, fontweight=weight)

def arr(ax, x0, y0, x1, y1, c="k", lw=0.6, ls="-", rad=0.0, mut=4.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=mut, lw=lw, color=c, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}", zorder=2,
                 shrinkA=0, shrinkB=0))

def plabel(ax, s):
    ax.text(-0.085, 1.06, s, transform=ax.transAxes, fontsize=8.0,
            fontweight="bold", va="bottom", ha="left")

def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# ================================================================= figure
fig = plt.figure(figsize=(7.20, 5.90))
gs = GridSpec(2, 12, figure=fig, height_ratios=[0.95, 1.05],
              hspace=0.52, wspace=6.5,
              left=0.055, right=0.985, top=0.945, bottom=0.075)

# ----------------------------------------------------------------- a
axa = fig.add_subplot(gs[0, 0:7]); clean(axa)
axa.text(-0.045, 1.06, "a", transform=axa.transAxes, fontsize=8.0,
         fontweight="bold", va="bottom")
axa.text(0.0, 1.055, "Same data, opposite order of operations",
         transform=axa.transAxes, fontsize=6.8, fontweight="bold",
         va="bottom", ha="left")

# raw-data rain (identical input)
rng = np.random.default_rng(3)
cols8 = plt.cm.viridis(np.linspace(0.1, 0.9, M))
yy0 = 0.30
for i in range(90):
    m = rng.integers(0, M)
    x = 0.020 + 0.085 * rng.random()
    y = yy0 + 0.52 * rng.random()
    axa.plot([x], [y], marker="|", ms=3.2, mew=0.7, color=cols8[m], alpha=0.9)
axa.text(0.062, yy0 - 0.075, "raw round data\n(one experiment)",
         fontsize=5.2, ha="center", va="top", linespacing=1.25)
axa.add_patch(Rectangle((0.012, yy0 - 0.02), 0.100, 0.56, fill=False,
              ec="#999999", lw=0.5))

# split point
arr(axa, 0.115, 0.55, 0.165, 0.755, c=C_GLB, lw=0.8, rad=0.18)
arr(axa, 0.115, 0.35, 0.165, 0.155, c=C_SEC, lw=0.8, rad=-0.18)

# --- global branch (top)
ytop = 0.680
axa.text(0.170, ytop + 0.205, "GLOBAL: average first, entropy second",
         fontsize=5.6, color=C_GLB, fontweight="bold", va="center")
box(axa, 0.170, ytop, 0.170, 0.150, "#EAF2F9", C_GLB,
    "pool\n$\\bar e=\\sum_m p_m e_m=13.8\\%$", fs=5.4)
box(axa, 0.385, ytop, 0.170, 0.150, "#EAF2F9", C_GLB,
    "entropy of average\n$h_2(\\bar e)=0.579$ bits", fs=5.4)
box(axa, 0.600, ytop, 0.170, 0.150, "#EAF2F9", C_GLB,
    "PA cost\n$0.579$ bits/bit", fs=5.4)
box(axa, 0.815, ytop, 0.160, 0.150, C_GLB, C_GLB,
    "key rate\nLOW", fs=5.8, tc="w", weight="bold")
for x0 in (0.340, 0.555, 0.770):
    arr(axa, x0, ytop + 0.075, x0 + 0.045, ytop + 0.075, c=C_GLB, lw=0.7)

# --- sector branch (bottom)
ybot = 0.085
axa.text(0.170, ybot + 0.205, "SECTOR: entropy first, average second",
         fontsize=5.6, color=C_SEC, fontweight="bold", va="center")
box(axa, 0.170, ybot, 0.170, 0.150, "#FBEFE6", C_SEC,
    "keep $\\{e_m\\}_{m=0}^{7}$\n$0.5\\%\\rightarrow 26\\%$", fs=5.4)
box(axa, 0.385, ybot, 0.170, 0.150, "#FBEFE6", C_SEC,
    "average of entropies\n$\\sum_m p_m h_2(e_m)=0.486$", fs=5.2)
box(axa, 0.600, ybot, 0.170, 0.150, "#FBEFE6", C_SEC,
    "PA cost\n$0.486$ bits/bit", fs=5.4)
box(axa, 0.815, ybot, 0.160, 0.150, C_SEC, C_SEC,
    "key rate\nHIGH", fs=5.8, tc="w", weight="bold")
for x0 in (0.340, 0.555, 0.770):
    arr(axa, x0, ybot + 0.075, x0 + 0.045, ybot + 0.075, c=C_SEC, lw=0.7)

# gap bracket between PA boxes
axa.annotate("", xy=(0.685, ybot + 0.155), xytext=(0.685, ytop - 0.005),
             arrowprops=dict(arrowstyle="<->", lw=0.7, color=C_OK))
axa.text(0.700, (ytop + ybot + 0.150) / 2,
         "Jensen gap\n$0.093$ bits\n(strong concavity)",
         fontsize=5.4, color=C_OK, va="center", ha="left", linespacing=1.3)

# ----------------------------------------------------------------- b
axb = fig.add_subplot(gs[0, 7:12])
plabel(axb, "b")
axb.set_title("What pooling hides", loc="left", fontweight="bold", pad=3)
idx = np.arange(M)
barcols = [C_TOX if e > 0.11 else C_OK for e in e_m]
axb.bar(idx, e_m * 100, width=0.68, color=barcols, ec="white", lw=0.4, zorder=3)
axb.axhline(11.0, color="k", lw=0.7, ls=":", zorder=4)
axb.axhline(e_bar * 100, color=C_GLB, lw=0.9, ls="--", zorder=4)
axb.text(M - 0.45, 11.0 + 0.7, "abort threshold 11%", fontsize=5.2,
         ha="right", va="bottom")
axb.text(M - 0.45, e_bar * 100 + 0.7, f"pooled $\\bar e$ = {e_bar*100:.1f}%",
         fontsize=5.2, color=C_GLB, ha="right", va="bottom")
axb.text(1.5, 27.6, "toxic", fontsize=5.4, color=C_TOX, ha="center",
         fontweight="bold")
axb.text(5.5, 8.2, "pure\n(harvested)", fontsize=5.4, color=C_OK,
         ha="center", fontweight="bold", linespacing=1.2)
axb.set_xticks(idx); axb.set_xlabel("public sector $m$")
axb.set_ylabel("QBER (%)"); axb.set_ylim(0, 30)
axb.spines[["top", "right"]].set_visible(False)
axb.text(0.98, 0.62, f"$\\mathrm{{Var}}_m={Var:.4f}$",
         transform=axb.transAxes, fontsize=5.4, va="top", ha="right", color=C_OK)

# ----------------------------------------------------------------- c
axc = fig.add_subplot(gs[1, 0:4])
plabel(axc, "c")
axc.set_title("Concavity converts variance into key", loc="left",
              fontweight="bold", pad=3)
xs = np.linspace(0.001, 0.32, 500)
axc.plot(xs, h2(xs), color="k", lw=0.9, zorder=4)
axc.plot(e_m, h2(e_m), "o", ms=2.6, color=C_SEC, zorder=6,
         label="sectors $(e_m,h_2(e_m))$")
# weighted-mean chord point
axc.plot([e_bar], [h2(e_bar)], "s", ms=3.2, color=C_GLB, zorder=6)
axc.plot([e_bar], [np.sum(p_m * h2(e_m))], "D", ms=3.0, color=C_SEC, zorder=6)
axc.vlines(e_bar, np.sum(p_m * h2(e_m)), h2(e_bar), color=C_OK, lw=1.8,
           zorder=5)
# faint chords from each sector to the mean point
for e in e_m:
    axc.plot([e, e_bar], [h2(e), np.sum(p_m * h2(e_m))], color=C_SEC,
             lw=0.35, alpha=0.35, zorder=2)
axc.annotate(f"gap = {gap:.3f} bits", xy=(e_bar, h2(e_bar) - gap / 2),
             xytext=(0.028, 0.72), fontsize=5.6, color=C_OK,
             arrowprops=dict(arrowstyle="-", lw=0.5, color=C_OK))
axc.text(0.200, 0.13,
         f"theorem bound\n$\\frac{{2}}{{\\ln 2}}\\mathrm{{Var}}_m={bound:.3f}$ bits",
         fontsize=5.4, color=C_GRY, linespacing=1.3)
axc.annotate("$h_2(\\bar e)$", xy=(e_bar, h2(e_bar)), xytext=(e_bar + 0.035, h2(e_bar) + 0.045),
             fontsize=5.4, color=C_GLB, ha="left")
axc.annotate("$\\sum p_m h_2(e_m)$", xy=(e_bar, np.sum(p_m*h2(e_m))),
             xytext=(e_bar + 0.045, np.sum(p_m*h2(e_m)) - 0.16),
             fontsize=5.4, color=C_SEC)
axc.set_xlim(0, 0.32); axc.set_ylim(0, 1.0)
axc.set_xlabel("phase-error rate $e$")
axc.set_ylabel("$h_2(e)$ (bits)")
axc.spines[["top", "right"]].set_visible(False)
axc.legend(loc="upper left", handletextpad=0.3, borderpad=0.1)

# ----------------------------------------------------------------- d (hero)
axd = fig.add_subplot(gs[1, 4:9])
plabel(axd, "d")
axd.set_title("Identical hardware, 14–20$\\times$ more key + 49 km",
              loc="left", fontweight="bold", pad=3)
Rg_p = np.where(Rg > 1e-15, Rg, np.nan)
Rs_p = np.where(Rs > 1e-15, Rs, np.nan)
plob = -np.log2(1 - 10 ** (-ALPHA * Lv / 10))
axd.semilogy(Lv, plob, color="#999999", lw=0.6, ls=(0, (1, 1)), zorder=2)
axd.text(12, 3e-1, "PLOB", fontsize=5.2, color="#777777")
ok = np.isfinite(Rg_p) & np.isfinite(Rs_p)
axd.fill_between(Lv, Rg_p, Rs_p, where=ok, color=C_OK, alpha=0.15, lw=0,
                 zorder=1)
ext = np.isnan(Rg_p) & np.isfinite(Rs_p)
axd.fill_between(Lv, 1e-9, Rs_p, where=ext, color=C_EXT, alpha=0.35, lw=0,
                 zorder=1)
axd.semilogy(Lv, Rg_p, color=C_GLB, lw=1.1, ls="--", zorder=4,
             label="global ($\\bar e=13.8\\%$)")
axd.semilogy(Lv, Rs_p, color=C_SEC, lw=1.2, zorder=5,
             label="sector-resolved")
axd.axvline(L_cut, color=C_GLB, lw=0.5, ls=":", zorder=3)
axd.annotate(f"global aborts\n{L_cut:.0f} km", xy=(L_cut, 3e-8),
             xytext=(L_cut - 105, 1.2e-8), fontsize=5.4, color=C_GLB,
             ha="center", linespacing=1.2,
             arrowprops=dict(arrowstyle="-|>", lw=0.6, color=C_GLB,
                             mutation_scale=5))
axd.annotate("sector key\nsurvives", xy=(300, 4e-8), xytext=(255, 3e-5),
             fontsize=5.4, color="#8a6100", ha="center", linespacing=1.2,
             arrowprops=dict(arrowstyle="-|>", lw=0.6, color=C_EXT,
                             mutation_scale=5))
# ratio callouts
for Lq, dy in [(50, 3.5), (150, 3.5)]:
    i = np.argmin(abs(Lv - Lq))
    axd.annotate("", xy=(Lq, Rs[i]), xytext=(Lq, Rg[i]),
                 arrowprops=dict(arrowstyle="<->", lw=0.6, color=C_OK))
    axd.text(Lq + 6, np.sqrt(Rs[i] * Rg[i]),
             f"$\\times${Rs[i]/Rg[i]:.0f}", fontsize=5.6, color=C_OK,
             va="center")
axd.set_xlim(0, 320); axd.set_ylim(1e-9, 1.5)
axd.set_xlabel("fibre distance $L$ (km)")
axd.set_ylabel("secret key rate (bits/pulse)")
axd.spines[["top", "right"]].set_visible(False)
axd.legend(loc="upper right", handlelength=1.6, borderpad=0.2,
           labelspacing=0.3)

# ----------------------------------------------------------------- e
axe = fig.add_subplot(gs[1, 9:12])
plabel(axe, "e")
axe.set_title("Falsifiable: no variance, no gain", loc="left",
              fontweight="bold", pad=3)
vb = np.linspace(0, g_var.max() * 1.05, 200)
axe.plot(vb * 1e3, qs * Q1_50 * (2 / np.log(2)) * vb * 1e3, color=C_GRY,
         lw=0.8, ls="--", zorder=3,
         label="bound $q\\,Q_1\\frac{2}{\\ln 2}\\mathrm{Var}$")
axe.plot(g_var * 1e3, g_gain * 1e3, "o-", ms=2.4, lw=0.9, color=C_OK,
         zorder=4, label="model gain $\\Delta R$ (50 km)")
axe.plot([0], [0], "s", ms=4.0, color="k", zorder=6)
axe.annotate("homogeneous channel:\n$\\mathrm{Var}=0\\Rightarrow\\Delta R=0$",
             xy=(0, 0), xytext=(0.30 * g_var.max() * 1e3, 0.18 * g_gain.max() * 1e3),
             fontsize=5.4, linespacing=1.3,
             arrowprops=dict(arrowstyle="-|>", lw=0.6, color="k",
                             mutation_scale=5))
axe.set_xlim(0, g_var.max() * 1.05e3)
axe.set_ylim(0, g_gain.max() * 1.15e3)
axe.set_xlabel("$\\mathrm{Var}_m(e_m)\\times 10^{3}$")
axe.set_ylabel("$\\Delta R\\times 10^{3}$ (bits/pulse)")
axe.spines[["top", "right"]].set_visible(False)
axe.legend(loc="upper left", handlelength=1.5, borderpad=0.1,
           labelspacing=0.3)

fig.savefig("fig_sector_vs_global.pdf",
            bbox_inches="tight", pad_inches=0.02)
fig.savefig("fig_sector_vs_global.png",
            dpi=600, bbox_inches="tight", pad_inches=0.02)
print("done")