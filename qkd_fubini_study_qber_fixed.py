import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path


# Publication-style defaults.  The screen figure uses a moderate DPI; exported
# files are still saved at 300 DPI below.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "mathtext.fontset": "cm",
        "figure.autolayout": False,
    }
)

BLUE = "#0f4c81"
GREEN = "#2e7d32"
RED = "#d62728"
GREY = "#777777"


fig = plt.figure(figsize=(13.5, 6.8), dpi=140, constrained_layout=True)
fig.patch.set_facecolor("white")
fig.get_layout_engine().set(w_pad=0.08, h_pad=0.08, wspace=0.10, hspace=0.08)

outer = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.38))
right = outer[1].subgridspec(2, 1, height_ratios=(1.32, 4.2))

ax_a = fig.add_subplot(outer[0])
ax_b_info = fig.add_subplot(right[0])
ax_b = fig.add_subplot(right[1])

# ---------------------------------------------------------------------------
# Panel A: geometric interpretation
# ---------------------------------------------------------------------------
ax_a.set_title(
    r"$\mathbf{A}$   Simplified quantum-state geometric interpretation",
    loc="left",
    pad=10,
    fontweight="bold",
)
ax_a.set_aspect("equal", adjustable="box")
ax_a.set_anchor("N")
ax_a.axis("off")
ax_a.set_xlim(-0.035, 1.0)
ax_a.set_ylim(0.0, 1.0)

# Bloch-sphere cross-section.  The two state vectors start at the same origin
# and terminate on the circumference; no Euclidean chord is used as d_FS.
origin = np.array((0.34, 0.22))
sphere_radius = 0.35
theta_reference_deg = 90.0
theta_residual_deg = 38.0

theta_sphere = np.radians(np.linspace(3.0, 177.0, 350))
ax_a.plot(
    origin[0] + sphere_radius * np.cos(theta_sphere),
    origin[1] + sphere_radius * np.sin(theta_sphere),
    color="black",
    lw=1.45,
    zorder=1,
)

pt_intended = origin + sphere_radius * np.array(
    (
        np.cos(np.radians(theta_reference_deg)),
        np.sin(np.radians(theta_reference_deg)),
    )
)
pt_imperfect = origin + sphere_radius * np.array(
    (
        np.cos(np.radians(theta_residual_deg)),
        np.sin(np.radians(theta_residual_deg)),
    )
)

# Reference and residual Bloch radii: both start exactly at O and meet the arc.
ax_a.plot(
    [origin[0], pt_intended[0]],
    [origin[1], pt_intended[1]],
    color=GREY,
    ls="--",
    lw=1.15,
    zorder=2,
)
ax_a.add_patch(
    patches.FancyArrowPatch(
        origin,
        pt_imperfect,
        arrowstyle="-|>",
        mutation_scale=10,
        color="black",
        lw=1.45,
        shrinkA=0,
        shrinkB=1,
        zorder=3,
    )
)

# The blue curve is the geodesic arc between the two state endpoints.
theta_geodesic = np.radians(
    np.linspace(theta_residual_deg, theta_reference_deg, 150)
)
ax_a.plot(
    origin[0] + sphere_radius * np.cos(theta_geodesic),
    origin[1] + sphere_radius * np.sin(theta_geodesic),
    color=BLUE,
    lw=3.0,
    solid_capstyle="round",
    zorder=4,
)

ax_a.plot(*origin, "ko", markersize=5.5, zorder=6)
ax_a.plot(*pt_intended, "ko", markersize=6.5, zorder=6)
ax_a.plot(*pt_imperfect, "ko", markersize=6.5, zorder=6)

# Delta_m is the central angle between the two Bloch radii.
angle_radius = 0.125
angle_values = np.radians(
    np.linspace(theta_residual_deg, theta_reference_deg, 50)
)
angle_vertices = np.column_stack(
    (
        origin[0] + angle_radius * np.cos(angle_values),
        origin[1] + angle_radius * np.sin(angle_values),
    )
)
angle_path = Path(
    angle_vertices,
    [Path.MOVETO] + [Path.LINETO] * (len(angle_vertices) - 1),
)
ax_a.add_patch(
    patches.FancyArrowPatch(
        path=angle_path,
        arrowstyle="<->",
        mutation_scale=8,
        color="black",
        lw=1.15,
        zorder=5,
    )
)

theta_mid_deg = 0.5 * (theta_reference_deg + theta_residual_deg)
angle_label_position = origin + 0.17 * np.array(
    (
        np.cos(np.radians(theta_mid_deg)),
        np.sin(np.radians(theta_mid_deg)),
    )
)
ax_a.text(
    angle_label_position[0],
    angle_label_position[1],
    r"$|\Delta_m|$",
    fontsize=11.5,
    ha="center",
    va="center",
)

# Labels: angular separation and Fubini--Study distance are associated with
# the geodesic arc, never with the straight radial vector.
ax_a.text(
    0.49,
    0.67,
    "Residual Bloch-sphere\nangular mismatch: " + r"$|\Delta_m|$",
    fontsize=8.7,
    color=BLUE,
    ha="center",
    va="bottom",
    fontweight="bold",
)
ax_a.text(
    0.76,
    0.32,
    "Fubini--Study distance:\n" + r"$d_{\mathrm{FS}}^{(m)}=|\Delta_m|/2$",
    fontsize=9.0,
    color=BLUE,
    ha="center",
    va="center",
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="#9bb8cf",
        lw=0.8,
    ),
)
ax_a.text(
    0.17,
    0.63,
    "Intended BB84\nreference state",
    ha="center",
    va="bottom",
    fontsize=9.0,
)
ax_a.text(
    0.69,
    0.51,
    "State after imperfect\ncompensation",
    ha="center",
    va="bottom",
    fontsize=9.0,
)
ax_a.text(
    origin[0],
    origin[1] - 0.045,
    r"Bloch-sphere origin $O$",
    ha="center",
    va="top",
    fontsize=8.2,
    color="#444444",
)
ax_a.text(
    0.72,
    0.17,
    "Residual sector mismatch\n"
    r"$\longrightarrow$ quantum-state distance" "\n"
    r"$\longrightarrow$ sector-conditioned QBER",
    ha="center",
    va="top",
    fontsize=8.3,
    color="#333333",
    style="italic",
)

# ---------------------------------------------------------------------------
# Panel B: equation/parameters in a dedicated information band
# ---------------------------------------------------------------------------
ax_b_info.set_title(
    r"$\mathbf{B}$   Sector-conditioned receiver response",
    loc="left",
    pad=10,
    fontweight="bold",
)
ax_b_info.axis("off")

info_text = (
    r"$e^{(m)}=e_d+V\sin^2\!\left(d_{\mathrm{FS}}^{(m)}\right)$"
    "\n"
    r"$V=0.97,\qquad e_d=(1-V)/2=0.015\;(1.5\%)$" "\n"
    "Under the declared interferometric receiver model"
)
ax_b_info.text(
    0.5,
    0.47,
    info_text,
    transform=ax_b_info.transAxes,
    ha="center",
    va="center",
    fontsize=9.5,
    linespacing=1.35,
    bbox=dict(
        boxstyle="round,pad=0.55",
        facecolor="#fbfbfb",
        edgecolor="#c7c7c7",
        lw=0.9,
    ),
)


# Model and domain
V = 0.97
e_d = (1.0 - V) / 2.0
e_abort = 0.11
d_abort = np.arcsin(np.sqrt((e_abort - e_d) / V))

d_FS = np.linspace(0, np.pi / 2, 500)
e_m = e_d + V * np.sin(d_FS) ** 2

ax_b.set_xlabel(r"Fubini--Study distance $d_{\mathrm{FS}}^{(m)}$ (rad)")
ax_b.set_ylabel(r"Sector-conditioned QBER $e^{(m)}$")

# The illustrative acceptance area is bounded in both distance and QBER.
acceptance_box = patches.Rectangle(
    (0, 0),
    d_abort,
    e_abort,
    facecolor="#d9d9d9",
    edgecolor="#555555",
    lw=0.9,
    alpha=0.62,
    zorder=1,
)
ax_b.add_patch(acceptance_box)
ax_b.axvline(
    d_abort,
    color="#555555",
    ls="-.",
    lw=1.0,
    zorder=2,
)
ax_b.text(
    d_abort - 0.014,
    0.50,
    rf"$d_{{\mathrm{{FS}}}}^{{\max}}\approx {d_abort:.3f}\ \mathrm{{rad}}$",
    rotation=90,
    rotation_mode="anchor",
    ha="center",
    va="top",
    fontsize=8.2,
    color="#333333",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.5),
    zorder=5,
)
ax_b.plot(d_FS, e_m, color=BLUE, lw=2.8, zorder=4)

# Illustrative guide: dashed styling remains distinguishable without color.
ax_b.axhline(e_abort, color=RED, ls=(0, (6, 3)), lw=1.55, zorder=3)
ax_b.annotate(
    r"Illustrative abort guide:  $e_{\mathrm{abort}}=11\%$",
    xy=(np.pi / 2, e_abort),
    xytext=(-8, 8),
    textcoords="offset points",
    ha="right",
    va="bottom",
    color=RED,
    fontsize=9.2,
    fontweight="bold",
)
ax_b.text(
    0.035,
    0.20,
    "Illustrative acceptance region\n"
    + rf"$0\leq d_{{\mathrm{{FS}}}}^{{(m)}}\leq {d_abort:.3f}\ \mathrm{{rad}}$",
    transform=ax_b.transAxes,
    color=GREEN,
    fontsize=8.4,
    ha="left",
    va="center",
    fontweight="bold",
)

# Reference points: every callout keeps distance and QBER as distinct quantities.
points_data = [
    (
        0.0,
        e_d,
        r"$d_{\mathrm{FS}}^{(m)}=0\ \mathrm{rad}$" "\n"
        r"$e^{(m)}=0.015=1.5\%$",
        (10, 8),
        "left",
        "bottom",
    ),
    (
        np.pi / 8,
        e_d + V * np.sin(np.pi / 8) ** 2,
        r"$d_{\mathrm{FS}}^{(m)}=\pi/8\ \mathrm{rad}$" "\n"
        r"$e^{(m)}\approx0.157=15.7\%$",
        (10, 12),
        "left",
        "bottom",
    ),
    (
        np.pi / 4,
        0.5,
        r"$d_{\mathrm{FS}}^{(m)}=\pi/4\ \mathrm{rad}$" "\n"
        r"$e^{(m)}=0.500=50.0\%$",
        (10, -8),
        "left",
        "top",
    ),
    (
        3 * np.pi / 8,
        e_d + V * np.sin(3 * np.pi / 8) ** 2,
        r"$d_{\mathrm{FS}}^{(m)}=3\pi/8\ \mathrm{rad}$" "\n"
        r"$e^{(m)}\approx0.843=84.3\%$",
        (-10, -8),
        "right",
        "top",
    ),
    (
        np.pi / 2,
        e_d + V,
        r"$d_{\mathrm{FS}}^{(m)}=\pi/2\ \mathrm{rad}$" "\n"
        r"$e^{(m)}=0.985=98.5\%$",
        (-10, -8),
        "right",
        "top",
    ),
]

for x, y, label, offset, ha, va in points_data:
    ax_b.plot(x, y, "ko", markersize=6, zorder=6)
    if x > 0:
        ax_b.plot([x, x], [0, y], color="#aaaaaa", ls="--", lw=0.8, zorder=2)
        ax_b.plot([0, x], [y, y], color="#aaaaaa", ls="--", lw=0.8, zorder=2)
    ax_b.annotate(
        label,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=7.7,
        ha=ha,
        va=va,
        linespacing=1.15,
        zorder=7,
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="none",
            alpha=0.82,
        ),
    )

ax_b.set_xticks(
    [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2],
    [r"$0$", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"],
)
ax_b.set_yticks(np.linspace(0, 1, 6))
ax_b.set_xlim(-0.02, np.pi / 2 + 0.02)
ax_b.set_ylim(-0.02, 1.02)
ax_b.tick_params(direction="out", length=4, width=0.9, pad=5)

for spine in ax_b.spines.values():
    spine.set_linewidth(0.9)
    spine.set_color("black")


fig.savefig("qkd_fubini_study_qber_fixed.png", dpi=300, facecolor="white")
fig.savefig("qkd_fubini_study_qber_fixed.pdf", facecolor="white")
if "agg" not in plt.get_backend().lower():
    plt.show()
