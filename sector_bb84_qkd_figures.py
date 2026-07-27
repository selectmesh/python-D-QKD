"""Five round-conditioned BB84 constellations on Bloch spheres.

Each panel corresponds to one row of the five-round table:

    k, x_tilde^(k), lambda_k, gamma_k, sector S_m

For visualisation, the round-dependent BB84 frame is defined by

    U_k = R_z(gamma_k),

so that

    B_k = {
        U_k |0>,
        U_k |1>,
        U_k |+>,
        U_k |->
    }.

Under R_z(gamma_k), the Z-basis states remain at the poles, while the
X-basis states rotate around the equator. Therefore, every panel still
contains two orthonormal and mutually unbiased BB84 bases.

If the manuscript defines a different sector unitary U_m, replace the
function `round_unitary`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)


# ============================================================
# Exact values from the five-round table
# ============================================================

ROUNDS = np.array([1, 2, 3, 4, 5], dtype=int)

X_TILDE = np.array(
    [-0.80, -0.20, 0.30, 0.85, -0.45],
    dtype=float,
)

LAMBDA_K = np.array(
    [0.600, 0.900, 1.150, 1.425, 0.775],
    dtype=float,
)

GAMMA_K = np.array(
    [6.264969, 0.374222, 2.061549, 3.957887, 0.001391],
    dtype=float,
)

SECTORS = ["S4", "S0", "S1", "S3", "S0"]

SECTOR_COLORS = {
    "S0": "#4C78A8",  # blue
    "S1": "#F58518",  # orange
    "S2": "#54A24B",  # green
    "S3": "#B279A2",  # purple
    "S4": "#E45756",  # red
}


# ============================================================
# Standard BB84 states
# ============================================================

KET_0 = np.array([1.0, 0.0], dtype=complex)
KET_1 = np.array([0.0, 1.0], dtype=complex)

KET_PLUS = (KET_0 + KET_1) / np.sqrt(2.0)
KET_MINUS = (KET_0 - KET_1) / np.sqrt(2.0)

STANDARD_BB84 = {
    "0": KET_0,
    "1": KET_1,
    "+": KET_PLUS,
    "-": KET_MINUS,
}


# ============================================================
# Quantum operations
# ============================================================

def rz(angle: float) -> np.ndarray:
    """Single-qubit rotation around the Bloch z-axis."""

    return np.array(
        [
            [np.exp(-0.5j * angle), 0.0],
            [0.0, np.exp(0.5j * angle)],
        ],
        dtype=complex,
    )


def round_unitary(gamma_k: float) -> np.ndarray:
    """Return the illustrative round-dependent BB84 frame unitary.

    Current choice:
        U_k = R_z(gamma_k)

    Replace this function if the protocol defines U_m differently.
    """

    return rz(gamma_k)


def bloch_vector(state: np.ndarray) -> np.ndarray:
    """Convert a pure one-qubit state into a unit Bloch vector."""

    psi = np.asarray(state, dtype=complex)
    norm = np.linalg.norm(psi)

    if norm == 0:
        raise ValueError("A quantum state cannot be the zero vector.")

    psi = psi / norm
    alpha, beta = psi

    x = 2.0 * np.real(np.conjugate(alpha) * beta)
    y = 2.0 * np.imag(np.conjugate(alpha) * beta)
    z = np.abs(alpha) ** 2 - np.abs(beta) ** 2

    vector = np.array([x, y, z], dtype=float)
    return vector / np.linalg.norm(vector)


# ============================================================
# Bloch-sphere drawing
# ============================================================

def draw_bloch_sphere(ax) -> None:
    """Draw the unit Bloch sphere, equator, meridians, and axes."""

    azimuth = np.linspace(0.0, 2.0 * np.pi, 72)
    polar = np.linspace(0.0, np.pi, 36)

    x = np.outer(np.cos(azimuth), np.sin(polar))
    y = np.outer(np.sin(azimuth), np.sin(polar))
    z = np.outer(np.ones_like(azimuth), np.cos(polar))

    ax.plot_wireframe(
        x,
        y,
        z,
        rstride=6,
        cstride=4,
        linewidth=0.35,
        alpha=0.20,
    )

    # Equator.
    ax.plot(
        np.cos(azimuth),
        np.sin(azimuth),
        np.zeros_like(azimuth),
        linewidth=0.8,
        alpha=0.45,
    )

    # Two reference meridians.
    ax.plot(
        np.sin(polar),
        np.zeros_like(polar),
        np.cos(polar),
        linewidth=0.7,
        alpha=0.30,
    )

    ax.plot(
        np.zeros_like(polar),
        np.sin(polar),
        np.cos(polar),
        linewidth=0.7,
        alpha=0.30,
    )

    # Cartesian axes.
    ax.plot([-1.18, 1.18], [0, 0], [0, 0], linewidth=0.75, alpha=0.55)
    ax.plot([0, 0], [-1.18, 1.18], [0, 0], linewidth=0.75, alpha=0.55)
    ax.plot([0, 0], [0, 0], [-1.18, 1.18], linewidth=0.75, alpha=0.55)

    ax.text(1.24, 0, 0, r"$x$", fontsize=10, color="#444444")
    ax.text(0, 1.24, 0, r"$y$", fontsize=10, color="#444444")
    ax.text(0, 0, 1.24, r"$z$", fontsize=10, color="#444444")


def configure_axis(ax) -> None:
    """Set equal scale and a consistent camera view."""

    ax.set_xlim(-1.24, 1.24)
    ax.set_ylim(-1.24, 1.24)
    ax.set_zlim(-1.24, 1.24)

    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=23.0, azim=38.0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()


def draw_state(
    ax,
    vector: np.ndarray,
    label: str,
    line_style: str,
    color: str,
) -> None:
    """Draw one state vector from the origin and label its endpoint."""

    v = np.asarray(vector, dtype=float)

    ax.plot(
        [0.0, v[0]],
        [0.0, v[1]],
        [0.0, v[2]],
        linestyle=line_style,
        linewidth=2.2,
        color=color,
    )

    ax.scatter(
        [v[0]],
        [v[1]],
        [v[2]],
        s=42,
        color=color,
        depthshade=False,
    )

    # Slightly different label radius for polar and equatorial states.
    radius = 1.10 if abs(v[2]) < 0.8 else 1.06

    ax.text(
        radius * v[0],
        radius * v[1],
        radius * v[2],
        label,
        color=color,
        fontsize=10,
        ha="center",
        va="center",
    )


def draw_basis_diameter(
    ax,
    positive_vector: np.ndarray,
    negative_vector: np.ndarray,
    line_style: str,
    color: str,
) -> None:
    """Draw the full diameter joining the two states of one basis."""

    ax.plot(
        [negative_vector[0], positive_vector[0]],
        [negative_vector[1], positive_vector[1]],
        [negative_vector[2], positive_vector[2]],
        linestyle=line_style,
        linewidth=1.0,
        color=color,
        alpha=0.70,
    )


def transformed_bb84_vectors(gamma_k: float) -> dict[str, np.ndarray]:
    """Return the four BB84 Bloch vectors after U_k = R_z(gamma_k)."""

    unitary = round_unitary(gamma_k)

    return {
        "0": bloch_vector(unitary @ KET_0),
        "1": bloch_vector(unitary @ KET_1),
        "+": bloch_vector(unitary @ KET_PLUS),
        "-": bloch_vector(unitary @ KET_MINUS),
    }


def draw_round_panel(
    ax,
    round_number: int,
    x_tilde: float,
    lambda_k: float,
    gamma_k: float,
    sector: str,
) -> None:
    """Draw one complete round-conditioned BB84 constellation."""

    draw_bloch_sphere(ax)
    configure_axis(ax)

    vectors = transformed_bb84_vectors(gamma_k)
    color = SECTOR_COLORS.get(sector, "#2c3e50")

    # The Z and X bases are shown with different line styles.
    draw_basis_diameter(ax, vectors["0"], vectors["1"], "-", color)
    draw_basis_diameter(ax, vectors["+"], vectors["-"], "--", color)

    draw_state(
        ax,
        vectors["0"],
        rf"$|0_{{{round_number}}}\rangle$",
        "-",
        color,
    )
    draw_state(
        ax,
        vectors["1"],
        rf"$|1_{{{round_number}}}\rangle$",
        "-",
        color,
    )
    draw_state(
        ax,
        vectors["+"],
        rf"$|+_{{{round_number}}}\rangle$",
        "--",
        color,
    )
    draw_state(
        ax,
        vectors["-"],
        rf"$-_{{{round_number}}}\rangle$",
        "--",
        color,
    )

    ax.set_title(
        rf"Round {round_number} — ${sector}$"
        "\n"
        rf"$\tilde{{x}}^{{({round_number})}}={x_tilde:.2f}$, "
        rf"$\lambda_{round_number}={lambda_k:.3f}$"
        "\n"
        rf"$\gamma_{round_number}={gamma_k:.6f}\ \mathrm{{rad}}$",
        fontsize=14,
        pad=12,
        color=color,
    )

    ax.text2D(
        0.50,
        0.02,
        rf"$\mathcal{{B}}_{round_number}"
        rf"=R_z(\gamma_{round_number})"
        r"\{|0\rangle,|1\rangle,|+\rangle,|-\rangle\}$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        color="#000000",
    )


# ============================================================
# Main figure
# ============================================================

def main() -> None:
    figure = plt.figure(figsize=(16, 10), facecolor="none")

    # Five equally sized panels:
    # top row: rounds 1, 2, 3
    # bottom row: rounds 4, 5 centred
    grid = figure.add_gridspec(
        nrows=2,
        ncols=6,
        hspace=0.12,
        wspace=0.10,
    )

    panel_positions = [
        grid[0, 0:2],
        grid[0, 2:4],
        grid[0, 4:6],
        grid[1, 1:3],
        grid[1, 3:5],
    ]

    round_entries = list(zip(ROUNDS, X_TILDE, LAMBDA_K, GAMMA_K, SECTORS))
    round_entries.sort(key=lambda item: item[4])

    for panel_position, (round_number, x_tilde, lambda_k, gamma_k, sector) in zip(
        panel_positions,
        round_entries,
    ):
        axis = figure.add_subplot(panel_position, projection="3d")

        draw_round_panel(
            axis,
            round_number=int(round_number),
            x_tilde=float(x_tilde),
            lambda_k=float(lambda_k),
            gamma_k=float(gamma_k),
            sector=sector,
        )

    # figure.suptitle(
    #     "Five round-conditioned BB84 constellations"
    #     "\n"
    #     r"$\mathcal{B}_k"
    #     r"=\{U_k|0\rangle,U_k|1\rangle,U_k|+\rangle,U_k|-\rangle\},"
    #     r"\qquad U_k=R_z(\gamma_k)$",
    #     fontsize=18,
    #     y=0.985,
    # )

    # figure.text(
    #     0.50,
    #     0.025,
    #     r"Every round contains a complete BB84 alphabet: "
    #     r"$\langle0_k|1_k\rangle=0$, "
    #     r"$\langle+_k|-_k\rangle=0$, and "
    #     r"$|\langle0_k|+_k\rangle|^2=1/2$. "
    #     r"The sector label is public and identifies the round-conditioned BB84 frame.",
    #     ha="center",
    #     va="bottom",
    #     fontsize=11,
    # )

    # figure.suptitle(
    #     "Five-round BB84 sector-conditioned Bloch-sphere constellations",
    #     fontsize=18,
    #     fontweight="bold",
    #     color="#1f3b5b",
    #     y=0.97,
    # )

    figure.subplots_adjust(left=0.025, right=0.985, top=0.90, bottom=0.055)

    output = Path(__file__).with_suffix(".svg")
    figure.savefig(output, bbox_inches="tight", transparent=True)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()