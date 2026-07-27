"""Figure 4: three-dimensional evolution of round, lambda_k, and gamma_k."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from model import SimulationConfig, simulate_five_rounds
from psb_utils import open_path_geometric_phase, phase_to_sector


def main() -> None:
    config = SimulationConfig()
    rounds = simulate_five_rounds(config)

    round_numbers = np.array([int(item["round"]) for item in rounds])
    lambdas = np.array([float(item["lambda"]) for item in rounds])
    phases = np.array(
        [
            open_path_geometric_phase(
                item["times"],
                item["states"],
                float(item["lambda"]),
                config,
            )
            for item in rounds
        ]
    )
    sectors = np.array([phase_to_sector(value, sectors=5) for value in phases])

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    figure = plt.figure(figsize=(9.0, 8.0), facecolor="none")
    axis = figure.add_subplot(111, projection="3d", facecolor="none")
    figure.patch.set_facecolor("none")
    axis.set_facecolor("none")
    figure.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.28)

    sector_colors = {
        0: "#4C78A8",
        1: "#F58518",
        2: "#54A24B",
        3: "#B279A2",
        4: "#E45756",
    }

    center = np.array([np.mean(round_numbers), np.mean(lambdas), np.mean(phases)])
    radius = max(np.ptp(round_numbers), np.ptp(lambdas), np.ptp(phases)) * 0.7

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    sphere_x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    sphere_y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    sphere_z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

    axis.plot_surface(
        sphere_x,
        sphere_y,
        sphere_z,
        color="#ffffff",
        alpha=0.24,
        edgecolor="#d9d9d9",
        linewidth=0.6,
        zorder=0,
    )

    axis.plot(
        round_numbers,
        lambdas,
        phases,
        color="#2c3e50",
        linewidth=2.3,
        alpha=0.95,
        zorder=2,
    )

    for k, lam, gamma, sector in zip(round_numbers, lambdas, phases, sectors):
        sector_idx = int(sector)
        colour = sector_colors.get(sector_idx, "#2c3e50")
        axis.scatter(
            k,
            lam,
            gamma,
            s=80,
            color=colour,
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
        )
        axis.text(
            k,
            lam,
            gamma + 0.05,
            rf"$S_{sector_idx}$",
            color=colour,
            fontsize=9,
            fontweight="bold",
            zorder=5,
        )

    axis.set_xlabel(r"Round $k$", labelpad=10)
    axis.set_ylabel(r"Control $\lambda_k$", labelpad=10)
    axis.set_zlabel(r"PSB phase $\gamma_k$ (rad)", labelpad=10)
    axis.set_title(
        r"Five-round control-to-phase evolution "
        r"$\lambda_k=\lambda^{(0)}+\alpha\tilde{x}^{(k)}$",
        pad=18,
    )
    axis.set_xticks(round_numbers)
    axis.set_xlim(center[0] - radius * 1.08, center[0] + radius * 1.08)
    axis.set_ylim(center[1] - radius * 1.08, center[1] + radius * 1.08)
    axis.set_zlim(center[2] - radius * 1.08, center[2] + radius * 1.08)
    axis.view_init(elev=24, azim=38)
    axis.set_box_aspect((1.35, 1.0, 0.9))
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=colour, markeredgecolor="white", markersize=9)
        for colour in sector_colors.values()
    ]
    axis.legend(
        legend_handles,
        [r"$S_0$", r"$S_1$", r"$S_2$", r"$S_3$", r"$S_4$"],
        title="Sector",
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#d0d0d0",
        fontsize=9,
        title_fontsize=10,
    )

    table_ax = figure.add_axes([0.07, 0.05, 0.86, 0.20])
    table_ax.axis("off")
    table_data = [
        [
            "round",
            r"$(\tilde{x}^{(k)})$",
            r"$(\lambda_k)$",
            r"$(\gamma_k)$ rad",
            "sector",
        ],
        ["1", r"$(-0.80)$", r"$(0.600)$", r"$(6.264969)$", r"$(S_4)$"],
        ["2", r"$(-0.20)$", r"$(0.900)$", r"$(0.374222)$", r"$(S_0)$"],
        ["3", r"$(0.30)$", r"$(1.150)$", r"$(2.061549)$", r"$(S_1)$"],
        ["4", r"$(0.85)$", r"$(1.425)$", r"$(3.957887)$", r"$(S_3)$"],
        ["5", r"$(-0.45)$", r"$(0.775)$", r"$(0.001391)$", r"$(S_0)$"],
    ]
    table = table_ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
        colWidths=[0.10, 0.17, 0.15, 0.18, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.15)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E9EFF7")
            cell.set_text_props(weight="bold", color="#23395B")
        elif row % 2 == 0:
            cell.set_facecolor("#F8FAFC")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#D6DCE5")

    output = (
        Path(__file__).parent
        / "figures"
        / "04_lambda_gamma_evolution.svg"
    )
    figure.savefig(output, bbox_inches="tight", transparent=True)
    print(f"Saved: {output}")

    print("\nRound | x_tilde | lambda | gamma(rad) | sector")
    print("-" * 49)
    for item, gamma, sector in zip(rounds, phases, sectors):
        print(
            f"{int(item['round']):5d} | "
            f"{float(item['normalized_value']):7.2f} | "
            f"{float(item['lambda']):6.3f} | "
            f"{gamma:10.6f} | "
            f"S{sector}"
        )


if __name__ == "__main__":
    main()
