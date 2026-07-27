"""Figure 1: five Eq. (10)-controlled trajectories on one Bloch sphere."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from bloch_utils import draw_bloch_sphere, set_bloch_axes
from model import SimulationConfig, simulate_five_rounds


def main() -> None:
    config = SimulationConfig()
    rounds = simulate_five_rounds(config)

    figure = plt.figure(figsize=(9.0, 8.0), facecolor="none")
    axis = figure.add_subplot(111, projection="3d", facecolor="none")
    figure.patch.set_facecolor("none")
    axis.set_facecolor("none")

    draw_bloch_sphere(axis)

    start = rounds[0]["bloch"][0]
    axis.scatter(
        start[0],
        start[1],
        start[2],
        s=70,
        depthshade=False,
        label="Shared start",
    )
    axis.text(
        1.06 * start[0],
        1.06 * start[1],
        1.06 * start[2],
        r"$S$",
    )

    for item in rounds:
        path = item["bloch"]
        round_number = int(item["round"])
        lambda_k = float(item["lambda"])

        line = axis.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            linewidth=2.2,
            label=rf"Round {round_number}: $\lambda_k={lambda_k:.3f}$",
        )[0]
        colour = line.get_color()

        end = path[-1]
        axis.scatter(
            end[0],
            end[1],
            end[2],
            s=58,
            color=colour,
            depthshade=False,
        )
        axis.text(
            1.06 * end[0],
            1.06 * end[1],
            1.06 * end[2],
            rf"$R_{round_number}$",
            color=colour,
        )

    # set_bloch_axes(
    #     axis,
    #     r"Five rounds driven by "
    #     r"$\lambda_k=\lambda^{(0)}+\alpha\tilde{x}^{(k)}$",
    # )
    axis.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98))
    figure.tight_layout()

    output = Path(__file__).parent / "figures" / "01_bloch_five_rounds.svg"
    figure.savefig(output, bbox_inches="tight", transparent=True)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
