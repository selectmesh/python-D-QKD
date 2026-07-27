"""Figure 3: open path, shortest geodesic closure, and PSB phase."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from bloch_utils import draw_bloch_sphere, set_bloch_axes
from model import SimulationConfig, simulate_five_rounds
from psb_utils import (
    close_bloch_path_with_geodesic,
    closed_path_solid_angle,
    open_path_geometric_phase,
    phase_from_solid_angle,
)


SELECTED_ROUND = 3


def add_area_ribbons(
    axis,
    path: np.ndarray,
    geodesic: np.ndarray,
    segments: int = 110,
) -> None:
    """Create an illustrative ruled surface between path and geodesic.

    This shading helps the reader see the enclosed spherical region.
    It is a visual aid; the numerical solid angle is computed separately.
    """

    path_indices = np.linspace(0, len(path) - 1, segments).astype(int)
    geo_indices = np.linspace(len(geodesic) - 1, 0, segments).astype(int)

    path_sample = path[path_indices]
    geodesic_sample = geodesic[geo_indices]

    polygons = []
    for index in range(segments - 1):
        polygons.append(
            [
                path_sample[index],
                path_sample[index + 1],
                geodesic_sample[index + 1],
                geodesic_sample[index],
            ]
        )

    surface = Poly3DCollection(
        polygons,
        alpha=0.12,
        linewidths=0.2,
    )
    axis.add_collection3d(surface)


def main() -> None:
    config = SimulationConfig()
    item = simulate_five_rounds(config)[SELECTED_ROUND - 1]

    path = item["bloch"]
    states = item["states"]
    times = item["times"]
    lambda_k = float(item["lambda"])

    closed_path, geodesic = close_bloch_path_with_geodesic(path)
    omega = closed_path_solid_angle(closed_path)
    gamma_area = phase_from_solid_angle(omega)
    gamma_state = open_path_geometric_phase(times, states, lambda_k, config)

    figure = plt.figure(figsize=(9.0, 8.0), facecolor="none")
    axis = figure.add_subplot(111, projection="3d", facecolor="none")
    figure.patch.set_facecolor("none")
    axis.set_facecolor("none")

    draw_bloch_sphere(axis)

    path_line = axis.plot(
        path[:, 0],
        path[:, 1],
        path[:, 2],
        linewidth=2.5,
        label="Hamiltonian trajectory",
    )[0]
    path_colour = path_line.get_color()

    geo_line = axis.plot(
        geodesic[:, 0],
        geodesic[:, 1],
        geodesic[:, 2],
        linestyle="--",
        linewidth=2.2,
        label="Shortest geodesic closure",
    )[0]

    add_area_ribbons(axis, path, geodesic)

    start = path[0]
    end = path[-1]
    axis.scatter(
        start[0],
        start[1],
        start[2],
        s=70,
        color=path_colour,
        depthshade=False,
    )
    axis.scatter(
        end[0],
        end[1],
        end[2],
        s=70,
        color=path_colour,
        depthshade=False,
    )
    axis.text(
        1.07 * start[0],
        1.07 * start[1],
        1.07 * start[2],
        r"$A$",
        color=path_colour,
    )
    axis.text(
        1.07 * end[0],
        1.07 * end[1],
        1.07 * end[2],
        r"$B$",
        color=path_colour,
    )

    # set_bloch_axes(
    #     axis,
    #     rf"Geodesic closure and spherical area — round {SELECTED_ROUND}",
    # )

    legend_title = (
        rf"$\Omega={omega:.4f}\ {{\rm sr}}$" "\n"
        rf"$-\Omega/2={gamma_area:.4f}\ {{\rm rad}}$" "\n"
        rf"$\gamma_{{\rm SB}}={gamma_state:.4f}\ {{\rm rad}}$"
    )
    axis.legend(
        title=legend_title,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.98),
    )
    figure.tight_layout()

    output = (
        Path(__file__).parent
        / "figures"
        / "03_psb_geodesic_closure.svg"
    )
    figure.savefig(output, bbox_inches="tight", transparent=True)
    print(f"Saved: {output}")
    print(f"Signed solid angle Omega: {omega:.10f} sr")
    print(f"Phase from -Omega/2:       {gamma_area:.10f} rad")
    print(f"State-based SB phase:      {gamma_state:.10f} rad")


if __name__ == "__main__":
    main()
