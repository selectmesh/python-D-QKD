"""Reusable three-dimensional Bloch-sphere drawing utilities."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes


def draw_bloch_sphere(ax: Axes, mesh_points: int = 36) -> None:
    """Draw a transparent unit-sphere wireframe and Cartesian axes."""

    u = np.linspace(0.0, 2.0 * np.pi, mesh_points)
    v = np.linspace(0.0, np.pi, mesh_points // 2)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_wireframe(
        x,
        y,
        z,
        rstride=3,
        cstride=2,
        linewidth=0.35,
        alpha=0.24,
    )

    # Main axes.
    ax.plot([-1.2, 1.2], [0.0, 0.0], [0.0, 0.0], linewidth=0.8, alpha=0.65)
    ax.plot([0.0, 0.0], [-1.2, 1.2], [0.0, 0.0], linewidth=0.8, alpha=0.65)
    ax.plot([0.0, 0.0], [0.0, 0.0], [-1.2, 1.2], linewidth=0.8, alpha=0.65)

    ax.text(1.27, 0.0, 0.0, r"$x$")
    ax.text(0.0, 1.27, 0.0, r"$y$")
    ax.text(0.0, 0.0, 1.27, r"$z$")
    ax.text(0.0, 0.0, 1.08, r"$|0\rangle$")
    ax.text(0.0, 0.0, -1.13, r"$|1\rangle$")


def set_bloch_axes(ax: Axes, title: str) -> None:
    """Set equal scale, camera, labels, and publication-style limits."""

    ax.set_title(title, pad=18)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_zlim(-1.25, 1.25)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=23.0, azim=38.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()


def spherical_geodesic(
    start: np.ndarray,
    end: np.ndarray,
    samples: int = 220,
) -> np.ndarray:
    """Return the shortest great-circle arc from `start` to `end`.

    Both inputs must be nonzero three-dimensional vectors. The output
    includes both endpoints.
    """

    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    omega = float(np.arccos(dot))

    if np.isclose(omega, 0.0):
        return np.repeat(a[None, :], samples, axis=0)

    if np.isclose(np.sin(omega), 0.0):
        raise ValueError(
            "Start and end are antipodal; the shortest geodesic is not unique."
        )

    s = np.linspace(0.0, 1.0, samples)
    weights_a = np.sin((1.0 - s) * omega) / np.sin(omega)
    weights_b = np.sin(s * omega) / np.sin(omega)
    curve = weights_a[:, None] * a + weights_b[:, None] * b
    curve /= np.linalg.norm(curve, axis=1, keepdims=True)
    return curve


def draw_state_vector(
    ax: Axes,
    vector: np.ndarray,
    label: str,
    *,
    linewidth: float = 1.5,
) -> None:
    """Draw a radial Bloch vector and label its endpoint."""

    v = np.asarray(vector, dtype=float)
    line = ax.plot(
        [0.0, v[0]],
        [0.0, v[1]],
        [0.0, v[2]],
        linewidth=linewidth,
    )[0]
    colour = line.get_color()
    ax.scatter(v[0], v[1], v[2], s=42, color=colour, depthshade=False)
    ax.text(1.06 * v[0], 1.06 * v[1], 1.06 * v[2], label, color=colour)
