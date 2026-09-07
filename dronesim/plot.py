"""Plots. Matplotlib only, and only these two functions."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .drone import State
from .sensor import beam_angles
from .world import World


def plot_world(
    world: World,
    state: Optional[State] = None,
    scan: Optional[np.ndarray] = None,
    trajectory: Optional[np.ndarray] = None,
    ax=None,
):
    """Draw the field, the obstacles and optionally the drone, its scan and its path.

    Args:
        world: the world.
        state: drone state, drawn as a dot with an arrow in the heading direction.
        scan: range scan of that state, drawn as rays (needs ``state``).
        trajectory: path of shape (n, 2) as returned by :func:`simulate`.
        ax: matplotlib axes to draw on. A new figure is created if omitted.

    Returns:
        The matplotlib axes.

    Example:
        >>> import matplotlib; matplotlib.use("Agg")
        >>> from dronesim import make_world, plot_world
        >>> ax = plot_world(make_world(seed=0))
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.add_patch(Rectangle((0, 0), world.width, world.height, fill=False, lw=1.5, color="black"))
    for (x, y, w, h) in world.obstacles:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="0.6", edgecolor="0.3"))

    if state is not None:
        if scan is not None:
            scan = np.asarray(scan, dtype=float).ravel()
            angles = beam_angles(state.heading, scan.size)
            ends_x = state.x + scan * np.cos(angles)
            ends_y = state.y + scan * np.sin(angles)
            for ex, ey in zip(ends_x, ends_y):
                ax.plot([state.x, ex], [state.y, ey], color="tab:orange", lw=0.6, alpha=0.7)
        ax.plot([state.x], [state.y], "o", color="tab:blue", ms=8, zorder=3)
        if state.speed > 0:
            ax.arrow(
                state.x, state.y,
                state.vx / state.speed, state.vy / state.speed,
                head_width=0.3, color="tab:blue", zorder=3, length_includes_head=True,
            )

    if trajectory is not None:
        trajectory = np.asarray(trajectory, dtype=float)
        ax.plot(trajectory[:, 0], trajectory[:, 1], "-", color="tab:red", lw=2, zorder=4)

    ax.set_xlim(-1, world.width + 1)
    ax.set_ylim(-1, world.height + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return ax


def plot_scan(scan: np.ndarray, ax=None):
    """Polar plot of a range scan, 0 degrees is straight ahead.

    Args:
        scan: array of shape (n_beams,) from :func:`range_scan`.
        ax: polar matplotlib axes. A new figure is created if omitted.

    Returns:
        The matplotlib axes.

    Example:
        >>> import matplotlib; matplotlib.use("Agg")
        >>> from dronesim import World, State, range_scan, plot_scan
        >>> w = World(width=10.0, height=10.0)
        >>> s = State(x=5.0, y=5.0, vx=1.0, vy=0.0)
        >>> ax = plot_scan(range_scan(w, s))
    """
    scan = np.asarray(scan, dtype=float).ravel()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})

    angles = beam_angles(0.0, scan.size)
    closed_angles = np.append(angles, angles[0])
    closed_scan = np.append(scan, scan[0])
    ax.plot(closed_angles, closed_scan, "-o", ms=3, color="tab:orange")
    ax.set_theta_zero_location("N")
    ax.set_title("range scan [m], 0 = heading")
    return ax
