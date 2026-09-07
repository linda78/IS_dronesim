"""Extension points. Only assignment D4 needs anything from this module."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .dataset import features_from_scan
from .drone import State
from .sensor import beam_angles
from .world import World

Controller = Callable[[State, np.ndarray, World], str]
"""A controller maps ``(state, scan, world)`` to the name of an evasive maneuver."""

MANEUVERS = ("none", "brake", "yaw_left", "yaw_right", "climb")
"""The five maneuver classes."""


def simple_controller(state: State, scan: np.ndarray, world: World, max_range: float = 5.0) -> str:
    """Reference controller with sharp, non overlapping thresholds.

    This is a starting point for assignment D4, not a good controller. The classes are
    almost perfectly separable on purpose, so that you can make them overlap yourself,
    for example by adding noise to the scan, by softening the thresholds or by making
    the decisions random near a threshold.

    Rules, checked in this order:
        brake:     time to collision below 0.5 s
        climb:     something closer than 0.8 m anywhere around the drone
        yaw_left:  obstacle ahead and more free space on the left
        yaw_right: obstacle ahead and more free space on the right
        none:      everything else

    Args:
        state: the drone state.
        scan: the range scan of that state.
        world: the world (unused here, but part of the controller interface).
        max_range: the ``max_range`` that was used for the scan.

    Returns:
        One of :data:`MANEUVERS`.

    Example:
        >>> from dronesim import World, State, range_scan, simple_controller
        >>> w = World(width=10.0, height=10.0)
        >>> s = State(x=5.0, y=5.0, vx=1.0, vy=0.0)
        >>> simple_controller(s, range_scan(w, s), w)
        'none'
    """
    scan = np.asarray(scan, dtype=float).ravel()
    f = features_from_scan(scan, state, max_range=max_range)

    if f["ttc"] < 0.5:
        return "brake"
    if f["d_min"] < 0.8:
        return "climb"
    if f["d_front"] < 3.0:
        # compare the free space on both sides of the heading
        angles = beam_angles(0.0, scan.size)
        left = scan[(angles > 0.1) & (angles < np.pi)]
        right = scan[(angles > np.pi) & (angles < 2 * np.pi - 0.1)]
        left_space = float(left.mean()) if left.size else 0.0
        right_space = float(right.mean()) if right.size else 0.0
        return "yaw_left" if left_space >= right_space else "yaw_right"
    return "none"
