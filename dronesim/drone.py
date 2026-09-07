"""The drone: state, motion, and the collision label."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .world import World, occupied

DRONE_RADIUS = 0.35
"""Radius of the drone in meters. Used for every collision check.

The drone is deliberately not a point. Because of its size it also clips obstacles that
the beam pointing straight ahead does not see, which is what keeps the task from being
solvable by a single threshold on ``ttc`` (see tests/test_dataset.py)."""


@dataclass
class State:
    """Position and velocity of the drone.

    The heading (direction of flight) is derived from the velocity, it is not stored.

    Attributes:
        x: position in x in meters.
        y: position in y in meters.
        vx: velocity in x in m/s.
        vy: velocity in y in m/s.

    Example:
        >>> from dronesim import State
        >>> s = State(x=1.0, y=2.0, vx=3.0, vy=4.0)
        >>> s.speed
        5.0
        >>> round(float(np.degrees(s.heading)), 1)
        53.1
    """

    x: float
    y: float
    vx: float
    vy: float

    @property
    def speed(self) -> float:
        """Magnitude of the velocity in m/s."""
        return float(np.hypot(self.vx, self.vy))

    @property
    def heading(self) -> float:
        """Direction of flight in radians, measured from the x axis (``atan2(vy, vx)``)."""
        return float(np.arctan2(self.vy, self.vx))


def sample_state(
    world: World,
    rng,
    v_min: float = 0.5,
    v_max: float = 4.0,
    drone_radius: float = DRONE_RADIUS,
) -> State:
    """Draw one random drone state in the free space of the world.

    The position is uniform over the free space, the speed is uniform in
    ``[v_min, v_max]`` and the direction is uniform over 360 degrees. There is no
    filtering for "interesting" states: the resulting class balance is intended.

    Args:
        world: the world.
        rng: a ``numpy.random.Generator`` (or an int seed).
        v_min: smallest speed in m/s.
        v_max: largest speed in m/s.
        drone_radius: radius of the drone in meters.

    Returns:
        A :class:`State`.

    Example:
        >>> import numpy as np
        >>> from dronesim import World, sample_state
        >>> w = World(width=10.0, height=10.0)
        >>> s = sample_state(w, np.random.default_rng(0))
        >>> 0 <= s.x <= 10 and 0.5 <= s.speed <= 4.0
        True
    """
    rng = np.random.default_rng(rng)
    if v_min <= 0 or v_max < v_min:
        raise ValueError(
            f"expected 0 < v_min <= v_max, got v_min={v_min}, v_max={v_max}"
        )

    for _ in range(1000):
        x = float(rng.uniform(0.0, world.width))
        y = float(rng.uniform(0.0, world.height))
        if not occupied(world, [(x, y)], drone_radius)[0]:
            break
    else:
        raise ValueError(
            "could not find a free position after 1000 attempts. The world is almost "
            "completely blocked: use fewer or smaller obstacles."
        )

    speed = float(rng.uniform(v_min, v_max))
    angle = float(rng.uniform(0.0, 2 * np.pi))
    return State(x=x, y=y, vx=speed * np.cos(angle), vy=speed * np.sin(angle))


def simulate(
    world: World,
    state: State,
    horizon: float = 1.5,
    dt: float = 0.05,
    rng=None,
    drone_radius: float = DRONE_RADIUS,
) -> Tuple[bool, np.ndarray]:
    """Fly the drone forward and report whether it crashes within the horizon.

    The drone keeps its velocity and is pushed by the wind of the world. Per step it
    moves ``(velocity + wind + gust) * dt``. Without wind and gusts the path is a
    straight line. A collision happens as soon as a disc of ``drone_radius`` around the
    drone touches an obstacle or leaves the field.

    This function is the only source of the label used in all five assignments.

    Args:
        world: the world.
        state: where the drone starts and how fast it flies.
        horizon: how far into the future we look, in seconds.
        dt: time step in seconds.
        rng: a ``numpy.random.Generator``. Only needed for gusts.
        drone_radius: radius of the drone in meters.

    Returns:
        ``(collided, trajectory)`` where ``collided`` is a bool and ``trajectory`` is an
        array of shape (n_points, 2) with the flown path, including the start position
        and ending at the collision point.

    Example:
        >>> from dronesim import World, State, simulate
        >>> w = World(width=10.0, height=10.0)
        >>> collided, traj = simulate(w, State(x=9.0, y=5.0, vx=4.0, vy=0.0))
        >>> collided                      # flies into the right border
        True
        >>> collided, traj = simulate(w, State(x=9.0, y=5.0, vx=-4.0, vy=0.0))
        >>> collided                      # flies away from it
        False
    """
    if horizon <= 0 or dt <= 0:
        raise ValueError(f"expected horizon > 0 and dt > 0, got horizon={horizon}, dt={dt}")
    n_steps = max(1, int(round(horizon / dt)))

    steps = np.tile(
        np.array([state.vx + world.wind[0], state.vy + world.wind[1]]) * dt, (n_steps, 1)
    )
    if world.gust_strength > 0.0:
        if rng is None:
            raise ValueError(
                "the world has gust_strength > 0, so simulate() needs an rng. "
                "Pass rng=np.random.default_rng(seed)."
            )
        steps = steps + np.random.default_rng(rng).normal(
            0.0, world.gust_strength, size=(n_steps, 2)
        ) * dt

    start = np.array([[state.x, state.y]], dtype=float)
    trajectory = np.vstack([start, start + np.cumsum(steps, axis=0)])

    blocked = occupied(world, trajectory[1:], drone_radius)
    if bool(blocked.any()):
        first = int(np.argmax(blocked))
        return True, trajectory[: first + 2]
    return False, trajectory
