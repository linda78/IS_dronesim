"""The range sensor: a 2D laser scan around the drone."""

from __future__ import annotations

import numpy as np

from .drone import State
from .world import World


def beam_angles(heading: float, n_beams: int = 32, fov_deg: float = 360.0) -> np.ndarray:
    """Absolute angles of the sensor beams in radians.

    For ``fov_deg=360`` the beams are spread evenly over the full circle and beam 0
    points exactly in the heading direction. For a smaller field of view the beams are
    centered on the heading, from ``-fov/2`` to ``+fov/2``.

    Example:
        >>> import numpy as np
        >>> from dronesim import beam_angles
        >>> np.degrees(beam_angles(0.0, n_beams=4)).tolist()
        [0.0, 90.0, 180.0, 270.0]
    """
    if n_beams < 1:
        raise ValueError(f"n_beams must be >= 1, got {n_beams}")
    if not 0 < fov_deg <= 360:
        raise ValueError(f"fov_deg must be in (0, 360], got {fov_deg}")

    if fov_deg == 360.0:
        offsets = np.linspace(0.0, 2 * np.pi, n_beams, endpoint=False)
    else:
        fov = np.radians(fov_deg)
        offsets = np.linspace(-fov / 2, fov / 2, n_beams)
    return heading + offsets


def _slab(origin: float, direction: np.ndarray, lo: float, hi: float):
    """Interval of ray parameters t for which the ray stays inside the strip lo..hi."""
    parallel = np.abs(direction) < 1e-12
    safe = np.where(parallel, 1.0, direction)
    a = (lo - origin) / safe
    b = (hi - origin) / safe
    t_near = np.minimum(a, b)
    t_far = np.maximum(a, b)
    # A ray parallel to the strip is either always inside it or never.
    inside = lo <= origin <= hi
    t_near = np.where(parallel, -np.inf if inside else np.inf, t_near)
    t_far = np.where(parallel, np.inf if inside else -np.inf, t_far)
    return t_near, t_far


def _ray_rect(ox: float, oy: float, dx: np.ndarray, dy: np.ndarray, rect) -> np.ndarray:
    """Distance from the origin to a rectangle along each beam, inf if the beam misses.

    Uses the slab method: intersect the x interval and the y interval of the rectangle
    in ray parameter space. No stepping along the ray.
    """
    rx, ry, rw, rh = rect
    t_near_x, t_far_x = _slab(ox, dx, rx, rx + rw)
    t_near_y, t_far_y = _slab(oy, dy, ry, ry + rh)
    t_enter = np.maximum(np.maximum(t_near_x, t_near_y), 0.0)
    t_exit = np.minimum(t_far_x, t_far_y)
    return np.where(t_exit >= t_enter, t_enter, np.inf)


def range_scan(
    world: World,
    state: State,
    n_beams: int = 32,
    max_range: float = 5.0,
    fov_deg: float = 360.0,
) -> np.ndarray:
    """Measure the distance to the nearest obstacle along each beam.

    The values are clean, there is no sensor noise. Adding noise is part of assignment D1.

    Args:
        world: the world.
        state: position and heading of the drone.
        n_beams: number of beams.
        max_range: maximum measurable distance in meters. Longer distances are clipped.
        fov_deg: field of view in degrees, centered on the heading.

    Returns:
        Array of shape (n_beams,) with distances in meters. With the default
        ``fov_deg=360`` beam 0 points straight ahead and the beams continue
        counter clockwise.

    Example:
        >>> from dronesim import World, State, range_scan
        >>> w = World(width=10.0, height=10.0)
        >>> scan = range_scan(w, State(x=5.0, y=5.0, vx=1.0, vy=0.0), n_beams=4, max_range=10.0)
        >>> scan.round(2).tolist()        # distance to the four borders
        [5.0, 5.0, 5.0, 5.0]
    """
    if max_range <= 0:
        raise ValueError(f"max_range must be > 0, got {max_range}")

    angles = beam_angles(state.heading, n_beams, fov_deg)
    dx, dy = np.cos(angles), np.sin(angles)
    ox, oy = float(state.x), float(state.y)

    # distance to the border of the field: where the ray leaves the field rectangle
    _, t_far_x = _slab(ox, dx, 0.0, world.width)
    _, t_far_y = _slab(oy, dy, 0.0, world.height)
    dist = np.maximum(np.minimum(t_far_x, t_far_y), 0.0)

    for rect in world.obstacles:
        dist = np.minimum(dist, _ray_rect(ox, oy, dx, dy, rect))

    return np.minimum(dist, max_range)
