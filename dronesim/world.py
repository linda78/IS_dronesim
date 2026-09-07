"""The world: a rectangular field with rectangular obstacles and wind."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

Rect = Tuple[float, float, float, float]


@dataclass
class World:
    """A rectangular field of size ``width`` x ``height`` with rectangular obstacles.

    Attributes:
        width: field size in x direction in meters. The field spans (0, 0) to (width, height).
        height: field size in y direction in meters.
        obstacles: list of rectangles ``(x, y, w, h)`` in meters, where (x, y) is the
            lower left corner.
        wind: constant velocity offset ``(wx, wy)`` in m/s that is added to the drone.
        gust_strength: standard deviation of the random gust per step in m/s.
            0.0 means no gusts.
        seed: the seed that was used to build this world (bookkeeping only).

    Example:
        >>> from dronesim import World, is_free
        >>> w = World(width=10.0, height=10.0, obstacles=[(4.0, 4.0, 2.0, 2.0)])
        >>> is_free(w, 5.0, 5.0)          # inside the obstacle
        False
        >>> is_free(w, 1.0, 1.0)          # open space
        True
    """

    width: float = 20.0
    height: float = 20.0
    obstacles: List[Rect] = field(default_factory=list)
    wind: Tuple[float, float] = (0.0, 0.0)
    gust_strength: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"width and height must be positive, got width={self.width}, height={self.height}"
            )
        self.obstacles = [tuple(float(v) for v in rect) for rect in self.obstacles]
        for rect in self.obstacles:
            if len(rect) != 4:
                raise ValueError(f"every obstacle must be a tuple (x, y, w, h), got {rect}")
            if rect[2] <= 0 or rect[3] <= 0:
                raise ValueError(f"obstacle width and height must be positive, got {rect}")
        if len(self.wind) != 2:
            raise ValueError(f"wind must be a tuple (wx, wy), got {self.wind}")
        self.wind = (float(self.wind[0]), float(self.wind[1]))
        if self.gust_strength < 0:
            raise ValueError(f"gust_strength must be >= 0, got {self.gust_strength}")


def occupied(world: World, points, radius: float = 0.0) -> np.ndarray:
    """Check many positions at once. True where the drone would be in collision.

    A position counts as occupied if a disc of ``radius`` around it touches an
    obstacle or sticks out of the field.

    Args:
        world: the world.
        points: array of shape (n, 2) with x/y positions in meters.
        radius: radius of the drone in meters.

    Returns:
        Boolean array of shape (n,).

    Example:
        >>> from dronesim import World, occupied
        >>> w = World(width=10.0, height=10.0, obstacles=[(4.0, 4.0, 2.0, 2.0)])
        >>> occupied(w, [(5.0, 5.0), (1.0, 1.0)]).tolist()
        [True, False]
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (n, 2), got {pts.shape}")
    x, y = pts[:, 0], pts[:, 1]

    out_of_field = (
        (x - radius < 0.0)
        | (x + radius > world.width)
        | (y - radius < 0.0)
        | (y + radius > world.height)
    )
    hit = out_of_field
    for (rx, ry, rw, rh) in world.obstacles:
        # closest point of the rectangle to (x, y)
        cx = np.clip(x, rx, rx + rw)
        cy = np.clip(y, ry, ry + rh)
        hit = hit | ((x - cx) ** 2 + (y - cy) ** 2 <= radius**2)
    return hit


def is_free(world: World, x: float, y: float, radius: float = 0.0) -> bool:
    """True if a drone of ``radius`` at (x, y) is inside the field and hits no obstacle.

    Example:
        >>> from dronesim import World, is_free
        >>> w = World(width=10.0, height=10.0)
        >>> is_free(w, 0.05, 5.0, radius=0.15)   # too close to the left border
        False
    """
    return not bool(occupied(world, [(float(x), float(y))], radius)[0])


def _rect_gap(a: Rect, b: Rect) -> float:
    """Smallest distance between two axis aligned rectangles (0 if they overlap)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(0.0, max(ax - (bx + bw), bx - (ax + aw)))
    dy = max(0.0, max(ay - (by + bh), by - (ay + ah)))
    return float(np.hypot(dx, dy))


def make_world(
    seed: int,
    n_obstacles: int = 3,
    min_size: float = 0.5,
    max_size: float = 3.0,
    min_gap: float = 1.0,
    width: float = 20.0,
    height: float = 20.0,
    wind: Sequence[float] = (0.0, 0.0),
    gust_strength: float = 0.0,
) -> World:
    """Build a world with randomly placed, non overlapping rectangular obstacles.

    Obstacles keep a distance of at least ``min_gap`` to each other and to the border,
    so there is always a corridor the drone can fly through.

    Args:
        seed: makes the world reproducible.
        n_obstacles: how many obstacles to place.
        min_size: smallest obstacle side length in meters.
        max_size: largest obstacle side length in meters.
        min_gap: minimum free distance between obstacles and to the border in meters.
        width: field size in x direction in meters.
        height: field size in y direction in meters.
        wind: constant velocity offset (wx, wy) in m/s.
        gust_strength: standard deviation of the gust per step in m/s.

    Returns:
        A :class:`World`.

    Raises:
        ValueError: if the obstacles do not fit into the field.

    Example:
        >>> from dronesim import make_world
        >>> w = make_world(seed=0, n_obstacles=3)
        >>> len(w.obstacles)
        3
        >>> make_world(seed=0, n_obstacles=3) == make_world(seed=0, n_obstacles=3)
        True
    """
    if n_obstacles < 0:
        raise ValueError(f"n_obstacles must be >= 0, got {n_obstacles}")
    if min_size <= 0 or max_size < min_size:
        raise ValueError(
            f"expected 0 < min_size <= max_size, got min_size={min_size}, max_size={max_size}"
        )
    if min_gap < 0:
        raise ValueError(f"min_gap must be >= 0, got {min_gap}")
    if max_size + 2 * min_gap >= min(width, height):
        raise ValueError(
            f"an obstacle of size {max_size} m plus a gap of {min_gap} m on both sides does not "
            f"fit into a field of {width} x {height} m. Reduce max_size or min_gap."
        )

    rng = np.random.default_rng(seed)
    obstacles: List[Rect] = []
    max_attempts = 200

    for i in range(n_obstacles):
        for attempt in range(max_attempts):
            w = float(rng.uniform(min_size, max_size))
            h = float(rng.uniform(min_size, max_size))
            x = float(rng.uniform(min_gap, width - min_gap - w))
            y = float(rng.uniform(min_gap, height - min_gap - h))
            candidate: Rect = (x, y, w, h)
            if all(_rect_gap(candidate, other) >= min_gap for other in obstacles):
                obstacles.append(candidate)
                break
        else:
            raise ValueError(
                f"could not place obstacle {i + 1} of {n_obstacles} after {max_attempts} attempts. "
                f"The field is too dense: use fewer obstacles, a smaller max_size, a smaller "
                f"min_gap, or a larger field."
            )

    return World(
        width=float(width),
        height=float(height),
        obstacles=obstacles,
        wind=(float(wind[0]), float(wind[1])),
        gust_strength=float(gust_strength),
        seed=int(seed),
    )
