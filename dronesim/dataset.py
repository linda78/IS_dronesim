"""Features and dataset generation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .drone import State, sample_state, simulate
from .sensor import range_scan
from .world import World

TTC_MAX = 99.0
"""Value used for the time to collision when nothing is in sight."""

FEATURE_NAMES = ["d_min", "angle_min", "d_front", "speed", "ttc", "n_close", "frac_blocked"]


def features_from_scan(scan, state: State, max_range: float = 5.0) -> Dict[str, float]:
    """Turn one range scan plus the drone state into tabular features.

    All features can be recomputed from a scan, so you can change the scan (for example
    add noise or change the units) and simply call this function again.

    The scan is assumed to be a full 360 degree scan, i.e. beam 0 points straight ahead.

    Features:
        d_min: smallest distance in the scan, in meters.
        angle_min: angle of that beam relative to the heading, in degrees in (-180, 180].
        d_front: distance of the beam pointing straight ahead, in meters.
        speed: magnitude of the velocity in m/s.
        ttc: time to collision, ``d_front / speed``, capped at ``TTC_MAX``.
        n_close: number of beams closer than 1 m.
        frac_blocked: fraction of beams that see something within ``max_range``.

    Note:
        ``ttc`` is a *feature*, not the label. The label comes from :func:`simulate`.

    Args:
        scan: array of shape (n_beams,) as returned by :func:`range_scan`.
        state: the drone state that produced the scan.
        max_range: the ``max_range`` that was used for the scan, in meters.

    Returns:
        dict of feature name to float.

    Example:
        >>> from dronesim import World, State, range_scan, features_from_scan
        >>> w = World(width=10.0, height=10.0)
        >>> s = State(x=5.0, y=5.0, vx=2.0, vy=0.0)
        >>> f = features_from_scan(range_scan(w, s), s)
        >>> f["d_front"], f["speed"], f["frac_blocked"]
        (5.0, 2.0, 0.0)
    """
    scan = np.asarray(scan, dtype=float).ravel()
    if scan.size == 0:
        raise ValueError("scan is empty, expected an array of shape (n_beams,)")

    n_beams = scan.size
    i_min = int(np.argmin(scan))
    angle_min = (i_min * 360.0 / n_beams) % 360.0
    if angle_min > 180.0:
        angle_min -= 360.0

    d_front = float(scan[0])
    speed = state.speed
    if speed <= 0.0 or d_front >= max_range - 1e-9:
        ttc = TTC_MAX
    else:
        ttc = min(d_front / speed, TTC_MAX)

    return {
        "d_min": float(scan.min()),
        "angle_min": float(angle_min),
        "d_front": d_front,
        "speed": float(speed),
        "ttc": float(ttc),
        "n_close": float((scan < 1.0).sum()),
        "frac_blocked": float((scan < max_range - 1e-9).mean()),
    }


def generate_dataset(
    world: World,
    n_samples: int,
    seed: int,
    horizon: float = 1.5,
    n_beams: int = 32,
    max_range: float = 5.0,
    controller: Optional[Callable[[State, np.ndarray, World], str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate a dataset of drone states, range scans and collision labels.

    For every sample: draw a random state, take a range scan, fly the drone forward for
    ``horizon`` seconds and record whether it crashed.

    Args:
        world: the world the samples are drawn in.
        n_samples: number of samples.
        seed: makes the dataset reproducible.
        horizon: look ahead time in seconds, passed to :func:`simulate`.
        n_beams: number of sensor beams.
        max_range: sensor range in meters.
        controller: optional function ``(state, scan, world) -> str``. If given, its
            maneuver is stored in the column ``maneuver`` (see ``dronesim.hooks``).

    Returns:
        ``(df, scans)``. ``df`` has the columns x, y, vx, vy, the features from
        :func:`features_from_scan`, optionally ``maneuver`` and finally ``collided``
        (0 or 1). ``scans`` has shape (n_samples, n_beams) and its rows line up with
        the rows of ``df``.

    Example:
        >>> from dronesim import make_world, generate_dataset
        >>> w = make_world(seed=0)
        >>> df, scans = generate_dataset(w, n_samples=100, seed=1)
        >>> len(df), scans.shape
        (100, (100, 32))
        >>> list(df.columns)[-1]
        'collided'
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    rng = np.random.default_rng(seed)
    scans = np.empty((n_samples, n_beams), dtype=float)
    rows = []

    for i in range(n_samples):
        state = sample_state(world, rng)
        scan = range_scan(world, state, n_beams=n_beams, max_range=max_range)
        collided, _ = simulate(world, state, horizon=horizon, rng=rng)

        row = {"x": state.x, "y": state.y, "vx": state.vx, "vy": state.vy}
        row.update(features_from_scan(scan, state, max_range=max_range))
        if controller is not None:
            row["maneuver"] = controller(state, scan, world)
        row["collided"] = int(collided)

        scans[i] = scan
        rows.append(row)

    return pd.DataFrame(rows), scans


def _paths(path) -> Tuple[Path, Path]:
    """Split a path into the csv file and the npy file that belong together."""
    path = Path(path)
    if path.suffix in (".csv", ".npy"):
        path = path.with_suffix("")
    return path.with_suffix(".csv"), path.with_name(path.name + "_scans.npy")


def save_dataset(df: pd.DataFrame, scans: np.ndarray, path) -> Tuple[Path, Path]:
    """Save a dataset as one CSV file and one npy file.

    Args:
        df: the table returned by :func:`generate_dataset`.
        scans: the scans returned by :func:`generate_dataset`.
        path: base path without extension, e.g. ``"data/train"``. This writes
            ``data/train.csv`` and ``data/train_scans.npy``.

    Returns:
        The two paths that were written.

    Example:
        >>> import tempfile, os
        >>> from dronesim import make_world, generate_dataset, save_dataset, load_dataset
        >>> w = make_world(seed=0)
        >>> df, scans = generate_dataset(w, 10, seed=1)
        >>> base = os.path.join(tempfile.mkdtemp(), "demo")
        >>> csv_path, npy_path = save_dataset(df, scans, base)
        >>> df2, scans2 = load_dataset(base)
        >>> len(df2) == len(df)
        True
    """
    if len(df) != len(scans):
        raise ValueError(
            f"df and scans must have the same number of rows, got {len(df)} and {len(scans)}"
        )
    csv_path, npy_path = _paths(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    np.save(npy_path, scans)
    return csv_path, npy_path


def load_dataset(path) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load a dataset that was written by :func:`save_dataset`.

    Args:
        path: the same base path that was passed to :func:`save_dataset`.

    Returns:
        ``(df, scans)``.
    """
    csv_path, npy_path = _paths(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} does not exist. Did you run save_dataset first?")
    if not npy_path.exists():
        raise FileNotFoundError(f"{npy_path} does not exist. Did you run save_dataset first?")
    return pd.read_csv(csv_path), np.load(npy_path)
