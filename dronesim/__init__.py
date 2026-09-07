"""dronesim: a tiny drone collision dataset generator.

The package builds a world with obstacles, drops a drone into it, measures a range
scan and records whether the drone crashes within the next 1.5 seconds.

Typical use:
    >>> from dronesim import make_world, generate_dataset
    >>> world = make_world(seed=0)
    >>> df, scans = generate_dataset(world, n_samples=1000, seed=1)
    >>> scans.shape
    (1000, 32)
"""

from .dataset import (
    FEATURE_NAMES,
    TTC_MAX,
    features_from_scan,
    generate_dataset,
    load_dataset,
    save_dataset,
)
from .drone import DRONE_RADIUS, State, sample_state, simulate
from .hooks import MANEUVERS, Controller, simple_controller
from .plot import plot_scan, plot_world
from .sensor import beam_angles, range_scan
from .world import World, is_free, make_world, occupied

__version__ = "0.1.0"

__all__ = [
    "World",
    "make_world",
    "is_free",
    "occupied",
    "State",
    "sample_state",
    "simulate",
    "DRONE_RADIUS",
    "range_scan",
    "beam_angles",
    "features_from_scan",
    "generate_dataset",
    "save_dataset",
    "load_dataset",
    "FEATURE_NAMES",
    "TTC_MAX",
    "plot_world",
    "plot_scan",
    "Controller",
    "simple_controller",
    "MANEUVERS",
    "__version__",
]
