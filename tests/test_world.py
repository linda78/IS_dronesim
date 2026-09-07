"""Tests for the world: placement, overlap, density."""

import numpy as np
import pytest

from dronesim import World, is_free, make_world, occupied
from dronesim.world import _rect_gap


def test_obstacles_are_inside_the_field():
    min_gap = 1.0
    world = make_world(seed=3, n_obstacles=5, min_gap=min_gap)
    for (x, y, w, h) in world.obstacles:
        assert x >= min_gap - 1e-9
        assert y >= min_gap - 1e-9
        assert x + w <= world.width - min_gap + 1e-9
        assert y + h <= world.height - min_gap + 1e-9


def test_obstacles_keep_the_minimum_gap():
    min_gap = 1.0
    world = make_world(seed=7, n_obstacles=6, min_gap=min_gap)
    for i, a in enumerate(world.obstacles):
        for b in world.obstacles[i + 1:]:
            assert _rect_gap(a, b) >= min_gap - 1e-9


def test_same_seed_gives_the_same_world():
    assert make_world(seed=1) == make_world(seed=1)
    assert make_world(seed=1) != make_world(seed=2)


def test_too_dense_raises_instead_of_looping_forever():
    with pytest.raises(ValueError, match="attempts"):
        make_world(seed=0, n_obstacles=50, min_size=2.5, max_size=3.0, min_gap=2.0)


def test_field_that_is_too_small_for_the_obstacles_raises():
    with pytest.raises(ValueError, match="does not fit"):
        make_world(seed=0, n_obstacles=1, max_size=3.0, min_gap=1.0, width=4.0, height=4.0)


def test_is_free_knows_obstacles_and_border():
    world = World(width=10.0, height=10.0, obstacles=[(4.0, 4.0, 2.0, 2.0)])
    assert is_free(world, 1.0, 1.0)
    assert not is_free(world, 5.0, 5.0)          # inside the obstacle
    assert not is_free(world, -0.1, 5.0)         # outside the field
    assert is_free(world, 3.5, 5.0)              # next to the obstacle
    assert not is_free(world, 3.5, 5.0, radius=0.6)   # ... but a fat drone touches it


def test_occupied_handles_many_points_at_once():
    world = World(width=10.0, height=10.0, obstacles=[(4.0, 4.0, 2.0, 2.0)])
    mask = occupied(world, [(1.0, 1.0), (5.0, 5.0), (20.0, 1.0)])
    assert mask.tolist() == [False, True, True]


def test_bad_arguments_explain_themselves():
    with pytest.raises(ValueError, match="n_obstacles"):
        make_world(seed=0, n_obstacles=-1)
    with pytest.raises(ValueError, match="min_size"):
        make_world(seed=0, min_size=2.0, max_size=1.0)
    with pytest.raises(ValueError, match="positive"):
        World(width=-1.0)
