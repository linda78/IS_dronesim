"""Tests for the range scan."""

import numpy as np
import pytest

from dronesim import State, World, make_world, range_scan
from dronesim.world import occupied


def test_empty_world_measures_the_distance_to_the_border():
    world = World(width=10.0, height=10.0)
    state = State(x=3.0, y=4.0, vx=1.0, vy=0.0)          # heading = +x
    scan = range_scan(world, state, n_beams=4, max_range=20.0)
    # beams point right, up, left, down
    assert scan == pytest.approx([7.0, 6.0, 3.0, 4.0])


def test_beam_zero_points_in_the_heading_direction():
    world = World(width=10.0, height=10.0)
    # heading = +y, so beam 0 must measure the distance to the top border
    state = State(x=3.0, y=4.0, vx=0.0, vy=2.0)
    scan = range_scan(world, state, n_beams=4, max_range=20.0)
    assert scan[0] == pytest.approx(6.0)
    assert scan == pytest.approx([6.0, 3.0, 4.0, 7.0])


def test_an_obstacle_in_front_shortens_exactly_that_beam():
    empty = World(width=10.0, height=10.0)
    blocked = World(width=10.0, height=10.0, obstacles=[(7.0, 4.5, 1.0, 1.0)])
    state = State(x=5.0, y=5.0, vx=1.0, vy=0.0)
    scan_empty = range_scan(empty, state, n_beams=4, max_range=20.0)
    scan_blocked = range_scan(blocked, state, n_beams=4, max_range=20.0)
    assert scan_blocked[0] == pytest.approx(2.0)          # front beam hits the obstacle
    assert scan_blocked[1:] == pytest.approx(scan_empty[1:])   # the others are unchanged


def test_values_are_clipped_to_max_range():
    world = World(width=100.0, height=100.0)
    scan = range_scan(world, State(x=50.0, y=50.0, vx=1.0, vy=0.0), max_range=5.0)
    assert scan == pytest.approx(np.full(32, 5.0))


def test_scan_matches_a_brute_force_reference():
    """The analytic slab intersection must agree with stepping along the ray."""
    world = make_world(seed=2, n_obstacles=5)
    state = State(x=6.0, y=7.0, vx=1.0, vy=0.7)
    max_range = 5.0
    scan = range_scan(world, state, n_beams=16, max_range=max_range)

    from dronesim.sensor import beam_angles

    step = 1e-3
    ts = np.arange(0.0, max_range + step, step)
    for beam, angle in enumerate(beam_angles(state.heading, 16)):
        points = np.column_stack(
            [state.x + ts * np.cos(angle), state.y + ts * np.sin(angle)]
        )
        blocked = occupied(world, points)
        expected = ts[np.argmax(blocked)] if blocked.any() else max_range
        assert scan[beam] == pytest.approx(expected, abs=2 * step)


def test_bad_arguments_explain_themselves():
    world = World()
    state = State(x=1.0, y=1.0, vx=1.0, vy=0.0)
    with pytest.raises(ValueError, match="n_beams"):
        range_scan(world, state, n_beams=0)
    with pytest.raises(ValueError, match="fov_deg"):
        range_scan(world, state, fov_deg=400.0)
    with pytest.raises(ValueError, match="max_range"):
        range_scan(world, state, max_range=0.0)
