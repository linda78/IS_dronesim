"""Tests for the motion and the collision label."""

import numpy as np
import pytest

from dronesim import DRONE_RADIUS, State, World, simulate


def wall_world():
    """A 20 x 20 field with a wall from x = 10 to x = 11 over the full height."""
    return World(width=20.0, height=20.0, obstacles=[(10.0, 0.0, 1.0, 20.0)])


def test_flying_into_a_wall_collides():
    collided, traj = simulate(wall_world(), State(x=9.0, y=10.0, vx=3.0, vy=0.0))
    assert collided
    # the path stops at the wall, it does not continue through it
    assert traj[-1, 0] <= 10.0 + 1e-9


def test_flying_away_from_the_wall_does_not_collide():
    collided, traj = simulate(wall_world(), State(x=9.0, y=10.0, vx=-3.0, vy=0.0))
    assert not collided
    assert traj[-1, 0] == pytest.approx(9.0 - 3.0 * 1.5)


def test_leaving_the_field_counts_as_collision():
    world = World(width=20.0, height=20.0)
    assert simulate(world, State(x=19.0, y=10.0, vx=4.0, vy=0.0))[0]
    assert not simulate(world, State(x=10.0, y=10.0, vx=1.0, vy=0.0))[0]


def test_the_drone_has_a_size():
    """A drone that passes the wall closer than its radius still crashes."""
    world = World(width=20.0, height=20.0, obstacles=[(10.0, 10.0, 1.0, 1.0)])
    # flying along y = 10 - 0.5 * DRONE_RADIUS: a point would pass below the obstacle
    y = 10.0 - 0.5 * DRONE_RADIUS
    assert simulate(world, State(x=8.0, y=y, vx=3.0, vy=0.0))[0]
    # far enough below, no contact
    assert not simulate(world, State(x=8.0, y=10.0 - 2 * DRONE_RADIUS, vx=3.0, vy=0.0))[0]


def test_without_wind_the_path_is_a_straight_line():
    _, traj = simulate(World(), State(x=5.0, y=5.0, vx=1.0, vy=2.0), horizon=1.0, dt=0.1)
    steps = np.diff(traj, axis=0)
    assert np.allclose(steps, steps[0])
    assert traj.shape == (11, 2)


def test_wind_can_change_the_outcome():
    calm = World(width=20.0, height=20.0, obstacles=[(10.0, 0.0, 1.0, 20.0)])
    breezy = World(width=20.0, height=20.0, obstacles=[(10.0, 0.0, 1.0, 20.0)], wind=(4.0, 0.0))
    state = State(x=8.0, y=10.0, vx=0.5, vy=0.0)
    assert not simulate(calm, state)[0]
    assert simulate(breezy, state)[0]


def test_gusts_need_an_rng_and_are_reproducible():
    world = World(width=20.0, height=20.0, gust_strength=2.0)
    state = State(x=10.0, y=10.0, vx=1.0, vy=0.0)
    with pytest.raises(ValueError, match="rng"):
        simulate(world, state)
    a = simulate(world, state, rng=np.random.default_rng(0))[1]
    b = simulate(world, state, rng=np.random.default_rng(0))[1]
    assert np.allclose(a, b)


def test_bad_arguments_explain_themselves():
    with pytest.raises(ValueError, match="horizon"):
        simulate(World(), State(x=1.0, y=1.0, vx=1.0, vy=0.0), horizon=0.0)
