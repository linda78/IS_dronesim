"""Tests for the dataset: reproducibility, class balance, difficulty."""

import numpy as np
import pytest

from dronesim import (
    State,
    features_from_scan,
    generate_dataset,
    load_dataset,
    make_world,
    range_scan,
    save_dataset,
    simple_controller,
)

N_SAMPLES = 5000


@pytest.fixture(scope="module")
def standard_dataset():
    """The dataset the students get with default settings."""
    world = make_world(seed=0)
    return generate_dataset(world, N_SAMPLES, seed=1)


def best_threshold_accuracy(values, labels):
    """Best accuracy any single threshold rule on `values` can reach."""
    order = np.argsort(values)
    y = np.asarray(labels)[order]
    n = y.size
    n_pos_before = np.concatenate([[0], np.cumsum(y)])       # positives left of the cut
    k = np.arange(n + 1)
    tp = n_pos_before                                        # predict "collided" left of the cut
    fn = y.sum() - n_pos_before
    tn = (n - k) - fn
    return float(((tp + tn) / n).max())


def test_dataset_has_the_expected_columns(standard_dataset):
    df, scans = standard_dataset
    expected = [
        "x", "y", "vx", "vy",
        "d_min", "angle_min", "d_front", "speed", "ttc", "n_close", "frac_blocked",
        "collided",
    ]
    assert list(df.columns) == expected
    assert scans.shape == (N_SAMPLES, 32)
    assert set(df["collided"].unique()) == {0, 1}


def test_same_seed_gives_the_same_dataset():
    world = make_world(seed=0)
    df_a, scans_a = generate_dataset(world, 200, seed=42)
    df_b, scans_b = generate_dataset(world, 200, seed=42)
    df_c, _ = generate_dataset(world, 200, seed=43)
    assert df_a.equals(df_b)
    assert np.array_equal(scans_a, scans_b)
    assert not df_a.equals(df_c)


def test_features_can_be_recomputed_from_the_scans(standard_dataset):
    """Students change `scans` and recompute the features, so both must line up."""
    df, scans = standard_dataset
    row = df.iloc[7]
    state = State(x=row["x"], y=row["y"], vx=row["vx"], vy=row["vy"])
    recomputed = features_from_scan(scans[7], state, max_range=5.0)
    for name, value in recomputed.items():
        assert value == pytest.approx(row[name])


def test_class_balance_is_workable(standard_dataset):
    """Between 10 and 40 percent collisions: usable for D1, D3, D4 and D5 as is."""
    df, _ = standard_dataset
    rate = df["collided"].mean()
    assert 0.10 < rate < 0.40, f"collision rate {rate:.3f} is outside the intended range"


def test_the_task_is_not_trivial(standard_dataset):
    """No single threshold on `ttc` may solve the task.

    If this test fails the task has become too easy. Adjust the horizon, the drone
    radius or the obstacle sizes until it passes again.
    """
    df, _ = standard_dataset
    accuracy = best_threshold_accuracy(df["ttc"].to_numpy(), df["collided"].to_numpy())
    assert accuracy < 0.97, f"a threshold on ttc alone already reaches {accuracy:.3f}"


def test_ttc_is_still_a_useful_feature(standard_dataset):
    """... but it must be clearly better than guessing, otherwise the task is broken."""
    df, _ = standard_dataset
    accuracy = best_threshold_accuracy(df["ttc"].to_numpy(), df["collided"].to_numpy())
    majority = max(df["collided"].mean(), 1 - df["collided"].mean())
    assert accuracy > majority + 0.1


def test_controller_adds_a_maneuver_column():
    world = make_world(seed=0)
    df, _ = generate_dataset(world, 300, seed=1, controller=simple_controller)
    assert "maneuver" in df.columns
    assert list(df.columns)[-1] == "collided"
    assert set(df["maneuver"]).issubset({"none", "brake", "yaw_left", "yaw_right", "climb"})
    assert df["maneuver"].nunique() >= 2


def test_save_and_load_roundtrip(tmp_path):
    world = make_world(seed=0)
    df, scans = generate_dataset(world, 50, seed=1)
    csv_path, npy_path = save_dataset(df, scans, tmp_path / "demo")
    assert csv_path.exists() and npy_path.exists()
    df2, scans2 = load_dataset(tmp_path / "demo")
    assert np.array_equal(scans, scans2)
    assert np.allclose(df.drop(columns=["collided"]), df2.drop(columns=["collided"]))
    assert (df["collided"] == df2["collided"]).all()


def test_load_without_saving_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="save_dataset"):
        load_dataset(tmp_path / "missing")
