# dronesim

A tiny drone simulator that produces datasets for the five assignments D1 to D5.

A quadrotor sits somewhere in a field with rectangular obstacles and flies straight
ahead. A 32 beam range scan tells you what is around it. The label says whether it
crashes within the next 1.5 seconds. That is all the package does: it is a data source,
not a physics engine and not an animation.

## Installation

```bash
pip install -e .          # from this folder
```

or simply copy the `dronesim/` folder next to your own code. You need Python 3.10 or
newer and `numpy`, `pandas` and `matplotlib`. For the baseline example you also need
`scikit-learn`.

Check that everything works:

```bash
python examples/01_generate_and_plot.py     # writes data/train.csv and data/train_scans.npy
python examples/02_train_baseline.py        # Random Forest on the tabular features
```

## The five functions you need

```python
from dronesim import make_world, sample_state, range_scan, generate_dataset, plot_world
```

| function | what it does |
| --- | --- |
| `make_world(seed, n_obstacles=3, wind=(0,0), gust_strength=0.0, ...)` | Builds the field with randomly placed rectangular obstacles. Everything about the world is controlled here. |
| `sample_state(world, rng)` | Draws one random drone state: position in free space, speed 0.5 to 4.0 m/s, uniform direction. |
| `range_scan(world, state, n_beams=32, max_range=5.0)` | The sensor. Returns 32 distances in meters, beam 0 points straight ahead, the rest continue counter clockwise. |
| `generate_dataset(world, n_samples, seed)` | Does all of the above `n_samples` times and returns `(df, scans)`: a table of features plus labels and the raw scans. |
| `plot_world(world, state, scan, trajectory)` | Draws one situation so you can see what your numbers mean. `plot_scan(scan)` gives the polar view. |

A five line start:

```python
from dronesim import make_world, generate_dataset

world = make_world(seed=0)
df, scans = generate_dataset(world, n_samples=5000, seed=1)
print(df.head(), df["collided"].mean())
```

`df` has one row per sample with the columns

`x, y, vx, vy` (the state), `d_min, angle_min, d_front, speed, ttc, n_close,
frac_blocked` (the tabular features) and `collided` (the label, 0 or 1).

`scans` is an array of shape `(n_samples, 32)` with the raw range scan of each row, in
the same order as `df`. Use `df[FEATURE_NAMES]` for the classic models and `scans` for
the 1D CNN.

Save and load with `save_dataset(df, scans, "data/train")` and
`load_dataset("data/train")`, which write and read `data/train.csv` and
`data/train_scans.npy`.

## What the label means

`collided` comes from `simulate(world, state)` and from nowhere else. The drone keeps
its velocity, is pushed by the wind of the world, and is stepped forward for 1.5 seconds
in steps of 0.05 s. If the disc of radius 0.35 m around the drone touches an obstacle or
leaves the field at any step, the label is 1.

Two consequences worth knowing:

* `ttc` (`d_front / speed`) is a **feature**, not the label. It looks like the answer but
  it is not: the drone is wider than a single beam, the sensor only sees 5 m, and wind
  bends the path. A single threshold on `ttc` gets stuck around 96 % accuracy.
* The class balance you get with the default settings is roughly 28 % collisions. It is
  what the geometry produces, nothing was filtered.

## What you may change

**Nothing inside `dronesim/`. Everything outside of it.**

The package is deliberately neutral: it knows nothing about your topic. You build your
dataset by choosing the arguments of `make_world` and `generate_dataset`, and by
transforming `df` and `scans` afterwards in your own script. If you change a scan, call
`features_from_scan(scan, state, max_range)` again so that the features match the scan
you actually feed to the model.

### D1 - Collision risk under sensor noise

Generate one clean dataset, then make noisy copies of `scans` in your own code: add
Gaussian noise proportional to the measured range (0 %, 5 %, 10 %, 20 %, 40 %) and turn
a share of the beams into outliers (dropouts to 0 or readings at `max_range`). Recompute
the features with `features_from_scan` for every noise level. The label stays the same,
only the input degrades. Train the CNN directly on the noisy `scans`.

### D2 - Rare collisions

Generate a large pool (for example 100 000 samples, about 28 % of them collisions), then
subsample rows from `df` to build datasets of the same total size with collision ratios
of 50 %, 20 %, 5 %, 1 % and 0.2 %. Always subsample `df` and `scans` with the same row
indices so they stay aligned. Do not change the simulator to get rare collisions, build
your ratio yourself.

### D3 - Units and correlated features

Take the tabular features and change the geometry of the feature space in your own code:
convert distances to millimetres, keep angles in degrees, and mix columns with a rotation
matrix to create correlations. Then compare raw, standardised and whitened versions of
the same data. The samples do not change at all, only their coordinates.

### D4 - Evasive maneuver classification

Pass a controller to get a `maneuver` column:

```python
from dronesim import generate_dataset, simple_controller
df, scans = generate_dataset(world, 20000, seed=1, controller=simple_controller)
```

`simple_controller` in `dronesim/hooks.py` uses sharp thresholds, so the five classes are
almost perfectly separable. That is your low overlap case. Write your own controller with
the same signature `(state, scan, world) -> str` and widen the ambiguous zone, for example
by drawing the class at random near a threshold, to get the moderate and high overlap
cases.

### D5 - Train in the field, test in the city

Everything you need is already in `make_world`:

```python
field = make_world(seed=0, n_obstacles=3)                                  # source
city  = make_world(seed=0, n_obstacles=12, max_size=2.5, min_gap=0.8,
                   wind=(2.0, 1.0), gust_strength=1.5)                     # target
```

Increase `n_obstacles` and `wind` / `gust_strength` in steps to get target environments
with increasing shift. Train on the field only, test on each city, then add 1 %, 5 % and
20 % of the target data to your training set and repeat.

## Neural networks

The examples use scikit-learn only, so that they run on any laptop without a GPU. The MLP
and the CNN of your project are implemented in PyTorch. The CNN takes the raw range scan
(`scans`, shape `(n_samples, 32)`) as a 1D input with one channel. Because the scan is a
full circle, a circular padding is the natural choice.

## Tests

```bash
pytest
```

About 50 checks (the tests in `tests/` plus the examples in every docstring), roughly a
second in total. Two of them protect the difficulty of the task: the
collision rate must stay between 10 % and 40 %, and a threshold on `ttc` alone must stay
below 97 % accuracy. If you change anything inside the package, run the tests. If those
two fail, your dataset is no longer the one the assignment was designed for.
