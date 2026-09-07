"""Generate a dataset, look at one situation, save everything to disk.

Run it from the project root:

    python examples/01_generate_and_plot.py
"""

import matplotlib.pyplot as plt
import numpy as np

from dronesim import (
    generate_dataset,
    make_world,
    plot_scan,
    plot_world,
    range_scan,
    sample_state,
    save_dataset,
    simulate,
)

# 1. build a world -----------------------------------------------------------
world = make_world(seed=0, n_obstacles=3)
print(f"world: {world.width} x {world.height} m, {len(world.obstacles)} obstacles")

# 2. generate a dataset ------------------------------------------------------
df, scans = generate_dataset(world, n_samples=2000, seed=1)
rate = df["collided"].mean()
print(f"samples: {len(df)},  collisions: {rate:.1%},  scans: {scans.shape}")
print(df.head())

# 3. look at one single situation -------------------------------------------
rng = np.random.default_rng(2)
state = sample_state(world, rng)
scan = range_scan(world, state)
collided, trajectory = simulate(world, state)
print(f"\nexample state: x={state.x:.1f} y={state.y:.1f} speed={state.speed:.1f} m/s")
print(f"collides within the next 1.5 s: {collided}")

fig = plt.figure(figsize=(11, 5))
ax1 = fig.add_subplot(1, 2, 1)
plot_world(world, state=state, scan=scan, trajectory=trajectory, ax=ax1)
ax1.set_title(f"collided = {collided}")
ax2 = fig.add_subplot(1, 2, 2, projection="polar")
plot_scan(scan, ax=ax2)
fig.tight_layout()

# 4. save --------------------------------------------------------------------
csv_path, npy_path = save_dataset(df, scans, "data/train")
print(f"\nsaved {csv_path} and {npy_path}")

fig.savefig("data/example_situation.png", dpi=120)
print("saved data/example_situation.png")
plt.show()
