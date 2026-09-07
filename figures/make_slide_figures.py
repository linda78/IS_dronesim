"""One illustration per assignment topic, rendered from real dronesim data.

The world is 2D, but that means a horizontal slice through a world of vertical
walls with the drone at constant altitude. Drawing it isometrically shows that
honestly: extruded obstacles, the range scan as a ring at flight height.

    python figures/make_slide_figures.py

Writes figures/d1_noise.png ... d5_shift.png plus two generic scenes. Every file
is transparent, so it drops onto any slide background.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from dronesim import (  # noqa: E402
    FEATURE_NAMES,
    generate_dataset,
    make_world,
    range_scan,
    sample_state,
    simple_controller,
    simulate,
)

BLUE, MAGENTA, INK, MUTED, RED, AMBER = (
    "#0f5aa8", "#b5136c", "#1a1a1a", "#5a5a5a", "#8a1a1a", "#d98c00",
)
GROUND, WALL_TOP, WALL_L, WALL_R = "#eef1f6", "#dbe4f0", "#b9c9e0", "#9fb5d4"

WALL_HEIGHT = 2.2   # metres, purely for the drawing
FLIGHT_ALT = 1.1    # the slice the 2D simulator represents
MAX_RANGE = 5.0
DPI = 220
WORLD_FIG = (4.7, 3.2)   # matches the aspect of the isometric field
WORLD_ASPECT = WORLD_FIG[0] / WORLD_FIG[1]

MANEUVER_COLORS = {
    "none": "#9fb5d4", "brake": RED, "yaw_left": BLUE,
    "yaw_right": AMBER, "climb": MAGENTA,
}


# --------------------------------------------------------------------------------------
# isometric primitives
# --------------------------------------------------------------------------------------
def iso(x, y, z=0.0):
    """Isometric projection of a world point onto the drawing plane."""
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    return ((x - y) * c, (x + y) * s + z)


def _poly(ax, pts3d, **kw):
    ax.add_patch(Polygon([iso(*p) for p in pts3d], closed=True, **kw))


def draw_ground(ax, world, grid=5):
    _poly(ax, [(0, 0, 0), (world.width, 0, 0), (world.width, world.height, 0),
               (0, world.height, 0)],
          facecolor=GROUND, edgecolor="#c8cdd6", linewidth=1.0, zorder=0)
    if grid:
        for t in range(0, int(world.width) + 1, grid):
            ax.plot(*zip(iso(t, 0), iso(t, world.height)), color="#d5dae2", lw=0.6, zorder=1)
            ax.plot(*zip(iso(0, t), iso(world.width, t)), color="#d5dae2", lw=0.6, zorder=1)


def draw_wall(ax, rect, height=WALL_HEIGHT):
    """One obstacle as an extruded box: top plus the two faces turned to the viewer."""
    x, y, w, h = rect
    _poly(ax, [(x, y, 0), (x + w, y, 0), (x + w, y, height), (x, y, height)],
          facecolor=WALL_L, edgecolor=MUTED, linewidth=0.7, zorder=2)
    _poly(ax, [(x, y, 0), (x, y + h, 0), (x, y + h, height), (x, y, height)],
          facecolor=WALL_R, edgecolor=MUTED, linewidth=0.7, zorder=2)
    _poly(ax, [(x, y, height), (x + w, y, height), (x + w, y + h, height),
               (x, y + h, height)],
          facecolor=WALL_TOP, edgecolor=MUTED, linewidth=0.7, zorder=2)


def draw_world(ax, world, grid=5):
    """Ground plus every obstacle, far ones first so near ones cover them."""
    draw_ground(ax, world, grid=grid)
    for rect in sorted(world.obstacles, key=lambda r: -(r[0] + r[1] + (r[2] + r[3]) / 2)):
        draw_wall(ax, rect)


def draw_scan(ax, state, scan, alt=FLIGHT_ALT, color=BLUE, ring=True, zorder=6):
    """The beams as a ring at flight altitude, drawn from the drone outwards."""
    angles = state.heading + np.linspace(0, 2 * np.pi, len(scan), endpoint=False)
    tips = []
    for angle, dist in zip(angles, scan):
        ex, ey = state.x + dist * np.cos(angle), state.y + dist * np.sin(angle)
        hit = dist < MAX_RANGE - 1e-9
        ax.plot(*zip(iso(state.x, state.y, alt), iso(ex, ey, alt)),
                color=color, lw=1.0 if hit else 0.7,
                alpha=0.7 if hit else 0.16, zorder=zorder)
        tips.append(iso(ex, ey, alt))
        if hit:
            ax.plot(*iso(ex, ey, alt), marker="o", ms=3.2, color=color, zorder=zorder + 1)
    if ring:
        closed = tips + [tips[0]]
        ax.plot([p[0] for p in closed], [p[1] for p in closed],
                color=color, lw=1.2, alpha=0.45, zorder=zorder)


def draw_drone(ax, state, alt=FLIGHT_ALT, color=MAGENTA, size=1.0, arrow=True, rotors=True):
    px, py = iso(state.x, state.y, alt)
    if rotors:
        ax.plot(*zip(iso(state.x, state.y, 0), (px, py)),
                color=MUTED, lw=0.8, ls=":", zorder=5)
        for dx, dy in ((-0.42, -0.42), (-0.42, 0.42), (0.42, -0.42), (0.42, 0.42)):
            rx, ry = iso(state.x + dx * size, state.y + dy * size, alt)
            ax.plot([px, rx], [py, ry], color=INK, lw=1.6, zorder=9)
            ax.plot(rx, ry, marker="o", ms=6 * size, mfc="white", mec=INK, mew=1.3, zorder=10)
    ax.plot(px, py, marker="o", ms=7 * size, color=color, zorder=11)
    if arrow:
        speed = math.hypot(state.vx, state.vy)
        hx, hy = iso(state.x + 1.3 * size * state.vx / speed,
                     state.y + 1.3 * size * state.vy / speed, alt)
        ax.annotate("", xy=(hx, hy), xytext=(px, py), zorder=11,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0 * size,
                                    shrinkA=0, shrinkB=0))


def draw_trajectory(ax, trajectory, collided, alt=FLIGHT_ALT, lw=1.8, alpha=0.9):
    pts = [iso(x, y, alt) for x, y in trajectory]
    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            color=MAGENTA if collided else MUTED, lw=lw,
            ls=(0, (4, 3)) if collided else "-",
            alpha=alpha if collided else 0.45, zorder=8)
    if collided:
        ax.plot(*pts[-1], marker="X", ms=13, color=RED, mec="white", mew=1.2, zorder=12)


def finish(ax, state=None, zoom=None):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale_view()
    if zoom is not None and state is not None:
        cx, cy = iso(state.x, state.y, FLIGHT_ALT)
        ax.set_xlim(cx - zoom, cx + zoom)
        ax.set_ylim(cy - zoom / WORLD_ASPECT, cy + zoom / WORLD_ASPECT)


def crop_to_world(ax, world, pad=0.4, top=WALL_HEIGHT):
    """Fit the axes to the field, so no slide space is wasted on empty margin."""
    corners = [(0, 0), (world.width, 0), (world.width, world.height), (0, world.height)]
    pts = [iso(x, y, z) for x, y in corners for z in (0.0, top)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)


def label(ax, text, x=0.02, y=0.97, size=11, color=INK, weight="bold"):
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=size, color=color, fontweight=weight)


def save(fig, path, tight=True):
    if tight:
        fig.tight_layout(pad=0.15)
    fig.savefig(path, dpi=DPI, transparent=True)
    plt.close(fig)
    print(f"  wrote {path}")


def degrade(scan, sigma_frac, outlier_frac, rng, max_range=MAX_RANGE):
    """Illustrative sensor corruption: proportional Gaussian noise, then broken beams.

    Deliberately self contained. These figures are teaching material and must not
    depend on any solution code for the assignments they advertise.
    """
    rng = np.random.default_rng(rng)
    noisy = np.clip(scan + rng.normal(0.0, sigma_frac * scan), 0.0, max_range)
    is_outlier = rng.random(noisy.shape) < outlier_frac
    is_dropout = rng.random(noisy.shape) < 0.5
    return np.where(is_outlier, np.where(is_dropout, 0.0, max_range), noisy)


# --------------------------------------------------------------------------------------
# picking situations worth drawing
# --------------------------------------------------------------------------------------
def pick_state(world, seed_range=range(400), want_collision=True, margin=6.0):
    """A state with an obstacle ahead and enough room that the scan stays in the field."""
    best = None
    for seed in seed_range:
        rng = np.random.default_rng(seed)
        state = sample_state(world, rng)
        if not (margin < state.x < world.width - margin
                and margin < state.y < world.height - margin):
            continue
        scan = range_scan(world, state)
        collided, trajectory = simulate(world, state, rng=rng)
        if collided != want_collision:
            continue
        if 0.12 < float((scan < 4.99).mean()) < 0.6 and scan[0] < 4.5:
            score = -abs(scan[0] - 3.0)
            if best is None or score > best[0]:
                best = (score, state, scan, trajectory, collided)
    if best is None:
        raise RuntimeError("no suitable state found. Widen seed_range or reduce margin.")
    return best[1:]


# --------------------------------------------------------------------------------------
# one figure per topic
# --------------------------------------------------------------------------------------
def figure_d1(out):
    """D1: the same scan, clean against degraded. The corruption is the subject."""
    world = make_world(seed=0, n_obstacles=3)
    state, scan, _, _ = pick_state(world)
    # pick a seed that shows what the topic is about: a few clearly broken beams
    for seed in range(200):
        noisy = degrade(scan, sigma_frac=0.20, outlier_frac=0.10, rng=seed)
        broken = (noisy <= 1e-9) | ((noisy >= MAX_RANGE - 1e-9) & (scan < MAX_RANGE - 1e-9))
        if 3 <= int(broken.sum()) <= 5:
            break

    fig, ax = plt.subplots(figsize=WORLD_FIG)
    draw_world(ax, world)

    angles = state.heading + np.linspace(0, 2 * np.pi, len(scan), endpoint=False)
    clean_ring = [iso(state.x + d * np.cos(a), state.y + d * np.sin(a), FLIGHT_ALT)
                  for a, d in zip(angles, scan)]
    clean_ring.append(clean_ring[0])
    ax.plot([p[0] for p in clean_ring], [p[1] for p in clean_ring],
            color=MUTED, lw=1.3, ls=(0, (3, 2)), alpha=0.65, zorder=5)

    for angle, clean_d, noisy_d in zip(angles, scan, noisy):
        dropout = noisy_d <= 1e-9
        false_max = noisy_d >= MAX_RANGE - 1e-9 and clean_d < MAX_RANGE - 1e-9
        # a dropout is a reading of zero: drawn as a stub, because that is exactly what
        # the tabular features will see - an obstacle at distance zero
        draw_d = 0.45 if dropout else noisy_d
        ex, ey = state.x + draw_d * np.cos(angle), state.y + draw_d * np.sin(angle)
        broken = dropout or false_max
        ax.plot(*zip(iso(state.x, state.y, FLIGHT_ALT), iso(ex, ey, FLIGHT_ALT)),
                color=RED if broken else BLUE, lw=1.6 if broken else 1.0,
                alpha=0.95 if broken else 0.55, zorder=7 if broken else 6)
        ax.plot(*iso(ex, ey, FLIGHT_ALT), marker="o", ms=5.0 if broken else 3.0,
                color="white" if dropout else (RED if false_max else BLUE),
                mec=RED if broken else BLUE, mew=1.4 if dropout else 0.0, zorder=8)

    draw_drone(ax, state, arrow=False)
    finish(ax, state, zoom=7.6)
    ax.legend(handles=[
        Line2D([], [], color=MUTED, ls=(0, (3, 2)), lw=1.3, label="true range"),
        Line2D([], [], color=BLUE, lw=1.2, label="noisy reading"),
        Line2D([], [], color=RED, lw=1.6, marker="o", ms=4.5, mfc="white",
               mec=RED, label="dropout to 0"),
        Line2D([], [], color=RED, lw=1.6, marker="o", ms=4.5, mfc=RED,
               label="false max range"),
    ], loc="lower left", fontsize=7.5, frameon=False, handlelength=1.6,
       handletextpad=0.5, labelspacing=0.35)
    save(fig, out / "d1_noise.png")


def figure_d2(out):
    """D2: many flights, almost none of them a collision."""
    world = make_world(seed=0, n_obstacles=3)
    rng = np.random.default_rng(7)
    free, hits = [], []
    while len(free) < 46 or len(hits) < 2:
        state = sample_state(world, rng)
        collided, trajectory = simulate(world, state, rng=rng)
        if collided:
            # a collision after one step is a dot on the slide, not a flight
            if len(hits) < 2 and len(trajectory) > 12:
                hits.append((state, trajectory, collided))
        elif len(free) < 46:
            free.append((state, trajectory, collided))

    fig, ax = plt.subplots(figsize=WORLD_FIG)
    draw_world(ax, world)
    for state, trajectory, collided in free:
        draw_trajectory(ax, trajectory, collided, lw=1.1)
        ax.plot(*iso(state.x, state.y, FLIGHT_ALT), marker="o", ms=2.6,
                color=MUTED, alpha=0.55, zorder=8)
    for state, trajectory, collided in hits:
        draw_trajectory(ax, trajectory, collided, lw=2.2)
        draw_drone(ax, state, size=0.9, arrow=False, rotors=False)
    finish(ax)
    crop_to_world(ax, world)
    ax.legend(handles=[
        Line2D([], [], color=MUTED, lw=1.4, alpha=0.6, label="no collision"),
        Line2D([], [], color=MAGENTA, ls=(0, (4, 3)), lw=2.0, label="collision"),
    ], loc="lower left", fontsize=8, frameon=False, handlelength=2.6,
       handletextpad=0.6, labelspacing=0.4)
    save(fig, out / "d2_imbalance.png")


def figure_d3(out):
    """D3: the same samples, seen in two different coordinate systems."""
    world = make_world(seed=0, n_obstacles=3)
    df, _ = generate_dataset(world, n_samples=3500, seed=4)
    # keep the situations that actually see something ahead: d_front piles up hard at
    # max_range otherwise, and that spike would be the only thing the picture shows
    df = df[df["d_front"] < MAX_RANGE - 0.05]
    a = df["d_front"].to_numpy()
    b = df["speed"].to_numpy()
    y = df["collided"].to_numpy().astype(bool)

    def standardise(v):
        return (v - v.mean()) / v.std()

    xs, ys = standardise(a), standardise(b)
    angle = np.radians(35)
    xa = 3.2 * (xs * np.cos(angle) - ys * np.sin(angle))     # stretched and rotated
    ya = 0.75 * (xs * np.sin(angle) + ys * np.cos(angle))

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.5))
    for ax, (u, v), title in zip(axes, [(xs, ys), (xa, ya)],
                                 ["isotropic", "scaled + correlated"]):
        ax.scatter(u[~y], v[~y], s=6, c=BLUE, alpha=0.5, linewidths=0, label="free")
        ax.scatter(u[y], v[y], s=6, c=MAGENTA, alpha=0.6, linewidths=0, label="collision")
        ax.set_title(title, fontsize=10, color=INK, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color("#c8cdd6")
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.02), ncol=2,
                   fontsize=8, frameon=False, markerscale=2.2,
                   handletextpad=0.3, columnspacing=1.4)
    save(fig, out / "d3_features.png")


def figure_d4(out):
    """D4: the manoeuvre the reference controller picks, per situation."""
    world = make_world(seed=0, n_obstacles=3)
    rng = np.random.default_rng(11)
    picked, seen = [], {name: 0 for name in MANEUVER_COLORS}
    while len(picked) < 26:
        state = sample_state(world, rng)
        scan = range_scan(world, state)
        manoeuvre = simple_controller(state, scan, world)
        if seen[manoeuvre] >= (8 if manoeuvre == "none" else 5):
            continue
        seen[manoeuvre] += 1
        picked.append((state, manoeuvre))

    fig, ax = plt.subplots(figsize=WORLD_FIG)
    draw_world(ax, world)
    for state, manoeuvre in picked:
        draw_drone(ax, state, alt=0.35, color=MANEUVER_COLORS[manoeuvre],
                   size=1.05, rotors=False)
    finish(ax)
    crop_to_world(ax, world)
    ax.legend(handles=[Line2D([], [], color=c, marker="o", ls="", ms=6, label=n)
                       for n, c in MANEUVER_COLORS.items()],
              loc="lower left", fontsize=7.5, frameon=False, ncol=2,
              handletextpad=0.3, columnspacing=0.8, labelspacing=0.35)
    save(fig, out / "d4_maneuvers.png")


def figure_d5(out):
    """D5: the world the model is trained in, and the world it is flown in."""
    field = make_world(seed=0, n_obstacles=3)
    city = make_world(seed=5, n_obstacles=12, max_size=2.5, min_gap=0.8,
                      wind=(2.0, 1.0), gust_strength=1.5)

    fig, axes = plt.subplots(2, 1, figsize=(4.7, 6.2))
    for ax, world, title in zip(axes, [field, city], ["field (training)", "city (deployment)"]):
        draw_world(ax, world, grid=0)
        state, _, trajectory, collided = pick_state(world, seed_range=range(300), margin=5.0)
        draw_trajectory(ax, trajectory, collided, lw=1.8)
        draw_drone(ax, state, size=1.1, rotors=False)
        if world.wind != (0.0, 0.0):
            for gy in (2.0, 8.0, 14.0):                       # wind, low across the field
                sx, sy = iso(1.0, gy, 0.15)
                ex, ey = iso(6.0, gy + 2.5, 0.15)
                ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6,
                                            alpha=0.5, shrinkA=0, shrinkB=0))
            ax.text(*iso(0.5, 19.0, 0.15), "wind", fontsize=8.5, color=BLUE,
                    alpha=0.9, ha="center", va="bottom")
        finish(ax)
        crop_to_world(ax, world)
        label(ax, title, size=10)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.02)
    save(fig, out / "d5_shift.png", tight=False)


def figure_generic(out):
    """The shared scene: one drone, one 360 degree scan, one label."""
    world = make_world(seed=0, n_obstacles=3)
    state, scan, trajectory, collided = pick_state(world)

    for name, size, zoom in [("world_iso.png", (6.6, 4.5), None),
                             ("scene_iso.png", WORLD_FIG, 7.6)]:
        fig, ax = plt.subplots(figsize=size)
        draw_world(ax, world)
        draw_scan(ax, state, scan)
        draw_trajectory(ax, trajectory, collided)
        draw_drone(ax, state)
        finish(ax, state, zoom=zoom)
        if zoom is None:
            crop_to_world(ax, world)
        save(fig, out / name)


def main() -> int:
    out = Path("figures")
    out.mkdir(exist_ok=True)
    print("rendering:")
    figure_generic(out)
    figure_d1(out)
    figure_d2(out)
    figure_d3(out)
    figure_d4(out)
    figure_d5(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
