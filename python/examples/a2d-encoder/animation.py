"""One-off illustration — the A2D encoder's adaptive bins finding a clumpy distribution.

    python examples/a2d-encoder/animation.py [output_dir]

A standalone demo, not tied to any classifier: synthetic 2-D data with several clumps, encoded by
an A2DEncoder at a higher bit depth than the Iris example. The bin grid starts at the uniform
binning and migrates toward equal-occupancy as `encode_adapt` walks the data — the bins bunch up
inside the clumps and stretch across the empty gaps, which is the whole point of adaptive binning.
Writes `bin-adaptation-clumpy.gif` via matplotlib's PillowWriter (no ffmpeg).

Default output_dir is this lesson's own figures/ (gitignored); pass a path to write elsewhere.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # python/ on path

import numpy as np                                                    # noqa: E402
import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")                                                # headless
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter          # noqa: E402

from ktram_neural_core.encode import A2DEncoder                       # noqa: E402

DEFAULT_OUT = str(pathlib.Path(__file__).resolve().parent / "figures")

# Smoothness note: each tree threshold only moves when an input lands on its path, so the deep
# nodes that set the FINE bin edges (~1/2**depth of the data each) update rarely and twitch if the
# steps are large or few are averaged per frame. A small LRATE (small EMA steps) + plenty of data
# + many updates integrated per frame smooths it; the deep edges then refine gradually and late,
# after the coarse structure has settled.
BITS = 5               # 2**BITS = 32 bins per axis (higher than the Iris example's 8)
LRATE = 0.005          # A2D bin-migration rate (encode_adapt EMA); smaller = smoother motion
N = 4000               # synthetic points — enough that the deep, rarely-hit bins still converge
EPOCHS = 6             # passes over the data
SEED = 0
SNAPSHOT_EVERY = 150   # integrate many updates per frame -> smooth (~160 frames total)
FPS = 20
HOLD_START = 8
HOLD_END = 28
PLOT_N = 1200          # scatter a subsample so the cloud stays readable at high N
EDGE_COLOR = "0.5"
DOT_COLOR = "tab:blue"

# (center, std, weight) — chosen so BOTH marginals are clearly multimodal, since A2D bins each
# axis by its own marginal. The bins should concentrate near each clump and thin out in between.
CLUMPS = [
    ((0.18, 0.22), 0.035, 0.28),
    ((0.30, 0.78), 0.045, 0.22),
    ((0.72, 0.30), 0.040, 0.25),
    ((0.85, 0.82), 0.030, 0.15),
    ((0.55, 0.55), 0.060, 0.10),
]


def make_data(n, seed):
    rng = np.random.default_rng(seed)
    weights = np.array([w for _, _, w in CLUMPS])
    weights = weights / weights.sum()
    counts = rng.multinomial(n, weights)
    pts = [rng.normal((cx, cy), sd, size=(k, 2))
           for ((cx, cy), sd, _), k in zip(CLUMPS, counts)]
    X = np.clip(np.vstack(pts), 0.0, 1.0)
    rng.shuffle(X)                              # so a plotted subsample is representative
    return X


def bin_edges(encoder, dim, lo, hi, ndims=2, n=2000):
    """Black-box read of one dimension's adaptive bin boundaries: sweep the value and record
    where the bin index changes (A2D dimensions are independent)."""
    xs = np.linspace(lo, hi, n)
    base = np.zeros(ndims)
    edges, prev = [], None
    for x in xs:
        v = base.copy()
        v[dim] = x
        b = encoder.encode(v)[dim]
        if prev is not None and b != prev:
            edges.append(x)
        prev = b
    return edges


def collect_frames(X):
    enc = A2DEncoder(dims=2, bits=BITS, init_min=X.min(0), init_max=X.max(0), l=LRATE)
    lo, hi = X.min(0), X.max(0)

    def snap(step):
        ex = bin_edges(enc, 0, lo[0], hi[0])
        ey = bin_edges(enc, 1, lo[1], hi[1])
        return (step, ex, ey)

    rng = np.random.default_rng(SEED + 1)
    frames = [snap(0)]
    step = 0
    for _ in range(EPOCHS):
        for i in rng.permutation(len(X)):
            enc.encode_adapt(X[i])
            step += 1
            if step % SNAPSHOT_EVERY == 0:
                frames.append(snap(step))
    if frames[-1][0] != step:
        frames.append(snap(step))

    total = step
    frames = [frames[0]] * HOLD_START + frames + [frames[-1]] * HOLD_END
    return lo, hi, frames, total


def main(out_dir):
    X = make_data(N, SEED)
    lo, hi, frames, total = collect_frames(X)

    fig, ax = plt.subplots(figsize=(6.5, 6.2), dpi=90)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.scatter(X[:PLOT_N, 0], X[:PLOT_N, 1], s=9, color=DOT_COLOR, alpha=0.18,
               edgecolor="none", zorder=2)
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")

    grid = []

    def update(k):
        for ln in grid:
            ln.remove()
        grid.clear()
        step, ex, ey = frames[k]
        for e in ex:
            grid.append(ax.axvline(e, color=EDGE_COLOR, lw=0.6, zorder=1))
        for e in ey:
            grid.append(ax.axhline(e, color=EDGE_COLOR, lw=0.6, zorder=1))
        pct = 100.0 * step / total
        ax.set_title(f"A2D adaptive binning on a clumpy distribution — {pct:3.0f}%\n"
                     f"{2 ** BITS} bins/axis: uniform → equal-occupancy", fontsize=11)
        return grid

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000.0 / FPS, blit=False)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "bin-adaptation-clumpy.gif"
    anim.save(str(path), writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"wrote {path}  ({len(frames)} frames, {2 ** BITS} bins/axis, {total} steps)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
