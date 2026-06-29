"""Animate the A2D bins adapting to the data — phase 1 of the Iris benchmark.

    python examples/iris-classifier/animation.py [output_dir]

The petal-space scatter is fixed; the A2D bin grid starts at the uniform binning and migrates
toward equal-occupancy as `encode_adapt` walks the training data (the same phase-1 adaptation
the classifier runs before it freezes the encoder). Writes `bin-adaptation.gif` via matplotlib's
PillowWriter — no ffmpeg needed. Same encoder and config as the benchmark (shared.py), fixed seed.

Default output_dir is this lesson's own figures/ (gitignored); pass a path to write elsewhere,
e.g. the website article folder.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))      # local shared.py
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # python/ on path

import numpy as np                                                    # noqa: E402
import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")                                                # headless
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter          # noqa: E402

import shared                                                          # noqa: E402

DEFAULT_OUT = str(pathlib.Path(__file__).resolve().parent / "figures")

# Iris columns: 0 sepal len, 1 sepal wid, 2 petal len, 3 petal wid. Petals separate best.
PX, PY = 2, 3
CLASS_COLORS = ["tab:blue", "tab:green", "tab:red"]
SEED = 0
SNAPSHOT_EVERY = 5     # record the bin grid every N adaptation steps -> one frame
FPS = 20
HOLD_START = 8         # frames to linger on the uniform grid
HOLD_END = 24          # frames to linger on the final equal-occupancy grid
EDGE_COLOR = "0.55"


def collect_frames():
    """Walk phase-1 adaptation exactly as LinearClassifier.adapt_encoder does (same seed/order),
    snapshotting the petal-dimension bin edges. Returns the scatter data and the frame list."""
    data, X_tr, _, y_tr, _ = shared.load_split(SEED)
    enc = shared.build_encoder(X_tr)
    lo, hi = X_tr.min(0), X_tr.max(0)
    n = len(X_tr)

    def snap(step):
        ex = shared.bin_edges(enc, PX, lo[PX], hi[PX], n=1500)
        ey = shared.bin_edges(enc, PY, lo[PY], hi[PY], n=1500)
        return (step, ex, ey)

    rng = np.random.default_rng(SEED)
    frames = [snap(0)]
    step = 0
    for _ in range(shared.ENCODER_EPOCHS):
        for i in rng.permutation(n):
            enc.encode_adapt(X_tr[i])
            step += 1
            if step % SNAPSHOT_EVERY == 0:
                frames.append(snap(step))
    if frames[-1][0] != step:
        frames.append(snap(step))           # always end on the final state

    total = step
    # hold on the first and last frames so the eye can read start and end
    frames = [frames[0]] * HOLD_START + frames + [frames[-1]] * HOLD_END
    return data, X_tr, y_tr, lo, hi, frames, total


def main(out_dir):
    data, X, y, lo, hi, frames, total = collect_frames()

    fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=90)
    padx = 0.05 * (hi[PX] - lo[PX])
    pady = 0.05 * (hi[PY] - lo[PY])
    ax.set_xlim(lo[PX] - padx, hi[PX] + padx)
    ax.set_ylim(lo[PY] - pady, hi[PY] + pady)
    for c in range(len(data.target_names)):
        m = y == c
        ax.scatter(X[m, PX], X[m, PY], s=18, color=CLASS_COLORS[c],
                   label=data.target_names[c], alpha=0.8, edgecolor="none", zorder=2)
    ax.set_xlabel(data.feature_names[PX])
    ax.set_ylabel(data.feature_names[PY])
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    grid = []   # the moving bin lines, redrawn each frame

    def update(k):
        for ln in grid:
            ln.remove()
        grid.clear()
        step, ex, ey = frames[k]
        for e in ex:
            grid.append(ax.axvline(e, color=EDGE_COLOR, lw=0.8, zorder=1))
        for e in ey:
            grid.append(ax.axhline(e, color=EDGE_COLOR, lw=0.8, zorder=1))
        pct = 100.0 * step / total
        ax.set_title(f"A2D bins adapting to the data — {pct:3.0f}%\n"
                     "uniform binning → equal-occupancy", fontsize=11)
        return grid

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000.0 / FPS, blit=False)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "bin-adaptation.gif"
    anim.save(str(path), writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"wrote {path}  ({len(frames)} frames, {shared.ENCODER_EPOCHS} epochs, {total} steps)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
