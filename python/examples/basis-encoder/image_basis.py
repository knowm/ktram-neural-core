"""Shared render for the image-basis lessons (fashion_patches + fashion_mnist_prune).

Both run the encoder in the self-pruning mode (exclusion on, recruitment off, cycle reset on
gather-abandon): the lanes that keep winning sharpen, the rest are never rewarded and fade toward
init. So both want the same two renders, and they live here once:

  render_forming(npz, gif, png)
    GIF  — the whole field on ONE global scale, so a lane that loses the competition fades to dark
           on screen while the survivors brighten: you watch the codebook prune itself.
    PNG  — the survivors only (win count above a fraction of the top lane's), each on its own
           contrast. No dead lanes to stretch into speckle, so it reads as a clean basis.

The npz must carry `frames` [n_frames, S, H, W], `samples`, `s`, and `win` (the final per-lane read
win counts that decide who survived).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Black -> warm intensity: a confidently-dark pixel sits at ~0 and sinks into the background while
# the learned structure glows.
KWARM = LinearSegmentedColormap.from_list("kwarm",
                                          ["#000000", "#5a1a7a", "#e0562b", "#ffd24a", "#ffffff"])
BG = "#000000"


def survivors(win, frac=0.05):
    """Lane ids that survived the self-pruning, strongest first: win count above `frac` of the top
    lane's. This is the hardware-native readout — one counter per lane, one threshold."""
    win = np.asarray(win)
    keep = np.where(win > frac * (win.max() or 1))[0]
    return keep[np.argsort(win[keep])[::-1]]


def _grid(n):
    cols = int(np.ceil(np.sqrt(n)))
    return int(np.ceil(n / cols)), cols


def render_forming(npz, gif, png, *, cell=1.25, fps=12, hold_seconds=3.0, survivor_frac=0.05,
                   counter_label="images seen"):
    d = np.load(npz)
    frames, samples, s = d["frames"], d["samples"], int(d["s"])
    win = d["win"] if "win" in d else np.ones(s)
    switch_at = int(d["switch_at"]) if "switch_at" in d else None
    if switch_at is not None and switch_at < 0:   # -1 sentinel = no switch (pure formation run)
        switch_at = None
    strip = 0.32

    # -- GIF: the whole field, one global scale, forming (losers fade, survivors brighten) --------
    rows, cols = _grid(s)
    gmax = float(np.percentile(frames[-1], 99.9)) or 1.0
    global_vmax = float(np.percentile(frames[-1], 99.5)) or 1.0
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell, rows * cell + strip))
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor(BG)
    ims = []
    for j, ax in enumerate(axes.ravel()):
        ax.axis("off")
        ax.set_facecolor(BG)
        ims.append(ax.imshow(frames[0][j], cmap=KWARM, vmin=0, vmax=global_vmax,
                             interpolation="nearest") if j < s else None)
    counter = fig.text(0.99, 0.985, "", ha="right", va="top", fontsize=8.5, color="#b8b8b8",
                       family="monospace")

    def update(f):
        for j, im in enumerate(ims):
            if im is not None:
                im.set_data(frames[f][j])
        seen = int(samples[f])
        tag = "  ·  sharpening (recruitment off)" if switch_at is not None and seen >= switch_at else ""
        counter.set_text(f"{counter_label} = {seen:,}{tag}")
        return [im for im in ims if im is not None]

    top = (rows * cell) / (rows * cell + strip)
    fig.subplots_adjust(left=0.01, right=0.99, top=top, bottom=0.01, wspace=0.06, hspace=0.06)
    seq = list(range(len(frames))) + [len(frames) - 1] * int(fps * hold_seconds)
    FuncAnimation(fig, update, frames=seq, blit=False).save(gif, writer=PillowWriter(fps=fps))
    plt.close(fig)

    # -- PNG: the survivors only, each on its own contrast (a clean self-pruned basis) ------------
    surv = survivors(win, survivor_frac)
    n = len(surv)
    rows, cols = _grid(n)
    final = frames[-1]
    gmax = float(np.percentile(final[surv], 99.9)) or 1.0
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell, rows * cell + strip))
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor(BG)
    for j, ax in enumerate(axes.ravel()):
        ax.axis("off")
        ax.set_facecolor(BG)
        if j < n:
            vmax = max(float(np.percentile(final[surv[j]], 99.5)), 0.35 * gmax)
            ax.imshow(final[surv[j]], cmap=KWARM, vmin=0, vmax=vmax, interpolation="nearest")
    fig.text(0.99, 0.985, f"self-pruned basis: {n} of {s} lanes survive", ha="right", va="top",
             fontsize=8.5, color="#b8b8b8", family="monospace")
    top = (rows * cell) / (rows * cell + strip)
    fig.subplots_adjust(left=0.01, right=0.99, top=top, bottom=0.01, wspace=0.06, hspace=0.06)
    fig.savefig(png, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"wrote {gif} and {png}  ({n}/{s} lanes survive)")
