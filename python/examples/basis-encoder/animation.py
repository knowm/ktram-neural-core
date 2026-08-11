"""Render the basis-encoder animations from the cached snapshots (see experiments.py).

    python animation.py
Two GIFs, both from the ablation cache's snapshots:
  synthetic-forming.gif   — the both-on win-count matrix resolving from noise into a block-diagonal
  synthetic-ablation.gif  — the three arms forming side by side (collapse vs smear vs clean)

Row order is fixed to the FINAL both-on codebook so the diagonal emerges in place instead of the
lanes reshuffling every frame.
"""

import pickle
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "figures"
INK = "#1a1a1a"
CMAP = "magma"


def _colnorm(m):
    col = m.sum(axis=0, keepdims=True).astype(float)
    col[col == 0] = 1
    return m / col


def _final_order(m):
    n, k = m.shape
    keyed = []
    for ch in range(n):
        if m[ch].sum():
            keyed.append((int(np.argmax(m[ch])), -int(m[ch].sum()), ch))
        else:
            keyed.append((k + 1, 0, ch))
    return [ch for _, _, ch in sorted(keyed)]


def _shuffled_ticks(ax, order):
    """Label a handful of rows with their arbitrary lane ids in reordered position, so the diagonal
    can't be read as an imposed 'lane i = class i' alignment."""
    ticks = list(range(0, len(order), max(1, len(order) // 8)))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"L{order[t]}" for t in ticks], fontsize=6)


def forming(data, arm="both on", fps=6):
    snaps = data["arms"][arm]["snaps"]
    order = _final_order(data["arms"][arm]["matrix"])
    fig, ax = plt.subplots(figsize=(4.8, 5.1))
    im = ax.imshow(_colnorm(snaps[0][1].astype(float))[order], aspect="auto", cmap=CMAP,
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel("true generator (source pattern)", color=INK)
    ax.set_ylabel("lane id (unlabeled — rows fixed to final order)", color=INK, fontsize=9)
    _shuffled_ticks(ax, order)
    title = ax.set_title("", color=INK, fontsize=11)
    fig.text(0.5, 0.01,
             "Unsupervised — lanes have no identity.\n"
             "Rows fixed to the final order (shuffled L-ids); the matching emerges in place.",
             ha="center", va="bottom", fontsize=7, color="#555")
    fig.subplots_adjust(bottom=0.17)

    def update(i):
        frac, m = snaps[i]
        im.set_data(_colnorm(m.astype(float))[order])
        title.set_text(f"codebook forming — {int(frac * 100):3d}% through training")
        return im, title

    anim = FuncAnimation(fig, update, frames=len(snaps), blit=False)
    p = OUT / "synthetic-forming.gif"
    anim.save(p, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("wrote", p)


def triptych(data, fps=6):
    labels = [l for l in ("both on", "exclusion off", "recruitment off") if l in data["arms"]]
    orders = {l: _final_order(data["arms"][l]["matrix"]) for l in labels}
    n = min(len(data["arms"][l]["snaps"]) for l in labels)
    fig, axes = plt.subplots(1, len(labels), figsize=(4.0 * len(labels), 4.4))
    ims, titles = {}, {}
    for ax, l in zip(axes, labels):
        snaps = data["arms"][l]["snaps"]
        ims[l] = ax.imshow(_colnorm(snaps[0][1].astype(float))[orders[l]], aspect="auto",
                           cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
        titles[l] = ax.set_title(l, color=INK, fontsize=11)
        ax.set_xlabel("generator (source pattern)", fontsize=9, color=INK)
        if l == labels[0]:
            ax.set_ylabel("lane id (unlabeled — reordered)", fontsize=8, color=INK)
            _shuffled_ticks(ax, orders[l])
        else:
            ax.set_yticks([])
    sup = fig.suptitle("", color=INK, fontsize=12, y=0.99)
    fig.text(0.5, 0.005,
             "Unsupervised: lanes carry no identity; rows reordered per panel to reveal the matching "
             "each group found. The diagonal is emergent, not imposed.",
             ha="center", va="bottom", fontsize=7.5, color="#555")

    def update(i):
        for l in labels:
            frac, m = data["arms"][l]["snaps"][i]
            ims[l].set_data(_colnorm(m.astype(float))[orders[l]])
        sup.set_text(f"three WTA groups forming — {int(data['arms'][labels[0]]['snaps'][i][0]*100)}% "
                     "through training")
        return list(ims.values())

    anim = FuncAnimation(fig, update, frames=n, blit=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    p = OUT / "synthetic-ablation.gif"
    anim.save(p, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    with open(OUT / "ablation.pkl", "rb") as f:
        data = pickle.load(f)
    forming(data)
    triptych(data)
