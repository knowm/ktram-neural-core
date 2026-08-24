"""Render the basis-encoder static figures from the cached experiment results (see experiments.py).

    python figures.py            # all static figures
Outputs land in the gitignored figures/ dir; the final article copies live in the website repo.
"""

import pickle
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "figures"

INK = "#1a1a1a"
GRID = "#d8d8d8"
ARM_COLORS = {"both on": "#2a9d8f", "exclusion off": "#e76f51", "recruitment off": "#e9c46a"}
CMAP = "magma"


def _colnorm(m):
    """Normalize each generator column to a fraction (where does that generator's mass land)."""
    col = m.sum(axis=0, keepdims=True)
    col[col == 0] = 1
    return m / col


def _sort_rows(m):
    """Lanes onto their plurality generator, sorted by it; idle lanes last. Returns the reordered
    matrix and the original (arbitrary) lane indices in their new row order."""
    n, k = m.shape
    keyed = []
    for ch in range(n):
        if m[ch].sum():
            keyed.append((int(np.argmax(m[ch])), -int(m[ch].sum()), ch))
        else:
            keyed.append((k + 1, 0, ch))
    order = [ch for _, _, ch in sorted(keyed)]
    return m[order], order


def fig_triptych(data):
    arms = data["arms"]
    labels = [lbl for lbl, *_ in [("both on",), ("exclusion off",), ("recruitment off",)]
              if lbl in arms]
    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 4.8))
    for ax, label in zip(axes, labels):
        m, order = _sort_rows(arms[label]["matrix"].astype(float))
        ax.imshow(_colnorm(m), aspect="auto", cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
        mean = arms[label]["mean"]
        ax.set_title(f"{label}\ncoverage {mean['coverage']:.2f}   purity {mean['purity']:.2f}",
                     fontsize=11, color=INK)
        ax.set_xlabel("true generator (source pattern)", fontsize=9, color=INK)
        # Show the arbitrary lane indices in their reordered positions — a handful of ticks makes
        # it plain the rows were shuffled, so no reader mistakes the diagonal for "lane i = class i".
        ticks = list(range(0, len(order), max(1, len(order) // 8)))
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"L{order[t]}" for t in ticks], fontsize=6)
        if label == labels[0]:
            ax.set_ylabel("lane id (unlabeled — rows reordered)", fontsize=9, color=INK)
        ax.tick_params(labelsize=7, colors=INK)
    fig.suptitle("One WTA group finds a codebook with no labels — each source pattern lands on one lane",
                 fontsize=12.5, color=INK, y=1.02)
    fig.text(0.5, -0.02,
             "Unsupervised: lanes carry no identity. Within each panel the rows are reordered "
             "afterward so each lane sits beside the pattern it specialized on (note the shuffled "
             "L-ids on the y-axis).\nThe diagonal is the one-to-one matching the group discovered on "
             "its own — not an imposed alignment. Colour = fraction of a pattern's samples won by "
             "that lane.",
             ha="center", va="top", fontsize=8, color="#555")
    fig.tight_layout()
    p = OUT / "synthetic-ablation.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_metrics_bars(data):
    arms = data["arms"]
    labels = [l for l in ("both on", "exclusion off", "recruitment off") if l in arms]
    metrics = ["coverage", "purity", "utilization"]
    x = np.arange(len(metrics))
    w = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, label in enumerate(labels):
        means = [arms[label]["mean"][k] for k in metrics]
        stds = [arms[label]["std"][k] for k in metrics]
        ax.bar(x + i * w, means, w, yerr=stds, capsize=3, label=label,
               color=ARM_COLORS.get(label, "#888"), edgecolor="white", linewidth=0.6)
    ax.set_xticks(x + w * (len(labels) - 1) / 2)
    ax.set_xticklabels([m.capitalize() for m in metrics], color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score", color=INK)
    ax.set_title(f"Ablation over {len(data['seeds'])} seeds — exclusion prevents collapse, "
                 "recruitment keeps the bank in play", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = OUT / "synthetic-ablation-bars.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_sweep(sweep):
    rows = sweep["rows"]
    corr = [r["corruption"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for ax, metric in zip(axes, ("coverage", "purity")):
        for label in ("both on", "exclusion off", "recruitment off"):
            if label not in rows[0]:
                continue
            ys = [r[label][metric] for r in rows]
            ax.plot(corr, ys, "-o", color=ARM_COLORS.get(label, "#888"), label=label, markersize=4)
        ax.set_xlabel("corruption", color=INK)
        ax.set_title(metric.capitalize(), color=INK)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("score", color=INK)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Difficulty sweep — the codebook holds until corruption swamps the signal",
                 fontsize=11, color=INK)
    fig.tight_layout()
    p = OUT / "synthetic-sweep.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_separability(sep):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    bars = ["raw input", "frozen basis code"]
    vals = [sep["raw"], sep["basis"]]
    ax.bar(bars, vals, color=["#adb5bd", "#2a9d8f"], edgecolor="white", width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=11, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("linear-decoder accuracy", color=INK)
    ax.set_title(f"Unsupervised features lift linear separability\n({sep['n_labels']} classes, "
                 f"{sep['n_groups']} groups x {sep['channels']} lanes)", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = OUT / "synthetic-separability.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    if (OUT / "separability.pkl").exists():
        with open(OUT / "separability.pkl", "rb") as f:
            fig_separability(pickle.load(f))
    if (OUT / "ablation.pkl").exists():
        with open(OUT / "ablation.pkl", "rb") as f:
            data = pickle.load(f)
        fig_triptych(data)
        fig_metrics_bars(data)
    if (OUT / "sweep.pkl").exists():
        with open(OUT / "sweep.pkl", "rb") as f:
            fig_sweep(pickle.load(f))
