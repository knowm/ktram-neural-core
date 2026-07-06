"""Regenerate the Iris-classifier figures for the blog chapter.

    python examples/iris-classifier/figures.py [output_dir]

Default output_dir is a local figures/ folder (git-ignored); pass the article folder to write
straight into the prose. Everything runs on the same emulator + comparison harness the text
benchmark uses (shared.py), at fixed seeds.

Three figures:
  1. accuracy.png      — bar chart, all methods, mean ± std across seeds. AAT encoding can change
                         a linear classifier's accuracy (raw vs encoded); the kT-RAM lane is as
                         good as the reference linear models on the same encoding.
  2. confusion.png     — the kT-RAM lane vs the reference LogReg, same encoded AATs, side by side.
  3. encoding.png      — three 2D projections of the same four-dimensional Iris data. The lane
                         classifies in all four dimensions at once; any single view flattens that.
                         Setosa separates in every view; versicolor and virginica stay tangled in
                         all of them, and the lane's few misses sit in that overlap a line cannot
                         split — the baseline the reference linear solvers hit too.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))      # local shared.py
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # python/ on path

import numpy as np                                                    # noqa: E402
import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")                                                # headless
import matplotlib.pyplot as plt                                       # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix          # noqa: E402

import shared                                                          # noqa: E402

DEFAULT_OUT = str(pathlib.Path(__file__).resolve().parent / "figures")

CLASS_COLORS = ["tab:blue", "tab:green", "tab:red"]
GRID = "0.92"
SEED = 0
SEEDS = range(20)
# bars: reference linear models in grey, ours highlighted, raw baseline set apart
BAR_COLORS = {
    "LogReg (raw)": "0.75",
    "LogReg (AAT)": "tab:orange",
    "LinearSVC (AAT)": "tab:orange",
    "kT-RAM lane (AAT)": "tab:blue",
}


# ----------------------------------------------------------------- 1. accuracy comparison

def fig_accuracy(ax=None):
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(8.5, 5.0))
    accs = shared.multi_seed(SEEDS)
    names = shared.METHOD_ORDER
    means = [accs[n].mean() for n in names]
    stds = [accs[n].std() for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=[BAR_COLORS[n] for n in names], edgecolor="0.3")
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.80, 1.01)
    ax.axvline(0.5, color="0.8", lw=1.0, ls="--")   # raw | encoded divider
    ax.text(0.0, 0.815, "raw features", ha="center", fontsize=8, color="0.5")
    ax.text(2.0, 0.815, "same AAT encoding", ha="center", fontsize=8, color="0.5")
    ax.set_title(f"Iris → AATs — the kT-RAM lane vs reference linear classifiers "
                 f"(mean ± std, {len(list(SEEDS))} seeds)", fontsize=10.5)
    ax.grid(True, axis="y", color=GRID)
    return ax


# --------------------------------------------------------------- 2. confusion matrices

def _confusion_panel(ax, cm, names, title):
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
    ax.set_yticklabels(names, fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > thresh else "0.2")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)


def fig_confusion(ax=None):
    own = ax is None
    if own:
        _, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    r = shared.run_once(SEED)
    names = list(r["data"].target_names)
    for a, method in zip(ax, [shared.OURS, "LogReg (AAT)"]):
        cm = confusion_matrix(r["y_te"], r["preds"][method])
        acc = accuracy_score(r["y_te"], r["preds"][method])
        _confusion_panel(a, cm, names, f"{method}\nacc {acc:.3f}")
    return ax


# ----------------------------------------------------------------- 3. the dimensionality picture

# Iris feature columns: 0 sepal len, 1 sepal wid, 2 petal len, 3 petal wid.
# Three 2D projections of the same 4D data. The lane reads all four dimensions at once; any one
# view flattens that. Petals separate best; sepals barely; the cross view sits between.
PROJECTIONS = [(2, 3), (0, 1), (2, 0)]
ENC_SEED = 5   # a split with a couple of misses, so the overlap point is visible (SEED 0 is perfect)


def fig_encoding(axes=None):
    own = axes is None
    if own:
        _, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    r = shared.run_once(ENC_SEED)
    data = r["data"]
    X_all = np.vstack([r["X_tr"], r["X_te"]])
    y_all = np.concatenate([r["y_tr"], r["y_te"]])
    wrong = r["preds"][shared.OURS] != r["y_te"]   # the lane's misses, shown in every view

    for ax, (px, py) in zip(axes, PROJECTIONS):
        for c in range(len(data.target_names)):
            m = y_all == c
            ax.scatter(X_all[m, px], X_all[m, py], s=16, color=CLASS_COLORS[c],
                       label=data.target_names[c], alpha=0.8, zorder=2, edgecolor="none")
        ax.scatter(r["X_te"][wrong, px], r["X_te"][wrong, py], s=130, facecolor="none",
                   edgecolor="k", lw=1.5, zorder=3, label="lane missed")
        ax.set_xlabel(data.feature_names[px], fontsize=9)
        ax.set_ylabel(data.feature_names[py], fontsize=9)
        ax.grid(False)

    axes[0].legend(loc="best", fontsize=8, framealpha=0.9)
    plt.gcf().suptitle(
        "Three 2D views of the same four-dimensional Iris — the lane classifies in all four at "
        "once.\nSetosa separates everywhere; versicolor and virginica stay tangled in every "
        "view, and that overlap is the baseline a straight cut cannot beat.", fontsize=10.5)
    return axes


# ------------------------------------------------------------------------------------ main

FIGURES = {
    "accuracy.png": fig_accuracy,
    "confusion.png": fig_confusion,
    "encoding.png": fig_encoding,
}


def main(out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        fn()
        plt.tight_layout()
        plt.savefig(out / name, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
