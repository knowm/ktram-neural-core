"""Render recorded Chapter 8 results or the output of a new public run."""

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
BLUE, GREEN, ORANGE, INK = "#32658a", "#318274", "#cd8647", "#253645"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": INK, "text.color": INK, "axes.titleweight": "bold"})


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_run(report, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained")
    for name, color in (("fitted", BLUE), ("learned", GREEN)):
        for stage, training in enumerate(report["arms"][name]["teaching"]):
            curve = training["curve"]
            axes[0].plot([p["epoch"] for p in curve], [100 * p["validation_agreement"] for p in curve],
                         color=color, linestyle=["-", "--", ":"][stage],
                         label=f"{name}, stage {stage + 1}")
    axes[0].set(xlabel="Teaching epoch", ylabel="Validation target agreement (%)",
                title="Local teaching")
    axes[0].legend(fontsize=8)
    values = [report["teacher_test_accuracy"], report["proxy_test"]["accuracy"],
              report["arms"]["fitted"]["test"]["accuracy"], report["arms"]["learned"]["test"]["accuracy"]]
    axes[1].bar(["Teacher", "Proxy", "Fitted", "Learned"], np.array(values) * 100,
                color=[INK, ORANGE, BLUE, GREEN])
    axes[1].set(ylabel="Test accuracy (%)", ylim=(0, 105), title="Selected models on the test split")
    for i, v in enumerate(values):
        axes[1].text(i, 100 * v + 1.5, f"{100 * v:.2f}", ha="center", fontsize=10)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"{report['config']['dataset']} · seed {report['config']['seed']} · "
                 f"{report['config']['n_train']:,} training images", fontsize=12)
    save(fig, out, "run-results.png")


def historical(out):
    data = json.loads((HERE / "historical-results.json").read_text())
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    x = np.arange(4)
    for i, row in enumerate(data["runs"]):
        ax.bar(i + 0.18, 100 * row["taught_accuracy"], width=0.34, color=GREEN,
               label="Taught lane chain" if i == 0 else None)
        ax.text(i + 0.18, 100 * row["taught_accuracy"] + 0.22,
                f"{100 * row['taught_accuracy']:.2f}", ha="center", fontsize=10)
        if row["proxy_accuracy"] is not None:
            ax.bar(i - 0.18, 100 * row["proxy_accuracy"], width=0.34, color=BLUE,
                   label="Hard-symbol proxy" if i == 1 else None)
            ax.text(i - 0.18, 100 * row["proxy_accuracy"] + 0.22,
                    f"{100 * row['proxy_accuracy']:.2f}", ha="center", fontsize=10)
    ax.axhline(88.05, color=INK, linestyle="--", linewidth=1.2, label="Frozen teacher: 88.05%")
    ax.set_xticks(x, [f"{r['id']}\n{r['label']}\n{r['synapses']:,} synapses" for r in data["runs"]])
    ax.set(ylabel="Evaluation accuracy (%)", ylim=(75, 91), title="Recorded MLP assimilation experiments")
    ax.legend(loc="upper left", ncol=3, fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    save(fig, out, "historical-accuracy.png")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.7), layout="constrained")
    rows = [data["runs"][i] for i in (1, 2, 3)]
    labels = ["Teacher", "A4", "A6 micro", "A7"]
    digital = [210600] + [4 * r["encoder_parameters"] for r in rows]
    lanes = [0] + [2 * r["synapses"] for r in rows]
    ax.bar(labels, np.array(digital) / 1000, color=ORANGE, label="Digital parameters (fp32)")
    ax.bar(labels, np.array(lanes) / 1000, bottom=np.array(digital) / 1000,
           color=GREEN, label="Lane payload (2 bytes/synapse)")
    for i, total in enumerate(np.array(digital) + np.array(lanes)):
        ax.text(i, total / 1000 + 22, f"{total / 1000:,.1f}", ha="center", fontsize=10)
    ax.set(ylabel="Parameter payload (decimal kB)", ylim=(0, 1950), title="The input encoder must be counted")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    macs = [52544] + [r["encoder_macs"] for r in rows]
    bx.bar(labels, np.array(macs) / 1000, color=[INK, ORANGE, ORANGE, ORANGE])
    for i, value in enumerate(macs):
        bx.text(i, value / 1000 + 3, f"{value / 1000:.1f}", ha="center", fontsize=10)
    bx.set(ylabel="Digital MACs per image (thousands)", ylim=(0, 190), title="Encoder arithmetic alone")
    bx.text(0.5, 0.94, "Teacher: entire MLP\nA4–A7: encoder only; lane work additional",
            ha="center", va="top", transform=bx.transAxes, fontsize=9)
    for panel in (ax, bx):
        panel.grid(axis="y", alpha=0.2)
        panel.set_axisbelow(True)
    save(fig, out, "historical-costs.png")


def workflow(out):
    fig, ax = plt.subplots(figsize=(11.8, 5.6), layout="constrained")
    ax.set(xlim=(0, 12), ylim=(0, 5.5))
    ax.axis("off")

    def box(x, y, text, color, width=2.0):
        ax.add_patch(FancyBboxPatch((x, y), width, 0.72, boxstyle="round,pad=0.08",
                                    facecolor=color, edgecolor="none"))
        ax.text(x + width / 2, y + 0.36, text, ha="center", va="center", color="white", fontsize=10)

    def arrow(x, y, end):
        ax.annotate("", xy=(end, y), xytext=(x, y), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))

    ax.text(0, 5.18, "1. Train a conventional teacher, then freeze it", weight="bold", fontsize=13)
    box(0, 4.14, "Raw pixels", INK, 1.7)
    box(2.3, 4.14, "MLP\n784 → 64 → 32 → 10", INK, 3.0)
    box(6, 4.14, "Predictions + hidden activations\nSupervision for assimilation", INK, 5.5)
    arrow(1.78, 4.5, 2.18); arrow(5.38, 4.5, 5.88)
    ax.text(0, 3.55, "2. Learn the interfaces through a differentiable proxy", weight="bold", fontsize=13)
    box(0, 2.48, "Local encoder\n16 × 64 symbols", ORANGE, 2.2)
    box(2.9, 2.48, "Additive stage\n32 × 16 symbols", BLUE, 2.2)
    box(5.8, 2.48, "Additive stage\n16 × 16 symbols", BLUE, 2.2)
    box(8.7, 2.48, "Additive head\n10 class scores", BLUE, 2.2)
    for x in (2.28, 5.18, 8.08):
        arrow(x, 2.84, x + 0.5)
    ax.text(0, 1.91, "3. Teach fresh lane banks, freeze each one, and harvest its output", weight="bold", fontsize=13)
    box(0, 0.79, "Frozen encoder\nDigital arithmetic", ORANGE, 2.2)
    box(2.9, 0.79, "Lane stage 1\nLocal adapt updates", GREEN, 2.2)
    box(5.8, 0.79, "Lane stage 2\nLocal adapt updates", GREEN, 2.2)
    box(8.7, 0.79, "Lane head\nPredicted class", GREEN, 2.2)
    for x in (2.28, 5.18, 8.08):
        arrow(x, 1.15, x + 0.5)
    ax.text(0, 0.23, "Inference retains the bottom row. The teacher, proxy tables and activation probes are removed.", fontsize=11)
    save(fig, out, "assimilation-workflow.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=pathlib.Path, nargs="?", default=HERE / "figures")
    parser.add_argument("--results", type=pathlib.Path, help="plot a new run instead of historical chapter results")
    args = parser.parse_args()
    if args.results:
        plot_run(json.loads(args.results.read_text()), args.out)
    else:
        historical(args.out)
        workflow(args.out)
