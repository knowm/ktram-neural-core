"""Regenerate the figures for the "kT-bit Up Close" web article (Chapter 3b).

Run:  python examples/kt-bit/figures.py [output_dir]

Default output_dir is this lesson's own figures/ (gitignored); pass a path to write elsewhere,
e.g. the website article folder. Imports the shared single-synapse helpers, so the article and
the notebook can never drift from the code. The data plots are reproduced exactly (fixed
seeds); the two conceptual diagrams (2-1 vs 1-2, "two numbers") are schematic and meant as
drafts a designer can redraw in the chapter's hand-drawn style.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # examples/ -> import _common

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

from _common import experiments as ex

BLUE, GREEN, ORANGE, GREY = "tab:blue", "tab:green", "tab:orange", "0.55"
# own figures/ by default; pass the article path to write into the website blog folder
# (…/src/content/blog/thermodynamic-bit-up-close)
DEFAULT_OUT = str(pathlib.Path(__file__).resolve().parent / "figures")


def w_of(gas, gbs):
    return [(a - b) / (a + b) for a, b in zip(gas, gbs)]


def m_of(gas, gbs):
    return [a + b for a, b in zip(gas, gbs)]


# ----------------------------------------------------------------------------- data plots

def fig_inertia():
    """(G) Matched weight, mismatched magnitude — w(t) for a low-m vs high-m pair."""
    cores, traces, levels = ex.inertia_pair(model="float", init="medium", y0=0.3,
                                            levels=(0.05, 0.5), feedback_instr="RH",
                                            n=300, seed=1)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    labels = ["young  (small m = 0.01)", "mature  (large m = 0.10)"]
    colors = [BLUE, ORANGE]
    for (ys, gas, gbs), lab, c in zip(traces, labels, colors):
        # prepend the matched start (recorded traces begin one step in)
        w = [0.3] + w_of(gas, gbs)
        ax.plot(w, color=c, lw=1.8, label=lab)
    ax.axhline(1.0, color=GREY, lw=0.8, ls="--")
    ax.set_xlabel("feedback step  (FF, RH)")
    ax.set_ylabel("weight  w = (Ga - Gb) / (Ga + Gb)")
    ax.set_ylim(0.25, 1.03)
    ax.set_title("Same starting weight, identical feedback — magnitude sets the plasticity")
    ax.legend(loc="lower right")
    ax.grid(True, color="0.92")
    return fig


def fig_evidence():
    """(H) FF (evidence piles up, lean washes out) vs RF (the lean compounds to a decision)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=False)
    for ax, read, title in zip(axes, ("FF", "RF"),
                               ("FF repeated  —  evidence accumulates, w forgets its lean",
                                "RF repeated  —  the lean compounds into a decision")):
        core, (ys, gas, gbs) = ex.read_decay_vs_growth(read, model="float",
                                                       init="medium", n=4000, seed=2)
        w, m = w_of(gas, gbs), m_of(gas, gbs)
        ax.plot(w, color=BLUE, lw=1.6, label="weight  w")
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.set_xlabel("read step")
        ax.set_ylabel("weight  w", color=BLUE)
        ax.tick_params(axis="y", labelcolor=BLUE)
        ax.set_title(title, fontsize=10)
        ax.grid(True, color="0.92")
        axm = ax.twinx()
        axm.plot(m, color=ORANGE, lw=1.6, label="magnitude  m = Ga + Gb")
        axm.set_ylabel("magnitude  m", color=ORANGE)
        axm.tick_params(axis="y", labelcolor=ORANGE)
        axm.set_ylim(0, 0.21)
    return fig


def fig_devices():
    """(I) The same (FF,RH)xN then (FF,RL)xN pulse on four device models.

    The canonical Knowm-synapse signature — ramp up to a plateau, ramp back down — run on
    each model so the device's texture shows: Float infinitely fine, Byte an 8-bit integer
    staircase that plateaus at the quantization ceiling, MSS and RS the real stochastic
    physics. The single-pair lesson is the same on each; the model sets resolution and
    noise, nothing else.
    """
    panels = (("float", BLUE, "Float  —  infinitely fine", 400),
              ("byte", GREEN, "Byte  —  8-bit integer staircase", 120),
              ("mss", ORANGE, "MSS  —  stochastic (real device)", 400),
              ("rs", "tab:red", "RS  —  stochastic (real device)", 400))
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    for ax, (model, c, title, n) in zip(axes.flat, panels):
        core, (ys, gas, gbs) = ex.pulse_up_down(model=model, init="medium", n=n, seed=1)
        half = len(ys) // 2
        ax.plot(ys, color=c, lw=1.5)
        ax.axvline(half, color=GREY, lw=0.8, ls="--")     # the RH -> RL switch
        ax.axhline(0, color="0.85", lw=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("feedback step   (FF, RH) then (FF, RL)")
        ax.set_ylabel("weight  w")
        ax.grid(True, color="0.93")
    return fig


def fig_feedback_strip():
    """(H ref) The reverse-feedback instruction set at a glance, from one shared start."""
    combos = ["RH", "RL", "RU", "RA", "RZ"]
    captions = ["RH  — up", "RL  — down", "RU  — runs from 0",
                "RA  — back to 0", "RZ  — decay to 0"]
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.8), sharey=True)
    from _common.experiments import single_synapse_core, execute_n, Z
    for ax, combo, cap in zip(axes, combos, captions):
        core = single_synapse_core("float", "medium", seed=1)
        execute_n(core, "FF", "RH", 600)         # nudge to a clear positive start (w ~ 0.67)
        ys, gas, gbs = execute_n(core, "FF", combo, 1500)
        ax.plot(ys, color=BLUE, lw=1.4)
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.set_title(cap, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(True, color="0.93")
    axes[0].set_ylabel("weight  w")
    return fig


def fig_low_voltage():
    """(J) FFLV on Float — w and m flat: a read that adds no evidence."""
    core, (ys, gas, gbs) = ex.low_voltage_read("FFLV", model="float", init="medium",
                                              n=500, seed=1)
    w, m = w_of(gas, gbs), m_of(gas, gbs)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(w, color=BLUE, lw=1.8, label="weight  w")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="0.85", lw=0.8)
    ax.set_xlabel("low-voltage read step  (FFLV)")
    ax.set_ylabel("weight  w", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax.set_title("FFLV x 500 — the pair is interrogated, not disturbed")
    ax.grid(True, color="0.92")
    axm = ax.twinx()
    axm.plot(m, color=ORANGE, lw=1.8, label="magnitude  m")
    axm.set_ylabel("magnitude  m", color=ORANGE)
    axm.tick_params(axis="y", labelcolor=ORANGE)
    axm.set_ylim(0, max(m) * 2)
    return fig


# --------------------------------------------------------------------- conceptual diagrams

def _device_stack(ax, x, ga_frac, gb_frac):
    """Draw a two-resistor series stack (a over b) as filled bars; fracs in [0,1]."""
    w = 0.5
    ax.add_patch(Rectangle((x - w / 2, 1.0), w, ga_frac, facecolor=BLUE, alpha=0.55,
                           edgecolor="k"))
    ax.add_patch(Rectangle((x - w / 2, 1.0 - gb_frac), w, gb_frac, facecolor=GREEN,
                           alpha=0.55, edgecolor="k"))


def fig_two_numbers():
    """Hero/concept — one pair carries a weight (what y reads) AND a magnitude (what a float drops)."""
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.9, 3.0)

    # two divider stacks at the same weight, different magnitude
    for x, ga, gb, tag in ((2.0, 0.55, 0.30, "near the noise floor"),
                           (5.0, 1.5, 0.82, "near the rails")):
        ax.plot([x, x], [1.0 - gb - 0.15, 1.0 + ga + 0.15], color="k", lw=1.0)
        ax.add_patch(Rectangle((x - 0.28, 1.0), 0.56, ga, facecolor=BLUE, alpha=0.55,
                               edgecolor="k"))
        ax.add_patch(Rectangle((x - 0.28, 1.0 - gb), 0.56, gb, facecolor=GREEN, alpha=0.55,
                               edgecolor="k"))
        ax.text(x - 0.42, 1.0 + ga / 2, "Ga", ha="right", color=BLUE, fontsize=11, va="center")
        ax.text(x - 0.42, 1.0 - gb / 2, "Gb", ha="right", color=GREEN, fontsize=11, va="center")
        ax.plot(x, 1.0, "o", color="crimson", ms=7)
        ax.text(x + 0.42, 1.0, "y", color="crimson", fontsize=12, va="center")
        ax.text(x, -0.55, tag, ha="center", fontsize=9.5, color="0.4")

    ax.annotate("same weight  w", xy=(3.5, 1.0), xytext=(3.5, 2.55), ha="center",
                fontsize=11, arrowprops=dict(arrowstyle="-", color="crimson", lw=1.0))
    ax.text(3.5, 2.78, "(same y on both)", ha="center", fontsize=9, color="crimson")

    # the two numbers, written out
    ax.text(7.7, 2.15, r"weight   $w = \dfrac{G_a - G_b}{G_a + G_b}\in[-1,1]$",
            fontsize=13, va="center")
    ax.text(7.7, 1.35, r"magnitude   $m = G_a + G_b$", fontsize=13, va="center")
    ax.text(7.7, 2.55, "differential / common mode — all a read returns", fontsize=9,
            color=BLUE, va="center")
    ax.text(7.7, 0.95, "the common mode — divided out by a read", fontsize=9,
            color=GREEN, va="center")
    ax.plot([6.9, 6.9], [0.8, 2.3], color="0.8", lw=1.0)
    return fig


def fig_21_vs_12():
    """Part I (D) — the 2-1 series divider (a voltage, an average) vs the 1-2 current sum."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    for ax in axes:
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.6, 10)

    # ---- 2-1 : series voltage divider ----
    ax = axes[0]
    ax.set_title("2-1  —  series voltage divider", fontsize=12)
    ax.plot([5, 5], [8.0, 9.3], color="k"); ax.text(5, 9.55, "+V", ha="center")
    ax.plot([5, 5], [0.7, 2.0], color="k"); ax.text(5, 0.35, "-V", ha="center")
    ax.add_patch(Rectangle((4.4, 6.2), 1.2, 1.8, facecolor=BLUE, alpha=0.45, edgecolor="k"))
    ax.text(6.1, 7.1, "Ga", color=BLUE, va="center")
    ax.add_patch(Rectangle((4.4, 2.0), 1.2, 1.8, facecolor=GREEN, alpha=0.45, edgecolor="k"))
    ax.text(6.1, 2.9, "Gb", color=GREEN, va="center")
    ax.plot([5, 5], [3.8, 6.2], color="k")
    ax.plot(5, 5.0, "o", color="crimson", ms=8)
    ax.plot([5, 7.8], [5.0, 5.0], color="crimson", lw=1.2)
    ax.text(7.9, 5.0, "y", color="crimson", fontsize=13, va="center")
    ax.text(5, -1.4, r"$y = \dfrac{\sum (G_a - G_b)}{\sum (G_a + G_b)} \in [-1,1]$",
            ha="center", fontsize=12)
    ax.text(5, -0.5, "a voltage — already an average", ha="center", fontsize=9.5, color="0.4")

    # ---- 1-2 : current sum ----
    ax = axes[1]
    ax.set_title("1-2  —  current sum", fontsize=12)
    ax.plot([1.0, 1.0], [1, 9], color="k"); ax.text(0.7, 9.2, "SS", ha="center")
    ax.text(1.0, 0.5, "drive", ha="center", fontsize=9)
    for yy, c, lab, liney in ((6.6, BLUE, "Ia", 7.6), (3.4, GREEN, "Ib", 2.4)):
        ax.add_patch(Rectangle((3.0, yy - 0.5), 1.4, 1.0, facecolor=c, alpha=0.45,
                               edgecolor="k"))
        ax.plot([1.0, 3.0], [yy, yy], color="k")
        ax.plot([4.4, 8.0], [yy, liney], color=c, lw=1.2)
        ax.text(8.1, liney, lab.replace("I", "y"), color=c, fontsize=12, va="center")
    ax.text(3.7, 6.6, "Ga", color=BLUE, ha="center", va="center", fontsize=9)
    ax.text(3.7, 3.4, "Gb", color=GREEN, ha="center", va="center", fontsize=9)
    ax.add_patch(FancyArrowPatch((8.4, 7.6), (8.4, 2.4), arrowstyle="-", color="0.5"))
    ax.text(8.7, 5.0, r"$-$", fontsize=16, va="center")
    ax.text(5, -1.4, r"$y \propto \sum I_a - \sum I_b$", ha="center", fontsize=12)
    ax.text(5, -0.5, "a current — a sum that grows with how many fire", ha="center",
            fontsize=9.5, color="0.4")
    return fig


# ------------------------------------------------------------------------------------ main

FIGURES = {
    "two-numbers.png": fig_two_numbers,
    "21-vs-12.png": fig_21_vs_12,
    "inertia.png": fig_inertia,
    "evidence.png": fig_evidence,
    "feedback-strip.png": fig_feedback_strip,
    "devices.png": fig_devices,
    "low-voltage-read.png": fig_low_voltage,
}


def main(out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        fig = fn()
        fig.tight_layout()
        path = out / name
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    main(out_dir)
