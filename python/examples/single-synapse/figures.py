"""Regenerate the single-synapse (Milestone-1) headline figures as PNGs.

Run:  python examples/single-synapse/figures.py [output_dir]

Emits the pulse up/down plots (Byte/Medium, Float/Medium) plus one MSS and one RS figure,
so they can be eyeballed against 04_Synapse_REVIEW/_img/. No install required.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # examples/ -> import _common

import matplotlib
matplotlib.use("Agg")          # headless; save PNGs without a display
import matplotlib.pyplot as plt

from _common import experiments as ex
from _common.plotting import plot_synapse, rails


def main(out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures = []

    # Headline pulse up/down: Byte/Medium (plateau +/-0.5) and Float/Medium (+/-1).
    for model in ("byte", "float"):
        core, traces = ex.pulse_up_down(model=model, init="medium", n=5000, seed=1)
        ax = plot_synapse(traces, f"{model.upper()} | MEDIUM | FF-RH then FF-RL",
                          *rails(core))
        figures.append((f"pulse_{model}_medium.png", ax))

    # RS FF-XX (float-like saturation, model-aware pulse_width=1e-8).
    core, traces = ex.read_decay_vs_growth("FF", model="rs", init="low_noise", n=500, seed=4)
    figures.append(("rs_ff_xx.png",
                    plot_synapse(traces, "RS | LOW_NOISE | FF-XX", *rails(core))))

    # MSS RNG demo: oscillates about zero; final read digitized to a bit.
    core, ys_gab, bit = ex.mss_rng_demo(n=500, seed=6)
    ax = plot_synapse(ys_gab, f"MSS | MEDIUM_NOISE | (FF-RA)x500 (FF-RZ)x500  ->  bit={bit}",
                      *rails(core))
    figures.append(("mss_rng_demo.png", ax))

    for name, ax in figures:
        fig = ax[0].get_figure()
        fig.tight_layout()
        path = out / name
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "figures"
    main(out_dir)
