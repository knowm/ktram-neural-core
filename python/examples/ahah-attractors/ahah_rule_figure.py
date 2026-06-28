"""Regenerate the dW-vs-Y "AHaH rule" figure for Chapter 5b (The Push and the Pull).

Run:  python examples/ahah-attractors/ahah_rule_figure.py [output_dir]

Reproduces, on the *new* emulator, the classic AHaH-rule scatter: the per-cycle weight
update Δw of a synapse plotted against the node output y that drove it. A 2-synapse AHaH
node is run through the unsupervised cycle (FF then RU) on the three overlapping patterns;
at each step we read the weights, do the FF read (which returns y), do the RU feedback, then
read the weights again — Δw is the net change over the cycle, exactly the old getdW(w[i] -
w[i-1]) method. One panel per device model (float, byte, rs, mss). No two devices draw the
same curve, but every one bends the same way: large updates near y = 0, shrinking and then
turning anti-Hebbian (negative) as |y| grows.

Reuses node_core / _w / the three AATs from the companion figures.py.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # python/ on path

import numpy as np                                   # noqa: E402
import matplotlib                                    # noqa: E402
matplotlib.use("Agg")                                # headless
import matplotlib.pyplot as plt                      # noqa: E402

# pull the shared experiment helpers from the sibling figures module
import importlib.util                                # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "ahah_figs", str(pathlib.Path(__file__).resolve().parent / "figures.py"))
_figs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figs)
node_core, _w = _figs.node_core, _figs._w
P0, P1, P01 = _figs.P0, _figs.P1, _figs.P01
BLUE, ORANGE, GRID = _figs.BLUE, _figs.ORANGE, _figs.GRID

OUT = ("/Users/alexnugent/Companies/Knowm/Code/GIT/knowm-ai-website/"
       "src/content/blog/the-ahah-rule")

PANELS = (("float", "Float — infinitely fine"),
          ("byte",  "Byte — 8-bit staircase"),
          ("rs",    "RS — stochastic (real device)"),
          ("mss",   "MSS — stochastic (real device)"))


def collect_natural(model, seeds=300, rec=130):
    """Δw vs y from real FF-RU trajectories, sampled in the transient (the Hebbian rise).

    The attractor app's own measurement: per cycle, record y (the FF read) and each synapse's
    net Δw over that cycle (w[i] - w[i-1]). A node sprints from y ≈ 0 out to its attractor and
    then sits there, so most of a long run is dead weight piled at one y. We take the first
    `rec` cycles of many random nodes — the transient, where y sweeps from 0 toward the
    equilibrium and the magnitude grows on its own. This covers the central, Hebbian part of
    the rule but never overshoots, so it cannot show the anti-Hebbian pull-back on its own.
    """
    pats = [P0, P1, P01]
    ys, dws = [], []
    for s in range(seeds):
        core = node_core(2, model, "medium", s)
        lane = core.lane(0)
        rng = np.random.default_rng(s)
        for _ in range(rec):
            p = pats[rng.integers(3)]
            active = [sp for sp, a in zip((0, 1), p) if a is not None]
            before = {sp: _w(core, sp, 2) for sp in active}
            y = lane.evaluate(p, "FF")        # read: returns y, applies the anti-Hebbian nudge
            lane.evaluate(p, "RU")            # unsupervised Hebbian feedback off H(y)
            for sp in active:
                ys.append(y); dws.append(_w(core, sp, 2) - before[sp])
    return np.asarray(ys), np.asarray(dws)


def collect_forced(model, samples=9000, level=0.5):
    """Δw vs y for nodes started *past* equilibrium, to reach the anti-Hebbian end.

    The natural transient never overshoots, so to see the rule turn anti-Hebbian we have to put
    the node there: place both synapses at a spread of states at high magnitude (set_start_y at
    level = half-full conductance), do one real FF-RU cycle, and record Δw against y. Out at
    large |y| the node is over-committed and the update reverses — the pull-back the natural
    runs can't reach. (Forcing the magnitude this way overstates the update size for the
    stochastic models, so we keep these only to map the *sign* and reach of the far lobe.)
    """
    rng = np.random.default_rng((hash(model) ^ 0x9E3779B9) & 0xFFFF)
    pats = [P0, P1, P01]
    ys, dws = [], []
    for i in range(samples):
        core = node_core(2, model, "medium", i)
        lane = core.lane(0)
        core.set_start_y(0, (0, None), float(rng.uniform(-0.98, 0.98)), level=level)
        core.set_start_y(0, (None, 0), float(rng.uniform(-0.98, 0.98)), level=level)
        p = pats[rng.integers(3)]
        active = [sp for sp, a in zip((0, 1), p) if a is not None]
        before = {sp: _w(core, sp, 2) for sp in active}
        y = lane.evaluate(p, "FF")
        lane.evaluate(p, "RU")
        for sp in active:
            ys.append(y); dws.append(_w(core, sp, 2) - before[sp])
    return np.asarray(ys), np.asarray(dws)


def collect(model):
    """Full curve: the natural Hebbian transient plus the forced anti-Hebbian far lobe."""
    yn, dwn = collect_natural(model)
    yf, dwf = collect_forced(model)
    return np.concatenate([yn, yf]), np.concatenate([dwn, dwf])


def fig_ahah_rule():
    fig, ax = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for a, (model, title) in zip(ax.flat, PANELS):
        y, dw = collect(model)
        # blue where the update reinforces the lean (sign dw == sign y), grey where it pulls back
        same = np.sign(dw) == np.sign(y)
        a.scatter(y[~same], dw[~same], s=3, alpha=0.18, color="0.6", edgecolors="none")
        a.scatter(y[same], dw[same], s=3, alpha=0.18, color=BLUE, edgecolors="none")
        a.axhline(0, color="0.55", lw=0.8)
        a.axvline(0, color="0.55", lw=0.8)
        lim = np.percentile(np.abs(dw), 99.0) if len(dw) else 1.0
        a.set_ylim(-1.2 * lim, 1.2 * lim)
        a.set_xlim(-1.05, 1.05)
        a.set_title(title, fontsize=10)
        a.set_xlabel("node output  $y$")
        a.set_ylabel("weight update  $\\Delta w$")
        a.grid(True, color=GRID)
    fig.suptitle("The AHaH rule, measured on the emulator: biggest updates where the node is "
                 "unsure ($y \\approx 0$), reversing into\nanti-Hebbian pull-back (grey) past "
                 "the balance point on the physical models (unsupervised FF–RU)", fontsize=10.5)
    return fig


def main(out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig = fig_ahah_rule()
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out / "emulator-rules.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OUT)
