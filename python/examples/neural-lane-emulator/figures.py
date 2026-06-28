"""Chapter 4b figures — verification of the emulator on a single synapse.

Each figure overlays the emulator's MEASURED output on the THEORETICAL law it should obey,
so the figure is also a test: matching dots-on-a-line means the emulator reproduces the
Chapter 3b/4 physics. The emphasis is the read noise — the kT — measured against its law.

Default output_dir is the 4b article folder in the website repo, so re-running updates the
images the article embeds. Run:  python examples/neural-lane-emulator/figures.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # examples/ on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # python/ on path

import numpy as np                                   # noqa: E402
import matplotlib.pyplot as plt                      # noqa: E402
from math import erf                                 # noqa: E402

from ktram_neural_core import Core, READ_NOISE       # noqa: E402
from _common.experiments import (                    # noqa: E402
    single_synapse_core, execute_n, Z,
)

DEFAULT_OUT = ("/Users/alexnugent/Companies/Knowm/Code/GIT/knowm-ai-website/"
               "src/content/blog/the-neural-lane-emulator")

GRID = "0.92"


def _gmax():
    c = single_synapse_core("float", "medium", seed=1)
    return c.lane(0).spaces[0].a.device_at(0).GMAX


GMAX = _gmax()                       # float reference magnitude m_ref
V_REF = 0.05                         # reference / low read voltage (FFLV)


def _Phi(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _read_std(core, aat, n=6000, noise=0.0):
    """std of n sub-threshold (FFLV) reads at a pinned state — the empirical read noise."""
    lane = core.lane(0)
    return float(np.std([lane.evaluate(aat, "FFLV", noise=noise) for _ in range(n)]))


# ---------------------------------------------------------------- 1. the state matches
def fig_state(ax=None):
    """The deterministic state reproduces Chapter 3b: inertia (step ~ 1/m) and a weight that
    counts evidence toward 2p-1."""
    own = ax is None
    if own:
        _, ax = plt.subplots(1, 2, figsize=(11, 3.8))

    # inertia: same start weight, 10x magnitude apart, identical (FF, RH) drive
    for level, lab in ((0.05, "small m"), (0.5, "large m (10x)")):
        core = single_synapse_core("float", "medium", seed=1)
        core.set_start_y(0, Z, 0.3, level)
        ys, gas, gbs = execute_n(core, "FF", "RH", 300)
        w = [(a - b) / (a + b) for a, b in zip(gas, gbs)]
        ax[0].plot([0.3] + w, lw=1.4, label=lab)
    ax[0].axhline(1, color="0.7", ls="--", lw=0.8)
    ax[0].set_ylim(0.25, 1.03)
    ax[0].set_xlabel("feedback step (FF, RH)")
    ax[0].set_ylabel("weight  w")
    ax[0].set_title("magnitude is inertia: step ∝ 1/m")
    ax[0].legend(loc="lower right")

    # evidence: vote a Bernoulli(p) stream in, weight tracks the running frequency -> 2p-1
    import random
    for p in (0.75, 0.30):
        core = single_synapse_core("float", "low", seed=1)
        lane = core.lane(0)
        rng = random.Random(42)
        ones, w_dev = 0, []
        N = 600
        for t in range(1, N + 1):
            lane.evaluate(Z, "FFLV")
            x = 1 if rng.random() < p else 0
            lane.evaluate(Z, "FH" if x else "FL")
            ones += x
            ga, gb = core.read_gab(0, Z)
            w_dev.append((ga - gb) / (ga + gb))
        line, = ax[1].plot(w_dev, lw=1.3, label=f"p={p}")
        ax[1].axhline(2 * p - 1, color=line.get_color(), ls="--", lw=0.9)
    ax[1].set_ylim(-1.05, 1.05)
    ax[1].set_xlabel("votes cast (FFLV; FH/FL)")
    ax[1].set_ylabel("weight  w")
    ax[1].set_title("magnitude is evidence: w → 2p − 1")
    ax[1].legend(loc="center right")

    for a in ax:
        a.grid(True, color=GRID)
    return ax


# ---------------------------------------------------------------- 2c. pooling across pairs
def fig_pooling(ax=None):
    """Read N equally-confident pairs through the one 2-1 divider and the read noise falls as
    1/√N — the lane pooling precision (inverse-variance weighting) for free."""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7.0, 4.0))

    Ns = [1, 2, 4, 8, 16]
    meas = []
    for N in Ns:
        c = Core(1, 1, spaces_per_lane=N, num_lanes=1, model="float",
                 init="medium", seed=1, read_noise=READ_NOISE)
        aat = (0,) * N
        c.set_start_y(0, aat, 0.0, 0.5)               # N equally-confident pairs, w = 0, m = m_ref each
        meas.append(_read_std(c, aat))
    theory = meas[0] / np.sqrt(Ns)
    ax.plot(Ns, theory, color="0.6", lw=1.6, label="1/√N")
    ax.plot(Ns, meas, "o", color="tab:blue", ms=6, label="emulator")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels(Ns)
    ax.set_xlabel("pairs read together through the lane  N")
    ax.set_ylabel("read noise  σ")
    ax.set_title("read more pairs, read quieter:  σ ∝ 1/√N")
    ax.legend(loc="upper right")
    ax.grid(True, color=GRID)
    return ax


# ---------------------------------------------------------------- 2b. the law as a map
def fig_noise_map(ax=None):
    """The read-noise law as a map: σ over read voltage and magnitude, at three fixed weights.
    The bright thermal corner (low V, low m) sits in every panel; the rest darkens as the weight
    leaves zero — the flicker floor collapsing by (1 - w^2)."""
    from matplotlib.colors import LogNorm
    own = ax is None
    if own:
        _, ax = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True, constrained_layout=True)

    c = single_synapse_core("float", "medium", seed=1, read_noise=READ_NOISE)
    RN, NT, NF = c.read_noise, c.noise_thermal, c.noise_flicker     # gain, thermal & flicker weights
    MREF = c.read_noise_ref_m                                       # reference magnitude (m_ref)
    VFULL = c.forward_low_voltage                                   # full sub-threshold read voltage

    def sigma(w, m, v):                                            # the law, in code (room temp, default pw)
        s_thermal = RN * NT * (V_REF / v) * np.sqrt(MREF / m)      # ~ 1/V, ~ 1/sqrt(m), flat in w
        s_flicker = RN * NF * (1 - w ** 2) * np.sqrt(MREF / m)     # the floor: (1 - w^2), ~1/sqrt(m), flat in V
        return np.hypot(s_thermal, s_flicker)

    v_axis = np.geomspace(5e-4, VFULL, 120)       # read voltage: full read (right) down toward 0 (left)
    m_axis = np.geomspace(5e-3, 2 * GMAX, 120)    # magnitude: low .. 2*GMax (the ceiling)
    V, M = np.meshgrid(v_axis, m_axis)
    norm = LogNorm(vmin=0.003, vmax=0.3)
    for a, w in zip(ax, (0.0, 0.5, 0.9)):
        pcm = a.pcolormesh(v_axis, m_axis, sigma(w, M, V), shading="auto", cmap="magma", norm=norm)
        a.set_xscale("log")
        a.set_yscale("log")
        a.axvline(VFULL, color="w", ls=":", lw=1, alpha=0.7)      # the full read voltage
        a.set_xlabel("read voltage  V   (← lower V = louder)")
        a.set_title(f"weight  w = {w}")
    ax[0].set_ylabel("magnitude  m = Ga + Gb")
    plt.colorbar(pcm, ax=list(ax), label="read noise  σ", shrink=0.9)
    return ax


# ---------------------------------------------------------------- 3. the writable coin
def fig_coin(ax=None):
    """The kT-bit as a writable random source: a single sub-threshold read's sign is a biased
    bit, P(+) = Φ(w/σ). The read noise sets the band: with σ ≈ READ_NOISE the coin is soft only
    for weights within a few σ of zero, a near-certain bit past that — and the weight is the bias."""
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7.0, 4.0))

    core = single_synapse_core("float", "medium", seed=1, read_noise=READ_NOISE)
    lane = core.lane(0)
    core.set_start_y(0, Z, 0.0, 0.5)
    sigma = _read_std(core, Z)                        # σ of a normal FFLV read at m_ref

    # bias curve: measured P(+) on the Gaussian CDF Φ(w/σ)
    w0s = np.linspace(-3.5 * sigma, 3.5 * sigma, 29)
    emp = []
    for w0 in w0s:
        core.set_start_y(0, Z, float(w0), 0.5)
        emp.append(np.mean([1 if lane.evaluate(Z, "FFLV") > 0 else 0 for _ in range(3000)]))
    ax.plot(w0s, [_Phi(w / sigma) for w in w0s], color="0.6", lw=1.6, label="Φ(w/σ)")
    ax.plot(w0s, emp, "o", color="tab:blue", ms=5, label="emulator")
    ax.axhline(0.5, color="0.85", lw=0.8)
    ax.axvline(0.0, color="0.85", lw=0.8)
    ax.set_xlabel("stored weight  w")
    ax.set_ylabel("P( read > 0 )")
    ax.set_title(f"a writable coin: P(+) = Φ(w/σ),  σ ≈ {sigma:.3f}")
    ax.legend(loc="upper left")
    ax.grid(True, color=GRID)
    return ax


FIGURES = {
    "verify-state.png": fig_state,
    "noise-map.png": fig_noise_map,
    "pooling.png": fig_pooling,
    "verify-coin.png": fig_coin,
}


def main(out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        fn()
        if not plt.gcf().get_constrained_layout():     # figures that manage their own layout opt out
            plt.tight_layout()
        plt.savefig(out / name, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
