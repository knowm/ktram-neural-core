"""Regenerate the figures for the "AHaH Attractors" web article (Chapter 5).

Run:  python examples/ahah-attractors/figures.py [output_dir]

Default output_dir is the article folder in the website repo, so re-running updates the
prose's images in one step. Everything here runs on the same emulator the companion
notebook uses; the data plots are reproduced exactly (fixed seeds). A 2-synapse AHaH node
is a lane with TWO address spaces (one differential pair per space): the AAT selects which
synapses are coupled to the shared 2-1 readout. z = (0, None) lights synapse 0 alone,
z = (None, 0) lights synapse 1 alone, z = (0, 0) couples both. Read FF then feed back RU
(the AHaH cycle) and the pair settles into an attractor.
"""

import sys
import pathlib
import collections

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # python/ on path

import numpy as np                                   # noqa: E402
import matplotlib                                    # noqa: E402
matplotlib.use("Agg")                                # headless
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import Rectangle, Patch       # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402

from ktram_neural_core import Core                   # noqa: E402

DEFAULT_OUT = ("/Users/alexnugent/Companies/Knowm/Code/GIT/knowm-ai-website/"
               "src/content/blog/ahah-attractors")

BLUE, ORANGE, GREEN, RED, GREY = "tab:blue", "tab:orange", "tab:green", "tab:red", "0.55"
GRID = "0.92"

# the three two-input AATs on a two-space lane (a 2-synapse AHaH node)
P0  = (0, None)      # synapse 0 alone        ("[0]")
P1  = (None, 0)      # synapse 1 alone        ("[1]")
P01 = (0, 0)         # both synapses coupled  ("[0,1]")


# --------------------------------------------------------------------------- experiment core

def node_core(spaces=2, model="float", init="medium", seed=0):
    """A single lane of `spaces` differential pairs — a `spaces`-synapse AHaH node.

    1x1 unit crossbars (one device per side per space). Read noise off: these are the
    deterministic attractor-dynamics figures.
    """
    return Core(1, 1, spaces_per_lane=spaces, num_lanes=1,
                model=model, init=init, seed=seed, read_noise=0.0)


def _w(core, space, n_spaces):
    """Weight of one synapse, read on its own."""
    aat = [None] * n_spaces
    aat[space] = 0
    ga, gb = core.read_gab(0, tuple(aat))
    return (ga - gb) / (ga + gb)


def run_node(seed, patterns, n=400, model="float", init="medium"):
    """Drive a 2-synapse node with the AHaH cycle (FF then RU) on random AATs.

    Each step picks one of `patterns` uniformly, reads it (FF), and feeds back unsupervised
    (RU). Returns the (w0, w1) path including the random start point.
    """
    core = node_core(2, model, init, seed)
    lane = core.lane(0)
    rng = np.random.default_rng(seed)
    w0 = [_w(core, 0, 2)]
    w1 = [_w(core, 1, 2)]
    for _ in range(n):
        p = patterns[rng.integers(len(patterns))]
        lane.evaluate(p, "FF")
        lane.evaluate(p, "RU")
        w0.append(_w(core, 0, 2))
        w1.append(_w(core, 1, 2))
    return np.array(w0), np.array(w1)


def classify(core):
    """The PLOS attractor-state label from the three sub-threshold reads (no disturbance).

    y10 = sign y([0]), y01 = sign y([1]), y11 = sign y([0,1]). The eight sign triples map to
    A/A', B/B', C/C', D/D' exactly as the AHaH Computing paper's table.
    """
    lane = core.lane(0)
    s = tuple(int(np.sign(lane.evaluate(p, "FFLV"))) for p in (P0, P1, P01))
    table = {
        (-1, -1, -1): "A",  (1, 1, 1): "A'",
        (-1, 1, -1): "B",   (1, -1, 1): "B'",
        (-1, 1, 1): "C",    (1, -1, -1): "C'",
        (1, 1, -1): "D",    (-1, -1, 1): "D'",
    }
    return table.get(s, "?")


def end_state(seed, patterns, n=400, model="float", init="medium"):
    """Run one node to convergence; return (w0, w1, state-label)."""
    core = node_core(2, model, init, seed)
    lane = core.lane(0)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        p = patterns[rng.integers(len(patterns))]
        lane.evaluate(p, "FF")
        lane.evaluate(p, "RU")
    return _w(core, 0, 2), _w(core, 1, 2), classify(core)


def quadrant_color(w0, w1):
    return {(1, 1): ORANGE, (-1, -1): ORANGE, (1, -1): BLUE, (-1, 1): BLUE}[
        (int(np.sign(w0)), int(np.sign(w1)))
    ]


# --------------------------------------------------------------------- 1. two-pattern basin

def fig_two_pattern(ax=None):
    """Orthogonal AATs ([0] and [1]) -> four clean attractors, one per sign quadrant.

    Each synapse is read and reinforced on its own, so they never interact: two synapses x
    two signs = 2^2 = 4 attractor basins.
    """
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(6.0, 6.0))
    trials = 140
    for s in range(trials):
        w0, w1 = run_node(s, [P0, P1], n=300, init="medium")
        ax.plot(w0, w1, color="0.8", lw=0.5, zorder=1)
        ax.plot(w0[0], w1[0], ".", color="0.6", ms=3, zorder=2)
        ax.plot(w0[-1], w1[-1], "o", color=quadrant_color(w0[-1], w1[-1]),
                ms=5, zorder=3)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.75, 0.75)
    ax.set_aspect("equal")
    ax.set_xlabel("w0")
    ax.set_ylabel("w1")
    ax.set_title("Two orthogonal AATs → four attractors")
    ax.grid(True, color=GRID)
    return ax


# ----------------------------------------------------------------- 2. the same on 4 devices

def fig_devices(ax=None):
    """Four device models, same two-pattern node — same four attractors, different texture.

    Float is infinitely fine, Byte an 8-bit staircase, MSS and RS the real stochastic
    physics. The four basins are in the same place on every model; the device sets resolution
    and scatter, not the computation.
    """
    own = ax is None
    if own:
        _, ax = plt.subplots(2, 2, figsize=(9.0, 9.0))
    panels = (("float", "Float — infinitely fine"),
              ("byte", "Byte — 8-bit staircase"),
              ("mss", "MSS — stochastic (real device)"),
              ("rs", "RS — stochastic (real device)"))
    for a, (model, title) in zip(ax.flat, panels):
        for s in range(90):
            w0, w1, _ = end_state(s, [P0, P1], n=400, model=model, init="medium")
            a.plot(w0, w1, "o", color=quadrant_color(w0, w1), ms=4, alpha=0.8)
        a.axhline(0, color="0.6", lw=0.8)
        a.axvline(0, color="0.6", lw=0.8)
        a.set_xlim(-1.05, 1.05)
        a.set_ylim(-1.05, 1.05)
        a.set_aspect("equal")
        a.set_title(title, fontsize=10)
        a.set_xlabel("w0")
        a.set_ylabel("w1")
        a.grid(True, color=GRID)
    return ax


# ------------------------------------------------------- 2b. add the overlap: the null state

def fig_three_pattern(ax=None):
    """Add the overlapping AAT [0,1] and the synapses interact — the null state appears.

    Left: the (w0, w1) paths collapse mostly onto the +/+ and -/- diagonal (the null state),
    where both synapses agree. Right: the occupancy over the eight states — the two null states
    (A, A') swallow most nodes, B/C survive at the edges, and D never forms.
    """
    own = ax is None
    if own:
        _, ax = plt.subplots(1, 2, figsize=(12, 5.6))

    # left: collapsing paths
    counts = collections.Counter()
    for s in range(160):
        core = node_core(2, "float", "medium", s)
        lane = core.lane(0)
        rng = np.random.default_rng(s)
        w0 = [_w(core, 0, 2)]
        w1 = [_w(core, 1, 2)]
        for _ in range(400):
            p = [P0, P1, P01][rng.integers(3)]
            lane.evaluate(p, "FF")
            lane.evaluate(p, "RU")
            w0.append(_w(core, 0, 2))
            w1.append(_w(core, 1, 2))
        w0, w1 = np.array(w0), np.array(w1)
        null = np.sign(w0[-1]) == np.sign(w1[-1])
        col = RED if null else GREEN
        ax[0].plot(w0, w1, color=col, lw=0.5, alpha=0.5, zorder=1)
        ax[0].plot(w0[-1], w1[-1], "o", color=col, ms=4, zorder=3)
        counts[classify(core)] += 1
    ax[0].plot([-1, 1], [-1, 1], color="0.4", lw=1.0, ls="--", zorder=2)  # the null diagonal
    ax[0].axhline(0, color="0.7", lw=0.8)
    ax[0].axvline(0, color="0.7", lw=0.8)
    ax[0].set_xlim(-0.75, 0.75)
    ax[0].set_ylim(-0.75, 0.75)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("w0")
    ax[0].set_ylabel("w1")
    ax[0].set_title("Overlapping AATs → the null state (red) devours the node")
    ax[0].grid(True, color=GRID)

    # right: state occupancy
    order = ["A", "A'", "B", "B'", "C", "C'", "D", "D'"]
    vals = [counts.get(k, 0) for k in order]
    cols = [RED if k in ("A", "A'") else "0.6" for k in order]
    ax[1].bar(order, vals, color=cols)
    ax[1].set_ylabel("nodes ending in state")
    ax[1].set_title("The null states A, A' (red) dominate; D never forms")
    ax[1].grid(True, axis="y", color=GRID)
    return ax


# --------------------------------------------------------------- the behavioral fingerprint

def H(y):
    """One output bit: + (>=0) -> 1, - -> 0. The atom of the attractor fingerprint."""
    return 1 if y >= 0 else 0


# the four 2-bit inputs, in the order that fixes the logic-gate numbering (00, 01, 10, 11)
INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]

# the 16 two-input logic functions, keyed by gate number = (o00 o01 o10 o11) read as binary
GATE_NAMES = {
    0: "FALSE", 1: "AND", 2: "A·¬B", 3: "A", 4: "¬A·B", 5: "B", 6: "XOR", 7: "OR",
    8: "NOR", 9: "XNOR", 10: "¬B", 11: "A∨¬B", 12: "¬A", 13: "¬A∨B", 14: "NAND", 15: "TRUE",
}
NONLINEAR = {6, 9}        # XOR / XNOR — not linearly separable, unreachable by a shared lane
CONSTANT = {0, 15}        # FALSE / TRUE — the "do-nothing" constant gates (the fat basins)


def _id_overlap(seed, n=600):
    """Overlapping 2-synapse encoding: 3 live patterns ([0], [1], [0,1]); (0,0) is not an input.
    Fingerprint is a 3-bit number 0..7 over those three answers."""
    core = node_core(2, "float", "medium", seed)
    lane = core.lane(0)
    rng = np.random.default_rng(seed)
    pats = [P0, P1, P01]
    for _ in range(n):
        p = pats[rng.integers(3)]
        lane.evaluate(p, "FF")
        lane.evaluate(p, "RU")
    b = [H(lane.evaluate(p, "FFLV")) for p in pats]
    return (b[0] << 2) | (b[1] << 1) | b[2]


def _id_twohot(seed, n=600):
    """Two-hot (dual-rail): 2 spaces x 2 channels, AAT = (x0, x1). All four inputs are real, so
    the fingerprint is the full 4-bit gate number 0..15 over (00, 01, 10, 11)."""
    core = Core(1, 2, spaces_per_lane=2, num_lanes=1, model="float",
                init="medium", seed=seed, read_noise=0.0)
    lane = core.lane(0)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x = INPUTS[rng.integers(4)]
        lane.evaluate(x, "FF")
        lane.evaluate(x, "RU")
    o = [H(lane.evaluate(x, "FFLV")) for x in INPUTS]
    return (o[0] << 3) | (o[1] << 2) | (o[2] << 1) | o[3]


def _id_onehot(seed, n=600):
    """One-hot: one space of 4, each whole input owns a synapse. Outputs are independent, so all
    16 gate numbers — XOR/XNOR included — are reachable. A four-entry lookup table."""
    core = Core(1, 4, spaces_per_lane=1, num_lanes=1, model="float",
                init="medium", seed=seed, read_noise=0.0)
    lane = core.lane(0)
    rng = np.random.default_rng(seed)
    idx = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    for _ in range(n):
        x = INPUTS[rng.integers(4)]
        lane.evaluate((idx[x],), "FF")
        lane.evaluate((idx[x],), "RU")
    o = [H(lane.evaluate((idx[x],), "FFLV")) for x in INPUTS]
    return (o[0] << 3) | (o[1] << 2) | (o[2] << 1) | o[3]


def _occupancy(id_fn, n_states, n_trials=200):
    """Fraction of random seeds landing in each attractor index — the basin-size histogram."""
    counts = np.zeros(n_states)
    for s in range(n_trials):
        counts[id_fn(s)] += 1
    return counts / n_trials


# ------------------------------------------------------- 3. the fingerprint, vs the picture

def fig_fingerprint(ax=None):
    """Bridge the weight-space scatter to the behavioral index: the four corners ARE states.

    For the orthogonal node the only patterns are [0] and [1], so the fingerprint is a 2-bit
    number — MSB the sign of y([1]), LSB the sign of y([0]). The four sign quadrants become the
    four integers 0..3, which lets us trust the index where there is no scatter left to draw.
    """
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(6.0, 6.0))
    for s in range(140):
        w0, w1, _ = end_state(s, [P0, P1], n=400, init="medium")
        ax.plot(w0, w1, "o", color=quadrant_color(w0, w1), ms=4, alpha=0.7, zorder=2)
    # label each quadrant in its empty OUTER corner, well clear of the cluster (~±0.37)
    corners = [((-0.66, -0.66), "(−,−)", 0),
               ((0.66, -0.66), "(+,−)", 1),
               ((-0.66, 0.66), "(−,+)", 2),
               ((0.66, 0.66), "(+,+)", 3)]
    for (x, y), sgn, idx in corners:
        ax.text(x, y, f"{sgn}\nstate {idx}", ha="center", va="center", fontsize=11,
                fontweight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-0.85, 0.85)
    ax.set_aspect("equal")
    ax.set_xlabel("w0")
    ax.set_ylabel("w1")
    ax.set_title("The four corners are states 0–3")
    ax.grid(True, color=GRID)
    return ax


# ----------------------------------------------------- 4. counting + basins, three encodings

def fig_landscapes(ax=None):
    """One histogram per encoding: bar count = number of attractors, bar height = basin size.

    Dimension-free — it reads the same for 2 synapses or 200. Constant ("do-nothing") gates in
    red, the nonlinear XOR/XNOR slots hatched, everything else grey. The landscape is reshaped
    purely by how much the encoding shares synapses; the learning rule never changed.
    """
    own = ax is None
    if own:
        _, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # --- overlapping: 8 states, its own all-same-answer constants are 0 (000) and 7 (111) ---
    occ = _occupancy(_id_overlap, 8)
    cols = [RED if i in (0, 7) else "0.6" for i in range(8)]
    ax[0].bar(range(8), occ, color=cols)
    ax[0].set_xticks(range(8))
    ax[0].set_xlim(-0.6, 7.6)
    ax[0].set_title("overlapping — 2 synapses\n(3 answers → 8 possible states)", fontsize=10)
    ax[0].set_xlabel("attractor index")
    ax[0].set_ylabel("fraction of nodes (basin size)")

    # --- two-hot and one-hot: full 16 gate numbers ---
    for col, (id_fn, title) in zip(
        ax[1:],
        [(_id_twohot, "two-hot — 2×2\n(4 answers → 16 possible)"),
         (_id_onehot, "one-hot — 4 synapses\n(lookup table → all 16)")],
    ):
        occ = _occupancy(id_fn, 16)
        cols = []
        for i in range(16):
            if i in CONSTANT:
                cols.append(RED)
            elif i in NONLINEAR:
                cols.append(ORANGE)
            else:
                cols.append("0.6")
        bars = col.bar(range(16), occ, color=cols)
        # hatch the nonlinear slots so an empty XOR bar still reads as "this is XOR/XNOR"
        for i in NONLINEAR:
            bars[i].set_hatch("///")
            bars[i].set_edgecolor(ORANGE)
        col.set_xticks(range(16))
        col.set_xlim(-0.6, 15.6)
        col.set_title(title, fontsize=10)
        col.set_xlabel("attractor index")

    for a in ax:
        a.grid(True, axis="y", color=GRID)
    # shared legend (kept in attractor terms — the logic reading comes later)
    ax[1].text(0.5, 0.92, "red = all-same-answer states   orange/hatched = "
               "the two a shared encoding misses",
               transform=ax[1].transAxes, ha="center", fontsize=8.5, color="0.3")
    return ax


# ------------------------------------------------------ 5. the reveal: every state is a gate

def fig_logic_gates(ax=None):
    """The fingerprint IS a truth table, so every attractor is a 2-input logic gate.

    Left: the 16 gates as a 4x4 map, each cell shaded by how often the two-hot node lands in it
    (the same basin sizes as the middle landscape panel). Constants are the darkest cells; the
    XOR/XNOR cells are hatched and labelled unreachable. Right: why — a bias-free node is one
    straight line through the origin, which can isolate any gate except XOR.
    """
    own = ax is None
    if own:
        _, ax = plt.subplots(1, 2, figsize=(13, 5.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    occ = _occupancy(_id_twohot, 16)
    occ_n = occ / occ.max() if occ.max() > 0 else occ

    a = ax[0]
    a.set_xlim(0, 4)
    a.set_ylim(0, 4)
    a.set_aspect("equal")
    a.axis("off")
    a.set_title("Sixteen attractors, sixteen logic gates "
                "(shaded by basin size, two-hot)", fontsize=11)
    for g in range(16):
        r, c = divmod(g, 4)
        y = 3 - r            # top row = gates 0..3
        x = c
        shade = 0.93 - 0.65 * occ_n[g]           # darker = bigger basin
        face = (shade, shade, shade)
        rect = Rectangle((x + 0.05, y + 0.05), 0.9, 0.9, facecolor=face, edgecolor="k")
        if g in NONLINEAR:
            rect.set_hatch("///")
            rect.set_edgecolor(ORANGE)
        a.add_patch(rect)
        bits = f"{g >> 3 & 1}{g >> 2 & 1}{g >> 1 & 1}{g & 1}"
        txtcol = "white" if occ_n[g] > 0.6 else "k"
        a.text(x + 0.5, y + 0.66, f"#{g}", ha="center", fontsize=10.5,
               fontweight="bold", color=txtcol)
        a.text(x + 0.5, y + 0.45, GATE_NAMES[g], ha="center", fontsize=9.5, color=txtcol)
        a.text(x + 0.5, y + 0.22, bits, ha="center", fontsize=8, family="monospace",
               color=ORANGE if g in NONLINEAR else txtcol)

    # --- right: linear separability of the four inputs ---
    b = ax[1]
    b.set_aspect("equal")
    b.set_xlim(-0.6, 1.6)
    b.set_ylim(-0.6, 1.9)
    # AND is separable: shade the (1,1) corner off from the rest with one line
    for (x, y) in INPUTS:
        xor = x ^ y
        b.plot(x, y, "o", ms=16, zorder=3,
               color=GREEN if xor else BLUE,
               markeredgecolor="k", markeredgewidth=1.0)
        b.annotate(f"({x},{y})", (x, y), textcoords="offset points",
                   xytext=(10, 8), fontsize=9)
    b.text(0.5, 1.55, "XOR: opposite corners share a color", ha="center", fontsize=9.5)
    b.text(0.5, -0.45, "no straight line separates the greens from the blues",
           ha="center", fontsize=9, color=GREEN)
    b.set_xlabel("input bit 0")
    b.set_ylabel("input bit 1")
    b.set_title("One node = one straight line", fontsize=11)
    b.grid(True, color=GRID)
    return ax


# ------------------------------------------------ 6. the bias and the split write (null fix)

# A bias is one or more always-on synapses appended after the two inputs. The "split write"
# reads the whole node once (FF, bias included), then drives the inputs with RU (Hebbian) and
# the bias with RA (anti-Hebbian) off that same H(y). The attractor STATE is the output
# fingerprint — sign of y on [0], [1], [0,1] with the bias in the read — mapped to A..D'. You
# cannot read it from the input weight signs once a bias offsets the boundary off the origin.
_BIAS_STATE = {0: "A", 1: "D'", 2: "B", 3: "C", 4: "C'", 5: "B'", 6: "D", 7: "A'"}


def _split_write_run(seed, n_bias, init, n_steps, track=False):
    """One node: 2 inputs + n_bias always-on bias synapses, driven by the split write.

    Returns (state, path) where state is the 3-bit output fingerprint and path is the weight
    trajectory array (n_sp, n_steps+1) when track else None.
    """
    n_sp = 2 + n_bias
    core = Core(1, 1, spaces_per_lane=n_sp, num_lanes=1, model="float",
                init=init, seed=seed, read_noise=0.0)
    lane = core.lane(0)
    rng = np.random.default_rng(seed)
    keys = ["0", "1", "01"]

    def full(k):                       # FF read: active inputs + every bias
        a = [None] * n_sp
        if k != "1": a[0] = 0
        if k != "0": a[1] = 0
        for b in range(2, n_sp): a[b] = 0
        return tuple(a)

    def inputs(k):                     # RU: active inputs only
        a = [None] * n_sp
        if k != "1": a[0] = 0
        if k != "0": a[1] = 0
        return tuple(a)

    aat_bias = tuple(0 if i >= 2 else None for i in range(n_sp))   # RA: bias synapses only

    def wt(sp):
        a = [None] * n_sp; a[sp] = 0
        ga, gb = core.read_gab(0, tuple(a))
        return (ga - gb) / (ga + gb)

    path = [[wt(j)] for j in range(n_sp)] if track else None
    for _ in range(n_steps):
        k = keys[rng.integers(3)]
        lane.evaluate(full(k), "FF")        # read the whole node (bias included)
        lane.evaluate(inputs(k), "RU")      # Hebbian on the inputs
        lane.evaluate(aat_bias, "RA")       # anti-Hebbian on the bias
        if track:
            for j in range(n_sp): path[j].append(wt(j))
    bits = [1 if lane.evaluate(full(k), "FFLV") >= 0 else 0 for k in keys]
    state = (bits[0] << 2) | (bits[1] << 1) | bits[2]
    return state, (np.array(path) if track else None)


def fig_split_write(n_bias, init, n_nodes=500, n_steps=2000, sample=120, ax=None):
    """Left: weights over time (inputs + bias). Right: the output-state histogram (A..D')."""
    own = ax is None
    if own:
        _, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))
    states, paths = [], []
    for s in range(n_nodes):
        st, pth = _split_write_run(s, n_bias, init, n_steps, track=(s < sample))
        states.append(st)
        if pth is not None: paths.append(pth)
    states = np.array(states)
    counts = collections.Counter(states.tolist())
    bias_cols = [GREEN, "tab:purple"]

    # left: weights over time
    for p in paths[::6]:
        t = np.arange(p.shape[1])
        ax[0].plot(t, p[0], color=BLUE, lw=0.5, alpha=0.5)
        ax[0].plot(t, p[1], color=ORANGE, lw=0.5, alpha=0.5)
        for b in range(n_bias):
            ax[0].plot(t, p[2 + b], color=bias_cols[b % 2], lw=0.6, alpha=0.4)
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].set_ylim(-1.05, 1.05)
    ax[0].set_xlabel("step"); ax[0].set_ylabel("weight")
    ax[0].set_title("Weights over time")
    handles = [Line2D([], [], color=BLUE, label="$w_0$ (input)"),
               Line2D([], [], color=ORANGE, label="$w_1$ (input)")]
    for b in range(n_bias):
        handles.append(Line2D([], [], color=bias_cols[b % 2],
                              label="bias" if n_bias == 1 else f"bias {b + 1}"))
    ax[0].legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    ax[0].grid(True, color=GRID)

    # right: output-state histogram
    vals = [counts.get(i, 0) for i in range(8)]
    col = lambda i: RED if i in (0, 7) else (GREEN if i in (1, 6) else "0.6")
    ax[1].bar(range(8), vals, color=[col(i) for i in range(8)])
    ax[1].set_xticks(range(8))
    ax[1].set_xticklabels([f"{_BIAS_STATE[i]}\n{i:03b}" for i in range(8)], fontsize=9)
    ax[1].set_xlabel("state   (output sign on  [0], [1], [0,1])")
    ax[1].set_ylabel("nodes")
    ax[1].set_title("Where the nodes land")
    ax[1].legend(handles=[Patch(color=RED, label="null (A/A′)"),
                          Patch(color=GREEN, label="D / D′"),
                          Patch(color="0.6", label="B / B′ / C / C′")],
                 loc="upper right", fontsize=8, framealpha=0.9)
    ax[1].grid(True, axis="y", color=GRID)

    null = int(np.isin(states, [0, 7]).sum())
    print(f"  n_bias={n_bias} init={init}: null={null}/{n_nodes}={null/n_nodes:.0%}  " +
          "  ".join(f"{_BIAS_STATE[i]}={counts.get(i, 0)}" for i in range(8)))
    return ax


def fig_one_bias(ax=None):
    """One low-magnitude bias: the null state is gone, but D/D′ are still out of reach — a
    single weak bias can kill the collapse without carving the off-origin boundary D needs."""
    return fig_split_write(1, "low", ax=ax)


def fig_two_bias(ax=None):
    """Two low-magnitude biases: D and D′ start to appear — more bias authority reaches the
    off-origin states, but at low magnitude their basins are still small."""
    return fig_split_write(2, "low", ax=ax)


def fig_bias_high_init(ax=None):
    """Two biases at higher init magnitude (mean 0.5): the D/D′ basins fill out and the whole
    eight-state set balances — the bias's reach scales with its magnitude."""
    return fig_split_write(2, "medium", ax=ax)


# ------------------------------------------------------------------------------------ main

FIGURES = {
    "two-pattern.png": fig_two_pattern,
    "devices.png": fig_devices,
    "three-pattern.png": fig_three_pattern,
    "fingerprint.png": fig_fingerprint,
    "landscapes.png": fig_landscapes,
    "logic-gates.png": fig_logic_gates,
    "one-bias.png": fig_one_bias,
    "two-bias.png": fig_two_bias,
    "bias-high-init.png": fig_bias_high_init,
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
