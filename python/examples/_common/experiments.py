"""Single-synapse experiment helpers (Milestone 1) — shared by the scripts and the notebook.

The smallest lane is one address space, one differential pair, one device per side. Selection
is the fixed AAT z = (0,). One instruction per evaluate() call; the old combined
execute(read, feedback) is just two sequential calls.
"""

import sys
import pathlib

# Make the package importable when run straight from the repo (no install needed):
# this file is python/examples/_common/experiments.py, so parents[2] is python/.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ktram_neural_core import Core   # noqa: E402

Z = (0,)


def execute_n(core, read, feedback_instr, n, lane_index=0):
    """Repeat `evaluate(read); evaluate(feedback_instr)` n times, recording y/Ga/Gb each step.

    `feedback_instr=None` means make only the read call (the old XX no-op). Returns (ys, gas, gbs).
    """
    lane = core.lane(lane_index)
    ys, gas, gbs = [], [], []
    for _ in range(n):
        y = lane.evaluate(Z, read)
        if feedback_instr is not None:
            lane.evaluate(Z, feedback_instr)
        ga, gb = core.read_gab(lane_index, Z)
        ys.append(y)
        gas.append(ga)
        gbs.append(gb)
    return ys, gas, gbs


def single_synapse_core(model, init, seed=None, read_noise=0.0, **kwargs):
    """A 1x1, one-space, one-lane Core — the single synapse.

    The emulator ships with read noise ON; these lesson/figure helpers default it
    OFF (read_noise=0) so the article's static figures and the recorded traces stay the
    deterministic state curves. Pass read_noise=READ_NOISE to exercise the noisy read.
    """
    return Core(1, 1, spaces_per_lane=1, num_lanes=1,
                model=model, init=init, seed=seed, read_noise=read_noise, **kwargs)


# ----- the experiment matrix (each returns a Core + recorded traces) -----

def pulse_up_down(model="byte", init="medium", n=5000, seed=1):
    """(FF, RH)xN then (FF, RL)xN — the headline plot. y up to a rail, then down."""
    core = single_synapse_core(model, init, seed)
    up = execute_n(core, "FF", "RH", n)
    dn = execute_n(core, "FF", "RL", n)
    return core, _concat(up, dn)


def feedback_combo(combo, model="float", init="medium", n=2000, seed=1):
    """One of the five FF-feedback combos: RH, RL, RU, RA, RZ."""
    core = single_synapse_core(model, init, seed)
    return core, execute_n(core, "FF", combo, n)


def read_decay_vs_growth(read, model="float", init="medium", n=5000, seed=2):
    """FF-XX (anti-Hebbian, y->0) or RF-XX (Hebbian, y->+/-1). Read-only, no feedback."""
    core = single_synapse_core(model, init, seed)
    return core, execute_n(core, read, None, n)


def low_voltage_read(instr="FFLV", model="rs", init="medium", n=500, seed=5):
    """FFLV/RFLV repeated — sub-threshold, state essentially unmoved."""
    core = single_synapse_core(model, init, seed)
    return core, execute_n(core, instr, None, n)


def inertia_pair(model="float", init="medium", y0=0.3,
                 levels=(0.05, 0.5), feedback_instr="RH", n=300, seed=1):
    """Matched weight, mismatched magnitude.

    Two synapse pairs are set to the SAME starting weight w = y0 but different magnitude
    m = Ga + Gb = 2 * level * GMax, using set_start_y(level=...). Both receive the identical
    (FF, feedback_instr) x n stream. Returns (cores, traces, levels) with traces[i] = (ys, gas, gbs)
    for levels[i]. The low-magnitude pair swings; the high-magnitude pair barely moves — the
    per-step change in w is proportional to 1/m.
    """
    cores, traces = [], []
    for level in levels:
        core = single_synapse_core(model, init, seed=seed)
        core.set_start_y(0, Z, y0, level)
        cores.append(core)
        traces.append(execute_n(core, "FF", feedback_instr, n))
    return cores, traces, levels


def mss_rng_demo(n=500, seed=6):
    """(FF, RA)xN then (FF, RZ)xN on MSS/MEDIUM_NOISE — oscillates about zero; the last
    read digitized (sign) is a random bit."""
    core = single_synapse_core("mss", "medium_noise", seed)
    a = execute_n(core, "FF", "RA", n)
    b = execute_n(core, "FF", "RZ", n)
    ys = _concat(a, b)
    bit = 1 if ys[0][-1] > 0 else 0
    return core, ys, bit


def _concat(*runs):
    ys, gas, gbs = [], [], []
    for y, ga, gb in runs:
        ys += y
        gas += ga
        gbs += gb
    return ys, gas, gbs
