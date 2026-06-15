"""Behavioral checks that the single-synapse experiments reproduce the lesson shapes.

These assert qualitative outcomes (saturation values, Hebbian vs anti-Hebbian direction),
not bit-exact traces. They run with read_noise=0 so the threshold assertions key off the
clean attractor value rather than a single thermal read sample; the thermal read noise is a
separate effect (the returned y is a draw whose spread shrinks as 1/m). Final validation
against the lesson figures is Alex's.
"""

import pytest

from ktram_neural_core import Core

Z = (0,)


def pulse(core, read, feedback, n):
    lane = core.lane(0)
    for _ in range(n):
        lane.evaluate(Z, read)
        lane.evaluate(Z, feedback)
    return lane.evaluate(Z, read)


def test_float_medium_pulse_reaches_plus_minus_one():
    core = Core(1, 1, model="float", init="medium", seed=1, read_noise=0)
    assert pulse(core, "FF", "RH", 5000) > 0.95
    assert pulse(core, "FF", "RL", 5000) < -0.95


def test_byte_medium_plateaus_at_half():
    # Byte_Medium.png plateaus at +/-0.5 — the half-up rounding ceiling (dVa=1-y rounds to
    # 0 once y>0.5, freezing Ga).
    core = Core(1, 1, model="byte", init="medium", seed=1, read_noise=0)
    assert 0.49 < pulse(core, "FF", "RH", 5000) < 0.51


def test_ff_xx_float_is_anti_hebbian():
    # FF repeated drives both devices up -> y -> 0.
    core = Core(1, 1, model="float", init="low_noise", seed=2, read_noise=0)
    lane = core.lane(0)
    for _ in range(5000):
        lane.evaluate(Z, "FF")
    assert abs(lane.evaluate(Z, "FF")) < 0.05


def test_rf_xx_float_is_hebbian():
    # RF repeated is a positive-feedback split -> y -> +/-1.
    core = Core(1, 1, model="float", init="medium", seed=3, read_noise=0)
    lane = core.lane(0)
    for _ in range(5000):
        lane.evaluate(Z, "RF")
    assert abs(lane.evaluate(Z, "RF")) > 0.95


@pytest.mark.parametrize("model,instr", [("float", "FFLV"), ("byte", "RFLV")])
def test_low_voltage_reads_do_not_move_state(model, instr):
    core = Core(1, 1, model=model, init="medium", seed=7)
    lane = core.lane(0)
    before = core.read_gab(0, Z)
    for _ in range(500):
        lane.evaluate(Z, instr)
    assert core.read_gab(0, Z) == before


def test_rs_default_pulse_width_is_model_aware():
    assert Core(1, 1, model="rs").pulse_width == 1e-8
    assert Core(1, 1, model="float").pulse_width == 1e-6


def test_mss_default_drive_is_model_aware():
    core = Core(1, 1, model="mss")
    assert core.forward_voltage == 0.25
    assert core.reverse_voltage == -0.25


def test_seed_makes_mss_run_bit_reproducible():
    def run(seed):
        core = Core(1, 1, model="mss", init="medium_noise", seed=seed)
        lane = core.lane(0)
        ys = []
        for _ in range(200):
            lane.evaluate(Z, "FF")
            lane.evaluate(Z, "RA")
            ys.append(core.lane(0).y)
        return ys

    assert run(42) == run(42)        # same seed -> identical stochastic trace
    assert run(42) != run(43)
