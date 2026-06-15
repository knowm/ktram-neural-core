"""Readout, update-voltage mapping, the instruction->Vy table, and AAT selection."""

import pytest

from ktram_neural_core import Core
from ktram_neural_core.topology import TwoOne

Z = (0,)


def test_readout_formula():
    top, bottom = TwoOne.readout([(1e-3, 1e-4), (2e-4, 2e-4)])
    assert top == pytest.approx(1e-3 - 1e-4 + 0.0)
    assert bottom == pytest.approx(1e-3 + 1e-4 + 4e-4)


def test_update_voltages():
    assert TwoOne.update_voltages(1.0, 0.5) == (0.5, 1.5)
    assert TwoOne.update_voltages(-1.0, -1.0) == (0.0, -2.0)   # RH case


def test_byte_handcheck_ff_then_rh():
    # Byte/medium_noiseless: 128/128. FF reads y=0 (Vy=0) and adapts both +1 -> 129/129.
    # RH: Vy=-1, dVa=0, dVb=-2 -> Ga stays 129, Gb 129-2=127.
    core = Core(1, 1, model="byte", init="medium_noiseless")
    lane = core.lane(0)
    lane.evaluate(Z, "FF")
    assert core.read_gab(0, Z) == (129.0, 129.0)
    lane.evaluate(Z, "RH")
    assert core.read_gab(0, Z) == (129.0, 127.0)


def test_read_y_sign_same_forward_or_reverse():
    # y = top/bottom regardless of read direction. read_noise=0: this checks the clean
    # readout, not the thermal sample (which would randomize the sign of a barely-leaning pair).
    core = Core(1, 1, model="float", init="medium_noiseless", read_noise=0)
    lane = core.lane(0)
    lane.evaluate(Z, "RH")                # nudge Ga>Gb a touch via reverse high
    yf = lane.evaluate(Z, "FF")
    yr = lane.evaluate(Z, "RF")
    assert (yf > 0) == (yr > 0)


def test_feedback_does_not_change_retained_y():
    core = Core(1, 1, model="float", init="low_noiseless")
    lane = core.lane(0)
    lane.evaluate(Z, "RH")               # drive Ga>Gb
    y_read = lane.evaluate(Z, "FF")
    y_after_feedback = lane.evaluate(Z, "FZ")   # feedback: y unchanged
    assert y_after_feedback == y_read


def test_fz_keeps_devices_balanced_increment():
    # FZ sets Vy=0 -> both devices see +Vf -> equal increment, y unchanged sign-wise.
    core = Core(1, 1, model="float", init="medium_noiseless")
    lane = core.lane(0)
    ga0, gb0 = core.read_gab(0, Z)
    lane.evaluate(Z, "FZ")
    ga1, gb1 = core.read_gab(0, Z)
    assert (ga1 - ga0) == pytest.approx(gb1 - gb0)


def test_aat_none_is_open_circuit():
    # Two spaces; second NONE contributes nothing -> readout equals single active space.
    core = Core(1, 1, spaces_per_lane=2, model="float", init="medium_noiseless", read_noise=0)
    lane = core.lane(0)
    y_one = lane.evaluate((0, None), "FF")
    assert y_one == pytest.approx(0.0)   # single balanced synapse


def test_aat_length_mismatch_raises():
    core = Core(1, 1, spaces_per_lane=2, model="float")
    with pytest.raises(ValueError):
        core.lane(0).evaluate((0,), "FF")


def test_unknown_instruction_raises():
    core = Core(1, 1, model="float")
    with pytest.raises(KeyError):
        core.lane(0).evaluate(Z, "ZZ")
