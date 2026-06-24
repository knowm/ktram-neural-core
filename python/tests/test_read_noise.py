"""Read-noise wiring and invariants.

These check the mechanism, not the model's calibration (that is Alex's to validate):
the two mechanisms have the right shape (thermal scales with 1/V_app, flicker is the
voltage-independent floor that vanishes at the rails); read noise rides only on the returned
read, never on the state; it is drawn from the seeded RNG; and disabling it reproduces the
deterministic readout bit-for-bit.
"""

import pytest

from ktram_neural_core import Core
from ktram_neural_core.core import (
    NOISE_FLICKER,
    NOISE_THERMAL,
    READ_NOISE,
    ROOM_TEMPERATURE_K,
)

Z = (0,)


def _gab_trace(core, n, read="FF", feedback="RH"):
    lane = core.lane(0)
    out = []
    for _ in range(n):
        lane.evaluate(Z, read)
        lane.evaluate(Z, feedback)
        out.append(core.read_gab(0, Z))
    return out


def test_defaults_are_noise_on_at_room_temperature():
    core = Core(1, 1, model="float")
    assert core.read_noise == READ_NOISE
    assert core.noise_thermal == NOISE_THERMAL
    assert core.noise_flicker == NOISE_FLICKER
    assert core.temperature == ROOM_TEMPERATURE_K
    assert core.read_noise_ref_m == 1e-1   # float GMAX


def test_ref_magnitude_tracks_model_gmax():
    assert Core(1, 1, model="byte").read_noise_ref_m == 255
    assert Core(1, 1, model="rs").read_noise_ref_m == 1e-3
    assert Core(1, 1, model="mss").read_noise_ref_m == pytest.approx(1e-3)


def test_disabled_returns_clean_read_and_draws_nothing():
    # read_noise=0: the reported y is the exact clean ratio, and two runs match bit-for-bit.
    def run():
        core = Core(1, 1, model="float", init="medium", seed=1, read_noise=0)
        lane = core.lane(0)
        return [lane.evaluate(Z, "FF") for _ in range(50)]

    a, b = run(), run()
    assert a == b
    # a balanced medium pair reads exactly 0 with no noise
    core = Core(1, 1, model="float", init="medium_noiseless", read_noise=0)
    assert core.lane(0).evaluate(Z, "FF") == 0.0


def test_noisy_read_drives_the_back_action():
    # Invariant: one junction node, one noisy Vy — the same hiss that lands on the reported read
    # also drives a full-voltage read's back-action. So with noise on, the conductance trace
    # diverges from the noise-off run (same seed); with read_noise=0 it is deterministic.
    noisy = _gab_trace(Core(1, 1, model="float", init="medium", seed=1), 200)
    clean = _gab_trace(Core(1, 1, model="float", init="medium", seed=1, read_noise=0), 200)
    assert noisy != clean
    # read_noise=0 is still bit-for-bit reproducible run to run.
    again = _gab_trace(Core(1, 1, model="float", init="medium", seed=1, read_noise=0), 200)
    assert clean == again


def test_low_voltage_read_is_non_disturbing_even_with_noise():
    core = Core(1, 1, model="float", init="medium", seed=7)   # noise on by default
    lane = core.lane(0)
    before = core.read_gab(0, Z)
    for _ in range(500):
        lane.evaluate(Z, "FFLV")
    assert core.read_gab(0, Z) == before


def test_seeded_noise_is_reproducible():
    def run(seed):
        core = Core(1, 1, model="float", init="medium", seed=seed)
        lane = core.lane(0)
        return [lane.evaluate(Z, "FF") for _ in range(100)]

    assert run(11) == run(11)
    assert run(11) != run(12)


def test_noise_zero_is_the_default_path():
    # evaluate(..., noise=0) takes the unscaled read voltage, so it matches the no-arg call
    # bit-for-bit (same seed).
    def run(**kw):
        core = Core(1, 1, model="float", init="medium", seed=1)
        return [core.lane(0).evaluate(Z, "FFLV", **kw) for _ in range(50)]

    assert run() == run(noise=0.0)


def test_thermal_knob_raises_sigma_without_disturbing():
    # With flicker off, the read-voltage dial isolates the thermal 1/V_app law: a lower read
    # voltage gives a much louder read, and the state still does not move.
    import numpy as np

    def sigma(noise):
        core = Core(1, 1, model="float", init="medium", seed=1,
                    noise_thermal=1.0, noise_flicker=0.0)
        core.set_start_y(0, Z, 0.0, 0.5)            # m = GMax, w = 0
        lane = core.lane(0)
        before = core.read_gab(0, Z)
        ys = [lane.evaluate(Z, "FFLV", noise=noise) for _ in range(4000)]
        assert core.read_gab(0, Z) == before        # noisy read still does not move the state
        return float(np.std(ys))

    s0, s_hi = sigma(0.0), sigma(0.9)
    assert s_hi > 5 * s0                            # lower read voltage -> much louder thermal read


def test_flicker_is_the_voltage_independent_floor():
    # Flicker is multiplicative, so it is flat in read voltage. With thermal off, dialing the
    # read voltage down does not raise sigma — it is the floor the voltage dial cannot clear.
    import numpy as np

    def sigma(noise):
        core = Core(1, 1, model="float", init="medium", seed=1,
                    noise_thermal=0.0, noise_flicker=1.0)
        core.set_start_y(0, Z, 0.0, 0.5)            # m = GMax, w = 0 -> flicker at its (1-y^2) max
        lane = core.lane(0)
        return float(np.std([lane.evaluate(Z, "FFLV", noise=noise) for _ in range(4000)]))

    assert sigma(0.9) == pytest.approx(sigma(0.0), rel=0.1)


def test_flicker_vanishes_at_the_rails():
    # Flicker carries (1 - y^2): loudest undecided at y = 0, silent at the rails. With thermal
    # off, a fully polarized pair reads essentially noiselessly.
    import numpy as np

    def sigma(y0):
        core = Core(1, 1, model="float", init="medium", seed=3,
                    noise_thermal=0.0, noise_flicker=1.0)
        core.set_start_y(0, Z, y0, 0.5)
        lane = core.lane(0)
        # FFLV: a sub-threshold, non-disturbing read, so the spread is pure noise, not drift.
        return float(np.std([lane.evaluate(Z, "FFLV") for _ in range(4000)]))

    assert sigma(0.0) > 20 * sigma(0.999)           # undecided reads far louder than confident


def test_set_read_noise_runtime_toggle():
    core = Core(1, 1, model="float", init="medium_noiseless", seed=1)
    core.set_read_noise(read_noise=0)
    assert core.lane(0).evaluate(Z, "FF") == 0.0          # clean
    core.set_read_noise(read_noise=READ_NOISE)
    # with noise back on a balanced pair no longer reads exactly zero
    assert core.lane(0).evaluate(Z, "FF") != 0.0
