"""Read-noise wiring and invariants.

These check the mechanism, not the model's calibration (that is Alex's to validate):
the two mechanisms have the right shape (thermal scales with 1/V_app, flicker is the
voltage-independent floor that vanishes at the rails); read noise rides only on the returned
read, never on the state; it is drawn from the seeded RNG; and disabling it reproduces the
deterministic readout bit-for-bit.

The tests that isolate a DEVICE mechanism pin comparator_enabled=False. The comparator is
periphery: flat in y and m, 1/|V_app|, and on by default, so it floors the rail read and rides
the voltage dial. Leaving it on would not make those tests harder, it would make them tests of
a different quantity. The comparator has its own tests at the bottom of this file.
"""

import pytest

from ktram_neural_core import Core
from ktram_neural_core.core import (
    COMPARATOR_CODE,
    NOISE_FLICKER,
    NOISE_THERMAL,
    READ_NOISE,
    ROOM_TEMPERATURE_K,
)

Z = (0,)


def _gab_trace(core, n, read="FF", feedback_instr="RH"):
    lane = core.lane(0)
    out = []
    for _ in range(n):
        lane.evaluate(Z, read)
        lane.evaluate(Z, feedback_instr)
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
    # With flicker off and the comparator off, the read-voltage dial isolates the thermal
    # 1/V_app law: a lower read voltage gives a much louder read, and the state still does not
    # move.
    import numpy as np

    def sigma(noise):
        core = Core(1, 1, model="float", init="medium", seed=1,
                    noise_thermal=1.0, noise_flicker=0.0, comparator_enabled=False)
        core.set_start_y(0, Z, 0.0, 0.5)            # m = GMax, w = 0
        lane = core.lane(0)
        before = core.read_gab(0, Z)
        ys = [lane.evaluate(Z, "FFLV", noise=noise) for _ in range(4000)]
        assert core.read_gab(0, Z) == before        # noisy read still does not move the state
        return float(np.std(ys))

    s0, s_hi = sigma(0.0), sigma(0.9)
    assert s_hi > 5 * s0                            # lower read voltage -> much louder thermal read


def test_flicker_is_the_voltage_independent_floor():
    # Flicker is multiplicative, so it is flat in read voltage. With thermal off and the
    # comparator off, dialing the read voltage down does not raise sigma — it is the floor the
    # voltage dial cannot clear. (The comparator is 1/|V_app| like thermal, so it has to be off
    # here or this measures the periphery instead.)
    import numpy as np

    def sigma(noise):
        core = Core(1, 1, model="float", init="medium", seed=1,
                    noise_thermal=0.0, noise_flicker=1.0, comparator_enabled=False)
        core.set_start_y(0, Z, 0.0, 0.5)            # m = GMax, w = 0 -> flicker at its (1-y^2) max
        lane = core.lane(0)
        return float(np.std([lane.evaluate(Z, "FFLV", noise=noise) for _ in range(4000)]))

    assert sigma(0.9) == pytest.approx(sigma(0.0), rel=0.1)


def test_flicker_vanishes_at_the_rails():
    # Flicker carries (1 - y^2): loudest undecided at y = 0, quiet at the rails. With thermal
    # and the comparator off, a fully polarized pair reads essentially noiselessly. (The
    # comparator is flat in y, so it would floor the rail read — that is its own test.)
    import numpy as np

    def sigma(y0):
        core = Core(1, 1, model="float", init="medium", seed=3,
                    noise_thermal=0.0, noise_flicker=1.0, comparator_enabled=False)
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


def test_read_pulse_width_defaults_to_update_pulse():
    # Decoupled read/update pulse widths default equal, across every model, so a Core built without
    # read_pulse_width reads exactly as before.
    for model in ("float", "byte", "rs", "mss"):
        core = Core(1, 1, model=model)
        assert core.read_pulse_width == core.pulse_width


def test_default_read_pulse_reproduces_the_shipped_read():
    # The guard: read_pulse_width == pulse_width (the default) leaves the read path bit-for-bit
    # identical to explicitly pinning the read pulse to the update pulse.
    def run(**kw):
        core = Core(1, 1, model="float", init="medium", seed=1, **kw)
        return [core.lane(0).evaluate(Z, "FFLV") for _ in range(80)]

    assert run() == run(read_pulse_width=Core(1, 1, model="float").pulse_width)


def test_read_pulse_width_is_the_thermal_bandwidth_dial():
    # A shorter READ pulse lifts thermal read noise (~1/sqrt(pw)) at a normal read voltage, and the
    # state still does not move. This is the pulse-width dial the thermal-reflection line samples
    # on. Comparator off: it is flat in pw, so it only dilutes the ratio under test.
    import numpy as np

    def sigma(read_pw):
        core = Core(1, 1, model="float", init="medium", seed=1,
                    noise_thermal=1.0, noise_flicker=0.0, comparator_enabled=False)
        core.set_start_y(0, Z, 0.0, 0.5)                 # m = GMax, w = 0
        core.set_read_pulse_width(read_pw)
        lane = core.lane(0)
        before = core.read_gab(0, Z)
        ys = [lane.evaluate(Z, "FFLV") for _ in range(4000)]
        assert core.read_gab(0, Z) == before             # a hotter read still does not disturb
        return float(np.std(ys))

    ref = Core(1, 1, model="float").pulse_width          # read_noise_ref_pw anchor
    assert sigma(ref / 100) > 5 * sigma(ref)             # 100x shorter -> ~10x thermal


def test_read_and_update_pulse_widths_are_independent():
    # The read dial and the update dial move only their own pulse width.
    core = Core(1, 1, model="byte")
    pw0 = core.pulse_width
    core.set_read_pulse_width(pw0 / 1000)
    assert core.pulse_width == pw0                        # read dial leaves the update pulse alone
    rpw = core.read_pulse_width
    core.set_pulse_width(pw0 / 7)
    assert core.read_pulse_width == rpw                   # update dial leaves the read bandwidth alone


# ---------------------------------------------------------------------------
# The comparator: periphery, on by default, set by an 8-bit register.
# ---------------------------------------------------------------------------

def _sigma(core, y0, level=0.5, n=6000, noise=0.0):
    """Measured sigma of a non-disturbing FFLV read at a pinned weight and magnitude."""
    import numpy as np
    core.set_start_y(0, Z, y0, level)
    lane = core.lane(0)
    return float(np.std([lane.evaluate(Z, "FFLV", noise=noise) for _ in range(n)]))


def test_comparator_defaults_are_on_at_code_10():
    core = Core(1, 1, model="float")
    assert core.comparator_enabled is True
    assert core.comparator_code == COMPARATOR_CODE == 10
    assert core.comparator_noise == pytest.approx(300e-6)
    assert Core(1, 1, model="float", comparator_enabled=False).comparator_noise == 0.0


def test_comparator_register_spans_the_documented_range():
    # Code 0 is the FLOOR, not the absence of a comparator: no comparator reaches zero
    # input-referred noise. The only way to remove the term is comparator_enabled=False.
    assert Core(1, 1, model="float", comparator_code=0).comparator_noise == pytest.approx(100e-6)
    assert Core(1, 1, model="float", comparator_code=255).comparator_noise == pytest.approx(5.2e-3)


def test_disabled_comparator_reproduces_the_pre_comparator_read():
    # The reproduction switch: with the comparator off the law is the two-term device law, and
    # two runs at one seed are bit-for-bit identical.
    def run():
        core = Core(1, 1, model="float", init="medium", seed=1, comparator_enabled=False)
        return [core.lane(0).evaluate(Z, "FFLV") for _ in range(80)]

    assert run() == run()
    # and a disabled comparator is exactly code-independent
    a = Core(1, 1, model="float", init="medium", seed=1, comparator_enabled=False,
             comparator_code=0)
    b = Core(1, 1, model="float", init="medium", seed=1, comparator_enabled=False,
             comparator_code=255)
    assert [a.lane(0).evaluate(Z, "FFLV") for _ in range(40)] == \
           [b.lane(0).evaluate(Z, "FFLV") for _ in range(40)]


def test_comparator_is_flat_in_y_and_in_m():
    # The periphery test. A low-m undecided read and a high-m confident read have very different
    # DEVICE noise; the comparator's contribution to each is the same number. Measured as the
    # difference of variances between comparator on and off.
    def parts(y0, level, code):
        on = Core(1, 1, model="float", init="medium", seed=5, comparator_code=code)
        off = Core(1, 1, model="float", init="medium", seed=5, comparator_enabled=False)
        s_on = _sigma(on, y0, level, n=20000)
        s_off = _sigma(off, y0, level, n=20000)
        return s_on * s_on - s_off * s_off, s_off   # the comparator's variance share, device sigma

    quiet, dev_quiet = parts(0.9, 0.9, 145)   # confident, high magnitude -> small device sigma
    loud, dev_loud = parts(0.0, 0.1, 145)     # undecided, low magnitude  -> large device sigma
    assert dev_loud > 4 * dev_quiet           # the DEVICE parts really are far apart
    want = (3e-3 / 0.05) ** 2                 # v_cmp / |V_FFLV|, squared
    assert quiet == pytest.approx(want, rel=0.12)
    assert loud == pytest.approx(want, rel=0.12)


def test_comparator_scales_as_one_over_read_voltage():
    # sigma_comparator = v_cmp / |V_app|. The read-voltage dial halves V_app, so the term doubles.
    import numpy as np

    def sigma_cmp(noise):
        on = Core(1, 1, model="float", init="medium", seed=7, comparator_code=145,
                  noise_thermal=0.0, noise_flicker=0.0)          # device silenced, comparator alone
        on.set_start_y(0, Z, 0.0, 0.5)
        lane = on.lane(0)
        return float(np.std([lane.evaluate(Z, "FFLV", noise=noise) for _ in range(8000)]))

    assert sigma_cmp(0.0) == pytest.approx(3e-3 / 0.05, rel=0.05)
    assert sigma_cmp(0.5) == pytest.approx(2 * sigma_cmp(0.0), rel=0.08)


def test_comparator_does_not_scale_with_read_noise_temperature_or_pulse_width():
    # It is periphery. read_noise is the DEVICE's calibration gain, and folding the comparator
    # into it would make this a device term — the exact error the term exists to remove.
    def cmp_var(**kw):
        on = Core(1, 1, model="float", init="medium", seed=9, comparator_code=145, **kw)
        off = Core(1, 1, model="float", init="medium", seed=9, comparator_enabled=False, **kw)
        s_on, s_off = _sigma(on, 0.0), _sigma(off, 0.0)
        return s_on * s_on - s_off * s_off

    base = cmp_var()
    assert cmp_var(read_noise=0.2) == pytest.approx(base, rel=0.15)          # 10x the gain
    assert cmp_var(temperature=4 * ROOM_TEMPERATURE_K) == pytest.approx(base, rel=0.15)
    assert cmp_var(read_pulse_width=1e-8) == pytest.approx(base, rel=0.15)   # 100x the bandwidth


def test_read_noise_zero_is_clean_even_with_a_loud_comparator():
    # The one deterministic test mode, and it stays a single switch. An ideal device behind a
    # noisy comparator is a nonphysical combination of the same kind as read_noise = 0.3.
    core = Core(1, 1, model="float", init="medium_noiseless", seed=1,
                read_noise=0, comparator_code=255)
    assert core.lane(0).evaluate(Z, "FF") == 0.0
    a = Core(1, 1, model="float", init="medium", seed=1, read_noise=0, comparator_code=255)
    b = Core(1, 1, model="float", init="medium", seed=1, read_noise=0, comparator_enabled=False)
    assert [a.lane(0).evaluate(Z, "FFLV") for _ in range(40)] == \
           [b.lane(0).evaluate(Z, "FFLV") for _ in range(40)]


def test_empty_lane_read_is_clean_with_the_comparator_on():
    # m <= 0: a lane with nothing selected has no comparator decision to make.
    core = Core(1, 1, spaces_per_lane=2, model="float", init="medium", seed=1)
    assert core.lane(0).evaluate((None, None), "FFLV") == 0.0


def test_comparator_code_is_validated_as_a_register():
    for bad in (-1, 256, 1000):
        with pytest.raises(ValueError):
            Core(1, 1, model="float", comparator_code=bad)
    for bad in (10.0, 10.5, True, "10"):
        with pytest.raises(TypeError):
            Core(1, 1, model="float", comparator_code=bad)


def test_set_comparator_noise_lands_on_the_right_code():
    # The float trap: int((1e-3 - 100e-6) / 20e-6) is 44, not 45, and
    # int((300e-6 - 100e-6) / 20e-6) is 9, not 10. Both land one code low. The conversion runs in
    # integer microvolts so the levels the program cares about are exact.
    core = Core(1, 1, model="float")
    assert core.set_comparator_noise(1e-3) == 45
    assert core.comparator_noise == pytest.approx(1e-3)
    assert core.set_comparator_noise(300e-6) == 10
    assert core.comparator_noise == pytest.approx(300e-6)
    assert core.set_comparator_noise(3e-3) == 145
    assert core.comparator_noise == pytest.approx(3e-3)


def test_set_comparator_noise_refuses_what_the_register_cannot_hold():
    core = Core(1, 1, model="float")
    with pytest.raises(ValueError, match="below the comparator floor"):
        core.set_comparator_noise(50e-6)          # a trimmed comparator: a different design class
    with pytest.raises(ValueError, match="not an exact register code"):
        core.set_comparator_noise(150e-6)         # falls between codes 2 and 3
    with pytest.raises(ValueError, match="above the register"):
        core.set_comparator_noise(7.5e-3)         # above code 255 = 5.20 mV
    assert core.comparator_code == COMPARATOR_CODE        # nothing was set by a failed request
    assert core.set_comparator_noise(150e-6, round_ok=True) == 2   # opt in, and read back what you got
    assert core.comparator_noise == pytest.approx(140e-6)


def test_set_comparator_toggles_and_sets_the_register():
    core = Core(1, 1, model="float")
    core.set_comparator(code=145)
    assert core.comparator_noise == pytest.approx(3e-3)
    core.set_comparator(enabled=False)
    assert core.comparator_noise == 0.0
    assert core.comparator_code == 145            # the register survives the switch
    core.set_comparator(enabled=True)
    assert core.comparator_noise == pytest.approx(3e-3)


def test_pre_comparator_pickle_loads_at_the_default():
    # An old snapshot has no comparator keys. It loads ENABLED at the default code, not at zero:
    # the comparator is part of the read model, not part of the artifact — an old Core was always
    # read by some comparator, the emulator simply was not modeling it.
    import pickle

    core = Core(1, 1, model="float", init="medium", seed=1)
    state = core.__dict__.copy()
    del state["comparator_enabled"], state["comparator_code"]
    old = pickle.loads(pickle.dumps(core))
    old.__setstate__(state)
    assert old.comparator_enabled is True
    assert old.comparator_code == COMPARATOR_CODE
    assert old.comparator_noise == pytest.approx(300e-6)


def test_comparator_state_round_trips_through_a_pickle():
    import pickle

    core = Core(1, 1, model="float", comparator_code=145, comparator_enabled=False)
    back = pickle.loads(pickle.dumps(core))
    assert back.comparator_code == 145            # the CODE round-trips, not the volts
    assert back.comparator_enabled is False
    assert back.comparator_noise == 0.0
