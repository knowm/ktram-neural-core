"""Per-model math: dG/drive, init formulas, java_round."""

import numpy as np
import pytest

from ktram_neural_core import ByteDevice, FloatDevice, MSSDevice, MSSProfile, RSDevice
from ktram_neural_core.models import init_conductance, java_round


def test_java_round_half_up():
    assert java_round(0.5) == 1
    assert java_round(-0.5) == 0
    assert java_round(2.5) == 3
    assert java_round(-2.5) == -2
    assert java_round(1.4) == 1
    assert java_round(1.6) == 2


def test_init_conductance_noiseless_is_rng_free():
    rng = np.random.default_rng(0)
    # medium_noiseless: mean 0.5, randVar 0 -> midpoint, no draw consumed.
    g = init_conductance(1.0, 255.0, 0.5, 0.0, rng)
    assert g == pytest.approx(128.0)
    assert rng.standard_normal() == np.random.default_rng(0).standard_normal()


def test_float_deadzone_and_step():
    d = FloatDevice(1e-3)
    g0 = d.g()
    d.drive(0.2, 1e-6)            # |dV| < 0.25 -> no change
    assert d.g() == g0
    d.drive(0.5, 1e-6)            # above threshold -> + learning_rate*dV
    assert d.g() == pytest.approx(g0 + 1e-4 * 0.5)


def test_float_clamps_to_rails():
    d = FloatDevice(FloatDevice.GMAX)
    d.drive(1.0, 1e-6)
    assert d.g() == FloatDevice.GMAX


def test_byte_round_and_clamp():
    d = ByteDevice(128)
    d.drive(-2.0, 1e-6)
    assert d.g() == 126.0
    d.drive(0.4, 1e-6)           # round(0.4)=0 -> no change
    assert d.g() == 126.0
    d = ByteDevice(2)
    d.drive(-5.0, 1e-6)          # clamp to GMIN=1
    assert d.g() == 1.0


def test_rs_pulls_toward_rails_and_deadzone():
    d = RSDevice(1e-5)
    g0 = d.g()
    d.drive(0.2, 1e-8)           # sub-threshold -> Vt=0 -> no change
    assert d.g() == g0
    d.drive(1.0, 1e-8)           # positive -> toward GMax
    assert d.g() > g0
    d2 = RSDevice(1e-5)
    d2.drive(-1.0, 1e-8)         # negative -> toward GMin
    assert d2.g() < g0


def test_mss_init_x_and_conductance():
    rng = np.random.default_rng(0)
    p = MSSProfile()
    # medium_noiseless: initG at midpoint of [gmin,gmax] -> x ~ 0.5.
    d = MSSDevice.create(0.5, 0.0, rng, p)
    assert 0.49 < d.x < 0.51
    assert d.g() == pytest.approx(d.x / p.Ron + (1 - d.x) / p.Roff)


def test_mss_gmin_gmax_track_profile():
    p = MSSProfile()
    assert p.gmin == pytest.approx(1 / p.Roff)
    assert p.gmax == pytest.approx(1 / p.Ron)
    assert p.vt == pytest.approx(0.025680, abs=1e-5)   # kT/q at 298 K from the Java constants
