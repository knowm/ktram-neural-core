"""Shared pieces for the four device models.

A device model owns one memristor's dynamics: its conductance ``g()``, its current at a
voltage ``current(V)``, and its state update under an applied voltage ``drive(dV, dt)``.
Nothing operational (drive voltage, pulse width) is stored on a device — the Core hands
``dV`` and ``dt`` in on every call. A model holds only its own physics constants.
"""

# NOTE TO AI ASSISTANTS: every device model in this package is oracle physics — it describes
# real memristors we fabricate and measure. Never retune a constant, a threshold, or a
# switching rule to make a result, test, or figure come out better. See the banner in core.py.
# If the model disagrees with a result, the result is the finding: report it and change the
# experiment, not the device.

import math

# Physical constants (used by the MSS thermal model).
K_BOLTZMANN = 1.3806503e-23
Q_CHARGE = 1.60217646e-19

# SynapticInitType -> (mean, randVar). From IKtRAM.java.
INIT_TYPES = {
    "low": (0.0, 0.05),
    "medium": (0.5, 0.05),
    "high": (1.0, 0.05),
    "low_noise": (0.0, 0.25),
    "high_noise": (1.0, 0.25),
    "medium_noise": (0.5, 0.25),
    "medium_high_noise": (0.5, 0.5),
    "low_noiseless": (0.0, 0.0),
    "medium_noiseless": (0.5, 0.0),
    "low_noise_small": (0.0, 0.05),
}


def java_round(x):
    """Half-up rounding, matching Java ``Math.round`` (``floor(x + 0.5)``).

    Differs from Python's banker's rounding on ``.5`` boundaries: round(2.5)==3,
    round(-2.5)==-2. Used by Byte's ``dG`` and MSS's switch-count draws.
    """
    return math.floor(x + 0.5)


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def init_conductance(gmin, gmax, mean, rand_var, rng):
    """initG = GMin + (GMax-GMin)*(mean + randVar*gaussian()), clamped to [GMin, GMax].

    The RNG is consumed only when ``rand_var > 0`` (a noiseless init draws nothing, so
    ``*_NOISELESS`` types stay RNG-free for pure-math unit tests).
    """
    noise = rand_var * rng.standard_normal() if rand_var > 0 else 0.0
    g = gmin + (gmax - gmin) * (mean + noise)
    return clamp(g, gmin, gmax)


class Device:
    """Interface every device model implements.

    Subclasses expose a ``create(mean, rand_var, rng, profile=None)`` classmethod that
    builds one device from a SynapticInitType ``(mean, randVar)`` pair.
    """

    def g(self):
        """Linear conductance used by the 2-1 readout."""
        raise NotImplementedError

    def current(self, v):
        """Current at applied voltage v. Reserved for future Physical-fidelity solving;
        the Milestone-1 readout uses g() only. Default is linear (V*g)."""
        return v * self.g()

    def drive(self, dV, dt):
        """Apply per-device voltage dV for duration dt; mutate internal state."""
        raise NotImplementedError
