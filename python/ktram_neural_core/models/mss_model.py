"""MSS (Mean Metastable Switch) device model. Stochastic.

From MSSMemristor.java + MSSMemristorNode.java, with the IdealMemristor.java default
profile (Knowm W+SDC, idealized). State is ``x`` in [0,1], the fraction of N metastable
switches in the On state. Conductance is derived from x.

The update is mean-field: one Normal pair per device per step (never one draw per switch).
``current(V)`` carries the optional Schottky term, but the Milestone-1 readout uses the
linear conductance ``g()`` only — the nonlinear-current path is Physical-fidelity work.

The constants below are a *device profile*, not hardcoded physics: the model can represent
many memristor types and fit measured devices. The defaults are the only thing pinned.
"""

import math
from dataclasses import dataclass

from .base import Device, K_BOLTZMANN, Q_CHARGE, clamp, init_conductance, java_round


@dataclass
class MSSProfile:
    """A settable MSS device profile. Defaults = idealized Knowm W+SDC (old IdealMemristor)."""
    Ron: float = 1000.0          # on resistance  (g = 1e-3)
    Roff: float = 10000.0        # off resistance (g = 1e-4)
    N: float = 1000.0            # number of metastable switches
    tau: float = 1e-5            # characteristic switching time (s)
    Von: float = 0.27           # off->on barrier potential (switching threshold, V)
    Voff: float = 0.27          # on->off barrier potential (V)
    phi: float = 1.0            # fraction of current from the MSS term (1 => linear)
    schottky_fa: float = 0.0    # Schottky forward alpha
    schottky_fb: float = 0.0    # Schottky forward beta
    schottky_ra: float = 0.0    # Schottky reverse alpha
    schottky_rb: float = 0.0    # Schottky reverse beta
    temperature: float = 298.0  # K

    @property
    def gmin(self):
        return 1.0 / self.Roff

    @property
    def gmax(self):
        return 1.0 / self.Ron

    @property
    def vt(self):
        """Thermal voltage kT/q (~0.025693 V at 298 K)."""
        return K_BOLTZMANN * self.temperature / Q_CHARGE


class MSSDevice(Device):
    def __init__(self, x, profile, rng):
        self.x = x
        self.p = profile
        self.rng = rng

    @property
    def GMIN(self):
        return self.p.gmin

    @property
    def GMAX(self):
        return self.p.gmax

    def g(self):
        p = self.p
        return self.x / p.Ron + (1.0 - self.x) / p.Roff   # linear conductance for the readout

    def set_g(self, g):
        p = self.p
        r = clamp(1.0 / g, p.Ron, p.Roff)
        self.x = (p.Ron * (r - p.Roff)) / (r * (p.Ron - p.Roff))

    def current(self, v):
        p = self.p
        return p.phi * v * self.g() + (1.0 - p.phi) * self._schottky(v)

    def _schottky(self, v):
        p = self.p
        return (p.schottky_ra * (-math.exp(-p.schottky_rb * v))
                + p.schottky_fa * (math.exp(p.schottky_fb * v)))

    def drive(self, dV, dt):
        p = self.p
        x = self.x
        alpha = dt / p.tau
        voff2on = alpha * 1.0 / (1.0 + math.exp(-(dV - p.Von) / p.vt))
        von2off = alpha * (1.0 - 1.0 / (1.0 + math.exp(-(dV + p.Voff) / p.vt)))

        u_on = (1.0 - x) * p.N * voff2on
        u_off = x * p.N * von2off
        # Binomial variance n*p*(1-p) >= 0. A clamped x=0/1 yields negative zero (or a tiny
        # float-rounding negative); take sqrt only when strictly positive so the RNG never
        # sees a negative scale.
        var_on = (1.0 - x) * p.N * voff2on * (1.0 - voff2on)
        var_off = x * p.N * von2off * (1.0 - von2off)
        sigma_on = math.sqrt(var_on) if var_on > 0.0 else 0.0
        sigma_off = math.sqrt(var_off) if var_off > 0.0 else 0.0

        n_off2on = java_round(self.rng.normal(u_on, sigma_on))   # Off -> On
        n_on2off = java_round(self.rng.normal(u_off, sigma_off))  # On  -> Off

        self.x = clamp(x + (n_off2on - n_on2off) / p.N, 0.0, 1.0)

    @classmethod
    def create(cls, mean, rand_var, rng, profile=None):
        p = profile or MSSProfile()
        g = init_conductance(p.gmin, p.gmax, mean, rand_var, rng)
        r_init = java_round(1.0 / g)
        r_init = clamp(r_init, p.Ron, p.Roff)
        x = (p.Ron * (r_init - p.Roff)) / (r_init * (p.Ron - p.Roff))
        return cls(x, p, rng)
