"""RS (resistive switch) device model. From RSMemristorNode.java — deterministic, no RNG.

Conductance is pulled toward GMax (dV>0) or GMin (dV<0) through a thresholded transfer
``Vt(dV)`` with a +-0.25 dead-zone. Time-integrating: it uses ``dt``.

The old Java node hardcoded its own ``dt = 1e-8`` (overriding the node pulse width), giving
an effective step factor alpha*dt = 0.01. Here RS holds no dt of its own — it consumes the
Core's pulse_width, and the Core's model-aware default for RS is 1e-8, so the old curves
reproduce by default.
"""

from .base import Device, clamp, init_conductance


class RSDevice(Device):
    GMIN = 1e-7
    GMAX = 1e-3

    def __init__(self, g, alpha=1e6, va=0.25, vb=0.25):
        self._g = g
        self.alpha = alpha
        self.va = va
        self.vb = vb

    def g(self):
        return self._g

    def set_g(self, g):
        self._g = clamp(g, self.GMIN, self.GMAX)

    def _vt(self, dV):
        if dV > self.va:
            return dV - self.va
        if dV < -self.vb:
            return -(dV + self.vb)
        return 0.0

    def drive(self, dV, dt):
        if dV > 0:
            dg = self.alpha * self._vt(dV) * dt * (self.GMAX - self._g)   # pull toward GMax
        else:
            dg = self.alpha * self._vt(dV) * dt * (self.GMIN - self._g)   # pull toward GMin
        self._g = clamp(self._g + dg, self.GMIN, self.GMAX)

    @classmethod
    def create(cls, mean, rand_var, rng, profile=None):
        return cls(init_conductance(cls.GMIN, cls.GMAX, mean, rand_var, rng))
