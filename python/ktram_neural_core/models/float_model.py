"""Float device model. From FloatNode.java — conductance *is* the state.

Symmetric threshold dead-zone: a sub-threshold |dV| produces exactly zero change, which is
what makes low-voltage reads non-disturbing for this model. ``dt`` is ignored (no time
dependence).
"""

from .base import Device, clamp, init_conductance


class FloatDevice(Device):
    GMIN = 1e-7
    GMAX = 1e-1

    def __init__(self, g, learning_rate=1e-4, va=0.25, vb=0.25):
        self._g = g
        self.learning_rate = learning_rate
        self.va = va
        self.vb = vb

    def g(self):
        return self._g

    def set_g(self, g):
        self._g = clamp(g, self.GMIN, self.GMAX)

    def drive(self, dV, dt):
        if -self.vb < dV < self.va:          # below threshold -> no change
            dg = 0.0
        else:
            dg = self.learning_rate * dV
        self._g = clamp(self._g + dg, self.GMIN, self.GMAX)

    @classmethod
    def create(cls, mean, rand_var, rng, profile=None):
        return cls(init_conductance(cls.GMIN, cls.GMAX, mean, rand_var, rng))
