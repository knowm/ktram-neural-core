"""Float device model. From FloatNode.java — conductance *is* the state.

Symmetric threshold dead-zone: a sub-threshold |dV| produces exactly zero change, which is
what makes low-voltage reads non-disturbing for this model. ``dt`` is ignored (no time
dependence).
"""

from .base import Device, clamp, init_conductance


class FloatDevice(Device):
    # GMAX/GMIN is the device on/off ratio. 4e-4 gives ~250:1, matching the byte model (255:1)
    # and the realistic ~10-300:1 of real memristors. (The former 1e-7 gave an unphysical 1e6:1,
    # which let a differential pair deplete to a vanishing magnitude under repeated feedback, where
    # the read noise blows up and the weight goes hair-trigger.)
    GMIN = 4e-4
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
