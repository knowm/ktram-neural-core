"""Byte device model. From ByteNode.java — conductance quantized to an unsigned byte [1,255].

The update is round(dV) with Java half-up rounding, so it is 0 for |dV|<0.5, +-1 for
0.5<=|dV|<1.5, etc. ``dt`` is ignored. The state is always integer-valued; the Java
``(byte)`` cast on the init float truncates toward zero, which ``int(...)`` reproduces.
"""

from .base import Device, clamp, init_conductance, java_round


class ByteDevice(Device):
    GMIN = 1
    GMAX = 255

    def __init__(self, g):
        self._g = int(g)

    def g(self):
        return float(self._g)

    def set_g(self, g):
        self._g = int(clamp(g, self.GMIN, self.GMAX))

    def drive(self, dV, dt):
        dg = java_round(dV)
        self._g = int(clamp(self._g + dg, self.GMIN, self.GMAX))

    @classmethod
    def create(cls, mean, rand_var, rng, profile=None):
        g = init_conductance(cls.GMIN, cls.GMAX, mean, rand_var, rng)
        return cls(int(g))   # truncate toward zero, matching Java's (byte) cast
