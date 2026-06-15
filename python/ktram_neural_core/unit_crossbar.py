"""UnitCrossbar and UnitCrossbarPair — the hardware-native carriers of memristors.

UnitCrossbar: an Nr x Nc array that carries individually selectable memristors. An address
is a flat index in [0, Nr*Nc); ``None`` is NONE (enable bit off -> open circuit). All
contribution/drive routes through the CrossbarFidelity strategy, so Physical fidelity drops
in later without touching this class.

UnitCrossbarPair: a differential pair of unit crossbars (a-side, b-side) — one signed synapse
(kT-bit) selectable per evaluation. The same address selects index ``addr`` in both sides,
yielding (Ga, Gb). A lane holds an ordered list of these; the AAT addresses them as its
per-entry "spaces" (entry i selects an address within pair i).
"""

from math import log2


class UnitCrossbar:
    def __init__(self, rows, cols, devices, fidelity):
        if len(devices) != rows * cols:
            raise ValueError(f"expected {rows * cols} devices, got {len(devices)}")
        self.rows = rows
        self.cols = cols
        self._devices = devices          # flat, length rows*cols
        self.fidelity = fidelity

    @property
    def size(self):
        return self.rows * self.cols

    @property
    def address_width(self):
        return int(log2(self.rows)) + int(log2(self.cols))

    def device_at(self, addr):
        return self._devices[addr]

    def conductance(self, addr):
        return self.fidelity.conductance(self, addr)

    def drive(self, addr, dV, dt):
        self.fidelity.drive(self, addr, dV, dt)


class UnitCrossbarPair:
    """A differential pair of unit crossbars: one signed synapse selectable per evaluation.

    A lane addresses these as its per-entry "spaces" (see the AAT model).
    """

    def __init__(self, a_side, b_side):
        self.a = a_side
        self.b = b_side

    @property
    def size(self):
        return self.a.size

    def conductances(self, addr):
        """(Ga, Gb) for the selected address (both 0 if NONE)."""
        return self.a.conductance(addr), self.b.conductance(addr)

    def drive(self, addr, dVa, dVb, dt):
        self.a.drive(addr, dVa, dt)
        self.b.drive(addr, dVb, dt)
