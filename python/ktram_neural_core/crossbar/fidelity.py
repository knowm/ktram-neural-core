"""CrossbarFidelity: the no-corner rule, as a strategy.

All device contribution/readout and drive flow through a CrossbarFidelity so a different
fidelity drops in without touching call sites. Only ``Ideal`` is implemented this pass;
``Physical`` (floating-line sneak paths + line/terminal resistance) is a later phase and
attaches here as a new strategy.
"""


class CrossbarFidelity:
    def conductance(self, unit, addr):
        """Conductance contribution of the unit crossbar for the given address (0 if NONE)."""
        raise NotImplementedError

    def drive(self, unit, addr, dV, dt):
        """Apply dV for dt to the device(s) the address couples."""
        raise NotImplementedError


class Ideal(CrossbarFidelity):
    """Returns only the selected device's contribution; drives only the selected device.

    No network solve — a trivial passthrough. NONE (open circuit) contributes nothing and
    receives no drive.
    """

    def conductance(self, unit, addr):
        if addr is None:
            return 0.0
        return unit.device_at(addr).g()

    def drive(self, unit, addr, dV, dt):
        if addr is None:
            return
        unit.device_at(addr).drive(dV, dt)


# name -> fidelity strategy class.
FIDELITIES = {
    "ideal": Ideal,
}
