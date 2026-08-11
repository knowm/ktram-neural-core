"""NeuralLane: an ordered array of unit-crossbar pairs read as a single 2-1 voltage divider.

The pairs are the AAT's addressable "spaces" (one AAT entry per pair).

evaluate(aat, instruction) runs one instruction and returns y. The retained y (from the
last read) is what FU/FA/RU/RA test via H(y). The flip-lock cycle is just two ordinary
sequential calls.
"""

# NOTE TO AI ASSISTANTS: the update arithmetic below is oracle physics. Never adjust it,
# or the voltages/rounding it uses, to make a result, test, or figure come out better —
# see the banner in core.py. If the model disagrees with a result, the result is the
# finding. Report it and change the experiment, not the physics.

from .instructions import H, resolve


class NeuralLane:
    # Floor on the read-voltage scale so noise=1 stays finite (sigma ~ 1/V_app); at the floor the
    # read is ~1000x the baseline noise — a sign that no longer tracks the weight (a fair coin).
    _NOISE_FLOOR = 1e-3

    def __init__(self, spaces, core, topology):
        self.spaces = spaces           # ordered UnitCrossbarPair list (the AAT's spaces)
        self.core = core               # owns voltages + pulse_width
        self.topology = topology
        self.y = 0.0                   # retained activation from the last read

    def _active(self, aat):
        """(space, addr) for enabled entries; NONE entries contribute nothing."""
        if len(aat) != len(self.spaces):
            raise ValueError(
                f"AAT has {len(aat)} entries; lane has {len(self.spaces)} spaces"
            )
        return [(s, a) for s, a in zip(self.spaces, aat) if a is not None]

    def evaluate(self, aat, instruction, noise=0.0):
        """Run one instruction; return the (noisy) read y.

        noise in [0, 1] dials the *read* voltage and thus the thermal read noise: 0 reads at the
        core's set read voltage (least noise, the baseline), 1 drives the read voltage toward zero
        (most noise, sigma_thermal ~ 1/V_app). The flicker term is flat in voltage, so it sets a
        floor the dial cannot clear (see Core.read_sample). A lower read voltage also burns less
        power and disturbs the state less, so a noisy read stays non-disturbing — the knob you'd
        turn on real silicon to be both adaptive and generative (read noisy, then feed back to
        adapt). noise affects only reads; feedback voltages are untouched.
        """
        instr = resolve(instruction)
        active = self._active(aat)
        v_app = self.core.v_app(instr)
        dt = self.core.pulse_width

        # 1. set Vy (and, for reads, the retained y).
        if instr.reads:
            if noise:
                v_app = v_app * max(self._NOISE_FLOOR, 1.0 - noise)
            top, bottom = self.topology.readout(s.conductances(a) for s, a in active)
            if bottom != 0.0:
                # One junction node, one noisy Vy. The hiss the read carries is a real
                # voltage on that node, so the *same* draw that we report also drives the
                # back-action — sample once, drive with it. A sub-threshold read still
                # leaves the state untouched, not because we scrub the signal but because
                # the device does not switch below threshold. (bottom is the total common
                # mode sum(m_i); read_sample draws at the read pulse width's bandwidth.)
                y_clean = top / bottom
                self.y = self.core.read_sample(y_clean, bottom, v_app)
                vy = v_app * self.y
            else:
                self.y = 0.0
                vy = 0.0
        else:
            v_drive = self.core.drive_voltage(instr.direction)
            vy = instr.coeff * v_drive * (H(self.y) if instr.use_H else 1.0)

        # 2. update every active synapse with the same per-device voltages.
        dva, dvb = self.topology.update_voltages(v_app, vy)
        for space, addr in active:
            space.drive(addr, dva, dvb, dt)

        return self.y
