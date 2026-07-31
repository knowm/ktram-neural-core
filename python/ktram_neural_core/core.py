"""Core: geometry + axes + control parameters, addressed only by AATs.

A Core is specified the way the hardware is — by unit-crossbar geometry and operating model —
and built as a fabric of neural lanes (each an array of differential-pair address spaces).
It owns the control parameters (forward/reverse drive and low voltages, pulse width); the
device models only consume what the Core hands them. Defaults are model-aware and the Core
sets them, but every parameter stays settable at construction and at runtime.
"""

import math

import numpy as np

from .crossbar.fidelity import FIDELITIES
from .instructions import FORWARD, resolve
from .lane import NeuralLane
from .models import INIT_TYPES, MODELS
from .models.mss_model import MSSProfile
from .topology import TwoOne
from .unit_crossbar import UnitCrossbar, UnitCrossbarPair

# Read noise — the kT-bit's hiss on a read. A reported read carries two physically distinct
# mechanisms, summed in quadrature and each referred to the weight (y = Vy / V_app):
#
#   sigma_y = sqrt(sigma_thermal^2 + sigma_flicker^2)
#
#   * thermal (Johnson-Nyquist) — additive voltage noise over the signal:
#       sigma_thermal = read_noise * noise_thermal * sqrt(T/T_ref) * (V_ref/|V_app|)
#                       * sqrt(m_ref/m) * sqrt(pw_ref/pw)
#     Scales as 1/|V_app| (lower the read voltage -> louder read; the operational noise dial),
#     sqrt(T) (temperature), 1/sqrt(m) (a high-magnitude pair reads quietly), and 1/sqrt(pw):
#     white power is proportional to the read bandwidth Df ~= 1/(2*pw), so a longer read pulse
#     integrates the hiss down. Flat in y.
#   * flicker / RTN (1/f) — multiplicative conductance fluctuation (dG/G), the dominant read
#     noise in real memristors:
#       sigma_flicker = read_noise * noise_flicker * (1 - y^2) * sqrt(m_ref/m) * bw_flicker
#     Flat in |V_app| — a floor the read-voltage dial cannot go below — also 1/sqrt(m), and
#     (1 - y^2): loudest at y = 0 (undecided), vanishing at the rails (a confident pair reads
#     quiet). 1/f power goes as ln(f_high/f_low); the read pulse sets the upper edge
#     f_high ~= 1/(2*pw), so bw_flicker carries a weak sqrt(ln) dependence on pw, spanning
#     FLICKER_DECADES of 1/f band at the reference pulse width. The Hooge factor is in noise_flicker.
#
# Both terms come from one junction node at the read pulse's bandwidth: the SAME draw is reported
# and drives the back-action (see NeuralLane.evaluate). read_noise is the master gain quoted at the
# reference operating point (read voltage READ_NOISE_REF_V, m = m_ref, T = ROOM_TEMPERATURE_K,
# pw = pw_ref); noise_thermal and noise_flicker set the mix, so at the reference point
# sigma_y = read_noise * sqrt(noise_thermal^2 + noise_flicker^2). The constant part is precomputed
# (see _recompute_noise_coeffs) and only m, V_app, y and pw are evaluated per read. Read noise is ON
# by default; read_noise = 0 gives a deterministic Core (draws nothing). A sub-threshold read stays
# non-disturbing because the device does not switch below threshold, not because the read is clean.
ROOM_TEMPERATURE_K = 298.0
READ_NOISE_REF_V = 0.05   # reference read voltage at which the noise constants are quoted
READ_NOISE = 0.02         # master read-noise gain (0 disables read noise entirely)
NOISE_THERMAL = 0.1       # thermal (Johnson) weight — a small floor
NOISE_FLICKER = 1.0       # flicker / RTN (1/f) weight — the dominant term
FLICKER_DECADES = 6.0     # decades of 1/f band at the reference pulse width (sets pw sensitivity)

# Model-aware Core defaults: drive/low voltages and pulse width. The Core initializes from
# the chosen model; all stay settable. RS gets pulse_width 1e-8 so its old alpha*dt = 0.01
# step reproduces by default; MSS drives at +-0.25 (threshold ~0.27 V).
MODEL_DEFAULTS = {
    "float": dict(fwd=1.0,  rev=-1.0,  fwd_lv=0.05, rev_lv=-0.05, pw=1e-6),
    "byte":  dict(fwd=1.0,  rev=-1.0,  fwd_lv=0.05, rev_lv=-0.05, pw=1e-6),
    "rs":    dict(fwd=1.0,  rev=-1.0,  fwd_lv=0.05, rev_lv=-0.05, pw=1e-8),
    "mss":   dict(fwd=0.25, rev=-0.25, fwd_lv=0.05, rev_lv=-0.05, pw=1e-6),
}


class Core:
    def __init__(
        self,
        crossbar_rows,
        crossbar_cols,
        spaces_per_lane=1,
        num_lanes=1,
        model="float",
        fidelity="ideal",
        init="medium",
        seed=None,
        profile=None,
        forward_voltage=None,
        reverse_voltage=None,
        forward_low_voltage=None,
        reverse_low_voltage=None,
        pulse_width=None,
        read_pulse_width=None,
        read_noise=READ_NOISE,
        noise_thermal=NOISE_THERMAL,
        noise_flicker=NOISE_FLICKER,
        temperature=ROOM_TEMPERATURE_K,
        read_noise_ref_m=None,
        read_noise_ref_pw=None,
        flicker_decades=FLICKER_DECADES,
    ):
        self.crossbar_rows = crossbar_rows
        self.crossbar_cols = crossbar_cols
        self.spaces_per_lane = spaces_per_lane
        self.num_lanes = num_lanes
        self.model_name = model.lower()
        self.fidelity_name = fidelity.lower()
        self.init_name = init.lower()

        if self.model_name not in MODELS:
            raise KeyError(f"unknown model {model!r}; valid: {sorted(MODELS)}")
        if self.fidelity_name not in FIDELITIES:
            raise KeyError(f"unknown fidelity {fidelity!r}; valid: {sorted(FIDELITIES)}")
        if self.init_name not in INIT_TYPES:
            raise KeyError(f"unknown init {init!r}; valid: {sorted(INIT_TYPES)}")

        # One seedable RNG covers init noise and MSS switching. Entropy-seeded by default;
        # pass a seed for reproducible runs/figures.
        self.rng = np.random.default_rng(seed)

        # Model-aware control-parameter defaults, with explicit overrides honored.
        d = MODEL_DEFAULTS[self.model_name]
        self.forward_voltage = d["fwd"] if forward_voltage is None else forward_voltage
        self.reverse_voltage = d["rev"] if reverse_voltage is None else reverse_voltage
        self.forward_low_voltage = (
            d["fwd_lv"] if forward_low_voltage is None else forward_low_voltage
        )
        self.reverse_low_voltage = (
            d["rev_lv"] if reverse_low_voltage is None else reverse_low_voltage
        )
        self.pulse_width = d["pw"] if pulse_width is None else pulse_width

        self.profile = profile if profile is not None else (
            MSSProfile() if self.model_name == "mss" else None
        )

        self._topology = TwoOne()
        self._lanes = self._build_lanes()

        # Read-noise control. The reference magnitude defaults to the model's own GMAX
        # (float 1e-1, byte 255, rs/mss 1e-3), so the calibration travels across models. The
        # references fix the operating point at which read_noise/noise_thermal/noise_flicker are
        # quoted (room temp, READ_NOISE_REF_V, m = m_ref); _recompute_noise_coeffs caches the
        # constant part so a read evaluates only the parts that change (m, V_app, y).
        self.read_noise = read_noise
        self.noise_thermal = noise_thermal
        self.noise_flicker = noise_flicker
        self.temperature = temperature
        self.read_noise_ref_T = ROOM_TEMPERATURE_K
        self.read_noise_ref_V = READ_NOISE_REF_V
        self.read_noise_ref_m = (
            read_noise_ref_m
            if read_noise_ref_m is not None
            else self._lanes[0].spaces[0].a.device_at(0).GMAX
        )
        # The read pulse width sets the noise bandwidth; the reference pulse width anchors it so
        # the quoted gains hold at the model's default pulse width and only deviations rescale.
        self.read_noise_ref_pw = (
            read_noise_ref_pw if read_noise_ref_pw is not None else self.pulse_width
        )
        # Read and update pulse widths are decoupled. read_sample keys its bandwidth off
        # read_pulse_width; the drive/update step (NeuralLane.evaluate) keys off pulse_width. They
        # default equal, so a Core built without read_pulse_width reads exactly as before — the
        # separation only matters when a short SAMPLING read must not shrink the write step (see the
        # thermal-reflection line). A shorter read_pulse_width lifts the thermal term as
        # sqrt(read_noise_ref_pw / read_pulse_width) at a normal read voltage.
        self.read_pulse_width = (
            self.pulse_width if read_pulse_width is None else read_pulse_width
        )
        self.flicker_decades = flicker_decades
        self._recompute_noise_coeffs()

    def __setstate__(self, state):
        # Back-fill attributes added after older pickles were written (snapshots are reused across
        # runs). read_pulse_width defaults to the update pulse, which is exactly how a pre-split Core
        # read, so an old snapshot deserializes to identical behavior.
        state.setdefault("read_pulse_width", state.get("pulse_width"))
        self.__dict__.update(state)

    # ----- construction -----

    def _build_lanes(self):
        model_cls = MODELS[self.model_name]
        fidelity = FIDELITIES[self.fidelity_name]()
        mean, rand_var = INIT_TYPES[self.init_name]
        n_dev = self.crossbar_rows * self.crossbar_cols

        def make_unit():
            devices = [
                model_cls.create(mean, rand_var, self.rng, self.profile)
                for _ in range(n_dev)
            ]
            return UnitCrossbar(self.crossbar_rows, self.crossbar_cols, devices, fidelity)

        lanes = []
        for _ in range(self.num_lanes):
            spaces = [
                UnitCrossbarPair(make_unit(), make_unit())   # a-side, then b-side
                for _ in range(self.spaces_per_lane)
            ]
            lanes.append(NeuralLane(spaces, self, self._topology))
        return lanes

    # ----- access -----

    def lane(self, index=0):
        return self._lanes[index]

    def evaluate(self, aat, instruction, lane_index=0, noise=0.0):
        """Run one instruction on a lane; return y. noise in [0,1] dials the read voltage and
        thus the read noise (0 = the set read voltage / least noise, 1 = toward 0 V / most
        noise). See NeuralLane.evaluate."""
        return self._lanes[lane_index].evaluate(aat, instruction, noise=noise)

    # ----- voltage helpers used by the lane -----

    def v_app(self, instruction):
        """Signed applied voltage for an instruction (direction x standard|low)."""
        instr = resolve(instruction)
        if instr.direction == FORWARD:
            return self.forward_low_voltage if instr.low else self.forward_voltage
        return self.reverse_low_voltage if instr.low else self.reverse_voltage

    def drive_voltage(self, direction):
        """Standard drive voltage of a direction (used to set feedback Vy)."""
        return self.forward_voltage if direction == FORWARD else self.reverse_voltage

    # ----- read noise (the kT) -----

    def _recompute_noise_coeffs(self):
        """Cache the constant part of the read-noise law.

        Everything that does not change between reads (the gains, temperature, and references)
        is folded here into three scalars, so read_sample touches only m, V_app and y. Called at
        construction and from set_read_noise whenever an input changes.
        """
        self._a_thermal = (
            self.read_noise
            * self.noise_thermal
            * math.sqrt(self.temperature / self.read_noise_ref_T)
            * self.read_noise_ref_V
        )
        self._a_flicker = self.read_noise * self.noise_flicker
        self._sqrt_ref_m = math.sqrt(self.read_noise_ref_m)
        # ln of the 1/f band at the reference pulse width (f_high_ref/f_low). The read pulse
        # only moves the upper edge, so off-reference the band is this plus ln(pw_ref/pw).
        self._flicker_ln_ref = self.flicker_decades * math.log(10.0)

    def read_sample(self, y_clean, m, v_app):
        """Read noise on the *reported* read — the kT-bit's hiss.

        Two physically distinct mechanisms, summed in quadrature and referred to the weight
        (y = Vy/V_app):

            Df            = 1 / (2 * pw)                              read bandwidth
            sigma_thermal = a_thermal * sqrt(m_ref/m) / |V_app| * sqrt(pw_ref/pw)   (Johnson-Nyquist)
            sigma_flicker = a_flicker * (1 - y^2) * sqrt(m_ref/m) * bw_flicker       (1/f flicker / RTN)
            sigma_y       = sqrt(sigma_thermal^2 + sigma_flicker^2)

        where a_thermal and a_flicker are the precomputed gains (see _recompute_noise_coeffs).
        Thermal is additive voltage noise over the signal: 1/|V_app| (the read-voltage noise
        dial), sqrt(T), 1/sqrt(m), 1/sqrt(pw) (white power follows the read bandwidth), flat in y.
        Flicker is the dominant memristor read noise — multiplicative conductance fluctuation: flat
        in |V_app| (a floor the dial cannot clear), 1/sqrt(m), (1 - y^2) (loudest undecided at
        y = 0, silent at the rails), and a weak sqrt(ln) growth as a longer read pulse lowers the
        upper edge of the 1/f band. m is the common mode summed over the read's active pairs, so a
        multi-pair lane's noise scales with the TOTAL magnitude with no per-pair handling.

        The single draw comes from the Core's seeded RNG and is the same hiss the caller drives the
        back-action with — one junction node at one bandwidth (see NeuralLane.evaluate). read_noise
        <= 0 returns the clean read untouched and draws nothing, so a noise-disabled Core is
        deterministic bit-for-bit.
        """
        if self.read_noise <= 0.0 or m <= 0.0:
            return y_clean
        f_m = self._sqrt_ref_m / math.sqrt(m)
        # Read bandwidth from the READ pulse width (decoupled from the update pulse): white (thermal)
        # power ~ Df ~ 1/pw -> sigma ~ 1/sqrt(pw); 1/f (flicker) power ~ ln(band), and the pulse only
        # moves the band's upper edge.
        bw_thermal = math.sqrt(self.read_noise_ref_pw / self.read_pulse_width)
        ln_band = self._flicker_ln_ref + math.log(self.read_noise_ref_pw / self.read_pulse_width)
        bw_flicker = math.sqrt(ln_band / self._flicker_ln_ref) if ln_band > 0.0 else 0.0
        sigma_thermal = self._a_thermal * f_m / abs(v_app) * bw_thermal
        sigma_flicker = self._a_flicker * (1.0 - y_clean * y_clean) * f_m * bw_flicker
        sigma = math.sqrt(sigma_thermal * sigma_thermal + sigma_flicker * sigma_flicker)
        y = y_clean + sigma * self.rng.standard_normal()
        return -1.0 if y < -1.0 else (1.0 if y > 1.0 else y)

    # ----- runtime setters (the low voltages get setters too — the Java gap) -----

    def set_voltages(
        self,
        forward_voltage=None,
        reverse_voltage=None,
        forward_low_voltage=None,
        reverse_low_voltage=None,
    ):
        if forward_voltage is not None:
            self.forward_voltage = forward_voltage
        if reverse_voltage is not None:
            self.reverse_voltage = reverse_voltage
        if forward_low_voltage is not None:
            self.forward_low_voltage = forward_low_voltage
        if reverse_low_voltage is not None:
            self.reverse_low_voltage = reverse_low_voltage

    def set_pulse_width(self, dt):
        """Set the UPDATE pulse width (the conductance step's dt). Leaves read_pulse_width alone, so
        the read-noise bandwidth is unchanged — set that with set_read_pulse_width."""
        self.pulse_width = dt

    def set_read_pulse_width(self, dt):
        """Set the READ pulse width — the read-noise bandwidth dial (read_sample), independent of the
        update pulse. A shorter read pulse lifts the thermal term as sqrt(read_noise_ref_pw/dt) at a
        normal read voltage; the write step (pulse_width) is untouched."""
        self.read_pulse_width = dt

    def set_read_noise(self, read_noise=None, noise_thermal=None, noise_flicker=None,
                       temperature=None, read_noise_ref_m=None, read_noise_ref_pw=None,
                       flicker_decades=None):
        """Tune the read-noise gains at runtime; read_noise=0 disables read noise entirely.

        read_noise is the master gain; noise_thermal and noise_flicker set the mix of the two
        mechanisms (see read_sample). For per-read control the primary dial is the READ VOLTAGE,
        not these gains: a lower read voltage raises the thermal term (sigma_thermal ~ 1/V_app)
        down to the flicker floor, which is flat in voltage. Set it with
        set_voltages(forward_low_voltage=...) and read sub-threshold (FFLV/RFLV) for noisy,
        non-disturbing reads. temperature scales the thermal term as sqrt(T) (room-temp
        reference)."""
        if read_noise is not None:
            self.read_noise = read_noise
        if noise_thermal is not None:
            self.noise_thermal = noise_thermal
        if noise_flicker is not None:
            self.noise_flicker = noise_flicker
        if temperature is not None:
            self.temperature = temperature
        if read_noise_ref_m is not None:
            self.read_noise_ref_m = read_noise_ref_m
        if read_noise_ref_pw is not None:
            self.read_noise_ref_pw = read_noise_ref_pw
        if flicker_decades is not None:
            self.flicker_decades = flicker_decades
        self._recompute_noise_coeffs()

    # ----- debug / visualization -----

    def read_gab(self, lane_index, aat):
        """Read back (Ga, Gb) for the enabled spaces of an AAT. Debug/plot only.

        Returns a single (Ga, Gb) tuple when exactly one space is enabled (the common
        single-synapse case), else a list of tuples in space order.
        """
        lane = self._lanes[lane_index]
        pairs = [
            space.conductances(addr)
            for space, addr in zip(lane.spaces, aat)
            if addr is not None
        ]
        return pairs[0] if len(pairs) == 1 else pairs

    def set_gab(self, lane_index, aat, ga, gb):
        """Force (Ga, Gb) on the enabled spaces of an AAT. Debug/setup only.

        Conductance-storing models set g directly; MSS converts g -> internal x. Lets you
        place different device types into the *same* state (same y) for matched comparisons.
        """
        lane = self._lanes[lane_index]
        for space, addr in zip(lane.spaces, aat):
            if addr is None:
                continue
            space.a.device_at(addr).set_g(ga)
            space.b.device_at(addr).set_g(gb)

    def set_start_y(self, lane_index, aat, y0, level=0.5):
        """Place the enabled synapses at a target activation y0 in [-1, 1].

        Uses Ga = c(1+y0), Gb = c(1-y0) with c = level*GMax, so Ga+Gb = 2c stays inside the
        device range and (Ga-Gb)/(Ga+Gb) = y0 for any model — identical starting y across
        types regardless of their conductance scale. level=0.5 puts the pair magnitude
        (Ga+Gb) at GMax for y0=0.
        """
        lane = self._lanes[lane_index]
        for space, addr in zip(lane.spaces, aat):
            if addr is None:
                continue
            c = level * space.a.device_at(addr).GMAX
            space.a.device_at(addr).set_g(c * (1 + y0))
            space.b.device_at(addr).set_g(c * (1 - y0))
