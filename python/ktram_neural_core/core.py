"""Core: geometry + axes + control parameters, addressed only by AATs.

A Core is specified the way the hardware is — by unit-crossbar geometry and operating model —
and built as a fabric of neural lanes (each an array of differential-pair address spaces).
It owns the control parameters (forward/reverse drive and low voltages, pulse width); the
device models only consume what the Core hands them. Defaults are model-aware and the Core
sets them, but every parameter stays settable at construction and at runtime.
"""

# ============================================================================
# NOTE TO AI ASSISTANTS — THE PHYSICS IN THIS FILE IS THE ORACLE. DO NOT TUNE IT.
#
# The device models, the read-noise law, the voltages, the pulse widths, and the
# update arithmetic describe what our hardware actually does. They are the
# standard every experiment, benchmark, figure, and published number in this
# project is measured against.
#
# You must NEVER change any of it to make a result come out better. Not to make a
# test pass, not to make an experiment work, not to close a gap with a baseline,
# not to match a claim already written in an article. Doing so silently
# invalidates every measurement in the repo at once, and there is no way to tell
# afterward which numbers were real. That is a catastrophic outcome for this
# project, whose entire value rests on the emulator predicting silicon we are
# building. A flattering number produced by a bent model is worth less than
# nothing: it is a lie we would then design hardware against.
#
# If a result disagrees with the model, THE RESULT IS THE FINDING. Report it.
# Change the experiment, the encoding, the routine, or the claim — never the
# physics. If you believe the model itself is wrong, say so plainly and stop;
# a physics revision is Alex's call, is made deliberately, and requires
# re-measuring everything downstream of it.
#
# Refactors that provably preserve numerics (renames, extracted helpers) are
# fine. Anything that moves a number is not yours to make.
# ============================================================================

import math

import numpy as np

from .crossbar.fidelity import FIDELITIES
from .instructions import FORWARD, resolve
from .lane import NeuralLane
from .models import INIT_TYPES, MODELS
from .models.mss_model import MSSProfile
from .topology import TwoOne
from .unit_crossbar import UnitCrossbar, UnitCrossbarPair

# Read noise — what a read actually carries. A reported read carries three physically distinct
# mechanisms, summed in quadrature and each referred to the weight (y = Vy / V_app):
#
#   sigma_y = sqrt(sigma_thermal^2 + sigma_flicker^2 + sigma_comparator^2)
#
# The first two are the DEVICE, and read_noise is their calibrated gain. The third is the
# PERIPHERY, and it is the only one a circuit designer sets.
#
#   * thermal (Johnson-Nyquist) — additive voltage noise over the signal:
#       sigma_thermal = read_noise * noise_thermal * sqrt(T/T_ref) * (V_ref/|V_app|)
#                       * sqrt(m_ref/m) * sqrt(pw_ref/pw)
#     Scales as 1/|V_app| (lower the read voltage -> louder read), sqrt(T) (temperature),
#     1/sqrt(m) (a high-magnitude pair reads quietly), and 1/sqrt(pw): white power is
#     proportional to the read bandwidth Df ~= 1/(2*pw), so a longer read pulse integrates the
#     noise down. Flat in y.
#   * flicker / RTN (1/f) — multiplicative conductance fluctuation (dG/G), the dominant read
#     noise in real memristors:
#       sigma_flicker = read_noise * noise_flicker * (1 - y^2) * sqrt(m_ref/m) * bw_flicker
#     Flat in |V_app| — a floor the read-voltage dial cannot go below — also 1/sqrt(m), and
#     (1 - y^2): loudest at y = 0 (undecided), vanishing at the rails (a confident pair reads
#     quietly). 1/f power goes as ln(f_high/f_low); the read pulse sets the upper edge
#     f_high ~= 1/(2*pw), so bw_flicker carries a weak sqrt(ln) dependence on pw, spanning
#     FLICKER_DECADES of 1/f band at the reference pulse width. The Hooge factor is in noise_flicker.
#   * comparator — input-referred noise of the comparator that resolves the read:
#       sigma_comparator = v_cmp / |V_app|
#     Nothing reads a lane directly. A comparator resolves every read, and every comparator has
#     input-referred noise: kT/C on the regeneration nodes plus the preamp's thermal noise
#     (Razavi, "The StrongARM Latch", IEEE SSC Magazine 7(2), 2015). Flat in y and flat in m,
#     because it is periphery and not device. NOT multiplied by read_noise — read_noise is the
#     device's gain, and scaling this term by it would make it a device term. Divides by |V_app|
#     because the read is referred to the weight. It is set by transistor size, capacitance and
#     bias current, so it is the one term a circuit actually controls, and it can be RAISED at no
#     cost (starve the preamp bias, strobe the latch earlier, inject at the input). Comparator
#     OFFSET is a different quantity (Pelgrom matching), static per lane, absorbed by learning —
#     it is not part of this law.
#
# The device terms come from one junction node at the read pulse's bandwidth. read_noise is their
# master gain quoted at the reference operating point (read voltage READ_NOISE_REF_V, m = m_ref,
# T = ROOM_TEMPERATURE_K, pw = pw_ref); noise_thermal and noise_flicker set the mix, so at the
# reference point their part of sigma_y is read_noise * sqrt(noise_thermal^2 + noise_flicker^2).
# read_noise is a CALIBRATION, not an operating dial: the knobs are the read voltage V_app, the
# read pulse width, and the comparator register. The constant part is precomputed (see
# _recompute_noise_coeffs) and only m, V_app, y and pw are evaluated per read.
#
# The comparator level is an 8-bit trim register measured UP FROM A FLOOR, because a comparator
# cannot have zero input-referred noise — the floor is set by the design class (raw dynamic latch
# 0.5-2 mV rms; auto-zeroed 50-200 uV; trimmed tens of uV, below this floor and deliberately not
# representable):
#
#   v_cmp = COMPARATOR_V_MIN + comparator_code * COMPARATOR_V_STEP     0..255 -> 100 uV .. 5.20 mV
#
# Code 0 is the quietest comparator modeled, not the absence of one. comparator_enabled = False is
# the only way to remove the term, and that is a statement about the model rather than a circuit
# setting. Both are ON by default: a read with no comparator is not a quiet read, it is not a read,
# and defaulting the term off would ship a case no hardware can perform.
#
# ALL of it is ON by default. read_noise = 0 gives a deterministic Core (draws nothing) REGARDLESS
# of the comparator — that is the one test mode, and an ideal device behind a noisy comparator is a
# nonphysical combination. A sub-threshold read stays non-disturbing because the device does not
# switch below threshold, not because the read is clean.
ROOM_TEMPERATURE_K = 298.0
READ_NOISE_REF_V = 0.05   # reference read voltage at which the noise constants are quoted
READ_NOISE = 0.02         # master read-noise gain (0 disables read noise entirely)
NOISE_THERMAL = 0.1       # thermal (Johnson) weight — a small floor
NOISE_FLICKER = 1.0       # flicker / RTN (1/f) weight — the dominant term
FLICKER_DECADES = 6.0     # decades of 1/f band at the reference pulse width (sets pw sensitivity)

# Comparator trim register. 8 bits counting up from a floor: no comparator reaches zero
# input-referred noise, so code 0 is the quietest one modeled. The 20 uV step is chosen to put the
# levels the program cares about on exact codes: 300 uV = 10, 1.00 mV = 45, 3.00 mV = 145.
COMPARATOR_ENABLED = True   # the comparator is modeled by default (a read without one is not a read)
COMPARATOR_V_MIN = 100e-6   # code 0 — the floor, the quietest comparator this emulator models
COMPARATOR_V_STEP = 20e-6   # volts per code step (code 255 -> 5.20 mV)
COMPARATOR_CODE = 10        # default register setting -> 300 uV rms
COMPARATOR_CODE_MAX = 255

# Model-aware Core defaults: drive/low voltages and pulse width. The Core initializes from
# the chosen model; all stay settable. RS gets pulse_width 1e-8 so its old alpha*dt = 0.01
# step reproduces by default; MSS drives at +-0.25 (threshold ~0.27 V).
MODEL_DEFAULTS = {
    "float": dict(fwd=1.0,  rev=-1.0,  fwd_lv=0.05, rev_lv=-0.05, pw=1e-6),
    "byte":  dict(fwd=1.0,  rev=-1.0,  fwd_lv=0.05, rev_lv=-0.05, pw=1e-6),
    "rs":    dict(fwd=1.0,  rev=-1.0,  fwd_lv=0.05, rev_lv=-0.05, pw=1e-8),
    "mss":   dict(fwd=0.25, rev=-0.25, fwd_lv=0.05, rev_lv=-0.05, pw=1e-6),
}


def _check_comparator_code(code):
    """The register is an integer 0-255. A float is rejected rather than truncated: the point of
    the register is that a result cannot depend on a level no hardware could be programmed to."""
    if isinstance(code, bool) or not isinstance(code, (int, np.integer)):
        raise TypeError(f"comparator_code must be an int 0-{COMPARATOR_CODE_MAX}, got {code!r}")
    if not 0 <= code <= COMPARATOR_CODE_MAX:
        raise ValueError(f"comparator_code must be 0-{COMPARATOR_CODE_MAX}, got {code}")
    return int(code)


def comparator_volts(code):
    """The register setting as input-referred rms volts."""
    return COMPARATOR_V_MIN + _check_comparator_code(code) * COMPARATOR_V_STEP


def comparator_code_for(v_n, round_ok=False):
    """Nearest register code for a requested input-referred noise, in integer microvolts.

    Do NOT do this in binary floating point: int((1e-3 - 100e-6) / 20e-6) is 44, not 45, and
    int((300e-6 - 100e-6) / 20e-6) is 9, not 10. Both land one code low, so the caller would
    silently get 980 uV where they asked for 1 mV.

    A request below the floor raises rather than clamping to code 0 — it means the caller wants a
    comparator class this emulator does not model, and quietly handing them a different one is how
    a wrong number gets published. A request that is not an exact multiple of the step raises
    unless round_ok.
    """
    uv = round(float(v_n) * 1e6, 6)
    lo_uv = COMPARATOR_V_MIN * 1e6
    step_uv = COMPARATOR_V_STEP * 1e6
    if uv < lo_uv:
        raise ValueError(
            f"{v_n * 1e3:.4g} mV is below the comparator floor "
            f"({COMPARATOR_V_MIN * 1e3:.4g} mV, code 0); that is a comparator class this "
            f"emulator does not model. Use comparator_enabled=False to remove the term.")
    steps = (uv - lo_uv) / step_uv
    code = int(round(steps))
    if not round_ok and abs(steps - code) > 1e-9:
        raise ValueError(
            f"{v_n * 1e3:.4g} mV is not an exact register code (nearest is {code}, "
            f"{comparator_volts(code) * 1e3:.4g} mV); pass round_ok=True to accept the nearest.")
    if code > COMPARATOR_CODE_MAX:
        raise ValueError(
            f"{v_n * 1e3:.4g} mV is above the register's ceiling "
            f"({comparator_volts(COMPARATOR_CODE_MAX) * 1e3:.4g} mV, code {COMPARATOR_CODE_MAX})")
    return code


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
        comparator_enabled=COMPARATOR_ENABLED,
        comparator_code=COMPARATOR_CODE,
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
        # The comparator that resolves the read. Periphery, not device: it is not scaled by
        # read_noise, and its level is a register rather than a float (see the law comment).
        self.comparator_enabled = bool(comparator_enabled)
        self.comparator_code = _check_comparator_code(comparator_code)
        self._recompute_noise_coeffs()

    def __setstate__(self, state):
        # Back-fill attributes added after older pickles were written (snapshots are reused across
        # runs). read_pulse_width defaults to the update pulse, which is exactly how a pre-split Core
        # read, so an old snapshot deserializes to identical behavior.
        state.setdefault("read_pulse_width", state.get("pulse_width"))
        # The comparator is part of the read model, not part of the snapshot: an old Core was
        # always read by SOME comparator, this emulator simply was not modeling it. So a
        # pre-comparator pickle deserializes at the default register setting, not at zero.
        state.setdefault("comparator_enabled", COMPARATOR_ENABLED)
        state.setdefault("comparator_code", COMPARATOR_CODE)
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

    @property
    def comparator_noise(self):
        """The comparator's input-referred rms noise in volts, derived from the register. Read
        only: the register is the source of truth. 0 when the comparator is not modeled."""
        return comparator_volts(self.comparator_code) if self.comparator_enabled else 0.0

    def read_sample(self, y_clean, m, v_app):
        """Read noise on the *reported* read.

        Three physically distinct mechanisms, summed in quadrature and referred to the weight
        (y = Vy/V_app):

            Df               = 1 / (2 * pw)                           read bandwidth
            sigma_thermal    = a_thermal * sqrt(m_ref/m) / |V_app| * sqrt(pw_ref/pw)  (Johnson-Nyquist)
            sigma_flicker    = a_flicker * (1 - y^2) * sqrt(m_ref/m) * bw_flicker     (1/f flicker / RTN)
            sigma_comparator = v_cmp / |V_app|                        (the comparator, periphery)
            sigma_y          = sqrt(sigma_thermal^2 + sigma_flicker^2 + sigma_comparator^2)

        where a_thermal and a_flicker are the precomputed gains (see _recompute_noise_coeffs).
        Thermal is additive voltage noise over the signal: 1/|V_app| (the read-voltage noise
        dial), sqrt(T), 1/sqrt(m), 1/sqrt(pw) (white power follows the read bandwidth), flat in y.
        Flicker is the dominant memristor read noise — multiplicative conductance fluctuation: flat
        in |V_app| (a floor the dial cannot clear), 1/sqrt(m), (1 - y^2) (loudest undecided at
        y = 0, silent at the rails), and a weak sqrt(ln) growth as a longer read pulse lowers the
        upper edge of the 1/f band. m is the common mode summed over the read's active pairs, so a
        multi-pair lane's noise scales with the TOTAL magnitude with no per-pair handling.

        The comparator term is the periphery: flat in y and flat in m, NOT scaled by read_noise,
        temperature or the read pulse width, and 1/|V_app| because the read is referred to the
        weight. Its level is the trim register (see comparator_noise). comparator_enabled = False
        drops it and gives the device-only law exactly.

        The single draw comes from the Core's seeded RNG and is the same noise the caller drives
        the back-action with — one junction node at one bandwidth (see NeuralLane.evaluate).
        read_noise <= 0 returns the clean read untouched and draws nothing REGARDLESS of the
        comparator, so a noise-disabled Core is deterministic bit-for-bit: that is the one test
        mode, and an ideal device behind a noisy comparator is a nonphysical combination. m <= 0
        is clean for the same reason a lane with nothing selected has no comparator decision to
        make.
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
        sigma_comparator = self.comparator_noise / abs(v_app)
        sigma = math.sqrt(sigma_thermal * sigma_thermal + sigma_flicker * sigma_flicker
                          + sigma_comparator * sigma_comparator)
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

    def set_comparator(self, enabled=None, code=None):
        """Set the comparator switch and/or its trim register (an int 0-255).

        This is the operating knob. read_noise is the device's calibration and is not one; the
        dials a circuit actually has are the read voltage, the read pulse width, and this
        register. Raising the level costs nothing on real hardware — starve the preamp bias,
        strobe the latch earlier, or inject at the input.
        """
        if enabled is not None:
            self.comparator_enabled = bool(enabled)
        if code is not None:
            self.comparator_code = _check_comparator_code(code)

    def set_comparator_noise(self, v_n, round_ok=False):
        """Set the register from a requested input-referred rms noise, in volts.

        A convenience over set_comparator: the register stays the source of truth, so the
        achieved level is whatever comparator_noise reads back afterwards, not what was asked
        for. Below the floor raises; a level that is not an exact code raises unless round_ok
        (see comparator_code_for). Returns the code that was set.
        """
        self.comparator_code = comparator_code_for(v_n, round_ok=round_ok)
        return self.comparator_code

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
