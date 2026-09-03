# Emulator Overview

`ktram-neural-core` is a Python emulator of **kT-RAM** — Thermodynamic RAM — built as a
**2-1 neural lane**. It is the software half of a build: it runs the kT-RAM instruction set
and the experiments ahead of the physical memristor-crossbar circuits it models, so hardware
and emulator can be developed side by side. It is for teaching and intuition, not production or
large-scale EDA simulation.

This document describes the classes, what they do, the parameters they take, and how to drive
the emulator. It is a Python port of Knowm's original Java implementation; where the behavior
intentionally diverges from that Java, the source files say so.

---

## The mental model in one paragraph

A **memristor pair** — two memristors read against each other — is one signed synapse, the
**kT-bit**. You read it by applying a voltage and measuring the divided voltage that comes back;
the ratio is the weight `y` in `[-1, 1]`. You write it by applying voltage pulses that nudge the
two devices in opposite directions. **Reading the memory is computing with it** — there is no
separate fetch, ALU, or trip to DRAM. The emulator wraps that single physical act (`evaluate`)
in an instruction set and an addressing scheme, then stacks it up into lanes and a core.

---

## Object model

The structure mirrors the hardware, not an idealized flat array of synapses:

```
Core                          geometry + operating model + control voltages; addressed only by AATs
 └─ NeuralLane[]              2-1 differential readout: y = Σ(Ga−Gb) / Σ(Ga+Gb)
      └─ UnitCrossbarPair[]   a differential pair = one signed synapse (kT-bit); a lane's "spaces"
           ├─ UnitCrossbar    a-side — an Nr×Nc array carrying selectable memristors
           └─ UnitCrossbar    b-side — the differential partner
                └─ Device     one memristor's dynamics (float / byte / mss / rs)
```

Everything is selected by an **Activation Address Tuple (AAT)** — one entry per space in the
lane. `None` in an entry means "this space is disabled" (open circuit, contributes nothing).

A `Device` holds *only* its own physics constants. Nothing operational (drive voltage, pulse
width) is stored on a device — the `Core` owns those and hands `dV` and `dt` to the device on
every call.

---

## Installation

Runtime dependency is NumPy only.

```bash
pip install -e python/                      # from a checkout
pip install -e "python/[examples,dev]"      # + matplotlib (plotting) and pytest (tests)
```

You can also run straight from the repo without installing — the example helpers prepend the
`python/` directory to the import path.

---

## Quickstart — a single synapse

The smallest lane is one address space, one differential pair, one device per side. The AAT is
the fixed tuple `(0,)`.

```python
from ktram_neural_core import Core

core = Core(1, 1, spaces_per_lane=1, num_lanes=1, model="byte", init="medium")
lane = core.lane(0)
z = (0,)                       # the AAT: address 0 in the one space

y = lane.evaluate(z, "FF")     # READ — computes y from the devices, returns it
    lane.evaluate(z, "RH")     # WRITE — one instruction per call

ga, gb = core.read_gab(0, z)   # read back the two conductances — debug/plotting only
```

`evaluate` runs **one instruction per call**. The old Java `execute(read, feedback)` is just two
sequential calls: a read, then a feedback write.

---

## `Core` — the hardware

`core.py`. A `Core` is specified the way the hardware is — by unit-crossbar geometry and an
operating model — and built as a fabric of neural lanes. It owns the control parameters (drive
voltages, low voltages, pulse width, read noise) and is addressed only by AATs.

### Constructor

```python
Core(crossbar_rows, crossbar_cols,
     spaces_per_lane=1, num_lanes=1,
     model="float", fidelity="ideal", init="medium",
     seed=None, profile=None,
     forward_voltage=None, reverse_voltage=None,
     forward_low_voltage=None, reverse_low_voltage=None,
     pulse_width=None,
     read_noise=READ_NOISE, noise_thermal=NOISE_THERMAL, noise_flicker=NOISE_FLICKER,
     temperature=ROOM_TEMPERATURE_K, read_noise_ref_m=None,
     comparator_enabled=True, comparator_code=COMPARATOR_CODE)
```

| Parameter | What it sets |
|---|---|
| `crossbar_rows`, `crossbar_cols` | Geometry of each unit crossbar (Nr × Nc devices per side). `1, 1` = a single device per side. |
| `spaces_per_lane` | Number of differential unit-crossbar pairs in each lane (one AAT entry per pair, so this is the AAT length). These are the kT-bits co-selected on a single evaluation — every space whose AAT entry is an address (not `None`) contributes to the readout; an entry of `None` disables that space for that evaluation. |
| `num_lanes` | Number of independent lanes in the core. |
| `model` | Device model: `"float"`, `"byte"`, `"mss"`, or `"rs"`. See below. |
| `fidelity` | Crossbar fidelity strategy. Only `"ideal"` is implemented (no sneak paths / line resistance). |
| `init` | Initial conductance distribution, e.g. `"medium"`, `"low"`, `"high"`, `"medium_noise"`. Full list below. |
| `seed` | Seed for the one RNG that covers init noise and MSS switching. `None` = entropy-seeded. Pass a seed for reproducible runs and figures. |
| `profile` | An `MSSProfile` (MSS model only); defaults to the idealized Knowm W+SDC profile. |
| `forward_voltage` / `reverse_voltage` | Standard drive voltages. `None` = model-aware default. |
| `forward_low_voltage` / `reverse_low_voltage` | Sub-threshold ("low") read voltages. `None` = model-aware default. |
| `pulse_width` | Write/update time `dt` in seconds. `None` = model-aware default. |
| `read_noise` | Master gain of the two **device** noise terms. The weight-referred σ is *quoted at the reference operating point* (room temp, reference read voltage and magnitude), **not** a fixed per-read value. Default `0.02`; `0` = a deterministic core. This is the device's **calibration**, not an operating dial — see "Read noise" below. |
| `noise_thermal` | Weight of the **thermal** (Johnson) noise term — the small voltage-dependent floor. Default `0.1`. |
| `noise_flicker` | Weight of the **flicker / RTN** (1/f) term — the dominant memristor read noise. Default `1.0`. |
| `temperature` | Operating temperature in kelvin (the `sqrt(T)` dependence of the thermal term). Default 298 K. |
| `read_noise_ref_m` | Reference magnitude for noise calibration. `None` = the model's own `GMAX`. |
| `comparator_enabled` | Whether the comparator that resolves the read is modeled. Default `True`. `False` gives the device-only law — that is a statement about the *model*, not a circuit setting, and it is the switch that reproduces pre-comparator results. |
| `comparator_code` | The comparator's 8-bit trim register, `0`–`255`. Default `10` (300 µV rms). Validated as an integer; a float raises. See "Read noise" below. |

Model-aware defaults (`MODEL_DEFAULTS`): float/byte/rs drive at ±1 with low voltages ±0.05; MSS
drives at ±0.25 (its switching threshold is ~0.27 V). RS uses `pulse_width=1e-8` so its old
`alpha*dt = 0.01` step reproduces; the others use `1e-6`.

Invalid `model` / `fidelity` / `init` names raise `KeyError` listing the valid choices.

### Methods

| Method | Purpose |
|---|---|
| `lane(index=0)` | Return the `NeuralLane` at `index`. |
| `evaluate(aat, instruction, lane_index=0, noise=0.0)` | Run one instruction on a lane; return `y`. Convenience wrapper over `lane.evaluate`. |
| `v_app(instruction)` | Signed applied voltage for an instruction (direction × standard/low). |
| `drive_voltage(direction)` | Standard drive voltage of a direction (used to set the feedback `Vy`). |
| `read_sample(y_clean, m, v_app)` | Apply read noise (thermal + flicker + comparator) to a clean read (see "Read noise" below). |
| `set_voltages(...)` | Change any of the four drive/low voltages at runtime. |
| `set_pulse_width(dt)` | Change the write/update time at runtime. |
| `set_read_noise(read_noise=None, noise_thermal=None, noise_flicker=None, temperature=None, read_noise_ref_m=None)` | Tune the **device** noise gains/temperature at runtime; `read_noise=0` disables read noise entirely, comparator included. Recomputes the cached coefficients. |
| `set_comparator(enabled=None, code=None)` | Set the comparator switch and/or its trim register at runtime. The register survives being switched off. |
| `set_comparator_noise(v_n, round_ok=False)` | Set the register from a requested rms voltage. Converts in integer microvolts; below the floor raises, and a level that is not an exact code raises unless `round_ok`. Returns the code set. |
| `comparator_noise` | *(property, read-only)* The register as volts, `0.0` when disabled. The register is the source of truth; this is derived. |
| `read_gab(lane_index, aat)` | Read back `(Ga, Gb)` for the enabled spaces of an AAT. **Debug/plot only.** Returns one tuple for a single enabled space, else a list. |
| `set_gab(lane_index, aat, ga, gb)` | Force `(Ga, Gb)` on the enabled spaces. **Debug/setup only.** Lets different device types start in the same state. |
| `set_start_y(lane_index, aat, y0, level=0.5)` | Place the enabled synapses at a target weight `y0 ∈ [-1, 1]`, identically across models regardless of conductance scale. `level` sets the pair magnitude. |

---

## `NeuralLane` — the 2-1 readout

`lane.py`. An ordered array of `UnitCrossbarPair`s read as a single 2-1 voltage divider. The
pairs are the AAT's addressable "spaces" (one AAT entry per pair). You normally get a lane from
`core.lane(...)` rather than constructing one directly.

- **`y`** — the retained activation from the last read. This is what the conditional feedback
  instructions (`FU`/`FA`/`RU`/`RA`) test via `H(y)`.
- **`evaluate(aat, instruction, noise=0.0)`** — run one instruction; return the (noisy) read `y`.
  - On a **read** instruction it computes `y = Σ(Ga−Gb)/Σ(Ga+Gb)` over the active spaces, stores
    it (with read noise applied to the *returned/retained* value), and drives the devices with
    the *clean* read so state evolution stays deterministic.
  - On a **feedback** instruction it forces `Vy` and drives the devices, leaving `y` unchanged.
  - `noise ∈ [0, 1]` dials the *read* voltage down toward zero, raising the weight-referred read
    noise (`σ_w ∝ 1/V_app`). `0` reads at the core's set read voltage (least noise); `1` drives
    it toward the floor (most noise). It affects reads only — feedback voltages are untouched.
  - An AAT whose length doesn't match the number of spaces raises `ValueError`.

---

## The instruction set

`instructions.py`. One instruction per `evaluate` call. Each `Instruction` is a frozen
dataclass with: `name`, `direction` (`forward`/`reverse`), `reads`, `low` (use sub-threshold
voltage), `coeff` (feedback `Vy` coefficient), and `use_H` (multiply feedback by `H(y)`).

A **read** computes `Vy` from the devices and sets the retained `y`. A **feedback** instruction
forces `Vy = coeff * Vdrive * (H(y) if use_H else 1)`, where `Vdrive` is the standard drive
voltage of the instruction's direction and `H(y) = +1 if y ≥ 0 else -1`.

| Name | Kind | Effect |
|---|---|---|
| `FF` | read | Forward read at standard voltage — the normal read. |
| `FFLV` | read | Forward read at the **low** (sub-threshold) voltage — non-disturbing / noisy read. |
| `RF` | read | Reverse read at standard voltage. |
| `RFLV` | read | Reverse read at the low voltage. |
| `FH` | forward feedback | `coeff=-1` |
| `FL` | forward feedback | `coeff=+1` |
| `FU` | forward feedback | `coeff=+1`, conditioned on `H(y)` |
| `FA` | forward feedback | `coeff=-1`, conditioned on `H(y)` |
| `FZ` | forward feedback | `coeff=0` (zero feedback) |
| `RH` | reverse feedback | `coeff=+1` |
| `RL` | reverse feedback | `coeff=-1` |
| `RU` | reverse feedback | `coeff=+1`, conditioned on `H(y)` |
| `RA` | reverse feedback | `coeff=-1`, conditioned on `H(y)` |
| `RZ` | reverse feedback | `coeff=0` |

You pass either the name string (`"FF"`) or an `Instruction` object. Unknown names raise
`KeyError`. The canonical learning cycle is a read followed by a feedback write, e.g.
`(FF, RH)` repeated drives `y` up to a rail; `(FF, RL)` drives it down. `FF`-then-no-feedback
is anti-Hebbian (`y → 0`); `RF`-then-no-feedback is Hebbian (`y → ±1`).

Two old Java instructions were dropped: `XX` (no-op — we run one instruction per call) and
`RCU` (branched on a state-change flag this port doesn't keep). `FFLV`/`RFLV` route through the
update path here, fixing the Java's `FFLV` early-return.

---

## `UnitCrossbar` / `UnitCrossbarPair` — the carriers

`unit_crossbar.py`. These are the hardware-native carriers of the memristors.

- **`UnitCrossbar(rows, cols, devices, fidelity)`** — an `Nr × Nc` array carrying individually
  selectable memristors. An address is a flat index in `[0, Nr*Nc)`; `None` is an open circuit.
  All contribution and drive route through the `CrossbarFidelity` strategy. Key members:
  `size`, `address_width`, `device_at(addr)`, `conductance(addr)`, `drive(addr, dV, dt)`.
- **`UnitCrossbarPair(a_side, b_side)`** — a differential pair = one signed synapse. The same
  address selects index `addr` in both sides, yielding `(Ga, Gb)`.
  - `conductances(addr)` → `(Ga, Gb)` (both `0` if `None`).
  - `drive(addr, dVa, dVb, dt)` — drive each side with its own voltage.

---

## `TwoOne` — the topology

`topology.py`. The fixed 2-1 readout and per-device update-voltage rule, kept in one place. It
is **not** an exposed constructor axis (you cannot build 1-2). Two static methods:

- `readout(pairs)` → `(top, bottom)` where `top = Σ(Ga−Gb)`, `bottom = Σ(Ga+Gb)`. Then
  `y = top/bottom ∈ [-1, 1]`.
- `update_voltages(v_app, vy)` → `(dVa, dVb) = (v_app − vy, vy + v_app)` — the per-device
  voltages applied to the two sides of the pair during a write.

---

## `CrossbarFidelity` — the no-corner rule

`crossbar/fidelity.py`. A strategy for how device contribution and drive resolve.

- **`Ideal`** (the only one implemented) — returns only the selected device's contribution and
  drives only the selected device. No network solve; `None` contributes nothing and receives no
  drive.
- `Physical` (floating-line sneak paths + line/terminal resistance) is a later phase and drops
  in here without touching call sites.

---

## Device models

`models/`. A device model owns one memristor's dynamics: its conductance `g()`, its current at a
voltage `current(V)`, and its state update `drive(dV, dt)`. Each model holds only its own physics
constants. Every model exposes a `create(mean, rand_var, rng, profile=None)` classmethod that
builds one device from an init `(mean, randVar)` pair, plus `set_g(g)` for forcing state.

Choose the model with the `Core(..., model=...)` argument.

| Model | `GMIN`–`GMAX` | RNG? | Time-dependent? | Character |
|---|---|---|---|---|
| `float` (`FloatDevice`) | `1e-7`–`1e-1` | no | no (`dt` ignored) | Conductance **is** the state. Symmetric ±0.25 dead-zone; sub-threshold `|dV|` → exactly zero change. Step `= learning_rate * dV`. Sweeps `y` to ±1. |
| `byte` (`ByteDevice`) | `1`–`255` | no | no (`dt` ignored) | Conductance quantized to an unsigned byte. Update is Java half-up `round(dV)`: 0 for `|dV|<0.5`, ±1 for `0.5≤|dV|<1.5`, … Pulse-up/down plateaus at ±0.5 (quantization ceiling). |
| `rs` (`RSDevice`) | `1e-7`–`1e-3` | no | **yes** (uses `dt`) | Resistive switch. Conductance pulled toward `GMax` (`dV>0`) or `GMin` (`dV<0`) through a thresholded transfer with a ±0.25 dead-zone. Deterministic. |
| `mss` (`MSSDevice`) | `1/Roff`–`1/Ron` | **yes** | **yes** (uses `dt`) | Mean Metastable Switch — **stochastic**. State `x ∈ [0,1]` is the fraction of `N` switches in the On state; conductance derived from `x`. Mean-field update (one Normal pair per step). Used for the random-bit-generation demo. |

### `MSSProfile` (MSS only)

A settable device profile — *not* hardcoded physics, so the model can fit measured devices.
Defaults are the idealized Knowm W+SDC device. Fields:

| Field | Default | Meaning |
|---|---|---|
| `Ron` / `Roff` | `1000` / `10000` | On/off resistance (Ω). |
| `N` | `1000` | Number of metastable switches. |
| `tau` | `1e-5` | Characteristic switching time (s). |
| `Von` / `Voff` | `0.27` / `0.27` | Off→on / on→off barrier potentials (switching threshold, V). |
| `phi` | `1.0` | Fraction of current from the MSS term (`1` ⇒ linear). |
| `schottky_fa/fb/ra/rb` | `0.0` | Schottky forward/reverse alpha/beta (nonlinear current term). |
| `temperature` | `298.0` | Device temperature (K). |

Derived: `gmin`, `gmax`, and `vt` (the thermal voltage `kT/q`, ~0.0257 V at 298 K).

### Initialization types (`init=`)

`INIT_TYPES` maps a name to `(mean, randVar)`, used as
`initG = GMin + (GMax−GMin)*(mean + randVar*gaussian())`, clamped to `[GMin, GMax]`. A noiseless
init (`randVar=0`) draws nothing from the RNG.

`low` (0, 0.05) · `medium` (0.5, 0.05) · `high` (1, 0.05) · `low_noise` (0, 0.25) ·
`high_noise` (1, 0.25) · `medium_noise` (0.5, 0.25) · `medium_high_noise` (0.5, 0.5) ·
`low_noiseless` (0, 0) · `medium_noiseless` (0.5, 0) · `low_noise_small` (0, 0.05).

---

## Read noise

Configured on the `Core` and applied in `read_sample`.
A read carries **three physically distinct noise mechanisms**, summed in quadrature and referred
to the weight (`y = Vy/V_app`):

```
σ_thermal    = read_noise * noise_thermal * sqrt(T / T_ref) * (V_ref / |V_app|) * sqrt(m_ref / m)
σ_flicker    = read_noise * noise_flicker * (1 − y²)        *                      sqrt(m_ref / m)
σ_comparator = v_cmp / |V_app|
σ_y          = sqrt(σ_thermal² + σ_flicker² + σ_comparator²)
```

The first two are the **device**, and `read_noise` is their calibrated gain. The third is the
**periphery**, and it is the only one a circuit designer sets. Each device term has its own
functional form and carries a tuning constant that folds in the parts the emulator does not model
from first principles (read bandwidth, 1/f corner, the Hooge factor).

**Thermal (Johnson–Nyquist)** — additive voltage noise over the signal. The node sees the two
memristors in parallel, so its thermal voltage noise is `4·k_B·T / (Ga+Gb)` per unit bandwidth;
referred to the weight that gives `1/|V_app|`, `sqrt(T)`, and `1/sqrt(m)`, and it is flat in `y`.
This is the small floor (`noise_thermal` default `0.1`).

**Flicker / RTN (1/f)** — multiplicative conductance fluctuation (`δG/G`), the *dominant* read
noise in real memristors. Propagating `δGa, δGb` through `y = (Ga−Gb)/(Ga+Gb)` gives a `(1 − y²)`
factor and **no `V_app` dependence**. This is the larger term (`noise_flicker` default `1.0`).

**Comparator** — the input-referred noise of the comparator that resolves the read. Nothing reads
a lane directly: a comparator decides every read, and every comparator has input-referred noise —
`kT/C` on the regeneration nodes plus the preamp's thermal noise (Razavi, *The StrongARM Latch*,
IEEE SSC Magazine 7(2), 2015). It is **periphery, not device**: flat in `y`, flat in `m`, and
**not** multiplied by `read_noise`, because `read_noise` is the device's gain and scaling this
term by it would make it a device term. It divides by `|V_app|` because the read is referred to
the weight. Comparator **offset** is a different quantity (Pelgrom matching), static per lane and
absorbed by learning; it is not part of this law.

Modeling a read *without* a comparator does not model an ideal comparator — it models the absence
of one, which no hardware can perform. So the term is **on by default**, and
`comparator_enabled=False` is the switch that reproduces pre-comparator results.

The level is an 8-bit trim register measured **up from a floor**, because no comparator reaches
zero input-referred noise:

```
v_cmp = COMPARATOR_V_MIN + comparator_code * COMPARATOR_V_STEP
      = 100 µV + code · 20 µV,   code ∈ [0, 255]   →   100 µV … 5.20 mV
```

Code `0` is the *quietest comparator modeled*, not the absence of one. The register is the source
of truth: `set_comparator_noise(v_n)` converts in integer microvolts and returns the code, and
`comparator_noise` reads back what was achieved. A request below the floor raises rather than
clamping — it asks for a design class this emulator does not model. The 20 µV step puts the levels
that matter on exact codes: 300 µV = 10, 1.00 mV = 45, 3.00 mV = 145.

| design class | input-referred noise | offset |
|---|---|---|
| raw dynamic latch | 0.5–2 mV rms | 5–20 mV |
| auto-zeroed | 50–200 µV rms | 0.1–1 mV |
| trimmed | tens of µV | — |

Trimmed comparators sit below the floor and are deliberately not representable: that is a
different design class at a different cost, and reaching it means moving `COMPARATOR_V_MIN`, which
is a visible edit rather than a silently available setting.

The knobs, in the order you'd actually reach for them:

1. **Read voltage `|V_app|` — the noise dial.** Lower it and the *thermal* term rises as
   `1/V_app` (the signal shrinks, the hiss does not). A sub-threshold read (below the dead zone)
   is non-disturbing, so `forward_low_voltage` + `FFLV`/`RFLV` is the knob for noisy,
   state-preserving reads — e.g. biased random-bit generation. The `noise ∈ [0,1]` argument to
   `evaluate` drives this same dial. **But the flicker term is flat in voltage**, so it sets a
   floor the dial cannot go below — the real device floor.
2. **Weight `y` — automatic.** The flicker term carries `(1 − y²)`: loudest when the synapse is
   undecided (`y = 0`), vanishing at the rails (`y = ±1`, a confident pair reads quiet).
3. **Magnitude `m`** — automatic. A confident (high-`m`) pair reads quietly, as `1/sqrt(m)` in
   both terms. `m` is the common-mode sum over the active pairs, so a multi-pair lane's noise
   scales with the total magnitude. Together the `(1 − y²)` and `1/sqrt(m)` factors track a
   Beta-posterior width, so a read's spread narrows toward certainty in both weight and
   magnitude — a read is a posterior-shaped sample.
4. **Temperature `T`** — the `sqrt(T)` dependence of the thermal term; settable, room-temp
   default.
5. **Comparator register `comparator_code` — the settable one.** Flat periphery noise, `1/|V_app|`
   like thermal but independent of `read_noise`, `T` and the pulse width. On real silicon its level
   is set by transistor size, capacitance and bias current, and it can be **raised at no cost**
   (starve the preamp bias, strobe the latch earlier, inject at the input). This is the dial a
   circuit actually has, and it is what makes the substrate usable as a sampler.

**`read_noise` is a calibration, not a dial.** It is the device's measured gain. The operating
knobs are the read voltage `V_app`, the read pulse width, and the comparator register — turning
`read_noise` up to raise a read's temperature asks the emulator for noise the modeled hardware
cannot produce.

`read_noise` is the **master gain**, quoted at the reference operating point (read voltage
`V_ref`, `m = m_ref`, `T = T_ref`, `y = 0`); `noise_thermal` and `noise_flicker` set the mix, so
at that point `σ_y = read_noise · sqrt(noise_thermal² + noise_flicker²)`. The constant part of
both terms is precomputed once (cached in `_recompute_noise_coeffs`, refreshed by
`set_read_noise`), so a read evaluates only the parts that change — `m`, `V_app`, and `y` — plus
one Gaussian draw.

Noise rides **only on the returned/retained value** (what `H()` and any threshold see). The
devices are driven by the clean read, so state stays deterministic and a sub-threshold read
stays exactly non-disturbing. `read_noise=0` (or `m≤0`) returns the clean read untouched and
draws nothing **regardless of the comparator** — that is the one deterministic test mode, and an
ideal device behind a noisy comparator is a nonphysical combination. A noise-disabled core
reproduces the deterministic output bit-for-bit. Output is clamped to `[-1, 1]`.

The mix and magnitude (`noise_thermal`, `noise_flicker`, and `read_noise` itself) are calibration
constants to be fit against measured devices; the defaults set the *shape*, not a validated noise
budget. At the reference read (`V_app` = 50 mV) the default register contributes `v_cmp/V = 0.006`
against the device's 0.02, raising σ by about 4.4%.

The torch modules carry the same law: `NoiseParams.sigma_unit` is **device only** — downstream
code reads it directly to build its own budgets — and `NoiseParams.sigma(y, m, T)` is the
composite. `sample_read(y, m, T, params, generator, comparator=None)` takes `comparator=False`
for a device-only draw at one call site, without disturbing params other readers share.

```python
# A noisy, non-disturbing read for biased random bits:
core = Core(1, 1, model="float", init="medium")          # read noise ON by default
core.set_voltages(forward_low_voltage=0.01)              # lower read voltage -> more thermal noise
y = core.lane(0).evaluate((0,), "FFLV")                  # sub-threshold: state barely moves
# or, equivalently, dial it per-call:
y = core.lane(0).evaluate((0,), "FF", noise=0.9)
# isolate one mechanism (e.g. study the flicker floor alone):
core.set_read_noise(noise_thermal=0.0, noise_flicker=1.0)
```

---

## `Neuron` — a partition (stub)

`neuron.py`. A view over a sub-window of a lane's spaces (`offset`, `size`) — a software view,
not hardware. It marks the seam for partitioned lanes; not exercised in Milestone 1.

---

## Worked example — the single-synapse lesson

The example helpers in `python/examples/_common/experiments.py` reproduce the canonical Knowm
Synapse lesson on this architecture. The pattern:

```python
from ktram_neural_core import Core

Z = (0,)

def execute_n(core, read, feedback, n, lane_index=0):
    """Repeat `evaluate(read); evaluate(feedback)` n times, recording y/Ga/Gb each step."""
    lane = core.lane(lane_index)
    ys, gas, gbs = [], [], []
    for _ in range(n):
        y = lane.evaluate(Z, read)            # read sets and returns y
        if feedback is not None:
            lane.evaluate(Z, feedback)        # feedback write (None = read-only)
        ga, gb = core.read_gab(lane_index, Z) # for plotting only
        ys.append(y); gas.append(ga); gbs.append(gb)
    return ys, gas, gbs

# Pulse up to a rail, then down:
core = Core(1, 1, model="byte", init="medium", seed=1, read_noise=0.0)
up = execute_n(core, "FF", "RH", 5000)        # y climbs
dn = execute_n(core, "FF", "RL", 5000)        # y falls
```

The lesson/figure helpers default `read_noise=0` so the static figures stay deterministic; pass
`read_noise=READ_NOISE` (or leave the default on) to exercise the noisy read. Other canonical
experiments in that file: the five FF-feedback combos, read decay-vs-growth (`FF` vs `RF`
read-only), low-voltage reads, the matched-weight/mismatched-magnitude "inertia" pair, and the
MSS random-bit-generation demo.

Runnable notebooks and scripts are in `python/examples/` (`synapse_review.ipynb`,
`compare_instructions.ipynb`, `iv_hysteresis.ipynb`, `generate_figures.py`). Tests are in
`python/tests/` — run `pytest` from `python/`.
