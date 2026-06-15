# ktram-neural-core

**kT-RAM** — Thermodynamic RAM — is a construct: addressable memristor-pair synapses
(kT-bits) driven by a small instruction set, where reading the memory *is* computing with it.
No separate fetch, no ALU, no trip out to DRAM. This package emulates one kT-RAM topology —
the **2-1 neural lane** — in Python. It is the software half of a build: the emulator runs the
instruction set and the experiments ahead of the physical memristor-crossbar circuits it
models, so the two can be developed side by side.

The emulator is built for teaching and intuition, not production. It pairs with the Knowm blog
series that works through the kT-RAM instruction set chapter by chapter — start with
[Chapter 4: The Neural Lane](https://knowm.ai/blog/building-the-neural-lane/) and its companion
[Chapter 4b: The Neural Lane Emulator](https://knowm.ai/blog/the-neural-lane-emulator/). It is
not an EDA component and it is not meant for large-scale simulation.

The emulator's internal structure mirrors the hardware it models — unit crossbars, differential
pairs, neural lanes — selected by Activation Address Tuples (AATs), rather than an idealized
flat-synapse construct.

```
Core
 └─ NeuralLane[]                      2-1 differential readout: y = Σ(Ga−Gb) / Σ(Ga+Gb)
      └─ UnitCrossbarPair[]           a differential pair = one signed synapse (kT-bit)
           ├─ UnitCrossbar  a-side    carries the selected memristor (Ideal fidelity)
           └─ UnitCrossbar  b-side    the differential partner
```

A **Core** is specified the way the hardware is — by unit-crossbar geometry and operating model —
and addressed only by AATs. A single synapse is the smallest lane.

## Install

```bash
pip install ktram-neural-core        # once published
# or, from a checkout:
pip install -e python/
```

Runtime dependency is NumPy only. Plotting (`matplotlib`) is an examples-only extra:
`pip install -e "python/[examples,dev]"`.

## Quickstart — a single synapse

```python
from ktram_neural_core import Core

core = Core(1, 1, spaces_per_lane=1, num_lanes=1, model="byte", init="medium")
lane = core.lane(0)
z = (0,)                              # the AAT: address 0 in the one space

y = lane.evaluate(z, "FF")           # read; sets and returns y
    lane.evaluate(z, "RH")           # write; one instruction per call
ga, gb = core.read_gab(0, z)         # debug/visualization only
```

The four device models — `float`, `byte`, `mss`, `rs` — are all available. Drive voltages
and pulse width are model-aware Core defaults and fully settable
(`set_voltages(...)`, `set_pulse_width(dt)`).

## Examples — the single-synapse lesson

Reproduces the canonical Knowm Synapse lesson on the new architecture. See
[`python/examples/`](python/examples):

- `synapse_review.ipynb` — Colab-ready notebook reproducing the lesson per device type.
- `compare_instructions.ipynb` — runs the **same instruction sequence across all four
  types** (pulse, FF-RH/RL, FF-RU/RA, FF-RZ, FF-XX/RF-XX, low-voltage), overlaying the
  normalized `y` from matched initial conditions.
- `iv_hysteresis.ipynb` — drives MSS/RS devices with a sinusoidal sweep and plots the
  pinched I-V hysteresis loop vs frequency; the device-physics parameters are the handles
  for fitting the model to measured devices.
- `generate_figures.py` — writes the headline PNGs (Byte/Float pulse, RS, MSS RNG).
- `synapse_experiments.py` — the experiment matrix as reusable functions.

```bash
cd python && pip install -e ".[examples,dev]"
python examples/generate_figures.py     # -> python/examples/figures/*.png
pytest                                    # 26 unit + behavioral tests
```

Byte pulse-up/down plateaus at ±0.5 (the quantization ceiling); Float sweeps to ±1; FF is
anti-Hebbian and RF is Hebbian; the MSS RNG demo digitizes to a fair bit. Final validation
against the lesson figures is by eye.

## License

MIT (see [`LICENSE`](LICENSE)) grants copyright. A separate [`PATENTS`](PATENTS) file reserves
Knowm's US hardware-patent rights: **software emulation is permitted with no patent license;
hardware realization in the US requires a separate license.**
