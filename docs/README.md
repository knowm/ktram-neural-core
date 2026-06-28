# Documentation

The guide to `ktram-neural-core` — what it is, how it's organized, and how to run it. Start
here; the other files in this folder go deeper.

## What this project is

`ktram-neural-core` is a Python emulator of **kT-RAM** (Thermodynamic RAM) built as a **2-1
neural lane**. It runs the kT-RAM instruction set and experiments ahead of the physical
memristor-crossbar circuits it models, so hardware and emulator develop side by side. It is
built for **teaching and intuition** — it pairs with the Knowm blog series chapter by chapter —
not for production or large-scale EDA simulation.

## The mental model in one paragraph

A **memristor pair** read against each other is one signed synapse, the **kT-bit**. You read it
by applying a voltage and measuring the divided voltage that returns; the ratio is a weight `y`
in `[-1, 1]`. You write it by pulsing the two devices in opposite directions. **Reading the
memory is computing with it** — no separate fetch, ALU, or trip to DRAM. The emulator wraps that
one physical act (`evaluate`) in an instruction set and an addressing scheme, then stacks it into
**lanes** and a **Core**. Everything is selected by an **Activation Address Tuple (AAT)** — one
entry per address space in a lane; `None` disables a space.

```
Core                       geometry + operating model + control voltages; addressed only by AATs
 └─ NeuralLane[]           2-1 differential readout: y = Σ(Ga−Gb) / Σ(Ga+Gb)
      └─ UnitCrossbarPair[] a differential pair = one signed synapse (kT-bit); a lane's "spaces"
           ├─ UnitCrossbar  a-side — an Nr×Nc array of selectable memristors
           └─ UnitCrossbar  b-side — the differential partner
                └─ Device   one memristor's dynamics (float / byte / mss / rs)
```

## Start here

| You want to… | Read |
|---|---|
| Understand the classes and the instruction set | [architecture.md](architecture.md) |
| Install, run tests, run examples, run benchmarks | [running.md](running.md) |
| Know where things live and how to add to the repo without making a mess | [repo-structure.md](repo-structure.md) |
| Open a runnable lesson notebook | [../python/examples/](../python/examples/) |

## The layers, in dependency order (each builds only on the ones above)

1. **Core / lane / instructions** — the L0 emulator: geometry, the `evaluate` instruction set,
   the 2-1 readout, read noise. See [architecture.md](architecture.md).
2. **Device models** — `float`, `byte`, `mss`, `rs`; one memristor's dynamics each.
3. **encode / recode / classify** — data → AAT (`encode`), lane-`y` vector → AAT (`recode`), and
   one neural lane per label (`classify`). The supervised-learning layer.

## License

MIT grants copyright (see [`../LICENSE`](../LICENSE)). A separate [`../PATENTS`](../PATENTS) file
reserves Knowm's US hardware-patent rights: **software emulation needs no patent license;
hardware realization in the US requires a separate license.**
