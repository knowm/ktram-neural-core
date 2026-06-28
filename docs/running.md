# Running the code

Install, the quickstart, the tests, the example figures and notebooks, and the benchmarks.

## Install

```bash
pip install ktram-neural-core            # once published
# or, from a checkout:
pip install -e python/                    # library only (NumPy)
pip install -e "python/[examples,dev]"    # + matplotlib, scikit-learn, pytest
```

The library depends on **NumPy only**. Plotting (`matplotlib`) and the benchmark loaders
(`scikit-learn`) are the `examples` extra; `pytest` is the `dev` extra.

## Quickstart — a single synapse

```python
from ktram_neural_core import Core

core = Core(1, 1, spaces_per_lane=1, num_lanes=1, model="byte", init="medium")
lane = core.lane(0)
z = (0,)                       # the AAT: address 0 in the one space

y = lane.evaluate(z, "FF")     # read; sets and returns y
    lane.evaluate(z, "RH")     # write; one instruction per call
ga, gb = core.read_gab(0, z)   # debug/visualization only
```

The four device models — `float`, `byte`, `mss`, `rs` — are all available; drive voltages and
pulse width are model-aware Core defaults and fully settable (`set_voltages`, `set_pulse_width`).

## Tests

```bash
cd python && pytest
```

The suite asserts the **mechanism** (hand-checked instruction math, read-noise law, the
classifier learning above chance) at a fixed seed with `read_noise=0`. It does not assert
accuracy bars or congruence with the original Java stack — that validation is done separately.

## Examples

Each folder under [`python/examples/`](../python/examples/) is one lesson: a runnable
`figures.py` and, for most, a Colab-ready notebook. See
[`python/examples/README.md`](../python/examples/README.md) for the catalog and Colab links.

```bash
cd python
python examples/single-synapse/figures.py     # writes examples/single-synapse/figures/ (gitignored)
python examples/single-synapse/figures.py PATH # write somewhere else (e.g. a website article folder)
```

Every `figures.py` defaults to **its own `figures/` subdir** (gitignored). Pass a path argument
to write elsewhere. Notebooks are self-contained (they `pip install` the package and inline their
helpers) so they open on Colab with no checkout.

## Benchmarks

Benchmarks **report** results; they do not gate on them. The headline metric is test accuracy
with a confusion matrix, deterministic at a fixed seed (`read_noise=0`).

```bash
cd python
python examples/iris-classifier/benchmark.py   # accuracy + confusion matrix, ours vs linear baselines
python examples/iris-classifier/figures.py      # the blog figures for this benchmark
```

### Data is never committed

The repo carries **code, not corpora**.

- **Bundled sets** (`load_iris`, `load_digits`, `load_wine`, `load_breast_cancer`) ship inside
  installed `scikit-learn` — they download nothing and write nothing to the working tree.
- **Downloaded sets** (`fetch_openml`, `fetch_*` — MNIST, etc.) cache to `~/scikit_learn_data/`,
  outside the repo. Leave `data_home` at its default; never point it into the repo.
- As a guardrail, [`.gitignore`](../.gitignore) ignores `data/`, `*.csv`, `*.npy`, `*.npz`. If a
  future benchmark needs a set sklearn cannot fetch, download it at runtime into an ignored
  `data/` dir — never check it in.

### Adding a benchmark

1. Create `python/examples/<dataset>-classifier/benchmark.py`.
2. Load via a sklearn loader (bundled) or `fetch_*` (home cache) — never add a data file.
3. Build the encoder (`encode/`), fit a `LinearClassifier`, read out with a `recode/` recoder.
4. Print accuracy + confusion matrix; keep it deterministic (fixed seed, `read_noise=0`).
