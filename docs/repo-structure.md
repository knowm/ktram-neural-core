# Repo structure & conventions

How the repo is laid out, and the rules for adding to it without making a mess. The project is
built to grow; these conventions keep it legible as it does.

## The tree

```
kt-ram-neural-core/
├── README.md            public front page (install, quickstart, license)
├── docs/                this documentation  ← you are here
├── planning/            private design specs (git-ignored; not published)
├── LICENSE  PATENTS     MIT copyright + reserved US hardware-patent rights
└── python/
    ├── pyproject.toml   package metadata; numpy core, [examples]/[dev] extras
    ├── ktram_neural_core/   the library (the only installed code)
    ├── examples/        runnable lessons + benchmarks (NOT part of the package)
    └── tests/           pytest suite
```

## The library — `python/ktram_neural_core/`

Dependency direction is **strictly downward onto the Core**. A module may use the layers above
it in this list, never below:

```
core.py · lane.py · instructions.py · topology.py · unit_crossbar.py   the L0 core
models/        device dynamics: float, byte, mss, rs
crossbar/      crossbar fidelity strategy
encode/        data  → AAT      (AATEncoder; A2DEncoder, ConstantEncoder, compose)
recode/        lane-y → AAT     (AATRecoder; Winner, AboveZero, WinnerAboveZero)
classify/      one neural lane per label (LinearClassifier)
```

Rules:
- **The Core never imports upward.** `encode`/`recode`/`classify` build on the Core; the Core
  knows nothing about them.
- **The library is NumPy-only.** Anything needing matplotlib or scikit-learn belongs in
  `examples/`, not here.
- **Export new public types** from [`ktram_neural_core/__init__.py`](../python/ktram_neural_core/__init__.py).
- New subpackages are siblings under `ktram_neural_core/` with a thin `base.py` + concrete
  implementations, mirroring `encode`/`recode`.

## Examples — `python/examples/`

One folder per lesson, **named for its topic, not numbered** (topics track concepts, not publish
order, so a new lesson slots in without renumbering). Each folder holds:

- `figures.py` — regenerates that lesson's figures. **Defaults to its own `figures/` subdir**
  (git-ignored); pass a path argument to write elsewhere (e.g. a website article folder). Never
  hardcode an absolute path.
- a Colab-ready `.ipynb` (most lessons) — self-contained: it `pip install`s the package and
  inlines its helpers, so it does **not** import `_common`. Keep it that way.
- benchmark folders (`*-classifier/`) additionally hold `benchmark.py` (the text report) and may
  hold a `shared.py` for logic shared between the benchmark and its figures.

Shared, examples-only helpers live in `_common/`. Lesson-specific code stays in the lesson folder.

> **Notebooks are load-bearing.** They back published blog chapters and run on Colab. Treat them
> as fixed unless a change is explicitly about them.

## Tests — `python/tests/`

`pytest`, one file per area (`test_core_math`, `test_models`, `test_read_noise`, `test_encode`,
`test_recode`, `test_classifier`, …). Tests assert the **mechanism** at a fixed seed with
`read_noise=0` for determinism. They do not assert accuracy bars or Java congruence.

## Where things go — quick reference

| Adding… | Put it in |
|---|---|
| A device model | `ktram_neural_core/models/` (register in its `__init__`) |
| An encoder / recoder / classifier | the matching `encode`/`recode`/`classify` subpackage |
| A lesson or benchmark | a new topic-named folder under `examples/` |
| Shared example helpers | `examples/_common/` |
| Generated figures | the lesson's own `figures/` subdir (git-ignored) — never committed |
| A dataset | nowhere — load it at runtime; data is never committed (see [running.md](running.md)) |
| Design notes / specs | `planning/` (private) or `docs/` (public) |
