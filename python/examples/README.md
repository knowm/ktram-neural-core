# Examples

Runnable lessons for [`ktram-neural-core`](https://pypi.org/project/ktram-neural-core/).
Each folder is one lesson: a Colab-ready notebook plus the script that regenerates that
lesson's article figures. Shared plumbing is in `_common/`.

## Layout

```
examples/
  _common/               # shared, examples-only helpers (NOT part of the package)
    experiments.py       #   single-pair experiment helpers (execute_n, single_synapse_core, ...)
    plotting.py          #   matplotlib helpers (plot_synapse, rails)
  single-synapse/        # Milestone 1 — the canonical Synapse lesson
    synapse_review.ipynb
    figures.py
  kt-bit/                # companion to "Chapter 3b: The kT-bit Up Close"
    kt_bit.ipynb
    figures.py
  neural-lane-emulator/  # companion to "Chapter 4b: The Neural Lane Emulator"
    figures.py
  ahah-attractors/       # companion to "Chapter 5: AHaH Attractors"
    ahah_attractors.ipynb
    figures.py
  iris-classifier/       # benchmark — encoders + linear classifier on Iris
    benchmark.py         #   text report: our lane vs reference linear models, same encoding
    shared.py            #   shared comparison logic
    figures.py
  instructions/
    compare_instructions.ipynb
  device-physics/
    iv_hysteresis.ipynb
```

New lessons get a new topic-named folder (not a number — they track topics, not publish
order, so an add-on can slot in without renumbering anything). Put shared code in `_common/`;
keep lesson-specific code in the lesson folder.

## Running

- **Notebooks** are self-contained: they `pip install ktram-neural-core` and inline their own
  helpers, so they run on Colab (badge at the top of the notebook) with nothing installed, or
  locally under `jupyter lab`. They do *not* import `_common` — that keeps Colab a one-click
  open with no repo checkout.
- **Figure scripts** import the shared helpers from `_common` and add the repo to the path
  themselves, so no install is needed. Each defaults to **its own `figures/` subdir**
  (gitignored); pass a path to write elsewhere (e.g. a website article folder):

  ```bash
  python examples/single-synapse/figures.py             # writes examples/single-synapse/figures/
  python examples/kt-bit/figures.py /path/to/article    # writes into that folder instead
  ```

## Colab

| Lesson | Notebook |
|---|---|
| The kT-bit | [Open in Colab](https://colab.research.google.com/github/knowm/ktram-neural-core/blob/main/python/examples/kt-bit/kt_bit.ipynb) |
| AHaH Attractors | [Open in Colab](https://colab.research.google.com/github/knowm/ktram-neural-core/blob/main/python/examples/ahah-attractors/ahah_attractors.ipynb) |
