# basis-encoder — companion to "The Basis Encoder"

The unsupervised L1 recoder ([`BasisEncoder`](../../ktram_neural_core/aat_recoder/basis_encoder/)): a
winner-take-all group of neural lanes that learns a codebook from raw AATs with no labels, using the
same read and feedback instructions as the supervised RankCut classifier — target = the read winner
instead of a label. Two stabilizers keep the competition from collapsing: **exclusion** (a lane
already rewarded this cycle steps aside) and **recruitment** (idle lanes get pulled in). Withdraw
recruitment once the basis has formed and the codebook prunes itself.

## Files

**Synthetic track — the measurable experiment** (a source with a known basis, so recovery is measured
not eyeballed):

- `source.py` — the synthetic AAT source and the recovery metrics (`coverage`, `purity`,
  `utilization`, `entropy`), all read off one channel × generator win-count matrix.
- `shared.py` — the locked article config and a `train_and_eval` helper that also snapshots the
  win-count matrix for the forming animation.
- `experiments.py` — runs the experiments and caches results (data-gen separated from rendering):
  `ablation` (both on / exclusion off / recruitment off, 3 seeds), `sweep` (coverage/purity vs
  corruption), `separability` (a linear decoder on the raw AAT vs on the frozen basis code).
- `figures.py` — the static figures from the caches.
- `animation.py` — the win-count-matrix forming GIF and the animated ablation.

**Image track — the visual heroes** (Fashion-MNIST, fetched at runtime, no labels used):

- `fashion_mnist.py` — one WTA group over whole 28×28 images, full width (recruitment on): all lanes
  form a garment prototype.
- `fashion_mnist_prune.py` — the same, but two-phase: form with recruitment, then withdraw it so the
  bank self-prunes to the crisp subset the data supports.
- `fashion_patches.py` — a wider bank over small image patches, two-phase self-pruning: the survivors
  are local basis patches (edges, corners, curves) — a dictionary of primitives.
- `image_basis.py` — shared render for the two self-pruning image runs: a full-field forming GIF
  (losers fade on one global scale) and a survivors-only still.

## Run

```bash
python experiments.py all      # ablation + sweep + separability -> figures/*.pkl
python figures.py              # synthetic-*.png
python animation.py            # synthetic-forming.gif, synthetic-ablation.gif
python fashion_mnist.py        # fashion-forming.gif, fashion-features.png       (full width)
python fashion_mnist_prune.py  # fashion-prune-forming.gif, fashion-prune-features.png  (self-pruned)
python fashion_patches.py      # patch-forming.gif, patch-features.png           (patch dictionary)
```

Outputs land in the gitignored `figures/` dir. The synthetic source is generated in memory;
Fashion-MNIST caches to `~/scikit_learn_data` via `fetch_openml`, outside the repo. Nothing here
commits a dataset or a rendered image.

## What the experiments show (measured on the emulator; Alex validates)

- **Exclusion prevents collapse.** Off, one lane wins every pattern (utilization → ~0).
- **Recruitment protects coverage.** Off, fewer than half the lanes ever win (utilization ~0.46) and
  coverage falls from ~0.94 to ~0.60: the lanes that start badly never get going, so the patterns
  they should have claimed go unclaimed. The arm runs `abandon_action="reset"`, so the cycle still
  turns and the comparison isolates recruitment rather than a frozen run.
- **The unsupervised code is more linearly separable than the raw input** (raw ≈ 0.81 → basis ≈ 0.95
  on the synthetic 8-class source).
- **On Fashion-MNIST**, recognizable garment prototypes emerge with no labels; withdrawing recruitment
  after they form prunes the whole-image bank to ~6 crisp garments and the patch bank to ~26 local
  primitives, with the survivors read out by win count alone.
