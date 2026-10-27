# Assimilation of a MLP

Companion to [Chapter 8: Assimilation of a MLP](https://knowm.ai/blog/assimilation-of-a-mlp/).
The chapter is a draft; its URL becomes public when the article is published.

Train a conventional image classifier, freeze it, and teach three kT-RAM
`Classifier` banks to approximate its behavior. Compare fitted hidden activation
codes with interfaces learned through an additive score proxy. Every synaptic
update uses `Classifier.adapt`; proxy tables are never copied into the banks.
These are emulator experiments, not physical hardware measurements.

## Run from a checkout

From the repository root, in a Python environment:

```bash
python -m pip install -e './python[torch,examples]' 'scikit-learn>=1.2'

# Short offline workflow check, using sklearn's bundled 8x8 digits dataset.
python python/examples/mlp-assimilation/experiment.py --smoke

# Full Fashion-MNIST experiment, including both interface arms.
python python/examples/mlp-assimilation/experiment.py

# Additional control: keep the same input encoder in both arms.
python python/examples/mlp-assimilation/experiment.py --freeze-input
```

Fashion-MNIST downloads through `fetch_openml` on first use and uses the standard
scikit-learn cache. The smoke run needs no network. It uses 512/128/128 images for
training/validation/test, two epochs per gradient phase and one per lane stage.
Its accuracy is a workflow check, not a Fashion-MNIST result. `--smoke` overrides
counts, schedules and batch size; other settings still apply.

Defaults: 784 → 64 → 32 → 10 ReLU teacher; student interfaces
16×64 → 32×16 → 16×16 → ten classes, **657,920 synapses**. The digital encoder
contains sixteen independent 49 → 32 → 64 networks, each confined to one 7×7
patch: 59,392 parameters and 57,856 MACs per image.

The full run uses 40 teacher epochs, 30 encoder warm-start epochs, 40 learned-proxy
epochs, and 20/10/10 teaching epochs in **each** arm. Use `--help` for sample
counts, epochs, seed, CPU threads and batch size. Runs execute sequentially.

## Experiment

1. Train the teacher on raw pixels and class labels; freeze it. Harvest predicted
   classes and the two hidden layers' pre-ReLU activations.
2. Fit one `AATCodec` per hidden activation space, using training images only.
   Train a local input encoder through an additive reader to predict the fitted
   first-layer codes. This is the common encoder warm start.
3. Teach the **fitted-interface arm**: the warm encoder supplies input symbols;
   the fitted books supply hidden targets. Train fresh banks stage by stage.
4. Starting from the same warm encoder, learn an end-to-end **proxy** with hard
   symbols. Loss = cross-entropy to teacher predictions + 0.3 × normalized hidden
   probe MSE − batch usage entropy, summed over the three interfaces. Temperature
   falls from 2.0 to 0.5; a straight-through estimator supplies gradients.
5. Freeze the proxy, harvest hidden codes and teach the **learned-interface arm**
   with fresh banks. The head still targets the original teacher's predictions.

After each bank is selected, the next receives its **actual emitted codes**,
including mistakes. Its targets remain the clean fitted/proxy codes for the
original image. Neither student uses training-set class labels directly.

By default, proxy training also refines the input encoder. The comparison includes
that refinement, hidden interface learning and end-to-end training. `--freeze-input`
holds the common encoder's parameters fixed. A generic nonlinear proxy and a
teaching-sample sweep would be further controls for a claim about local acquisition.

## Split and selection

Fashion-MNIST's 70,000 OpenML images are pooled and permuted with
`numpy.random.default_rng(seed)`. The first 32,000 train all components; the next
2,000 are validation; the last 2,000 are test. The rest are unused. These sets are
disjoint. This is a custom split, **not the official 60,000/10,000 benchmark split**.
At seed zero, training and test indices match the historical experiments, with a
separate validation set added.

The teacher checkpoint maximizes validation class accuracy. The encoder maximizes
validation agreement with fitted first-layer codes; the proxy maximizes validation
agreement with teacher predictions. Each lane stage maximizes validation agreement
with its targets. Ties retain the earlier checkpoint. Both students finish before
test evaluation. Use validation, rather than the test report, to choose settings.

## Lane semantics

For output slot `m` with `S` symbols, candidate `s` occupies flat lane `m*S+s`.
The target AAT contains one such lane index per output slot. `lane_codes` reshapes
`read_y` scores to `[batch, output_slots, alphabet]` and chooses one winner per
slot, including when every score is negative. Global `Classifier.read()` has a
different recoding policy.

`adapt(..., feedback="hard")` applies FF followed by RH/RL/RF. With a batch,
readings are stale within that update. Batch 256 matches the historical software
method; it is not 256 serial device updates. `--batch 1` uses the serial rule and
changes both learning dynamics and run time.

The proxy adds freely learned table entries over active input symbols. Real lane
scores divide summed conductance differences by summed conductance magnitudes.
The proxy does not constrain that denominator or simulate adaptation. Its achieved
accuracy is a reference, not an exact representation guarantee or a proven upper
bound on the taught network.

## Outputs and inference

An ignored `figures/<dataset>-<hash>/` directory contains:

- `results.json`: settings, source and split index hashes, versions, curves,
  selected epochs, accuracy, teacher agreement, stage agreement, usage and costs.
- `models.pt`: teacher, warm start, fitted book buffers, proxy and student state
  dictionaries. Load only checkpoints you trust.
- Six `.npz` inference packs, one per bank in each arm. Packing can change
  predictions; the report measures packed accuracy and live/packed agreement.
- `run-results.png`: validation teaching curves and final test comparison.

Completed runs are reused under the same settings, source hash and recorded library versions.
Incomplete runs restart. Remove a run directory to repeat that configuration.
No data, weights or generated run outputs need to be committed.

From this example directory, after installing the package:

```python
from experiment import load_student

student = load_student("figures/fashion-mnist-<hash>", arm="learned")
predicted_classes = student(images)  # [batch, 1, 28, 28] floats in [0, 1]
```

Inference needs only the encoder and taught banks. The run checks that reloading
the saved student exactly reproduces its live predictions.

Costs include digital encoder parameters and synaptic payload. An inference pack
uses two bytes per synapse for quantized difference/magnitude values, plus scales
and metadata. Live `ga`/`gb` buffers use two **int32** tensors, eight bytes per
synapse, before read caches and activations. MACs count the digital encoder only;
lane work is additional. Payload is not chip area or total Python runtime memory.

## Verified companion run

`reference-run.json` is the complete output of the default Fashion-MNIST command,
run while preparing Chapter 8. It includes all curves and checkpoint selections.
Environment: Python 3.11.7, PyTorch 2.2.2, NumPy 1.26.4, scikit-learn 1.9.0,
eight CPU threads. The full run took about 26 minutes on the development machine.
The record's `versions` field is authoritative for the exact environment.

| Model | Test accuracy | Agreement with teacher |
|---|---:|---:|
| Frozen teacher | 88.35% | 100% |
| Learned hard-symbol proxy | 85.80% | 90.30% |
| Taught fitted-interface chain | 78.90% | 82.50% |
| Taught learned-interface chain | 84.90% | 89.45% |

Both taught arms use 657,920 synapses and the same 20/10/10 teaching budget. The
learned arm refines the input encoder. The teacher checkpoint was selected at
epoch 35 and the proxy at epoch 11, using validation only. This run uses the
historical training/test image identities with a separate validation set added.

Saved-student reload reproduced live predictions exactly. Packed accuracy was
78.70% for the fitted chain and 84.90% for the learned chain; their predictions
agreed with their live versions on 99.55% and 99.85% of test images, respectively.
The three pack files occupy 1,325,494 bytes per arm, including pack overhead and
excluding the digital encoder.

Render this run's figure without training:

```bash
python python/examples/mlp-assimilation/figures.py --results python/examples/mlp-assimilation/reference-run.json
```

## Recorded development results

`historical-results.json` transcribes the August 2026 development runs and supplies
the chapter's plots. It contains no weights or private file dependencies. These
measurements used one seed and a repeatedly inspected 2,000-image evaluation set,
also used for configuration selection. They are not a fresh confirmatory test.

| Run | Proxy accuracy | Taught accuracy | Synapses |
|---|---:|---:|---:|
| A3, fitted interfaces | — | 78.60% | 657,920 |
| A4, learned interfaces | 85.75% | 85.40% | 657,920 |
| A6 micro | 85.30% | 84.60% | 83,200 |
| A7, larger encoder | 87.45% | 86.85% | 83,200 |

Teacher accuracy was 88.05%. A4 refined A3's encoder and used 20 first-stage
teaching epochs versus A3's 10. A6/A7 changed interfaces and used 60 proxy epochs.
A7's complete parameter payload was **799,232 bytes**, including its digital
encoder, versus the teacher's 210,600 bytes.

The public script implements the A3/A4 method with validation selection and equal
lane teaching budgets. It is **not a bit-for-bit historical reproduction**:
initializations and checkpoint choices differ. The A6/A7 sweeps are recorded
context, not extra CLI presets. Use repeated seeds and a new final holdout before
making confirmatory claims in a paper.

Regenerate article figures without training or network access:

```bash
python python/examples/mlp-assimilation/figures.py /path/to/article
python python/examples/mlp-assimilation/figures.py --results /path/to/results.json
```

Mechanism checks, with no accuracy thresholds:

```bash
python -m pip install pytest
python -m pytest python/examples/mlp-assimilation/test_mechanism.py
```
