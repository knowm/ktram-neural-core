# rank-cut-recoder

The **RankCut** L1 AAT Recoder, verified on Iris against the `iris-classifier` example.

`demo.py` runs RankCut on the **same Iris pipeline** as `examples/iris-classifier` and shows it
**reproduces that example's kT-RAM lane result exactly**. RankCut is the iris-classifier's
`LinearClassifier` + `Winner` recoder repackaged as one L1 recoder behind a clean interface:

- `recoder.adapt(in_aat, teach={label})` — learn (adapting `FF` read + the `RH`/`RL`/`RF` routine)
- `recoder.read(in_aat)` — infer (non-disturbing `FFLV` read, recoded to an output AAT)

Same frozen encoder, an identically-seeded core, the same training order, and the Winner readout
expressed as `(Vt=-inf, N=1)` → **identical predictions**, then placed in the iris-classifier
accuracy table next to the reference linear models. All the analog stays inside the recoder; you
hand it AATs and get AATs back. Verification by reproduction.

```
python demo.py        # needs the repo venv (sklearn), same as iris-classifier
```

The recoder lives in `ktram_neural_core/aat_recoder/rank_cut/`.
