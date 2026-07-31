"""Run the basis-encoder experiments and cache the results.

Data generation is separated from rendering: this script does the compute (slow) and writes a
pickle cache under the gitignored figures/ dir; figures.py and animation.py render from that cache
(fast), so styling iterates without recompute.

    python experiments.py ablation     # the 3-arm ablation, with animation snapshots
    python experiments.py sweep         # coverage/purity vs corruption
    python experiments.py all           # both
"""

import sys
import pickle
import pathlib

import numpy as np

from shared import ARMS, CONFIG, S, GATHER_ABANDON, train_and_eval

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)


def run_ablation(seeds=(0,), n_snapshots=24):
    """The 3-arm ablation. seeds[0] carries the snapshots (for the animation); extra seeds add
    metric spread for the headline table."""
    arms = {}
    for label, exc, rec in ARMS:
        per_seed = []
        snaps = None
        for i, seed in enumerate(seeds):
            r = train_and_eval(exc, rec, seed=seed,
                               n_snapshots=(n_snapshots if i == 0 else 0))
            per_seed.append(r["metrics"])
            if i == 0:
                snaps, final_matrix, k, channels = r["snaps"], r["matrix"], r["k"], r["channels"]
            print(f"  {label:16s} seed={seed} {r['metrics']}", flush=True)
        keys = per_seed[0].keys()
        mean = {kk: float(np.mean([d[kk] for d in per_seed])) for kk in keys}
        std = {kk: float(np.std([d[kk] for d in per_seed])) for kk in keys}
        arms[label] = dict(mean=mean, std=std, per_seed=per_seed,
                           snaps=snaps, matrix=final_matrix, k=k, channels=channels)
    with open(OUT / "ablation.pkl", "wb") as f:
        pickle.dump(dict(arms=arms, config=CONFIG, S=S, gather_abandon=GATHER_ABANDON,
                         seeds=list(seeds)), f)
    print("wrote", OUT / "ablation.pkl")


def run_sweep(corruptions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6), seed=0,
              arms=(("both on", True, True), ("recruitment off", True, False))):
    """Coverage / purity / utilization vs corruption, one seed (a search, so one seed per the
    compute-economy norm). Defaults to both-on plus recruitment-off, so the purity gap is visible
    across the difficulty range without paying for the collapsed exclusion-off arm at every point."""
    rows = []
    for corr in corruptions:
        cfg = dict(CONFIG, corruption=corr)
        point = {"corruption": corr}
        for label, exc, rec in arms:
            r = train_and_eval(exc, rec, config=cfg, seed=seed)
            point[label] = r["metrics"]
            print(f"  corr={corr} {label:16s} {r['metrics']}", flush=True)
        rows.append(point)
    with open(OUT / "sweep.pkl", "wb") as f:
        pickle.dump(dict(rows=rows, config=CONFIG, S=S), f)
    print("wrote", OUT / "sweep.pkl")


def _onehot(aats, sizes):
    import numpy as np
    offs = np.cumsum([0] + list(sizes))
    X = np.zeros((len(aats), offs[-1]), dtype=np.float32)
    for r, a in enumerate(aats):
        for s, ch in enumerate(a):
            if ch is not None:
                X[r, offs[s] + ch] = 1.0
    return X


def run_separability(n_groups=4, channels=48, epochs=4, n_train=4000, n_test=3000, seed=0):
    """The separability lift: a linear decoder on the raw input AAT vs on the frozen basis code.

    Train a bank of BasisGroups unsupervised, freeze it, encode train/test to the tuple-of-winners
    code, and fit a plain LogisticRegression on each representation. Unsupervised feature learning
    should make the classes more linearly separable (basis code > raw input)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from ktram_neural_core import BasisEncoder
    from source import AATSource

    src = AATSource(seed=seed, **CONFIG)
    enc = BasisEncoder(src.space_sizes, n_groups=n_groups, channels=channels,
                       gather_abandon=channels // 2, seed=100 + seed)
    tr_aats, _, tr_y = src.batch(n_train)
    te_aats, _, te_y = src.batch(n_test)
    for _ in range(epochs):
        for a in tr_aats:
            enc.adapt(a)
    # raw one-hot vs frozen basis code
    Xtr_raw, Xte_raw = _onehot(tr_aats, src.space_sizes), _onehot(te_aats, src.space_sizes)
    tr_code = [enc.read(a) for a in tr_aats]
    te_code = [enc.read(a) for a in te_aats]
    Xtr_b, Xte_b = _onehot(tr_code, enc.out_space_sizes), _onehot(te_code, enc.out_space_sizes)

    def fit_score(Xtr, Xte):
        clf = LogisticRegression(max_iter=2000).fit(Xtr, tr_y)
        return accuracy_score(te_y, clf.predict(Xte))

    raw, basis = fit_score(Xtr_raw, Xte_raw), fit_score(Xtr_b, Xte_b)
    out = dict(raw=raw, basis=basis, n_groups=n_groups, channels=channels, n_labels=src.n_labels)
    print(f"  linear decoder: raw={raw:.3f}  basis={basis:.3f}  ({src.n_labels} classes)", flush=True)
    with open(OUT / "separability.pkl", "wb") as f:
        pickle.dump(out, f)
    print("wrote", OUT / "separability.pkl")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "ablation"
    if what in ("ablation", "all"):
        print("== ablation ==")
        run_ablation(seeds=(0, 1, 2))
    if what in ("sweep", "all"):
        print("== sweep ==")
        run_sweep()
    if what in ("separability", "all"):
        print("== separability ==")
        run_separability()
