"""Iris benchmark — our kT-RAM neural-lane classifier vs reference linear classifiers, ALL on
the same encoded AATs.

    python examples/iris-classifier/benchmark.py

An AAT encoder is a feature transform, so the fair test of "is our lane as good as a linear
classifier?" runs both on the identical frozen AAT encoding (see shared.py). This script reports,
for one headline split and then across many seeds:

  * LogReg (raw)        — a linear classifier on the raw 4 features (the un-encoded baseline)
  * LogReg (AAT)        — the same linear classifier on the one-hot AAT encoding
  * LinearSVC (AAT)     — a second reference linear model on the AAT encoding
  * kT-RAM lane (AAT)   — our hardware-native online classifier on the AAT encoding

The gap between LogReg(raw) and LogReg(AAT) shows how the encoding affects a linear classifier
(it can help or hurt); the gap between LogReg(AAT) and the kT-RAM lane is our online rule vs a
batch linear solver on the same features.

The numbers are reported, not gated — whether they are "good" is a separate judgement.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))      # local shared.py
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # python/ on path

from sklearn.metrics import accuracy_score, confusion_matrix          # noqa: E402

import shared                                                          # noqa: E402

SEED = 0
SEEDS = range(20)     # the multi-seed sweep (45 test points per split is coarse on its own)


def _confusion(title, y_true, preds, names):
    cm = confusion_matrix(y_true, preds)
    print(f"\n{title}  (rows = true, cols = predicted)")
    print("          " + "".join(f"{n[:10]:>12}" for n in names))
    for i, row in enumerate(cm):
        print(f"{names[i][:10]:>10}" + "".join(f"{v:>12}" for v in row))


def main():
    print(f"Iris — A2D(bits={shared.BITS}, l={shared.LRATE}) + bias, "
          f"encoder {shared.ENCODER_EPOCHS} epochs then frozen, classifier {shared.EPOCHS} "
          f"epochs, {shared.MODEL}/{shared.INIT}, read_noise=0\n")

    # --- headline split ---
    r = shared.run_once(SEED)
    names = r["data"].target_names
    n_te = len(r["y_te"])
    print(f"single split (seed={SEED}, {n_te} test points):")
    for name in shared.METHOD_ORDER:
        acc = accuracy_score(r["y_te"], r["preds"][name])
        print(f"  {name:<22} {acc:.3f}  ({int((r['preds'][name] == r['y_te']).sum())}/{n_te})")

    _confusion("kT-RAM lane (AAT)", r["y_te"], r["preds"][shared.OURS], names)
    _confusion("LogReg (AAT)", r["y_te"], r["preds"]["LogReg (AAT)"], names)

    # --- multi-seed sweep ---
    accs = shared.multi_seed(SEEDS)
    print(f"\nacross {len(list(SEEDS))} seeds — test accuracy mean ± std  (min … max):")
    for name in shared.METHOD_ORDER:
        a = accs[name]
        print(f"  {name:<22} {a.mean():.3f} ± {a.std():.3f}   ({a.min():.3f} … {a.max():.3f})")

    gap_enc = accs["LogReg (AAT)"].mean() - accs["LogReg (raw)"].mean()
    gap_ours = accs["LogReg (AAT)"].mean() - accs[shared.OURS].mean()
    print(f"\n  encoding effect (LogReg AAT − raw):       {gap_enc:+.3f}")
    print(f"  our gap to the batch linear solver (AAT): {gap_ours:+.3f}")


if __name__ == "__main__":
    main()
