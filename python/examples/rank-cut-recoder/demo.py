"""rank-cut-recoder — the RankCut L1 AAT Recoder on Iris, against the iris-classifier benchmark.

Verifies the new RankCut recoder the honest way: run it on the SAME Iris pipeline as
examples/iris-classifier and show it reproduces that example's kT-RAM lane result (same supervised
routine, same frozen encoder, same seed -> same predictions), then sit it in the same accuracy
table as the reference linear models.

The point: RankCut is the iris-classifier's LinearClassifier+Winner repackaged as one L1 recoder
behind a clean read/adapt interface. Same lanes, same kT-RAM instructions, same answer.

Run:  python demo.py
"""

import sys
import pathlib

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))                        # python/ on path
sys.path.insert(0, str(_HERE.parents[1] / "iris-classifier"))   # reuse the iris example's setup

import numpy as np                                               # noqa: E402
from sklearn.linear_model import LogisticRegression             # noqa: E402
from sklearn.svm import LinearSVC                                # noqa: E402
from sklearn.metrics import accuracy_score                      # noqa: E402

from ktram_neural_core import Core, LinearClassifier, Winner     # noqa: E402
from ktram_neural_core.aat_recoder import RankCut                # noqa: E402

import shared                                                    # examples/iris-classifier/shared.py


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _core_like_classifier(encoder, n_labels, seed):
    """A Core built exactly as LinearClassifier builds its own — same geometry, model, init, seed —
    so the RankCut recoder starts from the identical substrate."""
    sizes = encoder.space_sizes
    cols = _next_pow2(max(1, max(sizes)))
    return Core(1, cols, spaces_per_lane=len(sizes), num_lanes=n_labels,
                model=shared.MODEL, init=shared.INIT, read_noise=0, seed=seed)


def main(seed=0):
    data, X_tr, X_te, y_tr, y_te = shared.load_split(seed)
    labels = list(range(len(data.target_names)))

    # --- the iris-classifier example's kT-RAM lane: LinearClassifier + the Winner recoder ---
    clf = LinearClassifier(shared.build_encoder(X_tr), labels=labels,
                           model=shared.MODEL, init=shared.INIT, read_noise=0,
                           recoder=Winner(), seed=seed)
    clf.adapt_encoder(X_tr, epochs=shared.ENCODER_EPOCHS, seed=seed)    # phase 1: adapt + freeze
    clf.fit(X_tr, y_tr, epochs=shared.EPOCHS, encoder_epochs=0, seed=seed)
    encoder = clf.encoder                                               # the now-frozen encoder

    # --- the SAME computation as a RankCut L1 recoder ---
    # Same frozen encoder, an identically-seeded core, the same training order (mirror fit's RNG),
    # and the Winner readout expressed as (Vt = -inf, N = 1) = unconditional argmax.
    rc = RankCut(_core_like_classifier(encoder, len(labels), seed), labels,
                 Vt=float("-inf"), N=1)
    rng = np.random.default_rng(seed)
    for _ in range(shared.EPOCHS):
        for i in rng.permutation(len(X_tr)):
            rc.adapt(encoder.encode(X_tr[i]), teach={y_tr[i]})

    # --- predictions on the held-out split ---
    clf_pred = np.array([clf.predict(x) for x in X_te])
    rc_pred = np.array([labels[rc.read(encoder.encode(x))[0]] for x in X_te])
    print(f"RankCut reproduces the iris-classifier lane exactly: {bool((clf_pred == rc_pred).all())}\n")

    # --- the iris-classifier benchmark table, with RankCut alongside (same frozen encoding) ---
    Xtr_enc, Xte_enc = shared.encode_matrix(encoder, X_tr), shared.encode_matrix(encoder, X_te)
    rows = [
        ("RankCut L1 recoder (ours)", rc_pred),
        ("LinearClassifier lane (iris ex.)", clf_pred),
        ("LogReg (AAT)", LogisticRegression(max_iter=5000).fit(Xtr_enc, y_tr).predict(Xte_enc)),
        ("LinearSVC (AAT)", LinearSVC(max_iter=20000).fit(Xtr_enc, y_tr).predict(Xte_enc)),
    ]
    print(f"Iris test accuracy (same frozen encoding, seed={seed}):")
    for name, pred in rows:
        print(f"  {name:34s} {accuracy_score(y_te, pred):.3f}")


if __name__ == "__main__":
    main()
