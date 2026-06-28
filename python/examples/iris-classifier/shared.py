"""Shared logic for the Iris benchmark + figures.

The fair comparison: an AAT encoder is a feature transform, so it can change a linear
classifier's accuracy (sometimes helping, sometimes hurting). So "is our kT-RAM neural-lane
classifier as good as a linear classifier?" must compare both on the SAME encoded AATs, not on
raw features. Here we adapt the A2D encoder ONCE, freeze it, then run our lane classifier and
reference linear models (LogisticRegression, LinearSVC) on the identical one-hot encoding of
those AATs. A raw-feature LogReg is included to show how the encoding affects accuracy.

All of this comparison machinery lives in the example. The library classifier
(ktram_neural_core.classify.LinearClassifier) is untouched — it only ever sees AATs.

The one-hot map is exactly what the lane reads: an AAT entry is a channel index within a space,
and the lane sums one differential-pair synapse per active (space, channel). So a faithful
linear baseline is a linear model over the binary vector with a 1 at each active (space, channel)
position — length sum(space_sizes), the bias space included.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # python/ on path

import numpy as np                                                    # noqa: E402
from sklearn.datasets import load_iris                                # noqa: E402
from sklearn.model_selection import train_test_split                  # noqa: E402
from sklearn.linear_model import LogisticRegression                   # noqa: E402
from sklearn.svm import LinearSVC                                      # noqa: E402
from sklearn.metrics import accuracy_score                            # noqa: E402

from ktram_neural_core.encode import A2DEncoder, ConstantEncoder, compose   # noqa: E402
from ktram_neural_core.classify import LinearClassifier                     # noqa: E402
from ktram_neural_core.recode import Winner                                 # noqa: E402

# --- configuration (the tuned operating point; one place for benchmark + figures) ---
BITS = 3
LRATE = 0.01            # A2D bin-migration rate (encode_adapt EMA)
EPOCHS = 5            # classifier passes (phase 2)
ENCODER_EPOCHS = 5    # encoder-adaptation passes (phase 1), then frozen
MODEL = "byte"
INIT = "low"
TEST_SIZE = 0.25

# Display order and the headline method (ours). Raw LogReg sits apart — it sees raw features.
METHOD_ORDER = ["LogReg (raw)", "LogReg (AAT)", "LinearSVC (AAT)", "kT-RAM lane (AAT)"]
OURS = "kT-RAM lane (AAT)"


def load_split(seed):
    data = load_iris()
    X, y = data.data, data.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    return data, X_tr, X_te, y_tr, y_te


def build_encoder(X_tr):
    """A2D over the four features (+ a bias space), seeded from the training-split ranges."""
    return compose(
        A2DEncoder(dims=4, bits=BITS, init_min=X_tr.min(0), init_max=X_tr.max(0), l=LRATE),
        ConstantEncoder(),
    )


def aat_to_onehot(aat, space_sizes):
    """The binary (space, channel) vector the lane reads: one 1 per active space."""
    vec = np.zeros(int(np.sum(space_sizes)))
    off = 0
    for entry, size in zip(aat, space_sizes):
        if entry is not None:
            vec[off + entry] = 1.0
        off += size
    return vec


def encode_matrix(encoder, X):
    """Encode a dataset through the FROZEN encoder into the one-hot AAT feature matrix."""
    sizes = encoder.space_sizes
    return np.array([aat_to_onehot(encoder.encode(x), sizes) for x in X])


def run_once(seed=0):
    """Train our lane classifier and the reference models on one split, all sharing the SAME
    frozen AAT encoding. Returns a result dict (splits, encoder, name -> test predictions)."""
    data, X_tr, X_te, y_tr, y_te = load_split(seed)
    labels = list(range(len(data.target_names)))

    encoder = build_encoder(X_tr)
    clf = LinearClassifier(encoder, labels=labels, model=MODEL, init=INIT,
                           read_noise=0, recoder=Winner(), seed=seed)
    clf.adapt_encoder(X_tr, epochs=ENCODER_EPOCHS, seed=seed)        # phase 1: adapt + freeze
    clf.fit(X_tr, y_tr, epochs=EPOCHS, encoder_epochs=0, seed=seed)  # phase 2: frozen encoding

    # The exact same frozen encoder, as one-hot matrices for the reference linear models.
    Xtr_enc = encode_matrix(encoder, X_tr)
    Xte_enc = encode_matrix(encoder, X_te)

    preds = {}
    preds[OURS] = np.array([clf.predict(x) for x in X_te])
    preds["LogReg (AAT)"] = (
        LogisticRegression(max_iter=5000).fit(Xtr_enc, y_tr).predict(Xte_enc)
    )
    preds["LinearSVC (AAT)"] = (
        LinearSVC(max_iter=20000).fit(Xtr_enc, y_tr).predict(Xte_enc)
    )
    preds["LogReg (raw)"] = (
        LogisticRegression(max_iter=5000).fit(X_tr, y_tr).predict(X_te)
    )

    return {
        "data": data, "encoder": encoder, "seed": seed,
        "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
        "preds": preds,
    }


def multi_seed(seeds):
    """Per-method test accuracy across seeds -> {name: np.array of accuracies}."""
    accs = {name: [] for name in METHOD_ORDER}
    for s in seeds:
        r = run_once(s)
        for name, p in r["preds"].items():
            accs[name].append(accuracy_score(r["y_te"], p))
    return {name: np.array(v) for name, v in accs.items()}


def bin_edges(encoder, dim, lo, hi, ndims=4, n=4000):
    """Black-box read of one dimension's adaptive bin boundaries: sweep the value and record
    where the bin index changes. A2D dimensions are independent, so the other entries are held
    at zero. Returns the interior edges in [lo, hi]."""
    xs = np.linspace(lo, hi, n)
    base = np.zeros(ndims)
    edges, prev = [], None
    for x in xs:
        v = base.copy()
        v[dim] = x
        b = encoder.encode(v)[dim]
        if prev is not None and b != prev:
            edges.append(x)
        prev = b
    return edges
