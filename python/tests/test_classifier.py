"""classify/ — the supervised train/predict loop runs and *learns*.

Per 04-repo-language-testing.md, this asserts the mechanism works (accuracy rises above
chance on a separable set, deterministic at a fixed seed with read_noise=0). It does NOT
assert congruence with the Java stack or any specific accuracy bar — that validation is Alex's.
"""

import numpy as np

from ktram_neural_core.encode import A2DEncoder, ConstantEncoder, compose
from ktram_neural_core.classify import LinearClassifier
from ktram_neural_core.recode import Winner


def _separable_set(n=120, seed=0):
    """Three well-separated 2-D Gaussian blobs -> a linearly separable 3-class problem."""
    rng = np.random.default_rng(seed)
    centers = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.85)]
    X, y = [], []
    for label, (cx, cy) in enumerate(centers):
        pts = rng.normal((cx, cy), 0.05, size=(n, 2))
        X.append(pts)
        y += [label] * n
    return np.clip(np.vstack(X), 0.0, 1.0), np.array(y)


def test_classifier_learns_above_chance():
    X, y = _separable_set(seed=0)
    encoder = compose(A2DEncoder(dims=2, bits=4, init_min=0.0, init_max=1.0),
                      ConstantEncoder())
    clf = LinearClassifier(encoder, labels=[0, 1, 2], model="byte",
                           read_noise=0, recoder=Winner(), seed=0)
    clf.fit(X, y, epochs=8, shuffle=True, seed=0)

    preds = np.array([clf.predict(x) for x in X])
    acc = float((preds == y).mean())
    assert acc > 1.0 / 3 + 0.2          # well above 3-class chance


def test_classifier_is_deterministic_at_fixed_seed():
    X, y = _separable_set(seed=1)
    enc = lambda: compose(A2DEncoder(dims=2, bits=4), ConstantEncoder())
    accs = []
    for _ in range(2):
        clf = LinearClassifier(enc(), labels=[0, 1, 2], read_noise=0, seed=0)
        clf.fit(X, y, epochs=4, seed=0)
        accs.append(float((np.array([clf.predict(x) for x in X]) == y).mean()))
    assert accs[0] == accs[1]
