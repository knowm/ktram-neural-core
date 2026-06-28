"""LinearClassifier — one neural lane per label.

Faithful to the old Java LinearClassifier instruction routine, adapted to the emulator's
single-instruction `evaluate` (the old two-phase execute(read, feedback) is two sequential
calls). Every active synapse in a lane gets the same update routine — no built-in bias, no
split-write. A user adds a bias at the encoding level with compose(..., ConstantEncoder()).

The encoder is adapted to convergence and FROZEN before the classifier learns, so the
classifier never chases a moving encoding. fit() runs the two phases in order; adapt_encoder()
and train() expose them separately.

Phase 1 — adapt the encoder (no classifier updates):
    encoder.encode_adapt(value)            # migrate bins toward the data, then stop

Phase 2 — train the classifier on the now-frozen encoding (online, per example):
    aat = encoder.encode(value)            # frozen — bins no longer move
    for each label-lane:
        y = evaluate(aat, "FF", lane)      # read at full V (this read adapts the synapses)
        correct label            -> "RH"
        false positive (y > 0)   -> "RL"
        true negative            -> "RF"

Infer:
    aat = encoder.encode(value)            # frozen
    y   = evaluate(aat, "FFLV", lane)      # low-V, non-disturbing, per lane
    recoder.recode(y_vector)               # -> output-space AAT

read_noise defaults to 0 so training and scoring are deterministic at a fixed seed; "train
clean, then run noisy" is a later demonstration, so it stays a constructor argument.
"""

import numpy as np

from ..core import Core
from ..recode import Winner


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _crossbar_geometry(max_channels):
    """A 1 x next_pow2 crossbar holds the largest declared space. The Core uses one crossbar
    shape for every space, so it is sized to the encoder's widest space; narrower spaces (the
    bias) simply use low addresses in it."""
    return 1, _next_pow2(max(1, max_channels))


class LinearClassifier:
    def __init__(
        self,
        encoder,
        labels,
        model="byte",
        init="medium",
        read_noise=0,
        recoder=None,
        seed=None,
        **core_kwargs,
    ):
        self.encoder = encoder
        self.labels = list(labels)
        self.recoder = recoder if recoder is not None else Winner()
        self._label_to_lane = {label: i for i, label in enumerate(self.labels)}

        sizes = encoder.space_sizes
        rows, cols = _crossbar_geometry(max(sizes))
        self.core = Core(
            rows,
            cols,
            spaces_per_lane=len(sizes),
            num_lanes=len(self.labels),
            model=model,
            init=init,
            read_noise=read_noise,
            seed=seed,
            **core_kwargs,
        )

    # ----- training -----

    def adapt_encoder(self, X, epochs=10, shuffle=True, seed=0):
        """Phase 1: migrate the encoder's bins over X, with NO classifier updates.

        Run this to convergence before training — the classifier must learn against a fixed
        encoding, not one still shifting under it. After this the encoding is frozen simply by
        never calling encode_adapt again (train/predict use encode). A non-adaptive encoder
        (e.g. ConstantEncoder) is a no-op here.
        """
        rng = np.random.default_rng(seed)
        n = len(X)
        for _ in range(epochs):
            order = rng.permutation(n) if shuffle else range(n)
            for i in order:
                self.encoder.encode_adapt(X[i])
        return self

    def train(self, value, label):
        """Phase 2: one supervised online update over all label-lanes (the Java FF -> RH/RL/RF
        table), on the FROZEN encoding. Assumes the encoder is already adapted."""
        aat = self.encoder.encode(value)                # frozen — bins no longer move
        target = self._label_to_lane[label]
        for lane in range(len(self.labels)):
            y = self.core.evaluate(aat, "FF", lane)
            if lane == target:
                self.core.evaluate(aat, "RH", lane)     # correct label
            elif y > 0:
                self.core.evaluate(aat, "RL", lane)     # false positive
            else:
                self.core.evaluate(aat, "RF", lane)     # true negative

    def fit(self, X, y, epochs=1, shuffle=True, seed=0, encoder_epochs=10):
        """Adapt the encoder (encoder_epochs passes), freeze it, then train the classifier for
        `epochs` passes against that fixed encoding. The two phases run strictly in order, so the
        classifier never chases a moving encoding. Shuffling uses a seeded RNG (reproducible).

        Set encoder_epochs=0 to skip encoder adaptation (e.g. a non-adaptive encoder, or one you
        adapted yourself via adapt_encoder)."""
        if encoder_epochs:
            self.adapt_encoder(X, epochs=encoder_epochs, shuffle=shuffle, seed=seed)
        rng = np.random.default_rng(seed)
        n = len(X)
        for _ in range(epochs):
            order = rng.permutation(n) if shuffle else range(n)
            for i in order:
                self.train(X[i], y[i])
        return self

    # ----- inference -----

    def scores(self, value):
        """The frozen low-voltage read vector, one `y` per label-lane."""
        aat = self.encoder.encode(value)
        return [self.core.evaluate(aat, "FFLV", lane) for lane in range(len(self.labels))]

    def recode(self, value):
        """The recoder's output-space AAT (channels are lane indices)."""
        return self.recoder.recode(self.scores(value))

    def predict(self, value):
        """The predicted label (first output channel), or None if the recoder abstains."""
        out = self.recode(value)
        return self.labels[out[0]] if out else None
