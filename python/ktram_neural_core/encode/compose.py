"""compose(*encoders) — combine encoders into one by concatenating their AATs and space_sizes.

The single mechanism for both multi-encoder composition and adding a bias. Each child receives
the same `value` (a ConstantEncoder ignores it); per-field input routing is not in scope. There
is no joiner — the assembled tuple *is* the spike pattern, one entry per declared space.

    encoder = compose(A2DEncoder(dims=4, bits=5), ConstantEncoder())
    # encode([5.1, 3.5, 1.4, 0.2]) -> (b0, b1, b2, b3, 0); space_sizes [32, 32, 32, 32, 1].
"""

from .base import AATEncoder


class _Composed(AATEncoder):
    def __init__(self, encoders):
        self._encoders = tuple(encoders)

    def encode(self, value):
        out = ()
        for e in self._encoders:
            out += tuple(e.encode(value))
        return out

    def encode_adapt(self, value):
        out = ()
        for e in self._encoders:
            out += tuple(e.encode_adapt(value))
        return out

    @property
    def space_sizes(self):
        sizes = []
        for e in self._encoders:
            sizes += list(e.space_sizes)
        return sizes


def compose(*encoders):
    return _Composed(encoders)
