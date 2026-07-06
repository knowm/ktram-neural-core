"""The concrete recode policies: Winner, AboveZero, WinnerAboveZero.

Pure functions over the lane-`y` vector, so they are trivially testable and reusable by any
later multi-lane readout. Each returns an output-space AAT whose channels are lane indices.
"""

from .base import RecodePolicy


def _argmax(y_vector):
    return max(range(len(y_vector)), key=lambda i: y_vector[i])


class Winner(RecodePolicy):
    """`(argmax,)` — the single active output channel; its index is the predicted class."""

    def recode(self, y_vector):
        return (_argmax(y_vector),)


class AboveZero(RecodePolicy):
    """The set of channels with `y > 0` (multi-label readout)."""

    def recode(self, y_vector):
        return tuple(i for i, y in enumerate(y_vector) if y > 0)


class WinnerAboveZero(RecodePolicy):
    """`(argmax,)` if its `y > 0`, else empty (no decision)."""

    def recode(self, y_vector):
        if not len(y_vector):
            return ()
        i = _argmax(y_vector)
        return (i,) if y_vector[i] > 0 else ()
