"""RecodePolicy — a pure, memory-less readout rule: lane-`y` vector -> output-space AAT.

One `y` per label-lane in, one AAT out (one channel per label/lane) — what a downstream lane
consumes if lanes are layered. A RecodePolicy holds no device state; it is only the `y -> AAT`
rule (Winner, AboveZero, WinnerAboveZero).

NOTE — do not confuse this with the `AATRecoder` construct in `aat_recoder/`. That one is a
hardware-level structure that *drives* neural lanes and executes kT-RAM instructions; this is just
the pure output rule applied to a read vector (by an L1 recoder, or by the LinearClassifier).
"""


class RecodePolicy:
    def recode(self, y_vector):
        """Return the output-space AAT for a vector of lane `y` values."""
        raise NotImplementedError
