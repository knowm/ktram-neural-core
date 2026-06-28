"""AATRecoder — lane-`y` vector -> AAT in the output space.

The Ch4 "AAT Recoder / A2D" in code: an analog vector in (one `y` per label-lane), an AAT out
(one channel per label/lane) — exactly what a downstream lane would consume if lanes are ever
layered. Recoders are pure functions over the read vector, no device state.
"""


class AATRecoder:
    def recode(self, y_vector):
        """Return the output-space AAT for a vector of lane `y` values."""
        raise NotImplementedError
