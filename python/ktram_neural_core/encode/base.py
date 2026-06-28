"""AATEncoder — data -> AAT.

The base mirrors the old Java SpikeEncoder encode/encodeAdapt split (frozen vs. adapting),
with a declared space layout (`space_sizes`) replacing both getSpikeSpace() and the deleted
SpikeStreamJoiner. An encoder emits one AAT entry per address space it declares; the AAT *is*
the pattern, so multi-field composition is just a longer tuple (see compose()).
"""


class AATEncoder:
    """One AAT entry per declared space; at most one active synapse per space."""

    def encode(self, value):
        """Frozen / inference path: data -> AAT (one entry per space)."""
        raise NotImplementedError

    def encode_adapt(self, value):
        """Training path: may migrate internal state, then encode. Non-adaptive encoders
        inherit this no-op default (encode_adapt == encode)."""
        return self.encode(value)

    @property
    def space_sizes(self):
        """Channel count of each space this encoder emits. The classifier reads this to size
        the Core geometry; compose() concatenates it across encoders."""
        raise NotImplementedError
