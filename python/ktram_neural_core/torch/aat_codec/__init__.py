"""aat_codec — compress a network's vectors into symbolic codes (AATs) and do the network's
arithmetic on the codes.

The codec side of the assimilation story (spec 08 §7): `AATCodec.encode` turns float vectors
into AAT tensors — the same ``[..., k]`` int64 contract the L1 modules read — and the kernels
(`inner`, `attention_scores`, `norm`, `combine`, `add`, `rope`) run the network's operations
directly in symbol space from lookup tables built off the codewords. One install, one namespace:
the lanes execute, the codec feeds them.
"""
from . import kernels
from .codec import RMS_EPS, SLOT_DIM, AATCodec, Codec, atoms
from .kernels import (add, attention_scores, attention_scores_table, combine, combine_table,
                      dot_table, inner, norm, rope, self_norms2)

__all__ = [
    "Codec", "AATCodec", "atoms", "SLOT_DIM", "RMS_EPS", "kernels",
    "dot_table", "inner", "attention_scores", "attention_scores_table",
    "norm", "self_norms2", "combine", "combine_table", "add", "rope",
]
