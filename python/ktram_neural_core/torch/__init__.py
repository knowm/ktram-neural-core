"""ktram_neural_core.torch — the fast L1 runtime (spec 08).

The two L1 modules as production ``torch.nn.Module``s, byte model only, verified against the
numpy oracle by the effective-congruence battery: AATs in, AATs out, integer buffers, no
autograd. ``aat_codec`` is the other half of the assimilation story — it turns float vectors
into the AAT tensors the lanes read, and runs the network's arithmetic on the codes. The
oracle (`ktram_neural_core` proper) stays the readable reference; importing this subpackage
requires ``torch`` (``pip install ktram-neural-core[torch]``).
"""

from . import aat_codec
from ._lane import NoiseParams, rank_cut
from .basis_encoder import BasisEncoder
from .classifier import Classifier

__all__ = ["Classifier", "BasisEncoder", "rank_cut", "NoiseParams", "aat_codec"]
