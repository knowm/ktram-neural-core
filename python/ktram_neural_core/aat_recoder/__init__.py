"""aat_recoder/ — the AAT-recoder construct and the L1 recoders built on it.

An AATRecoder is a hardware-level construct: it drives a bank of neural lanes by executing kT-RAM
instructions. The base IS the pass-through (raw instruction execution, for exploring the
instruction set directly); L1 recoders wrap a routine behind an AAT-level read/adapt interface
with the analog contained inside. RankCut is the first L1 recoder.

Distinct from recode.RecodePolicy (the pure, memory-less y_vector -> AAT rule). See
planning/06-aat-recoder-refactor.md.
"""

from .base import AATRecoder
from .basis_encoder import BasisEncoder, BasisGroup
from .rank_cut import RankCut, rank_cut

__all__ = ["AATRecoder", "RankCut", "rank_cut", "BasisEncoder", "BasisGroup"]
