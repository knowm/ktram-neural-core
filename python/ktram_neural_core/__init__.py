"""ktram-neural-core — an open emulator of kT-RAM realized as a 2-1 neural lane.

Object model (hardware-native): Core -> NeuralLane -> UnitCrossbarPair (differential pair) ->
UnitCrossbar -> Device. Addressed only by Activation Address Tuples (AATs).
"""

from .aat_recoder import AATRecoder, BasisEncoder, BasisGroup, RankCut, rank_cut
from .classify import LinearClassifier
from .core import (
    NOISE_FLICKER,
    NOISE_THERMAL,
    READ_NOISE,
    ROOM_TEMPERATURE_K,
    Core,
)
from .encode import A2DEncoder, AATEncoder, ConstantEncoder, compose
from .instructions import INSTRUCTIONS, Instruction
from .lane import NeuralLane
from .models import (
    INIT_TYPES,
    MODELS,
    ByteDevice,
    FloatDevice,
    MSSDevice,
    MSSProfile,
    RSDevice,
)
from .neuron import Neuron
from .recode import AboveZero, RecodePolicy, Winner, WinnerAboveZero
from .topology import TwoOne
from .unit_crossbar import UnitCrossbar, UnitCrossbarPair

__version__ = "0.0.1"

__all__ = [
    "Core",
    "READ_NOISE",
    "NOISE_THERMAL",
    "NOISE_FLICKER",
    "ROOM_TEMPERATURE_K",
    "NeuralLane",
    "Neuron",
    "UnitCrossbarPair",
    "UnitCrossbar",
    "TwoOne",
    "Instruction",
    "INSTRUCTIONS",
    "MODELS",
    "INIT_TYPES",
    "FloatDevice",
    "ByteDevice",
    "MSSDevice",
    "MSSProfile",
    "RSDevice",
    # encode/ — data -> AAT
    "AATEncoder",
    "A2DEncoder",
    "ConstantEncoder",
    "compose",
    # recode/ — lane-y vector -> AAT (pure recode policies)
    "RecodePolicy",
    "Winner",
    "AboveZero",
    "WinnerAboveZero",
    # aat_recoder/ — the lane-driving construct + L1 recoders
    "AATRecoder",
    "RankCut",
    "rank_cut",
    "BasisEncoder",
    "BasisGroup",
    # classify/ — one lane per label
    "LinearClassifier",
]
