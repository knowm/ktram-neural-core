"""ktram-neural-core — an open emulator of kT-RAM realized as a 2-1 neural lane.

Object model (hardware-native): Core -> NeuralLane -> UnitCrossbarPair (differential pair) ->
UnitCrossbar -> Device. Addressed only by Activation Address Tuples (AATs).
"""

from .core import (
    NOISE_FLICKER,
    NOISE_THERMAL,
    READ_NOISE,
    ROOM_TEMPERATURE_K,
    Core,
)
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
]
