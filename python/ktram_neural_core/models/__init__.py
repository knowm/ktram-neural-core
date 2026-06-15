"""Device models: one memristor's dynamics. All four are in Milestone 1."""

from .base import (
    INIT_TYPES,
    Device,
    clamp,
    init_conductance,
    java_round,
)
from .byte_model import ByteDevice
from .float_model import FloatDevice
from .mss_model import MSSDevice, MSSProfile
from .rs_model import RSDevice

# name -> device class, for Core construction.
MODELS = {
    "float": FloatDevice,
    "byte": ByteDevice,
    "mss": MSSDevice,
    "rs": RSDevice,
}

__all__ = [
    "Device",
    "FloatDevice",
    "ByteDevice",
    "MSSDevice",
    "MSSProfile",
    "RSDevice",
    "MODELS",
    "INIT_TYPES",
    "init_conductance",
    "java_round",
    "clamp",
]
