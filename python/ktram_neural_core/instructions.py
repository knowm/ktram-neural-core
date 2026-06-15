"""The instruction set and the instruction -> Vy contract.

Transcribed from AHaHNodeFloatOps.execute. One instruction per evaluate() call.

A read instruction (FF/FFLV/RF/RFLV) computes Vy from the devices and sets the retained y.
A feedback instruction forces Vy to a fixed value and leaves y unchanged; its Vy is

    Vy = coeff * Vdrive * (H(y) if use_H else 1)

where Vdrive is the standard drive voltage of the instruction's direction (Vf forward,
Vr reverse) and H(y) = +1 if y >= 0 else -1. Vf is positive, Vr is negative, so e.g. RH
(coeff=+1) gives Vy = Vr = -1 at default.

V_app (the signed applied voltage) is selected by direction x (standard|low):
forward standard -> Vf, forward low -> Vflv, reverse standard -> Vr, reverse low -> Vrlv.

Removed from the old enum: XX (no-op; we run one instruction per call) and RCU (branched on
a state-change flag we do not keep). FFLV/RFLV both route through the update path here — a
deliberate fix of the Java's FFLV early-return (see spec 02).
"""

from dataclasses import dataclass

FORWARD = "forward"
REVERSE = "reverse"


@dataclass(frozen=True)
class Instruction:
    name: str
    direction: str            # FORWARD | REVERSE
    reads: bool               # read instruction: compute Vy from devices, set y
    low: bool = False         # use the sub-threshold (low) voltage for V_app
    coeff: float = 0.0        # feedback Vy coefficient (unused for reads)
    use_H: bool = False       # feedback multiplies by H(y) (unused for reads)


INSTRUCTIONS = {
    # reads
    "FF":   Instruction("FF",   FORWARD, reads=True),
    "FFLV": Instruction("FFLV", FORWARD, reads=True, low=True),
    "RF":   Instruction("RF",   REVERSE, reads=True),
    "RFLV": Instruction("RFLV", REVERSE, reads=True, low=True),
    # forward feedback
    "FH":   Instruction("FH",   FORWARD, reads=False, coeff=-1.0, use_H=False),
    "FL":   Instruction("FL",   FORWARD, reads=False, coeff=+1.0, use_H=False),
    "FU":   Instruction("FU",   FORWARD, reads=False, coeff=+1.0, use_H=True),
    "FA":   Instruction("FA",   FORWARD, reads=False, coeff=-1.0, use_H=True),
    "FZ":   Instruction("FZ",   FORWARD, reads=False, coeff=0.0,  use_H=False),
    # reverse feedback
    "RH":   Instruction("RH",   REVERSE, reads=False, coeff=+1.0, use_H=False),
    "RL":   Instruction("RL",   REVERSE, reads=False, coeff=-1.0, use_H=False),
    "RU":   Instruction("RU",   REVERSE, reads=False, coeff=+1.0, use_H=True),
    "RA":   Instruction("RA",   REVERSE, reads=False, coeff=-1.0, use_H=True),
    "RZ":   Instruction("RZ",   REVERSE, reads=False, coeff=0.0,  use_H=False),
}


def resolve(instruction):
    """Accept an Instruction or its name string; return the Instruction."""
    if isinstance(instruction, Instruction):
        return instruction
    try:
        return INSTRUCTIONS[instruction]
    except KeyError:
        raise KeyError(
            f"unknown instruction {instruction!r}; valid: {sorted(INSTRUCTIONS)}"
        )


def H(y):
    """Heaviside step used by FU/FA/RU/RA. +1 at zero (matches Java)."""
    return 1.0 if y >= 0 else -1.0
