"""AATRecoder — the base AAT-recoder construct (the pass-through lane-driver).

An AATRecoder is a hardware-level construct: it drives a bank of neural lanes (one per output
channel / label) by executing kT-RAM instructions. The base IS the pass-through — it exposes raw
single-instruction execution on its lanes, which is how the instruction set is explored directly
(single-synapse lessons, hand-sequenced FF/RH/RL/RF, a classifier built by hand). L1 recoders
(e.g. RankCut) subclass it to wrap a routine behind an AAT-level read/adapt interface, with all
the analog contained inside.

Distinct from recode.RecodePolicy: that is the pure, memory-less y_vector -> AAT rule; this owns
lanes and runs the substrate.
"""


class AATRecoder:
    def __init__(self, core, labels):
        self.core = core
        self.labels = list(labels)
        self._label_to_lane = {label: i for i, label in enumerate(self.labels)}
        self._lanes = range(len(self.labels))

    def evaluate(self, aat, instruction, lane):
        """Pass-through: one L0 kT-RAM instruction on one lane; returns y. The raw window we
        explore the instruction set with."""
        return self.core.evaluate(aat, instruction, lane)
