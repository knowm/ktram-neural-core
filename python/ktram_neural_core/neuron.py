"""Neuron: a partition (a per-space (offset, size) window) over a lane — a view, not
hardware. Not needed for Milestone 1; this stub marks the seam so it can be added cleanly
without reshaping the lane.
"""


class Neuron:
    def __init__(self, lane, offset=0, size=None):
        self.lane = lane
        self.offset = offset
        self.size = size if size is not None else len(lane.spaces)

    # Milestone 1 does not exercise partitions. Methods (evaluate over the window, etc.)
    # attach here later.
