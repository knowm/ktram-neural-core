"""TwoOne: the 2-1 readout and per-device update-voltage mapping.

Topology is fixed at 2-1 and is not an exposed axis. This thin seam keeps the readout and
the update-voltage rule in one place so a different readout could be added later without
touching the lane. Do not expose it as a constructor option; do not build 1-2.
"""


class TwoOne:
    @staticmethod
    def readout(pairs):
        """Given (Ga, Gb) over the active synapses, return (top, bottom):

            top    = sum(Ga - Gb)
            bottom = sum(Ga + Gb)

        Vy = V_app * top / bottom, and y = top / bottom in [-1, 1].
        """
        top = 0.0
        bottom = 0.0
        for ga, gb in pairs:
            top += ga - gb
            bottom += ga + gb
        return top, bottom

    @staticmethod
    def update_voltages(v_app, vy):
        """Per-device applied voltages for the differential pair:

            dVa = V_app - Vy   (a-side)
            dVb = Vy + V_app   (b-side)
        """
        return v_app - vy, vy + v_app
