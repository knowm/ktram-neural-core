"""matplotlib helpers for the single-synapse figures.

Two panels per experiment, mirroring the lesson: y on the left, the differential pair
(Ga, Gb) on the right — so they line up with 04_Synapse_REVIEW/_img/*.png by eye.
matplotlib is an examples-only dependency; the core package never imports it.
"""

import matplotlib.pyplot as plt


def plot_synapse(traces, title, gmin=None, gmax=None, ax=None):
    """Plot one experiment's (ys, gas, gbs) as a y + conductance pair of panels.

    The conductance panel autoscales to the data with a margin, so a flat asymptote (e.g.
    Float Ga leveling at ~0.93*GMax under the dead-zone) sits below the border rather than
    jammed against it. gmin/gmax are accepted for back-compat but no longer pin the axis.
    """
    ys, gas, gbs = traces
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(11, 3.6))

    ax[0].plot(ys, color="tab:blue", lw=1.2)
    ax[0].set_ylim(-1.05, 1.05)
    ax[0].axhline(0, color="0.8", lw=0.8)
    ax[0].set_title(f"y | {title}")

    ax[1].plot(gas, color="tab:blue", lw=1.2, label="Ga")
    ax[1].plot(gbs, color="tab:green", lw=1.2, label="Gb")
    ax[1].margins(y=0.12)                 # headroom above/below the data, no hard clip
    ax[1].set_title(f"Ga, Gb | {title}")
    ax[1].legend(loc="best")
    for a in ax:
        a.grid(True, color="0.92")
    return ax


def rails(core):
    """(GMin, GMax) for the core's model, for the conductance panel's y-limits."""
    dev = core.lane(0).spaces[0].a.device_at(0)
    return dev.GMIN, dev.GMAX
