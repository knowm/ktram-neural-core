"""RankCut — the first L1 AAT recoder, and its rank_cut readout policy.

RankCut wraps the supervised kT-RAM routine behind an AAT-level interface (read / adapt), one
neural lane per label, with the analog contained inside.
"""

from ..base import AATRecoder

FEEDBACK = ("hard", "soft")


def check_feedback(feedback):
    """Validate the feedback rule: "hard" punishes a fired non-target lane with RL, "soft"
    lets it decay with RF. (Mirrored in torch/_lane.py; torch/ stays independent.)"""
    if feedback not in FEEDBACK:
        raise ValueError(f"feedback must be one of {FEEDBACK}, got {feedback!r}")
    return feedback


def rank_cut(y_vector, Vt=0.0, N=None):
    """Readout policy: keep lanes with y >= Vt, sort descending, return the first N as an
    output-space AAT (a tuple of lane indices, strongest first).

    A pure, memory-less policy over the lane outputs. (Vt, N) expresses every named readout:
        winner             -> (device-min, 1)   # Vt below every reachable y
        winner-above-zero  -> (0, 1)
        top-k above zero   -> (0, k)
        all above zero     -> (0, None)
    """
    above = [(lane, y) for lane, y in enumerate(y_vector) if y >= Vt]
    above.sort(key=lambda t: t[1], reverse=True)
    if N is not None:
        above = above[:N]
    return tuple(lane for lane, _ in above)


class RankCut(AATRecoder):
    """L1 supervised recoder over one neural lane per label:

        read(in_aat)         -> out_aat   non-disturbing FFLV read, recoded (inference)
        adapt(in_aat, teach) -> out_aat   adapting FF read + supervised RH/RL/RF, recoded (learn)

    (Vt, N) shape the readout only; teaching is pinned to the lane's own sign (y > 0) and
    is independent of (Vt, N) — every false positive is corrected, every time.
    """

    def __init__(self, core, labels, Vt=0.0, N=None):
        super().__init__(core, labels)
        self.Vt = Vt
        self.N = N

    def _target_lanes(self, teach):
        # target output AAT (labels or lane indices) -> set of target lanes; the lanes ARE labels.
        return {self._label_to_lane.get(t, t) for t in teach}

    def read(self, in_aat):
        """Inference: non-disturbing FFLV read on every lane, recoded. Leaves no mark."""
        y = [self.evaluate(in_aat, "FFLV", lane) for lane in self._lanes]
        return rank_cut(y, self.Vt, self.N)

    def adapt(self, in_aat, teach, feedback="hard"):
        """Supervised learn: adapting FF read on every lane, the RH/RL/RF routine vs `teach`,
        then recode the FF read it just saw.

        `feedback` picks the rule for a non-target lane that fired (y > 0). "hard" drives it
        down with RL, which maximizes the classification margin and is what a classifier wants.
        "soft" drops the RL term, so that lane decays with RF like every other non-target lane,
        and the bank keeps every answer the data supports instead of one winner — the weights
        you want when the read is going to be sampled rather than won.
        """
        check_feedback(feedback)
        target = self._target_lanes(teach)
        y = [self.evaluate(in_aat, "FF", lane) for lane in self._lanes]
        fired = {lane for lane in self._lanes if y[lane] > 0} if feedback == "hard" else set()
        for lane in self._lanes:
            instr = "RH" if lane in target else "RL" if lane in fired else "RF"
            self.evaluate(in_aat, instr, lane)
        return rank_cut(y, self.Vt, self.N)
