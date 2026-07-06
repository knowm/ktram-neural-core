"""aat_recoder/rank_cut — the rank_cut readout policy and the RankCut L1 recoder.

Per 04-repo-language-testing.md: assert the mechanism works (the policy table is exact; the recoder
learns above chance and is deterministic at a fixed seed with read_noise=0). It does NOT assert any
congruence or accuracy bar — that validation is Alex's.
"""

import numpy as np

from ktram_neural_core import Core
from ktram_neural_core.aat_recoder import RankCut, rank_cut


# ----- the pure readout policy: the (Vt, N) table -----

def test_rank_cut_sorts_above_zero_descending():
    assert rank_cut([-0.2, 0.7, 0.3, -0.1]) == (1, 2)        # all above zero, sorted
    assert rank_cut([0.1, 0.7, 0.3]) == (1, 2, 0)            # full descending order


def test_rank_cut_winner_above_zero():
    assert rank_cut([-0.2, 0.7, 0.3], Vt=0.0, N=1) == (1,)


def test_rank_cut_top_k_above_zero():
    assert rank_cut([0.1, 0.7, 0.3, -0.1], Vt=0.0, N=2) == (1, 2)


def test_rank_cut_winner_unconditional():
    # winner = (device-min, 1): a floor below every reachable y returns the argmax even if < 0.
    assert rank_cut([-0.5, -0.9, -0.1], Vt=-10.0, N=1) == (2,)


def test_rank_cut_threshold_excludes_below_floor():
    assert rank_cut([-0.2, -0.7]) == ()                      # nothing >= 0
    assert rank_cut([]) == ()


# ----- the RankCut recoder driving real neural lanes -----

def _toy_aats():
    """One distinct active channel per class -> a separable 3-class set of AATs. AAT for class k is
    (k,): channel k in the single space. Each lane learns its own channel positive, others negative."""
    labels = [0, 1, 2]
    aats = {k: (k,) for k in labels}
    return labels, aats


def _trained_recoder(seed=0, epochs=40, **rc_kwargs):
    labels, aats = _toy_aats()
    core = Core(1, 4, spaces_per_lane=1, num_lanes=len(labels),
                model="byte", init="low", read_noise=0, seed=seed)
    rc = RankCut(core, labels, **rc_kwargs)
    for _ in range(epochs):
        for k in labels:
            rc.adapt(aats[k], teach={k})
    return rc, labels, aats


def test_rankcut_learns_to_separate():
    rc, labels, aats = _trained_recoder()
    # after training, each class's own lane is the strongest read (argmax) for its input.
    for k in labels:
        out = rc.read(aats[k])
        assert out and out[0] == k


def test_rankcut_winner_readout_returns_single_address():
    rc, labels, aats = _trained_recoder(Vt=0.0, N=1)
    for k in labels:
        assert rc.read(aats[k]) == (k,)


def test_rankcut_is_deterministic_at_fixed_seed():
    outs = []
    for _ in range(2):
        rc, labels, aats = _trained_recoder(seed=0)
        outs.append(tuple(rc.read(aats[k]) for k in labels))
    assert outs[0] == outs[1]
