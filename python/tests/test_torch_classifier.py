"""Tier-1 congruence: the torch Classifier against the numpy oracle, bit for bit.

The byte model's integer state makes the sharp read and the B = 1 adapt sequence exactly
reproducible under lane-vectorization, so this tier asserts equality — it is the anchor and
the bug-catcher for the mechanical traps (Java half-up rounding, clamp order, int32
accumulation, NONE masking). Statistical tiers live in test_torch_noise.py and
test_torch_congruence.py.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ktram_neural_core import Core, RankCut, rank_cut as oracle_rank_cut  # noqa: E402
from ktram_neural_core.torch import Classifier, rank_cut  # noqa: E402

from _congruence import (  # noqa: E402
    oracle_rank_cut_padded,
    oracle_weights,
    record,
    to_oracle_aat,
)

LANES, SPACES, CHANNELS = 5, 4, 8


def _pair(seed=3, Vt=0.0, N=None, read_noise=0):
    core = Core(1, CHANNELS, spaces_per_lane=SPACES, num_lanes=LANES, model="byte",
                init="medium", read_noise=read_noise, seed=seed)
    oracle = RankCut(core, labels=list(range(LANES)), Vt=Vt, N=N)
    return oracle, Classifier.from_core(oracle)


def _random_aats(n, rng, none_frac=0.25):
    aats = rng.integers(0, CHANNELS, size=(n, SPACES))
    mask = rng.random((n, SPACES)) < none_frac
    return np.where(mask, -1, aats)


def test_from_core_lifts_exact_state():
    oracle, clf = _pair()
    ga, gb = oracle_weights(oracle.core)
    assert np.array_equal(clf.ga.numpy(), ga)
    assert np.array_equal(clf.gb.numpy(), gb)


def test_seeded_constructor_matches_oracle_init():
    """A seeded torch module starts bit-for-bit where a seeded oracle Core starts — the
    RNG stream is consumed in the oracle's device-creation order."""
    for init in ("medium", "low", "high_noise", "low_noiseless"):
        core = Core(1, CHANNELS, spaces_per_lane=SPACES, num_lanes=LANES, model="byte",
                    init=init, seed=11)
        clf = Classifier(LANES, SPACES, CHANNELS, init=init, seed=11)
        ga, gb = oracle_weights(core)
        assert np.array_equal(clf.ga.numpy(), ga), init
        assert np.array_equal(clf.gb.numpy(), gb), init
    record("Classifier", 1, "seeded constructor init == oracle Core init "
                            "(medium/low/high_noise/low_noiseless), bit-exact")


def test_sharp_read_bitexact():
    oracle, clf = _pair()
    rng = np.random.default_rng(0)
    aats = _random_aats(64, rng)
    y_torch = clf.read_y(torch.from_numpy(aats))
    for i, row in enumerate(aats):
        o_aat = to_oracle_aat(row)
        y_oracle = [oracle.evaluate(o_aat, "FFLV", lane) for lane in range(LANES)]
        assert y_torch[i].numpy().tolist() == y_oracle
    record("Classifier", 1, "sharp read y == oracle FFLV read on 64 random AATs "
                            "(25% NONE entries), float64 bit-exact")


def test_all_none_aat_reads_zero():
    _, clf = _pair()
    y = clf.read_y(torch.full((SPACES,), -1, dtype=torch.int64))
    assert (y == 0).all()


def test_rank_cut_matches_oracle_policy():
    rng = np.random.default_rng(1)
    for _ in range(200):
        y = rng.integers(-3, 4, size=LANES) / 4.0     # coarse grid to force ties
        for Vt, N in [(0.0, None), (0.0, 1), (0.0, 2), (-1.0, None), (0.25, 3)]:
            expected = oracle_rank_cut_padded(
                oracle_rank_cut(list(y), Vt, N), LANES if N is None else N)
            got = rank_cut(torch.from_numpy(y), Vt, N).numpy()
            assert np.array_equal(got, expected), (y, Vt, N)
    record("Classifier", 1, "rank_cut policy == oracle over tie-heavy grids, "
                            "all (Vt, N) shapes incl. truncation and padding")


def test_adapt_b1_sequence_bitexact():
    """The headline tier-1 check: a B = 1 supervised adapt stream leaves identical integer
    state at every step, and recodes identically."""
    oracle, clf = _pair(Vt=0.0, N=None)
    rng = np.random.default_rng(2)
    aats = _random_aats(300, rng)
    targets = rng.integers(0, LANES, size=300)
    for i in range(300):
        o_out = oracle.adapt(to_oracle_aat(aats[i]), {int(targets[i])})
        t_out = clf.adapt(torch.from_numpy(aats[i]), torch.tensor([targets[i]]))
        assert np.array_equal(t_out.numpy(), oracle_rank_cut_padded(o_out, LANES)), i
        if i % 50 == 0 or i == 299:
            ga, gb = oracle_weights(oracle.core)
            assert np.array_equal(clf.ga.numpy(), ga), i
            assert np.array_equal(clf.gb.numpy(), gb), i
    record("Classifier", 1, "300-step B=1 adapt sequence: weights and recoded outputs "
                            "bit-exact vs oracle RankCut (RH/RL/RF, incl. RF re-read)")


def test_adapt_multi_target_matches_per_pixel_routine():
    """The decoder-style many-heads case: independent per-pixel classifiers in one Core are
    one flat Classifier with a multi-lane target AAT (the generator's per-pixel RH lane)."""
    n_pix, levels, spaces, ch = 3, 2, 4, 8
    lanes = n_pix * levels
    core = Core(1, ch, spaces_per_lane=spaces, num_lanes=lanes, model="byte",
                init="medium", read_noise=0, seed=5)
    clf = Classifier.from_core(core)
    rng = np.random.default_rng(6)
    for step in range(100):
        aat = rng.integers(0, ch, size=spaces)
        pix_levels = rng.integers(0, levels, size=n_pix)
        # the oracle's NeuralPixelDecoder routine, verbatim
        o_aat = to_oracle_aat(aat)
        for p in range(n_pix):
            base = p * levels
            for lv in range(levels):
                lane = base + lv
                y = core.evaluate(o_aat, "FF", lane)
                if lv == pix_levels[p]:
                    core.evaluate(o_aat, "RH", lane)
                elif y > 0:
                    core.evaluate(o_aat, "RL", lane)
                else:
                    core.evaluate(o_aat, "RF", lane)
        target = torch.tensor([p * levels + int(pix_levels[p]) for p in range(n_pix)])
        clf.adapt(torch.from_numpy(aat), target)
        ga, gb = oracle_weights(core)
        assert np.array_equal(clf.ga.numpy(), ga), step
        assert np.array_equal(clf.gb.numpy(), gb), step
    record("Classifier", 1, "multi-target adapt == oracle per-pixel decoder routine "
                            "(3 pixels x 2 levels, 100 steps), bit-exact")


def test_none_target_padding_is_ignored():
    _, clf = _pair()
    _, clf2 = _pair()
    aat = torch.tensor([1, 2, 3, 4])
    clf.adapt(aat, torch.tensor([2]))
    clf2.adapt(aat, torch.tensor([2, -1, -1]))     # -1 padding must not touch lane 0
    assert torch.equal(clf.ga, clf2.ga)
    assert torch.equal(clf.gb, clf2.gb)


def test_read_is_nondisturbing_and_disturbing_flv_refused():
    _, clf = _pair()
    before = clf.ga.clone()
    clf.read(torch.tensor([0, 1, 2, 3]))
    clf.read_sampled(torch.tensor([0, 1, 2, 3]), T=0.5,
                     generator=torch.Generator().manual_seed(0))
    assert torch.equal(clf.ga, before)
    with pytest.raises(ValueError):
        Classifier(2, 2, 4, forward_low_voltage=0.3)


def test_float_model_lift_refused():
    core = Core(1, 8, spaces_per_lane=2, num_lanes=2, model="float", seed=0)
    with pytest.raises(NotImplementedError):
        Classifier.from_core(core)
    record("Classifier", 1, "non-byte device models refused by from_core "
                            "(raise, never approximate)")


def test_batched_read_equals_streamed_read():
    _, clf = _pair()
    rng = np.random.default_rng(7)
    aats = _random_aats(32, rng)
    batched = clf.read_y(torch.from_numpy(aats))
    streamed = torch.stack([clf.read_y(torch.from_numpy(a)) for a in aats])
    assert torch.equal(batched, streamed)
    record("Classifier", 1, "batched read == streamed read (lane/batch vectorization exact)")


def test_state_dict_round_trip():
    _, clf = _pair()
    clf.adapt(torch.tensor([1, 2, 3, 4]), torch.tensor([0]))
    clone = Classifier(LANES, SPACES, CHANNELS, seed=0)
    clone.load_state_dict(clf.state_dict())
    aat = torch.tensor([2, 0, 5, 1])
    assert torch.equal(clone.read_y(aat), clf.read_y(aat))
