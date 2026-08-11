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


# ---------------------------------------------------------------------------
# Grouped (num_groups=G): the F0 anchor — a group-stacked bank must be
# indistinguishable from G independent flat Classifiers, byte for byte.
# ---------------------------------------------------------------------------

GROUPS = 3


def _grouped_pair(seed=13, Vt=0.0, N=None):
    oracles = [RankCut(Core(1, CHANNELS, spaces_per_lane=SPACES, num_lanes=LANES,
                            model="byte", init="medium", read_noise=0, seed=seed + g),
                       labels=list(range(LANES)), Vt=Vt, N=N)
               for g in range(GROUPS)]
    return oracles, Classifier.from_core(oracles)


def _random_pg_aats(n, rng, none_frac=0.25):
    aats = rng.integers(0, CHANNELS, size=(n, GROUPS, SPACES))
    mask = rng.random((n, GROUPS, SPACES)) < none_frac
    return np.where(mask, -1, aats)


def test_grouped_from_core_lifts_exact_state():
    oracles, grp = _grouped_pair()
    assert grp.num_groups == GROUPS and grp.ga.shape == (GROUPS, LANES, SPACES, CHANNELS)
    for g, oracle in enumerate(oracles):
        ga, gb = oracle_weights(oracle.core)
        assert np.array_equal(grp.ga[g].numpy(), ga)
        assert np.array_equal(grp.gb[g].numpy(), gb)


def test_grouped_read_equals_independent_flats():
    """Shared and per-group reads, through BOTH read paths (embedding-bag table and the
    direct gather), against G separately-lifted flat Classifiers."""
    oracles, grp = _grouped_pair()
    flats = [Classifier.from_core(o) for o in oracles]
    rng = np.random.default_rng(20)

    shared = torch.from_numpy(_random_aats(32, rng))
    y_shared = grp.read_y(shared)                               # bag path
    assert y_shared.shape == (32, GROUPS, LANES)
    import ktram_neural_core.torch._lane as _lane
    t, b = _lane.read_sums(grp.ga, grp.gb, shared)              # direct path
    y_direct = _lane.divide(t, b, torch.float64)
    assert torch.equal(y_shared, y_direct)
    for g, flat in enumerate(flats):
        assert torch.equal(y_shared[:, g], flat.read_y(shared))

    pg = torch.from_numpy(_random_pg_aats(32, rng))
    y_pg = grp.read_y(pg, per_group=True)                       # bag path (per-group table)
    t, b = _lane.read_sums(grp.ga, grp.gb, pg, per_group=True)  # direct path
    assert torch.equal(y_pg, _lane.divide(t, b, torch.float64))
    for g, flat in enumerate(flats):
        assert torch.equal(y_pg[:, g], flat.read_y(pg[:, g]))
    record("Classifier", 1, "grouped read == G independent flat Classifiers, shared and "
                            "per-group AATs, bag table and direct gather paths all equal")


def test_grouped_broadcast_sanity():
    """A shared read is the per-group read with the AAT broadcast to every group."""
    _, grp = _grouped_pair()
    rng = np.random.default_rng(21)
    aat = torch.from_numpy(_random_aats(8, rng))
    pg = aat[:, None, :].expand(8, GROUPS, SPACES)
    assert torch.equal(grp.read_y(aat), grp.read_y(pg, per_group=True))


def test_grouped_adapt_b1_bitexact_vs_oracle():
    """B = 1 per-group adapt: every group walks its own oracle's serial sequence exactly —
    weights and recoded outputs, 200 steps with NONE-heavy per-group AATs."""
    oracles, grp = _grouped_pair(Vt=0.0, N=None)
    rng = np.random.default_rng(22)
    aats = _random_pg_aats(200, rng)
    targets = rng.integers(0, LANES, size=(200, GROUPS))
    for i in range(200):
        outs = [o.adapt(to_oracle_aat(aats[i, g]), {int(targets[i, g])})
                for g, o in enumerate(oracles)]
        t_out = grp.adapt(torch.from_numpy(aats[i]), torch.from_numpy(targets[i]),
                          per_group=True)
        assert t_out.shape == (GROUPS, LANES)
        for g in range(GROUPS):
            assert np.array_equal(t_out[g].numpy(),
                                  oracle_rank_cut_padded(outs[g], LANES)), (i, g)
        if i % 50 == 0 or i == 199:
            for g, o in enumerate(oracles):
                ga, gb = oracle_weights(o.core)
                assert np.array_equal(grp.ga[g].numpy(), ga), (i, g)
                assert np.array_equal(grp.gb[g].numpy(), gb), (i, g)
    record("Classifier", 1, "grouped 200-step B=1 per-group adapt: weights and outputs "
                            "bit-exact vs G independent oracle RankCuts")


def test_grouped_batched_adapt_equals_flat_batched():
    """B > 1: one grouped per-group adapt == each flat Classifier adapting its own slice
    of the batch (stale-batch semantics factor per group)."""
    oracles, grp = _grouped_pair()
    flats = [Classifier.from_core(o) for o in oracles]
    rng = np.random.default_rng(23)
    for step in range(5):
        aats = torch.from_numpy(_random_pg_aats(16, rng))
        targets = torch.from_numpy(rng.integers(0, LANES, size=(16, GROUPS)))
        grp.adapt(aats, targets, per_group=True)
        for g, flat in enumerate(flats):
            flat.adapt(aats[:, g], targets[:, g].reshape(16, 1))
            assert torch.equal(grp.ga[g], flat.ga), (step, g)
            assert torch.equal(grp.gb[g], flat.gb), (step, g)
    record("Classifier", 1, "grouped batched per-group adapt == G flat batched adapts, "
                            "byte-identical weights (16-example batches, 5 steps)")


def test_grouped_shared_input_adapt_equals_flats():
    """per_group=False on a grouped bank: every group taught on the same AAT, each with its
    own target — equals G flats teaching on that AAT."""
    oracles, grp = _grouped_pair()
    flats = [Classifier.from_core(o) for o in oracles]
    rng = np.random.default_rng(24)
    aats = torch.from_numpy(_random_aats(8, rng))
    targets = torch.from_numpy(rng.integers(0, LANES, size=(8, GROUPS)))
    grp.adapt(aats, targets)
    for g, flat in enumerate(flats):
        flat.adapt(aats, targets[:, g].reshape(8, 1))
        assert torch.equal(grp.ga[g], flat.ga), g
        assert torch.equal(grp.gb[g], flat.gb), g


def test_grouped_self_slot_none_is_inert():
    """The completion-bank wiring fact: with group g's own slot always NONE, group g's
    weights for that space are never driven — inert at init, through reads and adapts."""
    grp = Classifier(LANES, SPACES, CHANNELS, num_groups=GROUPS, init="medium", seed=31)
    slot = [g % SPACES for g in range(GROUPS)]
    ga0, gb0 = grp.ga.clone(), grp.gb.clone()
    rng = np.random.default_rng(32)
    for _ in range(20):
        aats = _random_pg_aats(8, rng, none_frac=0.0)
        for g in range(GROUPS):
            aats[:, g, slot[g]] = -1                   # the structural hole
        targets = torch.from_numpy(rng.integers(0, LANES, size=(8, GROUPS)))
        grp.adapt(torch.from_numpy(aats), targets, per_group=True)
    changed = False
    for g in range(GROUPS):
        assert torch.equal(grp.ga[g, :, slot[g]], ga0[g, :, slot[g]]), g
        assert torch.equal(grp.gb[g, :, slot[g]], gb0[g, :, slot[g]]), g
        other = [k for k in range(SPACES) if k != slot[g]]
        changed |= not torch.equal(grp.ga[g][:, other], ga0[g][:, other])
    assert changed                                     # the rest of the bank did learn
    record("Classifier", 1, "self-slot-NONE per-group teach leaves the self space's "
                            "synapses untouched (inert at init) while the rest learn")


def test_grouped_read_sampled_matches_read_at_t0():
    _, grp = _grouped_pair()
    rng = np.random.default_rng(25)
    pg = torch.from_numpy(_random_pg_aats(8, rng))
    out = grp.read_sampled(pg, T=0.0, per_group=True)
    assert torch.equal(out, grp.read(pg, per_group=True))
    noisy = grp.read_sampled(pg, T=0.5, per_group=True,
                             generator=torch.Generator().manual_seed(0))
    assert noisy.shape == (8, GROUPS, LANES)


def test_grouped_pack_round_trip(tmp_path):
    _, grp = _grouped_pair()
    path = tmp_path / "grouped-pack"
    grp.to_pack(path)
    fro = Classifier.from_pack(path)
    assert fro.num_groups == GROUPS and fro.frozen
    rng = np.random.default_rng(26)
    pg = torch.from_numpy(_random_pg_aats(16, rng))
    shared = torch.from_numpy(_random_aats(16, rng))
    # bag/fused path vs direct gather on the same frozen weights
    import ktram_neural_core.torch._lane as _lane
    y_bag = fro.read_y(pg, per_group=True)
    t, b = _lane.read_sums(fro.qdiff, fro.qmag, pg, per_group=True, paired=False)
    y_direct = _lane.divide(t, b, torch.float64, fro.diff_scale, fro.mag_scale)
    assert torch.equal(y_bag, y_direct)
    assert fro.read_y(shared).shape == (16, GROUPS, LANES)
    with pytest.raises(RuntimeError):
        fro.adapt(pg, torch.zeros(16, GROUPS, dtype=torch.int64), per_group=True)
    record("Classifier", 1, "grouped pack round trip: [G, L, K, S] section loads frozen, "
                            "fused/bag read == direct gather, adapt refused")


def test_per_group_requires_grouped_module():
    _, clf = _pair()
    aat = torch.zeros(2, 3, SPACES, dtype=torch.int64)
    with pytest.raises(ValueError):
        clf.read(aat, per_group=True)
    with pytest.raises(ValueError):
        clf.adapt(aat, torch.zeros(2, 3, dtype=torch.int64), per_group=True)


def test_grouped_state_dict_round_trip():
    _, grp = _grouped_pair()
    rng = np.random.default_rng(27)
    grp.adapt(torch.from_numpy(_random_pg_aats(4, rng)),
              torch.from_numpy(rng.integers(0, LANES, size=(4, GROUPS))), per_group=True)
    clone = Classifier(LANES, SPACES, CHANNELS, num_groups=GROUPS, seed=0)
    clone.load_state_dict(grp.state_dict())
    pg = torch.from_numpy(_random_pg_aats(4, rng))
    assert torch.equal(clone.read_y(pg, per_group=True), grp.read_y(pg, per_group=True))
