"""Tier-1 congruence: the torch BasisEncoder against the oracle BasisGroup, bit for bit.

B = 1 adapt streams must leave identical integer state AND identical bookkeeping (winners,
won-bits, cycle counts, throttle/recruit counters) across all three stall responses —
recruit, reset, and freeze — because the stall machinery is where the codebook's character
comes from. Statistical tiers live in test_torch_noise.py and test_torch_congruence.py.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ktram_neural_core import BasisGroup, Core  # noqa: E402
from ktram_neural_core.aat_recoder import BasisEncoder as OracleEncoder  # noqa: E402
from ktram_neural_core.torch import BasisEncoder  # noqa: E402

from _congruence import oracle_weights, record, to_oracle_aat  # noqa: E402

CHANNELS, SPACES, SYMS = 6, 5, 4


def _oracle_group(seed=3, **kwargs):
    core = Core(1, SYMS, spaces_per_lane=SPACES, num_lanes=CHANNELS, model="byte",
                init="low", read_noise=0, seed=seed)
    return BasisGroup(core, CHANNELS, **kwargs)


def _assert_state_equal(enc, groups, step=""):
    for g, grp in enumerate(groups):
        ga, gb = oracle_weights(grp.core)
        assert np.array_equal(enc.ga[g].numpy(), ga), (g, step)
        assert np.array_equal(enc.gb[g].numpy(), gb), (g, step)
        assert enc.win_counts[g].numpy().tolist() == grp.win_counts, (g, step)
        assert int(enc.n_throttled[g]) == grp.n_throttled, (g, step)
        assert int(enc.n_recruited[g]) == grp.n_recruited, (g, step)
        assert int(enc.n_instructions[g]) == grp.n_instructions, (g, step)
        assert int(enc.count[g]) == grp._ga.count, (g, step)
        assert int(enc.cycles[g]) == grp._ga.cycles, (g, step)
        assert enc.cycle_lengths[g] == grp._ga.cycle_lengths, (g, step)
        won = [grp._ga.has_won(i) for i in range(CHANNELS)]
        assert enc.won[g].numpy().tolist() == won, (g, step)


def _run_matched(mode_kwargs, n_steps=400, seed=9, label=""):
    grp = _oracle_group(seed=1, **mode_kwargs)
    enc = BasisEncoder.from_core(grp)
    rng = np.random.default_rng(seed)
    aats = rng.integers(0, SYMS, size=(n_steps, SPACES))
    for i in range(n_steps):
        w_oracle = grp.adapt(to_oracle_aat(aats[i]))
        w_torch = enc.adapt(torch.from_numpy(aats[i]))
        assert int(w_torch[0]) == w_oracle, (label, i)
    _assert_state_equal(enc, [grp], label)


def test_adapt_b1_recruit_mode_bitexact():
    """Recruitment on, the Java default: force-reward keeps the whole width live. The
    recruit correction re-reads post-correction state — the trap this test pins down."""
    _run_matched(dict(gather_abandon=4, exclusion=True, recruitment=True,
                      abandon_action="recruit"), label="recruit")
    record("BasisEncoder", 1, "400-step B=1 adapt, recruit mode (gather_abandon=4): weights, "
                              "winners, and all cycle bookkeeping bit-exact vs oracle")


def test_adapt_b1_reset_mode_bitexact():
    """Recruitment off + reset: the self-pruning codebook."""
    _run_matched(dict(gather_abandon=4, exclusion=True, recruitment=False,
                      abandon_action="reset"), label="reset")
    record("BasisEncoder", 1, "400-step B=1 adapt, reset mode: bit-exact incl. abandoned-"
                              "cycle lengths")


def test_adapt_b1_freeze_mode_bitexact():
    """Recruitment off, no reset: the ablation where exclusion freezes further reward."""
    _run_matched(dict(gather_abandon=4, exclusion=True, recruitment=False,
                      abandon_action="recruit"), label="freeze")
    record("BasisEncoder", 1, "400-step B=1 adapt, freeze ablation: bit-exact")


def test_adapt_b1_exclusion_off_bitexact():
    """Exclusion off: the collapse failure mode must collapse identically."""
    _run_matched(dict(gather_abandon=None, exclusion=False, recruitment=False,
                      abandon_action="recruit"), label="exclusion-off")
    record("BasisEncoder", 1, "400-step B=1 adapt, exclusion off (collapse ablation): bit-exact")


def test_mid_training_config_flip_bitexact():
    """The generator's form-then-sharpen flip (recruitment off + reset, mid-stream) must
    carry the cycle state across the flip exactly."""
    grp = _oracle_group(seed=2, gather_abandon=5, recruitment=True, abandon_action="recruit")
    enc = BasisEncoder.from_core(grp)
    rng = np.random.default_rng(12)
    aats = rng.integers(0, SYMS, size=(300, SPACES))
    for i in range(150):
        assert int(enc.adapt(torch.from_numpy(aats[i]))[0]) == grp.adapt(to_oracle_aat(aats[i]))
    grp.recruitment = False
    grp.abandon_action = "reset"
    enc.recruitment = False
    enc.abandon_action = "reset"
    for i in range(150, 300):
        assert int(enc.adapt(torch.from_numpy(aats[i]))[0]) == grp.adapt(to_oracle_aat(aats[i]))
    _assert_state_equal(enc, [grp], "flip")
    record("BasisEncoder", 1, "form-then-sharpen config flip mid-stream: bit-exact across "
                              "the flip (the generator's schedule)")


def test_multi_group_per_group_aats_bitexact():
    """Three independently-seeded oracle groups, each fed its own AAT (the generator's
    patch feed), against one stacked module with per_group=True."""
    groups = [_oracle_group(seed=100 + g, gather_abandon=4) for g in range(3)]
    enc = BasisEncoder.from_core(groups)
    rng = np.random.default_rng(13)
    aats = rng.integers(0, SYMS, size=(250, 3, SPACES))
    for i in range(250):
        winners = enc.adapt(torch.from_numpy(aats[i]), per_group=True)
        for g, grp in enumerate(groups):
            assert int(winners[g]) == grp.adapt(to_oracle_aat(aats[i, g])), (i, g)
    _assert_state_equal(enc, groups, "per-group")
    record("BasisEncoder", 1, "3-group stacked module, per-group AATs: bit-exact vs three "
                              "independent oracle groups (group axis is free batching)")


def test_shared_input_matches_oracle_encoder():
    """The oracle BasisEncoder semantics: every group reads the same AAT."""
    oracle = OracleEncoder([SYMS] * SPACES, n_groups=2, channels=CHANNELS, model="byte",
                           init="low", read_noise=0, seed=40, gather_abandon=4)
    enc = BasisEncoder.from_core(oracle)
    rng = np.random.default_rng(14)
    aats = rng.integers(0, SYMS, size=(200, SPACES))
    for i in range(200):
        w_oracle = oracle.adapt(to_oracle_aat(aats[i]))
        w_torch = enc.adapt(torch.from_numpy(aats[i]))
        assert tuple(int(w) for w in w_torch) == w_oracle, i
    _assert_state_equal(enc, oracle.groups, "shared")
    record("BasisEncoder", 1, "oracle BasisEncoder (shared input, 2 groups) reproduced "
                              "bit-exact incl. per-group seeding rule")


def test_seeded_constructor_matches_oracle_init():
    oracle = OracleEncoder([SYMS] * SPACES, n_groups=3, channels=CHANNELS, model="byte",
                           init="low", read_noise=0, seed=77)
    enc = BasisEncoder(3, CHANNELS, SPACES, SYMS, init="low", seed=77)
    for g, grp in enumerate(oracle.groups):
        ga, gb = oracle_weights(grp.core)
        assert np.array_equal(enc.ga[g].numpy(), ga)
        assert np.array_equal(enc.gb[g].numpy(), gb)
    record("BasisEncoder", 1, "seeded constructor init == oracle (seed + g per group), "
                              "bit-exact")


def test_read_is_nondisturbing():
    grp = _oracle_group(seed=6)
    enc = BasisEncoder.from_core(grp)
    before = enc.ga.clone()
    enc.read(torch.tensor([0, 1, 2, 3, 0]))
    enc.read_sampled(torch.tensor([0, 1, 2, 3, 0]), T=0.5,
                     generator=torch.Generator().manual_seed(0))
    assert torch.equal(enc.ga, before)


def test_winner_tie_breaks_to_first_lane():
    """Fresh low-init groups are tie-heavy; the argmax must break to the lowest lane, as the
    oracle's max() does. 60 fresh groups (one adapt each) exercise the tie path."""
    for seed in range(60):
        grp = _oracle_group(seed=seed)
        enc = BasisEncoder.from_core(grp)
        aat = np.full(SPACES, seed % SYMS)
        assert int(enc.adapt(torch.from_numpy(aat))[0]) == grp.adapt(to_oracle_aat(aat))
    record("BasisEncoder", 1, "winner ties break to first lane, matching the oracle (60 "
                              "fresh tie-heavy groups)")
