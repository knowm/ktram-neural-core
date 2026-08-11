"""Tier-2 congruence: the read-noise law, sigma for sigma.

``read_sampled`` must reproduce sigma(m, y) exactly as the oracle factors it — the same
closed-form Ch3b law the export validates — verified on the sigma curves at matched
(m, y, T) and on winner statistics over a trained group. Draw-for-draw RNG matching against
numpy is explicitly a non-goal (spec 08 §8): the law is the contract, not the stream.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ktram_neural_core import BasisGroup, Core  # noqa: E402
from ktram_neural_core.torch import BasisEncoder, NoiseParams  # noqa: E402

from _congruence import record, to_oracle_aat  # noqa: E402


def _oracle_sigma(core, y, m, v_app):
    """The oracle's sigma at read_noise gain T, computed from its cached coefficients —
    Core.read_sample's arithmetic without the draw."""
    import math
    f_m = core._sqrt_ref_m / math.sqrt(m)
    bw_thermal = math.sqrt(core.read_noise_ref_pw / core.read_pulse_width)
    ln_band = core._flicker_ln_ref + math.log(core.read_noise_ref_pw / core.read_pulse_width)
    bw_flicker = math.sqrt(ln_band / core._flicker_ln_ref) if ln_band > 0 else 0.0
    s_th = core._a_thermal * f_m / abs(v_app) * bw_thermal
    s_fl = core._a_flicker * (1.0 - y * y) * f_m * bw_flicker
    return math.sqrt(s_th * s_th + s_fl * s_fl)


def test_sigma_unit_matches_oracle_curves():
    """T * sigma_unit(m, y) == the oracle's sigma across the (m, y, T) grid, at both the
    default and a shortened read pulse width."""
    for read_pw in (1e-6, 2e-7):
        core = Core(1, 8, spaces_per_lane=4, num_lanes=1, model="byte", seed=0,
                    read_pulse_width=read_pw)
        params = NoiseParams.from_core(core)
        v = abs(core.forward_low_voltage)
        for T in (0.1, 0.5, 2.0):
            core.set_read_noise(read_noise=T)   # the oracle's cached gains now carry T
            for m in (4.0, 200.0, 2040.0):
                for y in (-0.99, -0.5, 0.0, 0.3, 0.9):
                    want = _oracle_sigma(core, y, m, v)
                    got = float(T * params.sigma_unit(
                        torch.tensor(y, dtype=torch.float64),
                        torch.tensor(m, dtype=torch.float64)))
                    assert got == pytest.approx(want, rel=1e-12), (read_pw, T, m, y)
    record("both modules", 2, "sigma(m, y, T) == oracle Core.read_sample law across the "
                              "grid (two read pulse widths), rel 1e-12")


def test_sampled_read_std_matches_law():
    """The empirical std of the sampled read matches T * sigma_unit at matched (m, y).
    T is kept small enough that the [-1, 1] clip never engages — the clip is part of the
    law, but a truncated distribution's std is not the sigma being checked here."""
    params = NoiseParams()
    y0, m0, T = 0.3, 300.0, 0.1
    n = 40000
    gen = torch.Generator().manual_seed(0)
    from ktram_neural_core.torch._lane import sample_read
    draws = sample_read(torch.full((n,), y0, dtype=torch.float64),
                        torch.full((n,), m0, dtype=torch.float64), T, params, gen)
    want = T * float(params.sigma_unit(torch.tensor(y0), torch.tensor(m0)))
    got = float(draws.std())
    assert got == pytest.approx(want, rel=0.02)
    record("both modules", 2, f"empirical sampled-read std {got:.5f} vs law {want:.5f} "
                              f"(n=40k, rel tol 2%)")


def test_winner_statistics_match_oracle_at_temperature():
    """Sampled-winner distributions over a trained group agree between oracle and torch at
    matched temperature — the statistic that matters for generation."""
    core = Core(1, 4, spaces_per_lane=6, num_lanes=8, model="byte", init="low",
                read_noise=0, seed=21)
    grp = BasisGroup(core, 8, gather_abandon=8)
    rng = np.random.default_rng(2)
    train = rng.integers(0, 4, size=(600, 6))
    for a in train:
        grp.adapt(to_oracle_aat(a))
    enc = BasisEncoder.from_core(grp)

    T = 0.5
    probe = to_oracle_aat(train[0])
    n = 3000
    core.set_read_noise(read_noise=T)
    oracle_hist = np.zeros(8)
    for _ in range(n):
        y = grp.read_scores(probe)
        oracle_hist[int(np.argmax(y))] += 1
    gen = torch.Generator().manual_seed(3)
    aat = torch.from_numpy(train[0].astype(np.int64))
    torch_wins = torch.stack([enc.read_sampled(aat, T, gen) for _ in range(n)])
    torch_hist = np.bincount(torch_wins.flatten().numpy(), minlength=8).astype(float)

    tv = 0.5 * np.abs(oracle_hist / n - torch_hist / n).sum()
    assert tv < 0.05, (oracle_hist, torch_hist)
    record("BasisEncoder", 2, f"sampled-winner distributions at T=0.5: total-variation "
                              f"distance {tv:.3f} over 3000 draws/side (< 0.05)")
