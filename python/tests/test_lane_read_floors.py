"""Lane-read throughput floors (08d deliverable 4) — ratios, never absolute rates.

Both floors compare kernels measured in the same run on the same host, so they hold on
any machine: a slow laptop slows both sides of each ratio. What they catch is a kernel
regression — the default read path silently falling back to (or being replaced by) a
formulation in the wrong kernel family. The margins sit far inside the measured gaps
(the 08d race measured the bag read ~10x the masked gather and ~30x slower than the
matched dense GEMM at this shape; the floors ask only 2x and 100x).
"""
import time

import pytest
import torch
import torch.nn.functional as F

from ktram_neural_core.torch import Classifier
from ktram_neural_core.torch import _lane

L, K, S = 1568, 16, 64          # the generator's decoder shape
B = 1024
REPS = 3


def _median_seconds(fn):
    fn()                                     # warmup
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


@pytest.fixture(scope="module")
def timings():
    torch.manual_seed(0)
    clf = Classifier(num_lanes=L, num_spaces=K, num_channels=S, init="medium", seed=0)
    g = torch.Generator().manual_seed(1)
    aat = torch.randint(0, S, (B, K), generator=g)
    aat[torch.rand(B, K, generator=g) < 0.1] = -1
    W = torch.randn(L, 2 * K, generator=g)
    x = torch.randn(B, 2 * K, generator=g)

    t_default = _median_seconds(lambda: clf.read_y(aat))
    dt = _lane.y_dtype(aat.device)
    t_gather = _median_seconds(
        lambda: _lane.divide(*_lane.read_sums(clf.ga, clf.gb, aat), dt))
    t_dense = _median_seconds(lambda: F.linear(x, W))
    return t_default, t_gather, t_dense


def test_default_read_beats_masked_gather(timings):
    """The default CPU read must stay in the embedding-bag kernel family: at least 2x
    the masked gather-and-sum on the same host (measured ~10x)."""
    t_default, t_gather, _ = timings
    assert t_default * 2.0 <= t_gather, \
        f"default read {t_default * 1e3:.1f} ms is not 2x faster than the " \
        f"masked gather at {t_gather * 1e3:.1f} ms — kernel regression"


def test_default_read_within_dense_envelope(timings):
    """The default CPU read must stay within 100x of the matched dense fp32 GEMM
    (measured ~30x). A fallback to the gather formulation (~300x) fails this."""
    t_default, _, t_dense = timings
    assert t_default <= 100.0 * t_dense, \
        f"default read {t_default * 1e3:.1f} ms vs dense {t_dense * 1e3:.1f} ms — " \
        f"more than 100x slower than the matched GEMM"
