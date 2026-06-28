"""encode/ — A2D binning, encode vs encode_adapt, space_sizes, compose, bias width."""

import pytest

from ktram_neural_core.encode import A2DEncoder, ConstantEncoder, compose


def test_a2d_space_sizes_and_active_count():
    enc = A2DEncoder(dims=4, bits=5)
    assert enc.space_sizes == [32, 32, 32, 32]
    aat = enc.encode([0.1, 0.2, 0.3, 0.4])
    assert len(aat) == 4                       # one active entry per dimension
    assert all(0 <= b < 32 for b in aat)


def test_a2d_bins_are_monotonic_in_value():
    # On a fixed [0, 1] tree, a larger value never lands in a lower bin.
    enc = A2DEncoder(dims=1, bits=4, init_min=0.0, init_max=1.0)
    bins = [enc.encode([x])[0] for x in [0.0, 0.25, 0.5, 0.75, 0.999]]
    assert bins == sorted(bins)
    assert bins[0] == 0 and bins[-1] == (1 << 4) - 1


def test_a2d_per_dim_ranges():
    # Different ranges per dimension: each dimension bins within its own [min, max].
    enc = A2DEncoder(dims=2, bits=3, init_min=[0.0, 100.0], init_max=[1.0, 200.0])
    assert enc.encode([0.0, 100.0]) == (0, 0)
    assert enc.encode([0.999, 199.0]) == (7, 7)


def test_a2d_encode_adapt_migrates_then_encode_is_frozen():
    enc = A2DEncoder(dims=1, bits=5, init_min=0.0, init_max=1.0, l=0.1)
    before = enc.encode([0.5])
    for _ in range(200):                       # feed a stream skewed high
        enc.encode_adapt([0.9])
    after = enc.encode([0.5])
    assert after != before                     # thresholds moved (bins migrated)
    frozen = enc.encode([0.5])
    enc.encode([0.5])
    assert enc.encode([0.5]) == frozen         # plain encode never migrates


def test_constant_encoder_is_the_bias():
    enc = ConstantEncoder(channel=0, count=3)
    assert enc.encode("ignored") == (0, 0, 0)
    assert enc.encode_adapt("ignored") == (0, 0, 0)   # non-adaptive: encode_adapt == encode
    assert enc.space_sizes == [1, 1, 1]               # count sets the bias width


def test_constant_channel_out_of_range_raises():
    with pytest.raises(ValueError):
        ConstantEncoder(channel=2, count=1, space_size=1)


def test_compose_concatenates_aats_and_space_sizes():
    enc = compose(A2DEncoder(dims=4, bits=5), ConstantEncoder(count=2))
    aat = enc.encode([5.1, 3.5, 1.4, 0.2])
    assert len(aat) == 6                              # 4 features + 2 bias spaces
    assert aat[4:] == (0, 0)
    assert enc.space_sizes == [32, 32, 32, 32, 1, 1]
