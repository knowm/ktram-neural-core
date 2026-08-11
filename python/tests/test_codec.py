"""The AAT Codec battery (spec 08 §7) — the ported, vectorized codec against three anchors.

1. Reference parity: the batched encode and the vectorized kernels reproduce the plain
   per-slot / per-group loop arithmetic (the staged implementation's form) on the same fit.
2. Frozen numbers: op counts and fidelity on a pinned synthetic distribution stay where the
   codec search left them — a regression here means the codec changed, not the data.
3. Production hygiene: every LUT is a registered buffer that survives ``state_dict`` and
   ``.to(device)``; kernels never build a table; fitting touches no global RNG.
"""
import math

import pytest
import torch

from ktram_neural_core.torch import aat_codec
from ktram_neural_core.torch.aat_codec import (AATCodec, add, attention_scores,
                                               attention_scores_table, combine,
                                               combine_table, dot_table, inner, kernels,
                                               norm, rope, self_norms2)

BITS, K, D = 6, 8, 64


# --------------------------------------------------------------------------- fixtures
def _data(n=20000, d=D, seed=0):
    """Correlated, per-dim-scaled Gaussian: routing and bit splits are non-trivial on it."""
    g = torch.Generator().manual_seed(seed)
    scale = torch.exp(1.2 * torch.randn(d, generator=g))
    W = torch.randn(d, d, generator=g) / d ** 0.5
    train = (torch.randn(n, d, generator=g) @ W) * scale
    test = (torch.randn(4000, d, generator=g) @ W) * scale
    return train, test


@pytest.fixture(scope="module")
def fitted():
    train, test = _data()
    return AATCodec(BITS, K=K).fit(train), train, test


# --------------------------------------------------------------------------- reference loops
def _encode_loop(codec, X):
    """The staged implementation's encode: per-group bucketize over unpadded edges."""
    U = aat_codec.atoms(X, codec.k) @ codec.R.T
    codes = torch.empty(*X.shape[:-1], codec.k, dtype=torch.long)
    for m in range(codec.k):
        su, sv = int(codec.su[m]), int(codec.sv[m])
        eu = codec.edges_u[m][: su - 1]
        ev = codec.edges_v[m][: sv - 1]
        au = torch.bucketize(U[..., m, 0].contiguous(), eu)
        av = torch.bucketize(U[..., m, 1].contiguous(), ev)
        codes[..., m] = au * sv + av
    return codes


def _inner_loop(codes_x, codes_y, table):
    out = None
    for m in range(table.shape[0]):
        term = table[m][codes_x[..., m], codes_y[..., m]]
        out = term if out is None else out + term
    return out


def _scores_loop(codes_q, codes_k, table, scaling=1.0):
    score = None
    for m in range(table.shape[0]):
        term = table[m][codes_q[..., m].unsqueeze(-1), codes_k[..., m].unsqueeze(-2)]
        score = term if score is None else score + term
    return score * scaling


def _combine_loop(codec_v, codes_v, weights):
    n, Tq, Tk = weights.shape
    k, S = codec_v.k, codec_v.S
    z = torch.empty(n, Tq, k, 2)
    for m in range(k):
        H = torch.zeros(n, Tq, S)
        H.scatter_add_(2, codes_v[..., m].unsqueeze(1).expand(n, Tq, Tk), weights)
        z[:, :, m, :] = H @ codec_v.C[m]
    return z.reshape(n, Tq, k * 2)


# --------------------------------------------------------------------------- interface
def test_package_exposes_codec_and_the_six_kernels():
    from ktram_neural_core.torch import aat_codec as pkg
    assert pkg.AATCodec is AATCodec and issubclass(AATCodec, torch.nn.Module)
    for op in (inner, attention_scores, norm, combine, add, rope,
               attention_scores_table, combine_table):
        assert getattr(pkg, op.__name__) is op


def test_shapes_and_leading_batch_axes(fitted):
    codec, _, test = fitted
    codes = codec.encode(test[:60].reshape(3, 4, 5, D))
    assert codes.shape == (3, 4, 5, codec.k) and codes.dtype == torch.int64
    assert codes.min() >= 0 and codes.max() < codec.S
    assert codec.decode(codes).shape == (3, 4, 5, D)
    lift = codec.onehot_lift(codes)
    assert lift.shape == (3, 4, 5, codec.k, codec.S)
    assert codec.onehot_lift(codes, flatten=True).shape == (3, 4, 5, codec.lift_width())


# --------------------------------------------------------------------------- parity
def test_batched_encode_matches_per_slot_loop(fitted):
    codec, _, test = fitted
    assert torch.equal(codec.encode(test), _encode_loop(codec, test))


def test_decode_is_linear_in_the_onehot_lift(fitted):
    codec, _, test = fitted
    codes = codec.encode(test[:256])
    via_lift = torch.einsum("nks,ksd->nkd", codec.onehot_lift(codes), codec.C)
    assert torch.allclose(codec.decode(codes), via_lift.reshape(-1, D))


def test_self_norms2_table_matches_decode(fitted):
    codec, _, test = fitted
    codes = codec.encode(test[:256])
    assert torch.allclose(codec.self_norms2(codes), codec.decode(codes).pow(2).sum(-1),
                          rtol=1e-4, atol=1e-3)
    assert torch.allclose(self_norms2(codec, codes), codec.self_norms2(codes))
    assert torch.allclose(norm(codec, codes), codec.decode(codes).norm(dim=-1),
                          rtol=1e-4, atol=1e-3)


def test_vectorized_kernels_match_loop_references(fitted):
    codec, train, test = fitted
    g = torch.Generator().manual_seed(1)
    other = AATCodec(BITS, K=K).fit(train + 0.1 * torch.randn(train.shape, generator=g))
    n, Tq, Tk = 3, 37, 53
    q = codec.encode(test[: n * Tq].reshape(n, Tq, D))
    kc = other.encode(test[n * Tq: n * Tq + n * Tk].reshape(n, Tk, D))
    G = dot_table(codec, other)

    assert torch.allclose(attention_scores_table(q, kc, G, scaling=0.3),
                          _scores_loop(q, kc, G, scaling=0.3), atol=1e-3)
    assert torch.allclose(inner(q, kc[:, :Tq], G), _inner_loop(q, kc[:, :Tq], G), atol=1e-3)
    w = torch.softmax(torch.randn(n, Tq, Tk, generator=g), dim=-1)
    assert torch.allclose(combine_table(other, kc, w), _combine_loop(other, kc, w), atol=1e-4)

    # inner against the dot of decodes — the table IS the decode inner product
    assert torch.allclose(inner(q, kc[:, :Tq], G),
                          (codec.decode(q) * other.decode(kc[:, :Tq])).sum(-1), atol=1e-2)
    assert torch.allclose(add(codec, q, other, kc[:, :Tq]),
                          codec.decode(q) + other.decode(kc[:, :Tq]))
    ang = torch.randn(n, Tq, codec.k, generator=g)
    rot = rope(codec, q, ang)
    assert torch.allclose(rot.norm(dim=-1), codec.decode(q).norm(dim=-1), rtol=1e-4, atol=1e-4)
    assert torch.allclose(rope(codec, q, torch.zeros_like(ang)), codec.decode(q), atol=1e-6)


def test_decode_path_equals_symbol_path(fitted):
    """The load-bearing identity behind the digital fast path: the table sum over slots IS
    the inner product of the decodes, so decode+GEMM and the symbol-space kernel are the
    same operation (float summation order aside)."""
    codec, train, test = fitted
    g = torch.Generator().manual_seed(3)
    other = AATCodec(BITS, K=K).fit(train + 0.1 * torch.randn(train.shape, generator=g))
    n, Tq, Tk = 3, 37, 53
    q = codec.encode(test[: n * Tq].reshape(n, Tq, D))
    kc = other.encode(test[n * Tq: n * Tq + n * Tk].reshape(n, Tk, D))
    G = dot_table(codec, other)

    assert torch.allclose(attention_scores(codec, q, other, kc, scaling=0.3),
                          attention_scores_table(q, kc, G, scaling=0.3), atol=1e-3)
    w = torch.softmax(torch.randn(n, Tq, Tk, generator=g), dim=-1)
    assert torch.allclose(combine(other, kc, w), combine_table(other, kc, w), atol=1e-4)


# --------------------------------------------------------------------------- frozen numbers
def test_pinned_op_counts():
    codec = AATCodec(BITS, K=K)
    codec.k, codec.D = D // 2, D                     # op counts need only the shape
    k = D // 2
    assert codec.op_counts("encode") == {"lookups": 2 * k, "mults": 2 * k, "adds": 2 * k}
    assert codec.op_counts("decode") == {"lookups": k, "mults": 0, "adds": 0}
    assert codec.op_counts("norm") == {"lookups": k, "mults": 0, "adds": k - 1, "sqrt": 1}
    assert codec.op_counts("dot") == {"lookups": k, "mults": 0, "adds": k - 1}
    fused = AATCodec(BITS, K=K, fuse_rmsnorm=True)
    fused.k, fused.D = k, D
    assert fused.op_counts("encode") == {"lookups": 2 * k, "mults": 3 * D + 2 * k,
                                         "adds": D + 2 * k, "sqrt": 1}
    with pytest.raises(KeyError):
        codec.op_counts("softmax")


def test_pinned_accounting(fitted):
    codec, _, _ = fitted
    assert codec.wire_bytes() == codec.k * BITS / 8.0
    lut = codec.lut_bytes()
    assert lut["total"] == (lut["decode_values_bytes"] + lut["encode_edges_bytes"]
                            + lut["route_bytes"])
    assert 0 < lut["total"] < 16384                  # tables stay tiny — the design's point


def test_pinned_fidelity(fitted):
    """The frozen fidelity numbers on the pinned distribution (measured at port time).
    bits=6, K=8: reconstruction cosine 0.973, score correlation 0.978, mean magnitude 1.006."""
    codec, _, test = fitted
    recon = codec.decode(codec.encode(test))
    cos = torch.nn.functional.cosine_similarity(test, recon, dim=-1)
    assert abs(cos.mean().item() - 0.9734) < 0.02
    assert torch.quantile(cos, 0.10).item() > 0.90
    mag = recon.norm(dim=-1) / test.norm(dim=-1)
    assert abs(mag.mean().item() - 1.006) < 0.02

    q, kk = test[:512], test[512:1024]
    est = attention_scores(codec, codec.encode(q).unsqueeze(0),
                           codec, codec.encode(kk).unsqueeze(0))[0]
    true = q @ kk.T
    corr = torch.corrcoef(torch.stack([true.flatten(), est.flatten()]))[0, 1]
    assert abs(corr.item() - 0.9785) < 0.02


def test_low_bit_fidelity_still_orders():
    """bits=4 on the same distribution: cosine 0.925, score correlation 0.915 (pinned)."""
    train, test = _data()
    codec = AATCodec(4, K=K).fit(train)
    cos = torch.nn.functional.cosine_similarity(test, codec.decode(codec.encode(test)), dim=-1)
    assert abs(cos.mean().item() - 0.9253) < 0.02


# --------------------------------------------------------------------------- fused RMSNorm
def test_fused_rmsnorm_folds_the_front_end():
    train, test = _data()
    g = torch.Generator().manual_seed(2)
    gamma = torch.rand(D, generator=g) + 0.5
    xhat = train * torch.rsqrt(train.pow(2).mean(-1, keepdim=True) + aat_codec.RMS_EPS) * gamma
    codec = AATCodec(BITS, K=K, fuse_rmsnorm=True).fit(xhat, gamma=gamma)
    that = test * torch.rsqrt(test.pow(2).mean(-1, keepdim=True) + aat_codec.RMS_EPS) * gamma
    assert torch.equal(codec.encode(test), codec.encode_normalized(that))
    assert "gamma" in codec.state_dict()


def test_fused_rmsnorm_requires_gamma():
    train, _ = _data(n=2000)
    with pytest.raises(ValueError):
        AATCodec(BITS, K=K, fuse_rmsnorm=True).fit(train)


# --------------------------------------------------------------------------- production hygiene
def test_luts_are_buffers_in_state_dict(fitted):
    codec, _, _ = fitted
    sd = codec.state_dict()
    for name in ("R", "assign", "su", "sv", "edges_u", "edges_v", "C", "norm2"):
        assert name in sd, f"{name} missing from state_dict"
    assert not any(p.requires_grad for p in codec.parameters())
    assert len(list(codec.parameters())) == 0        # buffers only, no autograd anywhere


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a second device")
def test_luts_survive_a_device_round_trip(fitted):
    codec, _, test = fitted
    cpu_codes = codec.encode(test[:64])
    codec.to("mps")
    try:
        assert codec.C.device.type == "mps" and codec.edges_u.device.type == "mps"
        mps_codes = codec.encode(test[:64].to("mps"))
        assert torch.equal(mps_codes.cpu(), cpu_codes)
    finally:
        codec.to("cpu")
    assert codec.C.device.type == "cpu"
    assert torch.equal(codec.encode(test[:64]), cpu_codes)


def test_kernels_take_the_table_never_build_it(fitted):
    codec, _, test = fitted
    codes = codec.encode(test[:8]).unsqueeze(0)
    with pytest.raises(TypeError):
        attention_scores_table(codes, codes)         # no table arg -> refuse, never rebuild
    with pytest.raises(TypeError):
        inner(codes, codes)


def test_fit_touches_no_global_rng():
    train, _ = _data(n=4000)
    torch.manual_seed(123)
    before = torch.randn(4)
    torch.manual_seed(123)
    AATCodec(BITS, K=K).fit(train)
    assert torch.equal(torch.randn(4), before)       # fit consumed nothing from the global stream


def test_fit_is_deterministic():
    train, test = _data(n=4000)
    a = AATCodec(BITS, K=K).fit(train)
    b = AATCodec(BITS, K=K).fit(train)
    assert torch.equal(a.C, b.C) and torch.equal(a.encode(test), b.encode(test))
