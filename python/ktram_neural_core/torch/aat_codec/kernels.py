"""Read kernels — the per-operation consumers of AAT codes, in two tiers.

A codec is a codebook (fixed, the lane target) + a read kernel (chosen per op at read time).
This module is the kernel menu, and every op that touches a codeword grid comes in two forms:

  * The default (`attention_scores`, `combine`) is the **digital fast path**: decode at the
    GEMM boundary and let the tuned matmul do the contraction. This is exact, not an
    approximation — the per-slot dot table's entries are codeword inner products, so summing
    table reads over slots IS ⟨decode(q), decode(k)⟩, and the histogram combine IS
    α @ decode(V). Codes stay the storage, wire, and cache format (that is where the memory
    shrink lives); floats exist only at the boundary of the one op that wants them. This is
    the same pattern quantized LLM inference uses (codebook weights dequantized into the
    matmul), and "decode only when an op needs floats" applied to hardware that multiplies
    for free.
  * The `*_table` forms are the **symbol-space kernels** — the exact model of what AAT-native
    hardware does: k table reads per pair, a slot sum, never a reconstructed float vector.
    About half the elementary ops of the float op they replace. On lanes and LUT datapaths
    the lookups are the cheap thing; on a CPU/GPU these lose to the GEMM and exist as the
    hardware reference and the small-shape fallback.

  inner / self_norms2 / norm  stay symbol-space even digitally: they are elementwise, with no
                              GEMM to ride. inner measures at native parity (0.86x CPU); norm
                              trails its fused native op on wall clock and stays symbol-space
                              anyway, because there is no faster float path to hand it to.
  add / rope                  produce a new vector off the codeword grid, so they return ℝ.

LUT lifecycle (spec 08 §7): `dot_table` builds a table ONCE; the caller registers it as a
buffer on whatever module composes the two codecs, and the `*_table` kernels take it as a
required argument — no kernel ever rebuilds a table per call. The decode-path kernels read
the codec's own `C` buffer. No per-slot Python loop anywhere.
"""
import torch

from .codec import SLOT_DIM


# --------------------------------------------------------------------- tables (built from codewords)
def dot_table(codec_x, codec_y):
    """Per-slot inner-product table G[m, s_x, s_y] = ⟨C_x[m,s_x], C_y[m,s_y]⟩ -> [k,S_x,S_y].

    Build it once per codec pair and register it as a buffer on the composing module
    (``self.register_buffer("G_qk", dot_table(cq, ck))``) so it moves with ``.to(device)``.
    """
    if codec_x.k != codec_y.k:
        raise ValueError("dot needs matching slot counts")
    return torch.einsum("msd,mtd->mst", codec_x.C, codec_y.C)


# --------------------------------------------------------------------- inner product / scores
def inner(codes_x, codes_y, table):
    """Elementwise inner product ⟨x,y⟩ over matching-shape codes -> [...]. One fused gather of
    the dot table (k reads per pair) and a slot sum; no reconstruction. Symbol-space is the
    fast digital form here too: the op is elementwise, so there is no GEMM to hand it to."""
    idx0 = torch.arange(table.shape[0], device=table.device)
    return table[idx0, codes_x, codes_y].sum(-1)


def attention_scores(codec_q, codes_q, codec_k, codes_k, scaling=1.0):
    """Q·K attention scores. codes_q [..., Tq, k], codes_k [..., Tk, k] -> [..., Tq, Tk]
    (× scaling). The digital fast path: decode both sides at the GEMM boundary and contract
    on the native matmul — numerically identical to the per-slot table sum
    (``attention_scores_table``), at GEMM speed. A caller reusing keys across calls (a KV
    cache) should decode them once and keep the float keys; only the queries then decode
    per call."""
    q = codec_q.decode(codes_q)
    if scaling != 1.0:
        q = q * scaling                    # fold into the small side, not the [Tq, Tk] scores
    return q @ codec_k.decode(codes_k).transpose(-1, -2)


def attention_scores_table(codes_q, codes_k, table, scaling=1.0):
    """The symbol-space form of `attention_scores` — what AAT-native hardware runs: read the
    per-slot dot table for every (query, key) symbol pair and sum over slots, never touching
    a float vector. Gathers each query's k table rows, lifts the keys to one-hot, and
    contracts with one batched matmul; on CPU/GPU the k·S-wide inner dimension makes this
    lose to the decode path by the lift ratio (k·S / D)."""
    k, Sq, Sk = table.shape
    lead = codes_q.shape[:-2]
    Tq, Tk = codes_q.shape[-2], codes_k.shape[-2]
    q = codes_q.reshape(-1, Tq, k)
    kk = codes_k.reshape(-1, Tk, k)
    idx0 = torch.arange(k, device=table.device)
    q_rows = table[idx0, q].reshape(-1, Tq, k * Sk)                # [n, Tq, k*Sk]
    k_lift = torch.zeros(kk.shape[0], Tk, k, Sk, device=table.device, dtype=table.dtype)
    k_lift.scatter_(-1, kk.unsqueeze(-1), 1.0)
    score = q_rows @ k_lift.reshape(-1, Tk, k * Sk).transpose(1, 2)
    return score.reshape(*lead, Tq, Tk) * scaling


# --------------------------------------------------------------------- norm / length
def self_norms2(codec, codes):
    """Squared length Σ_slot ‖codeword‖² -> [...]. Reads the codec's norm2 buffer (no decode)."""
    return codec.self_norms2(codes)


def norm(codec, codes):
    """Vector length -> [...]. Table read + sum + sqrt."""
    return codec.self_norms2(codes).clamp(min=0).sqrt()


# --------------------------------------------------------------------- value combine (kernel M)
def combine(codec_v, codes_v, weights):
    """Weighted value combine Σ_j α_ij V̂_j -> float output [n,Tq,D]. The digital fast path:
    α @ decode(V) on the native matmul — numerically identical to the histogram kernel
    (``combine_table``). weights [n,Tq,Tk], codes_v [n,Tk,k]. The output is float either
    way: a blend of symbols is not itself a symbol."""
    return weights @ codec_v.decode(codes_v)


def combine_table(codec_v, codes_v, weights):
    """The symbol-space form of `combine` (Codec M's histogram kernel) — what AAT-native
    hardware runs. For each slot, scatter α into a per-symbol histogram over the alphabet,
    then read it out against the book:
        z_i[m] = Σ_j α_ij C_m[c_jm] = Σ_s ( Σ_{j:c_jm=s} α_ij ) C_m[s]
    No value is ever decoded; the blend itself is the one unavoidable float."""
    n, Tq, Tk = weights.shape
    k, S = codec_v.k, codec_v.S
    v_lift = torch.zeros(n, Tk, k, S, device=weights.device, dtype=weights.dtype)
    v_lift.scatter_(-1, codes_v.unsqueeze(-1), 1.0)
    H = weights @ v_lift.reshape(n, Tk, k * S)                     # [n, Tq, k*S] histograms
    z = torch.einsum("bqms,msd->bqmd", H.reshape(n, Tq, k, S), codec_v.C.to(weights.dtype))
    return z.reshape(n, Tq, k * SLOT_DIM)


# --------------------------------------------------------------------- residual add (float)
def add(codec_x, codes_x, codec_y, codes_y):
    """Residual add x + y -> FLOAT vector [...,D]. A residual sum is off the codeword grid; the
    assimilation datapath adds it into the float residual accumulator (`h += decode(aat)`)."""
    return codec_x.decode(codes_x) + codec_y.decode(codes_y)


# --------------------------------------------------------------------- RoPE (provisional)
def rope(codec, codes, angles):
    """Rotate each slot's codeword by its per-slot angle -> FLOAT vector [...,D]. codes [...,k],
    angles [...,k]. A rotated codeword is off-grid; if a lane consumes it the caller snaps it."""
    A = codec.decode_atoms(codes)
    c, s = torch.cos(angles), torch.sin(angles)
    x, y = A[..., 0], A[..., 1]
    rot = torch.stack([x * c - y * s, x * s + y * c], dim=-1)
    return rot.reshape(*rot.shape[:-2], codec.k * SLOT_DIM)
