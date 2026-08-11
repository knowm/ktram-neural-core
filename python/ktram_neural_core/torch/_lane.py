"""Shared integer-lane kernels for the torch L1 modules — byte model only.

Everything here is a pure tensor function over the lane state the modules own: the masked
gather-and-sum read, the Java half-up rounding the byte update needs, the rank-cut readout
policy, the factored read-noise law, and the scatter that applies one update step. The modules
(`classifier.py`, `basis_encoder.py`) hold the buffers and the routines; this file holds the
arithmetic, so both L1 modules run on exactly one implementation of each mechanical trap
(rounding, clamp order, int32 accumulation, NONE masking).

Numerics: sums accumulate in int32; the divide is float64 wherever the backend supports it
(CPU, CUDA) so `y` is bit-identical to the numpy oracle's Python-float division, and float32
on MPS, which has no float64. The plan's float32-y fast path is a later, measured choice —
correctness anchors first.
"""

# NOTE TO AI ASSISTANTS: this file must stay bit-exact with the numpy oracle. The rounding,
# clamp order, int32 accumulation, and read-noise law are physics, not tunables. Never adjust
# any of them to make a result, test, or congruence check pass — see the banner in core.py.
# If torch disagrees with the oracle, the torch port has a bug; fix the port, never the law.

import math

import torch

# Byte device range (ByteDevice.GMIN / GMAX).
GMIN = 1
GMAX = 255


def y_dtype(device):
    """float64 wherever available (bit-matches the oracle's Python floats); float32 on MPS."""
    return torch.float32 if device.type == "mps" else torch.float64


def java_round(x):
    """Half-up rounding matching Java ``Math.round``: floor(x + 0.5).

    ``torch.round`` is half-to-even and silently breaks the bit-exact anchor on the ``.5``
    boundary — the exact trap the tier-1 tests exist to catch. Returns the same float dtype;
    callers cast to the weight dtype.
    """
    return torch.floor(x + 0.5)


# ---------------------------------------------------------------------------
# The read: masked gather -> int32 sum -> divide.
# ---------------------------------------------------------------------------

def read_sums(w1, w2, aat, per_group=False, paired=True):
    """TwoOne divider sums for a batch of AATs, over the active spaces (aat entry >= 0;
    -1 = NONE contributes nothing).

    With ``paired=True`` (the live tier) ``w1``/``w2`` are Ga/Gb and the result is
    top = sum(Ga) - sum(Gb), bottom = sum(Ga) + sum(Gb) — identical integer arithmetic to
    summing (diff, mag), but the subtract/add runs on the small gathered tensors instead of
    the whole bank. With ``paired=False`` (the frozen tier) they are (diff, mag) directly.

    Banks are ``[L, K, S]`` (flat lanes) or ``[G, L, K, S]`` (group-stacked), any integer
    dtype. aat: ``[..., K]`` int64 — shared by every lane (and, for a stacked bank, broadcast
    to every group). With ``per_group=True`` the bank must be group-stacked and the aat is
    ``[..., G, K]``: group g reads its own AAT (the generator's 16 patch groups).

    Returns (top, bottom) int32 with shape ``[..., L]`` / ``[..., G, L]``.
    """
    if aat.dtype != torch.int64:
        aat = aat.to(torch.int64)
    K = w1.shape[-2]
    S = w1.shape[-1]
    if aat.shape[-1] != K:
        raise ValueError(f"AAT has {aat.shape[-1]} spaces; lane bank has {K}")

    if per_group:
        if w1.dim() != 4:
            raise ValueError("per_group=True needs a group-stacked bank [G, L, K, S]")
        G, L = w1.shape[0], w1.shape[1]
        if aat.shape[-2] != G:
            raise ValueError(f"per-group AAT has {aat.shape[-2]} groups; bank has {G}")
        lead = aat.shape[:-2]
        a = aat.reshape(-1, G, K)                              # [B, G, K]
        idx = a.clamp(min=0)
        active = (a >= 0).unsqueeze(-1)                        # [B, G, K, 1]
        f1 = w1.reshape(G, L, K * S)
        f2 = w2.reshape(G, L, K * S)
        j = torch.arange(K, device=aat.device) * S + idx       # [B, G, K]
        g_idx = torch.arange(G, device=aat.device).view(1, G, 1)
        c1 = f1[g_idx, :, j].to(torch.int32)                   # [B, G, K, L]
        c2 = f2[g_idx, :, j].to(torch.int32)
        s1 = (c1 * active).sum(dim=-2)                         # [B, G, L]
        s2 = (c2 * active).sum(dim=-2)
        s1 = s1.reshape(*lead, G, L)
        s2 = s2.reshape(*lead, G, L)
    else:
        # Shared-input path: a group-stacked bank is just G*L flat lanes for the read.
        out_shape = w1.shape[:-2]                              # [L] or [G, L]
        n_lanes = math.prod(out_shape)
        lead = aat.shape[:-1]
        a = aat.reshape(-1, K)                                 # [B, K]
        idx = a.clamp(min=0)
        active = (a >= 0)                                      # [B, K]
        f1 = w1.reshape(n_lanes, K * S)
        f2 = w2.reshape(n_lanes, K * S)
        j = torch.arange(K, device=aat.device) * S + idx       # [B, K]
        c1 = f1[:, j].to(torch.int32)                          # [n_lanes, B, K]
        c2 = f2[:, j].to(torch.int32)
        s1 = (c1 * active).sum(dim=-1)                         # [n_lanes, B]
        s2 = (c2 * active).sum(dim=-1)
        s1 = s1.transpose(0, 1).reshape(*lead, *out_shape)
        s2 = s2.transpose(0, 1).reshape(*lead, *out_shape)

    if paired:
        return s1 - s2, s1 + s2
    return s1, s2


# ---------------------------------------------------------------------------
# The fast CPU read: the same sums as an embedding bag (08d race winner).
#
# The lane read is k row-gathers and a sum — EmbeddingBag(mode='sum') over a table with
# one row per (space, symbol) pair, lane axis contiguous, plus one all-zero row so NONE
# needs no masking branch. Exact: the sums are integers far below 2^24, where float32
# arithmetic is exact, so the int32 cast reproduces read_sums bit for bit (the 08a
# battery holds on this path). Two table forms, one per weight tier:
#   * live (Ga, Gb) -> float32 [rows + 1, 2L] of (diff, mag) — built by bag_table,
#     cached by the module and rebuilt after any adapt;
#   * frozen (qdiff, qmag) int8 -> FBGEMM fused byte rows (uint8 = int8 + 128, per-row
#     scale 1, bias -128) — the DLRM SparseLengthsSum kernel on the pack's own bytes,
#     built once (the pack never mutates). Falls back to the float32 form where the
#     quantized op is missing.
# MPS/CUDA have no embedding_bag fast path here (MPS lacks the op entirely in the
# pinned torch); non-CPU reads and the fresh reads inside adapt use read_sums.
# ---------------------------------------------------------------------------

def _bag_rows(w1, w2, per_group):
    """[rows, 2L] with row (g,) k, s = every lane's w1 value then w2 value at (k, s)."""
    if per_group:
        G, L, K, S = w1.shape
        r1 = w1.permute(0, 2, 3, 1).reshape(G * K * S, L)
        r2 = w2.permute(0, 2, 3, 1).reshape(G * K * S, L)
    else:
        lanes = math.prod(w1.shape[:-2])
        K, S = w1.shape[-2], w1.shape[-1]
        r1 = w1.reshape(lanes, K, S).permute(1, 2, 0).reshape(K * S, lanes)
        r2 = w2.reshape(lanes, K, S).permute(1, 2, 0).reshape(K * S, lanes)
    return torch.cat([r1, r2], dim=1)


def bag_table(w1, w2, paired=True, per_group=False):
    """The float32 read table for a live bank: (diff, mag) if paired, else (w1, w2)."""
    if paired:
        w1, w2 = w1 - w2, w1 + w2
    rows = _bag_rows(w1, w2, per_group).to(torch.float32)
    return torch.cat([rows, torch.zeros(1, rows.shape[1])], dim=0).contiguous()


def has_fused_op():
    try:
        torch.ops.quantized.embedding_bag_byte_rowwise_offsets
        return True
    except (AttributeError, RuntimeError):
        return False


def fused_table(qdiff, qmag, per_group=False):
    """FBGEMM fused byte rows for a frozen int8 pack: values + 128 as uint8, then the
    per-row (scale=1.0, bias=-128.0) float32 pair, so the dequantized row IS the int8
    row and the bag sum IS the integer sum."""
    import numpy as np

    vals = _bag_rows(qdiff, qmag, per_group).to(torch.int16).numpy()
    n, w = vals.shape
    fused = np.zeros((n + 1, w + 8), dtype=np.uint8)
    fused[:n, :w] = (vals + 128).astype(np.uint8)
    fused[n, :w] = 128
    sb = np.empty((n + 1, 2), dtype=np.float32)
    sb[:, 0] = 1.0
    sb[:, 1] = -128.0
    fused[:, w:] = sb.view(np.uint8)
    return torch.from_numpy(fused)


def _bag_indices(aat, K, S, G):
    """Flat table row indices, one bag per (example (, group)); NONE -> the zero row."""
    if G is None:
        a = aat.reshape(-1, K)
        idx = torch.arange(K, device=aat.device) * S + a.clamp(min=0)
        return idx.masked_fill(a < 0, K * S)
    a = aat.reshape(-1, G, K)
    base = (torch.arange(G, device=aat.device) * K * S).view(1, G, 1) + \
        (torch.arange(K, device=aat.device) * S).view(1, 1, K)
    idx = base + a.clamp(min=0)
    return idx.masked_fill(a < 0, G * K * S).reshape(-1, K)


def bag_read_sums(table, aat, K, S, G=None, out_shape=None):
    """The read as an embedding bag over a prebuilt table (float32 or FBGEMM fused).

    Returns (top, bottom) int32 shaped like read_sums: leading aat batch axes, then
    ``out_shape`` (default ``[L]``; pass e.g. ``(G, L)`` for stacked banks)."""
    if aat.dtype != torch.int64:
        aat = aat.to(torch.int64)
    idx = _bag_indices(aat, K, S, G)
    if table.dtype == torch.uint8:
        offsets = torch.arange(idx.shape[0], dtype=torch.int64) * K
        out = torch.ops.quantized.embedding_bag_byte_rowwise_offsets(
            table, idx.reshape(-1), offsets, False, 0, False, None, None)
    else:
        import torch.nn.functional as F
        out = F.embedding_bag(idx, table, mode="sum")
    L2 = out.shape[-1]
    lead = aat.shape[:-1] if G is None else aat.shape[:-2]
    shape = out_shape if out_shape is not None else \
        ((L2 // 2,) if G is None else (G, L2 // 2))
    out = out.to(torch.int32).reshape(*lead, *(() if G is None else (G,)), L2)
    return out[..., :L2 // 2].reshape(*lead, *shape), \
        out[..., L2 // 2:].reshape(*lead, *shape)


def divide(top, bot, dtype, diff_scale=1.0, mag_scale=1.0):
    """y = top/bottom in ``dtype``, 0 where bottom is 0 (an all-NONE read). For a frozen pack
    the global scales apply per term, exactly as the export's validation computes it."""
    t = top.to(dtype) * diff_scale
    b = bot.to(dtype) * mag_scale
    return torch.where(bot != 0, t / torch.where(bot != 0, b, torch.ones_like(b)),
                       torch.zeros_like(t))


# ---------------------------------------------------------------------------
# feedback: which rule a non-target lane that fired gets. Mirrors the oracle's
# check of the same name (torch/ stays independent of the numpy core).
# ---------------------------------------------------------------------------

FEEDBACK = ("hard", "soft")


def check_feedback(feedback):
    """Validate the feedback rule: "hard" punishes a fired non-target lane with RL, "soft"
    lets it decay with RF."""
    if feedback not in FEEDBACK:
        raise ValueError(f"feedback must be one of {FEEDBACK}, got {feedback!r}")
    return feedback


# ---------------------------------------------------------------------------
# rank_cut: the readout policy, tensor-shaped.
# ---------------------------------------------------------------------------

def rank_cut(y, Vt=0.0, N=None):
    """Keep lanes with y >= Vt, strongest first, as a fixed-width -1-padded output AAT.

    The same information as the oracle's variable-length tuple, tensor-shaped: ``[..., N]``
    int64 (N=None -> width = num lanes). The sort is stable, so ties keep lane order exactly
    as the oracle's stable Python sort does.
    """
    sy, si = torch.sort(y, dim=-1, descending=True, stable=True)
    out = torch.where(sy >= Vt, si, torch.full_like(si, -1))
    return out if N is None else out[..., :N]


# ---------------------------------------------------------------------------
# The read-noise law, factored as the export factors it: sigma_y = T * sigma_unit(m, y).
# ---------------------------------------------------------------------------

class NoiseParams:
    """The Ch3b read-noise coefficients at read_noise = 1, exactly the export's noise block.

    ``sigma_unit(m, y)`` is the per-read noise scale; the temperature knob T (the Core's
    read_noise gain) multiplies it linearly. Defaults are the Core's byte-model defaults at
    room temperature and the FFLV read voltage.
    """

    def __init__(self, a_thermal_unit=0.005, a_flicker_unit=1.0,
                 sqrt_ref_m=math.sqrt(GMAX), flicker_ln_ref=6.0 * math.log(10.0),
                 ref_pw=1e-6, read_pw=1e-6, v_read=0.05):
        self.a_thermal_unit = float(a_thermal_unit)
        self.a_flicker_unit = float(a_flicker_unit)
        self.sqrt_ref_m = float(sqrt_ref_m)
        self.flicker_ln_ref = float(flicker_ln_ref)
        self.ref_pw = float(ref_pw)
        self.read_pw = float(read_pw)
        self.v_read = float(abs(v_read))

    @classmethod
    def from_core(cls, core):
        """Read the coefficients off a live oracle Core (mirrors the generator's export)."""
        return cls(
            a_thermal_unit=core.noise_thermal
            * math.sqrt(core.temperature / core.read_noise_ref_T) * core.read_noise_ref_V,
            a_flicker_unit=core.noise_flicker,
            sqrt_ref_m=math.sqrt(core.read_noise_ref_m),
            flicker_ln_ref=core.flicker_decades * math.log(10.0),
            ref_pw=core.read_noise_ref_pw,
            read_pw=core.read_pulse_width,
            v_read=abs(core.forward_low_voltage),
        )

    def state(self):
        """The export's noise block, key for key."""
        return {"a_thermal_unit": self.a_thermal_unit, "a_flicker_unit": self.a_flicker_unit,
                "sqrt_ref_m": self.sqrt_ref_m, "flicker_ln_ref": self.flicker_ln_ref,
                "ref_pw": self.ref_pw, "read_pw": self.read_pw, "v_fflv": self.v_read}

    def sigma_unit(self, y, m):
        """sigma_y at T=1 for a clean read y with total active magnitude m (real device units).

        Thermal: a_thermal/|V| * sqrt(m_ref/m) * sqrt(pw_ref/pw). Flicker: a_flicker * (1-y^2)
        * sqrt(m_ref/m) * bw_flicker. Summed in quadrature; 0 where m <= 0 (Core.read_sample).
        """
        bw_thermal = math.sqrt(self.ref_pw / self.read_pw)
        ln_band = self.flicker_ln_ref + math.log(self.ref_pw / self.read_pw)
        bw_flicker = math.sqrt(ln_band / self.flicker_ln_ref) if ln_band > 0.0 else 0.0
        f_m = self.sqrt_ref_m / torch.sqrt(m.clamp(min=1e-30))
        s_th = self.a_thermal_unit * f_m / self.v_read * bw_thermal
        s_fl = self.a_flicker_unit * (1.0 - y * y) * f_m * bw_flicker
        sigma = torch.sqrt(s_th * s_th + s_fl * s_fl)
        return torch.where(m > 0, sigma, torch.zeros_like(sigma))


def sample_read(y, m, T, params, generator=None):
    """The opt-in noisy read: y + T * sigma_unit(m, y) * randn, clipped to [-1, 1].

    RNG is the explicit ``generator`` — no module holds global random state. Draw-for-draw
    matching against the oracle's numpy RNG is a non-goal (spec 08 §8); the law is the contract.
    """
    if T <= 0.0:
        return y
    sigma = params.sigma_unit(y, m)
    noise = torch.randn(y.shape, generator=generator, dtype=y.dtype, device=y.device)
    return (y + T * sigma * noise).clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# The update: scatter per-lane deltas onto the active synapses, clamp to the byte range.
# ---------------------------------------------------------------------------

def apply_update(ga, gb, aat, dga, dgb, lane_mask=None, per_group=False):
    """One drive step: every active synapse of a lane gets that lane's (dga, dgb), then the
    byte clamp. Mutates ga/gb in place.

    ga/gb: ``[L, K, S]`` or ``[G, L, K, S]`` integer weights. aat: ``[..., K]`` (shared) or
    ``[..., G, K]`` (per_group). dga/dgb: int32 per-lane deltas, ``[..., L]`` / ``[..., G, L]``.
    lane_mask (optional bool, same shape as dga) zeroes lanes that received no instruction.

    A batch (leading axes) is one adapt step: deltas accumulate, then ONE clamp — the stale-
    batch semantics of spec 08 §6. At B = 1 accumulate-then-clamp is a single update per
    synapse per phase, which is exactly the oracle's serial clamp.
    """
    grouped = ga.dim() == 4
    K, S = ga.shape[-2], ga.shape[-1]
    if lane_mask is not None:
        dga = dga * lane_mask
        dgb = dgb * lane_mask

    if grouped:
        G, L = ga.shape[0], ga.shape[1]
        if per_group:
            a = aat.reshape(-1, G, K)                          # [B, G, K]
        else:
            a = aat.reshape(-1, K)[:, None, :].expand(-1, G, K)
        B = a.shape[0]
        idx = a.clamp(min=0)
        active = (a >= 0)
        g_i = torch.arange(G, device=ga.device).view(1, G, 1, 1).expand(B, G, L, K)
        l_i = torch.arange(L, device=ga.device).view(1, 1, L, 1).expand(B, G, L, K)
        k_i = torch.arange(K, device=ga.device).view(1, 1, 1, K).expand(B, G, L, K)
        s_i = idx[:, :, None, :].expand(B, G, L, K)
        act = active[:, :, None, :]                            # [B, G, 1, K]
        va = (dga.reshape(B, G, L)[:, :, :, None] * act).to(ga.dtype)
        vb = (dgb.reshape(B, G, L)[:, :, :, None] * act).to(gb.dtype)
        index = (g_i.reshape(-1), l_i.reshape(-1), k_i.reshape(-1), s_i.reshape(-1))
    else:
        L = ga.shape[0]
        a = aat.reshape(-1, K)                                 # [B, K]
        B = a.shape[0]
        idx = a.clamp(min=0)
        active = (a >= 0)
        l_i = torch.arange(L, device=ga.device).view(1, L, 1).expand(B, L, K)
        k_i = torch.arange(K, device=ga.device).view(1, 1, K).expand(B, L, K)
        s_i = idx[:, None, :].expand(B, L, K)
        act = active[:, None, :]                               # [B, 1, K]
        va = (dga.reshape(B, L)[:, :, None] * act).to(ga.dtype)
        vb = (dgb.reshape(B, L)[:, :, None] * act).to(gb.dtype)
        index = (l_i.reshape(-1), k_i.reshape(-1), s_i.reshape(-1))

    ga.index_put_(index, va.reshape(-1), accumulate=True)
    gb.index_put_(index, vb.reshape(-1), accumulate=True)
    ga.clamp_(GMIN, GMAX)
    gb.clamp_(GMIN, GMAX)


# ---------------------------------------------------------------------------
# Oracle-matched initialization.
# ---------------------------------------------------------------------------

def init_weights(shape, init, seed):
    """Byte-model initial (Ga, Gb) drawn exactly as the oracle Core draws them.

    ``shape`` is ``[L, K, S]``. The oracle creates devices lane -> space -> a-side (S devices)
    -> b-side (S devices) from ``np.random.default_rng(seed)``; a C-ordered ``[L, K, 2, S]``
    standard-normal array consumes the stream in the identical order, so a seeded torch module
    starts bit-for-bit where a seeded oracle Core starts. A zero-variance init draws nothing
    (base.init_conductance), keeping *_noiseless types RNG-free.
    """
    import numpy as np

    from ..models.base import INIT_TYPES

    mean, rand_var = INIT_TYPES[init]
    L, K, S = shape
    rng = np.random.default_rng(seed)
    noise = rand_var * rng.standard_normal((L, K, 2, S)) if rand_var > 0 else \
        np.zeros((L, K, 2, S))
    g = GMIN + (GMAX - GMIN) * (mean + noise)
    g = np.clip(g, GMIN, GMAX)
    g = np.trunc(g).astype(np.int64)          # Java's (byte) cast truncates toward zero
    ga = torch.from_numpy(np.ascontiguousarray(g[:, :, 0, :]))
    gb = torch.from_numpy(np.ascontiguousarray(g[:, :, 1, :]))
    return ga.to(torch.int32), gb.to(torch.int32)
