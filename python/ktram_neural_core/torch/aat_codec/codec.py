"""The Codec interface, and the reference AAT codec.

An AAT is a product-quantization code: a D-dim vector becomes k = D/2 slots, each slot one
symbol in [0, S), S = 2^bits. A codec turns vectors into these codes and back, and exposes the
codewords so the read kernels (kernels.py) can run the network's operations on the codes.

`Codec` is the small interface a codec implements. `AATCodec` is the reference
implementation — the design the A–N codec search converged on:

  * d=2 slots.
  * Position-clustered routing (Codec G): the k slot positions cluster into K archetype groups
    by the shape of their atom distribution; each group gets its own separable book. Routing
    keys on the slot INDEX — known at encode and decode, zero wire bits.
  * Separable Hadamard books with companding (Codec N): each 2-d slot rotates into a fixed
    Hadamard basis; each axis is quantized by a fitted Panter-Dite p^(1/3) compander plus a
    uniform round. This matches 1-d Lloyd (k-means) fidelity with NO search — encode is a
    per-axis bucketize.
  * Pure-gather decode: each symbol maps to a stored reconstruction value. Decode is a lookup,
    linear in the one-hot lift, so a neural lane fed this code represents its target map exactly.

Two roles, one class (`fuse_rmsnorm`):
  * fuse_rmsnorm=False — quantize a vector already in its natural space (Q/K/V/attn-out books).
  * fuse_rmsnorm=True  — the lane-input encoder: fit books on x̂ = RMSNorm(h), then `encode(h)`
    takes the RAW pre-norm residual h and folds the RMSNorm normalize + γ scale into the encode.

Module idiom (spec 08 §5, same as the L1 modules): a codec is a ``torch.nn.Module`` whose
fitted state — the decode book ``C``, the per-symbol norm table ``norm2``, the encode edge
tables, the routing — is registered buffers, never Parameters, no autograd anywhere. Fit once,
``.to(device)`` moves every table, ``state_dict()`` serializes them, and no kernel call ever
rebuilds a table. Encode is one batched ``searchsorted`` over the per-slot edge tables; the only
Python loop left is over the K ≤ 8 archetype groups at fit time.
"""
import math
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

SLOT_DIM = 2                 # d=2
RMS_EPS = 1e-6
_HADAMARD = (1.0 / math.sqrt(2.0)) * torch.tensor([[1.0, 1.0], [1.0, -1.0]])   # symmetric: R@R=I


# ===================================================================== the interface
class Codec(nn.Module, ABC):
    """Interface a codec implements to run against the read kernels.

    Required attributes after `fit`:
      * k  — number of slots (D // 2)
      * S  — alphabet size (2^bits)
      * C  — codewords, [k, S, 2]: C[m, s] is the 2-d reconstruction of symbol s in slot m,
             registered as a buffer. The read kernels build their lookup tables from C. A codec
             whose representation is not a per-slot codeword table overrides the kernels instead.
    """

    k: int
    S: int

    @abstractmethod
    def fit(self, X, **kw) -> "Codec":
        """Fit the codebook on a train split (unsupervised). Returns self."""

    @abstractmethod
    def encode(self, X) -> torch.Tensor:
        """[...,D] -> codes [...,k] (long)."""

    @abstractmethod
    def decode(self, codes) -> torch.Tensor:
        """codes [...,k] -> [...,D]."""

    @abstractmethod
    def onehot_lift(self, codes, flatten=False) -> torch.Tensor:
        """codes [...,k] -> one-hot lift [...,k,S] (the neural-lane input)."""

    @abstractmethod
    def op_counts(self, op: str) -> dict:
        """Elementary-op counts for `op` (encode/decode/dot/norm), for cost accounting."""

    def self_norms2(self, codes) -> torch.Tensor:
        """Σ_slot ‖codeword‖² -> [...]. Default reads C; equals ‖decode‖²."""
        idx0 = torch.arange(self.k, device=codes.device).expand_as(codes)
        return (self.C ** 2).sum(-1)[idx0, codes].sum(-1)


# ===================================================================== small helpers
def atoms(X, k):
    """[...,D] -> [...,k,2] slot view."""
    return X.reshape(*X.shape[:-1], k, SLOT_DIM)


def _kmeans(X, S, iters=25, seed=0):
    """Vectorized Lloyd k-means for the slot routing. X [N,d] -> centroids [S,d]. Empty
    clusters re-seeded; deterministic under the explicit seed (no global RNG touched)."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    N, d = X.shape
    if N <= S:                                    # degenerate (fewer points than clusters)
        pad = X[torch.randint(N, (S - N,), generator=g, device=X.device)] if N < S else X
        return (X if N == S else torch.cat([X, pad]))[:S].clone()
    C = X[torch.randperm(N, generator=g, device=X.device)[:S]].clone()
    for _ in range(iters):
        a = torch.cdist(X, C).argmin(1)           # [N] nearest-centroid assignment
        sums = torch.zeros(S, d, device=X.device).index_add_(0, a, X)
        counts = torch.bincount(a, minlength=S).clamp(min=1).unsqueeze(1).float()
        newC = sums / counts
        empty = torch.bincount(a, minlength=S) == 0
        if empty.any():
            newC[empty] = X[torch.randint(N, (int(empty.sum()),), generator=g, device=X.device)]
        if torch.allclose(newC, C, atol=1e-6):
            C = newC
            break
        C = newC
    return C


def _subsample(A, cap, seed=0):
    if cap is None or A.shape[0] <= cap:
        return A
    g = torch.Generator().manual_seed(seed)
    return A[torch.randperm(A.shape[0], generator=g)[:cap]]


def _slot_features(A):
    """A [N,k,2] -> standardized per-slot feature [k,3] (log energy, log offset, log tail)."""
    e = A.pow(2).sum(-1)
    log_e = e.clamp(min=1e-9).log().mean(0)
    off = A.mean(0).norm(dim=-1).clamp(min=1e-9).log()
    p99 = torch.quantile(e, 0.99, dim=0).clamp(min=1e-9).log()
    f = torch.stack([log_e, off, p99], dim=-1)
    return (f - f.mean(0)) / f.std(0).clamp(min=1e-9)


def _build_compander(x, nbins=2048):
    """Panter-Dite warp: level density ∝ p(x)^(1/3). Returns (edges[nbins+1], cum[nbins+1])."""
    lo, hi = x.min().item(), x.max().item()
    if hi <= lo:
        hi = lo + 1e-6
    counts = torch.histc(x, bins=nbins, min=lo, max=hi)
    w = (counts + 1e-9) ** (1.0 / 3.0)
    cum = torch.cat([torch.zeros(1), torch.cumsum(w, 0)])
    cum = cum / cum[-1].clamp(min=1e-12)
    return torch.linspace(lo, hi, nbins + 1), cum


def _compander_interior(edges, cum, S):
    """S-1 interior level boundaries at equal warped-cumulative quantiles."""
    q = np.arange(1, S, dtype=np.float64) / S
    x = np.interp(q, cum.numpy().astype(np.float64), edges.numpy().astype(np.float64))
    return torch.from_numpy(x).float()


def _bin_means(xfit, interior):
    """Per-bin conditional mean of xfit — the decode reconstruction values. S = |interior|+1.
    An empty bin falls back to its nearest interior boundary."""
    S = interior.numel() + 1
    idx = torch.bucketize(xfit.contiguous(), interior)
    sums = torch.zeros(S).index_add_(0, idx, xfit)
    counts = torch.bincount(idx, minlength=S)
    vals = sums / counts.clamp(min=1).to(sums.dtype)
    if S > 1:
        fallback = interior[torch.arange(S).clamp(max=S - 2)]
        vals = torch.where(counts > 0, vals, fallback)
    return vals


def _bit_split(U, total_bits):
    """Split the per-slot bit budget across the two rotated axes by variance."""
    vu, vv = U[:, 0].var().item(), U[:, 1].var().item()
    bu = round(total_bits / 2 + 0.5 * math.log2(max(vu, 1e-9) / max(vv, 1e-9)))
    bu = min(max(bu, 1), total_bits - 1)
    return 2 ** bu, 2 ** (total_bits - bu)


# ===================================================================== the reference codec
class AATCodec(Codec):
    """The reference codec: companding separable books, position-clustered.

    Fit one per space (a Q-book, a K-book, a V-book, an x̂ lane-input book). `encode` is
    search-free, `decode` is a pure gather. See the module docstring for the design.

    Fitted state, all registered buffers: ``C`` [k,S,2] (decode book, original basis),
    ``norm2`` [k,S] (per-symbol ‖codeword‖²), ``assign`` [k] (slot -> group), ``su``/``sv``
    [k] (per-slot axis alphabet split, Su·Sv = S), ``edges_u``/``edges_v`` [k, ·] (per-slot
    interior encode boundaries, +inf padded to one width so a single batched ``searchsorted``
    encodes every slot at once), and ``gamma`` [D] when RMSNorm is fused.
    """

    def __init__(self, bits, K=8, fuse_rmsnorm=False):
        super().__init__()
        self.bits = int(bits)
        self.S = 2 ** self.bits
        self.K = int(K)
        self.fuse_rmsnorm = bool(fuse_rmsnorm)
        self.k = None                 # slots (set by fit)
        self.D = None                 # vector dim (set by fit)
        self.register_buffer("R", _HADAMARD.clone())   # 2x2 rotation

    def _set_buffer(self, name, tensor):
        """Register (or on a re-fit, replace) a fitted buffer."""
        if name in self._buffers:
            del self._buffers[name]
        self.register_buffer(name, tensor)

    # ---- fit -----------------------------------------------------------------
    def fit(self, X, gamma=None, max_atoms=200_000):
        """Fit books on X, the space the codebook represents. If fuse_rmsnorm, X is x̂ = RMSNorm(h)
        and `gamma` (γ, [D]) must be given (used at encode time to normalize raw h)."""
        self.D = X.shape[-1]
        self.k = self.D // SLOT_DIM
        if self.fuse_rmsnorm:
            if gamma is None:
                raise ValueError("fuse_rmsnorm=True needs gamma (γ, [D]) to normalize raw h at encode")
            self._set_buffer("gamma", gamma.reshape(-1).float())
        A = atoms(X, self.k).reshape(-1, self.k, SLOT_DIM)          # [N,k,2]
        assign = self._route(A)
        G = int(assign.max().item()) + 1
        books = torch.empty(G, self.S, SLOT_DIM)
        group_splits, group_edges = [], []
        for g in range(G):
            slots = (assign == g).nonzero().flatten()
            pooled = _subsample(A[:, slots].reshape(-1, SLOT_DIM), max_atoms)
            U = pooled @ self.R.T                                  # rotate into separable basis
            Su, Sv = _bit_split(U, self.bits)
            edges, vals = [], []
            for ax, S in ((0, Su), (1, Sv)):
                xfit = U[:, ax]
                interior = _compander_interior(*_build_compander(xfit), S).sort().values
                edges.append(interior)
                vals.append(_bin_means(xfit, interior))
            gu, gv = torch.meshgrid(vals[0], vals[1], indexing="ij")   # [Su,Sv]
            grid = torch.stack([gu, gv], dim=-1).reshape(-1, SLOT_DIM)  # [S,2] rotated
            books[g] = grid @ self.R                                # un-rotate to original basis
            group_splits.append((Su, Sv))
            group_edges.append((edges[0], edges[1]))

        # Broadcast the per-group tables out to per-slot buffers, +inf padded to one width,
        # so encode is a single batched searchsorted (padding never matches a finite value).
        su = torch.tensor([group_splits[g][0] for g in assign.tolist()])
        sv = torch.tensor([group_splits[g][1] for g in assign.tolist()])
        wu = max(int(s) - 1 for s, _ in group_splits)
        wv = max(int(s) - 1 for _, s in group_splits)
        edges_u = torch.full((self.k, wu), torch.inf)
        edges_v = torch.full((self.k, wv), torch.inf)
        for m, g in enumerate(assign.tolist()):
            eu, ev = group_edges[g]
            edges_u[m, :eu.numel()] = eu
            edges_v[m, :ev.numel()] = ev
        self._set_buffer("assign", assign)
        self._set_buffer("su", su)
        self._set_buffer("sv", sv)
        self._set_buffer("edges_u", edges_u)
        self._set_buffer("edges_v", edges_v)
        self._set_buffer("C", books[assign])                       # [k,S,2] per-slot decode book
        self._set_buffer("norm2", (self.C ** 2).sum(-1))           # [k,S] per-symbol ‖codeword‖²
        return self

    def _route(self, A):
        k = A.shape[1]
        Keff = min(self.K, k)
        if Keff <= 1:
            return torch.zeros(k, dtype=torch.long)
        if Keff >= k:
            return torch.arange(k)
        f = _slot_features(A)
        Cf = _kmeans(f, Keff, seed=0)
        return torch.cdist(f, Cf).argmin(1)

    # ---- encode / decode -----------------------------------------------------
    def encode(self, X):
        """[...,D] -> codes [...,k]. If fuse_rmsnorm, X is RAW pre-norm h (normalize+γ folded in).
        Search-free: rotate each slot, bucketize each axis against the slot's warp edges."""
        if self.fuse_rmsnorm:
            u = X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + RMS_EPS)
            X = self.gamma * u
        return self._quantize(X)

    def encode_normalized(self, Xhat):
        """Encode an already-normalized x̂ (skip the RMSNorm front end even when fused)."""
        return self._quantize(Xhat)

    def _quantize(self, X):
        return self.encode_atoms(atoms(X, self.k))

    def encode_atoms(self, A):
        """[...,k,2] slot atoms -> codes [...,k]. One rotation and one batched per-slot
        searchsorted per axis — no per-slot or per-group Python loop. Search-free."""
        U = A @ self.R.T
        lead = U.shape[:-2]
        u = U[..., 0].reshape(-1, self.k).T.contiguous()           # [k, B]
        v = U[..., 1].reshape(-1, self.k).T.contiguous()
        au = torch.searchsorted(self.edges_u, u)                   # [k, B] in [0, Su)
        av = torch.searchsorted(self.edges_v, v)
        codes = au * self.sv.unsqueeze(1) + av
        return codes.T.reshape(*lead, self.k)

    def decode(self, codes):
        """codes [...,k] -> [...,D]. Pure gather (one codeword per slot), no arithmetic."""
        return self.decode_atoms(codes).reshape(*codes.shape[:-1], self.k * SLOT_DIM)

    def decode_atoms(self, codes):
        """codes [...,k] -> [...,k,2] (decoded slot atoms, before flattening)."""
        idx0 = torch.arange(self.k, device=codes.device).expand_as(codes)
        return self.C[idx0, codes]

    # ---- the lane interface --------------------------------------------------
    def onehot_lift(self, codes, flatten=False):
        """codes [...,k] -> one-hot lift [...,k,S] (or [...,k*S] if flatten). decode(codes) ==
        einsum(lift, C), i.e. decode is linear in this binary vector — the lane's input."""
        lift = torch.zeros(*codes.shape, self.S, device=codes.device)
        lift.scatter_(-1, codes.unsqueeze(-1), 1.0)
        return lift.reshape(*codes.shape[:-1], self.k * self.S) if flatten else lift

    def self_norms2(self, codes):
        """Σ_slot ‖codeword‖² -> [...]. Reads the per-symbol table (k lookups, no reconstruction)."""
        idx0 = torch.arange(self.k, device=codes.device).expand_as(codes)
        return self.norm2[idx0, codes].sum(-1)

    def lift_width(self):
        """k·S — the one-hot lift dimension = the lane's input hardware size."""
        return self.k * self.S

    # ---- accounting ----------------------------------------------------------
    def _group_splits(self):
        """One (Su, Sv) per archetype group, read back off the per-slot buffers."""
        first = [(self.assign == g).nonzero()[0].item()
                 for g in range(int(self.assign.max().item()) + 1)]
        return [(int(self.su[m].item()), int(self.sv[m].item())) for m in first]

    def wire_bytes(self):
        return self.k * math.ceil(math.log2(self.S)) / 8.0

    def lut_bytes(self):
        splits = self._group_splits()
        edges = sum((Su - 1) + (Sv - 1) for (Su, Sv) in splits) * 2
        vals = sum(Su + Sv for (Su, Sv) in splits) * 2
        route = self.k * math.ceil(math.log2(max(self.K, 2))) / 8.0
        return {"decode_values_bytes": vals, "encode_edges_bytes": edges,
                "route_bytes": route, "total": edges + vals + route}

    def op_counts(self, op):
        """Elementary-op counts. encode splits into an unavoidable RMSNorm front end (~2.0× a
        native D-dim dot, only when fused) and the flat, budget-independent quantization (~1.5×)."""
        k, D = self.k, self.k * SLOT_DIM
        if op == "encode":
            rot_bucket = {"lookups": 2 * k, "mults": 2 * k, "adds": 2 * k}   # rotate + bucketize
            if not self.fuse_rmsnorm:
                return rot_bucket
            return {"lookups": 2 * k, "mults": 3 * D + 2 * k, "adds": D + 2 * k, "sqrt": 1}
        if op == "decode":
            return {"lookups": k, "mults": 0, "adds": 0}
        if op == "norm":
            return {"lookups": k, "mults": 0, "adds": k - 1, "sqrt": 1}
        if op == "dot":
            return {"lookups": k, "mults": 0, "adds": k - 1}
        raise KeyError(op)

    def extra_repr(self):
        G = len(self._group_splits()) if "assign" in self._buffers else 0
        return (f"bits={self.bits}, K={self.K}, fuse_rmsnorm={self.fuse_rmsnorm}, "
                f"k={self.k}, S={self.S}, groups={G}")
