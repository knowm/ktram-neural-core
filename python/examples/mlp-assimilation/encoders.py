"""Input encoders for the Chapter 8 encoder study.

Every encoder maps patches [B, P, D] (P patches, D pixels each, floats in [0, 1]) to
logits [B, P, S]; the symbol per patch is the argmax. `cost(side)` returns the digital
parameter, byte, MAC and comparison counts and the synapse count where the encoder is a
lane stage, so every entry lands in the same accounting table.

`a2d`: each pixel is quantized to L levels (784 x ceil(log2 L) comparisons, no MACs),
the patch becomes a D-slot AAT with L symbols per slot, and one lift table per patch
position scores S symbols -- exactly a lane group per patch (D slots x L channels -> S
lanes), so it can be taught by `Classifier.adapt` on a grouped bank.

Digital references: per-position linear (`linear`) and the Chapter 8 patch MLP (`mlp`).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def pixel_edges(train_patches, L, levels):
    """Shared-across-positions level edges, L-1 of them, from the training pixels only.

    uniform: k/L for k = 1..L-1. quantile: level 0 is exactly zero (Fashion-MNIST
    background), the other L-1 levels are quantiles of the NONZERO training pixels, so
    degenerate quantiles at zero cannot collapse the alphabet.
    """
    if L < 2:
        raise ValueError("L must be at least 2")
    if levels == "uniform":
        return torch.arange(1, L, dtype=torch.float32) / L
    if levels == "quantile":
        nz = train_patches[train_patches > 0].flatten().detach().cpu()
        if L == 2:
            return torch.tensor([1e-6])
        qs = torch.arange(1, L - 1, dtype=torch.float32) / (L - 1)
        inner = torch.quantile(nz[torch.randperm(len(nz))[:2_000_000]], qs)
        return torch.cat([torch.tensor([1e-6]), inner])
    raise ValueError(levels)


def quantize(patches, edges):
    """Pixel level index in 0..L-1 via the shared edges."""
    return torch.bucketize(patches.contiguous(), edges.to(patches.device))


class A2DLiftEncoder(nn.Module):
    """Lane encoder: quantized pixels -> one lift table per patch position -> S scores."""

    def __init__(self, n_patches, pixels, L, S, edges):
        super().__init__()
        self.L, self.S = L, S
        self.register_buffer("edges", edges.clone())
        self.table = nn.Parameter(torch.randn(n_patches, pixels, L, S) * 0.01)

    def levels(self, patches):
        return quantize(patches, self.edges)

    def forward(self, patches):
        lift = F.one_hot(self.levels(patches), self.L).to(patches.dtype)   # [B, P, D, L]
        return torch.einsum("bpdl,pdls->bps", lift, self.table)

    def cost(self, side):
        P, D, L, S = self.table.shape
        return dict(kind="lane", digital_params=L - 1, digital_bytes=4 * (L - 1),
                    digital_macs=0, comparisons=side * side * math.ceil(math.log2(L)),
                    encoder_synapses=int(P * D * L * S))


class LinearEncoder(nn.Module):
    """Per-position linear scorer D -> S."""

    def __init__(self, n_patches, pixels, S):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_patches, pixels, S) / pixels ** 0.5)
        self.b = nn.Parameter(torch.zeros(n_patches, S))

    def forward(self, patches):
        return torch.einsum("bpd,pds->bps", patches, self.w) + self.b

    def cost(self, side):
        P, D, S = self.w.shape
        n = P * D * S + P * S
        return dict(kind="digital", digital_params=n, digital_bytes=4 * n,
                    digital_macs=P * D * S, comparisons=0, encoder_synapses=0)


class MLPEncoder(nn.Module):
    """The Chapter 8 patch MLP, D -> H -> S per position."""

    def __init__(self, n_patches, pixels, S, hidden=32):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n_patches, pixels, hidden) / pixels ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(n_patches, hidden))
        self.w2 = nn.Parameter(torch.randn(n_patches, hidden, S) / hidden ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(n_patches, S))

    def forward(self, patches):
        h = F.relu(torch.einsum("bpd,pdh->bph", patches, self.w1) + self.b1)
        return torch.einsum("bph,phs->bps", h, self.w2) + self.b2

    def cost(self, side):
        P, D, H = self.w1.shape
        S = self.w2.shape[-1]
        macs = P * (D * H + H * S)
        n = macs + P * (H + S)
        return dict(kind="digital", digital_params=n, digital_bytes=4 * n,
                    digital_macs=macs, comparisons=0, encoder_synapses=0)


def make_encoder(name, n_patches, pixels, S, train_patches=None, L=4, levels="quantile", hidden=32):
    if name == "a2d":
        return A2DLiftEncoder(n_patches, pixels, L, S, pixel_edges(train_patches, L, levels))
    if name == "linear":
        return LinearEncoder(n_patches, pixels, S)
    if name == "mlp":
        return MLPEncoder(n_patches, pixels, S, hidden)
    raise ValueError(name)
