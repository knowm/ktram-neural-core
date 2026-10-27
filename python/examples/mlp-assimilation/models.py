"""Models for Chapter 8. Only the digital encoder survives proxy training.

LiftStage is an additive score surrogate. It does not simulate the Classifier's
conductance normalization or local update rule; transfer must be measured.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ktram_neural_core.torch import Classifier


def freeze(model):
    model.eval()
    model.requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    return model


def patchify(images):
    """Four by four nonoverlapping patches, for either 28x28 or 8x8 images."""
    n, side = len(images), images.shape[-1]
    if side % 4 or images.shape[1:] != (1, side, side):
        raise ValueError("expected square single-channel images divisible by four")
    p = side // 4
    return images.reshape(n, 4, p, 4, p).permute(0, 1, 3, 2, 4).reshape(n, 16, p * p)


def straight_through(logits, temperature):
    soft = (logits / temperature).softmax(-1)
    hard = F.one_hot(soft.argmax(-1), logits.shape[-1]).to(soft.dtype)
    return hard + (soft - soft.detach()), soft


def usage_entropy(soft):
    p = soft.mean(0)
    return -(p * p.clamp_min(1e-9).log()).sum(-1).mean()


class Teacher(nn.Module):
    def __init__(self, side):
        super().__init__()
        self.fc1 = nn.Linear(side * side, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, images):
        z1 = self.fc1(images.flatten(1))
        z2 = self.fc2(F.relu(z1))
        return z1, z2, self.fc3(F.relu(z2))


class PatchEncoder(nn.Module):
    """Sixteen independent local networks: patch pixels -> 32 -> 64 symbols."""
    def __init__(self, side):
        super().__init__()
        pixels = (side // 4) ** 2
        self.w1 = nn.Parameter(torch.randn(16, pixels, 32) / pixels ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(16, 32))
        self.w2 = nn.Parameter(torch.randn(16, 32, 64) / 32 ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(16, 64))

    def forward(self, patches):
        hidden = F.relu(torch.einsum("bkp,kph->bkh", patches, self.w1) + self.b1)
        return torch.einsum("bkh,khs->bks", hidden, self.w2) + self.b2


class LiftStage(nn.Module):
    """One score table per output candidate, summed over active input symbols."""
    def __init__(self, k_in, s_in, k_out, s_out):
        super().__init__()
        self.table = nn.Parameter(torch.randn(k_in, s_in, k_out, s_out) * 0.01)

    def forward(self, lift):
        return torch.einsum("bks,ksmt->bmt", lift, self.table)


class Proxy(nn.Module):
    """A4 topology: 16x64 -> 32x16 -> 16x16 -> ten class scores."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.stage1 = LiftStage(16, 64, 32, 16)
        self.stage2 = LiftStage(32, 16, 16, 16)
        self.head = LiftStage(16, 16, 1, 10)
        self.probe1 = nn.Linear(32 * 16, 64)
        self.probe2 = nn.Linear(16 * 16, 32)

    def forward(self, patches, temperature=1.0):
        l0, p0 = straight_through(self.encoder(patches), temperature)
        l1, p1 = straight_through(self.stage1(l0), temperature)
        l2, p2 = straight_through(self.stage2(l1), temperature)
        return self.head(l2).squeeze(1), (l1, l2), (p0, p1, p2)

    @torch.no_grad()
    def codes(self, patches):
        c0 = self.encoder(patches).argmax(-1)
        c1 = self.stage1(F.one_hot(c0, 64).float()).argmax(-1)
        c2 = self.stage2(F.one_hot(c1, 16).float()).argmax(-1)
        pred = self.head(F.one_hot(c2, 16).float()).squeeze(1).argmax(-1)
        return c0, c1, c2, pred


STAGES = ((16, 64, 32, 16), (32, 16, 16, 16), (16, 16, 1, 10))


def new_banks(seed):
    return nn.ModuleList([
        Classifier(ko * so, ki, si, init="medium", seed=seed)
        for ki, si, ko, so in STAGES
    ])


@torch.no_grad()
def lane_codes(bank, codes, k_out, batch=1024):
    """Unconditional winner in each output slot, including all-negative scores.

read_y supplies scores; this explicit slot-wise winner is the experiment's
recoder. A single global Classifier.read() is not equivalent.
"""
    return torch.cat([
        bank.read_y(b).reshape(len(b), k_out, -1).argmax(-1)
        for b in codes.split(batch)
    ])


class Student(nn.Module):
    """Inference uses the digital patch encoder and three taught lane banks."""
    def __init__(self, encoder, banks):
        super().__init__()
        self.encoder, self.banks = encoder, banks

    @torch.no_grad()
    def forward(self, images):
        codes = self.encoder(patchify(images)).argmax(-1)
        for bank, (_, _, k_out, _) in zip(self.banks, STAGES):
            codes = lane_codes(bank, codes, k_out)
        return codes.squeeze(-1)
