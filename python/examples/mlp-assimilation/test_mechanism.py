"""Guard the scientific protocol, rather than asserting a seed's accuracy."""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import experiment as exp
from models import PatchEncoder, Teacher, freeze, patchify, straight_through


def test_hard_forward_preserves_a_gradient():
    logits = torch.tensor([[0.1, 0.4, -0.2]], requires_grad=True)
    lift, _ = straight_through(logits, 0.7)
    assert torch.equal(lift, torch.tensor([[0.0, 1.0, 0.0]]))
    (lift * torch.tensor([[1.0, 3.0, -1.0]])).sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_local_encoder_cannot_mix_patches():
    torch.manual_seed(1)
    encoder = PatchEncoder(28)
    images = torch.zeros(2, 1, 28, 28)
    images[1, 0, :7, :7] = 1
    logits = encoder(patchify(images))
    assert torch.equal(logits[0, 1:], logits[1, 1:])
    assert not torch.equal(logits[0, 0], logits[1, 0])


def test_frozen_teacher_does_not_receive_student_gradients():
    teacher = freeze(Teacher(8))
    before = exp.snapshot(teacher)
    x = torch.randn(3, 1, 8, 8)
    target = teacher(x)[0]
    student = torch.nn.Linear(64, 64)
    torch.nn.functional.mse_loss(student(x.flatten(1)), target).backward()
    assert student.weight.grad is not None
    assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(v, teacher.state_dict()[k]) for k, v in before.items())


def test_splits_are_disjoint_and_reproducible():
    splits = exp.split_indices(70000, 32000, 2000, 2000, 0)
    assert len(np.unique(np.concatenate(splits))) == 36000
    assert all(np.array_equal(a, b) for a, b in zip(splits, exp.split_indices(70000, 32000, 2000, 2000, 0)))


def test_teaching_harvests_actual_outputs_and_keeps_clean_targets(monkeypatch):
    # Deliberately wrong upstream emissions must reach the following stage,
    # while its targets remain the original image's clean target symbols.
    observed = []
    banks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
    monkeypatch.setattr(exp, "new_banks", lambda seed: banks)

    def record(bank, inputs, targets, val_inputs, val_targets, ko, so, cfg, epochs, tag):
        observed.append((inputs.clone(), targets.clone(), val_inputs.clone(), val_targets.clone()))
        return {}

    monkeypatch.setattr(exp, "teach_stage", record)
    monkeypatch.setattr(exp, "lane_codes", lambda bank, codes, ko: torch.full((len(codes), ko), 3, dtype=torch.long))
    train = tuple(torch.zeros(5, k, dtype=torch.long) for k in (16, 32, 16))
    val = tuple(torch.zeros(2, k, dtype=torch.long) for k in (16, 32, 16))
    exp.teach_chain(PatchEncoder(8), train, val, torch.ones(5, dtype=torch.long),
                    torch.ones(2, dtype=torch.long), {"seed": 0, "teach_epochs": [1, 1, 1]}, "test")
    assert torch.all(observed[1][0] == 3) and torch.all(observed[1][2] == 3)
    assert torch.all(observed[1][1] == 0) and torch.all(observed[1][3] == 0)
    assert torch.all(observed[2][0] == 3) and torch.all(observed[2][1] == 1)
