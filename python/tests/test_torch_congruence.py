"""Tier-3 congruence: learning-outcome equivalence and the quality-vs-B curve.

B = 1 congruence is already bit-exact (tier 1), so matched runs land inside any band by
identity. The work here is qualifying the batched adapt path: stale reads within a batch
are a first-class operating mode, and the mode is congruent exactly when the outcome
metrics (codebook utilization, winner entropy, label accuracy) hold the bands the oracle's
own run sets. This is the measurement spec 08 §6/§8 calls for — small-scale in the suite;
the full-scale run is the generator re-run script.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ktram_neural_core import BasisGroup, Core  # noqa: E402
from ktram_neural_core.torch import BasisEncoder, Classifier  # noqa: E402

from _congruence import record, to_oracle_aat  # noqa: E402

CHANNELS, SPACES, SYMS = 8, 6, 4
N_TRAIN = 800


def _train_data(seed=0):
    rng = np.random.default_rng(seed)
    # four latent prototypes + symbol noise: a codebook-learnable stream
    protos = rng.integers(0, SYMS, size=(4, SPACES))
    idx = rng.integers(0, 4, size=N_TRAIN)
    data = protos[idx]
    flip = rng.random((N_TRAIN, SPACES)) < 0.15
    noise = rng.integers(0, SYMS, size=(N_TRAIN, SPACES))
    return np.where(flip, noise, data), idx


def _torch_encoder(seed=31):
    return BasisEncoder(1, CHANNELS, SPACES, SYMS, init="low", seed=seed, gather_abandon=8)


def test_encoder_outcomes_by_batch_size():
    """The quality-vs-B curve: utilization and entropy at B in {1, 8, 64} against the
    oracle's serial run. B = 1 is exact; larger B must hold the outcome bands."""
    data, _ = _train_data()
    core = Core(1, SYMS, spaces_per_lane=SPACES, num_lanes=CHANNELS, model="byte",
                init="low", read_noise=0, seed=31)
    grp = BasisGroup(core, CHANNELS, gather_abandon=8)
    for a in data:
        grp.adapt(to_oracle_aat(a))
    util_o = grp.codebook_utilization
    ent_o = grp.winner_entropy

    results = {}
    for B in (1, 8, 64):
        enc = _torch_encoder()
        batches = torch.from_numpy(data).split(B)
        for batch in batches:
            enc.adapt(batch)
        results[B] = (float(enc.codebook_utilization[0]), float(enc.winner_entropy[0]))

    assert results[1] == (pytest.approx(util_o), pytest.approx(ent_o))
    for B in (8, 64):
        util, ent = results[B]
        assert util >= util_o - 0.25, (B, results)
        assert ent >= ent_o - 1.0, (B, results)
    lines = ", ".join(f"B={B}: util {u:.2f} entropy {e:.2f}" for B, (u, e) in results.items())
    record("BasisEncoder", 3, f"quality-vs-B (oracle util {util_o:.2f}, entropy {ent_o:.2f}) "
                              f"-> {lines}")


def test_classifier_accuracy_by_batch_size():
    """Supervised outcome bands: label accuracy at B in {1, 16} within tolerance of the
    oracle's serial run on the same stream."""
    data, labels = _train_data(seed=5)
    n_labels = 4

    def acc_of(clf):
        preds = clf.read_y(torch.from_numpy(data)).argmax(dim=-1).numpy()
        return float((preds == labels).mean())

    accs = {}
    for B in (1, 16):
        clf = Classifier(n_labels, SPACES, SYMS, init="medium", seed=17)
        for i in range(0, N_TRAIN, B):
            batch = torch.from_numpy(data[i:i + B])
            tgt = torch.from_numpy(labels[i:i + B, None])
            clf.adapt(batch, tgt)
        accs[B] = acc_of(clf)

    assert accs[1] > 0.8, accs                    # learns the separable stream
    assert accs[16] >= accs[1] - 0.1, accs        # the stale batch holds the band
    record("Classifier", 3, f"label accuracy on 4-proto stream: B=1 {accs[1]:.3f}, "
                            f"B=16 {accs[16]:.3f} (band: within 0.1)")


def test_devices_move_and_read_identically():
    """The device matrix: CPU is the anchor; MPS/CUDA (where available) must agree on the
    sharp integer read. (MPS divides in float32, so y agrees to float32; the argmax-grade
    integer sums must match exactly.)"""
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    enc = _torch_encoder(seed=51)
    rng = np.random.default_rng(3)
    aats = torch.from_numpy(rng.integers(0, SYMS, size=(16, SPACES)))
    y_cpu = enc.read_scores(aats)
    for dev in devices[1:]:
        enc_d = _torch_encoder(seed=51).to(dev)
        y_d = enc_d.read_scores(aats.to(dev)).cpu()
        assert torch.allclose(y_cpu.to(torch.float32), y_d.to(torch.float32),
                              atol=1e-6), dev
    record("both modules", 3, f"device matrix: sharp reads agree across {devices}")


def test_sampled_read_deterministic_with_seeded_generator():
    enc = _torch_encoder(seed=61)
    aat = torch.tensor([0, 1, 2, 3, 0, 1])
    a = enc.read_sampled(aat, 0.6, torch.Generator().manual_seed(9))
    b = enc.read_sampled(aat, 0.6, torch.Generator().manual_seed(9))
    assert torch.equal(a, b)
    record("both modules", 3, "seeded read_sampled is reproducible (explicit torch.Generator, "
                              "no module-held RNG state)")
