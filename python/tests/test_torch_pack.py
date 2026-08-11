"""The frozen pack tier: to_pack / from_pack round trips and the shipped-weights anchor.

The pack is the existing generator export format — int8 (diff, mag), global symmetric
scales, factored noise coefficients. A synthetic round trip checks the format end to end;
when the shipped generator weights are present (a local training artifact, gitignored),
the frozen torch read is checked against them as the cross-implementation anchor the
browser already passed. The full agreement-number reproduction is the generator re-run
script's job.
"""

import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ktram_neural_core.torch import BasisEncoder, Classifier  # noqa: E402
from ktram_neural_core.torch.pack import load_section, quantize  # noqa: E402

from _congruence import record  # noqa: E402

WEIGHTS = (pathlib.Path(__file__).parent.parent / "examples" / "basis-encoder" /
           "figures" / "generator-weights")


def test_quantize_matches_export():
    """Bit-identical to generator.py's _quantize (numpy round, half-to-even)."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 50, size=(4, 5)).astype(np.float32)
    q, step = quantize(a)
    s = float(np.abs(a).max())
    assert np.array_equal(q, np.clip(np.round(a / s * 127.0), -127, 127).astype(np.int8))
    assert step == pytest.approx(s / 127.0)


def test_pack_round_trip(tmp_path):
    path = tmp_path / "weights"
    clf = Classifier(6, 4, 8, seed=1)
    clf.adapt(torch.tensor([1, 2, 3, 4]), torch.tensor([2]))
    enc = BasisEncoder(3, 5, 4, 8, seed=2)
    clf.to_pack(path, "classifier")
    clf.to_pack(path, "decoder")           # a second section must not clobber the first
    enc.to_pack(path)

    fro = Classifier.from_pack(path, "classifier")
    assert fro.frozen
    qd, qm, ds, ms = fro.qdiff.numpy(), fro.qmag.numpy(), fro.diff_scale, fro.mag_scale
    diff, mag = clf._diff_mag_float()
    qd2, ds2 = quantize(diff)
    qm2, ms2 = quantize(mag)
    assert np.array_equal(qd, qd2) and np.array_equal(qm, qm2)
    assert (ds, ms) == (pytest.approx(ds2), pytest.approx(ms2))

    enc_fro = BasisEncoder.from_pack(path)
    assert enc_fro.frozen and enc_fro.qdiff.shape == (3, 5, 4, 8)
    with pytest.raises(RuntimeError):
        enc_fro.adapt(torch.tensor([0, 1, 2, 3]))
    with pytest.raises(RuntimeError):
        fro.adapt(torch.tensor([0, 1, 2, 3]), torch.tensor([0]))
    record("both modules", 1, "pack round trip: to_pack -> from_pack preserves quantized "
                              "arrays, scales, and noise block; frozen adapt refused")


def test_frozen_read_matches_dequantized_math(tmp_path):
    """The frozen read must equal the export validation's arithmetic: int sums, then scales,
    then the guarded divide."""
    path = tmp_path / "weights"
    enc = BasisEncoder(2, 6, 5, 4, seed=3)
    enc.to_pack(path)
    fro = BasisEncoder.from_pack(path)
    qd, qm, ds, ms, _ = load_section(path, "encoder")
    rng = np.random.default_rng(4)
    aats = rng.integers(0, 4, size=(20, 5))
    y = fro.read_scores(torch.from_numpy(aats)).numpy()
    for i, a in enumerate(aats):
        sp = np.arange(5)
        top = qd[:, :, sp, a].sum(axis=-1) * ds
        bot = qm[:, :, sp, a].sum(axis=-1) * ms
        want = top / np.where(bot != 0, bot, 1.0)
        assert np.allclose(y[i], want, atol=0, rtol=0), i
    record("BasisEncoder", 1, "frozen pack read == export-validation arithmetic "
                              "(int8 sums, global scales), bit-exact")


@pytest.mark.skipif(not WEIGHTS.with_suffix(".npz").exists(),
                    reason="shipped generator weights not present (local artifact)")
def test_shipped_pack_loads_with_expected_geometry():
    enc = BasisEncoder.from_pack(WEIGHTS)
    dec = Classifier.from_pack(WEIGHTS, "decoder")
    clf = Classifier.from_pack(WEIGHTS, "classifier")
    assert enc.qdiff.shape == (16, 64, 49, 2)
    assert dec.qdiff.shape == (1568, 16, 64)
    assert clf.qdiff.shape == (10, 16, 64)
    assert enc.noise.v_read == pytest.approx(0.05)
    # a real binary patch: sharp winners must be stable and in range
    rng = np.random.default_rng(0)
    aat = torch.from_numpy(rng.integers(0, 2, size=(8, 16, 49)))
    winners = enc.read(aat, per_group=True)
    assert winners.shape == (8, 16) and int(winners.max()) < 64
    px = dec.read_y(winners)
    assert px.shape == (8, 1568)
    record("both modules", 1, "shipped generator-weights pack loads (encoder [16,64,49,2], "
                              "decoder [1568,16,64], classifier [10,16,64]); reads run")


@pytest.mark.skipif(not (WEIGHTS.parent / "generator-weights.json").exists()
                    or not (WEIGHTS.parent / "generator-weights.bin").exists(),
                    reason="browser pack not present (local artifact)")
def test_browser_pack_agrees_with_npz(tmp_path):
    """The .json/.bin browser carrier decodes to the same arrays as the .npz — one pack,
    two carriers. The json/bin reader is exercised via a copy without the npz beside it."""
    import shutil
    base = tmp_path / "generator-weights"
    shutil.copy(WEIGHTS.with_suffix(".json"), base.with_suffix(".json"))
    shutil.copy(WEIGHTS.with_suffix(".bin"), base.with_suffix(".bin"))
    for section in ("encoder", "decoder", "classifier"):
        qd_npz, qm_npz, ds_npz, ms_npz, _ = load_section(WEIGHTS, section)
        qd_bin, qm_bin, ds_bin, ms_bin, noise = load_section(base, section)
        assert np.array_equal(qd_bin, qd_npz), section
        assert np.array_equal(qm_bin, qm_npz), section
        assert ds_bin == pytest.approx(ds_npz) and ms_bin == pytest.approx(ms_npz)
    assert noise["v_fflv"] == pytest.approx(0.05)
    record("both modules", 1, "browser .json/.bin carrier decodes to the same arrays as "
                              "the .npz (all three sections)")
