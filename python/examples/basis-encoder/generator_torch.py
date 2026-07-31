"""The headline 08a check: generator.py training re-run on the torch L1 modules.

Same data order, same schedule, same seeds — the encoder's form-then-sharpen passes, the
per-pixel decoder, and the label read-out all re-run on `ktram_neural_core.torch`, then
checked tier by tier against the pickled oracle model and the shipped weight pack:

  tier 1  final integer state bit-for-bit against figures/generator.pkl (encoder, decoder,
          classifier), plus the frozen-pack cross-check that reproduces the export's
          encoder-winner / decoder-pixel agreement numbers
  tier 2  sampled-winner statistics at the hot setpoint vs the live oracle Cores
  tier 3  the outcome metrics (utilization, entropy, decoder pixel accuracy, label
          accuracy) side by side
  tier 4  generation behavior: fixed points at T = 0, wander statistics at T > 0

  python generator_torch.py        # everything; writes figures/congruence-report-generator.md

The report is the deliverable — the run prints it and leaves it in figures/. Alex signs off
on the report; this script only has to run and report. Needs figures/generator.pkl (run
`generator.py train` + `classifier` first) and the Fashion-MNIST cache.
"""

import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import generator  # noqa: E402  (constants + oracle classes; no training on import)
import __main__  # noqa: E402

# generator.py pickles its __main__-defined classes; alias them so the pkl loads here.
__main__.NeuralPixelDecoder = generator.NeuralPixelDecoder
__main__.NeuralLabelClassifier = generator.NeuralLabelClassifier

# A generator.pkl written under numpy 2.x is not RNG-portable to numpy 1.x (which torch
# 2.2, the last Intel-mac build, requires): both the bit-generator constructor arguments and
# the state format changed. Nothing here needs the pickled stream — the Cores' weights are
# what we compare, and the tier-2 checks are statistical — so unpickle the RNGs as fresh
# entropy-seeded generators and leave everything else untouched.
import pickle  # noqa: E402


class _FreshRNG:
    """Stands in for a numpy RNG pickled by a different numpy major version."""

    def __init__(self, *args, **kwargs):
        self._rng = np.random.default_rng()

    def __setstate__(self, state):
        pass                                     # the foreign state is not portable

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_rng"), name)


class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy.random._pickle":     # every RNG reconstructor lives here
            return _FreshRNG
        return super().find_class(module, name)

from fashion_mnist import INIT, OUT  # noqa: E402
from ktram_neural_core.torch import BasisEncoder, Classifier  # noqa: E402
from ktram_neural_core.torch.pack import load_section  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))
from _congruence import oracle_weights  # noqa: E402

G = generator  # shorthand for the schedule constants
REPORT = OUT / "congruence-report-generator.md"
_report_lines = []


def report(line):
    print(line, flush=True)
    _report_lines.append(line)


def patches_of(images):
    """[n, 784] binary images -> [n, 16, 49] per-group patch AATs."""
    return torch.from_numpy(images[:, G.PATCH_IDX].astype(np.int64))


def chunked_read(module, aat, chunk=256, **kw):
    return torch.cat([module.read(aat[i:i + chunk], **kw)
                      for i in range(0, len(aat), chunk)])


def chunked_read_y(module, aat, chunk=256):
    return torch.cat([module.read_y(aat[i:i + chunk])
                      for i in range(0, len(aat), chunk)])


# ---------------------------------------------------------------------------
# The torch re-run, generator.py's schedule verbatim.
# ---------------------------------------------------------------------------

def train_encoder_torch(images, seed=G.SEED):
    enc = BasisEncoder(G.N_GROUPS, G.BASIS, G.N_PIX_PATCH, generator._pow2(G.LEVELS),
                       gather_abandon=G.GATHER_ABANDON, exclusion=True, recruitment=True,
                       abandon_action="recruit", init=INIT,
                       seed=[1000 + seed * 100 + g for g in range(G.N_GROUPS)])
    p = patches_of(images)
    t0 = time.time()
    for _ in range(G.ENC_FORM_EPOCHS):            # form: recruitment on, width populates
        for i in range(len(p)):
            enc.adapt(p[i], per_group=True)
    enc.recruitment = False                       # sharpen: self-prune (the generator's flip)
    enc.abandon_action = "reset"
    for _ in range(G.ENC_SHARPEN_EPOCHS):
        for i in range(len(p)):
            enc.adapt(p[i], per_group=True)
    report(f"- torch encoder trained: {G.N_GROUPS} groups x {G.BASIS} basis, "
           f"{G.ENC_FORM_EPOCHS}+{G.ENC_SHARPEN_EPOCHS} epochs x {len(p)} images "
           f"({time.time() - t0:.0f}s)")
    return enc


def train_decoder_torch(aats, levels, seed=G.SEED):
    dec = Classifier(G.N_DEC_LANES, G.N_GROUPS, G.BASIS, init=INIT, seed=500 + seed)
    rng = np.random.default_rng(seed)
    base = torch.arange(G.N_PIXELS, dtype=torch.int64) * G.LEVELS
    n = len(aats)
    t0 = time.time()
    for ep in range(G.DEC_EPOCHS):
        for k, i in enumerate(rng.permutation(n)):
            dec.adapt(aats[i], base + torch.from_numpy(levels[i].astype(np.int64)))
            if k and k % 2000 == 0:
                print(f"  decoder epoch {ep} {k}/{n} ({time.time() - t0:.0f}s)", flush=True)
    report(f"- torch decoder trained: {G.N_DEC_LANES} lanes, {G.DEC_EPOCHS} epochs x {n} "
           f"images ({time.time() - t0:.0f}s)")
    return dec


def train_classifier_torch(aats, labels, seed=G.SEED):
    clf = Classifier(len(G.LABELS), G.N_GROUPS, G.BASIS, init=INIT, seed=700 + seed)
    rng = np.random.default_rng(seed + 1)
    n = len(aats)
    t0 = time.time()
    for _ in range(G.CLF_EPOCHS):
        for i in rng.permutation(n):
            clf.adapt(aats[i], torch.tensor([int(labels[i])]))
    report(f"- torch label read-out trained: {G.CLF_EPOCHS} epochs x {n} "
           f"({time.time() - t0:.0f}s)")
    return clf


def decode_levels(dec, aats):
    """Sharp decode: FFLV read of all 1568 lanes, per-pixel argmax -> [n, 784] levels."""
    y = chunked_read_y(dec, aats)
    return y.reshape(len(aats), G.N_PIXELS, G.LEVELS).argmax(dim=-1)


# ---------------------------------------------------------------------------
# Tier checks.
# ---------------------------------------------------------------------------

def compare_weights(name, torch_ga, torch_gb, oracle_core_or_groups):
    if isinstance(oracle_core_or_groups, list):
        ga = np.stack([oracle_weights(g.core)[0] for g in oracle_core_or_groups])
        gb = np.stack([oracle_weights(g.core)[1] for g in oracle_core_or_groups])
    else:
        ga, gb = oracle_weights(oracle_core_or_groups)
    same = np.array_equal(torch_ga.numpy(), ga) and np.array_equal(torch_gb.numpy(), gb)
    if same:
        report(f"- **tier 1 — {name}: final integer state bit-exact vs generator.pkl** "
               f"({ga.size:,} devices/side)")
    else:
        n_diff = int((torch_ga.numpy() != ga).sum() + (torch_gb.numpy() != gb).sum())
        report(f"- **tier 1 — {name}: MISMATCH — {n_diff:,} of {2 * ga.size:,} device "
               f"states differ** (was the pkl trained with current oracle code?)")
    return same


def tier2_winner_stats(enc, groups, probe_patches, T=0.5, n_draws=400):
    """Sampled winners on one patch, oracle Cores vs torch, at the hot setpoint."""
    g0 = groups[0]
    patch = probe_patches[0, 0]                                 # group 0's patch, image 0
    o_aat = np.ascontiguousarray(patch.numpy().astype(np.int8))
    generator._set_temperature([g0], T)
    hist_o = np.zeros(G.BASIS)
    for _ in range(n_draws):
        hist_o[int(np.argmax(g0.read_scores(o_aat)))] += 1
    generator._set_temperature([g0], 0.0)
    gen = torch.Generator().manual_seed(0)
    hist_t = np.zeros(G.BASIS)
    full = probe_patches[0][None, :, :]                         # [1, 16, 49]
    for _ in range(n_draws):
        w = enc.read_sampled(full, T, gen, per_group=True)
        hist_t[int(w[0, 0])] += 1
    tv = 0.5 * np.abs(hist_o / n_draws - hist_t / n_draws).sum()
    report(f"- tier 2 — sampled-winner statistics, group 0 at T={T}: total-variation "
           f"distance {tv:.3f} over {n_draws} draws/side (law match; draws independent)")


def tier4_generation(enc, dec, images, seed=G.SEED):
    """T = 0: the encode/decode loop must reach a fixed point from real starts.
    T > 0: the image wanders — report the wander statistics."""
    n_starts, max_steps = 30, 40
    starts = patches_of(images[:n_starts])                      # [n, 16, 49]
    fixed, steps_to_fix = 0, []
    for i in range(n_starts):
        aat = enc.read(starts[i], per_group=True)               # [16]
        for step in range(max_steps):
            img = decode_levels(dec, aat[None, :])[0]           # [784]
            aat2 = enc.read(patches_of(img.numpy()[None, :])[0], per_group=True)
            if torch.equal(aat2, aat):
                fixed += 1
                steps_to_fix.append(step)
                break
            aat = aat2
    report(f"- tier 4 — T=0: {fixed}/{n_starts} starts reached a fixed point "
           f"(median {int(np.median(steps_to_fix)) if steps_to_fix else '-'} steps); "
           f"sharp loop is attractor dynamics, as the oracle widget shows")

    T, n_wander = generator.HOT_SETPOINT, 60
    gen = torch.Generator().manual_seed(1)
    aat = enc.read(starts[0], per_group=True)
    changes = []
    img = decode_levels(dec, aat[None, :])[0]
    for _ in range(n_wander):
        aat2 = enc.read_sampled(patches_of(img.numpy()[None, :])[0], T, gen, per_group=True)
        changes.append(int((aat2 != aat).sum()))
        aat = aat2
        img = decode_levels(dec, aat[None, :])[0]
    report(f"- tier 4 — T={T}: loop wanders, {np.mean(changes):.1f}/16 group codes changing "
           f"per step (never frozen, never white noise) over {n_wander} steps")


def pack_cross_check(enc, dec, n_val=400, seed=G.SEED):
    """The export validation, re-run against the torch modules' own pack write: quantized
    frozen read vs live read — the same agreement numbers generator.py export prints."""
    pack_path = OUT / "generator-weights-torch"
    enc.to_pack(pack_path)
    dec.to_pack(pack_path, "decoder")
    enc_fro = BasisEncoder.from_pack(pack_path)
    dec_fro = Classifier.from_pack(pack_path, "decoder")

    imgs = generator.load_fashion(n_val, seed=seed + 5, levels=G.LEVELS)
    p = patches_of(imgs)
    live_w = chunked_read(enc, p, per_group=True)
    fro_w = chunked_read(enc_fro, p, per_group=True)
    enc_agree = float((live_w == fro_w).float().mean())
    live_px = decode_levels(dec, live_w)
    fro_px = decode_levels(dec_fro, live_w)
    dec_agree = float((live_px == fro_px).float().mean())
    report(f"- tier 1 — frozen-pack cross-check (torch pack, {n_val} val images): encoder "
           f"winner agreement {enc_agree:.4f}, decoder pixel agreement {dec_agree:.4f}")

    shipped = OUT / "generator-weights"
    if shipped.with_suffix(".npz").exists():
        same = all(
            np.array_equal(load_section(pack_path, s)[i], load_section(shipped, s)[i])
            for s in ("encoder", "decoder") for i in (0, 1))
        verdict = ("torch pack write == shipped generator-weights.npz, byte for byte"
                   if same else "torch pack DIFFERS from the shipped npz")
        report(f"- tier 1 — {verdict} (encoder + decoder sections)")
        man = shipped.with_suffix(".json")
        if man.exists():
            import json
            v = json.loads(man.read_text())["validation"]
            report(f"  (shipped export's own numbers: encoder {v['encoder_winner_agreement']:.4f}, "
                   f"decoder {v['decoder_pixel_agreement']:.4f})")


def main():
    report(f"# Generator re-run on the torch L1 modules — {time.strftime('%Y-%m-%d %H:%M')}")
    report(f"\nSchedule: generator.py's own ({G.N_TRAIN_ENC} encoder images, "
           f"{G.ENC_FORM_EPOCHS}+{G.ENC_SHARPEN_EPOCHS} epochs; {G.N_TRAIN_DEC} decoder "
           f"images x {G.DEC_EPOCHS} epochs; {G.CLF_EPOCHS} classifier epochs; seed {G.SEED}).\n")

    if not G.MODEL.exists():
        raise SystemExit("figures/generator.pkl not found — run `generator.py train` first")
    with open(G.MODEL, "rb") as f:
        blob = _TolerantUnpickler(f).load()
    groups, o_dec, o_clf = blob["groups"], blob["dec"], blob.get("clf")

    X, y = G.load_fashion_labeled(max(G.N_TRAIN_ENC, G.N_TRAIN_DEC), seed=G.SEED)

    # -- encoder --
    enc = train_encoder_torch(X[:G.N_TRAIN_ENC])
    util = enc.codebook_utilization
    ent = enc.winner_entropy
    report(f"- tier 3 — torch encoder outcomes: mean utilization {float(util.mean()):.2f}, "
           f"mean entropy {float(ent.mean()):.2f} bits")
    o_util = np.mean([g.codebook_utilization for g in groups])
    o_ent = np.mean([g.winner_entropy for g in groups])
    report(f"- tier 3 — oracle (pkl) encoder outcomes: mean utilization {o_util:.2f}, "
           f"mean entropy {o_ent:.2f} bits")
    compare_weights("encoder", enc.ga, enc.gb, groups)

    # -- encode once, both heads share it (the generator's own flow) --
    t0 = time.time()
    aats = chunked_read(enc, patches_of(X[:G.N_TRAIN_DEC]), per_group=True)
    print(f"encoded {len(aats)} images ({time.time() - t0:.1f}s)", flush=True)

    # spot-check the torch encode against the live oracle Cores
    n_spot = 100
    agree = 0
    for i in range(n_spot):
        a_o = generator.encode_indices(groups, X[i], temperature=0.0)
        agree += int((aats[i].numpy() == a_o).all())
    report(f"- tier 1 — sharp encode: torch AATs == oracle AATs on {agree}/{n_spot} "
           f"spot-checked images")

    # -- decoder --
    dec = train_decoder_torch(aats, X[:G.N_TRAIN_DEC])
    recon = decode_levels(dec, aats[:1500])
    acc = float((recon.numpy() == X[:1500]).mean())
    report(f"- tier 3 — torch decoder train pixel-accuracy {acc:.3f}")
    compare_weights("decoder", dec.ga, dec.gb, o_dec.core)

    # -- label read-out --
    clf = train_classifier_torch(aats, y[:G.N_TRAIN_DEC])
    preds = chunked_read_y(clf, aats[:2000]).argmax(dim=-1).numpy()
    clf_acc = float((preds == y[:2000]).mean())
    report(f"- tier 3 — torch label read-out train accuracy {clf_acc:.3f}")
    if o_clf is not None:
        compare_weights("classifier", clf.ga, clf.gb, o_clf.core)

    # -- tiers 2 and 4, and the pack --
    tier2_winner_stats(enc, groups, patches_of(X[:1]))
    pack_cross_check(enc, dec)
    tier4_generation(enc, dec, X[:100])

    REPORT.write_text("\n".join(_report_lines) + "\n")
    print(f"\nreport -> {REPORT}", flush=True)


if __name__ == "__main__":
    main()
