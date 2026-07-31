"""The thermal pattern generator — train the model and export the browser weights (Chapter 6b §10).

An encode/decode loop with a temperature knob, built end-to-end on kT-RAM neural lanes:

  encoder   a 4x4 grid of 16 non-overlapping 7x7 patches over a 28x28 image, one BasisGroup per
            patch (64 basis each), trained unsupervised form-then-sharpen. Group g emits its winner
            lane; the image code is the 16-tuple AAT (one channel per patch). Binary ink/no-ink.
  decoder   784 per-pixel classifiers as ONE Core of 1568 lanes (one lane per pixel-level), trained
            with the shipped LinearClassifier routine (FF read, then RH/RL/RF). Decode = FFLV read,
            argmax per pixel. Every pixel reads all 16 groups, so the decoder couples them globally.
  read-out  the same AAT feeds a supervised label classifier (one Core, one lane per class).
  loop      state = image; decode AAT -> image, encode image -> AAT, repeat.

Temperature is the encoder Cores' read_noise gain (Core.read_sample): at 0 the winner is the sharp
argmax and the loop settles to a fixed point; raise it and each read carries emulator read noise, so
the winner is sampled and the image wanders between attractors. Play with it via generator_widget.py.

  python generator.py train       # train encoder + decoder + label read-out -> figures/generator.pkl
  python generator.py classifier  # retrain just the label read-out on the frozen encoder
  python generator.py export      # int8 weights (diff + mag + noise coeffs) -> figures/generator-weights.*

Outputs land in the gitignored figures/ dir; Fashion-MNIST caches to ~/scikit_learn_data.
"""

import sys
import time
import pickle
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ktram_neural_core import BasisGroup, Core  # noqa: E402
from fashion_mnist import _pow2, OUT, INIT, SIDE, load_fashion  # noqa: E402

MODEL = OUT / "generator.pkl"

# --- 4x4 architecture, binary AAT --------------------------------------------
# 4x4 patches give the code enough capacity for a deep attractor landscape; BINARY ink/no-ink (a
# per-image-mean split) keeps interior detail instead of smearing a gray level into solid blobs.
PATCH = 7                         # patch side; 4 positions tile the 28px axis exactly, no overlap
STARTS = [0, 7, 14, 21]           # non-overlapping 7x7 tiles
GRID = len(STARTS)                # 4
N_GROUPS = GRID * GRID            # 16 patch positions
BASIS = 64                        # lanes per group; start WIDE and self-prune to sharp survivors
LEVELS = 2                        # BINARY: 0 = no ink, 1 = ink (per-image-mean split)
N_PIX_PATCH = PATCH * PATCH       # 49
N_PIXELS = SIDE * SIDE            # 784
N_DEC_LANES = N_PIXELS * LEVELS   # 1568 — one neural lane per (pixel, level)

# --- training budget --------------------------------------------------------
N_TRAIN_ENC = 12000               # images per encoder epoch (each group sees one patch per image)
ENC_FORM_EPOCHS = 1               # full forming pass(es), recruitment ON (populate the width)
ENC_SHARPEN_EPOCHS = 1            # full sharpening pass(es), recruitment OFF + reset (self-prune)
N_TRAIN_DEC = 8000                # images to train the heads on the frozen code (encoded once)
DEC_EPOCHS = 4                    # passes of the decoder's kT-RAM routine
CLF_EPOCHS = 6                    # passes of the label read-out's routine
GATHER_ABANDON = 96               # recruitment cadence (~1.5 x BASIS)
SEED = 0


LABELS = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
          "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]     # Fashion-MNIST class order


def load_fashion_labeled(n, seed=0):
    """n Fashion-MNIST images (binary, per-image-mean split) AND their int labels: ([n,784], [n])."""
    from sklearn.datasets import fetch_openml
    ds = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
    X = ds.data.astype(np.float32)
    y = ds.target.astype(np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[:n]
    X = X[idx]
    return (X > X.mean(axis=1, keepdims=True)).astype(np.int8), y[idx]


def _patch_indices():
    """Flat pixel indices of each of the 16 patches into a 28x28 image, [16, 49]."""
    grid = np.arange(SIDE * SIDE).reshape(SIDE, SIDE)
    return np.array([grid[r0:r0 + PATCH, c0:c0 + PATCH].ravel()
                     for r0 in STARTS for c0 in STARTS])


PATCH_IDX = _patch_indices()


# ---------------------------------------------------------------------------
# Encoder: 16 patch codebooks, form-then-sharpen. Trained sharp (read_noise=0).
# ---------------------------------------------------------------------------

def _train_group(patches, seed):
    """Form-then-sharpen with FULL epochs: ENC_FORM_EPOCHS passes with recruitment on (populate the
    64-wide bank), then ENC_SHARPEN_EPOCHS passes with recruitment off + reset (non-competitive lanes
    fade, survivors sharpen). Starting wide and self-pruning yields a sharp, high-quality basis."""
    core = Core(1, _pow2(LEVELS), spaces_per_lane=N_PIX_PATCH, num_lanes=BASIS, model="byte",
                init=INIT, read_noise=0, seed=seed)
    grp = BasisGroup(core, BASIS, gather_abandon=GATHER_ABANDON, exclusion=True, recruitment=True,
                     abandon_action="recruit")
    for _ in range(ENC_FORM_EPOCHS):            # form: recruitment on, whole width populates
        for p in patches:
            grp.adapt(p)
    grp.recruitment = False                     # sharpen: self-prune
    grp.abandon_action = "reset"
    for _ in range(ENC_SHARPEN_EPOCHS):
        for p in patches:
            grp.adapt(p)
    return grp


def train_encoder(images, seed=SEED):
    groups = []
    t0 = time.time()
    for g in range(N_GROUPS):
        grp = _train_group(images[:, PATCH_IDX[g]], seed=1000 + seed * 100 + g)
        groups.append(grp)
        print(f"  group {g:>2}: util={grp.codebook_utilization:.2f} "
              f"entropy={grp.winner_entropy:.2f} ({time.time()-t0:.0f}s)", flush=True)
    return groups


# ---------------------------------------------------------------------------
# Encoding — reads go THROUGH the live Cores. Temperature = the Cores' read_noise gain, so the
# winner argmax is sampled by the emulator's read_sample (kT-bit read noise), not by anything here.
# ---------------------------------------------------------------------------

def _set_temperature(groups, temperature):
    """Set every encoder Core's read-noise gain. T=0 -> deterministic sharp read."""
    for grp in groups:
        grp.core.set_read_noise(read_noise=float(temperature))


def encode_indices(groups, image, temperature=0.0):
    """image: int8 [784] level ids -> int8 [16] winner indices. At temperature>0 each lane's FFLV
    read carries emulator read noise, so the WTA winner is a sample, not the argmax."""
    _set_temperature(groups, temperature)
    aat = np.empty(N_GROUPS, dtype=np.int8)
    for g, grp in enumerate(groups):
        patch = np.ascontiguousarray(image[PATCH_IDX[g]])
        y = grp.read_scores(patch)                    # noisy iff read_noise>0 (drawn in the Core)
        aat[g] = int(np.argmax(y))
    return aat


# ---------------------------------------------------------------------------
# Decoder: 784 per-pixel neural-lane classifiers, ONE kT-RAM Core (1568 lanes).
# One neural lane per (pixel, level), all reading the same 16-space AAT. The shipped LinearClassifier
# routine per pixel: FF read at full V, then RH on the true level, RL on a false positive, RF on a
# true negative. Decode is the FFLV read of all lanes, argmax within each pixel's LEVELS lanes.
# ---------------------------------------------------------------------------

class NeuralPixelDecoder:
    def __init__(self, seed=0):
        self.core = Core(1, BASIS, spaces_per_lane=N_GROUPS, num_lanes=N_DEC_LANES,
                         model="byte", init=INIT, read_noise=0, seed=seed)

    def train(self, aat, levels):
        """aat: int [16] winner indices. levels: int8 [784] target level per pixel."""
        a = np.ascontiguousarray(aat, dtype=np.int8)
        for p in range(N_PIXELS):
            target = levels[p]
            base = p * LEVELS
            for lv in range(LEVELS):
                lane = base + lv
                y = self.core.evaluate(a, "FF", lane)
                if lv == target:
                    self.core.evaluate(a, "RH", lane)      # correct level
                elif y > 0:
                    self.core.evaluate(a, "RL", lane)      # false positive
                else:
                    self.core.evaluate(a, "RF", lane)      # true negative

    def decode(self, aat):
        """Sharp FFLV read of all lanes -> per-pixel argmax level. read_noise=0, so deterministic."""
        a = np.ascontiguousarray(aat, dtype=np.int8)
        y = np.array([self.core.evaluate(a, "FFLV", lane) for lane in range(N_DEC_LANES)])
        return y.reshape(N_PIXELS, LEVELS).argmax(axis=1).astype(np.int8)


class NeuralLabelClassifier:
    """The supervised label read-out over the SAME frozen AAT — one kT-RAM Core, one lane per class.
    Same instruction routine as the decoder / the shipped LinearClassifier (FF read → RH/RL/RF). This
    is the 'label optional' half of the chapter: the unsupervised codebook feeds both an unsupervised
    decoder and a supervised class read-out, no change to the encoder."""

    def __init__(self, n_labels=len(LABELS), seed=0):
        self.n_labels = n_labels
        self.core = Core(1, BASIS, spaces_per_lane=N_GROUPS, num_lanes=n_labels,
                         model="byte", init=INIT, read_noise=0, seed=seed)

    def train(self, aat, label):
        a = np.ascontiguousarray(aat, dtype=np.int8)
        for lane in range(self.n_labels):
            y = self.core.evaluate(a, "FF", lane)
            if lane == label:
                self.core.evaluate(a, "RH", lane)
            elif y > 0:
                self.core.evaluate(a, "RL", lane)
            else:
                self.core.evaluate(a, "RF", lane)

    def predict(self, aat):
        a = np.ascontiguousarray(aat, dtype=np.int8)
        y = [self.core.evaluate(a, "FFLV", lane) for lane in range(self.n_labels)]
        return int(np.argmax(y))


# ---------------------------------------------------------------------------
# Train + save.
# ---------------------------------------------------------------------------

def _reuse_encoder():
    """Reuse pickled encoder Cores if present and the geometry matches (iterate the decoder without
    retraining the encoder). Returns groups or None."""
    if not MODEL.exists():
        return None
    with open(MODEL, "rb") as f:
        blob = pickle.load(f)
    groups = blob.get("groups")
    if (groups and len(groups) == N_GROUPS and groups[0].core.spaces_per_lane == N_PIX_PATCH
            and groups[0].channels == BASIS):     # basis width must match too, else retrain
        print("reusing pickled encoder Cores from prior run", flush=True)
        return groups
    return None


def _train_decoder(aats, levels, seed):
    print(f"training decoder: {N_DEC_LANES} lanes, {DEC_EPOCHS} epochs x {len(aats)} images ...",
          flush=True)
    dec = NeuralPixelDecoder(seed=500 + seed)
    rng = np.random.default_rng(seed)
    n = len(aats)
    t0 = time.time()
    for ep in range(DEC_EPOCHS):
        for k, i in enumerate(rng.permutation(n)):
            dec.train(aats[i], levels[i])
            if k and k % 1000 == 0:
                el = time.time() - t0
                done = ep * n + k
                print(f"  epoch {ep} {k}/{n}  ({el:.0f}s, ETA {el/done*(DEC_EPOCHS*n-done)/60:.1f} min)",
                      flush=True)
    recon = np.stack([dec.decode(aats[i]) for i in range(min(1500, n))])
    acc = float((recon == levels[:len(recon)]).mean())
    print(f"  decoder train pixel-accuracy = {acc:.3f}", flush=True)
    return dec, acc


def _train_label_readout(aats, labels, seed, epochs=CLF_EPOCHS):
    print(f"training label read-out: {len(LABELS)} lanes, {epochs} epochs x {len(aats)} images ...",
          flush=True)
    clf = NeuralLabelClassifier(seed=700 + seed)
    rng = np.random.default_rng(seed + 1)
    n = len(aats)
    for _ in range(epochs):
        for i in rng.permutation(n):
            clf.train(aats[i], int(labels[i]))
    acc = float(np.mean([clf.predict(aats[i]) == labels[i] for i in range(min(2000, n))]))
    print(f"  label read-out train accuracy = {acc:.3f}", flush=True)
    return clf, acc


def _encode_set(groups, X):
    """Encode X to sharp AATs once; both heads share this."""
    print(f"encoding {len(X)} images to {N_GROUPS}-tuple AATs (sharp) ...", flush=True)
    t0 = time.time()
    aats = np.stack([encode_indices(groups, X[i], temperature=0.0) for i in range(len(X))])
    print(f"  aats {aats.shape} ({time.time()-t0:.0f}s)", flush=True)
    return aats


def train(seed=SEED):
    X, y = load_fashion_labeled(max(N_TRAIN_ENC, N_TRAIN_DEC), seed=seed)   # binary levels + labels
    groups = _reuse_encoder()
    if groups is None:
        print(f"training encoder: {N_GROUPS} groups x {BASIS} basis, "
              f"{ENC_FORM_EPOCHS} form + {ENC_SHARPEN_EPOCHS} sharpen epoch(s) x {N_TRAIN_ENC} "
              f"images (7x7 patches, binary, self-prune) ...", flush=True)
        groups = train_encoder(X[:N_TRAIN_ENC], seed=seed)
        avg_alive = np.mean([g.codebook_utilization for g in groups])
        print(f"  mean surviving basis fraction after self-prune = {avg_alive:.2f} "
              f"(~{avg_alive*BASIS:.0f}/{BASIS} lanes/group)", flush=True)

    aats = _encode_set(groups, X[:N_TRAIN_DEC])           # encode ONCE, both heads share it
    dec, acc = _train_decoder(aats, X[:N_TRAIN_DEC], seed)
    clf, clf_acc = _train_label_readout(aats, y[:N_TRAIN_DEC], seed)

    with open(MODEL, "wb") as f:
        pickle.dump({"groups": groups, "dec": dec, "clf": clf, "seed": seed}, f)
    print(f"saved model -> {MODEL}  (decoder acc {acc:.3f}, label acc {clf_acc:.3f})", flush=True)


def train_classifier(seed=SEED):
    """Augment an existing pickled model with the label read-out, without retraining encoder/decoder."""
    with open(MODEL, "rb") as f:
        blob = pickle.load(f)
    X, y = load_fashion_labeled(N_TRAIN_DEC, seed=seed)
    aats = _encode_set(blob["groups"], X)
    clf, acc = _train_label_readout(aats, y, seed)
    blob["clf"] = clf
    with open(MODEL, "wb") as f:
        pickle.dump(blob, f)
    print(f"updated model -> {MODEL}  (label read-out acc {acc:.3f})", flush=True)


def _load():
    with open(MODEL, "rb") as f:
        blob = pickle.load(f)
    return blob["groups"], blob["dec"]


def _load_clf():
    with open(MODEL, "rb") as f:
        return pickle.load(f).get("clf")


# ---------------------------------------------------------------------------
# Export the int8 weights (diff + mag + read-noise coefficients) so a numpy/JS reimplementation can
# reproduce the Core exactly — the TwoOne divider read AND the read_sample noise. Validated below
# against the live Cores.
# ---------------------------------------------------------------------------

EXPORT = OUT / "generator-weights"        # .npz (arrays) + .json (manifest) + .bin (packed int8)
HOT_SETPOINT = 0.5                        # a good "hot" read_noise gain for the widget's temperature


def _pairs(core, n_lanes, n_spaces, n_channels):
    """Read back the differential d = Ga - Gb and magnitude m = Ga + Gb for every (lane, space,
    channel). These are the two quantities a TwoOne read needs: top = sum(d_active), bottom =
    sum(m_active), y = top/bottom (see topology.TwoOne.readout). Shapes [n_lanes, n_spaces, n_ch]."""
    diff = np.zeros((n_lanes, n_spaces, n_channels), dtype=np.float32)
    mag = np.zeros((n_lanes, n_spaces, n_channels), dtype=np.float32)
    for c in range(n_channels):
        aat = np.full(n_spaces, c, dtype=np.int8)
        for lane in range(n_lanes):
            gab = core.read_gab(lane, aat)
            diff[lane, :, c] = [ga - gb for ga, gb in gab]
            mag[lane, :, c] = [ga + gb for ga, gb in gab]
    return diff, mag


def _quantize(a):
    """int8 with one global symmetric scale; returns (int8 array, real-units-per-step)."""
    s = float(np.abs(a).max()) or 1.0
    q = np.clip(np.round(a / s * 127.0), -127, 127).astype(np.int8)
    return q, s / 127.0


def _noise_params(core):
    """The read-noise coefficients read off the Core, factored so temperature is a linear multiplier:
    at read_noise gain T, sigma_y = T * sigma_unit(m, y) with the terms below (see Core.read_sample).
    read_noise itself is NOT included — it is the temperature knob the widget/browser supplies."""
    import math
    return {
        "a_thermal_unit": core.noise_thermal * math.sqrt(core.temperature / core.read_noise_ref_T)
        * core.read_noise_ref_V,                         # thermal gain at read_noise=1
        "a_flicker_unit": core.noise_flicker,            # flicker gain at read_noise=1
        "sqrt_ref_m": math.sqrt(core.read_noise_ref_m),  # magnitude reference (byte GMAX=255)
        "flicker_ln_ref": core.flicker_decades * math.log(10.0),
        "ref_pw": core.read_noise_ref_pw,
        "read_pw": core.read_pulse_width,
        "v_fflv": abs(core.forward_low_voltage),         # low read voltage (FFLV)
    }


def export(n_val=400, seed=SEED):
    """Export the full emulator read (diff + mag) plus the read-noise coefficients, so a numpy / JS
    reimplementation reproduces the Core exactly — the sharp divider read AND the read_sample noise.
    Validated below against the live Cores."""
    import json
    import gzip
    groups, dec = _load()

    enc_diff, enc_mag = [], []
    for grp in groups:
        d, m = _pairs(grp.core, BASIS, N_PIX_PATCH, LEVELS)
        enc_diff.append(d); enc_mag.append(m)
    enc_diff = np.stack(enc_diff); enc_mag = np.stack(enc_mag)          # [16,32,49,2]
    dec_diff, dec_mag = _pairs(dec.core, N_DEC_LANES, N_GROUPS, BASIS)  # [1568,16,32]

    clf = _load_clf()
    if clf is None:
        raise SystemExit("no label read-out in the model — run `generator.py classifier` first")
    clf_diff, clf_mag = _pairs(clf.core, clf.n_labels, N_GROUPS, BASIS)   # [10,16,32]

    ed_q, ed_s = _quantize(enc_diff); em_q, em_s = _quantize(enc_mag)
    dd_q, ds_s = _quantize(dec_diff); dm_q, dm_s = _quantize(dec_mag)
    cd_q, cd_s = _quantize(clf_diff); cm_q, cm_s = _quantize(clf_mag)
    noise = _noise_params(groups[0].core)

    # ---- validate the FULL-divider quantized read vs the Core (sharp, T=0) ----
    imgs = load_fashion(n_val, seed=seed + 5, levels=LEVELS)

    def enc_winner_q(g, patch):
        sp = np.arange(N_PIX_PATCH)
        top = (ed_q[g][:, sp, patch].sum(axis=1)) * ed_s
        bot = (em_q[g][:, sp, patch].sum(axis=1)) * em_s
        return int(np.argmax(top / np.where(bot != 0, bot, 1.0)))

    def dec_levels_q(aat):
        sp = np.arange(N_GROUPS)
        idx = np.asarray(aat, dtype=np.intp)
        top = (dd_q[:, sp, idx].sum(axis=1)) * ds_s
        bot = (dm_q[:, sp, idx].sum(axis=1)) * dm_s
        y = top / np.where(bot != 0, bot, 1.0)
        return y.reshape(N_PIXELS, LEVELS).argmax(axis=1).astype(np.int8)

    _set_temperature(groups, 0.0)
    enc_hits = enc_tot = dec_pix_hits = dec_pix_tot = 0
    for i in range(n_val):
        img = imgs[i]
        aat_core = np.empty(N_GROUPS, dtype=np.int8)
        for g, grp in enumerate(groups):
            patch = np.ascontiguousarray(img[PATCH_IDX[g]])
            core_win = int(np.argmax(grp.read_scores(patch)))
            aat_core[g] = core_win
            enc_hits += (enc_winner_q(g, patch) == core_win); enc_tot += 1
        core_img = dec.decode(aat_core)
        dec_pix_hits += int((dec_levels_q(aat_core) == core_img).sum()); dec_pix_tot += N_PIXELS

    enc_agree = enc_hits / enc_tot
    dec_agree = dec_pix_hits / dec_pix_tot
    print("[export] full-divider quantized read vs Core (T=0):", flush=True)
    print(f"  encoder winner agreement = {enc_agree:.4f}  ({enc_tot} group-reads)", flush=True)
    print(f"  decoder pixel agreement  = {dec_agree:.4f}  ({dec_pix_tot} pixels)", flush=True)

    # ---- write the asset ----
    manifest = {
        "grid": GRID, "n_groups": N_GROUPS, "basis": BASIS, "levels": LEVELS,
        "patch": PATCH, "starts": STARTS, "side": SIDE, "n_pixels": N_PIXELS,
        "hot_read_noise": HOT_SETPOINT,
        "read": "TwoOne divider: top=sum(diff_active), bottom=sum(mag_active), y=top/bottom; "
                "then read_sample noise scaled by temperature (read_noise gain).",
        "encoder": {"shape": list(ed_q.shape), "diff_scale": ed_s, "mag_scale": em_s, "dtype": "int8"},
        "decoder": {"shape": list(dd_q.shape), "diff_scale": ds_s, "mag_scale": dm_s, "dtype": "int8"},
        "classifier": {"shape": list(cd_q.shape), "diff_scale": cd_s, "mag_scale": cm_s,
                       "labels": LABELS, "dtype": "int8"},
        "noise": noise,
        "validation": {"encoder_winner_agreement": enc_agree, "decoder_pixel_agreement": dec_agree},
    }
    packed = np.concatenate([ed_q.ravel(), em_q.ravel(), dd_q.ravel(), dm_q.ravel(),
                             cd_q.ravel(), cm_q.ravel()])
    EXPORT.with_suffix(".bin").write_bytes(packed.tobytes())
    EXPORT.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    np.savez(EXPORT.with_suffix(".npz"),
             enc_diff=ed_q, enc_mag=em_q, dec_diff=dd_q, dec_mag=dm_q,
             clf_diff=cd_q, clf_mag=cm_q,
             enc_diff_scale=ed_s, enc_mag_scale=em_s, dec_diff_scale=ds_s, dec_mag_scale=dm_s,
             clf_diff_scale=cd_s, clf_mag_scale=cm_s, labels=np.array(LABELS),
             **{f"noise_{k}": v for k, v in noise.items()})

    raw = packed.nbytes
    gz = len(gzip.compress(packed.tobytes(), 9))
    print(f"[export] {packed.size:,} int8 (encoder, decoder, classifier — each diff + mag)", flush=True)
    print(f"  raw {raw/1e6:.2f} MB, gzipped {gz/1e6:.2f} MB -> generator-weights.bin/.json/.npz",
          flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "train"
    if what == "train":
        train()
    elif what == "classifier":
        train_classifier()
    elif what == "export":
        export()
    else:
        raise SystemExit(f"unknown mode {what!r}; valid: train, classifier, export")
