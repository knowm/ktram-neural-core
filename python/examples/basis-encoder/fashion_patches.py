"""Patch basis features forming — a wider, richer basis than the whole-image hero.

Instead of feeding a whole 28x28 garment to one lane (fashion_mnist.py), we sample small PATCH x PATCH
tiles from the images and feed those to a wider WTA bank. Each image yields many patches, so a single
pass over far FEWER images supplies many more training inputs, and the learned lane features become
LOCAL basis patches — the oriented edges, corners, and texture primitives of the sparse-coding
lineage — rather than whole-garment silhouettes. That is the classic "learn a dictionary of image
patches" result, here rendered as a grid of primitives sharpening from noise over time.

Sampling: draw random PATCH x PATCH tiles from the raw images and keep only the ones with real
structure (raw std above a threshold), so lanes learn edges and texture instead of the flat black
void. Encoding is the same 3-level per-pixel binning as the whole-image run: each patch is an int8
length-(PATCH*PATCH) AAT of level ids, streamed straight into the group.

Feature map: same as the whole-image hero — a lane's intensity-weighted light-channel differential
per pixel, reshaped to PATCH x PATCH, is the learned basis patch.

Self-pruning is two-phase — FORM then SHARPEN. Phase 1 runs with recruitment on so the whole width
populates and a full basis forms first. Phase 2 (halfway through) turns recruitment OFF and resets
the cycle: nothing props up a lane that stops winning, so non-competitive lanes are depressed back
toward init and fade while the well-formed lanes keep sharpening. The codebook prunes itself down to
the primitives the data supports; the survivors are read out by win count alone (one hardware counter
per lane), no post-hoc pruning step. The forming GIF shows the whole field on one scale so you watch
the losers fade after the switch; the still shows the survivors only.

    python fashion_patches.py run       # sample patches, stream, snapshot -> figures/patches.npz
    python fashion_patches.py render    # figures/patches.npz -> patch-forming.gif + patch-features.png
    python fashion_patches.py           # run then render

All outputs land in the gitignored figures/ dir; the dataset caches to ~/scikit_learn_data (outside
the repo).
"""

import sys
import time
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ktram_neural_core import BasisGroup, Core  # noqa: E402
from fashion_mnist import _pow2, OUT, LEVELS, INIT, SIDE  # noqa: E402

CACHE = OUT / "patches.npz"

# Config. Fewer images than the whole-image run — each image contributes many patches.
PATCH = 8              # tile side; the basis patch is PATCH x PATCH
N_PATCHES = 50000      # total patches streamed (one pass); the basis forms in the first several k
N_IMAGES = 6000        # sample patches from this many images (each gives several kept patches)
S = 64                 # lanes = basis patches (an 8x8 grid) — a much wider basis than the 16-lane run
GATHER_ABANDON = 96    # recruitment cadence (~1.5 x S, mirroring the whole-image 24/16 ratio)
STD_MIN = 45.0         # keep a patch only if its raw pixel std exceeds this (drops flat/background tiles)
SEED = 0
N_FRAMES = 60          # log-spaced toward the front, where the forming happens
N_PIX = PATCH * PATCH


def load_patches(n_patches=N_PATCHES, n_images=N_IMAGES, patch=PATCH, std_min=STD_MIN, seed=SEED,
                 levels=LEVELS):
    """`n_patches` structured PATCH x PATCH tiles as an int8 [n_patches, PATCH*PATCH] AAT array.

    Random offsets are drawn from the first `n_images` images; a tile is kept only if its raw pixel
    std exceeds `std_min` (so flat background tiles are rejected and the basis learns real edges and
    texture). Each kept patch is binned into `levels` intensity channels exactly like the whole-image
    encoder."""
    from sklearn.datasets import fetch_openml
    print("loading Fashion-MNIST (cached to ~/scikit_learn_data) ...", flush=True)
    ds = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
    X = ds.data.astype(np.float32)                       # [70000, 784], 0..255
    rng = np.random.default_rng(seed)
    X = X[rng.permutation(len(X))[:n_images]].reshape(-1, SIDE, SIDE)

    edges = np.linspace(0.0, 255.0, levels + 1)[1:-1]    # e.g. levels=3 -> [85, 170]
    maxoff = SIDE - patch
    out = np.empty((n_patches, N_PIX), dtype=np.int8)
    kept, tried = 0, 0
    while kept < n_patches:
        img = X[rng.integers(len(X))]
        r, c = rng.integers(maxoff + 1), rng.integers(maxoff + 1)
        tile = img[r:r + patch, c:c + patch]
        tried += 1
        if tile.std() < std_min:
            continue
        out[kept] = np.digitize(tile.ravel(), edges).astype(np.int8)
        kept += 1
    print(f"  kept {kept} patches from {tried} draws ({100*kept/tried:.0f}% pass std>{std_min}), "
          f"{n_images} images, patch={patch}x{patch}", flush=True)
    return out


def _patch_feature_maps(group, levels=LEVELS):
    """Every lane's expected-intensity map, [S, PATCH, PATCH] — the whole-image render, patch-sized.

    For each pixel, read the unnormalized differential (Ga - Gb) on each non-background channel and
    weight it by the intensity level: feature = sum_c c * relu(Ga_c - Gb_c). Unnormalized, so quiet
    pixels stay quiet."""
    full = [np.full(N_PIX, c, dtype=np.int8) for c in range(levels)]
    maps = np.zeros((group.channels, PATCH, PATCH), dtype=np.float32)
    for lane in range(group.channels):
        acc = np.zeros(N_PIX, dtype=np.float32)
        for c in range(1, levels):
            gab = group.core.read_gab(lane, full[c])
            d = np.array([ga - gb for ga, gb in gab], dtype=np.float32)
            acc += c * np.maximum(d, 0.0)
        maps[lane] = acc.reshape(PATCH, PATCH)
    return maps


def _read_stats(grp, patches, n_sample=20000):
    """Final per-lane win count from a read-only FFLV pass — the hardware-native survivor readout
    (one counter per lane). Survivors are the lanes whose count clears a fraction of the top lane's."""
    m = min(n_sample, len(patches))
    win = np.zeros(grp.channels)
    for i in range(m):
        win[grp.read(patches[i])] += 1
    return win


def run(n_patches=N_PATCHES, s=S, seed=SEED, n_frames=N_FRAMES, levels=LEVELS, switch_frac=0.5):
    """Two-phase: FORM then SHARPEN.

    Phase 1 (recruitment ON): the bank forms with idle lanes force-fed, so the whole width populates
    and a full basis develops first — nothing is starved before it has a chance to specialize.
    Phase 2 (recruitment OFF, cycle reset): at switch_frac of the run recruitment is turned off, so
    nothing props up a lane that stops winning; the strong lanes keep sharpening while the rest are
    depressed back toward init and fade. The codebook prunes itself down to what the data supports,
    and the survivors are read out by win count alone.
    """
    patches = load_patches(n_patches, seed=seed, levels=levels)
    core = Core(1, _pow2(levels), spaces_per_lane=N_PIX, num_lanes=s, model="byte", init=INIT,
                read_noise=0, seed=100 + seed)
    # Start in formation mode (recruitment on); the switch below turns it into self-pruning.
    grp = BasisGroup(core, s, gather_abandon=GATHER_ABANDON, exclusion=True,
                     recruitment=True, abandon_action="recruit")
    switch_at = int(switch_frac * n_patches)
    print(f"  levels={levels} S={s} patches={n_patches} patch={PATCH}x{PATCH} "
          f"gather_abandon={GATHER_ABANDON} switch_at={switch_at}", flush=True)

    sched = set([0] + [int(x) for x in np.geomspace(20, n_patches - 1, n_frames - 1)] + [n_patches - 1])
    frames, samples = [], []
    t0 = time.time()
    for i in range(n_patches):
        if i == switch_at:
            grp.recruitment = False        # stop force-feeding idle lanes ...
            grp.abandon_action = "reset"    # ... and reset the cycle so exclusion keeps the survivors going
            print(f"  SWITCH at {i}: recruitment off, reset (sharpen / self-prune)", flush=True)
        if i in sched:
            frames.append(_patch_feature_maps(grp, levels))
            samples.append(i)
            np.savez(CACHE, frames=np.array(frames), samples=np.array(samples), n=n_patches, s=s,
                     patch=PATCH, switch_at=switch_at)
        grp.adapt(patches[i])
        if i == 200:
            print(f"  [200/{n_patches}] ETA ~{(time.time()-t0)/201*n_patches/60:.1f} min", flush=True)
        if i and i % 2000 == 0:
            print(f"  {i}/{n_patches}  util={grp.codebook_utilization:.2f}  "
                  f"entropy={grp.winner_entropy:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    frames.append(_patch_feature_maps(grp, levels))
    samples.append(n_patches)
    # Hardware-native survivor readout: one read-only FFLV pass -> per-lane win count.
    win = _read_stats(grp, patches)
    np.savez(CACHE, frames=np.array(frames), samples=np.array(samples), n=n_patches, s=s, patch=PATCH,
             switch_at=switch_at, win=win)
    alive = int((win > 0.05 * (win.max() or 1)).sum())
    print(f"done: {len(frames)} frames, {alive}/{s} survive, {time.time()-t0:.0f}s -> {CACHE}",
          flush=True)


def render(fps=12, hold_seconds=3.0):
    from image_basis import render_forming
    render_forming(CACHE, OUT / "patch-forming.gif", OUT / "patch-features.png",
                   cell=0.9, fps=fps, hold_seconds=hold_seconds, counter_label="patches seen")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("run", "all"):
        run()
    if what in ("render", "all"):
        render()
