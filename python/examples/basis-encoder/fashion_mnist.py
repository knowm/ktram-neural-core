"""Fashion-MNIST basis features forming — the visual hero.

One WTA BasisGroup over raw Fashion-MNIST images, no labels, a single streaming pass. We snapshot
every lane's learned feature (its per-pixel weight map) at log-spaced points during the pass and
render a grid of all S features forming together.

Encoding (efficient, no per-pixel encoder object): threshold each image at its own mean, so every
pixel is one space of two channels (dark=0 / light=1) and the whole image is a length-784 AAT of
0/1. Balanced by construction — every image lights exactly 784 channels, so background is as much
of the code as the ink. AATs are precomputed once as an int8 array and streamed straight into the
group.

Feature map: a lane stores a differential pair per (pixel, channel). Reading its light-channel
weight y = (Ga-Gb)/(Ga+Gb) for every pixel and reshaping to 28x28 is the learned basis feature —
the thing that comes to look like a sleeve, a sole, a collar.

    python fashion_mnist.py run       # load, stream one pass, snapshot -> figures/fashion.npz
    python fashion_mnist.py render    # figures/fashion.npz -> fashion-forming.gif + fashion-features.png
    python fashion_mnist.py           # run then render

All outputs land in the gitignored figures/ dir; the dataset caches to ~/scikit_learn_data (outside
the repo). Halt and rerun with different N/S/seed freely — the action is in the first few thousand
images.
"""

import sys
import time
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ktram_neural_core import BasisGroup, Core  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
CACHE = OUT / "fashion.npz"

# Config. Single pass over N images; the codebook forms in the first few thousand. Halt/rerun to
# change these.
N = 60000              # one full epoch over the Fashion-MNIST training set (single pass)
S = 16                 # lanes = basis features (a 4x4 grid); expect only a partial fill
GATHER_ABANDON = 24
SEED = 0
N_FRAMES = 60          # log-spaced toward the front, where the forming happens
LEVELS = 3             # intensity bins per pixel: 2 = dark/light (binary), 3 = dark/mid/bright (gray)
INIT = "medium"        # device init: "low" starts near GMIN, "medium" starts mid-range
SIDE = 28
N_PIX = SIDE * SIDE


def _pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def load_fashion(n, seed=0, levels=LEVELS):
    """N Fashion-MNIST images as an int8 [n, 784] AAT array, each pixel binned into `levels`
    intensity channels. levels=2 is dark/light (per-image-mean threshold); levels=3 is
    dark/mid/bright on fixed global intensity edges, so mid-gray fabric and bright highlights land in
    different channels and a lane can learn a graded prototype."""
    from sklearn.datasets import fetch_openml
    print("loading Fashion-MNIST (cached to ~/scikit_learn_data) ...", flush=True)
    # parser="liac-arff" avoids a pandas dependency (the default "auto" needs pandas for dense data).
    ds = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
    X = ds.data.astype(np.float32)                       # [70000, 784], 0..255
    rng = np.random.default_rng(seed)
    X = X[rng.permutation(len(X))[:n]]
    if levels <= 2:
        return (X > X.mean(axis=1, keepdims=True)).astype(np.int8)
    edges = np.linspace(0.0, 255.0, levels + 1)[1:-1]    # e.g. levels=3 -> [85, 170]
    return np.digitize(X, edges).astype(np.int8)         # 0 .. levels-1


def _feature_maps(group, levels=LEVELS):
    """Every lane's expected-intensity map, [S, 28, 28].

    For each pixel we read the unnormalized differential (Ga - Gb) on each non-background channel
    c = 1 .. levels-1 and weight it by the intensity level: feature = sum_c c * relu(Ga_c - Gb_c).
    A pixel the lane confidently expects dark is rewarded only on channel 0, so channels >= 1 stay
    near the low init and the feature is ~0 there (black background). Mid-gray fabric drives
    channel 1, bright highlights channel 2, so the map reads as the lane's learned garment
    intensity. Unnormalized, so quiet pixels stay quiet instead of blowing up into speckle."""
    full = [np.full(N_PIX, c, dtype=np.int8) for c in range(levels)]
    maps = np.zeros((group.channels, SIDE, SIDE), dtype=np.float32)
    for lane in range(group.channels):
        acc = np.zeros(N_PIX, dtype=np.float32)
        for c in range(1, levels):
            gab = group.core.read_gab(lane, full[c])
            d = np.array([ga - gb for ga, gb in gab], dtype=np.float32)
            acc += c * np.maximum(d, 0.0)
        maps[lane] = acc.reshape(SIDE, SIDE)
    return maps


def run(n=N, s=S, seed=SEED, n_frames=N_FRAMES, levels=LEVELS):
    """The full-width run: exclusion and recruitment both on, so every one of the S lanes is kept
    live and the whole bank forms a garment prototype (fashion_mnist_prune.py is the self-pruning
    counterpart that keeps only the well-formed subset)."""
    aats = load_fashion(n, seed=seed, levels=levels)
    core = Core(1, _pow2(levels), spaces_per_lane=N_PIX, num_lanes=s, model="byte", init=INIT,
                read_noise=0, seed=100 + seed)
    grp = BasisGroup(core, s, gather_abandon=GATHER_ABANDON, exclusion=True, recruitment=True)
    print(f"  levels={levels} S={s} N={n} full-width (recruitment on)", flush=True)

    # Log-spaced snapshot schedule: dense early (where the basis forms), sparse late. Plus a frame
    # at 0 (pure noise) and at the end.
    sched = set([0] + [int(x) for x in np.geomspace(20, n - 1, n_frames - 1)] + [n - 1])

    frames, samples = [], []
    t0 = time.time()
    for i in range(n):
        if i in sched:
            frames.append(_feature_maps(grp, levels))
            samples.append(i)
            np.savez(CACHE, frames=np.array(frames), samples=np.array(samples), n=n, s=s)
        grp.adapt(aats[i])
        if i == 200:
            eta = (time.time() - t0) / 201 * n
            print(f"  [200/{n}] ETA ~{eta/60:.1f} min for the full pass", flush=True)
        if i and i % 1000 == 0:
            print(f"  {i}/{n}  util={grp.codebook_utilization:.2f}  "
                  f"entropy={grp.winner_entropy:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    frames.append(_feature_maps(grp, levels))
    samples.append(n)
    np.savez(CACHE, frames=np.array(frames), samples=np.array(samples), n=n, s=s)
    print(f"done: {len(frames)} frames, {time.time()-t0:.0f}s -> {CACHE}", flush=True)


def render(fps=12, hold_seconds=3.0):
    from matplotlib.colors import LinearSegmentedColormap
    # Black -> warm intensity map: the feature is the lane's expected garment intensity (>= 0), so a
    # confidently-dark pixel is ~0 and sinks into the black background while the garment glows.
    cmap = LinearSegmentedColormap.from_list("kwarm",
                                             ["#000000", "#5a1a7a", "#e0562b", "#ffd24a", "#ffffff"])
    bg = "#000000"

    d = np.load(CACHE)
    frames, samples, s = d["frames"], d["samples"], int(d["s"])
    cols = int(np.ceil(np.sqrt(s)))
    rows = int(np.ceil(s / cols))
    # Two scales. The ANIMATION uses one GLOBAL scale for every cell, so early frames read honestly
    # — all cells uniformly faint at the start, brightening only as a lane wins. The final STILL uses
    # per-lane contrast, so each finished feature fills its cell and a low-contrast (mid-gray) lane
    # reads as sharp as a bright one — with a floor so a genuinely idle lane stays dark, not stretched
    # into noise. All scales come from the final frame.
    gmax = float(np.percentile(frames[-1], 99.9)) or 1.0
    global_vmax = float(np.percentile(frames[-1], 99.5)) or 1.0
    vmax_lane = [max(float(np.percentile(frames[-1][j], 99.5)), 0.35 * gmax) for j in range(s)]

    cell = 1.25
    strip = 0.32                                    # thin top strip for the counter, no title
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell, rows * cell + strip))
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor(bg)
    ims = []
    for j, ax in enumerate(axes.ravel()):
        ax.axis("off")
        ax.set_facecolor(bg)
        if j < s:
            ims.append(ax.imshow(frames[0][j], cmap=cmap, vmin=0, vmax=global_vmax,
                                  interpolation="nearest"))
        else:
            ims.append(None)
    # No title — just a small counter in the top-right.
    counter = fig.text(0.99, 0.985, "", ha="right", va="top", fontsize=8.5, color="#b8b8b8",
                       family="monospace")

    def update(f):
        for j, im in enumerate(ims):
            if im is not None:
                im.set_data(frames[f][j])
        counter.set_text(f"images seen = {int(samples[f]):,}")
        return [im for im in ims if im is not None]

    top = (rows * cell) / (rows * cell + strip)
    fig.subplots_adjust(left=0.01, right=0.99, top=top, bottom=0.01, wspace=0.06, hspace=0.06)
    # Hold on the final frame for a few seconds before the GIF loops.
    seq = list(range(len(frames))) + [len(frames) - 1] * int(fps * hold_seconds)
    anim = FuncAnimation(fig, update, frames=seq, blit=False)
    gif = OUT / "fashion-forming.gif"
    anim.save(gif, writer=PillowWriter(fps=fps))     # animation drawn at the global scale
    # Final still: switch each cell to its own per-lane contrast.
    update(len(frames) - 1)
    for j, im in enumerate(ims):
        if im is not None:
            im.set_clim(0, vmax_lane[j])
    png = OUT / "fashion-features.png"
    fig.savefig(png, dpi=150, facecolor=bg)
    plt.close(fig)
    print("wrote", gif, "and", png)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("run", "all"):
        run()
    if what in ("render", "all"):
        render()
