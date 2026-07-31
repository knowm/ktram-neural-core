"""Fashion-MNIST basis that prunes itself — the self-pruning counterpart to fashion_mnist.py.

fashion_mnist.py runs the canonical mode (recruitment on the whole way): every one of the S lanes is
kept live, so a wide bank spreads itself thin and half the lanes stay diffuse. Here we run the SAME
bank two-phase — FORM then SHARPEN:

    phase 1 (recruitment on):  the bank forms with idle lanes force-fed, so the full basis develops
                               first and nothing is starved before it can specialize
    phase 2 (recruitment off, cycle reset):  halfway through, recruitment is turned off; lanes that
                               keep winning sharpen, lanes that stop winning fade back to init
    the survivors (win count above a fraction of the top lane's) are the basis

The pruning is done by the competition once recruitment is withdrawn, not by a hand-picked win-count
cut. This replaces the earlier arbitrary "train, then drop the low-win lanes at image 30k" approach —
same form-then-prune shape, but the dynamics do the pruning.

    python fashion_mnist_prune.py run     # stream one pass -> figures/fashion-prune.npz
    python fashion_mnist_prune.py render  # -> fashion-prune-forming.gif + fashion-prune-features.png
    python fashion_mnist_prune.py         # run then render

Outputs use a `-prune` suffix so they never collide with the canonical fashion_mnist.py run. The
shipped module is used unmodified; the self-pruning is just its recruitment=False, reset mode.
"""

import sys
import time
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ktram_neural_core import BasisGroup, Core  # noqa: E402
from fashion_mnist import (  # noqa: E402
    load_fashion, _feature_maps, _pow2, N_PIX, LEVELS, INIT, GATHER_ABANDON, OUT,
)

# Config mirrors the canonical run (same bank, same data) — only the mode differs.
N = 60000
S = 16
SEED = 0
N_FRAMES = 60
CACHE = OUT / "fashion-prune.npz"


def _read_stats(grp, aats, n_sample=20000):
    """Final per-lane win count from a read-only FFLV pass — the hardware-native survivor readout."""
    m = min(n_sample, len(aats))
    win = np.zeros(grp.channels)
    for i in range(m):
        win[grp.read(aats[i])] += 1
    return win


def run(n=N, s=S, seed=SEED, n_frames=N_FRAMES, levels=LEVELS, switch_frac=0.5):
    aats = load_fashion(n, seed=seed, levels=levels)
    core = Core(1, _pow2(levels), spaces_per_lane=N_PIX, num_lanes=s, model="byte", init=INIT,
                read_noise=0, seed=100 + seed)
    # Start in formation mode (recruitment on); switch to self-pruning partway through.
    grp = BasisGroup(core, s, gather_abandon=GATHER_ABANDON, exclusion=True,
                     recruitment=True, abandon_action="recruit")
    switch_at = int(switch_frac * n) if switch_frac is not None else None
    print(f"  levels={levels} S={s} N={n} form-then-sharpen switch_at={switch_at}", flush=True)

    sched = set([0] + [int(x) for x in np.geomspace(20, n - 1, n_frames - 1)] + [n - 1])
    frames, samples = [], []
    t0 = time.time()
    for i in range(n):
        if switch_at is not None and i == switch_at:
            grp.recruitment = False
            grp.abandon_action = "reset"
            print(f"  SWITCH at {i}: recruitment off, reset (sharpen / self-prune)", flush=True)
        if i in sched:
            frames.append(_feature_maps(grp, levels))
            samples.append(i)
            np.savez(CACHE, frames=np.array(frames), samples=np.array(samples), n=n, s=s,
                     switch_at=(switch_at if switch_at is not None else -1))
        grp.adapt(aats[i])
        if i == 200:
            print(f"  [200/{n}] ETA ~{(time.time()-t0)/201*n/60:.1f} min", flush=True)
        if i and i % 5000 == 0:
            print(f"  {i}/{n}  entropy={grp.winner_entropy:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    frames.append(_feature_maps(grp, levels))
    samples.append(n)
    win = _read_stats(grp, aats)
    np.savez(CACHE, frames=np.array(frames), samples=np.array(samples), n=n, s=s,
             switch_at=(switch_at if switch_at is not None else -1), win=win)
    alive = int((win > 0.05 * (win.max() or 1)).sum())
    print(f"done: {len(frames)} frames, {alive}/{s} survive, {time.time()-t0:.0f}s -> {CACHE}",
          flush=True)


def render(fps=12, hold_seconds=3.0):
    from image_basis import render_forming
    render_forming(CACHE, OUT / "fashion-prune-forming.gif", OUT / "fashion-prune-features.png",
                   cell=1.25, fps=fps, hold_seconds=hold_seconds, counter_label="images seen")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("run", "all"):
        run()
    if what in ("render", "all"):
        render()
