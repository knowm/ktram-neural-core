"""Shared setup for the basis-encoder lesson: the locked article config, the ablation arms, and a
train/eval helper that also snapshots the win-count matrix for the forming animation.

Run once, snapshot along the way, read the final metrics at the end — one training pass serves both
the static ablation figure and the animation.
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from source import AATSource, confusion, coverage, purity, utilization, entropy  # noqa: E402

from ktram_neural_core import BasisGroup, Core  # noqa: E402

# The article's main synthetic config: many generators (k), a lane bank wider than the basis (S),
# real corruption and overlap so the ablations bite. Chosen so both-on recovers a clean codebook,
# exclusion-off collapses onto a lane or two, and recruitment-off smears the codebook (purity
# falls while coverage holds).
CONFIG = dict(n_spaces=12, s_in=8, k=48, corruption=0.35, overlap=0.30, per_label=6)
S = 64                 # lanes per group (over-provisioned vs k=48)
GATHER_ABANDON = 32
EPOCHS = 4
N_TRAIN = 4000
N_EVAL = 3000

# (label, exclusion, recruitment, abandon_action) — the three-panel ablation.
#
# The recruitment-off arm resets the stalled cycle rather than leaving it to hang. With
# abandon_action="recruit" and recruitment off, nothing ever clears the won-buffer, so exclusion
# throttles almost every update and the arm measures a frozen run instead of the absence of
# recruitment. "reset" is the fair comparison: same exclusion, no force-feeding, cycle still turns.
ARMS = [
    ("both on", True, True, "recruit"),
    ("exclusion off", False, True, "recruit"),
    ("recruitment off", True, False, "reset"),
]


def _pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def train_and_eval(exclusion, recruitment, abandon_action="recruit", *, config=CONFIG, channels=S,
                   ga=GATHER_ABANDON, epochs=EPOCHS, n_train=N_TRAIN, n_eval=N_EVAL, seed=0,
                   n_snapshots=0, snap_eval=800):
    """Train one BasisGroup and evaluate it. Returns a dict of metrics, the final win-count matrix,
    and (if n_snapshots>0) a list of (fraction_trained, matrix) snapshots for the animation.

    Snapshots read a small held-out eval batch so the matrix reflects the current codebook state
    (not the cumulative history). The final metrics use the full n_eval batch.
    """
    src = AATSource(seed=seed, **config)
    k = src.k
    core = Core(1, _pow2(src.s_in), spaces_per_lane=src.n_spaces, num_lanes=channels,
                model="byte", init="low", read_noise=0, seed=100 + seed)
    grp = BasisGroup(core, channels, gather_abandon=ga,
                     exclusion=exclusion, recruitment=recruitment,
                     abandon_action=abandon_action)

    train_aats, _, _ = src.batch(n_train)
    eval_aats, eval_gens, _ = src.batch(n_eval)
    snap_aats, snap_gens = (eval_aats[:snap_eval], eval_gens[:snap_eval])

    total = epochs * n_train
    snap_at = set()
    if n_snapshots:
        snap_at = {int(round(i * (total - 1) / (n_snapshots - 1))) for i in range(n_snapshots)}

    snaps = []
    step = 0
    for _ in range(epochs):
        for a in train_aats:
            if step in snap_at:
                wr = [grp.read(x) for x in snap_aats]
                snaps.append((step / max(total - 1, 1),
                              confusion(wr, snap_gens, channels, k)))
            grp.adapt(a)
            step += 1
    if n_snapshots:  # final frame
        wr = [grp.read(x) for x in snap_aats]
        snaps.append((1.0, confusion(wr, snap_gens, channels, k)))

    winners = [grp.read(x) for x in eval_aats]
    m = confusion(winners, eval_gens, channels, k)
    metrics = dict(coverage=coverage(m), purity=purity(m), utilization=utilization(m),
                   entropy=entropy(m), n_instructions=grp.n_instructions,
                   n_throttled=grp.n_throttled, n_recruited=grp.n_recruited)
    return dict(metrics=metrics, matrix=m, snaps=snaps, k=k, channels=channels, win_counts=grp.win_counts)


def sort_matrix(m):
    """Order the lanes (rows) so the block-diagonal reads top-left to bottom-right: put lanes that
    won onto their plurality generator, sorted by that generator, first; idle lanes last."""
    n_channels, k = m.shape
    keyed = []
    for ch in range(n_channels):
        if m[ch].sum():
            keyed.append((int(np.argmax(m[ch])), -int(m[ch].sum()), ch))
        else:
            keyed.append((k + 1, 0, ch))  # idle lanes to the bottom
    order = [ch for _, _, ch in sorted(keyed)]
    return m[order], order
