"""Build examples/iris-classifier/iris_classifier.ipynb from source cells.

Run:  python examples/iris-classifier/_build_notebook.py
Then execute it (renders the inline plots):
      jupyter nbconvert --to notebook --execute --inplace examples/iris-classifier/iris_classifier.ipynb

Self-contained companion to "Chapter 6: Classification and Thermal Sampling on kT-RAM Neural
Lanes". Mirrors the article section by section: the Iris data, fixed and adaptive A2D bins, the
composed bias, the three-case supervised rule driven at L0, the RankCut L1 recoder, the fair
benchmark against batch linear solvers on the identical encoding, reads at temperature, and the
soft-feedback generator sampled chained against synchronous. Every helper is defined inline so
the notebook reads top-to-bottom with no hidden imports, and the seeds are there to be changed.
"""

import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md(r"""# The Iris classifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/knowm/ktram-neural-core/blob/main/python/examples/iris-classifier/iris_classifier.ipynb)

This notebook runs alongside *Chapter 6: Classification and Thermal Sampling on kT-RAM Neural
Lanes*. Three **neural lanes**, one per iris species, learn to name flowers from a supervised
routine written in kT-RAM instructions. Read each lane with `FF`. Drive the right one up with
`RH`, drive the wrong-but-fired ones down with `RL`, and leave the rest alone with `RF`.

We build the AAT encoding first, with fixed bins, adaptive bins, and a bias. Then we train at
L0 by hand, wrap the same routine as the `RankCut` L1 recoder, and measure it against batch
linear solvers on the identical encoding.

The last third of the notebook turns the read noise back on. Verdicts become samples. Fresh
banks then learn the opposite mapping, label in and pattern out, and sample a tilted
two-dimensional cloud through a two-step chain. Reading the banks synchronously breaks the
correlation, which shows what the chain carried.

Change the seeds and the knobs as you go: `bits`, `model`, `init`, the epochs, the
temperatures, and the feedback rule."""))

cells.append(md("""## Setup

On Colab this installs the package from GitHub. Locally it does nothing if the package is
already installed."""))

cells.append(code('''try:
    import ktram_neural_core  # noqa: F401
except ModuleNotFoundError:
    %pip install -q "git+https://github.com/knowm/ktram-neural-core.git#subdirectory=python" matplotlib scikit-learn'''))

cells.append(code('''import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

from ktram_neural_core import (Core, A2DEncoder, ConstantEncoder, compose,
                               LinearClassifier, RankCut, rank_cut, Winner)

SPECIES_COLORS = ["tab:blue", "tab:green", "tab:red"]
GRID = "0.92"

data = load_iris()
X, y = data.data, data.target
names = list(data.target_names)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
print(f"{len(X_tr)} training flowers, {len(X_te)} test flowers")
print("features:", data.feature_names)
print("species: ", names)'''))

cells.append(md("""## The data

Iris is a small table the statistician Ronald Fisher published in 1936: a hundred and fifty
flowers, fifty from each of three species, and four measurements per flower in centimeters.
The species is the label. Setosa (blue) sits in its own clump in every view. Versicolor (green)
and virginica (red) overlap, worst in the sepals. Every classifier puts its errors in that
overlap, ours included."""))

cells.append(code('''fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, (i, j) in zip(axes, [(2, 3), (0, 1)]):
    for s in range(3):
        m = y == s
        ax.scatter(X[m, i], X[m, j], s=14, color=SPECIES_COLORS[s], alpha=0.75, label=names[s])
    ax.set_xlabel(data.feature_names[i]); ax.set_ylabel(data.feature_names[j])
    ax.grid(True, color=GRID)
axes[0].set_title('petals'); axes[1].set_title('sepals'); axes[0].legend()
plt.tight_layout(); plt.show()'''))

cells.append(md("""## Encoding the measurements as AATs

A lane does not read numbers. It reads AATs, so every flower has to become a tuple of channel
indices before a lane sees it. The `A2DEncoder` does the analog-to-digital move. Each feature
gets its own space of bins, and a value's channel is whichever bin it lands in. `bits=3` cuts
each feature's range into eight bins, so a flower encodes to a four-entry AAT."""))

cells.append(code('''fixed = A2DEncoder(dims=4, bits=3, init_min=X_tr.min(0), init_max=X_tr.max(0))
flower = X_tr[0]
print("flower:     ", flower, "->", names[y_tr[0]])
print("AAT:        ", fixed.encode(flower))
print("space sizes:", fixed.space_sizes)'''))

cells.append(md("""## Fixed bins against adaptive bins

Left unadapted, the encoder keeps an even slicing of each feature's range. Call `encode_adapt`
on each training example instead, and every value tugs the nearest bin edges a little in its
direction. The edges migrate until every bin holds about the same number of points, so the
resolution goes where the data is.

The encoder gives no direct view of its edges. We read them from the outside: sweep a value
through one dimension and record where the reported bin index changes."""))

cells.append(code('''def bin_edges(encoder, dim, lo, hi, n=4000):
    """Sweep one dimension and record where the bin index changes. A2D dimensions are
    independent, so the other entries sit at zero."""
    edges, prev = [], None
    for v in np.linspace(lo, hi, n):
        q = np.zeros(4); q[dim] = v
        b = encoder.encode(q)[dim]
        if prev is not None and b != prev:
            edges.append(v)
        prev = b
    return edges

adaptive = A2DEncoder(dims=4, bits=3, init_min=X_tr.min(0), init_max=X_tr.max(0), l=0.01)
rng = np.random.default_rng(0)
for _ in range(5):                       # five passes of bin migration, then we stop
    for i in rng.permutation(len(X_tr)):
        adaptive.encode_adapt(X_tr[i])

PL, PW = 2, 3                            # petal length, petal width
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
for ax, enc, title in [(axes[0], fixed, 'fixed bins — even slices'),
                       (axes[1], adaptive, 'adapted bins — equal occupancy')]:
    for s in range(3):
        m = y_tr == s
        ax.scatter(X_tr[m, PL], X_tr[m, PW], s=12, color=SPECIES_COLORS[s], alpha=0.7)
    for e in bin_edges(enc, PL, X_tr[:, PL].min(), X_tr[:, PL].max()):
        ax.axvline(e, color='0.6', lw=0.7)
    for e in bin_edges(enc, PW, X_tr[:, PW].min(), X_tr[:, PW].max()):
        ax.axhline(e, color='0.6', lw=0.7)
    ax.set_title(title); ax.set_xlabel('petal length (cm)')
axes[0].set_ylabel('petal width (cm)')
plt.tight_layout(); plt.show()'''))

cells.append(md("""## Adding the bias

A bias is an always-on input. The `ConstantEncoder` ignores the value and always lights the
same channel, and `compose` lays its AAT after the A2D's. That gives five spaces: four
adaptive bins, then the bias. The balanced A2D encoding does not need the bias on this data,
but one extra synapse costs little and shows the mechanism. The lanes see this composed
encoder for the rest of the notebook."""))

cells.append(code('''encoder = compose(
    A2DEncoder(dims=4, bits=3, init_min=X_tr.min(0), init_max=X_tr.max(0), l=0.01),
    ConstantEncoder(),                    # one always-on synapse — the bias
)
print("AAT:        ", encoder.encode(flower))
print("space sizes:", encoder.space_sizes)'''))

cells.append(md("""## The supervised rule, driven at L0

Provision a core with three lanes, one per species, and five spaces each. The byte model makes
every synapse an 8-bit memristor. Read noise is off for training, so a fixed seed reproduces
exactly. Real hardware does not offer that switch.

Phase one adapts the encoder's bins with the classifier off. Then we freeze the encoder and
never call `encode_adapt` again, because a classifier cannot learn against an encoding that
slides out from under it. Phase two is the supervised rule, three cases of an `if`, written in
raw kT-RAM instructions:

- the lane that owns this flower's species gets `RH` — driven up toward *yes*
- a wrong lane that fired anyway gets `RL` — a false positive, driven down
- a wrong lane that correctly stayed below zero gets `RF` — left alone

Inference reads each lane with `FFLV`, the sub-threshold read that reports the weight without
changing it, and takes the winner."""))

cells.append(code('''LABELS = [0, 1, 2]
core = Core(1, 8, spaces_per_lane=len(encoder.space_sizes), num_lanes=len(LABELS),
            model='byte', init='low', read_noise=0.0, seed=0)

rng = np.random.default_rng(0)
for _ in range(5):                                 # phase one: adapt the bins, classifier off
    for i in rng.permutation(len(X_tr)):
        encoder.encode_adapt(X_tr[i])
# frozen from here on: only encode() gets called again

def predict(flower):
    aat = encoder.encode(flower)
    scores = [core.evaluate(aat, "FFLV", lane) for lane in LABELS]
    return int(np.argmax(scores))                  # the loudest lane names the class

def accuracy(Xs, ys):
    return float(np.mean([predict(f) == t for f, t in zip(Xs, ys)]))

for epoch in range(5):                             # phase two: the three-case rule, online
    for i in rng.permutation(len(X_tr)):
        aat = encoder.encode(X_tr[i])
        for lane in LABELS:
            yv = core.evaluate(aat, "FF", lane)
            if lane == y_tr[i]:
                core.evaluate(aat, "RH", lane)     # this IS the class — drive the answer up
            elif yv > 0:
                core.evaluate(aat, "RL", lane)     # wrong lane caught saying yes — drive it down
            else:
                core.evaluate(aat, "RF", lane)     # wrong lane correctly saying no — leave it be
    print(f"epoch {epoch + 1}:  train {accuracy(X_tr, y_tr):.3f}   test {accuracy(X_te, y_te):.3f}")'''))

cells.append(md("""## The same routine as an L1 recoder

The loop above reaches into the lanes and handles their analog outputs as floats. That works
for research, but we would not put it in hardware.

The `RankCut` recoder wraps the same routine behind an AAT-level interface, with `adapt` to
teach, `read` to answer, and the analog kept inside. Point it at the core we just trained by
hand. Its reads agree with our argmax, because underneath it runs the same instructions.

The readout policy is the rank-cut. Keep the lanes above a threshold `Vt`, sorted strongest
first, and cut after at most `N`. The winner is its smallest setting. In silicon a swept
reference voltage does this sort: the lanes surface in rank order as the waterline falls. That
circuit gets its own chapter."""))

cells.append(code('''rec = RankCut(core, labels=LABELS)            # wraps the SAME core we trained above

f = X_te[0]
aat = encoder.encode(f)
scores = [core.evaluate(aat, "FFLV", lane) for lane in LABELS]
print("lane scores:", [f"{s:+.3f}" for s in scores])
print("rec.read -> ", rec.read(aat), f"   true label: {y_te[0]} ({names[y_te[0]]})")
print()
print("the same score vector through the readout policy:")
print("  winner             rank_cut(y, Vt=-1, N=1) ->", rank_cut(scores, Vt=-1, N=1))
print("  all above zero     rank_cut(y, Vt=0)       ->", rank_cut(scores, Vt=0.0))
print("  top two above zero rank_cut(y, Vt=0, N=2)  ->", rank_cut(scores, Vt=0.0, N=2))

out = rec.adapt(encoder.encode(X_tr[0]), {int(y_tr[0])})   # one supervised step, recoded read
print()
print("rec.adapt ->", out, "  (one FF read + RH/RL/RF pass, then the recoded read)")'''))

cells.append(md("""## Does it work?

Accuracy on its own says little. A fair test freezes one encoder and runs three linear
classifiers on the identical AAT encoding: our lane, scikit-learn's `LogisticRegression`, and a
`LinearSVC`. The two reference models solve for their weights in one batch over the whole
dataset. The lane learns online, one example at a time, with local instructions. A fourth bar
runs logistic regression on the raw measurements, which shows what the encoding costs or gains.

The reference models read the same binary vector the lane reads, with a one at each active
(space, channel) position. The library's `LinearClassifier` packages the three-case loop we
wrote by hand, so we use it here to sweep twenty train/test splits. Give this cell a minute."""))

cells.append(code('''def aat_to_onehot(aat, sizes):
    """The binary (space, channel) vector the lane reads: one 1 per active space."""
    vec = np.zeros(int(np.sum(sizes))); off = 0
    for entry, size in zip(aat, sizes):
        if entry is not None:
            vec[off + entry] = 1.0
        off += size
    return vec

def run_once(seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    enc = compose(A2DEncoder(dims=4, bits=3, init_min=Xtr.min(0), init_max=Xtr.max(0), l=0.01),
                  ConstantEncoder())
    clf = LinearClassifier(enc, labels=LABELS, model='byte', init='low',
                           read_noise=0, recoder=Winner(), seed=seed)
    clf.fit(Xtr, ytr, epochs=5, encoder_epochs=5, seed=seed)   # adapt + freeze, then train
    enc_tr = np.array([aat_to_onehot(enc.encode(v), enc.space_sizes) for v in Xtr])
    enc_te = np.array([aat_to_onehot(enc.encode(v), enc.space_sizes) for v in Xte])
    return yte, {
        'kT-RAM lane (AAT)': np.array([clf.predict(v) for v in Xte]),
        'LogReg (AAT)': LogisticRegression(max_iter=5000).fit(enc_tr, ytr).predict(enc_te),
        'LinearSVC (AAT)': LinearSVC(max_iter=20000).fit(enc_tr, ytr).predict(enc_te),
        'LogReg (raw)': LogisticRegression(max_iter=5000).fit(Xtr, ytr).predict(Xte),
    }

ORDER = ['LogReg (raw)', 'LogReg (AAT)', 'LinearSVC (AAT)', 'kT-RAM lane (AAT)']
accs = {m: [] for m in ORDER}
for seed in range(20):
    yte, preds = run_once(seed)
    for m in ORDER:
        accs[m].append(accuracy_score(yte, preds[m]))

for m in ORDER:
    a = np.array(accs[m])
    print(f"{m:<20} {a.mean():.3f} ± {a.std():.3f}   ({a.min():.3f} … {a.max():.3f})")

fig, ax = plt.subplots(figsize=(7.5, 4.2))
means = [np.mean(accs[m]) for m in ORDER]
stds = [np.std(accs[m]) for m in ORDER]
ax.bar(range(4), means, yerr=stds, capsize=4,
       color=['0.75', 'tab:blue', 'tab:orange', 'tab:green'])
ax.set_xticks(range(4)); ax.set_xticklabels(ORDER, fontsize=9)
ax.set_ylim(0.8, 1.0); ax.set_ylabel('test accuracy (20 seeds)')
ax.set_title('An online instruction-level rule against batch solvers, same encoding')
ax.grid(True, axis='y', color=GRID)
plt.tight_layout(); plt.show()'''))

cells.append(md("""## Where the misses land

The lane and logistic regression miss the same flowers, the versicolor/virginica overlap that no
straight line splits. That ceiling comes from the data, not from the learning rule. A linear
classifier that reaches it does everything a linear classifier can do here."""))

cells.append(code('''yte, preds = run_once(0)
for m in ['kT-RAM lane (AAT)', 'LogReg (AAT)']:
    print(f"{m}  (rows = true, cols = predicted; order {names})")
    print(confusion_matrix(yte, preds[m]), "\\n")'''))

cells.append(md("""## Reading at temperature

Everything above read cold. Real hardware carries the kT-bit's read noise on every read, and the
read voltage is the dial for it. The `noise` argument slides the read voltage down from the
standard low-voltage read at `0`, which is nearly clean, toward zero volts at `1`, where the
hiss swallows the signal.

Turn the master gain back on and read the same trained lanes hot, many times. The thermal part
of the hiss grows as $1/V$, so most of the dial changes little and the action piles up near the
bottom.

At `noise=0.98` the flower on the versicolor/virginica overlap becomes a weighted draw between
its two contending lanes, while the setosa stays certain. The classifier now returns samples
rather than verdicts, and only where the data left the question open. Push to `noise=0.995` and
the hiss swallows everything, the setosa included. The useful sampler sits between those two
settings."""))

cells.append(code('''core.set_read_noise(0.02)                       # the hiss back on (0 disabled it for training)

def top_gap(f):
    a = encoder.encode(f)
    s = sorted((core.evaluate(a, "FFLV", lane) for lane in LABELS), reverse=True)
    return s[0] - s[1]

contested = min(range(len(X_te)), key=lambda i: top_gap(X_te[i]))
easy = next(i for i in range(len(X_te)) if y_te[i] == 0)          # any setosa

def winner_fractions(f, T, n=400):
    aat = encoder.encode(f)
    wins = np.zeros(3)
    for _ in range(n):
        s = [core.evaluate(aat, "FFLV", lane, noise=T) for lane in LABELS]
        wins[int(np.argmax(s))] += 1
    return wins / n

temps = [0.0, 0.9, 0.98, 0.995]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, idx, title in [(axes[0], easy, f'an easy flower (true: {names[y_te[easy]]})'),
                       (axes[1], contested, f'a boundary flower (true: {names[y_te[contested]]})')]:
    w = np.array([winner_fractions(X_te[idx], T) for T in temps])
    xpos = np.arange(len(temps))
    for s in range(3):
        ax.bar(xpos + (s - 1) * 0.25, w[:, s], 0.25, color=SPECIES_COLORS[s], label=names[s])
    ax.set_xticks(xpos); ax.set_xticklabels([f'noise={t}' for t in temps])
    ax.set_title(title); ax.grid(True, axis='y', color=GRID)
axes[0].set_ylabel('fraction of 400 reads won')
axes[1].legend()
plt.tight_layout(); plt.show()'''))

cells.append(md("""## The soft generator

A lane bank runs one direction only: an AAT in, an AAT out. The label is a plain tuple
coordinate, so it can sit on either side. The classifier learned the pattern-to-label mapping,
and that trained bank holds only that one direction. Hand it a label and no pattern comes back.

A generator is a new bank of lanes taught the opposite mapping, separately, with the same
supervised routine. The label enters as an input coordinate and the output lanes stand for
pattern bins. Clamp a label, read a pattern.

The data is a flat two-dimensional mixture: a tilted cloud with label 0 and a round cloud with
label 1, binned on both axes by the fixed-bin A2D idea, twenty bins each.

Swap the roles and give the *bins* the lanes. An x bank holds one lane per x bin and reads only
the label. A y bank holds one lane per y bin and reads the label together with the x bin.

Teach both banks **soft**, which means dropping the `RL` case so no lane is ever punished for
firing. Teach with slightly hot reads too, so the hiss dithers the byte-quantized updates. Where
the data spreads across many bins, soft feedback keeps every supported answer above the
waterline instead of leaving one loud winner.

The two banks do not face the same job. The x bank reads one thing, the label, so it has two
contexts to learn and settles quickly. The y bank reads the label *and* the committed x bin, so
it has one conditional distribution to learn for every x bin. That is twenty times the work from
the same stream of examples.

Starve the y bank and it falls back on the y marginal, which is the tilt-free answer the
synchronous draw gives. So this cell walks 2400 examples per class rather than a few hundred.
The conditional is expensive, and it carries the whole correlation.

One tuning note. These cores get a louder master noise gain, `read_noise=0.2` instead of the
device default `0.02`. The hiss then matches the gaps between lane scores in the middle of the
voltage dial rather than only in its last two percent. The mechanism does not change. The dial
is still the read voltage, turned up here for the demonstration."""))

cells.append(code('''NB = 20                       # bins per axis (and lanes per bank)
TEACH_T = 0.25                # slightly hot teach reads — the dither
rng = np.random.default_rng(0)

def make_data(n_per):
    pts, labs = [], []
    th = np.deg2rad(40); c, s = np.cos(th), np.sin(th)
    for _ in range(n_per):
        dx, dy = rng.normal() * 0.16, rng.normal() * 0.045      # the tilted cloud, label 0
        pts.append((0.45 + dx * c - dy * s, 0.48 + dx * s + dy * c)); labs.append(0)
        dx, dy = rng.normal() * 0.05, rng.normal() * 0.05       # the round cloud, label 1
        pts.append((0.78 + dx, 0.24 + dy)); labs.append(1)
    return np.clip(np.array(pts), 0.001, 0.999), np.array(labs)

train_pts, train_labs = make_data(2400)
bin_of = lambda v: min(NB - 1, int(v * NB))

# read noise ON and louder than the device default — the dither and the sampling both use it
GAIN = 0.2
gx = Core(1, 2, spaces_per_lane=1, num_lanes=NB, model='byte', init='medium', seed=1, read_noise=GAIN)
gy = Core(1, NB, spaces_per_lane=2, num_lanes=NB, model='byte', init='medium', seed=2, read_noise=GAIN)

def teach_soft(bank, aat, target):
    """FF, then RH on the target and RF on everyone else. No RL — the one-line difference."""
    for lane in range(NB):
        bank.evaluate(aat, "FF", lane, noise=TEACH_T)
        if lane == target:
            bank.evaluate(aat, "RH", lane)
        else:
            bank.evaluate(aat, "RF", lane, noise=TEACH_T)

for p, lab in zip(train_pts, train_labs):
    xb, yb = bin_of(p[0]), bin_of(p[1])
    teach_soft(gx, (lab,), xb)            # x bank reads the label alone
    teach_soft(gy, (lab, xb), yb)         # y bank reads the label AND the x bin
print("taught", len(train_pts), "examples soft")'''))

cells.append(md(r"""## Chained against synchronous sampling

Each sample is a two-step chain. Draw x from what the label makes likely, then draw y from what
the label *and that x* make likely. That is the chain rule of probability,
$p(x, y) = p(x)\,p(y \mid x)$, running as two lane reads. The committed x carries the dependence
from the first read into the second.

The **synchronous** draw skips the commit. Both banks read at the same instant with the y bank's
x space silent, so the y bank draws only from what the label supports. Both modes keep the same
marginals. Only the chained mode keeps the correlation, and the tilted cloud makes that visible:
chained samples come out tilted, and synchronous samples wash out into an axis-aligned blur.

The correlation coefficient of the label-0 samples says the same thing as the picture. The
chained value lands near the data's own and the synchronous value sits at zero. The correlation
rides on the committed x, and cutting the commit erases it.

The chained streak is still narrower than the data's and piles up toward one end. The
one-winner rank-cut causes that. Taking a single argmax per read is a greedier draw than the
odds the weights hold, so the sampler over-visits its strongest bins. Widening the cut with `N`
above 1, or reading the bank as a distribution rather than a winner, recovers that slack."""))

cells.append(code('''T = 0.5                                  # sampling temperature

def draw(lab, chained):
    ys = [gx.evaluate((lab,), "FFLV", lane, noise=T) for lane in range(NB)]
    pick = rank_cut(ys, Vt=0.0, N=1)
    if not pick:
        return None                       # a dry read — nothing surfaced
    xb = pick[0]
    aat = (lab, xb) if chained else (lab, None)   # synchronous: the x space stays silent
    ys = [gy.evaluate(aat, "FFLV", lane, noise=T) for lane in range(NB)]
    pick = rank_cut(ys, Vt=0.0, N=1)
    return (xb, pick[0]) if pick else None

def sample_map(chained, n=6000):
    H = np.zeros((NB, NB)); tilted = []; dry = 0
    for i in range(n):
        s = draw(i % 2, chained)
        if s is None:
            dry += 1
        else:
            H[s[1], s[0]] += 1
            if i % 2 == 0:
                tilted.append(s)
    return H, np.array(tilted), dry / n

Hc, tc, dry_c = sample_map(True)
Hs, ts, dry_s = sample_map(False)
Hd = np.histogram2d(train_pts[:, 1], train_pts[:, 0], bins=NB, range=[[0, 1], [0, 1]])[0]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
for ax, H, title in [(axes[0], Hd, 'the data'),
                     (axes[1], Hc, 'chained — commit x, then y | x'),
                     (axes[2], Hs, 'synchronous — label only')]:
    ax.imshow(H, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()

bins0 = np.minimum(NB - 1, (train_pts[train_labs == 0] * NB).astype(int))
r_d = np.corrcoef(bins0[:, 0], bins0[:, 1])[0, 1]
r_c = np.corrcoef(tc[:, 0], tc[:, 1])[0, 1]
r_s = np.corrcoef(ts[:, 0], ts[:, 1])[0, 1]
print(f"tilted-cloud correlation:  data r = {r_d:+.2f}   chained r = {r_c:+.2f}   synchronous r = {r_s:+.2f}")
print(f"dry reads:  chained {dry_c:.1%}   synchronous {dry_s:.1%}")'''))

cells.append(md("""## Hard feedback kills the generator

Put the `RL` case back and teach the x bank hard. In a generator bank the answer spreads across
many bins, so nearly every lane is wrong on nearly every example. The punishment then drives the
whole bank down toward the waterline, and reads start coming back empty. Nothing surfaces above
the rank-cut threshold at all, which is a dry read.

The soft bank never does this. The hard bank does it on a steady share of its reads, and every
dry read is a sample the generator failed to produce. Punish a bank for every wrong guess and it
learns to stop guessing."""))

cells.append(code('''gxh = Core(1, 2, spaces_per_lane=1, num_lanes=NB, model='byte', init='medium', seed=3, read_noise=GAIN)

def teach_hard(bank, aat, target):
    for lane in range(NB):
        yv = bank.evaluate(aat, "FF", lane, noise=TEACH_T)
        if lane == target:
            bank.evaluate(aat, "RH", lane)
        elif yv > 0:
            bank.evaluate(aat, "RL", lane)         # the one line that changes everything
        else:
            bank.evaluate(aat, "RF", lane, noise=TEACH_T)

for p, lab in zip(train_pts, train_labs):
    teach_hard(gxh, (lab,), bin_of(p[0]))

def dry_fraction(bank, n=600):
    dry = 0
    for i in range(n):
        ys = [bank.evaluate((i % 2,), "FFLV", lane, noise=T) for lane in range(NB)]
        if not rank_cut(ys, Vt=0.0, N=1):
            dry += 1
    return dry / n

print(f"x-bank dry reads at T={T}:   soft {dry_fraction(gx):.1%}   hard {dry_fraction(gxh):.1%}")'''))

cells.append(md("""## What to change

Three lanes and three cases of an `if` matched batch linear solvers on the identical encoding.
Fresh banks then learned the opposite mapping with the same routine, and they sampled patterns
once the feedback softened and the reads warmed up. Knobs worth turning:

- `bits` — coarser or finer bins. Watch the benchmark move.
- `model` — swap `byte` for `float` or `mss` and rerun the training cell.
- the temperatures — `TEACH_T` for the dither, `T` and the `noise` levels for the draws.
- the mixture in `make_data` — steeper tilts need the chain more.
- `Vt` and `N` on the rank-cut — two numbers that cover the whole readout family.

The chapter's browser widget runs the same byte-lane arithmetic live, with a switch for the
chained and synchronous draws. One straight cut per class is where supervised learning starts.
The next chapter stacks lanes on lanes."""))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

out = pathlib.Path(__file__).resolve().parent / "iris_classifier.ipynb"
nbf.write(nb, str(out))
print(f"wrote {out}  ({len(cells)} cells)")
