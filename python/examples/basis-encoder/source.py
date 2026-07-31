"""The synthetic AAT source, and the recovery metrics, for the basis-encoder lesson.

A generator is one AAT — a full assignment of one channel per input space. A sample picks a
generator, emits its channel in every space, then corrupts a fraction of the spaces by replacing
the channel with a random one. Because the true generator behind every sample is known, "did the
codebook recover the basis" is measured rather than eyeballed.

Two metrics separate the two halves of the stabilizer pair cleanly:

  coverage  fraction of true generators claimed by at least one lane   <- recruitment protects it
  purity    per lane, the fraction of its wins from a single generator <- exclusion protects it

Everything is read off one channel x generator win-count matrix (`confusion`), which is also the
figure the animation draws forming.
"""

import numpy as np


class AATSource:
    """A generator bank with a known basis.

    n_spaces    input spaces
    s_in        channels per input space
    k           generators (the true basis size)
    corruption  fraction of spaces re-drawn at random per sample (harder as it rises)
    overlap     fraction of spaces where a generator copies an earlier generator's channel — the
                case where a lazy codebook merges two generators onto one lane
    skew        frequency skew across generators. 0 is uniform; >0 makes some generators rarer
                (weights ~ 1/(rank+1)**skew), so rare patterns starve a lane that has no
                recruitment to pull it in.
    per_label   generators per label; the label is the generator index // per_label
    """

    def __init__(self, n_spaces=8, s_in=8, k=6, corruption=0.15, overlap=0.0,
                 skew=0.0, per_label=1, seed=0):
        self.n_spaces = n_spaces
        self.s_in = s_in
        self.k = k
        self.corruption = corruption
        self.overlap = overlap
        self.skew = skew
        self.per_label = per_label
        self.rng = np.random.default_rng(seed)
        self.generators = self._make_generators()
        w = 1.0 / (np.arange(1, k + 1) ** skew) if skew else np.ones(k)
        self.freq = w / w.sum()

    @property
    def space_sizes(self):
        return [self.s_in] * self.n_spaces

    @property
    def n_labels(self):
        return int(np.ceil(self.k / self.per_label))

    def label_of(self, generator):
        return generator // self.per_label

    def _make_generators(self):
        gens = []
        for g in range(self.k):
            aat = self.rng.integers(0, self.s_in, size=self.n_spaces)
            if g > 0 and self.overlap > 0:
                donor = gens[self.rng.integers(0, g)]
                shared = self.rng.random(self.n_spaces) < self.overlap
                aat = np.where(shared, donor, aat)
            gens.append(aat)
        return np.array(gens)

    def sample(self):
        """One (aat, generator, label) draw."""
        g = int(self.rng.choice(self.k, p=self.freq))
        aat = self.generators[g].copy()
        if self.corruption > 0:
            hit = self.rng.random(self.n_spaces) < self.corruption
            aat[hit] = self.rng.integers(0, self.s_in, size=int(hit.sum()))
        return tuple(int(c) for c in aat), g, self.label_of(g)

    def batch(self, n):
        """n samples as (aats, generators, labels)."""
        aats, gens, labels = [], [], []
        for _ in range(n):
            a, g, l = self.sample()
            aats.append(a)
            gens.append(g)
            labels.append(l)
        return aats, np.array(gens), np.array(labels)


def confusion(winners, generators, n_channels, k):
    """Channel x generator win counts — every metric below is read off this."""
    m = np.zeros((n_channels, k), dtype=int)
    for w, g in zip(winners, generators):
        if w is not None:
            m[w, g] += 1
    return m


def coverage(m):
    """Fraction of true generators claimed by at least one lane.

    A generator is claimed by the lane that wins on it most often. Two generators sharing one lane
    means only one is claimed, so a merged codebook reads as short coverage.
    """
    n_channels, k = m.shape
    claimed = set()
    for ch in range(n_channels):
        if m[ch].sum():
            claimed.add(int(np.argmax(m[ch])))
    return len(claimed) / k


def purity(m):
    """Per lane, the fraction of its wins from a single generator, averaged over lanes that won at
    all. A lane that answers for two generators equally scores 0.5."""
    used = [ch for ch in range(m.shape[0]) if m[ch].sum()]
    if not used:
        return 0.0
    return float(np.mean([m[ch].max() / m[ch].sum() for ch in used]))


def utilization(m):
    """Fraction of lanes that won at least once — coverage's cheap cousin, available when there is
    no ground-truth basis."""
    return float(np.mean([1.0 if m[ch].sum() else 0.0 for ch in range(m.shape[0])]))


def entropy(m):
    """Winner-histogram entropy in bits. Catches a codebook that uses every lane but leans hard on
    three."""
    counts = m.sum(axis=1).astype(float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())
