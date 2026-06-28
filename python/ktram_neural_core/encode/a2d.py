"""A2DEncoder — numeric feature vector -> multi-space AAT (one space per dimension).

Port of the Java FloatA2DEncoder: per-dimension adaptive binary-tree binning. Each dimension
owns a depth-`bits` tree whose split thresholds start at a uniform binning of [init_min,
init_max] and migrate by an EMA toward the data on `encode_adapt` (equal-occupancy bins). A
dimension encodes to a single bin index in [0, 2**bits) -> one active synapse in its space, so
`encode([..]) -> (b0, b1, ...)` with one entry per dimension and `space_sizes == [2**bits]*dims`.

init_min / init_max may be a scalar (shared by every dimension) or a per-dimension sequence, so
features on different scales (Iris sepal length vs. petal width) get their own starting bins
without external normalization.
"""

from .base import AATEncoder


class _BinNode:
    """One split threshold `w` in the binary tree (faithful to the Java Node)."""

    __slots__ = ("w", "a", "b")

    def __init__(self, w, dw, depth):
        self.w = w
        if depth == 0:
            self.a = self.b = None
            return
        self.a = _BinNode(w + dw, dw / 2, depth - 1)   # >= w branch (bit set)
        self.b = _BinNode(w - dw, dw / 2, depth - 1)   # <  w branch

    def rescale(self, lo, hi):
        self.w = self.w * (hi - lo) + lo
        if self.a is not None:
            self.a.rescale(lo, hi)
            self.b.rescale(lo, hi)

    def encode(self, x, depth, path):
        ge = x >= self.w                      # >= matters: matches the Java boundary
        if ge:
            path |= (1 << (depth - 1))
        if depth == 1:
            return path
        return (self.a if ge else self.b).encode(x, depth - 1, path)

    def encode_adapt(self, x, depth, path, l):
        ge = x >= self.w
        if ge:
            path |= (1 << (depth - 1))
        self.w = (1 - l) * self.w + l * x     # adaptive average (the bin migration)
        if depth == 1:
            return path
        return (self.a if ge else self.b).encode_adapt(x, depth - 1, path, l)


def _per_dim(v, dims):
    """A scalar broadcasts to every dimension; a sequence must match `dims`."""
    try:
        seq = list(v)
    except TypeError:
        return [v] * dims
    if len(seq) != dims:
        raise ValueError(f"expected {dims} values, got {len(seq)}")
    return seq


class A2DEncoder(AATEncoder):
    def __init__(self, dims, bits=5, init_min=0.0, init_max=1.0, l=0.01):
        if bits < 1:
            raise ValueError("bits must be >= 1")
        self.dims = dims
        self.bits = bits
        self.l = l
        lo = _per_dim(init_min, dims)
        hi = _per_dim(init_max, dims)
        self._roots = [_BinNode(0.5, 0.25, bits) for _ in range(dims)]
        for root, a, b in zip(self._roots, lo, hi):
            root.rescale(a, b)

    def encode(self, value):
        return tuple(
            self._roots[i].encode(float(value[i]), self.bits, 0) for i in range(self.dims)
        )

    def encode_adapt(self, value):
        return tuple(
            self._roots[i].encode_adapt(float(value[i]), self.bits, 0, self.l)
            for i in range(self.dims)
        )

    @property
    def space_sizes(self):
        return [1 << self.bits] * self.dims
