"""Benchmark the lane read as what it is — an embedding bag — against its dense twin.

Three sections, each emitting a markdown block into lane-read-bench-data.md (the curated
report, lane-read-bench.md, reads these numbers):

  race         four formulations of the read on identical inputs, each gated bit-exact
               against `_lane.read_sums` before its time counts, plus dense controls at
               the matched shape [B, 2k] @ [2k, L] (the op the lane replaces, not a read)
  conditional  RankCut fire rates through the digits lane bridge, and read cost as a
               function of active-space count (the offsets-form bag drops NONE entirely;
               a dense layer cannot skip anything)
  crossover    streaming (B calls of adapt at B=1, the bit-exact serial anchor) vs one
               batched adapt call, per local backend; quality at B is qualified by the
               08a tier-3 bands, not by this script

Run (from python/):  python benchmarks/bench_lane_read.py [--section race|conditional|crossover]
The full race takes ~1 h on a 10-core CPU (the old masked-gather formulation is the slow
cell — its slowness is the finding). --quick trims batch sizes for a smoke run.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F

from ktram_neural_core.torch import BasisEncoder, Classifier, _lane
from ktram_neural_core.torch.pack import quantize

OUT = pathlib.Path(__file__).parent / "lane-read-bench-data.md"
CHUNK_BYTES = 256e6            # cap on the incumbent's [L, chunk, K] int32 intermediate
BATCHES = [1, 64, 4096, 65536]
GATE_B = 257                   # gate batch: one all-NONE row, one full row, rest ~10% NONE

SHAPES = [
    # label, L, K, S, G (None = flat/shared-input read)
    ("decoder [1568, k16, S64]", 1568, 16, 64, None),
    ("encoder per-group [16x64, k49, S2]", 64, 49, 2, 16),
    ("xfmr [1024, k512, S64]", 1024, 512, 64, None),
    ("xfmr [1024, k512, S16]", 1024, 512, 16, None),
]
SWEEP_SHAPES = [
    ("decoder [1568, k16, S64]", 1568, 16, 64, [1, 2, 4, 8, 12, 16]),
    ("xfmr [1024, k512, S64]", 1024, 512, 64, [32, 64, 128, 256, 384, 512]),
]
CROSS_BS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


# ---------------------------------------------------------------------------
# Shared harness.
# ---------------------------------------------------------------------------

def bench(fn):
    """Median seconds per call, reps adapted to the call's cost."""
    fn()                                       # warmup
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter() - t0
    if t1 > 3.0:
        reps, inner = 2, 1
    elif t1 > 0.3:
        reps, inner = 5, 1
    else:
        reps, inner = 10, max(1, int(0.03 / max(t1, 1e-7)))
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        times.append((time.perf_counter() - t0) / inner)
    times.sort()
    return times[len(times) // 2]


def make_banks(L, K, S, G, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (L, K, S) if G is None else (G, L, K, S)
    ga = torch.randint(_lane.GMIN, _lane.GMAX + 1, shape, generator=g, dtype=torch.int32)
    gb = torch.randint(_lane.GMIN, _lane.GMAX + 1, shape, generator=g, dtype=torch.int32)
    qdiff, ds = quantize((ga - gb).numpy().astype(np.float32))
    qmag, ms = quantize((ga + gb).numpy().astype(np.float32))
    return ga, gb, torch.from_numpy(qdiff), torch.from_numpy(qmag), ds, ms


def make_aats(B, K, S, G, seed=1):
    g = torch.Generator().manual_seed(seed)
    shape = (B, K) if G is None else (B, G, K)
    a = torch.randint(0, S, shape, generator=g, dtype=torch.int64)
    return a.masked_fill(torch.rand(shape, generator=g) < 0.10, -1)


def y_from_sums(top, bot, ds=1.0, ms=1.0):
    t = top.to(torch.float64) * ds
    b = bot.to(torch.float64) * ms
    return torch.where(b != 0, t / torch.where(b != 0, b, torch.ones_like(b)),
                       torch.zeros_like(t))


def dense_controls(L, K, G, seed=3):
    """The matched dense op (per group, batched, when G is set)."""
    g = torch.Generator().manual_seed(seed)
    D = 2 * K
    if G is None:
        W = torch.randn(L, D, generator=g)
        lin = torch.nn.Linear(D, L, bias=False)
        with torch.no_grad():
            lin.weight.copy_(W)
        qlin = torch.quantization.quantize_dynamic(lin, {torch.nn.Linear},
                                                   dtype=torch.qint8)
        return (lambda x: F.linear(x, W)), (lambda x: qlin(x)), D
    W = torch.randn(G, D, L, generator=g)
    return (lambda x: torch.bmm(x, W)), None, D    # no off-the-shelf dyn-int8 bmm


# ---------------------------------------------------------------------------
# The race section.
# ---------------------------------------------------------------------------

def incumbent(w1, w2, paired, per_group, L, K):
    """The pre-08d formulation, batch-chunked so its [L, B, K] intermediate fits."""
    import math
    per_ex = (w1.shape[0] * L if per_group else math.prod(w1.shape[:-2])) * K * 4 * 2
    chunk = max(1, int(CHUNK_BYTES // per_ex))

    def run(aat):
        parts = [_lane.read_sums(w1, w2, aat[i:i + chunk], per_group=per_group,
                                 paired=paired)
                 for i in range(0, aat.shape[0], chunk)]
        return torch.cat([t for t, _ in parts]), torch.cat([b for _, b in parts])
    return run


def bag_variant(table, K, S, G):
    def run(aat):
        top, bot = _lane.bag_read_sums(table, aat, K, S, G=G)
        return top, bot
    return run


def gate_aats(K, S, G, seed=2):
    a = make_aats(GATE_B, K, S, G, seed)
    a[0] = -1                                   # an all-NONE read (bottom = 0 -> y = 0)
    a[1].clamp_(min=0)                          # a fully-active read
    return a


def run_race_shape(label, L, K, S, G, batches, rows):
    print(f"\n== {label} ==", flush=True)
    ga, gb, qd, qm, ds, ms = make_banks(L, K, S, G)
    pg = G is not None
    Gk = G if pg else None

    prep, tables = {}, {}
    for name, build in [("bag-fp32 (live)", lambda: _lane.bag_table(ga, gb, True, pg)),
                        ("bag-fp32 (frozen)", lambda: _lane.bag_table(qd, qm, False, pg)),
                        ("fbgemm-i8 (frozen)", lambda: _lane.fused_table(qd, qm, pg))]:
        t0 = time.perf_counter()
        tables[name] = build()
        prep[name] = time.perf_counter() - t0

    dense_f, dense_q, D = dense_controls(L, K, G)
    variants = {
        "incumbent (live)":   (incumbent(ga, gb, True, pg, L, K), 1.0, 1.0),
        "bag-fp32 (live)":    (bag_variant(tables["bag-fp32 (live)"], K, S, Gk), 1.0, 1.0),
        "incumbent (frozen)": (incumbent(qd, qm, False, pg, L, K), ds, ms),
        "bag-fp32 (frozen)":  (bag_variant(tables["bag-fp32 (frozen)"], K, S, Gk), ds, ms),
        "fbgemm-i8 (frozen)": (bag_variant(tables["fbgemm-i8 (frozen)"], K, S, Gk), ds, ms),
    }

    # The bit-exact gate: int sums and float64 y against the oracle formulation.
    ag = gate_aats(K, S, G)
    for name, (fn, vds, vms) in variants.items():
        w1, w2 = (qd, qm) if "frozen" in name else (ga, gb)
        top, bot = _lane.read_sums(w1, w2, ag, per_group=pg, paired="live" in name)
        t, b = fn(ag)
        ok = (torch.equal(t.to(torch.int32), top.to(torch.int32))
              and torch.equal(b.to(torch.int32), bot.to(torch.int32))
              and torch.equal(y_from_sums(t, b, vds, vms), y_from_sums(top, bot, vds, vms)))
        print(f"  gate {'PASS' if ok else 'FAIL'}  {name}", flush=True)
        if not ok:
            raise AssertionError(f"{label}: {name} failed the bit-exact gate")

    for B in batches:
        aat = make_aats(B, K, S, G, seed=10 + B)
        xd = torch.randn(B, D) if G is None else torch.randn(G, B, D)
        for name, (fn, vds, vms) in variants.items():
            t = bench(lambda: y_from_sums(*fn(aat), vds, vms))
            rows.append((label, name, B, B / t))
            print(f"  B={B:<6} {name:<20} {B / t:12,.0f} vec/s", flush=True)
        for name, ctl in [("dense fp32 (control)", dense_f), ("dense int8 (control)", dense_q)]:
            if ctl is None:
                continue
            t = bench(lambda: ctl(xd))
            rows.append((label, name, B, B / t))
            print(f"  B={B:<6} {name:<20} {B / t:12,.0f} vec/s", flush=True)

    aat = make_aats(4096, K, S, G, seed=99)
    y = y_from_sums(*variants["incumbent (live)"][0](aat))
    post = (lambda: y.argmax(dim=-1)) if pg else (lambda: _lane.rank_cut(y, 0.0, None))
    rows.append((label, "post (rank-cut/argmax) @B=4096", 4096, 4096 / bench(post)))
    for name, dt in prep.items():
        rows.append((label, f"prep: {name} (ms, one-time)", 0, dt * 1e3))


def race(lines, quick):
    batches = [1, 64, 4096] if quick else BATCHES
    rows = []
    for shape in SHAPES:
        run_race_shape(*shape, batches, rows)
    lines += ["## The race (median vec/s)", ""]
    for label in dict.fromkeys(r[0] for r in rows):
        lines += [f"### {label}", "",
                  "| variant | " + " | ".join(f"B={b}" for b in batches) + " |",
                  "|---|" + "---:|" * len(batches)]
        names = [n for n in dict.fromkeys(r[1] for r in rows if r[0] == label)
                 if not n.startswith(("post", "prep"))]
        cell = {(r[1], r[2]): r[3] for r in rows if r[0] == label}
        for n in names:
            lines.append(f"| {n} | " + " | ".join(
                f"{cell[(n, b)]:,.0f}" if (n, b) in cell else "—" for b in batches) + " |")
        lines += [""] + [f"- {r[1]}: {r[3]:,.1f}" for r in rows
                         if r[0] == label and r[1].startswith(("post", "prep"))] + [""]


# ---------------------------------------------------------------------------
# The conditional-compute section.
# ---------------------------------------------------------------------------

def varlen_bag(table, K, S):
    """The offsets form: NONE entries never enter the bag, so cost follows n_active."""
    def run(aat):
        active = aat >= 0
        idx = (torch.arange(K) * S + aat)[active]
        counts = active.sum(dim=1)
        offsets = torch.zeros(aat.shape[0], dtype=torch.int64)
        torch.cumsum(counts[:-1], 0, out=offsets[1:])
        out = F.embedding_bag(idx, table, offsets=offsets, mode="sum")
        L = out.shape[-1] // 2
        return out[..., :L].to(torch.int32), out[..., L:].to(torch.int32)
    return run


def sparse_aats(B, K, S, n_active, seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.full((B, K), -1, dtype=torch.int64)
    pos = torch.argsort(torch.rand(B, K, generator=g), dim=1)[:, :n_active]
    return a.scatter_(1, pos, torch.randint(0, S, (B, n_active), generator=g,
                                            dtype=torch.int64))


def conditional(lines, quick):
    B = 1024 if quick else 4096
    # Fire rates through the digits lane bridge (trained as the example trains it).
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from ktram_neural_core.torch.aat_codec import AATCodec

    digits = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(digits.data.astype("float32"), digits.target,
                                          test_size=0.25, random_state=0)
    codec = AATCodec(4, K=8).fit(torch.from_numpy(Xtr))
    codes_tr, codes_te = codec.encode(torch.from_numpy(Xtr)), codec.encode(torch.from_numpy(Xte))
    clf = Classifier(num_lanes=10, num_spaces=codec.k, num_channels=codec.S,
                     N=None, init="medium", seed=0)
    g = torch.Generator().manual_seed(0)
    for _ in range(2 if quick else 10):
        for i in torch.randperm(codes_tr.shape[0], generator=g):
            clf.adapt(codes_tr[i], torch.tensor([ytr[i]]))
    fired = (clf.read_y(codes_te) >= clf.Vt).to(torch.float64).mean(dim=-1)
    acc = float((clf.read(codes_te)[..., 0].numpy() == yte).mean())
    lines += ["## Conditional compute", "",
              f"Lane-bridge fire rate over {fired.numel()} held-out reads: mean "
              f"{fired.mean():.3f} (p10 {fired.quantile(0.1):.3f}, p90 "
              f"{fired.quantile(0.9):.3f}) at test accuracy {acc:.3f}.", ""]
    print(lines[-2], flush=True)

    for label, L, K, S, sweep in SWEEP_SHAPES:
        _, _, qd, qm, ds, ms = make_banks(L, K, S, None)
        table = _lane.bag_table(qd, qm, paired=False)
        var = varlen_bag(table, K, S)
        _, dense_q, D = dense_controls(L, K, None)
        xd = torch.randn(B, D)
        lines += [f"### {label} (B={B})", "",
                  "| active n | varlen bag (vec/s) | dense int8 (vec/s) "
                  "| lane bytes/vec | dense bytes/vec |", "|---:|---:|---:|---:|---:|"]
        for n in sweep:
            aat = sparse_aats(B, K, S, n, seed=n)
            top, bot = _lane.read_sums(qd, qm, aat, paired=False)
            t, b = var(aat)
            assert torch.equal(t, top) and torch.equal(b, bot), f"varlen gate n={n}"
            t_var = bench(lambda: y_from_sums(*var(aat), ds, ms))
            t_dq = bench(lambda: dense_q(xd))
            lines.append(f"| {n} | {B / t_var:,.0f} | {B / t_dq:,.0f} "
                         f"| {n * 2 * L:,} | {D * L:,} |")
            print(f"  {label} n={n}: varlen {B / t_var:,.0f} vec/s", flush=True)
        lines.append("")


# ---------------------------------------------------------------------------
# The crossover section.
# ---------------------------------------------------------------------------

def crossover(lines, quick):
    bs = CROSS_BS[:6] if quick else CROSS_BS
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    lines += ["## Streaming vs batched adapt (ex/s)", ""]
    for device in devices:
        lines += [f"### {device}", "",
                  "| B | Classifier batched | Classifier streaming "
                  "| BasisEncoder batched | BasisEncoder streaming |",
                  "|---:|---:|---:|---:|---:|"]
        for B in bs:
            g = torch.Generator().manual_seed(0)
            aat_c = torch.randint(0, 64, (B, 16), generator=g).to(device)
            tgt = torch.randint(0, 10, (B, 1), generator=g).to(device)
            aat_e = torch.randint(0, 2, (B, 16, 49), generator=g).to(device)
            cells = []
            for batched, streamed, make in [
                (lambda m: m.adapt(aat_c, tgt),
                 lambda m: [m.adapt(aat_c[i], tgt[i]) for i in range(B)],
                 lambda: Classifier(10, 16, 64, N=None, init="medium", seed=0).to(device)),
                (lambda m: m.adapt(aat_e, per_group=True),
                 lambda m: [m.adapt(aat_e[i], per_group=True) for i in range(B)],
                 lambda: BasisEncoder(16, 64, 49, 2, init="low", seed=0).to(device)),
            ]:
                for path in (batched, streamed):
                    m = make()
                    def timed(m=m, path=path):
                        path(m)
                        if device == "mps":
                            torch.mps.synchronize()
                    cells.append(f"{B / bench(timed):,.0f}")
            lines.append(f"| {B} | " + " | ".join(cells) + " |")
            print(f"  {device} B={B}: " + " / ".join(cells), flush=True)
        lines.append("")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--section", choices=["race", "conditional", "crossover", "all"],
                    default="all")
    ap.add_argument("--quick", action="store_true", help="smoke run: smaller batches")
    args = ap.parse_args()
    torch.manual_seed(0)
    lines = ["# Lane-read benchmark data",
             "",
             f"Generated by `benchmarks/bench_lane_read.py`, {time.strftime('%Y-%m-%d %H:%M')}. "
             f"torch {torch.__version__}, {torch.get_num_threads()} threads. Every lane-read "
             "variant is gated bit-exact against `_lane.read_sums` before timing; dense rows "
             "are controls at the matched shape, not reads. The curated commentary is "
             "lane-read-bench.md.",
             ""]
    if args.section in ("race", "all"):
        race(lines, args.quick)
    if args.section in ("conditional", "all"):
        conditional(lines, args.quick)
    if args.section in ("crossover", "all"):
        crossover(lines, args.quick)
    OUT.write_text("\n".join(lines))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
