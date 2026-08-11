"""Benchmark the AAT Codec read kernels against their native-float equivalents, per backend.

Each codec op is timed next to the float op it replaces, on transformer-shaped tensors
(head_dim-sized vectors, attention-length sequences). The attention-shaped ops run in two
forms, and the report carries two ratios:

  * The default kernels (`attention_scores`, `combine`) are the digital fast path: decode at
    the GEMM boundary, contract on the native matmul. Exact — the table sum over slots IS
    the inner product of the decodes — so the wall ratio lands near 1× plus the decode
    overhead, which the O(T²) contraction amortizes as shapes grow. Codes stay the storage
    and cache format. Fusing the decode into the GEMM tile (the quantized-inference pattern)
    is the 08c move that turns the smaller memory traffic into wall wins.
  * The `*_table` forms are the symbol-space kernels — the exact model of AAT-native
    hardware. Their **op ratio** (~0.5×, from op_counts) is the hardware-facing figure: half
    the elementary operations of the float op. Their **wall ratio** on CPU/GPU sits near the
    lift ratio k·S/D above native, because the one-hot lift hands the GEMM a k·S-wide inner
    dimension; they are the reference, not the digital production path.

encode/decode have no native twin; they report throughput.

Run:  python benchmarks/bench_codec.py [--device cpu|mps|cuda|all] [--out codec-bench.md]
"""
import argparse
import time
from pathlib import Path

import torch

from ktram_neural_core.torch.aat_codec import (AATCodec, attention_scores,
                                               attention_scores_table, combine,
                                               combine_table, dot_table)
from ktram_neural_core.torch.aat_codec import inner as inner_k
from ktram_neural_core.torch.aat_codec import norm as norm_k

BITS, K = 6, 8
SHAPES = [                       # (label, head batch n, Tq, Tk, D)
    ("small head", 8, 256, 256, 64),
    ("llama head", 8, 512, 512, 128),
    ("long context", 4, 2048, 2048, 128),
]
ROWWISE_N = 65536                # rows for the elementwise ops (inner / norm / encode / decode)


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def bench(fn, device, warmup=3, reps=10):
    """Median wall time of fn() in ms, synchronized on the accelerator."""
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1e3


def fit_codec(D, seed=0):
    g = torch.Generator().manual_seed(seed)
    scale = torch.exp(0.8 * torch.randn(D, generator=g))
    W = torch.randn(D, D, generator=g) / D ** 0.5
    X = (torch.randn(20000, D, generator=g) @ W) * scale
    return AATCodec(BITS, K=K).fit(X), (W, scale, g)


def _op_ratio(codec, op):
    """Elementary-op ratio codec/native for the dot-shaped reads (the frozen search figure)."""
    D = codec.k * 2
    counts = codec.op_counts(op)
    codec_ops = sum(counts.values())
    native_ops = (D + D - 1) + (1 if op == "norm" else 0)     # D mults + D-1 adds (+ sqrt)
    return codec_ops / native_ops


def _row(rows, device, op, shape, t_codec, t_native, op_ratio=None):
    rows.append((device, op, shape, t_codec, t_native, op_ratio))
    line = f"  {op:<18} {shape:<34} codec {t_codec:8.2f} ms"
    if t_native is not None:
        line += f"   native {t_native:8.2f} ms   wall {t_codec / t_native:5.2f}"
    if op_ratio is not None:
        line += f"   ops {op_ratio:.2f}"
    print(line)


def run_device(device, rows):
    print(f"\n== {device} ==")
    codecs = {}
    for D in sorted({s[-1] for s in SHAPES}):
        codec, (W, scale, g) = fit_codec(D)
        table = dot_table(codec, codec).to(device)
        codecs[D] = (codec.to(device), table, W, scale, g)

    # --- attention-shaped ops ---
    for label, n, Tq, Tk, D in SHAPES:
        codec, table, W, scale, g = codecs[D]
        Q = ((torch.randn(n, Tq, D, generator=g) @ W) * scale).to(device)
        Kf = ((torch.randn(n, Tk, D, generator=g) @ W) * scale).to(device)
        cq, ck = codec.encode(Q), codec.encode(Kf)
        alpha = torch.softmax(torch.randn(n, Tq, Tk, generator=g), dim=-1).to(device)
        shape = f"{label} [{n},{Tq},{Tk}] D={D}"

        t_native = bench(lambda: Q @ Kf.transpose(1, 2), device)
        t_codec = bench(lambda: attention_scores(codec, cq, codec, ck), device)
        _row(rows, device, "attention_scores", shape, t_codec, t_native)
        Kdec = codec.decode(ck)                      # the KV-cache steady state: keys decoded once
        t_codec = bench(lambda: codec.decode(cq) @ Kdec.transpose(-1, -2), device)
        _row(rows, device, "attention_scores (K cached)", shape, t_codec, t_native)
        t_codec = bench(lambda: attention_scores_table(cq, ck, table), device)
        _row(rows, device, "attention_scores (table)", shape, t_codec, t_native,
             _op_ratio(codec, "dot"))

        t_native = bench(lambda: alpha @ Kf, device)
        t_codec = bench(lambda: combine(codec, ck, alpha), device)
        _row(rows, device, "combine", shape, t_codec, t_native)
        t_codec = bench(lambda: combine_table(codec, ck, alpha), device)
        _row(rows, device, "combine (table)", shape, t_codec, t_native)

    # --- rowwise ops ---
    D = 128
    codec, table, W, scale, g = codecs[D]
    X = ((torch.randn(ROWWISE_N, D, generator=g) @ W) * scale).to(device)
    Y = ((torch.randn(ROWWISE_N, D, generator=g) @ W) * scale).to(device)
    cx, cy = codec.encode(X), codec.encode(Y)
    shape = f"[{ROWWISE_N},{D}]"

    t_native = bench(lambda: (X * Y).sum(-1), device)
    t_codec = bench(lambda: inner_k(cx, cy, table), device)
    _row(rows, device, "inner", shape, t_codec, t_native, _op_ratio(codec, "dot"))

    t_native = bench(lambda: X.norm(dim=-1), device)
    t_codec = bench(lambda: norm_k(codec, cx), device)
    _row(rows, device, "norm", shape, t_codec, t_native, _op_ratio(codec, "norm"))

    _row(rows, device, "encode", shape, bench(lambda: codec.encode(X), device), None)
    _row(rows, device, "decode", shape, bench(lambda: codec.decode(cx), device), None)


def emit(rows, out_path):
    lines = ["# AAT Codec read kernels vs native float",
             "",
             f"Generated by `benchmarks/bench_codec.py`, {time.strftime('%Y-%m-%d %H:%M')}. "
             f"torch {torch.__version__}. bits={BITS} (S={2**BITS}), K={K}. Median of 10 runs. "
             "op ratio = elementary operations codec/native (the frozen ~0.5× search figure, "
             "the hardware-facing number); wall ratio = measured torch time codec/native on "
             "this backend, where the native side is a tuned GEMM — the 08c performance pass "
             "works this column down toward the op column. encode/decode have no native twin "
             "and report time (and Mvec/s) only.",
             ""]
    for device in dict.fromkeys(r[0] for r in rows):
        lines += [f"## {device}", "",
                  "| op | shape | codec (ms) | native (ms) | wall ratio | op ratio |",
                  "|---|---|---:|---:|---:|---:|"]
        for dev, op, shape, tc, tn, oratio in rows:
            if dev != device:
                continue
            ocell = f"{oratio:.2f}" if oratio is not None else "—"
            if tn is None:
                n_rows = int(shape.strip("[]").split(",")[0])
                lines.append(f"| {op} | {shape} | {tc:.2f} | — | {n_rows / tc / 1e3:.1f} Mvec/s "
                             f"| {ocell} |")
            else:
                lines.append(f"| {op} | {shape} | {tc:.2f} | {tn:.2f} | {tc / tn:.2f} | {ocell} |")
        lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="all")
    ap.add_argument("--out", default=str(Path(__file__).parent / "codec-bench.md"))
    args = ap.parse_args()
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    if args.device != "all":
        devices = [args.device]
    torch.set_num_threads(torch.get_num_threads())   # whatever the host gives; report is per-host
    rows = []
    for device in devices:
        run_device(device, rows)
    emit(rows, Path(args.out))


if __name__ == "__main__":
    main()
