"""Train a frozen teacher, choose symbolic interfaces, and teach real lane banks.

Run from a checkout with the package's torch/examples extras installed. No private
files, pretrained downloads, or imports from drafts are required. See README.md.
"""

import argparse
import copy
import hashlib
import json
import pathlib
import platform
import sys
import time

import numpy as np
import sklearn
import torch
from torch.nn import functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from ktram_neural_core.torch import Classifier  # noqa: E402
from ktram_neural_core.torch.aat_codec import AATCodec  # noqa: E402
from models import (  # noqa: E402
    STAGES, LiftStage, PatchEncoder, Proxy, Student, Teacher, freeze,
    lane_codes, new_banks, patchify, straight_through, usage_entropy,
)


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def split_indices(n, n_train, n_val, n_test, seed):
    if min(n_train, n_val, n_test) < 1 or n_train + n_val + n_test > n:
        raise ValueError("train, validation, and test sizes must be positive and disjoint")
    p = np.random.default_rng(seed).permutation(n)
    return p[:n_train], p[n_train:n_train + n_val], p[-n_test:]


def load_data(cfg):
    from sklearn.datasets import fetch_openml, load_digits
    if cfg["dataset"] == "digits":
        ds = load_digits()
        x = ds.data.astype(np.float32) / 16.0
    else:
        print("Loading Fashion-MNIST from the scikit-learn cache (download on first use).", flush=True)
        ds = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
        x = ds.data.astype(np.float32) / 255.0
    y = np.asarray(ds.target, dtype=np.int64)
    side = int(x.shape[1] ** 0.5)
    indices = split_indices(len(x), cfg["n_train"], cfg["n_val"], cfg["n_test"], cfg["seed"])
    splits = [(torch.from_numpy(x[idx]).reshape(-1, 1, side, side),
               torch.from_numpy(y[idx])) for idx in indices]
    return splits, [hashlib.sha256(idx.astype("<i8").tobytes()).hexdigest() for idx in indices]


def batches(n, size, generator):
    return torch.randperm(n, generator=generator).split(size)


def fit_gradient(model, cfg, n, epochs, loss_fn, score_fn, tag):
    """Select by validation only; the test split is not an argument."""
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    gen = torch.Generator().manual_seed(cfg["seed"])
    best_score, best_state, best_epoch = -1.0, None, None
    curve = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        temperature = 2.0 - 1.5 * epoch / max(epochs - 1, 1)
        total = 0.0
        for b in batches(n, cfg["batch"], gen):
            loss = loss_fn(b, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            score = score_fn()
        curve.append(dict(epoch=epoch + 1, validation_agreement=score,
                          loss=total / n, temperature=temperature,
                          seconds=time.perf_counter() - start))
        if score > best_score:
            best_score, best_state, best_epoch = score, snapshot(model), epoch + 1
        print(f"{tag:20s} {epoch + 1:2d}/{epochs}  validation {score:.4f}", flush=True)
    model.load_state_dict(best_state)
    return dict(curve=curve, selected_epoch=best_epoch, selected_validation_agreement=best_score)


@torch.no_grad()
def teacher_outputs(teacher, x):
    parts = [teacher(b) for b in x.split(1024)]
    return tuple(torch.cat([p[i] for p in parts]) for i in range(3))


@torch.no_grad()
def proxy_codes(proxy, patches):
    parts = [proxy.codes(b) for b in patches.split(1024)]
    return tuple(torch.cat([p[i] for p in parts]) for i in range(4))


@torch.no_grad()
def encoder_codes(encoder, patches):
    return torch.cat([encoder(b).argmax(-1) for b in patches.split(1024)])


def agreement(a, b):
    return (a == b).float().mean().item()


def teach_stage(bank, inputs, targets, val_inputs, val_targets, k_out, s_out, cfg, epochs, tag):
    """All synaptic learning goes through adapt; Adam never sees these buffers."""
    if list(bank.parameters()):
        raise AssertionError("the lane bank must not expose optimizer parameters")
    target_lanes = targets + s_out * torch.arange(k_out)
    gen = torch.Generator().manual_seed(cfg["seed"])
    curve, best_state, best_score, best_epoch = [], None, -1.0, None
    start = time.perf_counter()
    for epoch in range(epochs):
        for b in batches(len(inputs), cfg["batch"], gen):
            bank.adapt(inputs[b], target_lanes[b], feedback="hard")
        score = agreement(lane_codes(bank, val_inputs, k_out), val_targets)
        curve.append(dict(epoch=epoch + 1, validation_agreement=score,
                          presentations=(epoch + 1) * len(inputs),
                          seconds=time.perf_counter() - start))
        if score > best_score:
            best_score, best_state, best_epoch = score, snapshot(bank), epoch + 1
        print(f"{tag:20s} {epoch + 1:2d}/{epochs}  validation {score:.4f}", flush=True)
    bank.load_state_dict(best_state)
    return dict(curve=curve, selected_epoch=best_epoch, selected_validation_agreement=best_score)


def teach_chain(encoder, train_codes, val_codes, pred_train, pred_val, cfg, tag):
    banks = new_banks(cfg["seed"])
    inputs, val_inputs = train_codes[0], val_codes[0]
    targets = (*train_codes[1:3], pred_train[:, None])
    val_targets = (*val_codes[1:3], pred_val[:, None])
    reports = []
    for i, (bank, (_, _, ko, so), epochs) in enumerate(zip(banks, STAGES, cfg["teach_epochs"])):
        reports.append(teach_stage(bank, inputs, targets[i], val_inputs, val_targets[i],
                                   ko, so, cfg, epochs, f"{tag} lane {i + 1}"))
        # Freeze this bank by stopping feedback, then harvest its ACTUAL outputs.
        # Clean targets for the next stage stay unchanged.
        inputs = lane_codes(bank, inputs, ko)
        val_inputs = lane_codes(bank, val_inputs, ko)
    return Student(freeze(copy.deepcopy(encoder)), banks), reports


def costs(side):
    teacher_parameters = side * side * 64 + 64 + 64 * 32 + 32 + 32 * 10 + 10
    teacher_macs = side * side * 64 + 64 * 32 + 32 * 10
    encoder_macs = 16 * ((side // 4) ** 2 * 32 + 32 * 64)
    encoder_parameters = encoder_macs + 16 * (32 + 64)
    synapses = sum(ki * si * ko * so for ki, si, ko, so in STAGES)
    return dict(teacher_parameters=teacher_parameters, teacher_fp32_bytes=4 * teacher_parameters,
                teacher_macs_per_image=teacher_macs, encoder_parameters=encoder_parameters,
                encoder_fp32_bytes=4 * encoder_parameters, encoder_macs_per_image=encoder_macs,
                synapses=synapses, packed_synapse_payload_bytes=2 * synapses,
                encoder_plus_packed_payload_bytes=4 * encoder_parameters + 2 * synapses,
                live_ga_gb_int32_bytes=8 * synapses,
                active_synapses_per_image=sum(ki * ko * so for ki, si, ko, so in STAGES),
                retained_teacher_parameters=0,
                note="Payload excludes pack headers/scales, temporary activations, read caches and host overhead. "
                     "MAC count covers the digital encoder only; lane work is additional.")


@torch.no_grad()
def predict(student, images):
    return torch.cat([student(b) for b in images.split(1024)])


def load_student(run_directory, arm="learned"):
    """Reload this example's own checkpoint; inference has no teacher or probes."""
    run_directory = pathlib.Path(run_directory)
    checkpoint = torch.load(run_directory / "models.pt", map_location="cpu", weights_only=True)
    encoder = PatchEncoder(checkpoint["side"])
    student = Student(encoder, new_banks(checkpoint["config"]["seed"]))
    student.load_state_dict(checkpoint["students"][arm])
    return freeze(student)


def run(cfg, out):
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    out.mkdir(parents=True, exist_ok=True)
    splits, index_hashes = load_data(cfg)
    (xt, yt), (xv, yv), (xe, ye) = splits
    side = xt.shape[-1]
    report = dict(config=cfg, split_index_sha256=index_hashes, costs=costs(side),
                  versions=dict(python=platform.python_version(), torch=str(torch.__version__),
                                numpy=np.__version__, sklearn=sklearn.__version__), arms={})
    print(json.dumps(report["costs"], indent=2), flush=True)
    teacher = Teacher(side)
    report["teacher_training"] = fit_gradient(
        teacher, cfg, len(xt), cfg["teacher_epochs"],
        lambda b, tau: F.cross_entropy(teacher(xt[b])[2], yt[b]),
        lambda: agreement(teacher_outputs(teacher, xv)[2].argmax(-1), yv), "teacher")
    freeze(teacher)
    teacher_state = snapshot(teacher)
    z1t, z2t, lt = teacher_outputs(teacher, xt)
    z1v, z2v, lv = teacher_outputs(teacher, xv)
    predt, predv = lt.argmax(-1), lv.argmax(-1)
    pt, pv = patchify(xt), patchify(xv)

    # Books use training activations only. They supply the fitted-interface arm
    # and the common encoder warm start; they are absent from student inference.
    book1, book2 = AATCodec(4, K=8).fit(z1t), AATCodec(4, K=8).fit(z2t)
    c1t, c2t = book1.encode(z1t), book2.encode(z2t)
    c1v, c2v = book1.encode(z1v), book2.encode(z2v)
    encoder, reader = PatchEncoder(side), LiftStage(16, 64, 32, 16)
    warm = torch.nn.ModuleList([encoder, reader])

    def warm_loss(b, tau):
        lift, soft = straight_through(encoder(pt[b]), tau)
        return F.cross_entropy(reader(lift).flatten(0, 1), c1t[b].flatten()) - usage_entropy(soft)

    def warm_score():
        codes = encoder_codes(encoder, pv)
        predicted = torch.cat([reader(F.one_hot(b, 64).float()).argmax(-1) for b in codes.split(1024)])
        return agreement(predicted, c1v)

    report["encoder_training"] = fit_gradient(
        warm, cfg, len(xt), cfg["encoder_epochs"], warm_loss, warm_score, "encoder warm start")
    c0t, c0v = encoder_codes(encoder, pt), encoder_codes(encoder, pv)
    students = {}
    students["fitted"], teaching = teach_chain(
        encoder, (c0t, c1t, c2t), (c0v, c1v, c2v), predt, predv, cfg, "fitted")
    report["arms"]["fitted"] = dict(teaching=teaching)

    # New tables are learned from scratch; the input encoder starts from the
    # SAME warm state used by the fitted arm. --freeze-input isolates its state.
    torch.manual_seed(cfg["seed"] + 1)
    proxy = Proxy(copy.deepcopy(encoder))
    if cfg["freeze_input"]:
        freeze(proxy.encoder)
    variance1, variance2 = z1t.var().clamp_min(1e-8), z2t.var().clamp_min(1e-8)

    def proxy_loss(b, tau):
        logits, (l1, l2), soft = proxy(pt[b], tau)
        hint = (F.mse_loss(proxy.probe1(l1.flatten(1)), z1t[b]) / variance1
                + F.mse_loss(proxy.probe2(l2.flatten(1)), z2t[b]) / variance2)
        return F.cross_entropy(logits, predt[b]) + 0.3 * hint - sum(usage_entropy(s) for s in soft)

    report["proxy_training"] = fit_gradient(
        proxy, cfg, len(xt), cfg["proxy_epochs"], proxy_loss,
        lambda: agreement(proxy_codes(proxy, pv)[3], predv), "learned proxy")
    if cfg["freeze_input"] and not all(
        torch.equal(value, proxy.encoder.state_dict()[key])
        for key, value in encoder.state_dict().items()
    ):
        raise AssertionError("the fixed-input control changed its encoder")
    freeze(proxy)
    pct, pcv = proxy_codes(proxy, pt), proxy_codes(proxy, pv)
    students["learned"], teaching = teach_chain(
        proxy.encoder, pct, pcv, predt, predv, cfg, "learned")
    report["arms"]["learned"] = dict(teaching=teaching)
    report["hard_training_symbols_used"] = [
        [int(torch.unique(codes[:, j]).numel()) for j in range(codes.shape[1])]
        for codes in pct[:3]
    ]
    if not all(torch.equal(value, teacher.state_dict()[key]) for key, value in teacher_state.items()):
        raise AssertionError("teacher changed during assimilation")

    # The first test evaluation happens after both students have been selected.
    z1e, z2e, le = teacher_outputs(teacher, xe)
    teacher_pred = le.argmax(-1)
    pce = proxy_codes(proxy, patchify(xe))
    report["teacher_test_accuracy"] = agreement(teacher_pred, ye)
    report["proxy_test"] = dict(accuracy=agreement(pce[3], ye),
                                teacher_agreement=agreement(pce[3], teacher_pred))
    torch.save(dict(config=cfg, side=side, teacher=teacher_state,
                    warm_start=warm.state_dict(), proxy=proxy.state_dict(),
                    fitted_books=[book1.state_dict(), book2.state_dict()],
                    students={name: s.state_dict() for name, s in students.items()}), out / "models.pt")
    for name, student in students.items():
        pred = predict(student, xe)
        target1, target2 = ((book1.encode(z1e), book2.encode(z2e)) if name == "fitted" else pce[1:3])
        code0 = encoder_codes(student.encoder, patchify(xe))
        code1 = lane_codes(student.banks[0], code0, 32)
        code2 = lane_codes(student.banks[1], code1, 16)
        report["arms"][name]["test"] = dict(
            accuracy=agreement(pred, ye), teacher_agreement=agreement(pred, teacher_pred),
            stage1_symbol_agreement=agreement(code1, target1),
            stage2_symbol_agreement=agreement(code2, target2))
        if not torch.equal(pred, predict(load_student(out, name), xe)):
            raise AssertionError("checkpoint reload changed predictions")
        pack_paths = [out / f"{name}-stage{i + 1}.npz" for i in range(3)]
        for bank, path in zip(student.banks, pack_paths):
            bank.to_pack(path)
        packed = Student(copy.deepcopy(student.encoder),
                         torch.nn.ModuleList([Classifier.from_pack(p) for p in pack_paths]))
        packed_pred = predict(packed, xe)
        report["arms"][name]["packed_test"] = dict(
            accuracy=agreement(packed_pred, ye), teacher_agreement=agreement(packed_pred, teacher_pred),
            live_prediction_agreement=agreement(packed_pred, pred),
            lane_pack_file_bytes=sum(p.stat().st_size for p in pack_paths))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="offline digits, 512/128/128 samples, short schedule")
    parser.add_argument("--freeze-input", action="store_true", help="hold the common input encoder fixed in proxy training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256, help="adapt batch; use 1 for serial updates")
    parser.add_argument("--teacher-epochs", type=int, default=40)
    parser.add_argument("--encoder-epochs", type=int, default=30)
    parser.add_argument("--proxy-epochs", type=int, default=40)
    parser.add_argument("--teach-epochs", type=int, nargs=3, default=[20, 10, 10])
    parser.add_argument("--n-train", type=int, default=32000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--out", type=pathlib.Path, default=HERE / "figures")
    args = parser.parse_args()
    cfg = vars(args).copy()
    output_root = cfg.pop("out")
    smoke = cfg.pop("smoke")
    cfg["dataset"] = "digits" if smoke else "Fashion-MNIST"
    if smoke:
        cfg.update(n_train=512, n_val=128, n_test=128, teacher_epochs=2,
                   encoder_epochs=2, proxy_epochs=2, teach_epochs=[1, 1, 1], batch=64)
    positive = [cfg[k] for k in ("threads", "batch", "teacher_epochs", "encoder_epochs", "proxy_epochs",
                                "n_train", "n_val", "n_test")] + cfg["teach_epochs"]
    if min(positive) < 1 or cfg["seed"] < 0:
        parser.error("counts must be positive and seed must be nonnegative")
    # Key outputs by settings AND source, so an edited example cannot reuse old results.
    digest = hashlib.sha256()
    for path in sorted(HERE.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    for path in sorted((HERE.parents[1] / "ktram_neural_core").rglob("*.py")):
        digest.update(str(path.relative_to(HERE.parents[1])).encode())
        digest.update(path.read_bytes())
    cfg["source_sha256"] = digest.hexdigest()
    cfg["torch_version"] = str(torch.__version__)
    cfg["numpy_version"] = np.__version__
    cfg["sklearn_version"] = sklearn.__version__
    key = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    out = output_root / f"{cfg['dataset'].lower()}-{key}"
    result_path = out / "results.json"
    start = time.perf_counter()
    if result_path.exists():
        report = json.loads(result_path.read_text())
        print(f"Reusing completed run: {out}")
    else:
        print(f"Outputs: {out}", flush=True)
        report = run(cfg, out)
        report["elapsed_seconds"] = time.perf_counter() - start
        result_path.write_text(json.dumps(report, indent=2) + "\n")
    from figures import plot_run
    plot_run(report, out)
    print(json.dumps({"teacher": report["teacher_test_accuracy"], "proxy": report["proxy_test"],
                      **{name: arm["test"] for name, arm in report["arms"].items()}}, indent=2))
    print(f"Results and figures: {out}")


if __name__ == "__main__":
    main()
