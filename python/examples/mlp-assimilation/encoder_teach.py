"""Encoder study, step 2: a screened encoder + proxy onto real lanes, freeze-and-harvest.

Loads <screen>/screen.pt (frozen encoder + proxy from encoder_screen.py). For the lane
encoder (`a2d`) the encoder itself is a grouped Classifier bank (one group per patch:
D slots x L channels -> S lanes) taught by `adapt` from the proxy's patch symbols, frozen,
and its ACTUAL emissions feed stage 1. For a digital encoder the frozen torch module emits
the codes. Then stage 1, stage 2 and the head exactly as the companion's `teach_chain`.

Every synapse is set by `Classifier.adapt`. Validation selects each bank's checkpoint;
the test sets are read once at the end: `test` is the 36,000 images after training and
validation, and `historical` is the companion's last 2,000 (a subset of `test`). A
head-only control adds a ten-lane head taught on the encoder's emissions.
Writes <screen>/teach-seed<k>.json and the packs.

    python encoder_teach.py --screen figures/encoders/a2d-S64-L4-quantile --seed 0
"""
import argparse
import hashlib
import json
import pathlib
import time

import numpy as np
import torch

from ktram_neural_core.torch import Classifier
from models import Teacher, patchify
from encoders import make_encoder
from encoder_screen import HID, ProxyS


def load_all(cfg):
    """Train / val / historical / test (everything after train+val), same permutation as
    the companion. Test includes the historical 2,000."""
    from sklearn.datasets import fetch_openml, load_digits
    if cfg["dataset"] == "digits":
        ds = load_digits(); x = ds.data.astype(np.float32) / 16.0
    else:
        ds = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
        x = ds.data.astype(np.float32) / 255.0
    y = np.asarray(ds.target, dtype=np.int64)
    side = int(x.shape[1] ** 0.5)
    p = np.random.default_rng(cfg["seed"]).permutation(len(x))
    nt, nv, ne = cfg["n_train"], cfg["n_val"], cfg["n_test"]
    idx = dict(train=p[:nt], val=p[nt:nt + nv], historical=p[-ne:], test=p[nt + nv:])
    out = {k: (torch.from_numpy(x[i]).reshape(-1, 1, side, side), torch.from_numpy(y[i])) for k, i in idx.items()}
    return out, {k: hashlib.sha256(i.astype("<i8").tobytes()).hexdigest() for k, i in idx.items()}


def agreement(a, b):
    return (a == b).float().mean().item()


@torch.no_grad()
def group_codes(bank, aat, k_out, per_group=False, batch=1024):
    """Per-slot winner by argmax over the slot's lanes (the companion's lane_codes);
    grouped banks read per group."""
    outs = []
    for b in aat.split(batch):
        y = bank.read_y(b, per_group=per_group) if per_group else bank.read_y(b).reshape(len(b), k_out, -1)
        outs.append(y.argmax(-1))
    return torch.cat(outs)


def teach_bank(bank, inputs, targets, val_inputs, val_targets, k_out, s_out, per_group,
               epochs, batch, seed, tag):
    if list(bank.parameters()):
        raise AssertionError("the lane bank must not expose optimizer parameters")
    tl = targets if per_group else targets + s_out * torch.arange(k_out)
    if per_group:
        tl = tl.unsqueeze(-1)                      # [B, G, 1]: one target lane per group
    gen = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, curve = -1.0, None, None, []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        for b in torch.randperm(len(inputs), generator=gen).split(batch):
            bank.adapt(inputs[b], tl[b], feedback="hard", per_group=per_group)
        score = agreement(group_codes(bank, val_inputs, k_out, per_group), val_targets)
        curve.append(dict(epoch=epoch + 1, validation_agreement=score,
                          presentations=(epoch + 1) * len(inputs), seconds=time.perf_counter() - t0))
        if score > best:
            best, best_epoch = score, epoch + 1
            best_state = {k: v.detach().clone() for k, v in bank.state_dict().items()}
        print(f"{tag:26s} {epoch + 1:2d}/{epochs}  validation {score:.4f}", flush=True)
    bank.load_state_dict(best_state)
    bank._bag = {}
    return dict(curve=curve, selected_epoch=best_epoch, selected_validation_agreement=best)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=0, help="bank init and shuffle seed")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--teach-epochs", type=int, nargs=4, default=[20, 20, 10, 10],
                    help="encoder bank (lane encoder only), stage 1, stage 2, head")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--enc-batch", type=int, default=64, help="adapt batch for the encoder bank only")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rep_path = a.screen / f"teach-seed{a.seed}.json"
    if rep_path.exists():
        print(f"cached: {rep_path}"); return
    scr = json.loads((a.screen / "screen.json").read_text())
    ck = torch.load(a.screen / "screen.pt", map_location="cpu", weights_only=True)
    c = scr["config"]
    cfg = scr["split"]                       # the effective split the screen used
    data, hashes = load_all(cfg)
    (xt, yt), (xv, yv) = data["train"], data["val"]
    side = xt.shape[-1]
    P, D, S = scr["patches"], scr["pixels"], c["S"]
    (k1, s1), (k2, s2) = HID

    # Frozen teacher (for predictions) and frozen proxy (for the targets).
    teacher = Teacher(side)
    if c["teacher"].startswith("trained here"):
        teacher = None
    else:
        teacher.load_state_dict(torch.load(c["teacher"], map_location="cpu", weights_only=True)["teacher"])
        teacher.eval().requires_grad_(False)
    enc = make_encoder(c["encoder"], P, D, S, train_patches=patchify(xt), L=c["L"], levels=c["levels"],
                       hidden=c["hidden"])
    proxy = ProxyS(enc, P, S)
    proxy.load_state_dict(ck["proxy"]); proxy.eval().requires_grad_(False)
    lane_encoder = c["encoder"] == "a2d"

    def codes_and_pred(x):
        with torch.no_grad():
            pt = patchify(x)
            c0, c1, c2, pp = (torch.cat(z) for z in zip(*[proxy.codes(b) for b in pt.split(1024)]))
            pred = pp if teacher is None else teacher(x)[2].argmax(-1)
            lv = enc.levels(pt) if lane_encoder else None
        return pt, lv, c0, c1, c2, pred

    pt_t, lv_t, c0t, c1t, c2t, predt = codes_and_pred(xt)
    pt_v, lv_v, c0v, c1v, c2v, predv = codes_and_pred(xv)

    report = dict(screen=scr["tag"], seed=a.seed, split_hashes=hashes, teach_epochs=a.teach_epochs,
                  batch=a.batch, enc_batch=a.enc_batch, stages={})
    t_all = time.perf_counter()
    banks = {}
    if lane_encoder:
        L = c["L"]
        banks["encoder"] = Classifier(S, D, L, num_groups=P, init="medium", seed=a.seed)
        report["stages"]["encoder"] = teach_bank(banks["encoder"], lv_t, c0t, lv_v, c0v, P, S, True,
                                                 a.teach_epochs[0], a.enc_batch, a.seed, f"{scr['tag']} enc-bank")
        in_t, in_v = group_codes(banks["encoder"], lv_t, P, True), group_codes(banks["encoder"], lv_v, P, True)
        report["stages"]["encoder"]["val_symbol_agreement_after_freeze"] = agreement(in_v, c0v)
    else:
        in_t, in_v = c0t, c0v                      # digital encoder
    enc_t, enc_v = in_t, in_v

    specs = [("stage1", (P, S, k1, s1), c1t, c1v), ("stage2", (k1, s1, k2, s2), c2t, c2v),
             ("head", (k2, s2, 1, 10), predt[:, None], predv[:, None])]
    for name, (ki, si, ko, so), tt, tv in specs:
        banks[name] = Classifier(ko * so, ki, si, init="medium", seed=a.seed)
        ep = a.teach_epochs[1 + list(n for n, *_ in specs).index(name)]
        report["stages"][name] = teach_bank(banks[name], in_t, tt, in_v, tv, ko, so, False, ep, a.batch,
                                            a.seed, f"{scr['tag']} {name}")
        in_t, in_v = group_codes(banks[name], in_t, ko), group_codes(banks[name], in_v, ko)

    # Head-only control: a ten-lane head taught on the encoder's emissions.
    head_only = Classifier(10, P, S, init="medium", seed=a.seed)
    report["head_only_teaching"] = teach_bank(head_only, enc_t, predt[:, None], enc_v, predv[:, None], 1, 10, False,
                                              a.teach_epochs[3], a.batch, a.seed, f"{scr['tag']} head-only")
    report["teach_seconds"] = time.perf_counter() - t_all

    # ---- test, read once ----
    def chain_pred(x):
        pt, lv, *_ = codes_and_pred(x)
        cur = group_codes(banks["encoder"], lv, P, True) if lane_encoder else proxy.codes(pt)[0]
        out = {"encoder": cur}
        for name, (_, _, ko, _), *_ in specs:
            cur = group_codes(banks[name], cur, ko); out[name] = cur
        return out

    for split in ("historical", "test"):
        xe, ye = data[split]
        em = chain_pred(xe)
        _, _, c0e, c1e, c2e, pred_e = codes_and_pred(xe)
        pred = em["head"].squeeze(-1)
        report[split] = dict(n=len(ye), accuracy=agreement(pred, ye), teacher_agreement=agreement(pred, pred_e),
                             head_only_accuracy=agreement(group_codes(head_only, em["encoder"], 1).squeeze(-1), ye),
                             symbol_agreement=dict(encoder=agreement(em["encoder"], c0e) if lane_encoder else 1.0,
                                                   stage1=agreement(em["stage1"], c1e), stage2=agreement(em["stage2"], c2e)))

    # packs and the accounting row
    pack_bytes = 0
    for name, bank in banks.items():
        path = a.screen / f"seed{a.seed}-{name}.npz"
        bank.to_pack(path); pack_bytes += path.stat().st_size
    syn_chain = P * S * k1 * s1 + k1 * s1 * k2 * s2 + k2 * s2 * 10
    cost = dict(scr["cost"])
    cost.update(chain_synapses=syn_chain, total_synapses=syn_chain + cost["encoder_synapses"],
                packed_bytes=2 * (syn_chain + cost["encoder_synapses"]), pack_file_bytes=pack_bytes,
                total_bytes=cost["digital_bytes"] + 2 * (syn_chain + cost["encoder_synapses"]),
                lane_reads_per_image=(P * S if lane_encoder else 0) + k1 * s1 + k2 * s2 + 10,
                teacher_params=side * side * 64 + 64 + 64 * 32 + 32 + 32 * 10 + 10,
                teacher_macs=side * side * 64 + 64 * 32 + 32 * 10)
    report["cost"] = cost
    rep_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(screen=scr["tag"], seed=a.seed, test=report["test"], historical=report["historical"],
                          total_bytes=cost["total_bytes"], digital_macs=cost["digital_macs"]), indent=2))


if __name__ == "__main__":
    main()
