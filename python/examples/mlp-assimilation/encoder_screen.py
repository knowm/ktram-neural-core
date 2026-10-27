"""Encoder study, step 1: the Chapter 8 proxy recipe with the input encoder swapped.

One encoder configuration per invocation. Trains (on the GPU when present) the companion's
warm start and learned proxy with this encoder. Validation only; no test split is touched
here. Writes <out>/<tag>/screen.json and screen.pt (encoder + proxy state for
encoder_teach.py).

    python encoder_screen.py --encoder a2d --S 64 --teacher figures/fashion-mnist-<hash>/models.pt
"""
import argparse
import json
import pathlib
import time

import torch
from torch.nn import functional as F

from ktram_neural_core.torch.aat_codec import AATCodec
from models import LiftStage, Teacher, patchify, straight_through, usage_entropy
import experiment as companion
from encoders import make_encoder

HERE = pathlib.Path(__file__).resolve().parent
HID = ((32, 16), (16, 16))          # the hidden interfaces, held fixed for every encoder


class ProxyS(torch.nn.Module):
    """The companion's Proxy with the input alphabet S as a parameter."""

    def __init__(self, encoder, P, S):
        super().__init__()
        self.encoder, self.S = encoder, S
        (k1, s1), (k2, s2) = HID
        self.stage1 = LiftStage(P, S, k1, s1)
        self.stage2 = LiftStage(k1, s1, k2, s2)
        self.head = LiftStage(k2, s2, 1, 10)
        self.probe1 = torch.nn.Linear(k1 * s1, 64)
        self.probe2 = torch.nn.Linear(k2 * s2, 32)

    def forward(self, patches, tau=1.0):
        l0, p0 = straight_through(self.encoder(patches), tau)
        l1, p1 = straight_through(self.stage1(l0), tau)
        l2, p2 = straight_through(self.stage2(l1), tau)
        return self.head(l2).squeeze(1), (l1, l2), (p0, p1, p2)

    @torch.no_grad()
    def codes(self, patches):
        (k1, s1), (k2, s2) = HID
        c0 = self.encoder(patches).argmax(-1)
        c1 = self.stage1(F.one_hot(c0, self.S).float()).argmax(-1)
        c2 = self.stage2(F.one_hot(c1, s1).float()).argmax(-1)
        pred = self.head(F.one_hot(c2, s2).float()).squeeze(1).argmax(-1)
        return c0, c1, c2, pred


def fit(params, n, epochs, batch, seed, loss_fn, score_fn, tag, model_for_state):
    """Adam, tau 2.0 -> 0.5, validation-selected checkpoint. Mirrors companion.fit_gradient."""
    opt = torch.optim.Adam(params, lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, curve = -1.0, None, None, []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        tau = 2.0 - 1.5 * epoch / max(epochs - 1, 1)
        model_for_state.train()
        total = 0.0
        for b in torch.randperm(n, generator=gen).split(batch):
            loss = loss_fn(b, tau)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        model_for_state.eval()
        with torch.no_grad():
            score = score_fn()
        curve.append(dict(epoch=epoch + 1, validation=score, loss=total / n, tau=tau,
                          seconds=time.perf_counter() - t0))
        if score > best:
            best, best_epoch = score, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model_for_state.state_dict().items()}
        print(f"{tag:22s} {epoch + 1:2d}/{epochs}  validation {score:.4f}", flush=True)
    model_for_state.load_state_dict(best_state)
    return dict(curve=curve, selected_epoch=best_epoch, selected_validation=best)


def agreement(a, b):
    return (a == b).float().mean().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="a2d", choices=["a2d", "linear", "mlp"])
    ap.add_argument("--init-seed", type=int, default=None,
                    help="init/shuffle seed for the encoder and proxy; the SPLIT stays at --seed (default: --seed)")
    ap.add_argument("--L", type=int, default=4, help="pixel levels (a2d)")
    ap.add_argument("--S", type=int, default=64, help="symbols per patch")
    ap.add_argument("--levels", default="quantile", choices=["uniform", "quantile"])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--teacher", type=pathlib.Path, default=None,
                    help="companion models.pt whose frozen teacher is reused (seed 0)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--encoder-epochs", type=int, default=30)
    ap.add_argument("--proxy-epochs", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=32000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "figures" / "encoders")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = dict(dataset="digits" if a.smoke else "Fashion-MNIST", n_train=a.n_train, n_val=a.n_val,
               n_test=a.n_test, seed=a.seed)
    if a.smoke:
        cfg.update(n_train=512, n_val=128, n_test=128)
        a.encoder_epochs = a.proxy_epochs = 2
    tag = f"{a.encoder}-S{a.S}" + (f"-L{a.L}-{a.levels}" if a.encoder == "a2d" else "") \
        + (f"-h{a.hidden}" if a.encoder == "mlp" else "") + (f"-seed{a.seed}" if a.seed else "") \
        + (f"-init{a.init_seed}" if a.init_seed is not None and a.init_seed != a.seed else "") + ("-smoke" if a.smoke else "")
    init_seed = a.seed if a.init_seed is None else a.init_seed
    out = a.out / tag
    if (out / "screen.json").exists():
        print(f"cached: {out / 'screen.json'}"); return
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(init_seed)
    dev = torch.device(a.device)
    splits, _ = companion.load_data(cfg)
    (xt, yt), (xv, yv), _ = splits
    side = xt.shape[-1]
    pt, pv = patchify(xt).to(dev), patchify(xv).to(dev)
    P, D = pt.shape[1], pt.shape[2]

    teacher = Teacher(side)
    if a.teacher is not None:
        ck = torch.load(a.teacher, map_location="cpu", weights_only=True)
        teacher.load_state_dict(ck["teacher"])
        teacher_src = str(a.teacher)
    else:   # smoke only: a quick throwaway teacher
        opt_t = torch.optim.Adam(teacher.parameters(), 1e-3)
        for _ in range(3):
            for b in torch.randperm(len(xt)).split(256):
                loss = F.cross_entropy(teacher(xt[b])[2], yt[b]); opt_t.zero_grad(); loss.backward(); opt_t.step()
        teacher_src = "trained here (smoke)"
    teacher.eval().requires_grad_(False)
    with torch.no_grad():
        z1t, z2t, lt = teacher(xt); z1v, z2v, lv = teacher(xv)
    predt, predv = lt.argmax(-1).to(dev), lv.argmax(-1).to(dev)
    teacher_val = agreement(lv.argmax(-1), yv)
    z1t, z2t = z1t.to(dev), z2t.to(dev)

    # Fitted first-layer book -> the common warm start, exactly the companion's step 2.
    book1 = AATCodec(4, K=8).fit(z1t.cpu())
    c1t, c1v = book1.encode(z1t.cpu()).to(dev), book1.encode(z1v).to(dev)
    (k1, s1), _ = HID
    enc = make_encoder(a.encoder, P, D, a.S, train_patches=pt, L=a.L, levels=a.levels, hidden=a.hidden).to(dev)
    reader = LiftStage(P, a.S, k1, s1).to(dev)
    warm = torch.nn.ModuleList([enc, reader])

    def warm_loss(b, tau):
        lift, soft = straight_through(enc(pt[b]), tau)
        return F.cross_entropy(reader(lift).flatten(0, 1), c1t[b].flatten()) - usage_entropy(soft)

    def warm_score():
        return agreement(reader(F.one_hot(enc(pv).argmax(-1), a.S).float()).argmax(-1), c1v)

    rep = dict(tag=tag, config=vars(a) | dict(out=str(a.out), teacher=teacher_src), split=cfg, patches=P, pixels=D,
               teacher_validation_accuracy=teacher_val, cost=enc.cost(side))
    rep["init_seed"] = init_seed
    rep["warm"] = fit(list(warm.parameters()), len(pt), a.encoder_epochs, a.batch, init_seed,
                      warm_loss, warm_score, f"{tag} warm", warm)

    # The learned proxy, encoder refined (the companion default).
    torch.manual_seed(init_seed + 1)
    proxy = ProxyS(enc, P, a.S).to(dev)
    v1, v2 = z1t.var().clamp_min(1e-8), z2t.var().clamp_min(1e-8)

    def proxy_loss(b, tau):
        logits, (l1, l2), soft = proxy(pt[b], tau)
        hint = F.mse_loss(proxy.probe1(l1.flatten(1)), z1t[b]) / v1 + F.mse_loss(proxy.probe2(l2.flatten(1)), z2t[b]) / v2
        return F.cross_entropy(logits, predt[b]) + 0.3 * hint - sum(usage_entropy(s) for s in soft)

    rep["proxy"] = fit(list(proxy.parameters()), len(pt), a.proxy_epochs, a.batch, init_seed,
                       proxy_loss, lambda: agreement(proxy.codes(pv)[3], predv), f"{tag} proxy", proxy)
    proxy.eval().requires_grad_(False)
    c0t, c1pt, c2pt, pp_t = proxy.codes(pt)
    c0v, c1pv, c2pv, pp_v = proxy.codes(pv)
    rep["proxy"]["validation_accuracy"] = agreement(pp_v, yv.to(dev))
    rep["proxy"]["validation_teacher_agreement"] = agreement(pp_v, predv)
    rep["symbols_in_use"] = [[int(torch.unique(c[:, j]).numel()) for j in range(c.shape[1])]
                             for c in (c0t, c1pt, c2pt)]

    torch.save(dict(config=rep["config"], encoder=enc.state_dict(), proxy=proxy.state_dict(),
                    book1=book1.state_dict()), out / "screen.pt")
    (out / "screen.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
    print(json.dumps(dict(tag=tag, teacher=teacher_val, proxy=rep["proxy"]["validation_accuracy"],
                          cost=rep["cost"]), indent=2))


if __name__ == "__main__":
    main()
