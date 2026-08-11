"""The lane bridge — codec output feeding the torch L1 lanes, end to end.

    python examples/aat-codec/lane_bridge.py

The whole assimilation datapath in one small script (spec 08 §7). Float vectors go in one
side; everything after the codec runs on AAT tensors:

    float x  --AATCodec.encode-->  AAT [.., k]  --Classifier.adapt/read-->  output AAT

The codec's codes ARE the lane's input contract — ``[..., k]`` int64, one symbol per space —
so the bridge is a plain function call, no adapter in between. The script fits a codec on the
sklearn digits features (D=64 -> k=32 slots), checks what the code preserves (reconstruction
cosine, score correlation through the dot-table kernel), then trains a 10-lane torch
Classifier directly on the codes and reports train/test accuracy.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # python/ on path

import torch                                                          # noqa: E402
from sklearn.datasets import load_digits                              # noqa: E402
from sklearn.model_selection import train_test_split                  # noqa: E402

from ktram_neural_core.torch import Classifier                        # noqa: E402
from ktram_neural_core.torch.aat_codec import AATCodec, attention_scores  # noqa: E402

BITS = 4          # S = 16 symbols per slot
K = 8             # archetype groups
EPOCHS = 10
SEED = 0


def main():
    digits = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(
        digits.data.astype("float32"), digits.target, test_size=0.25, random_state=SEED)
    Xtr = torch.from_numpy(Xtr)
    Xte = torch.from_numpy(Xte)

    # --- the codec side: floats -> AATs ---------------------------------------
    codec = AATCodec(BITS, K=K).fit(Xtr)
    codes_tr = codec.encode(Xtr)                       # [N, k] int64 — a batch of AATs
    codes_te = codec.encode(Xte)
    recon = codec.decode(codes_tr)
    cos = torch.nn.functional.cosine_similarity(Xtr, recon, dim=-1).mean()
    est = attention_scores(codec, codes_te[:256].unsqueeze(0),
                           codec, codes_te[256:512].unsqueeze(0))[0]
    true = Xte[:256] @ Xte[256:512].T
    corr = torch.corrcoef(torch.stack([true.flatten(), est.flatten()]))[0, 1]
    print(f"codec: {codec}")
    print(f"  wire: {codes_tr.shape[1]} symbols x {BITS} bits = "
          f"{codec.wire_bytes():.0f} bytes/vector (float input was {Xtr.shape[1] * 4} bytes)")
    print(f"  reconstruction cosine {cos:.3f}; kernel score correlation {corr:.3f}")

    # --- the lane side: AATs -> output AATs -----------------------------------
    clf = Classifier(num_lanes=10, num_spaces=codec.k, num_channels=codec.S,
                     N=1, init="medium", seed=SEED)
    print(f"lane bank: {clf.extra_repr()}")
    g = torch.Generator().manual_seed(SEED)
    for epoch in range(EPOCHS):
        order = torch.randperm(codes_tr.shape[0], generator=g)
        for i in order:                                # streaming adapt, B = 1
            clf.adapt(codes_tr[i], torch.tensor([ytr[i]]))
        pred = clf.read(codes_tr)[..., 0]
        acc = (pred.numpy() == ytr).mean()
        print(f"  epoch {epoch + 1:2d}: train accuracy {acc:.3f}")

    pred = clf.read(codes_te)[..., 0]                  # one batched read over the test set
    acc = (pred.numpy() == yte).mean()
    print(f"test accuracy {acc:.3f} on {len(yte)} held-out digits, "
          f"read entirely from {BITS}-bit symbol codes")


if __name__ == "__main__":
    main()
