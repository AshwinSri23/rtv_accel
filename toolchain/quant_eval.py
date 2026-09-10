#!/usr/bin/env python3
"""
quant_eval.py — evaluate a quantisation configuration against the float model.

A single prompt is not evidence. "The capital of France is" reached rank 0
under a configuration that agrees with the float model on only 5 of 8 prompts,
so every headline number in this project's history should be read as
"on one prompt" unless it came from here.

Metrics, both against ref_model as ground truth:
  * top-1 agreement -- how often the quantised model picks the same token
  * mean rank of the float model's choice -- a graded score, so a
    configuration that is close everywhere scores better than one that is
    perfect on half the prompts and lost on the rest

Usage:
    python3 toolchain/quant_eval.py                  # default sweep
    python3 toolchain/quant_eval.py --w 8 --a 16 --r 16 --per-channel
"""

import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import scale_sim as SS                              # noqa: E402
from scale_sim import ScaleQwen3                    # noqa: E402
from hf_loader import SafeTensorsModel              # noqa: E402
from ref_model import RefQwen3                      # noqa: E402

# Deliberately mixed: factual recall, arithmetic, code, and open-ended
# continuations. Quantisation error is not uniform across prompt types, and a
# set of near-identical prompts would hide that.
PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "Water boils at a temperature of",
    "The largest planet in our solar system is",
    "def add(a, b):\n    return",
    "One plus one equals",
    "The color of the sky on a clear day is",
    "Shakespeare wrote a play called Romeo and",
]

# (weight bits, activation bits, residual bits, per-channel residual)
SWEEP = [
    ("as designed        W8 A8  R8",  (8, 8, 8, False)),
    ("wider residual     W8 A8  R16", (8, 8, 16, False)),
    ("per-channel resid  W8 A8  R16", (8, 8, 16, True)),
    ("+ wider acts       W8 A12 R16", (8, 12, 16, True)),
    ("+ wider acts       W8 A16 R16", (8, 16, 16, True)),
]


def load(model_id="Qwen/Qwen3-0.6B"):
    path = model_id
    if not os.path.exists(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(model_id, allow_patterns=[
            "*.safetensors", "config.json", "tokenizer*", "vocab*", "merges*"])
    from transformers import AutoTokenizer
    return SafeTensorsModel(path), AutoTokenizer.from_pretrained(path)


def evaluate(model, tok, ref_targets, w, a, r, per_channel):
    SS.FLOAT_MODE = False
    SS.WEIGHT_BITS, SS.ACT_BITS = w, a
    SS.RESIDUAL_BITS, SS.RESIDUAL_PER_CHANNEL = r, per_channel
    ok, ranks = 0, []
    for ids, target in ref_targets:
        order = np.argsort(ScaleQwen3(model).logits(ids))[::-1]
        rank = int(np.where(order == target)[0][0])
        ranks.append(rank)
        ok += (rank == 0)
    return ok, float(np.mean(ranks))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    for f, d in (("w", None), ("a", None), ("r", None)):
        ap.add_argument(f"--{f}", type=int, default=d)
    ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="check FLOAT_MODE reproduces ref_model, then exit")
    args = ap.parse_args()

    model, tok = load(args.model)
    ref = RefQwen3(model)

    if args.validate:
        # The gate that must pass before any measurement is quoted: with all
        # quantisation bypassed, scale_sim must BE ref_model.
        SS.FLOAT_MODE = True
        ids = tok(PROMPTS[0]).input_ids
        rs, ss = ref.forward(ids), ScaleQwen3(model).forward(ids)
        worst = min(
            float(np.dot(x.ravel(), y.value.ravel())
                  / (np.linalg.norm(x) * np.linalg.norm(y.value) + 1e-30))
            for x, y in zip(rs, ss))
        print(f"FLOAT_MODE worst per-layer cosine vs ref_model: {worst:.6f}")
        print("PASS" if worst > 0.9999 else "FAIL — structural bug in scale_sim")
        return 0 if worst > 0.9999 else 1

    targets = []
    for p in PROMPTS:
        ids = tok(p).input_ids
        targets.append((ids, int(np.argmax(ref.logits(ids)[-1]))))

    if args.w:
        ok, mr = evaluate(model, tok, targets, args.w, args.a or 8,
                          args.r or 8, args.per_channel)
        print(f"top-1 agreement {ok}/{len(PROMPTS)}   mean rank {mr:.2f}")
        return 0

    print(f"{'configuration':36s} {'agree':>7}  {'mean rank':>9}")
    for label, (w, a, r, pc) in SWEEP:
        ok, mr = evaluate(model, tok, targets, w, a, r, pc)
        print(f"{label:36s} {ok:>5}/{len(PROMPTS)}  {mr:>9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
