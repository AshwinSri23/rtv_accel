#!/usr/bin/env python3
"""
layer_probe.py — locate where a pipeline first diverges from ground truth.

An end-to-end comparison ("the top token is wrong") gives one bit and says
nothing about where the error entered. This walks the layer stack and reports
per-layer cosine similarity against ref_model, so a gradual decay
(accumulating precision loss) is distinguishable from a cliff at one layer
(a bug in that step).

Cosine, not RMSE, because every INT8 stage carries an arbitrary scale factor;
direction is the part that has to survive, magnitude is bookkeeping.

Usage:
    python3 toolchain/layer_probe.py --mode float      # must be ~1.0 throughout
    python3 toolchain/layer_probe.py --mode w8         # INT8 weights only
    python3 toolchain/layer_probe.py --mode full       # everything INT8
"""

import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from hf_loader import SafeTensorsModel          # noqa: E402
from ref_model import RefQwen3                  # noqa: E402
from quant_sim import QuantSim, Opts            # noqa: E402


MODES = {
    # Everything off: proves the two implementations agree structurally.
    "float": dict(float_mode=True),
    # Only the weights are quantised. Per-channel INT8 reconstructs weights to
    # ~1% relative error, so anything worse than a near-1.0 cosine here is a
    # bug in the pipeline, not quantisation loss.
    "w8":    dict(int8_acts=False, n8=False, add8=False, int8_probs=False,
                  int8_rope=False, fold_gamma=False, gamma_exact=True,
                  w_per_channel=True, corr_exact=True),
    "w8_pt": dict(int8_acts=False, n8=False, add8=False, int8_probs=False,
                  int8_rope=False, fold_gamma=False, gamma_exact=True,
                  w_per_channel=False),
    "acts":  dict(n8=False, add8=False, int8_probs=False, int8_rope=False,
                  fold_gamma=False, gamma_exact=True, w_per_channel=True,
                  corr_exact=True),
    "norm":  dict(add8=False, int8_probs=False, int8_rope=False,
                  fold_gamma=False, gamma_exact=True, w_per_channel=True,
                  corr_exact=True),
    "full":  dict(),
}


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0 if na != nb else 1.0
    return float(a @ b / (na * nb))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--mode", default="w8", choices=sorted(MODES))
    ap.add_argument("--layers", type=int, default=None)
    args = ap.parse_args()

    path = args.model
    if not os.path.exists(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(args.model, allow_patterns=[
            "*.safetensors", "config.json", "tokenizer*", "vocab*", "merges*"])
    model = SafeTensorsModel(path)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    ids = tok(args.prompt).input_ids

    ref = RefQwen3(model)
    sim = QuantSim(model, Opts(**MODES[args.mode]))
    n = args.layers or ref.n_layers

    r_states = ref.forward(ids, upto_layer=n)
    s_states = sim.forward(ids, upto_layer=n)

    print(f"mode={args.mode}  prompt={args.prompt!r}  layers={n}")
    print(f"{'layer':>6}  {'cosine(last pos)':>17}  {'cosine(all pos)':>16}")
    worst, worst_at = 1.0, None
    for i, (r, s) in enumerate(zip(r_states, s_states)):
        c_last = cosine(r[-1], s[-1])
        c_all = cosine(r, s)
        tag = "embed" if i == 0 else str(i - 1)
        print(f"{tag:>6}  {c_last:17.5f}  {c_all:16.5f}")
        if i and c_last < worst:
            worst, worst_at = c_last, tag
    if worst_at is not None:
        print(f"\nlowest per-layer cosine {worst:.5f} at layer {worst_at}")


if __name__ == "__main__":
    main()
