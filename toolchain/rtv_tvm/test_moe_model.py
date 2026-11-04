#!/usr/bin/env python3
"""
test_moe_model.py -- a whole Qwen3-MoE model on the simulated chip vs Hugging Face.

Runs hf-internal-testing/tiny-random-Qwen3MoeForCausalLM (a real
Qwen3MoeForCausalLM: 2 layers, 4 experts, top-2, norm_topk_prob, q_norm/k_norm,
GQA) through toolchain/moe_model.py (host-segmented routing on a persistent
rtv_sim session) and compares with the recorded Hugging Face reference in
fixtures/moe_tiny_hf_reference.json: the generated token ids AND the top-5 logits at
every step (the model's logits are nearly tied, so this is a tight check).

Skips (exit 0) when the checkpoint is not in the local Hugging Face cache:
    hf download hf-internal-testing/tiny-random-Qwen3MoeForCausalLM

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_moe_model.py
"""
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(ROOT, "toolchain"))

import moe_model  # noqa: E402


def main():
    hits = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--hf-internal-testing--tiny-random-Qwen3MoeForCausalLM/snapshots/*")))
    if not hits:
        print("SKIP: tiny Qwen3-MoE checkpoint not in the local HF cache")
        return 0
    ref = json.load(open(os.path.join(_HERE, "fixtures", "moe_tiny_hf_reference.json")))
    prompt = ref["prompt"]
    n_new = len(ref["steps"])
    ids, steps = moe_model.generate_moe(
        hits[0], prompt, n_new, os.path.join(ROOT, "build", "rtv_sim"),
        os.path.join(ROOT, "toolchain", "hal_config.json"))
    ok_ids = ids == ref["ids"]
    worst = 0.0
    ok_top = True
    for s, r in zip(steps, ref["steps"]):
        if s[0] != r["top_ids"]:
            ok_top = False
        worst = max(worst, max(abs(a - b) for a, b in zip(s[1], r["top_logits"])))
    results = [("generated_ids_match_hf", ok_ids, f"chip {ids[len(prompt):]} vs hf {ref['ids'][len(prompt):]}"),
               ("top5_ids_match_every_step", ok_top, ""),
               ("top5_logits_within_2e-3", worst < 2e-3, f"max |delta| {worst:.2e}")]
    n = 0
    for name, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {msg}" if msg else ""))
        n += ok
    print(f"\n{n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
