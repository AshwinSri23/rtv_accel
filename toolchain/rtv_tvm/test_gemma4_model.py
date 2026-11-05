#!/usr/bin/env python3
"""
test_gemma4_model.py -- a whole (tiny, random) Gemma-4 text model on the simulated chip vs Hugging Face.

3 layers (sliding, sliding, full), a dense GELU MLP in parallel with 8 routed experts (top-2) in every layer,
sandwich norms, K=V and head_dim 32 on the full-attention layer (partial "proportional" RoPE), randomised
norm gains / router scales / per-expert scales / layer scalars, final logit softcap, tied embeddings.
Compares generated ids AND the top-5 logits at every step with fixtures/gemma4_tiny_hf_reference.json
(made by make_gemma4_fixture.py). Needs no download.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_gemma4_model.py
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(ROOT, "toolchain"))

import gemma4_model  # noqa: E402


def main():
    ref = json.load(open(os.path.join(_HERE, "fixtures", "gemma4_tiny_hf_reference.json")))
    ids, steps = gemma4_model.generate_gemma4(
        os.path.join(_HERE, "fixtures", "gemma4_tiny_model"), ref["prompt"], len(ref["steps"]),
        os.path.join(ROOT, "build", "rtv_sim"), os.path.join(ROOT, "toolchain", "hal_config.json"))
    worst, ok_top = 0.0, True
    for st, r in zip(steps, ref["steps"]):
        ok_top &= st[0] == r["top_ids"]
        worst = max(worst, max(abs(a - b) for a, b in zip(st[1], r["top_logits"])))
    results = [("generated_ids_match_hf", ids == ref["ids"], f"chip {ids[len(ref['prompt']):]} vs hf {ref['ids'][len(ref['prompt']):]}"),
               ("top5_ids_match_every_step", ok_top, ""),
               ("top5_logits_within_1e-4", worst < 1e-4, f"max |delta| {worst:.2e}")]
    n = 0
    for name, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {msg}" if msg else ""))
        n += ok
    print(f"\n{n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
