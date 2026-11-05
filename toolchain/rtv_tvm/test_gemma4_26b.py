#!/usr/bin/env python3
"""
test_gemma4_26b.py -- the REAL Gemma-4-26B-A4B (30 layers, 128 experts, top-8, Q4_0 -> INT4W g32) on the chip.

Compares against fixtures/gemma4_26b_a4b_q4_0_reference.npz, made by toolchain/gemma4_gguf_reference.py: Hugging
Face's own Gemma4TextDecoderLayer applied one layer at a time to the DEQUANTISED Q4_0 weights (the model does not fit
in RAM at once). Checks the hidden state after every layer at the last prompt position (relative error) and the
final top-5 ids / logits. Takes ~3 minutes and ~24 GB of RAM (16 GB DRAM image in the simulator), so it only runs
when RTV_REAL_GEMMA4=1 and the GGUF exists; otherwise it skips (exit 0), like the other checkpoint tests.

    RTV_REAL_GEMMA4=1 .venv-tvm/bin/python toolchain/rtv_tvm/test_gemma4_26b.py [path/to/gemma-4-26B_q4_0-it.gguf]
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(ROOT, "toolchain"))

PROMPT = [2, 105, 2364, 236764]
DEFAULT = os.path.expanduser("~/models/gguf/gemma-4-26B_q4_0-it.gguf")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.environ.get("RTV_REAL_GEMMA4") or not os.path.exists(path):
        print("SKIP  real Gemma-4-26B (set RTV_REAL_GEMMA4=1 and provide the GGUF)")
        return 0
    import gemma4_model
    from gemma4_gguf import load_gemma4_gguf
    ref = np.load(os.path.join(_HERE, "fixtures", "gemma4_26b_a4b_q4_0_reference.npz"))
    trace = []
    ids, steps = gemma4_model.generate_gemma4(
        load_gemma4_gguf(path), PROMPT, 1, os.path.join(ROOT, "build", "rtv_sim"),
        os.path.join(ROOT, "toolchain", "hal_config.json"), int4_group=32, trace=trace,
        image_dir=os.path.expanduser("~/.cache/rtv_accel"))
    err = [float(np.linalg.norm(t["hidden"] - ref["hidden"][i]) / np.linalg.norm(ref["hidden"][i]))
           for i, t in enumerate(trace)]
    dl = max(abs(a - b) for a, b in zip(steps[0][1], ref["top_logits"]))
    results = [("30 layers traced", len(trace) == 30, ""),
               ("every layer within 1e-4 of Hugging Face", max(err) < 1e-4, f"worst {max(err):.2e}"),
               ("top-5 ids match", steps[0][0] == ref["top_ids"].tolist(), f"{steps[0][0]}"),
               ("top-5 logits within 1e-2", dl < 1e-2, f"max |delta| {dl:.2e}")]
    n = 0
    for name, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {msg}" if msg else ""))
        n += ok
    print(f"\n{n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
