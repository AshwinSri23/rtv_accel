#!/usr/bin/env python3
"""
hf_compare.py -- compare the simulated chip against the real Hugging Face model.

Runs Hugging Face Qwen3-0.6B (greedy, FP32) and toolchain/generate.py (the
simulated chip, via the project's TVM venv) on the same prompt and checks that
they generate the SAME token ids. Exit 0 = identical, 1 = differ, 77 = skipped
(PyTorch/Transformers not available or no local checkpoint).

This is what caught the missing attention 1/sqrt(head_dim) score scale: every
other test compares against numpy written by the same hand that wrote the
compiler, so a shared misunderstanding passes everywhere. It is NOT in `ctest`:
PyTorch is not a project dependency (run it from any Python that has `torch` and
`transformers`), and a 28-layer run takes minutes per prompt.

    python hf_compare.py --prompt "The capital of France is" --steps 4
    python hf_compare.py --prompt "<26+ tokens, exercises multi-tile prefill>" --steps 4
"""
import argparse
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = 77


def find_checkpoint():
    hits = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*")))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=4, help="new tokens to generate")
    ap.add_argument("--chip-python", default=os.path.join(ROOT, ".venv-tvm", "bin", "python"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--chip-args", default="", help="extra args for generate.py, e.g. --bf16-weights")
    ap.add_argument("--ref-int8", action="store_true",
                    help="fake-quantize the HF reference's 7 per-layer weight matrices per output "
                         "channel to INT8 (the chip's --int8-weights scheme), so the comparison "
                         "isolates chip bugs from quantization error")
    ap.add_argument("--ref-int4", metavar="NPZ", default=None,
                    help="overwrite the HF reference's 7 per-layer weight matrices with the dequantised "
                         "INT4 weights from a quant_gptq.py file (the chip's --int4-weights)")
    a = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"SKIP: {e} (needs torch + transformers)")
        return SKIP
    model_dir = a.model or find_checkpoint()
    if not model_dir:
        print("SKIP: no local Qwen3-0.6B checkpoint (hf download Qwen/Qwen3-0.6B)")
        return SKIP

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    if a.ref_int8:
        import re as _re
        with torch.no_grad():
            for n, p in model.named_parameters():
                if _re.search(r"layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight", n):
                    sc = (p.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)   # [out,1] = per output channel
                    p.copy_(torch.clamp(torch.round(p / sc), -128, 127) * sc)
    if a.ref_int4:
        import numpy as _np
        z = _np.load(a.ref_int4)
        g = int(z["group"])
        names = {"wq": "self_attn.q_proj", "wk": "self_attn.k_proj", "wv": "self_attn.v_proj",
                 "wo": "self_attn.o_proj", "wgate": "mlp.gate_proj", "wup": "mlp.up_proj",
                 "wdown": "mlp.down_proj"}
        params = dict(model.named_parameters())
        with torch.no_grad():
            for key in z.files:
                if key.endswith(".q"):
                    li, chip = key.split(".")[:2]
                    deq = z[key].astype(_np.float32) * _np.repeat(z[key[:-2] + ".s"], g, axis=0)  # [K, N]
                    params[f"model.layers.{li}.{names[chip]}.weight"].copy_(torch.from_numpy(deq.T.copy()))
    ids = tok(a.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        hf_ids = model.generate(ids, max_new_tokens=a.steps, do_sample=False)[0].tolist()
    print(f"hf   : {hf_ids}\n       {tok.decode(hf_ids)!r}")

    r = subprocess.run([a.chip_python, os.path.join(ROOT, "toolchain", "generate.py"),
                        "--model", model_dir, "--prompt", a.prompt, "--steps", str(a.steps)] + a.chip_args.split(),
                       capture_output=True, text=True)
    m = re.search(r"^ids\s*:\s*(\[.*\])\s*$", r.stdout, re.M)
    if r.returncode != 0 or not m:
        print("chip run failed:\n" + (r.stdout + r.stderr)[-800:])
        return 1
    chip_ids = eval(m.group(1))  # noqa: S307 -- our own generate.py output, a list literal
    print(f"chip : {chip_ids}")
    if chip_ids == hf_ids:
        print("IDENTICAL")
        return 0
    first = next((i for i, (x, y) in enumerate(zip(chip_ids, hf_ids)) if x != y),
                 min(len(chip_ids), len(hf_ids)))
    print(f"DIFFER at token index {first}: chip {chip_ids[first:first+1]} vs hf {hf_ids[first:first+1]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
