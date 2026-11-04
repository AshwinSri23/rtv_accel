#!/usr/bin/env python3
"""
quant_study.py -- weight-only quantization accuracy study (no chip, no ISA change).

Fake-quantizes the decoder-layer linear weights of Qwen3 in PyTorch (quantize,
then dequantize back to FP32) and measures how far the model drifts from the
FP32 reference: greedy-token agreement on a few prompts, and perplexity on a
WikiText-2 slice. Embeddings / lm_head / norms stay FP32. This decides which
scheme (bits, per-channel vs per-group) is worth giving an ISA dtype.

Schemes: fp32, bf16, int8 (per output channel), int8g<N> / int4g<N>
(symmetric, groups of N along K), int4g<N>a (asymmetric, zero-point).

    python quant_study.py --schemes fp32,bf16,int8,int4g128,int4g32
Needs torch + transformers (not a project dependency; not in ctest).
"""
import argparse, glob, math, os, re, sys


def quantize(w, scheme):
    import torch
    if scheme == "fp32":
        return w
    if scheme == "bf16":
        return w.to(torch.bfloat16).to(torch.float32)
    m = re.fullmatch(r"int(8|4)(?:g(\d+))?(a?)", scheme)
    if not m:
        raise SystemExit(f"unknown scheme {scheme}")
    bits, g, asym = int(m.group(1)), m.group(2), bool(m.group(3))
    n, k = w.shape                                  # [out, in]; groups run along K
    g = int(g) if g else k
    assert k % g == 0
    x = w.reshape(n, k // g, g)
    if asym:
        lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
        qmax = 2 ** bits - 1
        scale = ((hi - lo) / qmax).clamp_min(1e-12)
        zp = torch.round(-lo / scale)
        q = torch.clamp(torch.round(x / scale) + zp, 0, qmax)
        return ((q - zp) * scale).reshape(n, k)
    qmax = 2 ** (bits - 1) - 1
    scale = (x.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return (q * scale).reshape(n, k)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", default="fp32,bf16,int8,int4g128,int4g32")
    ap.add_argument("--model", default=None)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--ppl-tokens", type=int, default=2048)
    ap.add_argument("--only", default="", help="regex of weight names to quantize (default: all layer linears)")
    a = ap.parse_args()
    model_dir = a.model or sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*")))[0]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    orig = {n: p.detach().clone() for n, p in model.named_parameters()}
    targets = [n for n in orig if re.search(r"layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight", n)
               and re.search(a.only, n)]
    prompts = ["The capital of France is", "def fibonacci(n):\n    ",
               "In 1969, the first humans landed on the Moon. The mission was",
               "Water boils at a temperature of"]
    text = ""
    try:
        import pandas as pd
        f = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet"))[0]
        text = "\n\n".join(pd.read_parquet(f)["text"].tolist())
    except Exception as e:  # noqa: BLE001
        print(f"(no wikitext: {e})")
    ids_ppl = tok(text, return_tensors="pt").input_ids[:, :a.ppl_tokens] if text else None

    def greedy(p):
        ids = tok(p, return_tensors="pt").input_ids
        with torch.no_grad():
            return model.generate(ids, max_new_tokens=a.steps, do_sample=False)[0].tolist()

    def ppl():
        if ids_ppl is None:
            return float("nan")
        nll, cnt = 0.0, 0
        for i in range(0, ids_ppl.shape[1] - 1, 512):          # 512-token windows
            win = ids_ppl[:, i:i + 513]
            if win.shape[1] < 2:
                break
            with torch.no_grad():
                lg = model(win[:, :-1]).logits
            nll += torch.nn.functional.cross_entropy(lg[0], win[0, 1:], reduction="sum").item()
            cnt += win.shape[1] - 1
        return math.exp(nll / cnt)

    ref = None
    print(f"{'scheme':10} {'bits/w':>7} {'ppl':>8} {'tokens matching FP32':>22}")
    for s in a.schemes.split(","):
        params = dict(model.named_parameters())
        with torch.no_grad():
            for n in targets:
                params[n].copy_(quantize(orig[n], s))
        outs = [greedy(p) for p in prompts]
        if ref is None:
            ref = outs
        match = sum(sum(x == y for x, y in zip(o, r)) for o, r in zip(outs, ref))
        total = sum(len(r) for r in ref)
        bits = {"fp32": 32, "bf16": 16}.get(s) or int(re.match(r"int(\d)", s).group(1))
        g = re.search(r"g(\d+)", s)
        eff = bits + (32 / int(g.group(1)) if g and bits < 16 else 0) + (bits / int(g.group(1)) if s.endswith("a") and g else 0)
        print(f"{s:10} {eff:7.2f} {ppl():8.3f} {match:>10}/{total}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
