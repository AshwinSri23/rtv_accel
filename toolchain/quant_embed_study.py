#!/usr/bin/env python3
"""
quant_embed_study.py -- does quantizing the tied embedding / lm_head table (151936 x 1024, 622 MB in
FP32, the biggest single tensor in Qwen3-0.6B) cost quality? quant_study.py leaves it FP32.

Fake-quantizes the table per vocab row (= per output channel of lm_head; groups of G along the hidden
dim for INT4) and measures WikiText-2 test perplexity and greedy-token agreement. Two placements:
  tied     the embedding lookup AND lm_head both read the quantized table (one table in DRAM)
  head     only lm_head is quantized (the lookup keeps the FP32 rows)
Layers stay FP32 unless --layers int8 (then they are per-channel INT8) to check errors do not compound.

    python quant_embed_study.py [--layers fp32|int8] [--ppl-tokens 2048]
Needs torch + transformers (not in ctest).
"""
import argparse, glob, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_study import quantize


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="fp32")
    ap.add_argument("--ppl-tokens", type=int, default=2048)
    ap.add_argument("--schemes", default="fp32,bf16,int8,int4g128,int4g32")
    a = ap.parse_args()
    d = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*")))[0]
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
    f = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet"))[0]
    ids = tok("\n\n".join(pd.read_parquet(f)["text"].tolist()), return_tensors="pt").input_ids[:, :a.ppl_tokens]
    prompts = ["The capital of France is", "def fibonacci(n):\n    ",
               "In 1969, the first humans landed on the Moon. The mission was", "Water boils at a temperature of"]
    table = model.model.embed_tokens.weight.detach().clone()
    layer_w = {n: p.detach().clone() for n, p in model.named_parameters()
               if ".layers." in n and n.endswith("_proj.weight")}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in layer_w:
                p.copy_(quantize(layer_w[n], a.layers))
    # untie so lm_head can differ from the lookup
    model.lm_head.weight = torch.nn.Parameter(table.clone())
    emb, head = model.model.embed_tokens.weight, model.lm_head.weight

    def ppl():
        nll, cnt = 0.0, 0
        for i in range(0, ids.shape[1] - 1, 512):
            win = ids[:, i:i + 513]
            if win.shape[1] < 2:
                break
            with torch.no_grad():
                lg = model(win[:, :-1]).logits
            nll += torch.nn.functional.cross_entropy(lg[0], win[0, 1:], reduction="sum").item()
            cnt += win.shape[1] - 1
        return math.exp(nll / cnt)

    def greedy(p):
        x = tok(p, return_tensors="pt").input_ids
        with torch.no_grad():
            return model.generate(x, max_new_tokens=16, do_sample=False)[0].tolist()

    ref = None
    print(f"layers={a.layers}  table {tuple(table.shape)}")
    print(f"{'scheme':9} {'MB':>7} {'tied ppl':>9} {'head-only ppl':>14} {'tokens=ref (tied)':>18}")
    for s in a.schemes.split(","):
        q = quantize(table, s)
        bits = {"fp32": 32, "bf16": 16}.get(s) or int(s[3])
        import re
        g = re.search(r"g(\d+)", s)
        eff = bits + (32 / int(g.group(1)) if g and bits < 16 else (32 / table.shape[1] if bits == 8 else 0))
        mb = table.numel() * eff / 8 / 1e6
        with torch.no_grad():
            emb.copy_(q); head.copy_(q)
            p_tied = ppl()
            outs = [greedy(p) for p in prompts]
            emb.copy_(table)
            p_head = ppl()
        if ref is None:
            ref = outs
        match = sum(sum(x == y for x, y in zip(o, r)) for o, r in zip(outs, ref))
        total = sum(len(r) for r in ref)
        print(f"{s:9} {mb:7.0f} {p_tied:9.3f} {p_head:14.3f} {match:>12}/{total}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
