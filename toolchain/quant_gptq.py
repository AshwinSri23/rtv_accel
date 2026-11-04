#!/usr/bin/env python3
"""
quant_gptq.py -- GPTQ-quantize Qwen3 dense decoder weights to symmetric INT4 and write the
integer codes + group scales in the chip's layout, for generate.py --int4-weights.

    python quant_gptq.py --out qwen3_int4_g32.npz --group 32

Output npz keys "<layer>.<chip name>.q" (int8 codes in [-8, 7], shape [K, N]) and
"<layer>.<chip name>.s" (FP32 scales, shape [K/group, N]); chip names are wq wk wv wo wgate wup
wdown. Also stores the group size. Needs torch + transformers (not a project dependency).
Calibration: WikiText-2 train, --calib-seqs x 512 tokens, Hessians from the FP32 model.
"""
import argparse, glob, os, re, sys
import numpy as np

HF2CHIP = {"self_attn.q_proj": "wq", "self_attn.k_proj": "wk", "self_attn.v_proj": "wv",
           "self_attn.o_proj": "wo", "mlp.gate_proj": "wgate", "mlp.up_proj": "wup",
           "mlp.down_proj": "wdown"}


def main():
    import torch
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from quant_study import gptq
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--calib-seqs", type=int, default=32)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    model_dir = a.model or sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*")))[0]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    targets = [n for n, _ in model.named_parameters()
               if re.search(r"layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight", n)]
    f = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/train-*.parquet"))[0]
    cal = tok("\n\n".join(pd.read_parquet(f)["text"].tolist()[:20000]), return_tensors="pt").input_ids
    mods, hess, cnt = dict(model.named_modules()), {}, {}

    def mk(name):
        def hook(_m, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).double()
            h = hess.setdefault(name, torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64))
            h += 2 * x.T @ x
            cnt[name] = cnt.get(name, 0) + x.shape[0]
        return hook
    hs = [mods[n[:-len(".weight")]].register_forward_pre_hook(mk(n)) for n in targets]
    with torch.no_grad():
        for i in range(a.calib_seqs):
            model(cal[:, i * 512:(i + 1) * 512])
    for h in hs:
        h.remove()
    out = {"group": np.array(a.group)}
    params = dict(model.named_parameters())
    for n in targets:
        m = re.search(r"layers\.(\d+)\.(.*)\.weight", n)
        li, chip = int(m.group(1)), HF2CHIP[m.group(2)]
        gptq(params[n].detach(), hess[n] / cnt[n], 4, a.group)
        codes, scales = gptq.last                       # [out, in], [out, in/group]
        out[f"{li}.{chip}.q"] = np.ascontiguousarray(codes.numpy().T)        # chip layout [K=in, N=out]
        out[f"{li}.{chip}.s"] = np.ascontiguousarray(scales.numpy().T.astype(np.float32))  # [K/group, N]
        print(f"layer {li} {chip}", flush=True)
    np.savez(a.out, **out)
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
