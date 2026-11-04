#!/usr/bin/env python3
"""
chip_ppl.py -- perplexity MEASURED ON THE SIMULATED CHIP, next to the same weights in PyTorch.

Runs the first N WikiText-2 test tokens through the chip's prefill (every 16-row tile is scored: the
final hidden rows come back via generate()'s prefill_hook, and the trailing norm + lm_head run on the
host as they do for generation), then the same tokens through Hugging Face with identical weights,
and prints both negative log-likelihoods. The per-token NLLs should agree to ~1e-3; the point is
that the +x% perplexity quantization study numbers hold on the device, not just in PyTorch.

  python toolchain/chip_ppl.py --tokens 128 [--int8-weights | --int4-weights NPZ | --bf16-weights]

Needs torch + transformers (the scratchpad hfenv) for the reference half; the chip half is done in
the same process via generate.py, so run it from an env that has both tvm and torch, or use
--chip-only / --ref-only with --save/--load to split them.
"""
import argparse, glob, json, math, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)


def wikitext_ids(tok, n):
    import pandas as pd
    f = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet"))[0]
    text = "\n\n".join(pd.read_parquet(f)["text"].tolist())
    return tok.encode(text).ids[:n]


def chip_nll(a, ids):
    import generate as G
    import checkpoint_loader
    model_dir = a.model or G._find_checkpoint()
    _, _, _, lm_head, final_norm = checkpoint_loader.load_qwen3_checkpoint(model_dir, n_layers=1)
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    eps = cfg["rms_norm_eps"]
    rows = []
    kw = {}
    if a.int4_weights:
        z = np.load(a.int4_weights)
        quant = {}
        for key in z.files:
            if key.endswith(".q"):
                li, name = key.split(".")[:2]
                quant[(int(li), name)] = (z[key], z[key[:-2] + ".s"])
        kw = dict(int4_group=int(z["group"]), int4_quant=quant)
    G.generate(model_dir, "", 0, sim=os.path.join(ROOT, "build", "rtv_sim"),
               hal=os.path.join(_HERE, "hal_config.json"), bf16_weights=a.bf16_weights,
               int8_weights=a.int8_weights, use_graph=True, prompt_ids=ids,
               prefill_hook=lambda c, r: rows.append(r), **kw)
    h = np.concatenate(rows).astype(np.float64)                  # [N, hidden]
    n = h / np.sqrt((h ** 2).mean(-1, keepdims=True) + eps) * final_norm.astype(np.float64)
    lg = n @ lm_head.T.astype(np.float64)                        # [N, vocab]
    lg = lg[:-1]
    lse = np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1)) + lg.max(-1)
    return lse - lg[np.arange(len(ids) - 1), ids[1:]]


def ref_nll(a, ids):
    import torch
    from transformers import AutoModelForCausalLM
    from hf_compare import find_checkpoint
    model_dir = a.model or find_checkpoint()
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    sd = m.model.layers
    names = dict(wq="self_attn.q_proj", wk="self_attn.k_proj", wv="self_attn.v_proj",
                 wo="self_attn.o_proj", wgate="mlp.gate_proj", wup="mlp.up_proj", wdown="mlp.down_proj")
    with torch.no_grad():
        if a.int4_weights:
            z = np.load(a.int4_weights)
            g = int(z["group"])
            for li, layer in enumerate(sd):
                for k, path in names.items():
                    q, s = z[f"{li}.{k}.q"].astype(np.float64), z[f"{li}.{k}.s"].astype(np.float64)
                    w = (q * np.repeat(s, g, axis=0)).T          # [out, in]
                    mod = layer.get_submodule(path)
                    mod.weight.copy_(torch.tensor(w, dtype=torch.float32))
        elif a.int8_weights or a.bf16_weights:
            for layer in sd:
                for path in names.values():
                    w = layer.get_submodule(path).weight
                    if a.int8_weights:
                        s = (w.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-12)
                        w.copy_(torch.clamp(torch.round(w / s), -128, 127) * s)
                    else:
                        w.copy_(w.bfloat16().float())
        lg = m(torch.tensor([ids])).logits[0, :-1]
        return torch.nn.functional.cross_entropy(lg, torch.tensor(ids[1:]), reduction="none").numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--bf16-weights", action="store_true")
    ap.add_argument("--int8-weights", action="store_true")
    ap.add_argument("--int4-weights", metavar="NPZ", default=None)
    ap.add_argument("--chip-only", metavar="OUT.npy", default=None)
    ap.add_argument("--ref-only", metavar="CHIP.npy", default=None,
                    help="compare against a chip NLL file written by --chip-only")
    ap.add_argument("--ids-file", default=None,
                    help="JSON list of token ids; written on first use (from a pandas-capable env) "
                         "and read by the TVM venv, which has no parquet reader")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    from hf_compare import find_checkpoint
    tok = Tokenizer.from_file(os.path.join(a.model or find_checkpoint(), "tokenizer.json"))
    if a.ids_file and os.path.exists(a.ids_file):
        ids = json.load(open(a.ids_file))[:a.tokens]
    else:
        ids = wikitext_ids(tok, a.tokens)          # needs pandas + pyarrow (not in the TVM venv)
        if a.ids_file:
            json.dump(ids, open(a.ids_file, "w"))
    if a.ref_only is None:
        chip = chip_nll(a, ids)
        if a.chip_only:
            np.save(a.chip_only, chip)
            print(f"chip NLL saved ({len(chip)} tokens), ppl {math.exp(chip.mean()):.3f}")
            return 0
    else:
        chip = np.load(a.ref_only)
    ref = ref_nll(a, ids)
    print(f"tokens {len(ids) - 1}")
    print(f"chip ppl {math.exp(chip.mean()):.4f}   torch ppl {math.exp(ref.mean()):.4f}   "
          f"max |dNLL| {np.abs(chip - ref).max():.2e}   mean |dNLL| {np.abs(chip - ref).mean():.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
