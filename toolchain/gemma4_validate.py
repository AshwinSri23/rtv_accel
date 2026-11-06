#!/usr/bin/env python3
"""gemma4_validate.py -- validate the real Gemma-4-26B-A4B on the chip against the layer-by-layer Hugging Face reference.

Two checks, both teacher-forced on the SAME token sequence so that one early disagreement cannot cascade:
  chat:  a few chat prompts, the chip generates a reply; the reference is then run over prompt+reply and its top-5 /
         logits are compared with the chip's at every generated step.
  ppl:   perplexity of a fixed text chunk (wikitext-2 test) on the chip vs the reference.

    .venv-tvm/bin/python toolchain/gemma4_validate.py --stage chip  --out DIR   # needs the chip env
    (then)  toolchain/gemma4_gguf_reference.py --all-positions ...              # run by --stage ref via the torch env
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

GGUF = os.path.expanduser("~/models/gguf/gemma-4-26B_q4_0-it.gguf")
TORCH_PY = "/opt/ai-inference/venv-ov/bin/python"
PROMPTS = ["What is the capital of France?", "Write one sentence about the sea.", "What is 12 times 12?"]


def tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)


def wikitext_ids(path):
    """Token ids of a wikitext-2 test chunk (bos + 47 tokens), pre-tokenised with the torch env (the TVM venv has no
    parquet reader); see --wiki-ids."""
    return json.load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=GGUF)
    ap.add_argument("--tokenizer", default=os.path.expanduser("~/models/hf/gemma-4-12B-it-assistant"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["chip", "ref", "compare"], required=True)
    ap.add_argument("--window", type=int, default=None, help="override the sliding window (chip and reference)")
    ap.add_argument("--embed-int8", action="store_true", help="INT8-per-row embedding / lm_head table on the host")
    ap.add_argument("--only-ppl", action="store_true", help="skip the chat prompts")
    ap.add_argument("--reply-tokens", type=int, default=8)
    ap.add_argument("--wiki-ids", required=True, help="JSON list of token ids for the perplexity text")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = tokenizer(a.tokenizer)
    seqs = {}
    if a.stage == "chip":
        import gemma4_model
        from gemma4_gguf import load_gemma4_gguf
        sim, hal = os.path.join(ROOT, "build", "rtv_sim"), os.path.join(_HERE, "hal_config.json")
        src = load_gemma4_gguf(a.gguf, embed_int8=a.embed_int8)
        if a.window:
            src = (dict(src[0], sliding_window=a.window),) + tuple(src[1:])
        chip = {}
        for i, p in enumerate([] if a.only_ppl else PROMPTS):
            ids = [2] + tok(f"<|turn>user\n{p}<turn|>\n<|turn>model\n", add_special_tokens=False)["input_ids"]
            full, steps = gemma4_model.generate_gemma4(src, ids, a.reply_tokens, sim, hal, int4_group=32,
                                                       image_dir=os.path.expanduser("~/.cache/rtv_accel"),
                                                       stop_ids=(1, 106))
            chip[f"chat{i}"] = dict(ids=full, n_prompt=len(ids), steps=steps)
            print(f"chat{i}: {tok.decode(full[len(ids):])!r}", flush=True)
            json.dump(chip, open(os.path.join(a.out, "chip.json"), "w"))
        ids = wikitext_ids(a.wiki_ids)
        sc = []
        gemma4_model.generate_gemma4(src, ids, 0, sim, hal, int4_group=32, score_out=sc,
                                     image_dir=os.path.expanduser("~/.cache/rtv_accel"))
        chip["ppl"] = dict(ids=ids, scores=sc)
        json.dump(chip, open(os.path.join(a.out, "chip.json"), "w"))
        print("chip ppl %.4f over %d tokens" % (np.exp(-np.mean([s[2] for s in sc])), len(sc)))
    elif a.stage == "ref":
        chip = json.load(open(os.path.join(a.out, "chip.json")))
        for name, c in chip.items():
            subprocess.run([TORCH_PY, os.path.join(_HERE, "gemma4_gguf_reference.py"), "--gguf", a.gguf,
                            "--prompt-ids", ",".join(map(str, c["ids"])), "--out", os.path.join(a.out, f"ref_{name}.npz"),
                            "--all-positions"] + (["--sliding-window", str(a.window)] if a.window else []), check=True)
    else:
        chip = json.load(open(os.path.join(a.out, "chip.json")))
        for name, c in chip.items():
            r = np.load(os.path.join(a.out, f"ref_{name}.npz"))
            if name == "ppl":
                lp_chip = np.array([s[2] for s in c["scores"]])
                lp_ref = r["pos_logprob"][:len(lp_chip)]
                print(f"ppl: chip {np.exp(-lp_chip.mean()):.4f}  reference {np.exp(-lp_ref.mean()):.4f}  "
                      f"max |dlogp| {np.abs(lp_chip - lp_ref).max():.2e}  tokens {len(lp_chip)}")
            else:
                n0 = c["n_prompt"]
                agree, worst = 0, 0.0
                for i, (t5, l5) in enumerate(c["steps"]):
                    pos = n0 - 1 + i
                    agree += t5 == r["pos_top_ids"][pos].tolist()
                    worst = max(worst, float(np.abs(np.array(l5) - r["pos_top_logits"][pos]).max()))
                print(f"{name}: {len(c['steps'])} steps, top-5 ids identical {agree}/{len(c['steps'])}, "
                      f"max |dlogit| {worst:.2e}  reply {tok.decode(c['ids'][n0:])!r}")


if __name__ == "__main__":
    main()
