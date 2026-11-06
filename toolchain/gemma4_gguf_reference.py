#!/usr/bin/env python3
"""gemma4_gguf_reference.py -- Hugging Face reference for the real Gemma-4-26B-A4B GGUF, one layer at a time.

The full model does not fit in RAM, so each Gemma4TextDecoderLayer is built from the DEQUANTISED Q4_0 weights
(exactly what the chip's INT4W g32 computes), applied to the whole prompt, then freed. Writes the per-layer hidden
state at the last prompt position plus the final top-5 logits to an .npz. Needs torch + transformers (not the
.venv-tvm env):

    /opt/ai-inference/venv-ov/bin/python toolchain/gemma4_gguf_reference.py --gguf <file> --prompt-ids 2,105,2364 \
        --out ref.npz [--layers N]
"""
import argparse
import os
import sys

import numpy as np
import torch
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemma4_gguf import load_gemma4_gguf, hf_layer_tensors  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prompt-ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--sliding-window", type=int, default=None,
                    help="override the sliding window (to exercise window masking with a short sequence)")
    ap.add_argument("--router-probs", action="store_true", help="save every layer's router probabilities [layers, T, E]")
    ap.add_argument("--all-positions", action="store_true",
                    help="also score every position: top-5 and the log-prob of the next token (teacher forcing)")
    a = ap.parse_args()
    ids = [int(x) for x in a.prompt_ids.split(",")]
    cfg_d, layers, embed, final_norm = load_gemma4_gguf(a.gguf, a.layers)
    if a.sliding_window:
        cfg_d["sliding_window"] = a.sliding_window
    cfg = Gemma4TextConfig(
        vocab_size=cfg_d["vocab_size"], hidden_size=cfg_d["hidden_size"], intermediate_size=cfg_d["intermediate_size"],
        moe_intermediate_size=cfg_d["moe_intermediate_size"], num_hidden_layers=cfg_d["num_hidden_layers"],
        num_attention_heads=cfg_d["num_attention_heads"], num_key_value_heads=cfg_d["num_key_value_heads"],
        head_dim=cfg_d["head_dim"], global_head_dim=cfg_d["global_head_dim"],
        num_global_key_value_heads=cfg_d["num_global_key_value_heads"], attention_k_eq_v=True,
        enable_moe_block=True, num_experts=cfg_d["num_experts"], top_k_experts=cfg_d["top_k_experts"],
        layer_types=cfg_d["layer_types"], sliding_window=cfg_d["sliding_window"], hidden_size_per_layer_input=0,
        final_logit_softcapping=cfg_d["final_logit_softcapping"], max_position_embeddings=4096,
        tie_word_embeddings=True, vocab_size_per_layer_input=cfg_d["vocab_size"], num_kv_shared_layers=0,
        rms_norm_eps=cfg_d["rms_norm_eps"],
        rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": cfg_d["rope_theta_full"],
                                            "rope_type": "proportional"},
                         "sliding_attention": {"rope_theta": cfg_d["rope_theta_sliding"], "rope_type": "default"}})
    cfg._attn_implementation = "eager"
    T = len(ids)
    x = torch.from_numpy(embed[ids] * np.float32(cfg_d["hidden_size"] ** 0.5)).unsqueeze(0)
    pos = torch.arange(T).unsqueeze(0)
    rot = Gemma4TextRotaryEmbedding(cfg)
    mask = torch.full((T, T), float("-inf")).triu(1)[None, None]
    # sliding layers: key j is visible to query i iff j > i - window (as in transformers' sliding_window_overlay)
    qi, kj = torch.arange(T)[:, None], torch.arange(T)[None, :]
    swa_mask = torch.where((kj <= qi) & (kj > qi - cfg_d["sliding_window"]), 0.0, float("-inf"))[None, None]
    out = {"hidden": []}
    gaps = []                                   # per layer: router prob [T, E] captured by a hook
    cap_probs = []
    with torch.no_grad():
        for li, lw in enumerate(layers):
            layer = Gemma4TextDecoderLayer(cfg, li).eval()
            sd = {k: torch.from_numpy(v) for k, v in hf_layer_tensors(lw).items()}
            missing = layer.load_state_dict(sd, strict=False)
            assert not missing.unexpected_keys, missing.unexpected_keys
            assert not [m for m in missing.missing_keys if m != "layer_scalar"], missing.missing_keys
            if a.router_probs:
                layer.router.register_forward_hook(lambda m, i, o: cap_probs.append(o[0].detach().numpy().copy()))
            pe = rot(x, pos, cfg_d["layer_types"][li])
            x = layer(x, shared_kv_states={}, position_embeddings=pe, attention_mask=(swa_mask if cfg_d["layer_types"][li] == "sliding_attention" else mask), position_ids=pos)
            out["hidden"].append(x[0, -1].numpy().copy())
            print(f"layer {li} done  |h|={x[0, -1].norm():.3f}", flush=True)
            del layer, sd
    h = x[0, -1].double()
    n = h / torch.sqrt((h ** 2).mean() + cfg_d["rms_norm_eps"]) * torch.from_numpy(final_norm).double()
    lg = torch.from_numpy(embed).float() @ n.float()
    cap = cfg_d["final_logit_softcapping"]
    if cap:
        lg = torch.tanh(lg / cap) * cap
    top = lg.topk(5)
    extra = {}
    if a.all_positions:
        hs = x[0].double()
        n = hs / torch.sqrt((hs ** 2).mean(-1, keepdim=True) + cfg_d["rms_norm_eps"]) * torch.from_numpy(final_norm).double()
        L = torch.from_numpy(embed).float() @ n.float().T          # [vocab, T]
        L = L.T
        if cap:
            L = torch.tanh(L / cap) * cap
        lp = torch.log_softmax(L.double(), dim=-1)
        t5 = L.topk(5, dim=-1)
        extra = dict(pos_top_ids=t5.indices.numpy(), pos_top_logits=t5.values.numpy(),
                     pos_logprob=lp[torch.arange(T - 1), torch.tensor(ids[1:])].numpy())
    if a.router_probs:
        extra["router_probs"] = np.stack(cap_probs)
    np.savez(a.out, hidden=np.stack(out["hidden"]), top_ids=top.indices.numpy(), top_logits=top.values.numpy(), **extra)
    print("top5", top.indices.tolist(), [round(v, 3) for v in top.values.tolist()])


if __name__ == "__main__":
    main()
