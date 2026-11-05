#!/usr/bin/env python3
"""
make_gemma4_fixture.py -- regenerate rtv_tvm/fixtures/gemma4_tiny_model/ and gemma4_tiny_hf_reference.json:
a small random Gemma-4 text model (3 layers: sliding, sliding, full; dense MLP + 8 experts top-2 on every
layer; K=V and head_dim 32 on the full layer; final logit softcap; tied embeddings) with all norm gains,
router scales, per-expert scales and layer scalars randomised so every Gemma-specific path matters.
Reference = Hugging Face greedy ids + top-5 logits per step. Needs torch + transformers.
"""
import json, os, sys
import torch
from transformers import Gemma4TextConfig, Gemma4ForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "rtv_tvm", "fixtures")
PROMPT, N_NEW = [5, 17, 101, 42, 7], 6


def tiny_config():
    cfg = Gemma4TextConfig(
        vocab_size=256, hidden_size=32, intermediate_size=64, moe_intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        global_head_dim=32, num_global_key_value_heads=1, attention_k_eq_v=True,
        enable_moe_block=True, num_experts=8, top_k_experts=2,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"], sliding_window=64,
        hidden_size_per_layer_input=0, final_logit_softcapping=30.0, max_position_embeddings=128,
        tie_word_embeddings=True, vocab_size_per_layer_input=256, num_kv_shared_layers=0,
        rms_norm_eps=1e-6)
    return cfg


LONG_PROMPT_LEN, LONG_N_NEW = 90, 6      # > the tiny model's sliding_window (64): exercises window masking


def make_long_reference():
    """Greedy ids + top-5 logits of the SAVED tiny model on a prompt longer than its sliding window."""
    d = os.path.join(FIX, "gemma4_tiny_model")
    from safetensors.torch import load_file
    m = Gemma4ForCausalLM(tiny_config()).eval()
    print(m.load_state_dict(load_file(os.path.join(d, "model.safetensors")), strict=False))
    g = torch.Generator().manual_seed(23)
    prompt = torch.randint(3, 256, (LONG_PROMPT_LEN,), generator=g).tolist()
    with torch.no_grad():
        out = m.generate(torch.tensor([prompt]), max_new_tokens=LONG_N_NEW, do_sample=False)[0].tolist()
        steps = []
        for t in range(len(prompt) - 1, len(out) - 1):
            lg = m(torch.tensor([out[:t + 1]])).logits[0, -1]
            top = lg.topk(5)
            steps.append(dict(top_ids=top.indices.tolist(), top_logits=top.values.tolist()))
    json.dump(dict(prompt=prompt, ids=out, steps=steps, sliding_window=m.config.sliding_window,
                   model="random Gemma4 text, 3 layers, long prompt"),
              open(os.path.join(FIX, "gemma4_tiny_hf_reference_long.json"), "w"), indent=1)
    print("wrote long reference", out[-LONG_N_NEW:])


def main():
    if "--long" in sys.argv:
        return make_long_reference()
    torch.manual_seed(7)
    cfg = tiny_config()
    m = Gemma4ForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("norm.weight") or "layernorm" in n or n.endswith("router.scale"):
                p.copy_(torch.empty_like(p).uniform_(0.6, 1.4, generator=g))
            elif n.endswith("per_expert_scale"):
                p.copy_(torch.empty_like(p).uniform_(0.7, 1.3, generator=g))
            elif n.endswith("router.proj.weight"):
                p.mul_(6.0)
        for n, b in m.named_buffers():
            if n.endswith("layer_scalar"):
                b.copy_(torch.empty_like(b).uniform_(0.7, 1.3, generator=g))
    d = os.path.join(FIX, "gemma4_tiny_model")
    m.save_pretrained(d, safe_serialization=True)
    with torch.no_grad():
        out = m.generate(torch.tensor([PROMPT]), max_new_tokens=N_NEW, do_sample=False)[0].tolist()
        steps, picks = [], []
        for t in range(len(PROMPT) - 1, len(out) - 1):
            lg = m(torch.tensor([out[:t + 1]])).logits[0, -1]
            top = lg.topk(5)
            steps.append(dict(top_ids=top.indices.tolist(), top_logits=top.values.tolist()))
    json.dump(dict(prompt=PROMPT, ids=out, steps=steps, model="random Gemma4 text, 3 layers"),
              open(os.path.join(FIX, "gemma4_tiny_hf_reference.json"), "w"), indent=1)
    print("wrote", d, out)


if __name__ == "__main__":
    sys.exit(main())
