#!/usr/bin/env python3
"""
make_moe_wide_fixture.py -- regenerate rtv_tvm/fixtures/moe_wide_model/ (a small random-init Qwen3-MoE with
40 experts, top-6: the router spans three 16-wide tiles, the last only partly used) and
rtv_tvm/fixtures/moe_wide_hf_reference.json (Hugging Face greedy ids + top-5 logits per step).
Needs torch + transformers (not a project dependency).
"""
import json, os, sys
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "rtv_tvm", "fixtures")
PROMPT, N_NEW = [5, 17, 101, 42, 7], 6


def main():
    torch.manual_seed(1234)
    cfg = Qwen3MoeConfig(vocab_size=256, hidden_size=32, intermediate_size=64, moe_intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         num_experts=40, num_experts_per_tok=6, norm_topk_prob=True,
                         max_position_embeddings=64, rope_theta=1e6, rms_norm_eps=1e-6,
                         decoder_sparse_step=1, mlp_only_layers=[], tie_word_embeddings=False)
    m = Qwen3MoeForCausalLM(cfg).eval()
    with torch.no_grad():                     # widen the router so many different experts get picked
        for layer in m.model.layers:
            layer.mlp.gate.weight.mul_(8.0)
    d = os.path.join(FIX, "moe_wide_model")
    m.save_pretrained(d, safe_serialization=True)
    with torch.no_grad():
        out = m.generate(torch.tensor([PROMPT]), max_new_tokens=N_NEW, do_sample=False)[0].tolist()
        steps = []
        for t in range(len(PROMPT) - 1, len(out) - 1):
            lg = m(torch.tensor([out[:t + 1]])).logits[0, -1]
            top = lg.topk(5)
            steps.append(dict(top_ids=top.indices.tolist(), top_logits=top.values.tolist()))
    json.dump(dict(prompt=PROMPT, ids=out, steps=steps, model="random Qwen3MoE 40 experts top-6"),
              open(os.path.join(FIX, "moe_wide_hf_reference.json"), "w"), indent=1)
    print("wrote", d, out)


if __name__ == "__main__":
    sys.exit(main())
