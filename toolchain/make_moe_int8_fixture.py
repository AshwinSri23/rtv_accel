#!/usr/bin/env python3
"""
make_moe_int8_fixture.py -- regenerate rtv_tvm/fixtures/moe_tiny_int8_hf_reference.json.

Hugging Face tiny Qwen3-MoE with attention q/k/v/o and every expert matrix
fake-quantized to INT8 per output channel (the chip's --int8-weights scheme; router,
norms, embeddings, lm_head stay FP32), then greedy generation + top-5 logits per step.
Needs torch + transformers (not a project dependency).
"""
import glob, json, os, sys
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

D = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--hf-internal-testing--tiny-random-Qwen3MoeForCausalLM/snapshots/*")))[0]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rtv_tvm", "fixtures",
                   "moe_tiny_int8_hf_reference.json")
PROMPT, N_NEW = [785, 6722, 315, 9625, 374], 6


def fq(w):                      # w[..., out, in]: one scale per output row
    s = (w.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-12)
    return torch.clamp(torch.round(w / s), -128, 127) * s


def main():
    m = AutoModelForCausalLM.from_pretrained(D, dtype=torch.float32).eval()
    f = safe_open(glob.glob(D + "/*.safetensors")[0], "pt")
    with torch.no_grad():
        for i, layer in enumerate(m.model.layers):
            layer.mlp.gate.weight.copy_(f.get_tensor(f"model.layers.{i}.mlp.router.weight"))
            for p in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
                      layer.self_attn.o_proj):
                p.weight.copy_(fq(p.weight))
            layer.mlp.experts.gate_up_proj.copy_(fq(layer.mlp.experts.gate_up_proj))
            layer.mlp.experts.down_proj.copy_(fq(layer.mlp.experts.down_proj))
        ids = torch.tensor([PROMPT])
        out = m.generate(ids, max_new_tokens=N_NEW, do_sample=False)[0].tolist()
        steps = []
        for t in range(len(PROMPT) - 1 + 0, len(out) - 1):
            lg = m(torch.tensor([out[:t + 1]])).logits[0, -1]
            top = lg.topk(5)
            steps.append(dict(top_ids=top.indices.tolist(), top_logits=top.values.tolist()))
    json.dump(dict(prompt=PROMPT, ids=out, steps=steps, model="tiny-random-Qwen3MoeForCausalLM",
                   note="INT8 per-output-channel fake-quantized attention + experts; see make_moe_int8_fixture.py"),
              open(OUT, "w"), indent=1)
    print("wrote", OUT, out)


if __name__ == "__main__":
    sys.exit(main())
