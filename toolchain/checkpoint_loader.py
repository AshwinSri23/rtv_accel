#!/usr/bin/env python3
"""
checkpoint_loader.py — a real Hugging Face safetensors checkpoint (Qwen3
architecture) into the exact shapes tvm_backend.py's FP32 TVM path expects,
built on hf_loader.py's SafeTensorsModel (unmodified -- it was already
dtype-agnostic; nothing about it was ever INT8-specific, only its
now-retired caller was).

The one real transformation needed: Hugging Face stores every nn.Linear
weight as (out_features, in_features) and computes `x @ W.T`, but this
project's GEMM consumes its B operand directly as (in, out) -- confirmed
as a real, previously-hit bug in this exact spot (see the spec doc's
"A layout bug found while wiring this up" note: feeding a genuine
checkpoint through untransposed silently computed `x @ W`, shape-
compatible only where in==out, wrong everywhere else, no error raised).
Every 2-D projection weight below is transposed on load specifically to
avoid repeating that.

What this does NOT do, and why that's a real, tracked gap rather than an
oversight: Qwen3 applies a per-head RMSNorm to Q and K (`q_norm.weight`/
`k_norm.weight`, each a (head_dim,) gain shared across every head) BEFORE
RoPE -- confirmed present in the real checkpoint (every layer has both).
tvm_backend._emit_attention_block_kvcache_fp32 has no such step. This
loader still extracts and returns those weights (as `q_norm`/`k_norm` in
each layer's dict) so nothing is silently dropped, but nothing in the
compiler consumes them yet -- running a real checkpoint through the
current pipeline is not yet a faithful reproduction of Qwen3's actual
forward pass for exactly this reason. See TODO.md.

Also host-side, not compiled to the NPU at all, consistent with the
established design (embedding gather and argmax sampling are already
host-side because the ISA has no indirect addressing or control flow --
see the spec doc's "Embedding lookup, lm_head, and sampling" section):
`embed_tokens`/`lm_head` (tied in the real Qwen3-0.6B checkpoint --
confirmed directly, `np.array_equal` on the two tensors) are returned as
plain numpy arrays for a host driver to index/matmul directly, not staged
into any DRAM image this module builds.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from hf_loader import SafeTensorsModel, hal_model_config  # noqa: E402


def load_qwen3_checkpoint(model_dir: str, n_layers: int = None):
    """Load a real Qwen3-architecture checkpoint (safetensors + config.json
    in `model_dir`, e.g. what `hf download Qwen/Qwen3-0.6B` produces).

    n_layers: load only the first N transformer layers instead of every
    layer the checkpoint has (config["num_layers"] stays the checkpoint's
    real value either way -- this only limits how many layer dicts are
    returned, e.g. for verifying against a handful of real layers without
    compiling and simulating all 28 of a real model every test run).

    Returns (config, layer_weights_np, embed_tokens, lm_head, final_norm):

      config: hal_model_config(...)'s dict (hidden_size, num_layers,
      num_heads, num_kv_heads, head_dim, vocab_size, rms_norm_eps,
      rope_theta) -- num_layers here is always the CHECKPOINT's real
      count, independent of the n_layers truncation above.

      layer_weights_np: a list of dicts, one per loaded layer, with
      gamma1/gamma2/wq/wk/wv/wo/wgate/wup/wdown in EXACTLY the shapes and
      orientation build_kvcache_model_dram_image/build_model_dram_image
      expect (every 2-D projection already transposed out of HF's
      (out, in) layout into this compiler's (in, out) one) -- plus
      q_norm/k_norm (each (head_dim,), NOT consumed by the compiler yet,
      see the module docstring), included so nothing is silently dropped.

      embed_tokens, lm_head: (vocab_size, hidden_size) numpy arrays, for a
      host driver's row-gather and final logits projection -- see the
      module docstring for why these stay off the NPU entirely.

      final_norm: (hidden_size,) gamma for the model's own trailing
      RMSNorm (`model.norm.weight`), applied after the last transformer
      layer and before lm_head -- also host-side for now, alongside
      embed_tokens/lm_head, since nothing here compiles a whole-model
      driver yet.
    """
    m = SafeTensorsModel(model_dir)
    cfg = hal_model_config(m.config(model_dir))
    total_layers = cfg["num_layers"]
    load_n = total_layers if n_layers is None else min(n_layers, total_layers)

    def t(name):
        return m.tensor(name)

    layer_weights_np = []
    for li in range(load_n):
        p = f"model.layers.{li}."
        layer_weights_np.append(dict(
            gamma1=t(p + "input_layernorm.weight"),
            gamma2=t(p + "post_attention_layernorm.weight"),
            # HF's nn.Linear stores (out_features, in_features) and
            # computes x @ W.T; this project's GEMM wants B as
            # (in, out) directly -- transpose every 2-D projection here,
            # once, rather than risk repeating the exact bug the spec
            # doc already documents hitting in this precise spot.
            wq=np.ascontiguousarray(t(p + "self_attn.q_proj.weight").T),
            wk=np.ascontiguousarray(t(p + "self_attn.k_proj.weight").T),
            wv=np.ascontiguousarray(t(p + "self_attn.v_proj.weight").T),
            wo=np.ascontiguousarray(t(p + "self_attn.o_proj.weight").T),
            wgate=np.ascontiguousarray(t(p + "mlp.gate_proj.weight").T),
            wup=np.ascontiguousarray(t(p + "mlp.up_proj.weight").T),
            wdown=np.ascontiguousarray(t(p + "mlp.down_proj.weight").T),
            # Per-head RMSNorm gain applied to Q/K before RoPE in real
            # Qwen3 -- extracted so nothing is silently dropped, but NOT
            # wired into _emit_attention_block_kvcache_fp32 yet. See the
            # module docstring.
            q_norm=t(p + "self_attn.q_norm.weight"),
            k_norm=t(p + "self_attn.k_norm.weight"),
        ))

    embed_tokens = t("model.embed_tokens.weight")
    lm_head = t("lm_head.weight") if "lm_head.weight" in m.header else embed_tokens
    final_norm = t("model.norm.weight")

    return cfg, layer_weights_np, embed_tokens, lm_head, final_norm


if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if model_dir is None:
        print("usage: checkpoint_loader.py <model_dir_or_safetensors_file>")
        sys.exit(1)
    cfg, layers, embed_tokens, lm_head, final_norm = load_qwen3_checkpoint(model_dir, n_layers=2)
    print("config:", cfg)
    print(f"loaded {len(layers)} layer(s) (of {cfg['num_layers']} total)")
    for k, v in layers[0].items():
        print(f"  layer 0 {k:10s} {v.shape} {v.dtype}")
    print("embed_tokens", embed_tokens.shape, embed_tokens.dtype)
    print("lm_head     ", lm_head.shape, lm_head.dtype)
    print("final_norm  ", final_norm.shape, final_norm.dtype)
