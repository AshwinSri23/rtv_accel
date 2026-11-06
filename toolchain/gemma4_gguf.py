#!/usr/bin/env python3
"""gemma4_gguf.py -- Gemma-4 (dense MLP + 128 experts) from a llama.cpp Q4_0 GGUF.

Q4_0 is value = (code - 8) * fp16_scale over 32 consecutive INPUT-dim elements, which is exactly the chip's
INT4W g32 grid, so the weights convert losslessly: a `Q4Mat` keeps the raw 18-byte blocks (a view into the mmap)
and only builds the INT4W DRAM blob (`tvm_backend.int4_weight_blob` layout) when it is placed -- the model is
never expanded to FP32 (it would be ~90 GB). `load_gemma4_gguf` returns the same (cfg, layers, embed, final_norm)
that gemma4_model.generate_gemma4 consumes; `hf_layer_tensors` gives dequantised Hugging Face-named tensors for
one layer (reference runs)."""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from gguf_reader import GGUF  # noqa: E402


class Q4Mat:
    """A [K, N] (input x output) weight stored as Q4_0 blocks `raw` of shape (N, K/32, 18)."""

    def __init__(self, raw):
        assert raw.ndim == 3 and raw.shape[2] == 18 and raw.dtype == np.uint8
        self.raw = raw
        self.N, self.nblk = raw.shape[0], raw.shape[1]
        self.K = self.nblk * 32

    def _codes_scales(self):
        r = self.raw
        d = np.ascontiguousarray(r[:, :, :2]).view(np.float16).reshape(self.N, self.nblk).astype(np.float32)
        q = r[:, :, 2:]
        codes = np.empty((self.N, self.nblk, 32), np.uint8)
        codes[:, :, :16] = q & 15
        codes[:, :, 16:] = q >> 4
        return codes, d

    def blob(self):
        """INT4W blob: K*N/2 packed bytes (row-major [K, N], two consecutive N per byte, low nibble = even n,
        two's-complement nibble of code-8), then the (K/32, N) fp32 scale rows."""
        codes, d = self._codes_scales()
        nib = (codes ^ 8).transpose(1, 2, 0).reshape(self.K, self.N)       # (code-8) & 15, laid out [K, N]
        packed = nib[:, 0::2] | (nib[:, 1::2] << 4)
        return np.concatenate([packed.reshape(-1), np.ascontiguousarray(d.T).reshape(-1).view(np.uint8)])

    def dense(self):
        codes, d = self._codes_scales()
        w = (codes.astype(np.float32) - 8.0) * d[:, :, None]               # (N, nblk, 32)
        return np.ascontiguousarray(w.reshape(self.N, self.K).T)          # [K, N]


def _q4(g, name, rows=None, expert=None):
    dims, typ, raw = g.raw(name)
    assert typ == "Q4_0", (name, typ)
    k_in = dims[0]
    if expert is None:
        r = raw.reshape(dims[1], k_in // 32, 18)
    else:
        r = raw.reshape(dims[2], dims[1], k_in // 32, 18)[expert]
    return Q4Mat(r[rows] if rows is not None else r)


def gguf_config(g):
    m = g.meta
    pat = m["gemma4.attention.sliding_window_pattern"]
    n = m["gemma4.block_count"]
    return dict(
        num_hidden_layers=n, hidden_size=m["gemma4.embedding_length"], intermediate_size=m["gemma4.feed_forward_length"],
        moe_intermediate_size=m["gemma4.expert_feed_forward_length"], num_experts=m["gemma4.expert_count"],
        top_k_experts=m["gemma4.expert_used_count"], num_attention_heads=m["gemma4.attention.head_count"],
        num_key_value_heads=max(m["gemma4.attention.head_count_kv"]), head_dim=m["gemma4.attention.key_length_swa"],
        global_head_dim=m["gemma4.attention.key_length"],
        num_global_key_value_heads=min(m["gemma4.attention.head_count_kv"]),
        rms_norm_eps=m["gemma4.attention.layer_norm_rms_epsilon"],
        final_logit_softcapping=m.get("gemma4.final_logit_softcapping"),
        sliding_window=m["gemma4.attention.sliding_window"],
        layer_types=["sliding_attention" if s else "full_attention" for s in pat],
        vocab_size=g.shape("token_embd.weight")[0], rope_theta_full=m["gemma4.rope.freq_base"],
        rope_theta_sliding=m["gemma4.rope.freq_base_swa"])


class Int8Rows:
    """The tied embedding / lm_head table as INT8 with one FP32 scale per row (0.74 GB instead of 2.95 GB for the
    262144 x 2816 table). Row lookup dequantises one row; `table @ v` (the lm_head) runs in chunks so the full FP32
    table never exists. Host-side only -- the chip never sees it."""

    def __init__(self, table):
        self.shape = table.shape
        amax = np.abs(table).max(axis=1)
        self.scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
        self.q = np.empty(table.shape, np.int8)
        for i in range(0, table.shape[0], 16384):
            self.q[i:i + 16384] = np.clip(np.rint(table[i:i + 16384] / self.scale[i:i + 16384, None]), -127, 127)

    def __getitem__(self, i):
        return self.q[i].astype(np.float32) * self.scale[i]

    def __matmul__(self, v):
        v = np.asarray(v, np.float32)
        out = np.empty(self.shape[0], np.float32)
        for i in range(0, self.shape[0], 16384):
            out[i:i + 16384] = (self.q[i:i + 16384].astype(np.float32) @ v) * self.scale[i:i + 16384]
        return out


def load_gemma4_gguf(path, n_layers=None, load_embed=True, embed_int8=False):
    """-> (cfg, layers, embed fp32 [vocab, hidden] (or Int8Rows), final_norm). Weight matrices are lazy Q4Mat objects."""
    g = GGUF(path)
    cfg = gguf_config(g)
    if n_layers is not None:
        cfg["num_hidden_layers"] = n_layers
        cfg["layer_types"] = cfg["layer_types"][:n_layers]
    E, I = cfg["num_experts"], cfg["moe_intermediate_size"]
    kv = g.meta["gemma4.attention.head_count_kv"]
    f = lambda n: g.dequant(n).astype(np.float32)  # noqa: E731
    layers = []
    for li in range(cfg["num_hidden_layers"]):
        p = f"blk.{li}."
        full = cfg["layer_types"][li] == "full_attention"
        hd = cfg["global_head_dim"] if full else cfg["head_dim"]
        has_v = (p + "attn_v.weight") in g.tensors
        layers.append(dict(
            type=cfg["layer_types"][li], head_dim=hd, n_kv=kv[li],
            theta=cfg["rope_theta_full"] if full else cfg["rope_theta_sliding"],
            rope_angles=int(0.25 * hd // 2) if full else None,
            wq=_q4(g, p + "attn_q.weight"), wk=_q4(g, p + "attn_k.weight"),
            wv=_q4(g, p + "attn_v.weight") if has_v else None, wo=_q4(g, p + "attn_output.weight"),
            q_norm=f(p + "attn_q_norm.weight"), k_norm=f(p + "attn_k_norm.weight"),
            g_in=f(p + "attn_norm.weight"), g_post_attn=f(p + "post_attention_norm.weight"),
            g_pre_ff=f(p + "ffn_norm.weight"), g_post_ff=f(p + "post_ffw_norm.weight"),
            g_pre_ff2=f(p + "pre_ffw_norm_2.weight"), g_post_ff1=f(p + "post_ffw_norm_1.weight"),
            g_post_ff2=f(p + "post_ffw_norm_2.weight"),
            mlp_gate=_q4(g, p + "ffn_gate.weight"), mlp_up=_q4(g, p + "ffn_up.weight"),
            mlp_down=_q4(g, p + "ffn_down.weight"),
            router=f(p + "ffn_gate_inp.weight"), router_scale=f(p + "ffn_gate_inp.scale"),
            per_expert_scale=f(p + "ffn_down_exps.scale"),
            experts=[dict(w_gate=_q4(g, p + "ffn_gate_up_exps.weight", slice(0, I), e),
                          w_up=_q4(g, p + "ffn_gate_up_exps.weight", slice(I, 2 * I), e),
                          w_down=_q4(g, p + "ffn_down_exps.weight", None, e)) for e in range(E)],
            layer_scalar=float(f(p + "layer_output_scale.weight").reshape(-1)[0])))
    embed = f("token_embd.weight") if load_embed else None
    if embed is not None and embed_int8:
        embed = Int8Rows(embed)
    return cfg, layers, embed, f("output_norm.weight")


def hf_layer_tensors(layer):
    """Hugging Face Gemma4TextDecoderLayer state-dict (float32 torch-ready numpy) from a loaded layer dict."""
    d = {
        "self_attn.q_proj.weight": layer["wq"].dense().T, "self_attn.k_proj.weight": layer["wk"].dense().T,
        "self_attn.o_proj.weight": layer["wo"].dense().T,
        "self_attn.q_norm.weight": layer["q_norm"], "self_attn.k_norm.weight": layer["k_norm"],
        "input_layernorm.weight": layer["g_in"], "post_attention_layernorm.weight": layer["g_post_attn"],
        "pre_feedforward_layernorm.weight": layer["g_pre_ff"], "post_feedforward_layernorm.weight": layer["g_post_ff"],
        "pre_feedforward_layernorm_2.weight": layer["g_pre_ff2"],
        "post_feedforward_layernorm_1.weight": layer["g_post_ff1"],
        "post_feedforward_layernorm_2.weight": layer["g_post_ff2"],
        "mlp.gate_proj.weight": layer["mlp_gate"].dense().T, "mlp.up_proj.weight": layer["mlp_up"].dense().T,
        "mlp.down_proj.weight": layer["mlp_down"].dense().T,
        "router.proj.weight": layer["router"], "router.scale": layer["router_scale"],
        "router.per_expert_scale": layer["per_expert_scale"],
        "experts.gate_up_proj": np.stack([np.concatenate([e["w_gate"].dense().T, e["w_up"].dense().T], axis=0)
                                          for e in layer["experts"]]),
        "experts.down_proj": np.stack([e["w_down"].dense().T for e in layer["experts"]]),
        "layer_scalar": np.array([layer["layer_scalar"]], np.float32),
    }
    if layer["wv"] is not None:
        d["self_attn.v_proj.weight"] = layer["wv"].dense().T
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in d.items()}


def render_chat(gguf_path, messages):
    """The GGUF's own Jinja chat template applied to `messages` ([{"role","content"}, ...]) with the generation
    prompt added; returns the prompt string (it starts with the literal '<bos>')."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    g = GGUF(gguf_path)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(msg):
        raise ValueError(msg)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(g.meta["tokenizer.chat_template"]).render(
        messages=messages, add_generation_prompt=True, bos_token="<bos>")
