#!/usr/bin/env python3
"""
ref_model.py — float32 numpy reference for Qwen3, used as ground truth.

The project had no golden model. Every test so far compared the device
against a reference that mirrored the DEVICE's integer semantics, which
verifies that the compiler lowers what it intended but cannot detect that the
intent itself is wrong -- exactly how the missing q_norm/k_norm survived: no
test referenced them, so no test could miss them.

This module instead implements Qwen3's forward pass as the architecture
defines it, in float32, reading the checkpoint directly. Divergence between
this and the device is the quantity layer_probe.py measures.

Deliberately unoptimised and written to match the reference implementation
statement-for-statement; it runs on 1024-wide hidden states for a handful of
positions, so clarity is worth more than speed.
"""

import numpy as np


def rmsnorm(x, gamma, eps=1e-6):
    """RMSNorm over the last axis: x / sqrt(mean(x^2) + eps) * gamma."""
    x = x.astype(np.float32)
    inv = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return x * inv * gamma.astype(np.float32)


def silu(x):
    return x / (1.0 + np.exp(-x, dtype=np.float32))


def rope_tables(seq, head_dim, theta=1e6, offset=0):
    """
    Half-split RoPE, matching HF's Qwen3 implementation.

    HF builds inv_freq over head_dim/2 and then CONCATENATES it with itself
    (`emb = cat(freqs, freqs)`), so cos/sin are head_dim wide and the rotation
    pairs element i with element i + head_dim/2 -- not adjacent pairs. Getting
    this wrong produces plausible-looking but wrong attention.
    """
    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
    pos = np.arange(offset, offset + seq, dtype=np.float32)[:, None]
    freqs = pos * inv[None, :]                       # (seq, half)
    emb = np.concatenate([freqs, freqs], axis=-1)    # (seq, head_dim)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope(x, cos, sin):
    """x: (seq, heads, head_dim); cos/sin: (seq, head_dim)."""
    half = x.shape[-1] // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


class RefQwen3:
    """Float32 Qwen3 forward pass over a full prompt (no KV cache)."""

    def __init__(self, model):
        # Tensors are read on demand rather than cached: Qwen3-0.6B in float32
        # is ~2.4 GB, and each weight is touched exactly once per forward pass,
        # so holding them all resident buys nothing.
        self.model = model
        self.shapes = {n: v.shape for n, v in model.named_parameters()}
        c = model.config()
        self.n_layers  = c["num_hidden_layers"]
        self.n_heads   = c["num_attention_heads"]
        self.n_kv      = c.get("num_key_value_heads", self.n_heads)
        self.head_dim  = c.get("head_dim",
                               c["hidden_size"] // c["num_attention_heads"])
        self.eps       = c.get("rms_norm_eps", 1e-6)
        self.theta     = c.get("rope_theta", 1e6)
        self.hidden    = c["hidden_size"]

    def _w(self, name):
        return np.asarray(self.model.tensor(name), dtype=np.float32)

    def forward(self, token_ids, upto_layer=None, apply_qk_norm=True):
        """
        Returns a list of per-layer hidden states, [x0, x1, ... xL], where x0
        is the embedding output and xI is the state after layer I-1.

        apply_qk_norm=False reproduces the device's current behaviour (which
        drops q_norm/k_norm) so the probe can attribute error to that gap
        specifically rather than lumping it in with quantisation loss.
        """
        n_layers = self.n_layers if upto_layer is None else upto_layer
        T = len(token_ids)
        hd, nh, nkv = self.head_dim, self.n_heads, self.n_kv

        x = self._w("model.embed_tokens.weight")[list(token_ids)]   # (T, hidden)
        states = [x.copy()]

        cos, sin = rope_tables(T, hd, self.theta)
        causal = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)

        for i in range(n_layers):
            b = f"model.layers.{i}."
            y = rmsnorm(x, self._w(b + "input_layernorm.weight"), self.eps)

            # HF stores nn.Linear as (out, in) and computes x @ W.T.
            q = (y @ self._w(b + "self_attn.q_proj.weight").T).reshape(T, nh,  hd)
            k = (y @ self._w(b + "self_attn.k_proj.weight").T).reshape(T, nkv, hd)
            v = (y @ self._w(b + "self_attn.v_proj.weight").T).reshape(T, nkv, hd)

            # Qwen3 normalises each head's Q and K before RoPE. This is the
            # step the compiler omits entirely.
            if apply_qk_norm:
                q = rmsnorm(q, self._w(b + "self_attn.q_norm.weight"), self.eps)
                k = rmsnorm(k, self._w(b + "self_attn.k_norm.weight"), self.eps)

            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            # GQA: each KV head serves nh/nkv query heads.
            rep = nh // nkv
            kx = np.repeat(k, rep, axis=1)      # (T, nh, hd)
            vx = np.repeat(v, rep, axis=1)

            qh = q.transpose(1, 0, 2)           # (nh, T, hd)
            kh = kx.transpose(1, 0, 2)
            vh = vx.transpose(1, 0, 2)

            scores = qh @ kh.transpose(0, 2, 1) / np.sqrt(np.float32(hd))
            scores = scores + causal[None]
            scores = scores - scores.max(axis=-1, keepdims=True)
            w = np.exp(scores)
            w /= w.sum(axis=-1, keepdims=True)
            ctx = (w @ vh).transpose(1, 0, 2).reshape(T, nh * hd)

            x = x + ctx @ self._w(b + "self_attn.o_proj.weight").T

            y2 = rmsnorm(x, self._w(b + "post_attention_layernorm.weight"), self.eps)
            g = y2 @ self._w(b + "mlp.gate_proj.weight").T
            u = y2 @ self._w(b + "mlp.up_proj.weight").T
            x = x + (silu(g) * u) @ self._w(b + "mlp.down_proj.weight").T

            states.append(x.copy())

        return states

    def logits(self, token_ids, apply_qk_norm=True):
        x = self.forward(token_ids, apply_qk_norm=apply_qk_norm)[-1]
        x = rmsnorm(x, self._w("model.norm.weight"), self.eps)
        return x @ self._w("lm_head.weight").T
