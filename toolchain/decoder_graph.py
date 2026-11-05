#!/usr/bin/env python3
"""
decoder_graph.py -- a Qwen3 decode step expressed as a Relax graph and lowered through
the BYOC path (byoc_graph.py), the graph-driven alternative to the hand-written
`tvm_backend._emit_model_kvcache_fp32`.

One decoder layer is six top-level graph ops:

    n1   = rms_norm(x, g1)                        claimed -> rmsnorm composite
    attn = rtv.attention_kv:<layer>(n1, wq, wk, wv, wo [, q_norm, k_norm])   EXTERN
    r1   = x + attn                               claimed -> add composite
    n2   = rms_norm(r1, g2)                       claimed -> rmsnorm composite
    ffn  = matmul/silu/multiply/matmul(n2, ...)   claimed -> swiglu_ffn composite
    out  = r1 + ffn                               claimed -> add composite

The attention block (q/k per-head norm, RoPE at the current position, KV-cache append, GQA
attention over the growing cache) cannot be discovered from graph structure (patterns.py says
why), so it travels through the graph as an extern call and is lowered by the existing
config-aware emitter `_emit_attention_block_kvcache_fp32`. Everything else is claimed by the
patterns. Acceptance (test_byoc_decoder.py): the instruction stream is BYTE-IDENTICAL to the
hand-driven path for a layer, for a multi-layer model, and under schedule relocation.

Binding layout (used for fixed ping-pong layer outputs): layer i occupies top-level bindings
6*i .. 6*i+5; its output is binding 6*i+5.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "rtv_tvm"))

from tvm import relax                      # noqa: E402
from tvm.relax import op as rop            # noqa: E402

import byoc_graph                          # noqa: E402
import tvm_backend                         # noqa: E402

BINDINGS_PER_LAYER = 6
_CACHE = {}


def _t(*shape):
    return relax.TensorType(tuple(shape), "float32")


def layer_param_names(i, qk_norm):
    names = [f"L{i}_{n}" for n in ("g1", "wq", "wk", "wv", "wo")]
    if qk_norm:
        names += [f"L{i}_qn", f"L{i}_kn"]
    return names + [f"L{i}_{n}" for n in ("g2", "wg", "wu", "wd")]


def build_decode_graph(cfg, n_layers, qk_norm=True, batched=False):
    """Relax module `main(x, <per-layer params>...) -> (rows, hidden)` for n_layers layers.
    cfg needs hidden, inter, n_heads, n_kv_heads, head_dim, eps.

    batched=False: single-token decode, x is (1, hidden), norms are the claimed `rms_norm`
    pattern. batched=True: a 16-row PREFILL tile; the norms become the extern
    `rtv.rmsnorm_rows` (a genuine per-row scale -- the claimed `rmsnorm` pattern shares one
    scale across the whole tile, which is wrong for 16 different tokens) and the attention is
    `rtv.attention_kv_batched` (causal, multi-tile, writes the KV-cache)."""
    key = (cfg["hidden"], cfg["inter"], cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"],
           float(cfg["eps"]), n_layers, qk_norm, batched)
    if key in _CACHE:
        return _CACHE[key]
    H, I = cfg["hidden"], cfg["inter"]
    q_dim, kv_dim = cfg["n_heads"] * cfg["head_dim"], cfg["n_kv_heads"] * cfg["head_dim"]
    eps = float(cfg["eps"])
    shapes = {"g1": (H,), "wq": (H, q_dim), "wk": (H, kv_dim), "wv": (H, kv_dim),
              "wo": (q_dim, H), "qn": (cfg["head_dim"],), "kn": (cfg["head_dim"],),
              "g2": (H,), "wg": (H, I), "wu": (H, I), "wd": (I, H)}
    R_ = 16 if batched else 1
    x = relax.Var("x", _t(R_, H))
    params, per_layer = [x], []
    for i in range(n_layers):
        d = {}
        for n in layer_param_names(i, qk_norm):
            d[n.split("_", 1)[1]] = relax.Var(n, _t(*shapes[n.split("_", 1)[1]]))
        per_layer.append(d)
        params += list(d.values())

    def norm(v, gamma):
        if batched:
            return relax.Call(relax.ExternFunc("rtv.rmsnorm_rows"), [v, gamma], None, [_t(R_, H)])
        return rop.nn.rms_norm(v, gamma, axes=[-1], epsilon=eps)

    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            cur = x
            for i, d in enumerate(per_layer):
                n1 = bb.emit(norm(cur, d["g1"]))
                ext_args = [n1, d["wq"], d["wk"], d["wv"], d["wo"]]
                if qk_norm:
                    ext_args += [d["qn"], d["kn"]]
                attn = bb.emit(relax.Call(relax.ExternFunc(
                    f"rtv.attention_kv_batched:{i}" if batched else f"rtv.attention_kv:{i}"),
                    ext_args, None, [_t(R_, H)]))
                r1 = bb.emit(rop.add(cur, attn))
                n2 = bb.emit(norm(r1, d["g2"]))
                gate = bb.emit(rop.matmul(n2, d["wg"]))
                up = bb.emit(rop.matmul(n2, d["wu"]))
                act = bb.emit(rop.nn.silu(gate))
                inter = bb.emit(rop.multiply(act, up))
                y = bb.emit(rop.matmul(inter, d["wd"]))
                if i == n_layers - 1:
                    cur = bb.emit_output(rop.add(r1, y))
                else:
                    cur = bb.emit(rop.add(r1, y))
        bb.emit_func_output(cur)
    mod = tvm_backend.partition(bb.get())
    _CACHE[key] = mod
    return mod


def attention_lowering(cfg, layer_weights, pos, rope_theta, attn_scale):
    """The extern lowering for "rtv.attention_kv": wraps the config-aware decode attention
    block. `suffix` is the layer index; its KV-cache comes from layer_weights[layer]."""
    def lower(compiler, suffix, args, out_addr):
        li = int(suffix)
        lw = layer_weights[li]
        n1, wq, wk, wv, wo = args[:5]
        qn, kn = (args[5], args[6]) if len(args) == 7 else (None, None)
        tvm_backend._emit_attention_block_kvcache_fp32(
            compiler, cfg["hidden"], cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"],
            x_addr=n1, wq_addr=wq, wk_addr=wk, wv_addr=wv, wo_addr=wo, out_addr=out_addr,
            pos=pos, k_cache_tiles=lw["k_cache"], v_cache_tiles=lw["v_cache"],
            rope_theta=rope_theta, q_norm_addr=qn, k_norm_addr=kn, attn_scale=attn_scale)
    return {"rtv.attention_kv": lower}


def prefill_lowering(cfg, layer_weights, n_real, tile_idx, rope_theta, attn_scale):
    """Extern lowerings for the batched (prefill) graph: per-row RMSNorm and the causal,
    multi-tile, KV-cache-writing attention block."""
    def rows_norm(compiler, suffix, args, out_addr):
        tvm_backend._emit_rmsnorm_rows_fp32(compiler, cfg["hidden"], compiler.tile_dim,
                                            args[0], args[1], out_addr)

    def attn(compiler, suffix, args, out_addr):
        lw = layer_weights[int(suffix)]
        n1, wq, wk, wv, wo = args[:5]
        qn, kn = (args[5], args[6]) if len(args) == 7 else (None, None)
        tvm_backend._emit_attention_block_kvcache_batched_fp32(
            compiler, cfg["hidden"], cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"],
            x_addr=n1, wq_addr=wq, wk_addr=wk, wv_addr=wv, wo_addr=wo, out_addr=out_addr,
            n_real=n_real, k_cache_tiles=lw["k_cache"], v_cache_tiles=lw["v_cache"],
            rope_theta=rope_theta, q_norm_addr=qn, k_norm_addr=kn, tile_idx=tile_idx,
            attn_scale=attn_scale)
    return {"rtv.rmsnorm_rows": rows_norm, "rtv.attention_kv_batched": attn}


def emit_prefill_graph(compiler, cfg, n_layers, x_addr, layer_weights, scratch_a, scratch_b,
                       n_real, tile_idx=0, rope_theta=1e6, attn_scale=None):
    """Graph-driven equivalent of tvm_backend._emit_model_kvcache_batched_fp32 (one 16-row
    prefill tile): same ping-pong buffers, same return value."""
    qk_norm = layer_weights[0].get("q_norm") is not None
    mod = build_decode_graph(cfg, n_layers, qk_norm, batched=True)
    bufs = [scratch_b, scratch_a]
    fixed = {BINDINGS_PER_LAYER * i + BINDINGS_PER_LAYER - 1: bufs[i % 2] for i in range(n_layers)}
    out, _ = byoc_graph.emit_graph(
        compiler, mod, _inputs(layer_weights, range(n_layers), x_addr, qk_norm),
        extern=prefill_lowering(cfg, layer_weights, n_real, tile_idx, rope_theta, attn_scale),
        fixed_outputs=fixed, batch_alloc=True)
    return out


def make_prefill_layer_emitter(cfg, n_real, tile_idx, rope_theta=1e6, attn_scale=None):
    """Prefill counterpart of make_layer_emitter, for tvm_backend._emit_model_kvcache_fp32_relocated:
    compiles ONE layer of the batched (16-row) graph; layers 2+ are address-patched."""
    def emit_layer(compiler, i, x_in, out, lw):
        qk_norm = lw.get("q_norm") is not None
        mod = build_decode_graph(cfg, 1, qk_norm, batched=True)
        byoc_graph.emit_graph(
            compiler, mod, _inputs({0: lw}, [0], x_in, qk_norm),
            extern=prefill_lowering(cfg, {0: lw}, n_real, tile_idx, rope_theta, attn_scale),
            fixed_outputs={BINDINGS_PER_LAYER - 1: out}, batch_alloc=True)
    return emit_layer


def _inputs(layer_weights, layer_ids, x_addr, qk_norm):
    inp = {"x": x_addr}
    for i in layer_ids:
        lw = layer_weights[i]
        pairs = {"g1": "gamma1", "wq": "wq", "wk": "wk", "wv": "wv", "wo": "wo",
                 "g2": "gamma2", "wg": "wgate", "wu": "wup", "wd": "wdown"}
        for short, key in pairs.items():
            inp[f"L{i}_{short}"] = lw[key]
        if qk_norm:
            inp[f"L{i}_qn"], inp[f"L{i}_kn"] = lw["q_norm"], lw["k_norm"]
    return inp


def emit_model_graph(compiler, cfg, n_layers, x_addr, layer_weights, scratch_a, scratch_b, pos,
                     rope_theta=1e6, attn_scale=None):
    """Graph-driven equivalent of tvm_backend._emit_model_kvcache_fp32: same ping-pong layer
    buffers, same return value."""
    qk_norm = layer_weights[0].get("q_norm") is not None
    mod = build_decode_graph(cfg, n_layers, qk_norm)
    bufs = [scratch_b, scratch_a]
    fixed = {BINDINGS_PER_LAYER * i + BINDINGS_PER_LAYER - 1: bufs[i % 2] for i in range(n_layers)}
    out, _ = byoc_graph.emit_graph(
        compiler, mod, _inputs(layer_weights, range(n_layers), x_addr, qk_norm),
        extern=attention_lowering(cfg, layer_weights, pos, rope_theta, attn_scale),
        fixed_outputs=fixed, batch_alloc=True)
    return out


def make_layer_emitter(cfg, layer_weights, pos, rope_theta=1e6, attn_scale=None):
    """A `layer_emitter(compiler, i, x_in, out, lw)` for
    tvm_backend._emit_model_kvcache_fp32_relocated: compiles layer i through the graph path."""
    def emit_layer(compiler, i, x_in, out, lw):
        qk_norm = lw.get("q_norm") is not None
        mod = build_decode_graph(cfg, 1, qk_norm)
        # one-layer graph: its parameters are named L0_*, the extern call's layer suffix is 0,
        # but the weights/caches belong to layer i.
        inp = _inputs({0: lw}, [0], x_in, qk_norm)
        byoc_graph.emit_graph(
            compiler, mod, inp, extern=attention_lowering(cfg, {0: lw}, pos, rope_theta, attn_scale),
            fixed_outputs={BINDINGS_PER_LAYER - 1: out}, batch_alloc=True)
    return emit_layer
