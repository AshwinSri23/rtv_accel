#!/usr/bin/env python3
"""
gemma4_model.py -- run a Gemma-4 text model (dense MLP + routed experts in every layer) on the simulated chip,
with HOST-SEGMENTED routing (same scheme as moe_model.py): each layer is two device programs and the host
picks the top-k experts between them, all on one persistent `rtv_sim --serve` session.

Per layer (Hugging Face Gemma4TextDecoderLayer, FP32 activations):

  device A:  a   = RMSNorm_postattn(Attn(RMSNorm_in(x)));            x1 = x + a
             h1  = RMSNorm_post1(GeluMLP(RMSNorm_preff(x1)))                      (dense branch)
             en  = RMSNorm_preff2(x1);   logits = (RMSNorm_noscale(x1) * router.scale * hidden^-.5) @ Wr
  host:      p = softmax(logits); top-k; renormalise; * per_expert_scale
  device B:  h2  = RMSNorm_post2(sum_j w_j GeluExpert_j(en))                      (expert branch)
             x'  = (x1 + RMSNorm_postff(h1 + h2)) * layer_scalar

Gemma specifics handled: attention scale 1.0 (q/k norms carry it), weightless V norm, K=V on full-attention
layers (no v_proj), per-layer head_dim / KV heads, "proportional" partial RoPE on full layers, GELU(tanh) built
from existing ISA ops (tvm_backend._emit_gelu_tanh_fp32), sqrt(hidden) embedding scale and final logit softcap
on the host. Sliding-window masking is NOT implemented (contexts must stay within the window).

    python toolchain/gemma4_model.py --model <dir> --prompt-ids 5,17,101 --steps 4
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "rtv_tvm"))

from compiler_state import NPUCompiler, DType, _padded  # noqa: E402
from hf_loader import SafeTensorsModel                   # noqa: E402
import tvm_backend                                       # noqa: E402
from generate import _SimSession, _SESSION_HEADROOM      # noqa: E402

M = 16


def _t(m, n):
    return m.tensor(n).astype(np.float32)


def load_gemma4_hf(model_dir):
    """Hugging Face Gemma4ForCausalLM / text checkpoint -> (cfg, layers, embed, final_norm).
    Weight matrices are returned as [K, N] (input x output) like the rest of the toolchain."""
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    cfg = cfg.get("text_config", cfg)
    m = SafeTensorsModel(model_dir)
    pre = "model." if any(n.startswith("model.layers.0.") for n, _ in m.named_parameters()) else \
        "model.language_model."
    E, I = cfg["num_experts"], cfg["moe_intermediate_size"]
    per_layer = cfg.get("per_layer_config") or {}
    layers = []
    for li in range(cfg["num_hidden_layers"]):
        p = f"{pre}layers.{li}."
        ltype = cfg["layer_types"][li]
        over = per_layer.get(str(li), {})
        head_dim = over.get("head_dim", cfg["head_dim"])
        n_kv = over.get("num_key_value_heads", cfg["num_key_value_heads"])
        rp = cfg["rope_parameters"][ltype]
        rope_angles = None
        if rp.get("rope_type") == "proportional":
            rope_angles = int(rp.get("partial_rotary_factor", 1.0) * head_dim // 2)
        gu = _t(m, p + "experts.gate_up_proj")           # (E, 2I, hidden)
        down = _t(m, p + "experts.down_proj")            # (E, hidden, I)
        has_v = (p + "self_attn.v_proj.weight") in {n for n, _ in m.named_parameters()}
        layers.append(dict(
            type=ltype, head_dim=head_dim, n_kv=n_kv, theta=rp["rope_theta"], rope_angles=rope_angles,
            wq=_t(m, p + "self_attn.q_proj.weight").T.copy(), wk=_t(m, p + "self_attn.k_proj.weight").T.copy(),
            wv=_t(m, p + "self_attn.v_proj.weight").T.copy() if has_v else None,
            wo=_t(m, p + "self_attn.o_proj.weight").T.copy(),
            q_norm=_t(m, p + "self_attn.q_norm.weight"), k_norm=_t(m, p + "self_attn.k_norm.weight"),
            g_in=_t(m, p + "input_layernorm.weight"), g_post_attn=_t(m, p + "post_attention_layernorm.weight"),
            g_pre_ff=_t(m, p + "pre_feedforward_layernorm.weight"),
            g_post_ff=_t(m, p + "post_feedforward_layernorm.weight"),
            g_pre_ff2=_t(m, p + "pre_feedforward_layernorm_2.weight"),
            g_post_ff1=_t(m, p + "post_feedforward_layernorm_1.weight"),
            g_post_ff2=_t(m, p + "post_feedforward_layernorm_2.weight"),
            mlp_gate=_t(m, p + "mlp.gate_proj.weight").T.copy(), mlp_up=_t(m, p + "mlp.up_proj.weight").T.copy(),
            mlp_down=_t(m, p + "mlp.down_proj.weight").T.copy(),
            router=_t(m, p + "router.proj.weight"), router_scale=_t(m, p + "router.scale"),
            per_expert_scale=_t(m, p + "router.per_expert_scale"),
            experts=[dict(w_gate=np.ascontiguousarray(gu[e, :I].T), w_up=np.ascontiguousarray(gu[e, I:].T),
                          w_down=np.ascontiguousarray(down[e].T)) for e in range(E)],
            layer_scalar=float(_t(m, p + "layer_scalar").reshape(-1)[0]),
        ))
    embed = _t(m, pre + "embed_tokens.weight")
    return cfg, layers, embed, _t(m, pre + "norm.weight")


def generate_gemma4(source, prompt_ids, n_new, sim, hal, verbose=False, int4_group=0, trace=None):
    """Greedy-generate n_new tokens. `source` is a checkpoint dir or an already loaded
    (cfg, layers, embed, final_norm). Returns (all_ids, per_step) with per_step[i] = (top5_ids, top5_logits)."""
    cfg, layers, embed, final_norm = load_gemma4_hf(source) if isinstance(source, str) else source
    hidden, I, E = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["num_experts"]
    inter, top_k = cfg["intermediate_size"], cfg["top_k_experts"]
    n_heads, eps = cfg["num_attention_heads"], cfg["rms_norm_eps"]
    softcap = cfg.get("final_logit_softcapping")
    E_pad = (E + M - 1) // M * M
    n_layers = len(layers)
    total = len(prompt_ids) - 1 + n_new
    max_tiles = -(-total // M)
    w4 = 4
    for d in (hidden, I, inter) + tuple(l["head_dim"] for l in layers):
        if d % M:
            raise NotImplementedError(f"dimension {d} must be a multiple of {M}")

    image_parts = []
    cursor = [0x1000]

    def place_bytes(b):
        a = cursor[0]
        cursor[0] += (len(b) + 0xFFF + 0x1000) & ~0xFFF
        image_parts.append((a, b))
        return a

    def place(arr, quant=False):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if quant and int4_group:
            return place_bytes(tvm_backend.int4_weight_blob(*tvm_backend.quantize_int4_group(arr, int4_group)).tobytes())
        return place_bytes(arr.tobytes())

    def place_zeros(nbytes):
        a = cursor[0]
        cursor[0] += (nbytes + 0xFFF + 0x1000) & ~0xFFF
        return a

    def gamma(v):
        return place(np.tile(np.asarray(v, np.float32), (M, 1)))

    x_addr = place(np.zeros((M, hidden), np.float32))
    resid_addr = place(np.zeros((M, hidden), np.float32))
    h1_addr = place(np.zeros((M, hidden), np.float32))
    en_addr = place(np.zeros((M, hidden), np.float32))
    logits_addr = place(np.zeros((M, E_pad), np.float32))
    L = []
    for lw in layers:
        hd, nkv = lw["head_dim"], lw["n_kv"]
        rproj = np.zeros((hidden, E_pad), np.float32)
        rproj[:, :E] = lw["router"].T
        d = dict(
            wq=place(lw["wq"], True), wk=place(lw["wk"], True),
            wv=place(lw["wv"], True) if lw["wv"] is not None else None, wo=place(lw["wo"], True),
            q_norm=gamma(lw["q_norm"]), k_norm=gamma(lw["k_norm"]),
            g_in=gamma(lw["g_in"]), g_post_attn=gamma(lw["g_post_attn"]), g_pre_ff=gamma(lw["g_pre_ff"]),
            g_post_ff=gamma(lw["g_post_ff"]), g_pre_ff2=gamma(lw["g_pre_ff2"]),
            g_post_ff1=gamma(lw["g_post_ff1"]), g_post_ff2=gamma(lw["g_post_ff2"]),
            g_router=gamma(lw["router_scale"] * hidden ** -0.5),
            mlp_gate=place(lw["mlp_gate"], True), mlp_up=place(lw["mlp_up"], True),
            mlp_down=place(lw["mlp_down"], True), router=place(rproj),
            experts=[dict(w_gate=place(e["w_gate"], True), w_up=place(e["w_up"], True),
                          w_down=place(e["w_down"], True)) for e in lw["experts"]],
            k_cache=[], v_cache=[])
        for _ in range(nkv):
            kb = place_zeros(max_tiles * M * hd * w4)
            d["k_cache"].append([kb + t * M * hd * w4 for t in range(max_tiles)])
            vb = place_zeros(max_tiles * M * hd * w4)
            d["v_cache"].append([vb + t * M * hd * w4 for t in range(max_tiles)])
        L.append(d)
    staging_base = (cursor[0] + 0xFFF) & ~0xFFF
    image = bytearray(staging_base)
    for a, b in image_parts:
        image[a:a + len(b)] = b

    def fresh():
        c = NPUCompiler(hal)
        c.dtype = DType.FP32
        c.instruction_stream = bytearray()
        c._dram_cursor = staging_base
        c.rms_eps = eps
        c.int4_group = int4_group
        return c

    def norm(c, src, g, dst):
        tvm_backend._emit_rmsnorm_row_fp32(
            c, hidden, x_addr=src, gamma_addr=g, out_addr=dst,
            scratch_addr=c.alloc_staging(4), stage_addr=c.alloc_staging(M * _padded(hidden) * 4))

    def buf(c, width=None):
        return c.alloc_staging(M * (width or hidden) * 4)

    dbg_addrs = {}

    def program_a(li, pos, xin):
        lw, a = layers[li], L[li]
        c = fresh()
        if xin is not None:
            c._const_regions[x_addr] = xin.tobytes()
        n1 = buf(c)
        norm(c, x_addr, a["g_in"], n1)
        attn = buf(c)
        hd, nkv = lw["head_dim"], lw["n_kv"]
        tvm_backend._emit_attention_block_kvcache_fp32(
            c, hidden, n_heads, nkv, hd, x_addr=n1, wq_addr=a["wq"], wk_addr=a["wk"],
            wv_addr=a["wv"] if a["wv"] is not None else a["wk"], wo_addr=a["wo"], out_addr=attn, pos=pos,
            k_cache_tiles=a["k_cache"], v_cache_tiles=a["v_cache"], rope_theta=lw["theta"],
            q_norm_addr=a["q_norm"], k_norm_addr=a["k_norm"], attn_scale=None, v_norm=True,
            v_from_k=a["wv"] is None, rope_angles=lw["rope_angles"])
        an = buf(c)
        norm(c, attn, a["g_post_attn"], an)
        tvm_backend._emit_residual_add_fp32(c, x_addr, an, resid_addr, hidden)
        n2 = buf(c)
        norm(c, resid_addr, a["g_pre_ff"], n2)
        mlp = buf(c)
        tvm_backend._emit_swiglu_ffn_fp32(c, hidden, inter, n2, a["mlp_gate"], a["mlp_up"], a["mlp_down"],
                                          mlp, act="gelu_tanh")
        dbg_addrs.update(n2=n2, mlp=mlp)
        norm(c, mlp, a["g_post_ff1"], h1_addr)
        rn = buf(c)
        norm(c, resid_addr, a["g_router"], rn)
        tvm_backend._emit_matmul_fp32(c, M, E_pad, hidden, rn, a["router"], logits_addr)
        norm(c, resid_addr, a["g_pre_ff2"], en_addr)
        return c

    def program_b(li, idx, wts):
        lw, a = layers[li], L[li]
        c = fresh()
        y = buf(c)
        tvm_backend._emit_moe_ffn_fp32(c, hidden, I, en_addr, [a["experts"][i] for i in idx],
                                       [float(v) for v in wts], y, act="gelu_tanh")
        h2 = buf(c)
        norm(c, y, a["g_post_ff2"], h2)
        hsum = buf(c)
        tvm_backend._emit_residual_add_fp32(c, h1_addr, h2, hsum, hidden)
        hn = buf(c)
        norm(c, hsum, a["g_post_ff"], hn)
        x2 = buf(c)
        tvm_backend._emit_residual_add_fp32(c, resid_addr, hn, x2, hidden)
        sc = tvm_backend._emit_scale_fp32(c, x2, hidden, lw["layer_scalar"])
        tvm_backend._emit_copy_slice_fp32(c, sc, x_addr, hidden, width=hidden)
        return c

    with tempfile.TemporaryDirectory(prefix="rtv_gemma4_") as wd:
        img = os.path.join(wd, "state.img")
        open(img, "wb").write(bytes(image))
        sess = _SimSession(sim, hal, img, len(image) + _SESSION_HEADROOM, wd)
        try:
            def step(token, pos):
                xin = np.zeros((M, hidden), np.float32)
                xin[0] = embed[token] * np.float32(hidden ** 0.5)
                last = None
                for li in range(n_layers):
                    ca = program_a(li, pos, xin if li == 0 else None)
                    tr = trace is not None and pos == len(prompt_ids) - 1
                    dbg = [(resid_addr, M * hidden * 4, "r"), (h1_addr, M * hidden * 4, "h"),
                           (en_addr, M * hidden * 4, "e"), (dbg_addrs["n2"], M * hidden * 4, "n"),
                           (dbg_addrs["mlp"], M * hidden * 4, "m")] if tr else []
                    raw, *dres = sess.run_step(ca, extra_dumps=[(logits_addr, M * E_pad * 4, "l")] + dbg)
                    lg = np.frombuffer(raw, np.float32).reshape(M, E_pad)[0, :E].astype(np.float64)
                    p = np.exp(lg - lg.max())
                    p /= p.sum()
                    idx = np.argsort(-p, kind="stable")[:top_k]
                    wts = p[idx] / p[idx].sum() * layers[li]["per_expert_scale"][idx]
                    cb = program_b(li, idx, wts)
                    dumps = [(x_addr, M * hidden * 4, "x")] if (li == n_layers - 1 or tr) else ()
                    res = sess.run_step(cb, extra_dumps=dumps)
                    if tr:
                        row = lambda b: np.frombuffer(b, np.float32).reshape(M, hidden)[0].copy()  # noqa: E731
                        trace.append(dict(layer=li, experts=[int(i) for i in idx], weights=[float(v) for v in wts],
                                          logits=lg.copy(), resid=row(dres[0]), h1=row(dres[1]), en=row(dres[2]),
                                          hidden=row(res[0]), n2=row(dres[3]), mlp=row(dres[4])))
                    if li == n_layers - 1:
                        last = np.frombuffer(res[0], np.float32).reshape(M, hidden)[0]
                return last

            def head(h):
                h = h.astype(np.float64)
                n = h / np.sqrt((h ** 2).mean() + eps) * final_norm.astype(np.float64)
                lg = n @ embed.T.astype(np.float64)
                if softcap:
                    lg = np.tanh(lg / softcap) * softcap
                return lg

            ids = list(prompt_ids)
            per_step = []
            for pos in range(total):
                h = step(ids[pos], pos)
                if pos >= len(prompt_ids) - 1:
                    lg = head(h)
                    top = np.argsort(-lg)[:5]
                    per_step.append(([int(i) for i in top], [float(lg[i]) for i in top]))
                    ids.append(int(top[0]))
                    if verbose:
                        print(f"  pos {pos}: token {ids[-1]}", file=sys.stderr)
        finally:
            sess.close()
    return ids, per_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-ids", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--sim", default=os.path.join(ROOT, "build", "rtv_sim"))
    ap.add_argument("--hal", default=os.path.join(_HERE, "hal_config.json"))
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    ids, steps = generate_gemma4(a.model, [int(x) for x in a.prompt_ids.split(",")], a.steps,
                                 a.sim, a.hal, verbose=a.verbose)
    print("ids:", ids)
    for i, (t, l) in enumerate(steps):
        print(f"  step {i}: top5 {t}  logits {[round(v, 4) for v in l]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
