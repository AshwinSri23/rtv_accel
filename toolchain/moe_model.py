#!/usr/bin/env python3
"""
moe_model.py -- run a Qwen3-MoE checkpoint on the simulated chip, with the experts
chosen by HOST-SEGMENTED routing (TODO Phase 5.1 option (a)).

The ISA has no data-dependent addressing, and which experts a token uses depends on
router logits the device itself computes mid-layer. So every layer is TWO device
programs with a host step between them, all on one persistent `rtv_sim --serve`
session (weights and the KV-cache stay resident):

  device A: n1 = RMSNorm(x); x1 = x + Attention(n1); n2 = RMSNorm(x1); logits = n2 @ Wr
  host:     softmax over all experts, top-k, renormalise (norm_topk_prob)
  device B: x' = x1 + sum_j w_j * SwiGLU_{e_j}(n2)      (expert addresses baked in)

Embedding lookup, the final RMSNorm, lm_head and argmax stay on the host exactly as in
generate.py. Prompt tokens run through single-token decode steps (MoE routing differs
per row, so the batched-prefill path does not apply).

Checkpoint layout handled here (transformers 5 Qwen3Moe): fused experts
  mlp.experts.gate_up_proj (E, 2*I, hidden)  -> gate = [:, :I], up = [:, I:]
  mlp.experts.down_proj    (E, hidden, I)
  mlp.router.weight        (E, hidden)
Per-expert keys (mlp.experts.N.{gate,up,down}_proj.weight) and the older mlp.gate.weight
router name are also accepted.

    python toolchain/moe_model.py --model <dir> --prompt-ids 785,6722,315 --steps 4
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


def load_moe_checkpoint(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    m = SafeTensorsModel(model_dir)
    names = {name for name, _ in m.named_parameters()}

    def has(n):
        return n in names

    def t(n):
        return m.tensor(n).astype(np.float32)

    E, I = cfg["num_experts"], cfg["moe_intermediate_size"]
    layers = []
    for li in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{li}."
        if cfg.get("mlp_only_layers") and li in cfg["mlp_only_layers"]:
            raise NotImplementedError("dense (mlp_only) layers inside an MoE model")
        router_name = p + ("mlp.router.weight" if has(p + "mlp.router.weight") else "mlp.gate.weight")
        if has(p + "mlp.experts.gate_up_proj"):
            gu = t(p + "mlp.experts.gate_up_proj")            # (E, 2I, hidden)
            down = t(p + "mlp.experts.down_proj")             # (E, hidden, I)
            experts = [dict(w_gate=np.ascontiguousarray(gu[e, :I].T),
                            w_up=np.ascontiguousarray(gu[e, I:].T),
                            w_down=np.ascontiguousarray(down[e].T)) for e in range(E)]
        else:
            experts = [dict(
                w_gate=np.ascontiguousarray(t(p + f"mlp.experts.{e}.gate_proj.weight").T),
                w_up=np.ascontiguousarray(t(p + f"mlp.experts.{e}.up_proj.weight").T),
                w_down=np.ascontiguousarray(t(p + f"mlp.experts.{e}.down_proj.weight").T))
                for e in range(E)]
        router = t(router_name)                               # (E, hidden)
        layers.append(dict(
            gamma1=t(p + "input_layernorm.weight"), gamma2=t(p + "post_attention_layernorm.weight"),
            wq=np.ascontiguousarray(t(p + "self_attn.q_proj.weight").T),
            wk=np.ascontiguousarray(t(p + "self_attn.k_proj.weight").T),
            wv=np.ascontiguousarray(t(p + "self_attn.v_proj.weight").T),
            wo=np.ascontiguousarray(t(p + "self_attn.o_proj.weight").T),
            q_norm=t(p + "self_attn.q_norm.weight"), k_norm=t(p + "self_attn.k_norm.weight"),
            router=router, experts=experts))
    embed = t("model.embed_tokens.weight")
    lm_head = t("lm_head.weight") if has("lm_head.weight") else embed
    return cfg, layers, embed, lm_head, t("model.norm.weight")


def generate_moe(model_dir, prompt_ids, n_new, sim, hal, verbose=False, trace=None,
                 int8_weights=False):
    """Greedy-generate n_new tokens. Returns (all_ids, per_step) where per_step[i] is the
    (top5_ids, top5_logits) at the position that produced generated token i."""
    cfg, layers, embed, lm_head, final_norm = load_moe_checkpoint(model_dir)
    hidden, I, E = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["num_experts"]
    top_k, norm_topk = cfg["num_experts_per_tok"], cfg.get("norm_topk_prob", True)
    n_heads, n_kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim") or hidden // n_heads
    eps, theta = cfg["rms_norm_eps"], cfg.get("rope_theta", 1e6)
    if E > M:
        raise NotImplementedError(f"router is one 16-wide tile; num_experts={E} > {M}")
    for d in (hidden, I, head_dim):
        if d % M:
            raise NotImplementedError(f"dimension {d} must be a multiple of {M}")
    n_layers = len(layers)
    total = len(prompt_ids) - 1 + n_new
    attn_scale = float(head_dim) ** -0.5

    # --- DRAM image: attention weights + KV-cache via the standard builder (it also
    # wants FFN weights per layer; give it tiny zero placeholders), then router and
    # experts placed after it.
    builder_layers = []
    for lw in layers:
        b = {k: lw[k] for k in ("gamma1", "gamma2", "wq", "wk", "wv", "wo", "q_norm", "k_norm")}
        b.update(wgate=np.zeros((hidden, M), np.float32), wup=np.zeros((hidden, M), np.float32),
                 wdown=np.zeros((M, hidden), np.float32))
        builder_layers.append(b)
    comp0 = NPUCompiler(hal)
    comp0.dtype = DType.FP32
    dram, x_addr, lws, _sa, _sb, _ = tvm_backend.build_kvcache_model_dram_image(
        comp0, hidden, n_layers, n_kv, head_dim, max_seq_len=total,
        layer_weights_np=builder_layers, int8_weights=int8_weights)
    cursor = [(comp0._dram_cursor + 0xFFF) & ~0xFFF]
    extra = []

    def place(arr, quant=False):
        # quant: store an expert matrix as INT8 bytes + FP32 per-channel scales (ISA v1.8)
        arr = tvm_backend.int8_weight_blob(arr) if (quant and int8_weights) else \
            np.ascontiguousarray(arr, dtype=np.float32)
        a = cursor[0]
        cursor[0] += (arr.nbytes + 0xFFF) & ~0xFFF
        extra.append((a, arr))
        return a

    routers, expert_addrs = [], []
    for lw in layers:
        rp = np.zeros((hidden, M), np.float32)
        rp[:, :E] = lw["router"].T
        routers.append(place(rp))
        expert_addrs.append([dict(w_gate=place(e["w_gate"], True), w_up=place(e["w_up"], True),
                                  w_down=place(e["w_down"], True)) for e in lw["experts"]])
    resid_addr = place(np.zeros((M, hidden), np.float32))
    norm2_addr = place(np.zeros((M, hidden), np.float32))
    logits_addr = place(np.zeros((M, M), np.float32))
    staging_base = cursor[0]
    image = bytearray(max(len(dram), staging_base))
    image[:len(dram)] = dram
    for a, arr in extra:
        image[a:a + arr.nbytes] = arr.tobytes()

    def fresh():
        c = NPUCompiler(hal)
        c.dtype = DType.FP32
        c.instruction_stream = bytearray()
        c._dram_cursor = staging_base
        c.rms_eps = eps
        c.int8_weights = int8_weights
        return c

    def norm(c, src, gamma, dst):
        tvm_backend._emit_rmsnorm_row_fp32(
            c, hidden, x_addr=src, gamma_addr=gamma, out_addr=dst,
            scratch_addr=c.alloc_staging(4), stage_addr=c.alloc_staging(M * _padded(hidden) * 4))

    with tempfile.TemporaryDirectory(prefix="rtv_moe_") as wd:
        img = os.path.join(wd, "state.img")
        open(img, "wb").write(bytes(image))
        sess = _SimSession(sim, hal, img, len(image) + _SESSION_HEADROOM, wd)
        try:
            def step(token, pos):
                """One token through every layer; returns the final hidden row."""
                xin = np.zeros((M, hidden), np.float32)
                xin[0] = embed[token]
                last = None
                for li, lw in enumerate(lws):
                    a = fresh()
                    if li == 0:
                        a._const_regions[x_addr] = xin.tobytes()
                    n1 = a.alloc_staging(M * hidden * 4)
                    norm(a, x_addr, lw["gamma1"], n1)
                    attn = a.alloc_staging(M * hidden * 4)
                    tvm_backend._emit_attention_block_kvcache_fp32(
                        a, hidden, n_heads, n_kv, head_dim, x_addr=n1, wq_addr=lw["wq"],
                        wk_addr=lw["wk"], wv_addr=lw["wv"], wo_addr=lw["wo"], out_addr=attn,
                        pos=pos, k_cache_tiles=lw["k_cache"], v_cache_tiles=lw["v_cache"],
                        rope_theta=theta, q_norm_addr=lw["q_norm"], k_norm_addr=lw["k_norm"],
                        attn_scale=attn_scale)
                    tvm_backend._emit_residual_add_fp32(a, x_addr, attn, resid_addr, hidden)
                    norm(a, resid_addr, lw["gamma2"], norm2_addr)
                    tvm_backend._emit_matmul_fp32(a, M, M, hidden, norm2_addr, routers[li],
                                                  logits_addr)
                    (raw,) = sess.run_step(a, extra_dumps=[(logits_addr, M * M * 4, "l")])
                    lg = np.frombuffer(raw, np.float32).reshape(M, M)[0, :E].astype(np.float64)

                    p = np.exp(lg - lg.max())
                    p /= p.sum()
                    idx = np.argsort(-p, kind="stable")[:top_k]
                    w = p[idx] / p[idx].sum() if norm_topk else p[idx]

                    b = fresh()
                    tvm_backend._emit_moe_block_fp32(
                        b, hidden, I, resid_addr, norm2_addr,
                        [expert_addrs[li][i] for i in idx], [float(v) for v in w], x_addr)
                    # `trace` (a list) collects, for the LAST prompt position only, each
                    # layer's chosen experts + weights and output hidden row, for
                    # layer-by-layer comparison against Hugging Face.
                    tr = trace is not None and pos == len(prompt_ids) - 1
                    dumps = [(x_addr, M * hidden * 4, "x")] if (li == n_layers - 1 or tr) else ()
                    res = sess.run_step(b, extra_dumps=dumps)
                    if tr:
                        trace.append(dict(layer=li, experts=[int(i) for i in idx],
                                          weights=[float(v) for v in w], probs=p.tolist(),
                                          hidden=np.frombuffer(res[0], np.float32)
                                          .reshape(M, hidden)[0].copy()))
                    if li == n_layers - 1:
                        last = np.frombuffer(res[0], np.float32).reshape(M, hidden)[0]
                return last

            def head(h):
                h = h.astype(np.float64)
                n = h / np.sqrt((h ** 2).mean() + eps) * final_norm.astype(np.float64)
                return n @ lm_head.T.astype(np.float64)

            ids = list(prompt_ids)
            per_step = []
            for pos in range(total):
                tok = ids[pos]
                h = step(tok, pos)
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
    ap.add_argument("--prompt-ids", required=True, help="comma-separated token ids")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--sim", default=os.path.join(ROOT, "build", "rtv_sim"))
    ap.add_argument("--hal", default=os.path.join(_HERE, "hal_config.json"))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--int8-weights", action="store_true",
                    help="attention and expert weights as INT8 + per-channel scales (ISA v1.8)")
    a = ap.parse_args()
    ids, steps = generate_moe(a.model, [int(x) for x in a.prompt_ids.split(",")], a.steps,
                              a.sim, a.hal, a.verbose, int8_weights=a.int8_weights)
    print("ids     :", ids)
    for i, (ti, tl) in enumerate(steps):
        print(f"  step {i}: top5 {ti}  logits {[round(v, 4) for v in tl]}")


if __name__ == "__main__":
    main()
