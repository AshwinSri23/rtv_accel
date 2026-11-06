#!/usr/bin/env python3
"""
gemma4_model.py -- run a Gemma-4 text model (dense MLP + routed experts in every layer) on the simulated chip,
with HOST-SEGMENTED routing (the host picks the experts between two device programs): each layer is two device programs and the host
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
on the host. Sliding-window attention (last `window` keys; older cache tiles skipped) on the sliding layers.

    python toolchain/gemma4_model.py --model <dir> --prompt-ids 5,17,101 --steps 4
"""
import argparse
import contextlib
import types
import json
import os
import sys
import tempfile
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "rtv_tvm"))

from compiler_state import NPUCompiler, DType, _padded  # noqa: E402
from hf_loader import SafeTensorsModel                   # noqa: E402
import quant_planes                                      # noqa: E402
import tvm_backend                                       # noqa: E402
from generate import _SimSession, _SESSION_HEADROOM      # noqa: E402
from decode_step_cache import DecodeStepCache            # noqa: E402

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


def _blob_bytes(K, N, int4_group):
    return tvm_backend.int4_blob_nbytes(K, N, int4_group) if int4_group else K * N * 4


_COMPILE_JOB = None


def _noop(_):
    return None


def _compiler_state(c):
    """What a derived decode step needs from a compiled program, as plain picklable data."""
    return dict(stream=bytes(c.instruction_stream), regions=dict(c._const_regions), cache=dict(c._const_cache),
                cursors=(c._dram_cursor, c._staging_cursor, c._staging_peak))


def _run_compile_job(rep, pos):
    return _COMPILE_JOB(rep, pos)


@contextlib.contextmanager
def open_gemma4(source, max_total, sim, hal, verbose=False, int4_group=0, trace=None, trace_pos=None,
                staging_reserve=1 << 30, stats=None, image_dir=None):
    """Open a model session: build the DRAM image, start `rtv_sim --serve`, yield `s` with
        s.step(token, pos) -> the final hidden row after running every layer for `token` at position `pos`
        s.head(h)          -> the next-token logits (final norm, tied lm_head, softcap; on the host)
        s.max_total        -> positions the KV cache holds
    Positions may be rewound: stepping `pos` again overwrites that KV row and later rows are masked out, so a chat
    can keep the cache of a common prefix and re-feed only the part that changed. `source` is a checkpoint dir or an
    already loaded (cfg, layers, embed, final_norm); matrices may be lazy gemma4_gguf.Q4Mat objects (int4_group=32).
    generate_gemma4 below is the one-shot driver built on this.

    Compile cost: the per-layer programs are NOT recompiled for every layer. Program A (attention + dense branch +
    router) is compiled for one representative layer per layer TYPE (sliding / full) and its LOAD/STORE addresses are
    relocated onto every other layer of that type (tvm_backend._relocate_instructions, same mechanism as the dense
    model); successive positions of one KV tile are additionally derived rather than recompiled (DecodeStepCache).
    Program B (the chosen experts) does not depend on the position at all: one template, relocated per (layer, chosen
    experts), with the route weights and layer scalar patched in as constant tiles. RTV_NO_GEMMA_RELOC=1 compiles
    every program for real (the oracle the relocation is tested against)."""
    cfg, layers, embed, final_norm = load_gemma4_hf(source) if isinstance(source, str) else source
    hidden, I, E = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["num_experts"]
    inter, top_k = cfg["intermediate_size"], cfg["top_k_experts"]
    n_heads, eps = cfg["num_attention_heads"], cfg["rms_norm_eps"]
    softcap = cfg.get("final_logit_softcapping")
    E_pad = (E + M - 1) // M * M
    n_layers = len(layers)
    total = max_total
    max_tiles = -(-total // M)
    w4 = 4
    reloc = not os.environ.get("RTV_NO_GEMMA_RELOC")
    for d in (hidden, I, inter) + tuple(l["head_dim"] for l in layers):
        if d % M:
            raise NotImplementedError(f"dimension {d} must be a multiple of {M}")
    window = cfg.get("sliding_window")

    if image_dir:
        os.makedirs(image_dir, exist_ok=True)
    wd_ctx = tempfile.TemporaryDirectory(prefix="rtv_gemma4_", dir=image_dir)
    wd = wd_ctx.name
    img = os.path.join(wd, "state.img")
    img_f = open(img, "wb")          # streamed: a real model's image is many GB and is never held in Python
    cursor = [0x1000]

    def place_bytes(b):
        a = cursor[0]
        cursor[0] += (len(b) + 0xFFF + 0x1000) & ~0xFFF
        img_f.seek(a)
        img_f.write(memoryview(b))
        return a

    qreg, gsums = {}, {}          # planar (Q4_K / Q8_0 / Q6_K) weights: primary address -> spec; (K, group) -> G blob

    def place(arr, quant=False):
        if hasattr(arr, "spec"):                   # QMat: INT4W planes + per-group correction (quant_planes)
            sp = arr.spec()
            planes = [(place_bytes(tvm_backend.int4_weight_blob(v, s).tobytes()), g) for v, s, g in sp["planes"]]
            reg = dict(planes=planes, corr=None)
            if sp["corr"] is not None:
                g = sp["corr"][1]
                reg["corr"] = (place_bytes(quant_planes.corr_matrix(sp, arr.K).tobytes()), g)
                if (arr.K, g) not in gsums:
                    gsums[(arr.K, g)] = place_bytes(tvm_backend.group_sum_blob(arr.K, g)[0].tobytes())
            qreg[planes[0][0]] = reg
            return planes[0][0]
        if hasattr(arr, "blob"):                   # lazy Q4_0 weight (gemma4_gguf.Q4Mat): already INT4W g32
            if int4_group != 32:
                raise ValueError("GGUF Q4_0 weights need int4_group=32")
            return place_bytes(arr.blob())
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

    # Layout: everything small (activations, norm gains, router, KV caches) and the staging window sit BELOW 4 GiB
    # (32-bit addresses); the big INT4 weights follow, above it if the model needs that (LOAD_TILE takes 36 bits).
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
            q_norm=gamma(lw["q_norm"]), k_norm=gamma(lw["k_norm"]),
            g_in=gamma(lw["g_in"]), g_post_attn=gamma(lw["g_post_attn"]), g_pre_ff=gamma(lw["g_pre_ff"]),
            g_post_ff=gamma(lw["g_post_ff"]), g_pre_ff2=gamma(lw["g_pre_ff2"]),
            g_post_ff1=gamma(lw["g_post_ff1"]), g_post_ff2=gamma(lw["g_post_ff2"]),
            g_router=gamma(lw["router_scale"] * hidden ** -0.5), router=place(rproj), k_cache=[], v_cache=[])
        for _ in range(nkv):
            kb = place_zeros(max_tiles * M * hd * w4)
            d["k_cache"].append([kb + t * M * hd * w4 for t in range(max_tiles)])
            vb = place_zeros(max_tiles * M * hd * w4)
            d["v_cache"].append([vb + t * M * hd * w4 for t in range(max_tiles)])
        L.append(d)
    staging_base = (cursor[0] + 0xFFF) & ~0xFFF
    cursor[0] = staging_base + staging_reserve
    for lw, d in zip(layers, L):
        d.update(wq=place(lw["wq"], True), wk=place(lw["wk"], True),
                 wv=place(lw["wv"], True) if lw["wv"] is not None else None, wo=place(lw["wo"], True),
                 mlp_gate=place(lw["mlp_gate"], True), mlp_up=place(lw["mlp_up"], True),
                 mlp_down=place(lw["mlp_down"], True),
                 experts=[dict(w_gate=place(e["w_gate"], True), w_up=place(e["w_up"], True),
                               w_down=place(e["w_down"], True)) for e in lw["experts"]])
        if verbose:
            print(f"  placed weights, image {cursor[0] / 2**30:.2f} GiB", file=sys.stderr)
    image_bytes = (cursor[0] + 0xFFF) & ~0xFFF
    img_f.truncate(image_bytes)
    img_f.close()

    def fresh():
        c = NPUCompiler(hal)
        c.dtype = DType.FP32
        c.instruction_stream = bytearray()
        c._dram_cursor = staging_base
        c.rms_eps = eps
        c.int4_group = int4_group
        c.qweights, c.group_sums = qreg, gsums
        if not os.environ.get("RTV_FULL_A_TILES"):
            c.gemm_a_rows = 1        # decode: one real row, load 64 B of activation per GEMM step, not a 1 KB tile
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
            v_from_k=a["wv"] is None, rope_angles=lw["rope_angles"],
            window=window if lw["type"] == "sliding_attention" else None)
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

    def program_b(li, idx, wts, scalar=None):
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
        sc = tvm_backend._emit_scale_fp32(c, x2, hidden, lw["layer_scalar"] if scalar is None else scalar)
        tvm_backend._emit_copy_slice_fp32(c, sc, x_addr, hidden, width=hidden)
        return c

    # ---- relocation -------------------------------------------------------------------------------------------
    def wroles(name, addr, K, N):
        """The DRAM regions of one weight matrix: one region for INT4W g32, or one per plane (+ the correction)."""
        reg = qreg.get(addr)
        if reg is None:
            return {name: (addr, _blob_bytes(K, N, int4_group))}
        r = {f"{name}#p{i}": (a_, tvm_backend.int4_blob_nbytes(K, N, g)) for i, (a_, g) in enumerate(reg["planes"])}
        if reg["corr"]:
            r[f"{name}#c"] = (reg["corr"][0], quant_planes.kg_pad(K, reg["corr"][1]) * N * 4)
        return r

    def a_roles(li, nt):
        lw, a = layers[li], L[li]
        hd, nkv = lw["head_dim"], lw["n_kv"]
        q_dim, kv_dim = n_heads * hd, nkv * hd
        r = {"router": (a["router"], hidden * E_pad * 4),
             "q_norm": (a["q_norm"], M * hd * 4), "k_norm": (a["k_norm"], M * hd * 4)}
        r.update(wroles("wq", a["wq"], hidden, q_dim))
        r.update(wroles("wk", a["wk"], hidden, kv_dim))
        r.update(wroles("wo", a["wo"], q_dim, hidden))
        r.update(wroles("mlp_gate", a["mlp_gate"], hidden, inter))
        r.update(wroles("mlp_up", a["mlp_up"], hidden, inter))
        r.update(wroles("mlp_down", a["mlp_down"], inter, hidden))
        if a["wv"] is not None:
            r.update(wroles("wv", a["wv"], hidden, kv_dim))
        for g in ("g_in", "g_post_attn", "g_pre_ff", "g_post_ff1", "g_router", "g_pre_ff2"):
            r[g] = (a[g], M * hidden * 4)
        for h in range(nkv):
            for t in range(nt):
                r[f"k{h}_{t}"] = (a["k_cache"][h][t], M * hd * 4)
                r[f"v{h}_{t}"] = (a["v_cache"][h][t], M * hd * 4)
        return r

    def b_roles(li, idx):
        a = L[li]
        r = {"g_post_ff2": (a["g_post_ff2"], M * hidden * 4), "g_post_ff": (a["g_post_ff"], M * hidden * 4)}
        for j, e in enumerate(idx):
            ex = a["experts"][int(e)]
            r.update(wroles(f"e{j}_gate", ex["w_gate"], hidden, I))
            r.update(wroles(f"e{j}_up", ex["w_up"], hidden, I))
            r.update(wroles(f"e{j}_down", ex["w_down"], I, hidden))
        return r

    def clone(tpl, stream, regions):
        c = fresh()
        c.instruction_stream = bytearray(stream)
        c._const_regions = dict(regions)
        c._const_cache = dict(tpl._const_cache)
        c._dram_cursor, c._staging_cursor, c._staging_peak = tpl._dram_cursor, tpl._staging_cursor, tpl._staging_peak
        return c

    kind = lambda m: None if m is None else getattr(m, "kind", "q4_0")  # noqa: E731
    qsig = lambda l: tuple(kind(l[k]) for k in ("wq", "wk", "wv", "wo", "mlp_gate", "mlp_up", "mlp_down"))  # noqa: E731
    esig = [tuple(kind(l["experts"][0][k]) for k in ("w_gate", "w_up", "w_down")) for l in layers]
    type_key = [(l["head_dim"], l["n_kv"], l["wv"] is not None, l["type"]) + qsig(l) for l in layers]
    rep_of = {}
    for li, k in enumerate(type_key):
        rep_of.setdefault(k, li)
    step_caches = {k: DecodeStepCache(M, x_addr) for k in rep_of}
    a_templates = {}                     # type key -> (pos, compiler) for the current position
    b_state = {}                         # template compiler, sentinel region addresses, rep roles
    SENT_W, SENT_S = 1234.5, 4321.25

    # ── Compile-ahead worker processes ──
    # The real attention compiles (TVM scheduling, Python emit) are CPU-bound Python, so threads cannot overlap them
    # (GIL). They depend only on (layer type, position), never on the token, so forked workers compile the programs
    # of the current and next KV tiles while this process drives the simulator. Workers inherit the placed weights
    # and layer tables copy-on-write; they return only the compiler state a derived step needs.
    n_workers = int(os.environ.get("RTV_COMPILE_WORKERS", min(3, max(0, (os.cpu_count() or 1) // 2 - 1))))
    pool, ahead = None, {}
    if reloc and n_workers > 0:
        import concurrent.futures as cf
        import multiprocessing as mp
        global _COMPILE_JOB
        _COMPILE_JOB = lambda rep, pos: _compiler_state(program_a(rep, pos, None))     # noqa: E731
        pool = cf.ProcessPoolExecutor(n_workers, mp_context=mp.get_context("fork"))
        list(pool.map(_noop, range(n_workers)))            # start every worker now: before the simulator's pipes exist

    def submit_ahead(pos):
        t = pos // M
        want = [(k, p) for k in rep_of for p in (pos, pos + 1) if p // M == t and p < total]
        want += [(k, p) for k in rep_of for tt in range(t + 1, t + 3) for p in (tt * M, tt * M + 1) if p < total]
        for k in [key for key in ahead if key[1] // M < t]:
            ahead.pop(k)
        for key in want:
            if key not in ahead:
                ahead[key] = pool.submit(_run_compile_job, rep_of[key[0]], key[1])

    def from_state(st):
        c = fresh()
        c.instruction_stream = bytearray(st["stream"])
        c._const_regions, c._const_cache = st["regions"], st["cache"]
        c._dram_cursor, c._staging_cursor, c._staging_peak = st["cursors"]
        return c

    def a_program(li, pos, xin):
        if not reloc:
            return program_a(li, pos, xin)
        k = type_key[li]
        tpl = a_templates.get(k)
        if tpl is None or tpl[0] != pos:
            rep = rep_of[k]
            comp = step_caches[k].derive(pos, fresh())
            if comp is None:
                if pool is not None:
                    submit_ahead(pos)
                    comp = from_state(ahead.pop((k, pos)).result())
                else:
                    comp = program_a(rep, pos, None)
                step_caches[k].observe(pos, comp)
                if stats is not None:
                    stats.setdefault("programs", {}).setdefault(f"A_{k[3]}", bytes(comp.instruction_stream))
            tpl = a_templates[k] = (pos, comp)
        comp = tpl[1]
        rep = rep_of[k]
        if li == rep:
            stream = comp.instruction_stream
        else:
            stream = tvm_backend._relocate_instructions(
                bytes(comp.instruction_stream), a_roles(rep, pos // M + 1), a_roles(li, pos // M + 1), 0, list(comp._const_cache.values()))
        c = clone(comp, stream, comp._const_regions)
        if xin is not None:
            c._const_regions[x_addr] = xin.tobytes()
        return c

    def b_program(li, idx, wts):
        if not reloc:
            return program_b(li, idx, wts)
        st = b_state.get(esig[li])               # one template per expert quantisation signature
        if st is None:
            rep = esig.index(esig[li])
            idx0 = list(range(top_k))
            tpl = program_b(rep, idx0, [SENT_W + j for j in range(top_k)], scalar=SENT_S)
            sent = {}
            for addr, data in tpl._const_regions.items():
                v = np.frombuffer(data, np.float32)
                if v.size and (v == v[0]).all():
                    for j in range(top_k):
                        if v[0] == np.float32(SENT_W + j):
                            sent[addr] = ("w", j)
                    if v[0] == np.float32(SENT_S):
                        sent[addr] = ("s", 0)
            if sum(1 for v in sent.values() if v[0] == "w") != top_k or sum(1 for v in sent.values() if v[0] == "s") != 1:
                raise RuntimeError("could not locate the route-weight / layer-scalar constant tiles in program B")
            st = b_state[esig[li]] = dict(tpl=tpl, sent=sent, roles=b_roles(rep, idx0))
            if stats is not None:
                stats.setdefault("programs", {}).setdefault("B_experts", bytes(tpl.instruction_stream))
        tpl = st["tpl"]
        stream = tvm_backend._relocate_instructions(
            bytes(tpl.instruction_stream), st["roles"], b_roles(li, idx), 0, list(tpl._const_cache.values()))
        regions = dict(tpl._const_regions)
        for addr, (kind_, j) in st["sent"].items():
            val = wts[j] if kind_ == "w" else layers[li]["layer_scalar"]
            regions[addr] = np.full((M, hidden), val, np.float32).tobytes()
        return clone(tpl, stream, regions)

    with wd_ctx:
        sess = _SimSession(sim, hal, img, image_bytes + _SESSION_HEADROOM, wd)
        try:
            def step(token, pos):
                xin = np.zeros((M, hidden), np.float32)
                xin[0] = embed[token] * np.float32(hidden ** 0.5)
                last = None
                for li in range(n_layers):
                    ca = a_program(li, pos, xin if li == 0 else None)
                    tr = trace is not None and pos == trace_pos
                    dbg = [(resid_addr, M * hidden * 4, "r"), (h1_addr, M * hidden * 4, "h"),
                           (en_addr, M * hidden * 4, "e"), (dbg_addrs["n2"], M * hidden * 4, "n"),
                           (dbg_addrs["mlp"], M * hidden * 4, "m")] if tr else []
                    raw, *dres = sess.run_step(ca, extra_dumps=[(logits_addr, M * E_pad * 4, "l")] + dbg)
                    lg = np.frombuffer(raw, np.float32).reshape(M, E_pad)[0, :E].astype(np.float64)
                    p = np.exp(lg - lg.max())
                    p /= p.sum()
                    idx = np.argsort(-p, kind="stable")[:top_k]
                    wts = p[idx] / p[idx].sum() * layers[li]["per_expert_scale"][idx]
                    cb = b_program(li, idx, wts)
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
                lg = (embed @ n.astype(np.float32)).astype(np.float64)
                if softcap:
                    lg = np.tanh(lg / softcap) * softcap
                return lg

            yield types.SimpleNamespace(step=step, head=head, max_total=total)
        finally:
            sess.close()
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
    if stats is not None:
        stats.update({str(k): dict(c.stats) for k, c in step_caches.items() if k != "programs"})


def generate_gemma4(source, prompt_ids, n_new, sim, hal, verbose=False, int4_group=0, trace=None,
                    staging_reserve=1 << 30, stats=None, image_dir=None, stop_ids=(), score_out=None, on_token=None):
    """Greedy-generate n_new tokens. Returns (all_ids, per_step) with per_step[i] = (top5_ids, top5_logits).

    score_out (a list): teacher-forced scoring. Every position's logits are computed and, if ids[pos+1] is known
    (prompt_ids is then the whole sequence, n_new=0), (top5_ids, top5_logits, logprob_of_next_token) is appended."""
    total = len(prompt_ids) - 1 + n_new
    ids = list(prompt_ids)
    per_step = []
    with open_gemma4(source, total, sim, hal, verbose, int4_group, trace, len(prompt_ids) - 1, staging_reserve, stats,
                     image_dir) as s:
        t0 = time.time()
        for pos in range(total):
            h = s.step(ids[pos], pos)
            if score_out is not None and (pos % 16 == 15 or pos + 1 == total):
                el = time.time() - t0
                print(f"  scored {pos + 1}/{total} positions, {el / 60:.1f} min elapsed, "
                      f"~{el / (pos + 1) * (total - pos - 1) / 60:.0f} min left at the average pace", file=sys.stderr, flush=True)
            if score_out is not None and pos + 1 < len(prompt_ids):
                lg = s.head(h)
                top = np.argsort(-lg)[:5]
                m = lg.max()
                logp = float(lg[prompt_ids[pos + 1]] - m - np.log(np.exp(lg - m).sum()))
                score_out.append(([int(i) for i in top], [float(lg[i]) for i in top], logp))
            if pos >= len(prompt_ids) - 1:
                lg = s.head(h)
                top = np.argsort(-lg)[:5]
                per_step.append(([int(i) for i in top], [float(lg[i]) for i in top]))
                ids.append(int(top[0]))
                if verbose:
                    print(f"  pos {pos}: token {ids[-1]}", file=sys.stderr)
                if on_token is not None:
                    on_token(ids[-1])
                if ids[-1] in stop_ids:
                    break
    return ids, per_step


class Chat:
    """Multi-turn driver over an open session. Keeps the token ids whose KV rows are in the cache; a new turn is
    the whole re-rendered conversation, and only the part after the longest common prefix is run (the cache is
    rewound to that point: stepping a position again overwrites its KV row and later rows are masked)."""

    def __init__(self, session, stop_ids=()):
        self.s, self.stop, self.cache, self.fed = session, tuple(stop_ids), [], 0

    def feed(self, ids):
        if len(ids) > self.s.max_total:
            raise ValueError(f"{len(ids)} tokens exceed the {self.s.max_total}-position context")
        c = 0
        for a, b in zip(self.cache, ids):
            if a != b:
                break
            c += 1
        c = min(c, len(ids) - 1)             # the last token is always re-run: its hidden state seeds the reply
        del self.cache[c:]
        h = None
        for pos in range(c, len(ids)):
            h = self.s.step(ids[pos], pos)
            self.cache.append(ids[pos])
        self.fed = len(ids) - c
        return h

    def reply(self, ids, max_new, on_token=None):
        """Feed `ids`, then greedy-decode up to max_new tokens (stops after a stop id or when the context is full).
        Returns the reply ids, stop id included."""
        h = self.feed(ids)
        out = []
        for _ in range(max_new):
            t = int(np.argmax(self.s.head(h)))
            out.append(t)
            if on_token is not None:
                on_token(t)
            if t in self.stop or len(self.cache) >= self.s.max_total:
                break
            h = self.s.step(t, len(self.cache))
            self.cache.append(t)
        return out


def chat_main(a, source, group):
    """Interactive multi-turn chat. Each turn re-renders the whole conversation with the GGUF's template; only the
    tokens after the longest common prefix with what the cache holds are run (Chat). An empty line or Ctrl-D ends."""
    from gemma4_gguf import render_chat
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    stop = (1, 106)
    msgs = [{"role": "system", "content": a.system}] if a.system else []
    print(f"loading (the image build takes a minute or two); context {a.max_ctx} tokens", file=sys.stderr)
    with open_gemma4(source, a.max_ctx, a.sim, a.hal, int4_group=group,
                     image_dir=a.image_dir or os.path.expanduser("~/.cache/rtv_accel")) as s:
        chat = Chat(s, stop)
        while True:
            try:
                user = input("\nyou> ").strip()
            except EOFError:
                break
            if not user:
                break
            msgs.append({"role": "user", "content": user})
            ids = tok(render_chat(a.gguf, msgs), add_special_tokens=False)["input_ids"]
            if len(ids) + a.max_new > a.max_ctx:
                print(f"[context full: {len(ids)} tokens + {a.max_new} reply > {a.max_ctx}; restart with a larger --max-ctx]")
                msgs.pop()
                break
            seen = []

            def on_token(t):
                if t in stop:
                    return
                seen.append(t)
                text = tok.decode(seen)
                print(text[on_token.printed:], end="", flush=True)
                on_token.printed = len(text)
            on_token.printed = 0
            t0 = time.time()
            reply = chat.reply(ids, a.max_new, on_token)
            print(f"\n[{chat.fed} tokens fed, {len(reply)} generated, {time.time() - t0:.0f} s]", file=sys.stderr)
            msgs.append({"role": "assistant", "content": tok.decode([t for t in reply if t not in stop])})
    return 0


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="Hugging Face Gemma-4 checkpoint directory")
    src.add_argument("--gguf", help="llama.cpp Q4_0 GGUF (runs as INT4W g32)")
    ap.add_argument("--layers", type=int, default=None, help="GGUF only: run just the first N layers")
    ap.add_argument("--prompt-ids")
    ap.add_argument("--prompt", help="user message; rendered with the GGUF's own chat template (needs --gguf)")
    ap.add_argument("--system", default=None, help="optional system message")
    ap.add_argument("--tokenizer", default=os.path.expanduser("~/models/hf/gemma-4-12B-it-assistant"),
                    help="directory with a Gemma-4 tokenizer.json (the vocab is shared across the family)")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--chat", action="store_true",
                    help="interactive multi-turn chat (needs --gguf): the model and KV cache stay loaded between turns")
    ap.add_argument("--max-ctx", type=int, default=2048, help="--chat: KV cache capacity in tokens")
    ap.add_argument("--max-new", type=int, default=256, help="--chat: longest reply in tokens")
    ap.add_argument("--int4-group", type=int, default=0)
    ap.add_argument("--image-dir", default=None,
                    help="where the DRAM image file is written (a real model is ~16 GB; /tmp is often a small tmpfs)")
    ap.add_argument("--trace-out", help="save the per-layer hidden state at the last prompt position (.npz)")
    ap.add_argument("--sim", default=os.path.join(ROOT, "build", "rtv_sim"))
    ap.add_argument("--hal", default=os.path.join(_HERE, "hal_config.json"))
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    group = a.int4_group
    if a.gguf:
        from gemma4_gguf import load_gemma4_gguf
        source, group = load_gemma4_gguf(a.gguf, a.layers), 32
    else:
        source = a.model
    if a.chat:
        if not a.gguf:
            ap.error("--chat needs --gguf")
        return chat_main(a, source, group)
    trace = [] if a.trace_out else None
    tok = None
    stop = ()
    on_token = None
    if a.prompt is not None:
        if not a.gguf:
            ap.error("--prompt needs --gguf (the chat template is read from the GGUF)")
        from gemma4_gguf import render_chat
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        msgs = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": a.prompt}]
        prompt_ids = tok(render_chat(a.gguf, msgs), add_special_tokens=False)["input_ids"]
        assert prompt_ids[0] == 2, "the rendered prompt must start with <bos>"
        stop = (1, 106)                     # <eos>, <turn|>
        n_prompt, seen = len(prompt_ids), []

        def on_token(t):                     # stream: print the newly decoded text as tokens arrive
            if t in stop:
                return
            seen.append(t)
            text = tok.decode(seen)
            print(text[on_token.printed:], end="", flush=True)
            on_token.printed = len(text)
        on_token.printed = 0
    elif a.prompt_ids:
        prompt_ids = [int(x) for x in a.prompt_ids.split(",")]
    else:
        ap.error("give --prompt or --prompt-ids")
    ids, steps = generate_gemma4(source, prompt_ids, a.steps,
                                 a.sim, a.hal, verbose=a.verbose, int4_group=group, trace=trace, stop_ids=stop, on_token=on_token,
                                 image_dir=a.image_dir or (os.path.expanduser("~/.cache/rtv_accel") if a.gguf else None))
    if tok is not None:
        print()                              # the reply was streamed above
    if tok is None or a.verbose:
        print("ids:", ids)
        for i, (t, l) in enumerate(steps):
            print(f"  step {i}: top5 {t}  logits {[round(v, 4) for v in l]}")
    if trace:
        np.savez(a.trace_out, hidden=np.stack([t["hidden"] for t in trace]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
