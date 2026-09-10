#!/usr/bin/env python3
"""
e2e_migrated.py — the FP32 cases from toolchain/e2e_test.py, re-verified
through the TVM path instead of npu_compiler.py.

Part of TODO Phase 1.5 ("migrate all test cases in toolchain/e2e_test.py to
use toolchain/rtv_tvm/"). Only two of e2e_test.py's 22 case classes are
actually migratable today:

    Fp32LinearCase  -> here, via emit_composite's "matmul" dispatch
    Fp32MlpCase     -> here, via emit_composite's "swiglu_ffn" dispatch

Everything else in e2e_test.py is either INT8/INT16-specific (out of scope
entirely post the FP32-only pivot -- see rtv-accel-fp32-only-target in
project memory) or needs infrastructure that doesn't exist yet:

    Fp32AttentionCase("fp32_attn_gqa", ...) -> here now too, via
                          tvm_backend._emit_attention_block_batched_fp32 --
                          real Q/K/V/O projections, GQA, and a genuine
                          per-row softmax over 16 DIFFERENT real query rows
                          (not one row replicated), which needed
                          codegen.py's Codegen to gain a real `dim_m`
                          (previously the literal constant TILE everywhere)
                          and tvm_backend._emit_softmax_rows_fp32 to drive
                          REDUCE/SCALAR_ALU at dim_m=1 once per row -- see
                          that function's docstring for why no schedule.py
                          primitive could express this. fp32_attn_1head
                          (head_dim=32) is NOT migrated: this dispatch (like
                          the rest of the attention block) still needs
                          head_dim == compiler.tile_dim, a separate,
                          pre-existing limit (K's per-head transpose only
                          handles one square tile) this work doesn't touch.

    Fp32FullLayerCase  -- needs Fp32AttentionCase's machinery (now built)
                          PLUS RoPE applied per-row at each row's OWN
                          position (tvm_backend._emit_rope_batched_fp32,
                          also now built) PLUS chaining across layers.
                          RMSNorm needed NO new work: _emit_rmsnorm_row_fp32
                          already computes ONE scale from a whole-tile
                          REDUCE_SUMSQ regardless of what's in the other 15
                          rows, which happens to be EXACTLY what
                          Fp32FullLayerCase's own reference expects
                          ("compile_rmsnorm_row shares ONE scale across all
                          16 tile rows even when they hold 16 REAL distinct
                          tokens... a real approximation for prefill" --
                          this dispatch reproduces that by construction, not
                          by a deliberate design choice to imitate a bug.
                          See run_fp32_full_layer_case below.

Test vectors, tolerances and instantiation parameters below are copied
verbatim from e2e_test.py's Fp32LinearCase/Fp32MlpCase and the exact
(m, n, k) / (hidden, inter) tuples main() constructs them with, so this is a
genuine parity check, not a fresh test with looser goalposts.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/e2e_migrated.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import tvm                                  # noqa: E402
from tvm.script import relax as R           # noqa: E402

from compiler_state import NPUCompiler, DType, _padded  # noqa: E402
import tvm_backend                           # noqa: E402
from rtv_tvm import patterns                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")

tvm_backend.register()


def _run_sim(instruction_stream, dram, addr, nbytes):
    with tempfile.TemporaryDirectory(prefix="rtv_e2e_migrated_") as wd:
        p, d = os.path.join(wd, "p.bin"), os.path.join(wd, "d.img")
        open(p, "wb").write(bytes(instruction_stream))
        open(d, "wb").write(bytes(dram))
        out = os.path.join(wd, "o.bin")
        mb = max(1, len(dram) // (1024 * 1024) + 2)
        r = subprocess.run(
            [SIM, p, HAL, str(mb), "--dram-image", d,
             "--dump", f"{addr}:{nbytes}:{out}"],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"simulator failed: {r.stdout[-400:]}{r.stderr[-400:]}")
        return open(out, "rb").read()


# ── Fp32LinearCase, migrated ──────────────────────────────────────────────────
#
# TVMScript's @R.function re-parses source text and can't resolve shape
# params captured from an enclosing Python function's closure (see
# test_composite_dispatch.py's own note on this) -- one literal-shaped
# module per (m, n, k) instance, matching e2e_test.py's main() exactly.

@tvm.script.ir_module
class _LinearMod16x16x16:
    @R.function
    def main(x: R.Tensor((16, 16), "float32"),
             w: R.Tensor((16, 16), "float32")) -> R.Tensor((16, 16), "float32"):
        with R.dataflow():
            out = R.matmul(x, w)
            R.output(out)
        return out


@tvm.script.ir_module
class _LinearMod16x32x32:
    @R.function
    def main(x: R.Tensor((16, 32), "float32"),
             w: R.Tensor((32, 32), "float32")) -> R.Tensor((16, 32), "float32"):
        with R.dataflow():
            out = R.matmul(x, w)
            R.output(out)
        return out


@tvm.script.ir_module
class _LinearMod16x48x64:
    @R.function
    def main(x: R.Tensor((16, 64), "float32"),
             w: R.Tensor((64, 48), "float32")) -> R.Tensor((16, 48), "float32"):
        with R.dataflow():
            out = R.matmul(x, w)
            R.output(out)
        return out


_LINEAR_MODS = {
    (16, 16, 16): _LinearMod16x16x16,
    (16, 32, 32): _LinearMod16x32x32,
    (16, 48, 64): _LinearMod16x48x64,
}


def run_fp32_linear_case(name, m, n, k):
    """Fp32LinearCase, verbatim: values spanning 1e-4..1e4 in one tensor --
    the point of the FP32 path -- checked as relative error against the
    magnitude of the TERMS actually summed, not against the result (which
    can cancel near zero by construction)."""
    rng = np.random.default_rng(abs(hash(name)) & 0xFFFF)
    a = (rng.normal(size=(m, k)) * np.logspace(-4, 4, k)[None, :]).astype(np.float32)
    b = (rng.normal(size=(k, n)) * np.logspace(4, -4, n)[None, :]).astype(np.float32)
    expected = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)

    mod = patterns.partition(_LINEAR_MODS[(m, n, k)])
    composites = tvm_backend.offloaded_composites(mod)
    if len(composites) != 1 or composites[0]["kind"] != "matmul":
        return False, f"expected exactly one matmul composite, got {composites}", 0

    a_ad, b_ad, c_ad = 0x1000, 0x10_0000, 0x20_0000
    dram = bytearray(c_ad + _padded(m) * _padded(n) * 4)
    for arr, addr, cols in ((a, a_ad, k), (b, b_ad, n)):
        pad = np.zeros((_padded(arr.shape[0]), _padded(cols)), dtype=np.float32)
        pad[:arr.shape[0], :cols] = arr
        dram[addr:addr + pad.nbytes] = pad.tobytes()

    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    tvm_backend.emit_composite(comp, composites[0], {"x": a_ad, "w": b_ad, "out": c_ad})
    n_instr = len(comp.instruction_stream) // 16

    raw = _run_sim(comp.instruction_stream, dram, c_ad, _padded(m) * _padded(n) * 4)
    got = np.frombuffer(raw, dtype=np.float32).reshape(_padded(m), _padded(n))[:m, :n]

    bound = np.abs(a).astype(np.float64) @ np.abs(b).astype(np.float64)
    rel = np.abs(got - expected) / np.maximum(bound, 1e-30)
    if float(np.max(rel)) < 1e-6:
        return True, "", n_instr
    bad = int(np.argmax(rel))
    return False, (f"max err/|terms| {float(np.max(rel)):.3g} at flat {bad} "
                   f"(got {got.ravel()[bad]:.6g} want {expected.ravel()[bad]:.6g})"), n_instr


# ── Fp32MlpCase, migrated ─────────────────────────────────────────────────────

@tvm.script.ir_module
class _MlpMod16x32:
    @R.function
    def main(x: R.Tensor((16, 16), "float32"),
             wg: R.Tensor((16, 32), "float32"),
             wu: R.Tensor((16, 32), "float32"),
             wd: R.Tensor((32, 16), "float32")) -> R.Tensor((16, 16), "float32"):
        with R.dataflow():
            g = R.matmul(x, wg)
            u = R.matmul(x, wu)
            act = R.nn.silu(g)
            inter = R.multiply(act, u)
            out = R.matmul(inter, wd)
            R.output(out)
        return out


@tvm.script.ir_module
class _MlpMod32x64:
    @R.function
    def main(x: R.Tensor((16, 32), "float32"),
             wg: R.Tensor((32, 64), "float32"),
             wu: R.Tensor((32, 64), "float32"),
             wd: R.Tensor((64, 32), "float32")) -> R.Tensor((16, 32), "float32"):
        with R.dataflow():
            g = R.matmul(x, wg)
            u = R.matmul(x, wu)
            act = R.nn.silu(g)
            inter = R.multiply(act, u)
            out = R.matmul(inter, wd)
            R.output(out)
        return out


_MLP_MODS = {(16, 32): _MlpMod16x32, (32, 64): _MlpMod32x64}
TILE = 16


def run_fp32_mlp_case(name, hidden, inter):
    """Fp32MlpCase, verbatim: two GEMMs, real SiLU, elementwise multiply, a
    third GEMM, all in float, checked as max absolute error relative to the
    output's own magnitude."""
    h, i = hidden, inter
    rng = np.random.default_rng(abs(hash(name)) & 0xFFFF)
    x = rng.normal(size=(TILE, h)).astype(np.float32)
    wg = (rng.normal(size=(h, i)) * 0.1).astype(np.float32)
    wu = (rng.normal(size=(h, i)) * 0.1).astype(np.float32)
    wd = (rng.normal(size=(i, h)) * 0.1).astype(np.float32)

    g = x @ wg
    u = x @ wu
    expected = ((g / (1.0 + np.exp(-g))) * u) @ wd

    mod = patterns.partition(_MLP_MODS[(h, i)])
    composites = tvm_backend.offloaded_composites(mod)
    if len(composites) != 1 or composites[0]["kind"] != "swiglu_ffn":
        return False, f"expected exactly one swiglu_ffn composite, got {composites}", 0

    x_ad, g_ad, u_ad, d_ad, o_ad = 0x1000, 0x10_0000, 0x20_0000, 0x30_0000, 0x40_0000
    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = o_ad + TILE * _padded(h) * 4 + 0x10000
    tvm_backend.emit_composite(comp, composites[0], {
        "x": x_ad, "w_gate": g_ad, "w_up": u_ad, "w_down": d_ad, "out": o_ad,
    })
    n_instr = len(comp.instruction_stream) // 16

    dram_size = max(o_ad + TILE * _padded(h) * 4, comp.dram_bytes_needed) + 0x10000
    dram = bytearray(dram_size)
    for arr, addr in ((x, x_ad), (wg, g_ad), (wu, u_ad), (wd, d_ad)):
        pad = np.zeros((_padded(arr.shape[0]), _padded(arr.shape[1])), dtype=np.float32)
        pad[:arr.shape[0], :arr.shape[1]] = arr
        dram[addr:addr + pad.nbytes] = pad.tobytes()

    raw = _run_sim(comp.instruction_stream, dram, o_ad, TILE * _padded(h) * 4)
    got = np.frombuffer(raw, dtype=np.float32).reshape(TILE, _padded(h))[:, :h]

    bound = np.abs(expected).max() or 1.0
    err = float(np.abs(got - expected).max()) / bound
    if err < 1e-5:
        return True, "", n_instr
    return False, f"max err/|out| {err:.3g}", n_instr


# ── Fp32AttentionCase, migrated (the head_dim==tile_dim instance only) ────────
#
# Unlike Linear/Mlp, the real attention BLOCK (with actual Q/K/V/O
# projections) is deliberately NOT what patterns.py's "attention" composite
# matches -- emit_composite's granularity decision dispatches only the
# already-projected matmul->softmax->matmul core (see its own docstring).
# So, like rtv_tvm/test_transformer_layer.py, this drives
# tvm_backend._emit_attention_block_batched_fp32 directly rather than
# through a Relax graph + patterns.partition -- there is no composite
# pattern for a whole attention block to partition in the first place.

def run_fp32_attention_case(name, hidden, n_heads, n_kv_heads, head_dim):
    """Fp32AttentionCase, verbatim math and tolerance: real Q/K/V/O
    projections, GQA repeat, full non-causal (16,16) attention with a
    genuine per-row softmax (16 DIFFERENT real rows, not one replicated),
    no 1/sqrt(head_dim), checked as max absolute error relative to the
    output's own magnitude -- exactly e2e_test.py's Fp32AttentionCase.check.
    """
    TILE = 16
    q_dim, kv_dim = n_heads * head_dim, n_kv_heads * head_dim
    rng = np.random.default_rng(abs(hash(name)) & 0xFFFF)
    x = rng.normal(size=(TILE, hidden)).astype(np.float32) * 0.5
    wq = (rng.normal(size=(hidden, q_dim)) * 0.2).astype(np.float32)
    wk = (rng.normal(size=(hidden, kv_dim)) * 0.2).astype(np.float32)
    wv = (rng.normal(size=(hidden, kv_dim)) * 0.2).astype(np.float32)
    wo = (rng.normal(size=(q_dim, hidden)) * 0.2).astype(np.float32)

    q = (x @ wq).reshape(TILE, n_heads, head_dim)
    k = (x @ wk).reshape(TILE, n_kv_heads, head_dim)
    v = (x @ wv).reshape(TILE, n_kv_heads, head_dim)
    rep = n_heads // n_kv_heads
    K = np.repeat(k, rep, axis=1)
    V = np.repeat(v, rep, axis=1)
    ctx = np.zeros((TILE, n_heads, head_dim), dtype=np.float32)
    for hh in range(n_heads):
        sc = q[:, hh] @ K[:, hh].T
        e = np.exp(sc - sc.max(axis=-1, keepdims=True))
        p = e / e.sum(axis=-1, keepdims=True)
        ctx[:, hh] = p @ V[:, hh]
    expected = ctx.reshape(TILE, q_dim) @ wo

    addrs = {}
    base = [0x1000]

    def place(nm, arr):
        addrs[nm] = base[0]
        base[0] += arr.nbytes + 0x1000
        return addrs[nm]

    place("x", x)
    place("wq", wq)
    place("wk", wk)
    place("wv", wv)
    place("wo", wo)
    out_addr = base[0]
    base[0] += TILE * hidden * 4 + 0x1000

    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = base[0] + 0x10000

    tvm_backend._emit_attention_block_batched_fp32(
        comp, hidden, n_heads, n_kv_heads, head_dim,
        x_addr=addrs["x"], wq_addr=addrs["wq"], wk_addr=addrs["wk"],
        wv_addr=addrs["wv"], wo_addr=addrs["wo"], out_addr=out_addr,
    )
    n_instr = len(comp.instruction_stream) // 16

    dram_size = max(base[0], comp.dram_bytes_needed) + 0x10000
    dram = bytearray(dram_size)
    for nm, arr in (("x", x), ("wq", wq), ("wk", wk), ("wv", wv), ("wo", wo)):
        a = addrs[nm]
        dram[a:a + arr.nbytes] = arr.tobytes()
    for addr, data in comp._const_regions.items():
        dram[addr:addr + len(data)] = data

    raw = _run_sim(comp.instruction_stream, dram, out_addr, TILE * hidden * 4)
    got = np.frombuffer(raw, dtype=np.float32).reshape(TILE, hidden)

    bound = float(np.abs(expected).max()) or 1.0
    err = float(np.abs(got - expected).max()) / bound
    if err < 1e-4:
        return True, "", n_instr
    return False, f"max err/|out| {err:.3g}", n_instr


# ── Fp32FullLayerCase, migrated (head_dim==tile_dim instances only) ───────────
#
# e2e_test.py's own two instances (fp32_layer_1L/2L) use head_dim=32, which
# exceeds compiler.tile_dim(16) -- the same pre-existing limit that keeps
# fp32_attn_1head unmigrated above (K's per-head transpose, and RoPE's
# intra-tile permutation, both only handle one square tile; head_dim=32
# would need the transpose generalized into per-column-tile square
# transposes stacked into the wide output, which this pass didn't build).
# So, like Fp32AttentionCase, this is verified at head_dim=16 -- exercising
# every piece of machinery (RMSNorm, batched attention with GQA, per-row
# RoPE, residual, SwiGLU, multi-layer chaining) with the EXACT reference
# math and tolerance, just not e2e_test.py's literal head_dim=32 parameters.

def run_fp32_full_layer_case(name, hidden, inter, n_heads, n_kv_heads, head_dim,
                             layers=1):
    """Fp32FullLayerCase, verbatim math and tolerance: RMSNorm (deliberately
    ONE shared scale across all 16 rows, per that reference's own rms()
    helper and comment -- see run_fp32_full_layer_case's module note),
    real Q/K/V/O projections, GQA, per-row RoPE (rope_tables' sign-folded
    sin table, rotate_half via swap-then-sign), full non-causal attention
    with a genuine per-row softmax, residual, SwiGLU, residual -- chained
    across `layers` layers, checked as max absolute error relative to the
    output's own magnitude, exactly e2e_test.py's Fp32FullLayerCase.check.
    """
    TILE = 16
    q_dim, kv_dim = n_heads * head_dim, n_kv_heads * head_dim
    rng = np.random.default_rng(abs(hash(name)) & 0xFFFF)

    def w(shape, s=0.2):
        return (rng.normal(size=shape) * s).astype(np.float32)

    layers_np = []
    for _ in range(layers):
        layers_np.append(dict(
            gamma1=np.ones(hidden, dtype=np.float32),
            gamma2=np.ones(hidden, dtype=np.float32),
            wq=w((hidden, q_dim)), wk=w((hidden, kv_dim)), wv=w((hidden, kv_dim)),
            wo=w((q_dim, hidden)), wgate=w((hidden, inter)), wup=w((hidden, inter)),
            wdown=w((inter, hidden)),
        ))

    x = rng.normal(size=(TILE, hidden)).astype(np.float32)

    def rms(z):
        ssq = float((z.astype(np.float64) ** 2).sum())
        return z * np.sqrt(z.shape[-1] / max(ssq, 1e-30))

    cos_t, sin_t = NPUCompiler.rope_tables(TILE, head_dim, 1e6, dtype=3)
    half = head_dim // 2

    def rope(z):
        sw = np.concatenate([z[..., half:], z[..., :half]], axis=-1)
        return z * cos_t[:, None, :] + sw * sin_t[:, None, :]

    cur = x.copy()
    for lw in layers_np:
        y = rms(cur)
        q = (y @ lw["wq"]).reshape(TILE, n_heads, head_dim)
        k = (y @ lw["wk"]).reshape(TILE, n_kv_heads, head_dim)
        v = (y @ lw["wv"]).reshape(TILE, n_kv_heads, head_dim)
        q, k = rope(q), rope(k)
        rep = n_heads // n_kv_heads
        K, V = np.repeat(k, rep, 1), np.repeat(v, rep, 1)
        ctx = np.zeros((TILE, n_heads, head_dim), dtype=np.float64)
        for hh in range(n_heads):
            sc = q[:, hh] @ K[:, hh].T
            e = np.exp(sc - sc.max(-1, keepdims=True))
            ctx[:, hh] = (e / e.sum(-1, keepdims=True)) @ V[:, hh]
        cur = cur + (ctx.reshape(TILE, n_heads * head_dim) @ lw["wo"]).astype(np.float32)
        y2 = rms(cur)
        g = y2 @ lw["wgate"]
        u = y2 @ lw["wup"]
        cur = cur + ((g / (1.0 + np.exp(-g))) * u) @ lw["wdown"]
    expected = cur.astype(np.float32)

    addrs = {}
    base = [0x1000]

    def place(nm, arr):
        addrs[nm] = base[0]
        base[0] += arr.nbytes + 0x1000
        return addrs[nm]

    place("x", x)
    layer_weights = []
    for li, lw in enumerate(layers_np):
        entry = {
            "gamma1": place(f"l{li}_gamma1", np.tile(lw["gamma1"], (TILE, 1))),
            "gamma2": place(f"l{li}_gamma2", np.tile(lw["gamma2"], (TILE, 1))),
        }
        for k in ("wq", "wk", "wv", "wo", "wgate", "wup", "wdown"):
            entry[k] = place(f"l{li}_{k}", lw[k])
        layer_weights.append(entry)
    scratch_a = base[0]; base[0] += TILE * hidden * 4 + 0x1000
    scratch_b = base[0]; base[0] += TILE * hidden * 4 + 0x1000

    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = base[0] + 0x10000

    final_addr = tvm_backend._emit_model_batched_fp32(
        comp, layers, hidden, inter, n_heads, n_kv_heads, head_dim,
        x_addr=addrs["x"], layer_weights=layer_weights,
        scratch_a=scratch_a, scratch_b=scratch_b, positions=list(range(TILE)),
    )
    n_instr = len(comp.instruction_stream) // 16

    dram_size = max(base[0], comp.dram_bytes_needed) + 0x10000
    dram = bytearray(dram_size)
    dram[addrs["x"]:addrs["x"] + x.nbytes] = x.tobytes()
    for li, lw in enumerate(layers_np):
        for nm, arr in ((f"l{li}_gamma1", np.tile(lw["gamma1"], (TILE, 1))),
                       (f"l{li}_gamma2", np.tile(lw["gamma2"], (TILE, 1))),
                       (f"l{li}_wq", lw["wq"]), (f"l{li}_wk", lw["wk"]),
                       (f"l{li}_wv", lw["wv"]), (f"l{li}_wo", lw["wo"]),
                       (f"l{li}_wgate", lw["wgate"]), (f"l{li}_wup", lw["wup"]),
                       (f"l{li}_wdown", lw["wdown"])):
            a = layer_weights[li][nm.split("_", 1)[1]]
            dram[a:a + arr.nbytes] = arr.astype(np.float32).tobytes()
    for addr, data in comp._const_regions.items():
        dram[addr:addr + len(data)] = data

    raw = _run_sim(comp.instruction_stream, dram, final_addr, TILE * hidden * 4)
    got = np.frombuffer(raw, dtype=np.float32).reshape(TILE, hidden)

    bound = float(np.abs(expected).max()) or 1.0
    err = float(np.abs(got - expected).max()) / bound
    if err < 1e-3:
        return True, "", n_instr
    return False, f"max err/|out| {err:.3g}", n_instr


def main():
    cases = [
        ("fp32_linear_16x16x16", lambda: run_fp32_linear_case("fp32_linear_16x16x16", 16, 16, 16)),
        ("fp32_linear_16x32x32", lambda: run_fp32_linear_case("fp32_linear_16x32x32", 16, 32, 32)),
        ("fp32_linear_16x48x64", lambda: run_fp32_linear_case("fp32_linear_16x48x64", 16, 48, 64)),
        ("fp32_mlp_16x32", lambda: run_fp32_mlp_case("fp32_mlp_16x32", 16, 32)),
        ("fp32_mlp_32x64", lambda: run_fp32_mlp_case("fp32_mlp_32x64", 32, 64)),
        ("fp32_attn_gqa", lambda: run_fp32_attention_case("fp32_attn_gqa", 32, 4, 2, 16)),
        ("fp32_layer_1L", lambda: run_fp32_full_layer_case("fp32_layer_1L", 64, 64, 2, 1, 16, layers=1)),
        ("fp32_layer_2L", lambda: run_fp32_full_layer_case("fp32_layer_2L", 64, 64, 2, 1, 16, layers=2)),
    ]
    n_pass = 0
    for name, fn in cases:
        try:
            ok, msg, n_instr = fn()
        except Exception as e:  # noqa: BLE001
            ok, msg, n_instr = False, f"{type(e).__name__}: {e}", 0
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({n_instr} instrs)"
              + (f"  -- {msg}" if msg else ""))
        n_pass += ok
    print(f"\n{n_pass}/{len(cases)} passed")
    return 0 if n_pass == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
