#!/usr/bin/env python3
"""
test_attention_wide_headdim.py — the attention BLOCK (real Q/K/V/O
projections, GQA, RoPE) generalized past head_dim == compiler.tile_dim,
closing the last real gap between the migrated Fp32AttentionCase/
Fp32FullLayerCase and e2e_test.py's ORIGINAL parametrization (both used
head_dim=32; head_dim=16 was a substitute this session used earlier
because the machinery genuinely didn't support anything wider yet).

Three real bugs, once tracked down (all size/shape mistakes copy-pasted
into a "generalize later" TODO, not conceptual gaps):

1. `_emit_rope_fp32`/`_emit_rope_batched_fp32`'s rotate_half permutation
   matrix was hardcoded to `(compiler.tile_dim, compiler.tile_dim)`
   instead of `(head_dim, head_dim)` -- for head_dim==tile_dim these are
   the same shape, which is exactly why this went unnoticed. The GEMM
   that applies the matrix already tiled M/N/K generically (proven
   elsewhere in this codebase), so sizing P correctly is the ENTIRE fix;
   nothing about the "rotate_half's partner is a whole other tile once
   head_dim >= 2*tile_dim" case (the thing `npu_compiler.py`'s
   `compile_rope` handled as a genuinely different code path) needed
   separate handling here -- one permutation matrix expresses both an
   intra-tile swap (head_dim==16) and a whole-tile swap (head_dim>=32)
   identically.
2. `_emit_copy_slice_fp32` (V's per-head de-interleave) hardcoded its
   output width to `compiler.tile_dim` instead of accepting the real
   head_dim -- the GEMM underneath was already general, only the call
   site's own literal needed to become a parameter.
3. `_emit_transpose_fp32` (K's per-head transpose) is a real hardware
   limit -- TRANSPOSE is in-place and square-single-tile-only -- so THIS
   one needed an actual new helper, `_emit_transpose_wide_fp32`, looping
   the single-tile transpose once per column-tile of a wider head_dim,
   each writing into its own row-block of a fresh, tightly-packed output
   (verified directly: for input A split into column-tiles A_0..A_{n-1},
   A^T's row-block t is exactly A_t^T).

Verified against numpy at head_dim 16 (regression), 32, 64, and 128
(Qwen3-0.6B's actual value), with and without RoPE, for both the
attention BLOCK alone and a full transformer layer -- including one case
at Qwen3's REAL full configuration (hidden=1024, 16 query heads over 8 KV
heads, head_dim=128).

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_attention_wide_headdim.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from compiler_state import NPUCompiler, DType  # noqa: E402
import tvm_backend                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")

M = 16


def _run_sim(instr_stream, dram, addr, nbytes, shape):
    with tempfile.TemporaryDirectory(prefix="rtv_attn_wide_") as wd:
        p, d = os.path.join(wd, "p.bin"), os.path.join(wd, "d.img")
        open(p, "wb").write(bytes(instr_stream))
        open(d, "wb").write(bytes(dram))
        out = os.path.join(wd, "o.bin")
        mb = max(1, len(dram) // (1024 * 1024) + 2)
        r = subprocess.run([SIM, p, HAL, str(mb), "--dram-image", d,
                            "--dump", f"{addr}:{nbytes}:{out}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"simulator failed: {r.stdout[-500:]}{r.stderr[-500:]}")
        return np.frombuffer(open(out, "rb").read(), dtype=np.float32).reshape(shape)


def _rope_ref(z, pos, head_dim, theta=1e6):
    cos_t, sin_t = NPUCompiler.rope_tables(pos + 1, head_dim, theta, dtype=3)
    half = head_dim // 2
    sw = np.concatenate([z[..., half:], z[..., :half]], axis=-1)
    return z * cos_t[pos].astype(np.float64) + sw * sin_t[pos].astype(np.float64)


def run_attention_block_case(hidden, n_heads, n_kv_heads, head_dim, pos=None, seed=3):
    """Real Q/K/V/O projections + GQA, self-attention over one distinct
    (therefore replicated) key -- softmax is uniform, so context = that
    head's own V, matching _emit_attention_block_fp32's own documented
    degenerate-but-exact case."""
    q_dim, kv_dim = n_heads * head_dim, n_kv_heads * head_dim
    rng = np.random.default_rng(seed)
    x_row = rng.normal(size=(hidden,)).astype(np.float32) * 0.5
    wq = (rng.normal(size=(hidden, q_dim)) * 0.2).astype(np.float32)
    wk = (rng.normal(size=(hidden, kv_dim)) * 0.2).astype(np.float32)
    wv = (rng.normal(size=(hidden, kv_dim)) * 0.2).astype(np.float32)
    wo = (rng.normal(size=(q_dim, hidden)) * 0.2).astype(np.float32)

    q = (x_row @ wq).reshape(n_heads, head_dim).astype(np.float64)
    k = (x_row @ wk).reshape(n_kv_heads, head_dim).astype(np.float64)
    v = (x_row @ wv).reshape(n_kv_heads, head_dim).astype(np.float64)
    if pos is not None:
        q, k = _rope_ref(q, pos, head_dim), _rope_ref(k, pos, head_dim)
    rep = n_heads // n_kv_heads
    ctx = np.stack([v[qh // rep] for qh in range(n_heads)])
    expected_row = (ctx.reshape(q_dim) @ wo).astype(np.float32)

    x = np.zeros((M, hidden), dtype=np.float32)
    x[0, :] = x_row
    addrs, base = {}, [0x1000]

    def place(name, arr):
        addrs[name] = base[0]
        base[0] += arr.nbytes + 0x1000
        return addrs[name]

    place("x", x)
    place("wq", wq); place("wk", wk); place("wv", wv); place("wo", wo)
    out_addr = base[0]
    base[0] += M * hidden * 4 + 0x1000

    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = base[0] + 0x10000

    tvm_backend._emit_attention_block_fp32(
        comp, hidden, n_heads, n_kv_heads, head_dim,
        x_addr=addrs["x"], wq_addr=addrs["wq"], wk_addr=addrs["wk"],
        wv_addr=addrs["wv"], wo_addr=addrs["wo"], out_addr=out_addr, pos=pos,
    )
    n_instr = len(comp.instruction_stream) // 16

    dram_size = max(base[0], comp.dram_bytes_needed) + 0x10000
    dram = bytearray(dram_size)
    dram[addrs["x"]:addrs["x"] + x.nbytes] = x.tobytes()
    for nm, arr in (("wq", wq), ("wk", wk), ("wv", wv), ("wo", wo)):
        a = addrs[nm]
        dram[a:a + arr.nbytes] = arr.tobytes()
    for addr, data in comp._const_regions.items():
        dram[addr:addr + len(data)] = data

    got = _run_sim(comp.instruction_stream, dram, out_addr, M * hidden * 4, (M, hidden))[0]
    if np.allclose(got, expected_row, atol=2e-2):
        return True, "", n_instr
    d = np.abs(got - expected_row)
    return False, f"max delta {float(d.max()):.4g}", n_instr


def _rmsnorm_ref(v, gamma):
    return v / np.sqrt((v.astype(np.float64) ** 2).mean() + 1e-6) * gamma.astype(np.float64)


def run_full_layer_case(hidden, inter, n_heads, n_kv_heads, head_dim, pos=None, seed=11):
    q_dim, kv_dim = n_heads * head_dim, n_kv_heads * head_dim
    rng = np.random.default_rng(seed)
    x_row = rng.normal(size=(hidden,)).astype(np.float32) * 0.3
    lw = dict(
        gamma1=rng.uniform(0.5, 1.5, size=(hidden,)).astype(np.float32),
        gamma2=rng.uniform(0.5, 1.5, size=(hidden,)).astype(np.float32),
        wq=(rng.normal(size=(hidden, q_dim)) * 0.2).astype(np.float32),
        wk=(rng.normal(size=(hidden, kv_dim)) * 0.2).astype(np.float32),
        wv=(rng.normal(size=(hidden, kv_dim)) * 0.2).astype(np.float32),
        wo=(rng.normal(size=(q_dim, hidden)) * 0.2).astype(np.float32),
        wgate=(rng.normal(size=(hidden, inter)) * 0.1).astype(np.float32),
        wup=(rng.normal(size=(hidden, inter)) * 0.1).astype(np.float32),
        wdown=(rng.normal(size=(inter, hidden)) * 0.1).astype(np.float32),
    )

    norm1 = _rmsnorm_ref(x_row, lw["gamma1"])
    q = (norm1 @ lw["wq"]).reshape(n_heads, head_dim)
    k = (norm1 @ lw["wk"]).reshape(n_kv_heads, head_dim)
    v = (norm1 @ lw["wv"]).reshape(n_kv_heads, head_dim)
    if pos is not None:
        q, k = _rope_ref(q, pos, head_dim), _rope_ref(k, pos, head_dim)
    rep = n_heads // n_kv_heads
    ctx = np.stack([v[qh // rep] for qh in range(n_heads)])
    attn = ctx.reshape(q_dim) @ lw["wo"]
    resid1 = x_row.astype(np.float64) + attn
    norm2 = _rmsnorm_ref(resid1, lw["gamma2"])
    g = norm2 @ lw["wgate"]
    u = norm2 @ lw["wup"]
    mlp = ((g / (1.0 + np.exp(-g))) * u) @ lw["wdown"]
    expected_row = (resid1 + mlp).astype(np.float32)

    x = np.zeros((M, hidden), dtype=np.float32)
    x[0, :] = x_row
    addrs, base = {}, [0x1000]

    def place(name, arr):
        addrs[name] = base[0]
        base[0] += arr.nbytes + 0x1000
        return addrs[name]

    place("x", x)
    place("gamma1", np.tile(lw["gamma1"], (M, 1)))
    place("gamma2", np.tile(lw["gamma2"], (M, 1)))
    for k_ in ("wq", "wk", "wv", "wo", "wgate", "wup", "wdown"):
        place(k_, lw[k_])
    out_addr = base[0]
    base[0] += M * hidden * 4 + 0x1000

    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = base[0] + 0x10000

    tvm_backend._emit_transformer_layer_fp32(
        comp, hidden, inter, n_heads, n_kv_heads, head_dim,
        x_addr=addrs["x"], gamma1_addr=addrs["gamma1"], gamma2_addr=addrs["gamma2"],
        wq_addr=addrs["wq"], wk_addr=addrs["wk"], wv_addr=addrs["wv"], wo_addr=addrs["wo"],
        wgate_addr=addrs["wgate"], wup_addr=addrs["wup"], wdown_addr=addrs["wdown"],
        out_addr=out_addr, pos=pos,
    )
    n_instr = len(comp.instruction_stream) // 16

    dram_size = max(base[0], comp.dram_bytes_needed) + 0x10000
    dram = bytearray(dram_size)
    dram[addrs["x"]:addrs["x"] + x.nbytes] = x.tobytes()
    for nm, arr in (("gamma1", np.tile(lw["gamma1"], (M, 1))),
                    ("gamma2", np.tile(lw["gamma2"], (M, 1))),
                    ("wq", lw["wq"]), ("wk", lw["wk"]), ("wv", lw["wv"]), ("wo", lw["wo"]),
                    ("wgate", lw["wgate"]), ("wup", lw["wup"]), ("wdown", lw["wdown"])):
        a = addrs[nm]
        dram[a:a + arr.astype(np.float32).nbytes] = arr.astype(np.float32).tobytes()
    for addr, data in comp._const_regions.items():
        dram[addr:addr + len(data)] = data

    got = _run_sim(comp.instruction_stream, dram, out_addr, M * hidden * 4, (M, hidden))[0]
    if np.allclose(got, expected_row, atol=2e-2):
        return True, "", n_instr
    d = np.abs(got - expected_row)
    return False, f"max delta {float(d.max()):.4g}", n_instr


def main():
    cases = [
        ("attn_block_hd16_regression", lambda: run_attention_block_case(64, 2, 1, 16)),
        ("attn_block_hd32", lambda: run_attention_block_case(64, 2, 1, 32)),
        ("attn_block_hd64", lambda: run_attention_block_case(64, 2, 1, 64)),
        ("attn_block_hd128_rope", lambda: run_attention_block_case(64, 2, 1, 128, pos=5)),
        ("attn_block_qwen3_real_dims_rope",
         lambda: run_attention_block_case(1024, 16, 8, 128, pos=3)),
        ("full_layer_hd32_original_e2e_params",
         lambda: run_full_layer_case(64, 64, 2, 1, 32)),
        ("full_layer_hd32_rope", lambda: run_full_layer_case(64, 64, 2, 1, 32, pos=2)),
        ("full_layer_hd64_rope", lambda: run_full_layer_case(128, 64, 2, 1, 64, pos=3)),
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
