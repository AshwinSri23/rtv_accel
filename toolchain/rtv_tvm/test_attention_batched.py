#!/usr/bin/env python3
"""
test_attention_batched.py — tvm_backend._emit_attention_core_batched_fp32 /
_emit_gqa_attention_batched_fp32: genuinely batched attention, where all 16
tile rows hold DIFFERENT real queries (and correspondingly different
keys/values), not one real row replicated across the tile.

This is the actual blocker TODO 1.1 named ("codegen.py's _reduce/
_tile_scalar hardcode dim_m=dim_n=TILE, so a real per-row softmax across
multiple DIFFERENT queries in one call isn't reachable without generalizing
that first") and it needed two real things, not one:

1. codegen.py's Codegen gained a `dim_m` instance attribute (default TILE,
   so every existing caller is unaffected) threaded through LOAD_TILE/
   STORE_TILE/EXEC_ALU/REDUCE/SCALAR_ALU's dim_m field, which used to be
   the literal constant TILE everywhere.
2. tvm_backend._emit_softmax_rows_fp32 drives that at dim_m=1, once per
   row, bypassing schedule.py's schedule_reduce/schedule_tile_scalar
   entirely -- they unconditionally split every axis into TILE-sized
   chunks (`sch.split(i, factors=[None, TILE])`), which cannot express a
   1-row axis, so there's no schedule-primitive route to this at all.

GEMM (Q@K^T, P@V) needed NO changes: COMPUTE_GEMM is inherently
row-parallel, so it was already correct for genuinely different rows.

Verified against an independent numpy reference matching e2e_test.py's
Fp32AttentionCase's own math exactly (full non-causal (TILE,TILE)
attention, no 1/sqrt(head_dim), softmax per query row over all TILE keys).

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_attention_batched.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from compiler_state import NPUCompiler, DType  # noqa: E402
import tvm_backend                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")

M = 16


def _run_sim(instr_stream, dram, addr, nbytes, shape):
    with tempfile.TemporaryDirectory(prefix="rtv_attn_batched_") as wd:
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


def _gqa_ref(q, k, v, n_heads, n_kv_heads, head_dim):
    """q: (M, n_heads, head_dim); k, v: (M, n_kv_heads, head_dim).
    Matches Fp32AttentionCase's per-head loop exactly: full (M, M)
    attention, no scale, per-row softmax."""
    rep = n_heads // n_kv_heads
    K = np.repeat(k, rep, axis=1)
    V = np.repeat(v, rep, axis=1)
    ctx = np.zeros((M, n_heads, head_dim), dtype=np.float64)
    for hh in range(n_heads):
        sc = q[:, hh].astype(np.float64) @ K[:, hh].astype(np.float64).T
        e = np.exp(sc - sc.max(axis=-1, keepdims=True))
        p = e / e.sum(axis=-1, keepdims=True)
        ctx[:, hh] = p @ V[:, hh].astype(np.float64)
    return ctx


def run_single_head_case(head_dim=16, seed=3):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(M, head_dim)).astype(np.float32)
    k = rng.normal(size=(M, head_dim)).astype(np.float32)   # (seq_k, head_dim), conceptually
    v = rng.normal(size=(M, head_dim)).astype(np.float32)
    expected = _gqa_ref(q[:, None, :], k[:, None, :], v[:, None, :], 1, 1, head_dim)[:, 0, :]

    # _emit_attention_core_batched_fp32 (like its non-batched counterpart)
    # computes matmul(q, k) LITERALLY -- no transpose inside the core -- so
    # the caller must hand it k ALREADY as (head_dim, seq_k), matching
    # test_attention_multitile.py's own convention.
    k_t = np.ascontiguousarray(k.T)

    addrs = {}
    base = [0x1000]

    def place(name, arr):
        addrs[name] = base[0]
        base[0] += arr.nbytes + 0x1000
        return addrs[name]

    place("q", q)
    place("k", k_t)
    place("v", v)
    out_addr = base[0]
    base[0] += M * head_dim * 4 + 0x1000

    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = base[0] + 0x10000

    tvm_backend._emit_attention_core_batched_fp32(
        comp, seq_k=M, head_dim=head_dim,
        q_addr=addrs["q"], k_addr=addrs["k"], v_addr=addrs["v"], out_addr=out_addr,
    )
    n_instr = len(comp.instruction_stream) // 16

    dram_size = max(base[0], comp.dram_bytes_needed) + 0x10000
    dram = bytearray(dram_size)
    dram[addrs["q"]:addrs["q"] + q.nbytes] = q.tobytes()
    dram[addrs["k"]:addrs["k"] + k_t.nbytes] = k_t.tobytes()
    dram[addrs["v"]:addrs["v"] + v.nbytes] = v.tobytes()
    for addr, data in comp._const_regions.items():
        dram[addr:addr + len(data)] = data

    got = _run_sim(comp.instruction_stream, dram, out_addr, M * head_dim * 4, (M, head_dim))
    if np.allclose(got, expected, atol=1e-4):
        return True, "", n_instr
    d = np.abs(got - expected)
    return False, f"max delta {float(d.max()):.4g}", n_instr


def run_gqa_case(n_heads, n_kv_heads, head_dim=16, seed=5):
    rng = np.random.default_rng(seed)
    q_dim, kv_dim = n_heads * head_dim, n_kv_heads * head_dim
    q = rng.normal(size=(M, n_heads, head_dim)).astype(np.float32)
    k = rng.normal(size=(M, n_kv_heads, head_dim)).astype(np.float32)
    v = rng.normal(size=(M, n_kv_heads, head_dim)).astype(np.float32)
    expected = _gqa_ref(q, k, v, n_heads, n_kv_heads, head_dim).reshape(M, q_dim)

    # Q: wide (M, q_dim) buffer, head h = columns [h*hd:(h+1)*hd] (real layout).
    q_wide = q.reshape(M, q_dim)
    # K/V: separate CONTIGUOUS per-kv-head blocks (what _emit_gqa_attention_
    # batched_fp32 requires -- see its docstring / _emit_gqa_attention_fp32's).
    # K additionally needs to be pre-TRANSPOSED per head to (head_dim, seq_k)
    # -- the attention core computes matmul(q, k) literally, no transpose
    # inside it (see _emit_attention_core_fp32's docstring).
    k_blocks = np.ascontiguousarray(k.transpose(1, 2, 0))   # (n_kv_heads, hd, M)
    v_blocks = np.ascontiguousarray(v.transpose(1, 0, 2))   # (n_kv_heads, M, hd)

    addrs = {}
    base = [0x1000]

    def place(name, arr):
        addrs[name] = base[0]
        base[0] += arr.nbytes + 0x1000
        return addrs[name]

    place("q", q_wide)
    place("k", k_blocks)
    place("v", v_blocks)
    out_addr = base[0]
    base[0] += M * q_dim * 4 + 0x1000

    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = base[0] + 0x10000

    tvm_backend._emit_gqa_attention_batched_fp32(
        comp, n_heads=n_heads, n_kv_heads=n_kv_heads, seq_k=M, head_dim=head_dim,
        q_addr=addrs["q"], k_addr=addrs["k"], v_addr=addrs["v"], out_addr=out_addr,
    )
    n_instr = len(comp.instruction_stream) // 16

    dram_size = max(base[0], comp.dram_bytes_needed) + 0x10000
    dram = bytearray(dram_size)
    dram[addrs["q"]:addrs["q"] + q_wide.nbytes] = q_wide.tobytes()
    dram[addrs["k"]:addrs["k"] + k_blocks.nbytes] = k_blocks.tobytes()
    dram[addrs["v"]:addrs["v"] + v_blocks.nbytes] = v_blocks.tobytes()
    for addr, data in comp._const_regions.items():
        dram[addr:addr + len(data)] = data

    got = _run_sim(comp.instruction_stream, dram, out_addr, M * q_dim * 4, (M, q_dim))
    if np.allclose(got, expected, atol=1e-4):
        return True, "", n_instr
    d = np.abs(got - expected)
    return False, f"max delta {float(d.max()):.4g}", n_instr


def main():
    cases = [
        ("batched_single_head", lambda: run_single_head_case()),
        ("batched_gqa_2h_1kv", lambda: run_gqa_case(2, 1, seed=11)),
        ("batched_gqa_4h_2kv", lambda: run_gqa_case(4, 2, seed=13)),
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
