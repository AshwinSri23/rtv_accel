#!/usr/bin/env python3
"""
test_attention_multitile.py — attention's core with seq_k/head_dim > one
tile, single head, driven directly through tvm_backend._emit_attention_
core_fp32 (not pattern-matched -- same reasoning as test_gqa_attention.py).

Covers TODO 1.1/1.2's ">1 K/V tile" / "non-square matrix dimensions" items.
Turned out to need no codegen.py changes at all: REDUCE's accumulate flag
already lets rowmax/rowsum span multiple 16-wide column tiles (the same
mechanism test_tvm_path.py's tvm_sumsq_16x64/tvm_rowmax_16x64 already prove
at N=64), and GEMM already tiles M/N/K generically (test_gemm_double_
buffer.py's own multi-K-tile cases). The only thing scoping
_emit_attention_core_fp32 to exactly one 16x16 tile was that its OWN
internal buffers (scores/probs/etc.) were hardcoded to (M, M) instead of
(M, seq_k) -- fixed by generalizing those, not by touching codegen.py.

This is a SEPARATE limitation from genuinely batched (non-replicated)
multi-row queries, which still isn't done -- see _emit_attention_core_
fp32's docstring. Q here is still one real query row, replicated across
all 16 tile rows; seq_k/head_dim being wider than one tile is what's new.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_attention_multitile.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from npu_compiler import NPUCompiler, DType  # noqa: E402
import tvm_backend                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")


def run_multitile_case(seq_k, head_dim, seed=5):
    rng = np.random.default_rng(seed)
    q_row = rng.uniform(-1, 1, size=(head_dim,)).astype(np.float32)
    q = np.tile(q_row, (16, 1)).astype(np.float32)
    k = rng.uniform(-1, 1, size=(head_dim, seq_k)).astype(np.float32)  # already "transposed"
    v = rng.uniform(-1, 1, size=(seq_k, head_dim)).astype(np.float32)

    sc = q_row.astype(np.float64) @ k.astype(np.float64)
    e = np.exp(sc - sc.max())
    p = e / e.sum()
    expected_row = (p @ v.astype(np.float64)).astype(np.float32)
    expected = np.tile(expected_row, (16, 1)).astype(np.float32)

    q_ad, k_ad, v_ad, out_ad = 0x1000, 0x40000, 0x80000, 0x100000
    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = out_ad + 16 * head_dim * 4 + 0x10000

    tvm_backend._emit_attention_core_fp32(
        comp, seq_k, head_dim,
        q_addr=q_ad, k_addr=k_ad, v_addr=v_ad, out_addr=out_ad,
    )
    n_instr = len(comp.instruction_stream) // 16

    dram_size = max(out_ad + 16 * head_dim * 4, comp.dram_bytes_needed) + 0x10000
    dram = bytearray(dram_size)
    dram[q_ad:q_ad + q.nbytes] = q.tobytes()
    dram[k_ad:k_ad + k.nbytes] = k.tobytes()
    dram[v_ad:v_ad + v.nbytes] = v.tobytes()
    for addr, data in comp._const_regions.items():
        dram[addr:addr + len(data)] = data

    with tempfile.TemporaryDirectory(prefix="rtv_attn_mt_") as wd:
        p, d = os.path.join(wd, "p.bin"), os.path.join(wd, "d.img")
        open(p, "wb").write(bytes(comp.instruction_stream))
        open(d, "wb").write(bytes(dram))
        out = os.path.join(wd, "o.bin")
        mb = max(1, len(dram) // (1024 * 1024) + 2)
        r = subprocess.run(
            [SIM, p, HAL, str(mb), "--dram-image", d,
             "--dump", f"{out_ad}:{16 * head_dim * 4}:{out}"],
            capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"simulator failed: {r.stdout[-400:]}{r.stderr[-400:]}", 0
        got = np.frombuffer(open(out, "rb").read(), dtype=np.float32).reshape(16, head_dim)

    if np.allclose(got, expected, atol=1e-4):
        return True, "", n_instr
    d = np.abs(got - expected)
    return False, f"{int((d > 1e-4).sum())}/{d.size} wrong, max delta {float(d.max()):.4g}", n_instr


def main():
    cases = [
        # Baseline: exactly one tile each way, should be unaffected.
        ("attn_1tile_16x16", lambda: run_multitile_case(16, 16)),
        # 2 column-tiles for softmax + P@V's K-axis, 2 K-tiles for Q@K.
        ("attn_2tile_32x32", lambda: run_multitile_case(32, 32, seed=6)),
        # Non-square: seq_k and head_dim independently multi-tile.
        ("attn_seqk48_hd16", lambda: run_multitile_case(48, 16, seed=7)),
        ("attn_seqk16_hd64", lambda: run_multitile_case(16, 64, seed=8)),
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
