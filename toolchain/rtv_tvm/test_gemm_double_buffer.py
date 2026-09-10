#!/usr/bin/env python3
"""
test_gemm_double_buffer.py — GEMM K-tile double buffering: correctness AND
a genuine cycle-count improvement, not just "doesn't crash."

Covers TODO 1.2's "SRAM double buffering" item. codegen.py's Codegen._gemm_
k_loop replaces the generic per-K-step recursion (LOAD/WAIT/COMPUTE, in
strict TIR order) with one that alternates the A and B operands between two
SRAM slots each and prefetches step k+1's tiles before step k's compute --
confirmed via rtv_sim's own cycle model that the naive version never
overlaps a tile's DMA with anything (the next LOAD_TILE isn't even issued
until the current COMPUTE_GEMM has already been emitted and executed).

Both operands are double-buffered, not just the weight (B) -- unlike
compile_linear's version, which leaves A single-slot. Measured directly why
that matters here: double-buffering B alone produced IDENTICAL cycle
estimates to no double buffering at all, because WAIT_SYNC stalls to the
LATEST unmet dependency, and A's un-overlapped reload's full latency
dominated the critical path every step regardless of how well B's prefetch
overlapped. Buffering both operands is what actually moves the number.

This only affects instruction ORDER and SRAM slot usage, never the
computed values -- so every one of the 43 existing tests already re-proves
correctness incidentally (several use K > 16). What's checked here
specifically: a real multi-K-tile case still gives the exact right answer,
and rtv_sim's own "serial" vs "db" cycle estimate shows the double-buffered
path is genuinely faster, not just differently-ordered by the same amount.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_gemm_double_buffer.py
"""

import os
import re
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from compiler_state import NPUCompiler, DType, _padded  # noqa: E402
import tvm_backend                                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")


def _run_sim(instr_stream, dram):
    with tempfile.TemporaryDirectory(prefix="rtv_dbuf_") as wd:
        p, d = os.path.join(wd, "p.bin"), os.path.join(wd, "d.img")
        open(p, "wb").write(bytes(instr_stream))
        open(d, "wb").write(bytes(dram))
        mb = max(1, len(dram) // (1024 * 1024) + 2)
        r = subprocess.run([SIM, p, HAL, str(mb), "--dram-image", d,
                            "--dump", f"0x80000:1024:{os.path.join(wd, 'o.bin')}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"simulator failed: {r.stdout[-400:]}{r.stderr[-400:]}")
        got = np.frombuffer(open(os.path.join(wd, "o.bin"), "rb").read(),
                            dtype=np.float32)
        serial = int(re.search(r"serial\s+(\d+)", r.stdout).group(1))
        db = int(re.search(r"db\s+(\d+)", r.stdout).group(1))
        return got, serial, db


def run_multitile_matmul_case(M, N, K, min_speedup, seed=0):
    """Correctness + a real cycle-count win for a multi-K-tile matmul."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, size=(M, K)).astype(np.float32)
    b = rng.uniform(-1, 1, size=(K, N)).astype(np.float32)
    expected = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)

    comp = NPUCompiler(HAL)
    comp.instruction_stream = bytearray()
    tvm_backend._emit_matmul_fp32(comp, M, N, K, 0x1000, 0x40000, 0x80000)
    n_instr = len(comp.instruction_stream) // 16

    dram = bytearray(0x80000 + _padded(M) * _padded(N) * 4 + 0x1000)
    dram[0x1000:0x1000 + a.nbytes] = a.tobytes()
    dram[0x40000:0x40000 + b.nbytes] = b.tobytes()

    got, serial, db = _run_sim(comp.instruction_stream, dram)
    got = got[:M * N].reshape(M, N)

    if not np.allclose(got, expected, atol=1e-3):
        d = np.abs(got - expected)
        return False, f"wrong result: max delta {float(d.max()):.4g}", n_instr
    speedup = serial / db if db else 0
    if speedup < min_speedup:
        return False, (f"correct, but speedup only {speedup:.2f}x "
                       f"(serial={serial}, db={db}, wanted >= {min_speedup}x)"), n_instr
    return True, f"speedup {speedup:.2f}x (serial={serial}, db={db})", n_instr


def main():
    cases = [
        # k_tiles=1: double buffering degenerates to the single-tile case,
        # nothing to overlap -- just confirms this path didn't break it.
        ("matmul_1ktile_16x16x16", lambda: run_multitile_matmul_case(16, 16, 16, 1.0)),
        # Real multi-K-tile cases: expect a genuine, not marginal, speedup.
        ("matmul_4ktile_16x16x64", lambda: run_multitile_matmul_case(16, 16, 64, 1.3)),
        ("matmul_8ktile_16x16x128", lambda: run_multitile_matmul_case(16, 16, 128, 1.5)),
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
