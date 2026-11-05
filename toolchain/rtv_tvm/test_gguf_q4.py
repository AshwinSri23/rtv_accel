#!/usr/bin/env python3
"""
test_gguf_q4.py -- llama.cpp Q4_0 blocks -> the chip's INT4W g32 format is lossless.

Random 18-byte Q4_0 blocks (fp16 scale + 16 bytes of nibbles, low nibble = element 0..15, high = 16..31) are wrapped
in gemma4_gguf.Q4Mat. Checks that (1) dense() equals an independent per-block dequantisation, (2) the INT4W blob
that goes into the DRAM image decodes (nibble unpack + per-(K-group, column) scale) to exactly the same matrix, and
(3) the same matrix pushed through the chip's INT4W GEMM gives the NumPy product of the dequantised weights.
"""
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(ROOT, "toolchain"))
sys.path.insert(0, _HERE)

from gemma4_gguf import Q4Mat            # noqa: E402
from gguf_reader import GGUF             # noqa: E402,F401  (import check)
import tvm_backend                       # noqa: E402
from compiler_state import NPUCompiler, DType  # noqa: E402


def reference_dense(raw):
    N, nblk, _ = raw.shape
    w = np.zeros((N, nblk * 32), np.float32)
    for n in range(N):
        for b in range(nblk):
            blk = raw[n, b]
            d = np.frombuffer(blk[:2].tobytes(), np.float16)[0].astype(np.float32)
            for i in range(16):
                w[n, b * 32 + i] = (int(blk[2 + i] & 15) - 8) * d
                w[n, b * 32 + 16 + i] = (int(blk[2 + i] >> 4) - 8) * d
    return w.T                                            # [K, N]


def main():
    rng = np.random.default_rng(0)
    N, nblk = 48, 4                                      # K = 128, N = 48 (output columns)
    raw = rng.integers(0, 256, (N, nblk, 18), dtype=np.uint8)
    raw[:, :, :2] = (rng.uniform(-0.5, 0.5, (N, nblk, 1)).astype(np.float16)).view(np.uint8)   # sane fp16 scales
    m = Q4Mat(raw)
    K = m.K
    results = []

    ref = reference_dense(raw)
    results.append(("dense() == per-block dequant", np.array_equal(m.dense(), ref), ""))

    b = m.blob()
    pk = b[:K * N // 2]
    lo, hi = (pk & 15).astype(np.int8), (pk >> 4).astype(np.int8)
    lo, hi = np.where(lo > 7, lo - 16, lo), np.where(hi > 7, hi - 16, hi)
    q = np.empty(K * N, np.int8)
    q[0::2], q[1::2] = lo, hi
    sc = b[K * N // 2:].view(np.float32).reshape(K // 32, N)
    dec = q.reshape(K, N).astype(np.float32) * np.repeat(sc, 32, axis=0)
    results.append(("blob decodes to the same matrix", np.array_equal(dec, ref),
                    f"blob {b.nbytes} B = {tvm_backend.int4_blob_nbytes(K, N, 32)}"))
    results.append(("blob size == int4_blob_nbytes", b.nbytes == tvm_backend.int4_blob_nbytes(K, N, 32), ""))

    # On the chip: y = x @ W through the INT4W GEMM.
    sim, hal = os.path.join(ROOT, "build", "rtv_sim"), os.path.join(ROOT, "toolchain", "hal_config.json")
    M = 16
    x = rng.standard_normal((M, K)).astype(np.float32)
    a_addr, b_addr = 0x1000, 0x4000
    c_addr = (b_addr + b.nbytes + 0xFFF) & ~0xFFF
    c = NPUCompiler(hal)
    c.dtype = DType.FP32
    c.instruction_stream = bytearray()
    c._dram_cursor = c_addr + 0x10000
    tvm_backend._emit_matmul_int4w_fp32(c, M, N, K, a_addr, b_addr, c_addr, group=32)
    image = bytearray(c_addr + M * N * 4 + 0x10000)
    image[a_addr:a_addr + x.nbytes] = x.tobytes()
    image[b_addr:b_addr + b.nbytes] = b.tobytes()
    import subprocess
    with tempfile.TemporaryDirectory() as wd:
        prog, img, out = (os.path.join(wd, n) for n in ("p.bin", "i.img", "o.bin"))
        open(prog, "wb").write(bytes(c.instruction_stream))
        open(img, "wb").write(bytes(image))
        subprocess.run([sim, prog, hal, "2", "--dram-image", img, "--dump", f"{c_addr}:{M * N * 4}:{out}"],
                       check=True, capture_output=True)
        y = np.frombuffer(open(out, "rb").read(), np.float32).reshape(M, N)
    want = x.astype(np.float64) @ ref.astype(np.float64)
    err = float(np.abs(y - want).max() / np.abs(want).max())
    results.append(("chip INT4W GEMM == x @ dequant", err < 1e-5, f"rel err {err:.2e}"))

    n = 0
    for name, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {msg}" if msg else ""))
        n += ok
    print(f"\n{n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
