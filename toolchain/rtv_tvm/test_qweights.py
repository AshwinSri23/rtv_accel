#!/usr/bin/env python3
"""
test_qweights.py -- Q4_K / Q8_0 / Q6_K style weights on the existing INT4W hardware (toolchain/quant_planes.py):
INT4W planes plus a per-group correction through the activation's group sums. For each scheme a random quantised
matrix is built in its native form, the chip multiplies an activation by it (1 row and a full tile, several K and N
tiles), and the result must equal the activation times the exactly dequantised matrix.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_qweights.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(ROOT, "toolchain"))
sys.path.insert(0, os.path.join(ROOT, "toolchain", "rtv_tvm"))

import quant_planes  # noqa: E402
import tvm_backend  # noqa: E402
from compiler_state import NPUCompiler, DType as CDType  # noqa: E402

SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")


def run(stream, dram, dump_addr, dump_len):
    with tempfile.TemporaryDirectory() as wd:
        p, d, o = (os.path.join(wd, n) for n in ("p.bin", "d.img", "o.bin"))
        open(p, "wb").write(bytes(stream))
        open(d, "wb").write(bytes(dram))
        r = subprocess.run([SIM, p, HAL, str(len(dram) // 2**20 + 1), "--dram-image", d, "--dump", f"{dump_addr}:{dump_len}:{o}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None, r.stdout + r.stderr
        return open(o, "rb").read(), ""


def make(kind, rng, K, N):
    if kind == "q4k":
        q = rng.integers(0, 16, size=(K, N))
        s = rng.uniform(0.01, 0.2, size=(K // 32, N))
        m = rng.uniform(-0.3, 0.5, size=(K // 32, N))
        return quant_planes.asym4(q, s, m, 32)
    if kind == "q8_0":
        return quant_planes.sym8(rng.integers(-128, 128, size=(K, N)), rng.uniform(0.001, 0.02, size=(K // 32, N)))
    if kind == "q6k":
        s = rng.integers(-60, 60, size=(K // 16, N)) * np.float32(0.0007)
        return quant_planes.sym6(rng.integers(0, 64, size=(K, N)), s)
    raise ValueError(kind)


def case(kind, M, N, K):
    rng = np.random.default_rng(hash((kind, M, N, K)) & 0xFFFF)
    spec = make(kind, rng, K, N)
    a = rng.normal(size=(M, K)).astype(np.float32)
    expected = a.astype(np.float64) @ quant_planes.dequant(spec, K)

    comp = NPUCompiler(HAL)
    comp.dtype = CDType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = 0x4000000
    dram = bytearray(0x4000000 + 0x1000000)
    cur = [0x1000]

    def put(b):
        b = bytes(b)
        addr = cur[0]
        dram[addr:addr + len(b)] = b
        cur[0] = (addr + len(b) + 0xFFF) & ~0xFFF
        return addr

    ap = np.zeros((16, K), np.float32)
    ap[:M] = a
    a_addr = put(ap.tobytes())
    c_addr = put(bytes(16 * N * 4))
    planes = [(put(tvm_backend.int4_weight_blob(v, s).tobytes()), g) for v, s, g in spec["planes"]]
    reg = dict(planes=planes, corr=None)
    comp.group_sums = {}
    if spec["corr"] is not None:
        g = spec["corr"][1]
        reg["corr"] = (put(quant_planes.corr_matrix(spec, K).tobytes()), g)
        blob, _ = tvm_backend.group_sum_blob(K, g)
        comp.group_sums[(K, g)] = put(blob.tobytes())
    comp.qweights = {planes[0][0]: reg}
    tvm_backend._emit_weight_matmul_fp32(comp, M, N, K, a_addr, planes[0][0], c_addr)
    raw, err = run(comp.instruction_stream, dram, c_addr, 16 * N * 4)
    if raw is None:
        return False, err[-300:]
    got = np.frombuffer(raw, np.float32).reshape(16, N)[:M]
    rel = float(np.abs(got - expected).max() / np.abs(expected).max())
    return rel < 2e-5, f"{len(comp.instruction_stream) // 16} instrs, max rel err {rel:.2e}"


def main():
    cases = [(k, M, N, K) for k in ("q4k", "q8_0", "q6k") for (M, N, K) in ((1, 32, 64), (16, 48, 96), (1, 64, 224))]
    n = 0
    for k, M, N, K in cases:
        try:
            ok, msg = case(k, M, N, K)
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {e}"
        print(f"{'PASS' if ok else 'FAIL'}  {k} M={M} N={N} K={K}  -- {msg}")
        n += ok
    print(f"\n{n}/{len(cases)} passed")
    return 0 if n == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
