"""
test_int8w.py -- ISA v1.8 DType.INT8W and the compiler's INT8 weight matmul.

  * INT8W is rejected by the simulator on a non-GEMM opcode
  * tvm_backend._emit_matmul_int8w_fp32 equals numpy against the DEQUANTISED
    weights (q * per-column scale) for 1-row (decode) and full-tile (prefill)
    shapes, multiple K tiles and N tiles
  * the weight blob is K*N + 4N bytes, vs 4*K*N for FP32 (about 4x smaller)
  * per-channel quantisation error is bounded by half a step per column

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_int8w.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from rtv_isa import Opcode, ActivationMode, DType, encode, to_bytes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")


def run(words, dram, dump_addr, dump_len):
    with tempfile.TemporaryDirectory() as wd:
        p, d, o = (os.path.join(wd, n) for n in ("p.bin", "d.img", "o.bin"))
        open(p, "wb").write(b"".join(to_bytes(w) for w in words))
        open(d, "wb").write(bytes(dram))
        r = subprocess.run([SIM, p, HAL, "2", "--dram-image", d,
                            "--dump", f"{dump_addr}:{dump_len}:{o}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None, (r.stdout + r.stderr)
        return open(o, "rb").read(), ""


def case_rejected_elsewhere():
    words = [encode(opcode=Opcode.EXEC_ALU, dep_id=1, sram_addr=0, dim_m=16, dim_n=16,
                    activation=int(ActivationMode.NONE), dtype=int(DType.INT8W))]
    raw, err = run(words, bytearray(0x1000), 0, 4)
    if raw is not None:
        return False, "simulator accepted INT8W on EXEC_ALU"
    return ("INT8W" in err), err[-120:]


def case_quant_error():
    sys.path.insert(0, os.path.join(ROOT, "toolchain"))
    import tvm_backend
    w = np.random.default_rng(1).normal(size=(64, 48)).astype(np.float32)
    q, s = tvm_backend.quantize_int8_per_channel(w)
    err = np.abs(q.astype(np.float32) * s - w)
    if not np.all(err <= s * 0.5 + 1e-7):
        return False, "quantisation error exceeds half a step"
    blob = tvm_backend.int8_weight_blob(w)
    if blob.nbytes != 64 * 48 + 48 * 4:
        return False, f"blob size {blob.nbytes}"
    return True, f"blob {blob.nbytes} B vs fp32 {w.nbytes} B"


def case_compiler_matmul(M, N, K):
    sys.path.insert(0, os.path.join(ROOT, "toolchain"))
    from compiler_state import NPUCompiler, DType as CDType  # noqa: E402
    import tvm_backend  # noqa: E402
    rng = np.random.default_rng(M + N + K)
    a = rng.normal(size=(M, K)).astype(np.float32)
    b = rng.normal(size=(K, N)).astype(np.float32)
    q, s = tvm_backend.quantize_int8_per_channel(b)
    expected = a.astype(np.float64) @ (q.astype(np.float64) * s.astype(np.float64))
    comp = NPUCompiler(HAL)
    comp.dtype = CDType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = 0x180000        # staging well clear of the test's A/B/C regions
    a_addr, b_addr, c_addr = 0x1000, 0x40000, 0x100000
    tvm_backend._emit_matmul_int8w_fp32(comp, M, N, K, a_addr, b_addr, c_addr)
    dram = bytearray(max(0x200000, comp.dram_bytes_needed + 0x1000))
    ap = np.zeros((16, K), np.float32)
    ap[:M] = a
    dram[a_addr:a_addr + ap.nbytes] = ap.tobytes()
    blob = tvm_backend.int8_weight_blob(b)
    dram[b_addr:b_addr + blob.nbytes] = blob.tobytes()
    raw, err = run([int.from_bytes(bytes(comp.instruction_stream[i:i + 16]), "big")
                    for i in range(0, len(comp.instruction_stream), 16)],
                   dram, c_addr, 16 * N * 4)
    if raw is None:
        return False, err[-300:]
    got = np.frombuffer(raw, dtype=np.float32).reshape(16, N)[:M]
    if not np.allclose(got, expected, rtol=1e-4, atol=1e-3):
        return False, f"max delta {float(np.abs(got - expected).max()):.4g}"
    rel = float(np.linalg.norm(got - a @ b) / np.linalg.norm(a @ b))
    return True, f"{len(comp.instruction_stream) // 16} instrs, rel err vs fp32 {rel:.4f}"


def main():
    cases = [("int8w_rejected_on_non_gemm", case_rejected_elsewhere),
             ("quantisation_error_and_blob_size", case_quant_error),
             ("compiler_matmul_16x16x16", lambda: case_compiler_matmul(16, 16, 16)),
             ("compiler_matmul_16x64x128", lambda: case_compiler_matmul(16, 64, 128)),
             ("compiler_matmul_1row_48x96", lambda: case_compiler_matmul(1, 96, 48))]
    n = 0
    for name, fn in cases:
        try:
            ok, msg = fn()
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {e}"
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {msg}" if msg else ""))
        n += ok
    print(f"\n{n}/{len(cases)} passed")
    return 0 if n == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
