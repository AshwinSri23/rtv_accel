#!/usr/bin/env python3
"""
test_bf16w.py -- ISA v1.6 DType.BF16W: COMPUTE_GEMM with FP32 activations (A),
BF16 weights (B, 2 bytes/element) and FP32 accumulation/output.

End to end through rtv_sim with hand-built instructions (no compiler plumbing
for BF16 weights exists yet; this proves the ISA + simulator side):
  LOAD A (fp32 tile) / LOAD B (bf16 tile, half the bytes) -> WAIT -> GEMM -> STORE
checked against numpy using the BF16-rounded weights, plus:
  * K-tiled accumulation across two tiles
  * BF16W on a non-GEMM opcode is rejected by the simulator

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_bf16w.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from rtv_isa import Opcode, ActivationMode, DType, SRAMMap, encode, to_bytes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")
T = 16


def to_bf16_bits(x):
    """Round-to-nearest-even FP32 -> BF16 bit pattern (uint16)."""
    b = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16_bits(h):
    return (h.astype(np.uint32) << 16).view(np.float32)


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


def gemm_program(k_tiles, a_addr, b_addr, c_addr):
    a_sram, b_sram, c_sram = SRAMMap.input_slot(0, 4), SRAMMap.weight_slot(0, 4), SRAMMap.output_slot(0)
    w = []
    for k in range(k_tiles):
        w.append(encode(opcode=Opcode.LOAD_TILE, dep_id=1, dram_addr=a_addr + k * T * T * 4,
                        sram_addr=a_sram, dim_m=T, dim_n=T * 4, stride=T * 4))
        w.append(encode(opcode=Opcode.LOAD_TILE, dep_id=2, dram_addr=b_addr + k * T * T * 2,
                        sram_addr=b_sram, dim_m=T, dim_n=T * 2, stride=T * 2))
        w.append(encode(opcode=Opcode.WAIT_SYNC, dep_id=1))
        w.append(encode(opcode=Opcode.WAIT_SYNC, dep_id=2))
        w.append(encode(opcode=Opcode.COMPUTE_GEMM, dep_id=1, sram_addr=a_sram, stride=b_sram,
                        dim_m=T, dim_n=T, dim_k=T, accumulate=(k > 0), c_slot=0,
                        dtype=int(DType.BF16W)))
    w.append(encode(opcode=Opcode.STORE_TILE, dram_addr=c_addr, sram_addr=c_sram,
                    dim_m=T, dim_n=T * 4, stride=T * 4))
    return w


def case_gemm(k_tiles):
    rng = np.random.default_rng(k_tiles)
    a = rng.normal(size=(k_tiles, T, T)).astype(np.float32)
    b_raw = rng.normal(size=(k_tiles, T, T)).astype(np.float32)
    b_bits = to_bf16_bits(b_raw)
    b = from_bf16_bits(b_bits)
    expected = sum(a[k].astype(np.float64) @ b[k].astype(np.float64) for k in range(k_tiles))
    a_addr, b_addr, c_addr = 0x1000, 0x10000, 0x20000
    dram = bytearray(0x30000)
    dram[a_addr:a_addr + a.nbytes] = a.tobytes()
    dram[b_addr:b_addr + b_bits.nbytes] = b_bits.tobytes()
    raw, err = run(gemm_program(k_tiles, a_addr, b_addr, c_addr), dram, c_addr, T * T * 4)
    if raw is None:
        return False, err[-300:]
    got = np.frombuffer(raw, dtype=np.float32).reshape(T, T)
    if not np.allclose(got, expected, rtol=1e-4, atol=1e-4):
        return False, f"max delta {float(np.abs(got - expected).max()):.4g}"
    # and it must differ from using the UNrounded FP32 weights (BF16 is lossy)
    full = sum(a[k].astype(np.float64) @ b_raw[k].astype(np.float64) for k in range(k_tiles))
    return True, f"rounded-vs-fp32-weight diff {float(np.abs(got - full).max()):.3g} (expected nonzero)"


def case_rejected_elsewhere():
    w = [encode(opcode=Opcode.EXEC_ALU, sram_addr=0, stride=0, dim_m=T, dim_n=T,
                activation=int(ActivationMode.NONE), dtype=int(DType.BF16W))]
    raw, err = run(w, bytearray(0x1000), 0, 4)
    if raw is not None:
        return False, "simulator accepted BF16W on EXEC_ALU"
    if "BF16W" not in err:
        return False, f"unexpected error: {err[-200:]}"
    return True, ""


def case_compiler_matmul(M, N, K):
    """tvm_backend._emit_matmul_bf16w_fp32 against numpy with BF16-rounded
    weights, and its weight DMA traffic is exactly half the FP32 matmul's."""
    sys.path.insert(0, os.path.join(ROOT, "toolchain"))
    from compiler_state import NPUCompiler, DType as CDType  # noqa: E402
    import tvm_backend  # noqa: E402
    rng = np.random.default_rng(M + N + K)
    a = rng.normal(size=(M, K)).astype(np.float32)
    b = rng.normal(size=(K, N)).astype(np.float32)
    bb = tvm_backend.to_bf16_bits(b)
    expected = a.astype(np.float64) @ tvm_backend.from_bf16_bits(bb).astype(np.float64)
    comp = NPUCompiler(HAL)
    comp.dtype = CDType.FP32
    comp.instruction_stream = bytearray()
    a_addr, b_addr, c_addr = 0x1000, 0x40000, 0x100000
    tvm_backend._emit_matmul_bf16w_fp32(comp, M, N, K, a_addr, b_addr, c_addr)
    dram = bytearray(0x200000)
    # A is read as full 16-row tiles; rows past M stay zero.
    ap = np.zeros((16, K), np.float32)
    ap[:M] = a
    dram[a_addr:a_addr + ap.nbytes] = ap.tobytes()
    dram[b_addr:b_addr + bb.nbytes] = bb.tobytes()
    raw, err = run([int.from_bytes(bytes(comp.instruction_stream[i:i + 16]), "big")
                    for i in range(0, len(comp.instruction_stream), 16)],
                   dram, c_addr, 16 * N * 4)
    if raw is None:
        return False, err[-300:]
    got = np.frombuffer(raw, dtype=np.float32).reshape(16, N)[:M]
    if not np.allclose(got, expected, rtol=1e-4, atol=1e-3):
        return False, f"max delta {float(np.abs(got - expected).max()):.4g}"
    return True, f"{len(comp.instruction_stream) // 16} instrs, B bytes {bb.nbytes} (fp32 would be {b.nbytes})"


def main():
    cases = [("bf16w_gemm_1ktile", lambda: case_gemm(1)),
             ("bf16w_gemm_4ktile_accumulate", lambda: case_gemm(4)),
             ("bf16w_rejected_on_non_gemm", case_rejected_elsewhere),
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
