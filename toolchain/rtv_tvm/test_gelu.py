#!/usr/bin/env python3
"""test_gelu.py -- GELU(tanh) composed from existing ISA ops (multiply, add, RELU, exp; no new hardware
mode) must match the exact function on the simulator, across small, mid and saturating inputs."""
import os, subprocess, sys, tempfile
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from compiler_state import NPUCompiler, DType  # noqa: E402
import tvm_backend                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")


def gelu_ref(x):
    x = x.astype(np.float64)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def run(x, after_reduce=False):
    M, W = x.shape
    c = NPUCompiler(HAL)
    c.dtype = DType.FP32
    c.instruction_stream = bytearray()
    c._dram_cursor = 0x1000
    xa = c.alloc_staging(M * W * 4)
    oa = c.alloc_staging(M * W * 4)
    c._const_regions[xa] = x.astype(np.float32).tobytes()
    if after_reduce:
        # an RMSNorm first leaves a NONZERO scalar in the SRAM scalar slot; GELU's exp must not use it
        g = c.alloc_staging(M * 16 * 4)
        c._const_regions[g] = np.ones((M, 16), np.float32).tobytes()
        src = c.alloc_staging(M * 16 * 4)
        row = np.zeros((M, 16), np.float32)
        row[0] = np.linspace(2.0, 9.0, 16)
        c._const_regions[src] = row.tobytes()
        c.rms_eps = 1e-6
        tvm_backend._emit_rmsnorm_row_fp32(c, 16, x_addr=src, gamma_addr=g, out_addr=c.alloc_staging(M * 16 * 4),
                                           scratch_addr=c.alloc_staging(4),
                                           stage_addr=c.alloc_staging(M * 16 * 4))
    tvm_backend._emit_gelu_tanh_fp32(c, W, xa, oa)
    with tempfile.TemporaryDirectory() as wd:
        img, prog, out = (os.path.join(wd, n) for n in ("img", "prog", "out"))
        buf = bytearray(c.dram_bytes_needed + 4096)
        for a, d in c._const_regions.items():
            buf[a:a + len(d)] = bytes(d)
        open(img, "wb").write(bytes(buf))
        open(prog, "wb").write(bytes(c.instruction_stream))
        mb = max(1, len(buf) // (1024 * 1024) + 2)
        r = subprocess.run([SIM, prog, HAL, str(mb), "--dram-image", img,
                            "--dump", f"{oa}:{M * W * 4}:{out}"], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stdout[-600:] + r.stderr[-600:])
        return np.frombuffer(open(out, "rb").read(), np.float32).reshape(M, W), len(c.instruction_stream) // 16


def main():
    rng = np.random.default_rng(0)
    cases = {
        "normal": rng.normal(size=(16, 32)) * 2,
        "wide_range": rng.uniform(-12, 12, size=(16, 32)),
        "tiny": rng.normal(size=(16, 32)) * 1e-4,
        "huge": rng.uniform(-300, 300, size=(16, 16)),
    }
    ok = True
    cases["after_norm"] = cases["normal"]
    for name, x in cases.items():
        got, n = run(x, after_reduce=(name == "after_norm"))
        ref = gelu_ref(x.astype(np.float32))
        err = np.abs(got - ref)
        rel = float((err / np.maximum(np.abs(ref), 1e-3)).max())
        good = err.max() < 2e-5 and np.isfinite(got).all()
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  gelu_{name}  max abs {err.max():.2e}  ({n} instrs)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
