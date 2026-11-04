#!/usr/bin/env python3
"""
test_staging_reuse.py -- compiler.reuse_layer_staging: layers 2+ run in layer 1's staging region
instead of each taking fresh DRAM (TODO Phase 5.2, liveness-based memory planning, per-layer form).

Why it is safe: layer 0 is cold (builds the shared constants), layer 1 is warm, so layer 1's staging
is per-layer-only transients; a layer's output goes to the ping-pong scratch, its KV-cache and
weights are separate roles. What could break it is an emitter that assumes freshly zeroed staging
(row-0-real/rest-zero buffers), so the checks are numeric, not just structural:

  1. footprint: staging peak with reuse is ~2 layers' worth instead of n_layers' worth
  2. full path == relocated path, byte for byte, with reuse on (the relocation acceptance bar)
  3. simulator output with reuse is BIT-IDENTICAL to without, at pos 0 and at pos 2, 6 layers,
     random weights (stale data from earlier layers is non-zero)
  4. a layer that would build a new shared constant after layer 1 is rejected, not silently clobbered

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_staging_reuse.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from compiler_state import NPUCompiler, DType     # noqa: E402
import tvm_backend                                # noqa: E402
from test_schedule_relocation import _make_model, SIM, HAL, M   # noqa: E402

CFG = dict(hidden=32, inter=48, n_heads=4, n_kv=2, head_dim=16, n_layers=6)


def build(reuse, relocated, pos, n_layers=None):
    c = dict(CFG, n_layers=n_layers or CFG["n_layers"])
    layers_np, rng = _make_model(c["hidden"], c["inter"], c["n_heads"], c["n_kv"], c["head_dim"],
                                 c["n_layers"], 5)
    tok = rng.normal(size=(c["hidden"],)).astype(np.float32) * 0.3
    comp0 = NPUCompiler(HAL)
    comp0.dtype = DType.FP32
    dram, x_addr, lws, sa, sb, _ = tvm_backend.build_kvcache_model_dram_image(
        comp0, c["hidden"], c["n_layers"], c["n_kv"], c["head_dim"], max_seq_len=pos + 1,
        layer_weights_np=layers_np)
    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = comp0._dram_cursor
    comp.reuse_layer_staging = reuse
    xr = np.zeros((M, c["hidden"]), np.float32)
    xr[0] = tok
    comp._const_regions[x_addr] = xr.tobytes()
    fn = (tvm_backend._emit_model_kvcache_fp32_relocated if relocated
          else tvm_backend._emit_model_kvcache_fp32)
    final = fn(comp, c["n_layers"], c["hidden"], c["inter"], c["n_heads"], c["n_kv"], c["head_dim"],
               x_addr=x_addr, layer_weights=lws, scratch_a=sa, scratch_b=sb, pos=pos)
    return comp, dram, final


def run(comp, dram, final):
    dram = bytearray(dram)
    need = max(comp.dram_bytes_needed, len(dram))
    dram += bytearray(need - len(dram) + 0x10000)
    for a, d in comp._const_regions.items():
        dram[a:a + len(d)] = d
    with tempfile.TemporaryDirectory() as wd:
        p, d, o = (os.path.join(wd, n) for n in ("p", "d", "o"))
        open(p, "wb").write(bytes(comp.instruction_stream))
        open(d, "wb").write(bytes(dram))
        r = subprocess.run([SIM, p, HAL, str(len(dram) // (1 << 20) + 2), "--dram-image", d,
                            "--dump", f"{final}:{M * CFG['hidden'] * 4}:{o}"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-300:] + r.stderr[-300:]
        return open(o, "rb").read()


def case_footprint():
    used = lambda c: c._staging_peak - c._staging_base      # staging sits above the weights
    a, _, _ = build(False, True, 0)
    b, _, _ = build(True, True, 0)
    b4, _, _ = build(True, True, 0, n_layers=4)
    # with reuse the staging used is layer 0 + layer 1 whatever the depth; without it grows per layer
    ok = used(b) == used(b4) < used(a)
    return ok, (f"6 layers: {used(a)} -> {used(b)} bytes of staging; with reuse at 4 layers "
                f"{used(b4)} (depth-independent)")


def case_byte_identical():
    a, _, fa = build(True, False, 1)
    b, _, fb = build(True, True, 1)
    ok = fa == fb and bytes(a.instruction_stream) == bytes(b.instruction_stream)
    return ok, f"{len(a.instruction_stream) // 16} instructions"


def case_numerics(pos):
    def go(reuse, relocated):
        comp, dram, final = build(reuse, relocated, pos)
        return run(comp, dram, final)
    base = go(False, True)
    ok = go(True, True) == base and go(True, False) == base
    return ok, "output bit-identical with and without reuse (relocated and full paths)"


def case_new_const_rejected():
    c = CFG
    comp, _, _ = build(True, False, 0)          # a normal run does not raise
    # emulate a layer-2 emitter that creates a shared constant layer 1 did not
    orig = tvm_backend._emit_transformer_layer_kvcache_fp32
    calls = [0]

    def spy(compiler, *a, **k):
        calls[0] += 1
        if calls[0] == 3:
            tvm_backend._get_or_build_const(compiler, ("test_only", 1), 256,
                                            lambda: np.zeros(64, np.float32))
        return orig(compiler, *a, **k)

    tvm_backend._emit_transformer_layer_kvcache_fp32 = spy
    try:
        try:
            build(True, False, 0)
        except RuntimeError as e:
            return "shared constant" in str(e), "raised as intended"
        return False, "no error"
    finally:
        tvm_backend._emit_transformer_layer_kvcache_fp32 = orig


def main():
    cases = [("footprint", case_footprint), ("full_equals_relocated_bytes", case_byte_identical),
             ("numerics_pos0", lambda: case_numerics(0)), ("numerics_pos2", lambda: case_numerics(2)),
             ("new_shared_const_after_layer1_rejected", case_new_const_rejected)]
    n = 0
    for name, fn in cases:
        try:
            ok, msg = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok, msg = False, f"{type(e).__name__}: {e}"
        print(f"{'PASS' if ok else 'FAIL'}  {name}  -- {msg}")
        n += ok
    print(f"\n{n}/{len(cases)} passed")
    return 0 if n == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
