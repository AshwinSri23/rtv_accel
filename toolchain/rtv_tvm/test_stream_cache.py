#!/usr/bin/env python3
"""
test_stream_cache.py -- codegen.emit's instruction-stream cache (template with fake DRAM bases,
real bases patched in) must give BYTE-IDENTICAL streams to a normal emit.

  1. matmuls of several shapes/strides at several address triples, including addresses above 4 GiB
     (the 36-bit split), emitted twice (miss then hit) and compared with RTV_NO_STREAM_CACHE
  2. a whole 4-layer model decode step (relocated path), cached vs uncached
  3. hits actually happen (the cache is not silently bypassed) and indirect loads are not cached

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_stream_cache.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from compiler_state import NPUCompiler, DType     # noqa: E402
import tvm_backend                                # noqa: E402
from rtv_tvm import codegen                      # noqa: E402
from test_schedule_relocation import _make_model, HAL, M   # noqa: E402


def stream(fn, cached):
    if cached:
        os.environ.pop("RTV_NO_STREAM_CACHE", None)
    else:
        os.environ["RTV_NO_STREAM_CACHE"] = "1"
    try:
        c = NPUCompiler(HAL)
        c.dtype = DType.FP32
        c.instruction_stream = bytearray()
        fn(c)
        return bytes(c.instruction_stream)
    finally:
        os.environ.pop("RTV_NO_STREAM_CACHE", None)


def case_matmuls():
    shapes = [(16, 16, 16, None, None, None), (16, 48, 32, None, None, None),
              (16, 16, 64, 128, None, None), (16, 32, 48, None, 64, 96), (16, 128, 16, None, None, None)]
    addrs = [(0x1000, 0x20000, 0x40000), (0x500000, 0x10000, 0x300000),
             (0x1_2000_0000, 0x4000, 0x2_0000_8000), (0xF_0000_0000, 0x1_0000_0000, 0x2_0000_0000)]
    bad = []
    for sh in shapes:
        Mm, N, K, a_s, b_s, c_s = sh
        for ad in addrs * 2:                                     # second pass is all hits
            f = lambda c: tvm_backend._emit_matmul_fp32(c, Mm, N, K, *ad, a_stride=a_s,
                                                        b_stride=b_s, c_stride=c_s)
            if stream(f, True) != stream(f, False):
                bad.append((sh, ad))
    return not bad, f"{len(shapes) * len(addrs) * 2} emits compared" + (f"; MISMATCH {bad[:2]}" if bad else "")


def case_model():
    layers_np, rng = _make_model(32, 48, 4, 2, 16, 4, 9)
    tok = rng.normal(size=(32,)).astype(np.float32) * 0.3

    def build(cached):
        def go(_c):
            comp0 = NPUCompiler(HAL)
            comp0.dtype = DType.FP32
            dram, x_addr, lws, sa, sb, _ = tvm_backend.build_kvcache_model_dram_image(
                comp0, 32, 4, 2, 16, max_seq_len=3, layer_weights_np=layers_np)
            _c._dram_cursor = comp0._dram_cursor
            xr = np.zeros((M, 32), np.float32)
            xr[0] = tok
            _c._const_regions[x_addr] = xr.tobytes()
            tvm_backend._emit_model_kvcache_fp32_relocated(
                _c, 4, 32, 48, 4, 2, 16, x_addr=x_addr, layer_weights=lws, scratch_a=sa,
                scratch_b=sb, pos=2)
        return stream(go, cached)
    a, b = build(True), build(False)
    return a == b, f"{len(a) // 16} instructions"


def case_hits_and_indirect():
    codegen._STREAM_CACHE.clear()
    for k in codegen.STREAM_CACHE_STATS:
        codegen.STREAM_CACHE_STATS[k] = 0
    for ad in ((0x1000, 0x20000, 0x40000), (0x9000, 0x80000, 0x70000)):
        c = NPUCompiler(HAL)
        c.dtype = DType.FP32
        c.instruction_stream = bytearray()
        tvm_backend._emit_matmul_fp32(c, 16, 32, 32, *ad)
    st = dict(codegen.STREAM_CACHE_STATS)
    c = NPUCompiler(HAL)
    c.dtype = DType.FP32
    c.instruction_stream = bytearray()
    tvm_backend._emit_matmul_fp32(c, 16, 32, 32, 0x1000, 0x20000, 0x40000, indirect_b=0x3F00)
    ok = st["miss"] == 1 and st["hit"] == 1 and codegen.STREAM_CACHE_STATS == st
    return ok, f"{st}; indirect emit left the cache untouched"


def main():
    cases = [("matmuls_cached_equals_uncached", case_matmuls),
             ("whole_model_step_cached_equals_uncached", case_model),
             ("hits_happen_indirect_not_cached", case_hits_and_indirect)]
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
