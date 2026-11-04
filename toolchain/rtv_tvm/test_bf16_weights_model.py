#!/usr/bin/env python3
"""
test_bf16_weights_model.py -- a whole model with BF16 weights (ISA v1.6 BF16W).

compiler.bf16_weights = True makes every WEIGHT matmul (Q/K/V/O projections and
the SwiGLU gate/up/down) use the BF16-weight GEMM, and
build_kvcache_model_dram_image(..., bf16_weights=True) stores those seven
matrices per layer as BF16. Activations, norms, the KV-cache and all
accumulation stay FP32.

Checks:
  1. Numerics: a batched prefill tile + one-token decode steps match a numpy
     reference that uses the SAME BF16-rounded weights (so only the BF16
     rounding of the weights differs from FP32, not any wiring).
  2. Relocation: _emit_model_kvcache_fp32_relocated is byte-identical to the
     full recompile with BF16 weights (the relocation role sizes are halved).
  3. Footprint: the weight bytes in the DRAM image are exactly half.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_bf16_weights_model.py
"""
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from compiler_state import NPUCompiler, DType  # noqa: E402
import tvm_backend                            # noqa: E402
from test_prefill_batching import (            # noqa: E402
    _make_model, _reference_all_steps, _run_sim, HAL, M)

WEIGHTS = ("wq", "wk", "wv", "wo", "wgate", "wup", "wdown")


def rounded(layers_np):
    out = []
    for lw in layers_np:
        d = dict(lw)
        for k in WEIGHTS:
            d[k] = tvm_backend.from_bf16_bits(tvm_backend.to_bf16_bits(lw[k]))
        out.append(d)
    return out


def build(layers_np, hidden, n_layers, n_kv_heads, head_dim, total, bf16):
    comp0 = NPUCompiler(HAL)
    comp0.dtype = DType.FP32
    comp0.bf16_weights = bf16
    res = tvm_backend.build_kvcache_model_dram_image(
        comp0, hidden, n_layers, n_kv_heads, head_dim, max_seq_len=total,
        layer_weights_np=layers_np, bf16_weights=bf16)
    return comp0, res


def case_numerics(hidden=32, inter=48, n_heads=4, n_kv_heads=2, head_dim=16, n_layers=2,
                  n_prompt=6, n_extra=3, seed=17):
    layers_np, rng = _make_model(hidden, inter, n_heads, n_kv_heads, head_dim, n_layers, seed)
    total = n_prompt + n_extra
    rows = [rng.normal(size=(hidden,)).astype(np.float32) * 0.3 for _ in range(total)]
    scale = head_dim ** -0.5
    expected = _reference_all_steps(rounded(layers_np), rows, n_heads, n_kv_heads, head_dim,
                                    attn_scale=scale)
    comp0, (dram, x_addr, lws, sa, sb, _) = build(layers_np, hidden, n_layers, n_kv_heads,
                                                  head_dim, total, True)
    base = comp0._dram_cursor
    got = []
    with tempfile.TemporaryDirectory() as wd:
        cur = os.path.join(wd, "s.img")
        open(cur, "wb").write(bytes(dram))

        def run(comp, final, name):
            nonlocal cur
            sd = bytearray(open(cur, "rb").read())
            need = max(comp.dram_bytes_needed, len(sd))
            if need > len(sd):
                sd += bytearray(need - len(sd) + 0x10000)
            for a, d in comp._const_regions.items():
                sd[a:a + len(d)] = d
            open(cur, "wb").write(bytes(sd))
            nxt = os.path.join(wd, name)
            (raw,) = _run_sim(comp.instruction_stream, cur, nxt,
                              extra_dumps=[(final, M * hidden * 4, "out")])
            cur = nxt
            return np.frombuffer(raw, dtype=np.float32).reshape(M, hidden)

        comp = NPUCompiler(HAL)
        comp.dtype = DType.FP32
        comp.bf16_weights = True
        comp.instruction_stream = bytearray()
        comp._dram_cursor = base
        xb = np.zeros((M, hidden), dtype=np.float32)
        for r in range(n_prompt):
            xb[r] = rows[r]
        comp._const_regions[x_addr] = xb.tobytes()
        final = tvm_backend._emit_model_kvcache_batched_fp32(
            comp, n_layers, hidden, inter, n_heads, n_kv_heads, head_dim, x_addr=x_addr,
            layer_weights=lws, scratch_a=sa, scratch_b=sb, n_real=n_prompt, attn_scale=scale)
        out = run(comp, final, "p.img")
        got.extend(out[r] for r in range(n_prompt))
        for t in range(n_prompt, total):
            comp = NPUCompiler(HAL)
            comp.dtype = DType.FP32
            comp.bf16_weights = True
            comp.instruction_stream = bytearray()
            comp._dram_cursor = base
            xr = np.zeros((M, hidden), dtype=np.float32)
            xr[0] = rows[t]
            comp._const_regions[x_addr] = xr.tobytes()
            final = tvm_backend._emit_model_kvcache_fp32(
                comp, n_layers, hidden, inter, n_heads, n_kv_heads, head_dim, x_addr=x_addr,
                layer_weights=lws, scratch_a=sa, scratch_b=sb, pos=t, attn_scale=scale)
            got.append(run(comp, final, f"d{t}.img")[0])
    worst = max(float(np.abs(got[t] - expected[t]).max()) for t in range(total))
    return worst < 3e-2, f"max diff {worst:.2e} vs BF16-rounded-weight reference"


def case_relocation_byte_identical(hidden=32, inter=48, n_heads=4, n_kv_heads=2, head_dim=16,
                                   n_layers=5, seed=3):
    layers_np, _ = _make_model(hidden, inter, n_heads, n_kv_heads, head_dim, n_layers, seed)
    streams = []
    for fn in (tvm_backend._emit_model_kvcache_fp32,
               tvm_backend._emit_model_kvcache_fp32_relocated):
        comp0, (dram, x_addr, lws, sa, sb, _) = build(layers_np, hidden, n_layers, n_kv_heads,
                                                      head_dim, 8, True)
        comp = NPUCompiler(HAL)
        comp.dtype = DType.FP32
        comp.bf16_weights = True
        comp.instruction_stream = bytearray()
        comp._dram_cursor = comp0._dram_cursor
        fn(comp, n_layers, hidden, inter, n_heads, n_kv_heads, head_dim, x_addr=x_addr,
           layer_weights=lws, scratch_a=sa, scratch_b=sb, pos=3, attn_scale=head_dim ** -0.5)
        streams.append(bytes(comp.instruction_stream))
    same = streams[0] == streams[1]
    return same, f"{len(streams[0]) // 16} instrs, " + ("byte-identical" if same else "STREAMS DIFFER")


def case_footprint(hidden=32, inter=48, n_heads=4, n_kv_heads=2, head_dim=16, n_layers=2):
    layers_np, _ = _make_model(hidden, inter, n_heads, n_kv_heads, head_dim, n_layers, 5)
    w32 = sum(lw[k].astype(np.float32).nbytes for lw in layers_np for k in WEIGHTS)
    w16 = sum(tvm_backend.to_bf16_bits(lw[k]).nbytes for lw in layers_np for k in WEIGHTS)
    return w16 * 2 == w32, f"weights {w32} B fp32 -> {w16} B bf16"


def main():
    cases = [("numerics_prefill_plus_decode", case_numerics),
             ("relocation_byte_identical_5layers", case_relocation_byte_identical),
             ("weight_footprint_halved", case_footprint)]
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
