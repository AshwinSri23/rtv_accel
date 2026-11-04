#!/usr/bin/env python3
"""
test_prefill_multitile.py -- prefill of a prompt LONGER than one 16-row tile.

test_prefill_batching.py covers up to 16 tokens in one causal batch. Past that,
tile t's rows must attend to every earlier tile's cached K/V as well as causally
to their own tile. `_emit_model_kvcache_batched_fp32(..., tile_idx=t)` does that:
RoPE positions are 16t+r, K/V go into cache tile t, attention runs over cache
tiles 0..t with a mask that shows earlier tiles in full and this tile causally,
and the softmax spans all 16(t+1) key columns.

Checked the strong way, as test_prefill_batching does: the model output for EVERY
prompt token (from the tile outputs) and for the tokens decoded afterwards one at a
time (reading the cache the tiles wrote) must match a reference that processes the
whole sequence purely one token at a time from the start.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_prefill_multitile.py
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


def run_case(hidden, inter, n_heads, n_kv_heads, head_dim, n_layers, n_prompt, n_extra,
             seed=71, attn_scale=None):
    layers_np, rng = _make_model(hidden, inter, n_heads, n_kv_heads, head_dim, n_layers, seed)
    total = n_prompt + n_extra
    token_rows = [rng.normal(size=(hidden,)).astype(np.float32) * 0.3 for _ in range(total)]
    expected = _reference_all_steps(layers_np, token_rows, n_heads, n_kv_heads, head_dim,
                                    attn_scale=attn_scale)

    comp0 = NPUCompiler(HAL)
    comp0.dtype = DType.FP32
    dram, x_addr, layer_weights, scratch_a, scratch_b, max_tiles = \
        tvm_backend.build_kvcache_model_dram_image(
            comp0, hidden, n_layers, n_kv_heads, head_dim,
            max_seq_len=total, layer_weights_np=layers_np)
    base = comp0._dram_cursor
    got = []
    n_tiles = -(-n_prompt // M)

    def step(comp, wd, cur, name):
        sd = bytearray(open(cur, "rb").read())
        need = max(comp.dram_bytes_needed, len(sd))
        if need > len(sd):
            sd += bytearray(need - len(sd) + 0x10000)
        for a, d in comp._const_regions.items():
            sd[a:a + len(d)] = d
        open(cur, "wb").write(bytes(sd))
        nxt = os.path.join(wd, name)
        return nxt

    with tempfile.TemporaryDirectory(prefix="rtv_prefill_mt_") as wd:
        cur = os.path.join(wd, "state.img")
        open(cur, "wb").write(bytes(dram))
        n_instr = 0
        for c in range(n_tiles):
            n_real = min(M, n_prompt - c * M)
            comp = NPUCompiler(HAL)
            comp.dtype = DType.FP32
            comp.instruction_stream = bytearray()
            comp._dram_cursor = base
            xb = np.zeros((M, hidden), dtype=np.float32)
            for r in range(n_real):
                xb[r] = token_rows[c * M + r]
            comp._const_regions[x_addr] = xb.tobytes()
            final = tvm_backend._emit_model_kvcache_batched_fp32(
                comp, n_layers, hidden, inter, n_heads, n_kv_heads, head_dim,
                x_addr=x_addr, layer_weights=layer_weights, scratch_a=scratch_a,
                scratch_b=scratch_b, n_real=n_real, tile_idx=c,
                attn_scale=attn_scale)
            n_instr = len(comp.instruction_stream) // 16
            nxt = step(comp, wd, cur, f"t{c}.img")
            (raw,) = _run_sim(comp.instruction_stream, cur, nxt,
                              extra_dumps=[(final, M * hidden * 4, "out")])
            cur = nxt
            out = np.frombuffer(raw, dtype=np.float32).reshape(M, hidden)
            got.extend(out[r] for r in range(n_real))
        for t in range(n_prompt, total):
            comp = NPUCompiler(HAL)
            comp.dtype = DType.FP32
            comp.instruction_stream = bytearray()
            comp._dram_cursor = base
            xr = np.zeros((M, hidden), dtype=np.float32)
            xr[0] = token_rows[t]
            comp._const_regions[x_addr] = xr.tobytes()
            final = tvm_backend._emit_model_kvcache_fp32(
                comp, n_layers, hidden, inter, n_heads, n_kv_heads, head_dim,
                x_addr=x_addr, layer_weights=layer_weights, scratch_a=scratch_a,
                scratch_b=scratch_b, pos=t, attn_scale=attn_scale)
            nxt = step(comp, wd, cur, f"d{t}.img")
            (raw,) = _run_sim(comp.instruction_stream, cur, nxt,
                              extra_dumps=[(final, M * hidden * 4, "out")])
            cur = nxt
            got.append(np.frombuffer(raw, dtype=np.float32).reshape(M, hidden)[0])

    worst, at = 0.0, -1
    for t in range(total):
        d = float(np.abs(got[t] - expected[t]).max())
        if d > worst:
            worst, at = d, t
    if worst < 3e-2:
        return True, f"{n_tiles} tiles, max diff {worst:.2e}", n_instr
    return False, f"token {at} wrong, max diff {worst:.4g}", n_instr


def main():
    cases = [
        ("two_tiles_16_plus_4_then_decode", lambda: run_case(32, 48, 4, 2, 16, 2, 20, 3)),
        ("two_full_tiles_then_decode", lambda: run_case(32, 48, 4, 2, 16, 2, 32, 2)),
        ("three_tiles_40_headdim32", lambda: run_case(48, 64, 2, 1, 32, 2, 40, 2)),
        # attn_scale = head_dim**-0.5: real attention scores are q.k/sqrt(head_dim).
        # Missing from the production path until the Hugging Face comparison caught it.
        ("scaled_two_tiles_then_decode", lambda: run_case(32, 48, 4, 2, 16, 2, 20, 3,
                                                         attn_scale=16 ** -0.5)),
        ("scaled_headdim32_one_tile", lambda: run_case(48, 64, 2, 1, 32, 2, 10, 2,
                                                       attn_scale=32 ** -0.5)),
    ]
    n = 0
    for name, fn in cases:
        try:
            ok, msg, ni = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok, msg, ni = False, f"{type(e).__name__}: {e}", 0
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({ni} instrs/tile)  -- {msg}")
        n += ok
    print(f"\n{n}/{len(cases)} passed")
    return 0 if n == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
