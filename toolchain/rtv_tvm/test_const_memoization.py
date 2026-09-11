#!/usr/bin/env python3
"""
test_const_memoization.py — tvm_backend._get_or_build_const: several
"constant matrix via GEMM" helpers (RoPE's rotate_half permutation
matrix, the row0<->broadcast masks, the KV-cache tail/append masks)
build content that depends only on decode-step-global values (head_dim,
seq_k, pos, theta, ...), never on which head or layer is calling --
before this, every call rebuilt and re-staged a fresh, byte-identical
copy anyway.

This is NOT a numeric-correctness test on its own -- every existing
test in this directory (test_decode_loop.py, test_checkpoint_decode_
loop.py, etc.) already re-verifies numeric correctness with memoization
active, since it's now just always-on inside the functions those tests
already call. What this file checks is the thing that's actually new
and could silently regress without changing any number: that the
CACHE HIT RATE is real, not just harmless-but-inert code. Two things:

1. Emitting the SAME decode step with MORE layers does not grow
   `compiler._const_cache`/the const-region byte total proportionally
   with layer count -- if it did, the memoization would have silently
   stopped working (e.g. a cache key accidentally including something
   that varies per layer) while every other test would still pass,
   since correctness doesn't depend on whether sharing happened.
2. The specific high-multiplicity case (RoPE, called 24x/layer at
   Qwen3-like head counts) collapses to exactly ONE cache entry
   regardless of how many layers/heads asked for it this step.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_const_memoization.py
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from compiler_state import NPUCompiler, DType  # noqa: E402
import tvm_backend                             # noqa: E402

M = 16


def _build_and_emit(n_layers, hidden, inter, n_heads, n_kv_heads, head_dim, seed=3):
    rng = np.random.default_rng(seed)
    q_dim, kv_dim = n_heads * head_dim, n_kv_heads * head_dim

    def w_(shape, s=0.2):
        return (rng.normal(size=shape) * s).astype(np.float32)

    layers_np = []
    for _ in range(n_layers):
        layers_np.append(dict(
            gamma1=rng.uniform(0.5, 1.5, size=(hidden,)).astype(np.float32),
            gamma2=rng.uniform(0.5, 1.5, size=(hidden,)).astype(np.float32),
            wq=w_((hidden, q_dim)), wk=w_((hidden, kv_dim)), wv=w_((hidden, kv_dim)),
            wo=w_((q_dim, hidden)), wgate=w_((hidden, inter)), wup=w_((hidden, inter)),
            wdown=w_((inter, hidden)),
        ))

    comp0 = NPUCompiler(os.path.join(os.path.dirname(_HERE), "hal_config.json"))
    comp0.dtype = DType.FP32
    dram, x_addr, layer_weights, scratch_a, scratch_b, max_tiles = \
        tvm_backend.build_kvcache_model_dram_image(
            comp0, hidden, n_layers, n_kv_heads, head_dim,
            max_seq_len=1, layer_weights_np=layers_np,
        )

    comp = NPUCompiler(os.path.join(os.path.dirname(_HERE), "hal_config.json"))
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = comp0._dram_cursor
    x_row = np.zeros((M, hidden), dtype=np.float32)
    comp._const_regions[x_addr] = x_row.tobytes()

    tvm_backend._emit_model_kvcache_fp32(
        comp, n_layers, hidden, inter, n_heads, n_kv_heads, head_dim,
        x_addr=x_addr, layer_weights=layer_weights, scratch_a=scratch_a,
        scratch_b=scratch_b, pos=0,
    )
    return comp


def run_cache_flat_across_layers_case():
    """Same config, 1 layer vs. 4 layers: the number of DISTINCT cached
    constants must be IDENTICAL (everything here is decode-step-global,
    not layer-specific), even though the number of CALLS quadruples."""
    hidden, inter, n_heads, n_kv_heads, head_dim = 32, 48, 4, 2, 16

    comp_1 = _build_and_emit(1, hidden, inter, n_heads, n_kv_heads, head_dim)
    comp_4 = _build_and_emit(4, hidden, inter, n_heads, n_kv_heads, head_dim)

    n1, n4 = len(comp_1._const_cache), len(comp_4._const_cache)
    if n1 != n4:
        return False, f"cache entries grew with layer count: 1L={n1}, 4L={n4}", 0
    if n1 == 0:
        return False, "no cache entries at all -- memoization path never hit", 0
    return True, f"{n1} distinct cached constants, unchanged from 1 to 4 layers", n1


def run_rope_single_entry_case():
    """head_dim=32, 4 Q heads + 2 KV heads, 3 layers -> _emit_rope_fp32
    is called (4+2)*3 = 18 times this step, all with the SAME
    (head_dim, pos, theta) -- must collapse to exactly one
    "rope_cossin" entry and one "rope_perm" entry, not 18 of each."""
    hidden, inter, n_heads, n_kv_heads, head_dim = 32, 48, 4, 2, 16
    comp = _build_and_emit(3, hidden, inter, n_heads, n_kv_heads, head_dim)

    rope_cossin = [k for k in comp._const_cache if k[0] == "rope_cossin"]
    rope_perm = [k for k in comp._const_cache if k[0] == "rope_perm"]
    if len(rope_cossin) != 1:
        return False, f"expected 1 rope_cossin cache entry, got {len(rope_cossin)}", 0
    if len(rope_perm) != 1:
        return False, f"expected 1 rope_perm cache entry, got {len(rope_perm)}", 0
    return True, "", len(comp._const_cache)


def main():
    cases = [
        ("cache_flat_across_layers", run_cache_flat_across_layers_case),
        ("rope_single_entry_regardless_of_heads_and_layers", run_rope_single_entry_case),
    ]
    n_pass = 0
    for name, fn in cases:
        try:
            ok, msg, n_entries = fn()
        except Exception as e:  # noqa: BLE001
            ok, msg, n_entries = False, f"{type(e).__name__}: {e}", 0
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({n_entries} cache entries)"
              + (f"  -- {msg}" if msg else ""))
        n_pass += ok
    print(f"\n{n_pass}/{len(cases)} passed")
    return 0 if n_pass == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
