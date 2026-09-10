#!/usr/bin/env python3
"""
test_gqa_attention.py — multi-head attention with GQA, driven directly
through tvm_backend._emit_gqa_attention_fp32 rather than pattern-matched
dispatch.

Unlike RMSNorm/SwiGLU/attention-core in test_composite_dispatch.py, this
isn't something patterns.py discovers from graph structure -- see
_emit_gqa_attention_fp32's own docstring on why: a caller that already
knows its head/GQA config (hidden_size/num_heads/num_kv_heads, the same way
compile_attention_block does) drives this directly, informed by the model
config rather than DFPattern inference. So there is no Relax module here to
partition; this calls tvm_backend._emit_gqa_attention_fp32 directly and
checks the result against a numpy reference matching e2e_test.py's
Fp32AttentionCase's per-head loop -- minus the x@Wq/Wk/Wv/Wo projections
(this dispatch starts from already-projected Q/K/V, per the granularity
decision documented in emit_composite) and minus 1/sqrt(head_dim) (the
attention-core pattern this builds on has no scale step).

Q is one real query row, replicated across all 16 tile rows (same
convention _emit_attention_core_fp32 uses, for the same REDUCE-collapses-
a-whole-tile reason). K and V are laid out as n_kv_heads separate,
CONTIGUOUS per-head blocks -- K already transposed to (head_dim, seq_k)
per head (matching matmul(q, k)'s literal semantics, no transpose inside
this composite), V as-is (seq_k, head_dim) -- not column-sliced from one
wide buffer the way Q/out are. See _emit_gqa_attention_fp32's docstring
for why those are different layouts, not an inconsistency.

Found and fixed a real bug while building this: codegen.py's STORE_TILE
path (_tile_copy's `else` branch) had its OWN separate pitch calculation
that, unlike the LOAD_TILE path, was never fixed to respect an explicit
match_buffer stride -- so each head's output write used head_dim as the
row pitch instead of n_heads*head_dim, silently landing row 1 of head h's
write inside row 0's own span rather than n_heads*head_dim further along.
Every row past 0 was corrupted by the next head's store. Caught because
this is the first caller to write a STRIDED buffer as an output rather
than only reading one as an input.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_gqa_attention.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from compiler_state import NPUCompiler, DType  # noqa: E402
import tvm_backend                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")


def run_gqa_case(n_heads, n_kv_heads, head_dim, seq_k=16, seed=7):
    q_dim = n_heads * head_dim
    rep = n_heads // n_kv_heads

    rng = np.random.default_rng(seed)
    q_row = rng.uniform(-1, 1, size=(q_dim,)).astype(np.float32)
    q = np.tile(q_row, (16, 1)).astype(np.float32)
    k_heads = [rng.uniform(-1, 1, size=(head_dim, seq_k)).astype(np.float32)
               for _ in range(n_kv_heads)]
    v_heads = [rng.uniform(-1, 1, size=(seq_k, head_dim)).astype(np.float32)
               for _ in range(n_kv_heads)]
    k_blob = np.concatenate(k_heads, axis=0)
    v_blob = np.concatenate(v_heads, axis=0)

    expected_row = np.zeros(q_dim, dtype=np.float64)
    for qh in range(n_heads):
        kv_h = qh // rep
        qh_slice = q_row[qh * head_dim:(qh + 1) * head_dim].astype(np.float64)
        k_block = k_heads[kv_h].astype(np.float64)
        v_block = v_heads[kv_h].astype(np.float64)
        sc = qh_slice @ k_block
        e = np.exp(sc - sc.max())
        p = e / e.sum()
        expected_row[qh * head_dim:(qh + 1) * head_dim] = p @ v_block
    expected = np.tile(expected_row, (16, 1)).astype(np.float32)

    q_ad, k_ad, v_ad, out_ad = 0x1000, 0x40000, 0x80000, 0x100000
    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = out_ad + 16 * q_dim * 4 + 0x10000

    tvm_backend._emit_gqa_attention_fp32(
        comp, n_heads, n_kv_heads, seq_k, head_dim,
        q_addr=q_ad, k_addr=k_ad, v_addr=v_ad, out_addr=out_ad,
    )
    n_instr = len(comp.instruction_stream) // 16

    dram_size = max(out_ad + 16 * q_dim * 4, comp.dram_bytes_needed) + 0x10000
    dram = bytearray(dram_size)
    dram[q_ad:q_ad + q.nbytes] = q.tobytes()
    dram[k_ad:k_ad + k_blob.nbytes] = k_blob.tobytes()
    dram[v_ad:v_ad + v_blob.nbytes] = v_blob.tobytes()
    for addr, data in comp._const_regions.items():
        dram[addr:addr + len(data)] = data

    with tempfile.TemporaryDirectory(prefix="rtv_gqa_") as wd:
        p, d = os.path.join(wd, "p.bin"), os.path.join(wd, "d.img")
        open(p, "wb").write(bytes(comp.instruction_stream))
        open(d, "wb").write(bytes(dram))
        out = os.path.join(wd, "o.bin")
        mb = max(1, len(dram) // (1024 * 1024) + 2)
        r = subprocess.run(
            [SIM, p, HAL, str(mb), "--dram-image", d,
             "--dump", f"{out_ad}:{16 * q_dim * 4}:{out}"],
            capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"simulator failed: {r.stdout[-400:]}{r.stderr[-400:]}", 0
        got = np.frombuffer(open(out, "rb").read(), dtype=np.float32).reshape(16, q_dim)

    if np.allclose(got, expected, atol=1e-4):
        return True, "", n_instr
    d = np.abs(got - expected)
    per_row = d.max(axis=1)
    return False, (f"{int((d > 1e-4).sum())}/{d.size} wrong, max delta "
                   f"{float(d.max()):.4g}, worst row {int(per_row.argmax())}"), n_instr


def main():
    cases = [
        # e2e_test.py's Fp32AttentionCase instances use head_dim=32 for its
        # single-head case; that exceeds _emit_attention_core_fp32's
        # single-K/V-tile scope (seq_k == head_dim == 16), a limit from
        # before this multi-head work and unrelated to it, so this uses
        # head_dim=16 for the degenerate 1-head/1-kv-head sanity check
        # instead (hidden itself is irrelevant here regardless, since
        # projections aren't part of this dispatch).
        ("gqa_1head_1kv_16hd", lambda: run_gqa_case(1, 1, 16)),
        ("gqa_4head_2kv_16hd", lambda: run_gqa_case(4, 2, 16)),
        ("gqa_8head_2kv_16hd", lambda: run_gqa_case(8, 2, 16, seed=3)),
    ]
    n_pass = 0
    for name, fn in cases:
        try:
            ok, msg, n_instr = fn()
        except Exception as e:  # noqa: BLE001
            ok, msg, n_instr = False, f"{type(e).__name__}: {e}", 0
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({n_instr} instrs)"
              + (f"  -- {msg}" if msg else ""))
        n_pass += ok
    print(f"\n{n_pass}/{len(cases)} passed")
    return 0 if n_pass == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
