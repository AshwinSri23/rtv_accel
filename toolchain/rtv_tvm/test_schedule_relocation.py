#!/usr/bin/env python3
"""
test_schedule_relocation.py — tvm_backend._emit_model_kvcache_fp32_relocated:
compiles only layer 0 for real through TVM's schedule+codegen and
RELOCATES that result (a cheap address patch on already-generated
instruction bytes) for every other layer instead of recompiling it,
since every layer at a given config is structurally identical and only
differs in which DRAM addresses it reads/writes.

Two independent checks, not one, because "the relocated path produces
the same final number" alone wouldn't catch a relocation bug that
happens to cancel out numerically for a specific random seed:

1. byte_identical_to_recompiled: run the SAME model/weights/pos through
   both _emit_model_kvcache_fp32 (the original, fully-recompiled path)
   and _emit_model_kvcache_fp32_relocated on two otherwise-identical
   fresh compilers, and assert the two instruction streams are
   BYTE-FOR-BYTE IDENTICAL, not just "produce the same output" -- the
   relocated path is supposed to reconstruct the exact same program a
   full recompile would have produced, just without paying for it.
2. real_sim_matches_numpy_reference: run the relocated path through
   rtv_sim against a genuine numpy reference (same shape as
   test_decode_loop.py's), so this is also checked independently of
   the original path, not just self-consistently with it.

Also reports the actual compile-time win: TVM schedule/codegen time
for a full recompile of every layer vs. compiling layer 0 once and
relocating the rest.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_schedule_relocation.py
"""

import os
import subprocess
import sys
import tempfile
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from compiler_state import NPUCompiler, DType  # noqa: E402
import tvm_backend                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")

M = 16


def _rmsnorm_ref(v, gamma):
    return v / np.sqrt((v.astype(np.float64) ** 2).mean() + 1e-6) * gamma.astype(np.float64)


def _rope_ref(row, pos, head_dim, theta=1e6):
    half = head_dim // 2
    inv_freq = theta ** (-np.arange(0, half, dtype=np.float64) * 2.0 / head_dim)
    ang = pos * inv_freq
    cos_full = np.concatenate([np.cos(ang), np.cos(ang)])
    x1, x2 = row[:half], row[half:]
    rotate_half = np.concatenate([-x2, x1])
    sin_full = np.concatenate([np.sin(ang), np.sin(ang)])
    return row * cos_full + rotate_half * sin_full


def _make_model(hidden, inter, n_heads, n_kv_heads, head_dim, n_layers, seed):
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
    return layers_np, rng


def run_byte_identical_case(hidden, inter, n_heads, n_kv_heads, head_dim, n_layers,
                            pos=3, seed=11):
    layers_np, rng = _make_model(hidden, inter, n_heads, n_kv_heads, head_dim,
                                 n_layers, seed)
    token_row = rng.normal(size=(hidden,)).astype(np.float32) * 0.3

    def _build_and_emit(relocated):
        comp0 = NPUCompiler(HAL)
        comp0.dtype = DType.FP32
        dram, x_addr, layer_weights, scratch_a, scratch_b, max_tiles = \
            tvm_backend.build_kvcache_model_dram_image(
                comp0, hidden, n_layers, n_kv_heads, head_dim,
                max_seq_len=pos + 1, layer_weights_np=layers_np,
            )
        comp = NPUCompiler(HAL)
        comp.dtype = DType.FP32
        comp.instruction_stream = bytearray()
        comp._dram_cursor = comp0._dram_cursor
        x_row = np.zeros((M, hidden), dtype=np.float32)
        x_row[0, :] = token_row
        comp._const_regions[x_addr] = x_row.tobytes()

        fn = (tvm_backend._emit_model_kvcache_fp32_relocated if relocated
             else tvm_backend._emit_model_kvcache_fp32)
        final_addr = fn(comp, n_layers, hidden, inter, n_heads, n_kv_heads, head_dim,
                        x_addr=x_addr, layer_weights=layer_weights,
                        scratch_a=scratch_a, scratch_b=scratch_b, pos=pos)
        return bytes(comp.instruction_stream), final_addr

    t0 = time.time()
    stream_full, addr_full = _build_and_emit(relocated=False)
    t1 = time.time()
    stream_reloc, addr_reloc = _build_and_emit(relocated=True)
    t2 = time.time()

    if addr_full != addr_reloc:
        return False, f"final_addr mismatch: {addr_full} vs {addr_reloc}", 0, (t1 - t0, t2 - t1)
    if stream_full != stream_reloc:
        n = min(len(stream_full), len(stream_reloc))
        first_diff = next((i for i in range(0, n, 16)
                           if stream_full[i:i + 16] != stream_reloc[i:i + 16]), n)
        return (False,
               f"instruction streams differ (len {len(stream_full)} vs "
               f"{len(stream_reloc)}, first differing word at byte {first_diff})",
               0, (t1 - t0, t2 - t1))
    return True, "", len(stream_full) // 16, (t1 - t0, t2 - t1)


def run_real_sim_case(hidden, inter, n_heads, n_kv_heads, head_dim, n_layers,
                      pos=0, seed=23):
    layers_np, rng = _make_model(hidden, inter, n_heads, n_kv_heads, head_dim,
                                 n_layers, seed)
    token_row = rng.normal(size=(hidden,)).astype(np.float32) * 0.3
    rep = n_heads // n_kv_heads

    # Single-step numpy reference matching _emit_model_kvcache_fp32's own
    # documented degenerate case: self-attention over one real key (this
    # step's own), the same simplification every non-checkpoint test here
    # uses -- this test's job is to prove RELOCATION preserves that
    # behavior exactly, not to re-litigate multi-token attention.
    # Requires pos=0: at any pos>0 on a freshly zero-initialized cache,
    # the tail mask would mark positions 0..pos-1 "valid" too, but those
    # cache rows were never actually written by a real prior decode step
    # (this test runs exactly one step) -- they're zero-vectors, not a
    # second real key, so "one real key gets 100% weight" only holds at
    # pos=0 (valid_len=1). Caught directly: pos=2 failed against BOTH
    # the relocated path AND the original _emit_model_kvcache_fp32,
    # proving the bug was in this reference, not in relocation.
    cur = token_row.astype(np.float64)
    for lw in layers_np:
        norm1 = _rmsnorm_ref(cur, lw["gamma1"])
        q = (norm1 @ lw["wq"]).reshape(n_heads, head_dim)
        k = (norm1 @ lw["wk"]).reshape(n_kv_heads, head_dim)
        v = (norm1 @ lw["wv"]).reshape(n_kv_heads, head_dim)
        q = np.stack([_rope_ref(q[h], pos, head_dim) for h in range(n_heads)])
        k = np.stack([_rope_ref(k[h], pos, head_dim) for h in range(n_kv_heads)])
        ctx = np.stack([v[qh // rep] for qh in range(n_heads)])
        attn = ctx.reshape(n_heads * head_dim) @ lw["wo"]
        resid1 = cur + attn
        norm2 = _rmsnorm_ref(resid1, lw["gamma2"])
        g = norm2 @ lw["wgate"]
        u = norm2 @ lw["wup"]
        mlp = ((g / (1.0 + np.exp(-g))) * u) @ lw["wdown"]
        cur = resid1 + mlp
    expected = cur.astype(np.float32)

    comp0 = NPUCompiler(HAL)
    comp0.dtype = DType.FP32
    dram, x_addr, layer_weights, scratch_a, scratch_b, max_tiles = \
        tvm_backend.build_kvcache_model_dram_image(
            comp0, hidden, n_layers, n_kv_heads, head_dim,
            max_seq_len=pos + 1, layer_weights_np=layers_np,
        )
    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = comp0._dram_cursor
    x_row = np.zeros((M, hidden), dtype=np.float32)
    x_row[0, :] = token_row
    comp._const_regions[x_addr] = x_row.tobytes()

    final_addr = tvm_backend._emit_model_kvcache_fp32_relocated(
        comp, n_layers, hidden, inter, n_heads, n_kv_heads, head_dim,
        x_addr=x_addr, layer_weights=layer_weights, scratch_a=scratch_a,
        scratch_b=scratch_b, pos=pos,
    )
    n_instr = len(comp.instruction_stream) // 16

    needed = max(comp.dram_bytes_needed, len(dram))
    if needed > len(dram):
        dram = dram + bytearray(needed - len(dram) + 0x10000)
    for addr, data in comp._const_regions.items():
        dram[addr:addr + len(data)] = data

    with tempfile.TemporaryDirectory(prefix="rtv_reloc_") as wd:
        p, d, o = (os.path.join(wd, n) for n in ("p.bin", "d.img", "o.bin"))
        open(p, "wb").write(bytes(comp.instruction_stream))
        open(d, "wb").write(bytes(dram))
        mb = max(1, len(dram) // (1024 * 1024) + 2)
        r = subprocess.run([SIM, p, HAL, str(mb), "--dram-image", d,
                            "--dump", f"{final_addr}:{M * hidden * 4}:{o}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"simulator failed: {r.stdout[-500:]}{r.stderr[-500:]}", n_instr
        got = np.frombuffer(open(o, "rb").read(), dtype=np.float32).reshape(M, hidden)[0]

    if np.allclose(got, expected, atol=3e-2):
        return True, "", n_instr
    d = np.abs(got - expected)
    return False, f"max diff {float(d.max()):.4g}", n_instr


def main():
    cases_byte = [
        # n_layers must be >=3 for these to actually exercise relocation:
        # layers 0 and 1 are ALWAYS compiled for real (see
        # _emit_model_kvcache_fp32_relocated's docstring on why layer 1,
        # not layer 0, is the template) -- only layer 2 onward is ever
        # relocated.
        ("byte_identical_4layer_gqa", lambda: run_byte_identical_case(
            32, 48, 4, 2, 16, n_layers=4)),
        ("byte_identical_6layer_headdim32", lambda: run_byte_identical_case(
            48, 64, 2, 1, 32, n_layers=6)),
    ]
    cases_sim = [
        ("real_sim_4layer_gqa", lambda: run_real_sim_case(32, 48, 4, 2, 16, n_layers=4)),
    ]
    n_pass, n_total = 0, 0
    total_full, total_reloc = 0.0, 0.0
    for name, fn in cases_byte:
        n_total += 1
        try:
            ok, msg, n_instr, (t_full, t_reloc) = fn()
            total_full += t_full
            total_reloc += t_reloc
        except Exception as e:  # noqa: BLE001
            ok, msg, n_instr = False, f"{type(e).__name__}: {e}", 0
            t_full = t_reloc = 0.0
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({n_instr} instrs)"
              f"  full={t_full:.2f}s relocated={t_reloc:.2f}s"
              + (f"  -- {msg}" if msg else ""))
        n_pass += ok
    for name, fn in cases_sim:
        n_total += 1
        try:
            ok, msg, n_instr = fn()
        except Exception as e:  # noqa: BLE001
            ok, msg, n_instr = False, f"{type(e).__name__}: {e}", 0
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({n_instr} instrs)"
              + (f"  -- {msg}" if msg else ""))
        n_pass += ok
    if total_full > 0:
        print(f"\ncompile time (byte-identical cases only): "
             f"full recompile {total_full:.2f}s vs relocated {total_reloc:.2f}s "
             f"({total_full / max(total_reloc, 1e-9):.1f}x)")
    print(f"\n{n_pass}/{n_total} passed")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
