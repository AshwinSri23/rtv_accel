#!/usr/bin/env python3
"""
test_rope_attention.py — RoPE (tvm_backend._emit_rope_fp32) wired into
_emit_attention_block_fp32/_emit_transformer_layer_fp32/_emit_model_fp32
via their new `pos=` parameter, closing out TODO 1.1's RoPE sub-item.

Two cases, deliberately NOT the same kind of check:

rope_perhead_wiring: _emit_attention_block_fp32 applies RoPE to Q and K
via a per-head loop (n_heads iterations for Q at stride q_dim, n_kv_heads
for K at stride kv_dim) BEFORE broadcasting row0 across the tile -- see
_emit_attention_block_fp32's docstring. This reproduces that exact loop
directly against a multi-head buffer with DIFFERENT data per head and
checks each head's rotated slice against an independent numpy RoPE
reference. This is the test that actually exercises the new wiring
(per-head address/stride arithmetic); see the other case for why a
full-layer check can't do that here.

layer_with_rope_matches_norope: runs a whole transformer layer through
_emit_transformer_layer_fp32 with pos=7 and confirms the output is
IDENTICAL to pos=None. This looks like it's testing nothing, but it's
verifying a real structural property, not a null test: this whole path's
self-attention has exactly one DISTINCT key (replicated 16x across the
tile -- see _emit_attention_block_fp32's docstring), so every row's
attention score is identical and softmax is uniform regardless of what Q
and K actually contain. Applying the SAME rotation angle to Q and K (they
share pos, since it's self-attention on one token) is an orthogonal
transform that cancels exactly in the q.k dot product feeding that
softmax (R(t)q . R(t)k = q.(R(t)^T R(t)).k = q.k), so RoPE is
mathematically a no-op for THIS architecture's attention core, and will
stay one until a genuine multi-token KV-cache gives distinct keys at
distinct positions. If this ever starts producing a different number,
either that invariant broke (bug) or a real KV-cache landed and this
test's premise needs revisiting.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_rope_attention.py
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

M = 16


def _run_sim(instr_stream, dram, addr, nbytes, shape):
    with tempfile.TemporaryDirectory(prefix="rtv_rope_") as wd:
        p, d = os.path.join(wd, "p.bin"), os.path.join(wd, "d.img")
        open(p, "wb").write(bytes(instr_stream))
        open(d, "wb").write(bytes(dram))
        out = os.path.join(wd, "o.bin")
        mb = max(1, len(dram) // (1024 * 1024) + 2)
        r = subprocess.run([SIM, p, HAL, str(mb), "--dram-image", d,
                            "--dump", f"{addr}:{nbytes}:{out}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"simulator failed: {r.stdout[-500:]}{r.stderr[-500:]}")
        return np.frombuffer(open(out, "rb").read(), dtype=np.float32).reshape(shape)


def _rope_ref(row, pos, head_dim, theta):
    half = head_dim // 2
    inv_freq = theta ** (-np.arange(0, half, dtype=np.float64) * 2.0 / head_dim)
    ang = pos * inv_freq
    cos_full = np.concatenate([np.cos(ang), np.cos(ang)])
    x1, x2 = row[:half].astype(np.float64), row[half:].astype(np.float64)
    rotate_half = np.concatenate([-x2, x1])
    sin_full = np.concatenate([np.sin(ang), np.sin(ang)])
    return (row.astype(np.float64) * cos_full + rotate_half * sin_full).astype(np.float32)


def run_perhead_wiring_case(n_heads, head_dim, pos, theta=1e6, seed=5):
    """Reproduces _emit_attention_block_fp32's per-head RoPE loop directly
    on a q_stage-shaped (M, n_heads*head_dim) buffer with distinct data per
    head, catching addressing/stride bugs the full-layer test can't (see
    module docstring)."""
    rng = np.random.default_rng(seed)
    q_dim = n_heads * head_dim
    q_row = rng.normal(size=(q_dim,)).astype(np.float32)

    expected_row = np.concatenate([
        _rope_ref(q_row[h * head_dim:(h + 1) * head_dim], pos, head_dim, theta)
        for h in range(n_heads)
    ])
    expected = np.zeros((M, q_dim), dtype=np.float32)
    expected[0, :] = expected_row  # row0-real-rest-zero: rows 1-15 stay 0

    q = np.zeros((M, q_dim), dtype=np.float32)
    q[0, :] = q_row

    q_addr, out_addr = 0x1000, 0x100000
    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = out_addr + M * q_dim * 4 + 0x10000

    w = DType.BYTES[DType.FP32]
    # exactly the loop _emit_attention_block_fp32 runs over q_stage
    for h in range(n_heads):
        off = h * head_dim * w
        tvm_backend._emit_rope_fp32(comp, head_dim, pos, q_addr + off, out_addr + off,
                                    theta=theta, in_stride=q_dim)
    n_instr = len(comp.instruction_stream) // 16

    dram_size = max(out_addr + M * q_dim * 4, comp.dram_bytes_needed) + 0x10000
    dram = bytearray(dram_size)
    dram[q_addr:q_addr + q.nbytes] = q.tobytes()
    for addr, data in comp._const_regions.items():
        dram[addr:addr + len(data)] = data

    got = _run_sim(comp.instruction_stream, dram, out_addr, M * q_dim * 4, (M, q_dim))

    if np.allclose(got, expected, atol=1e-4):
        return True, "", n_instr
    d = np.abs(got - expected)
    return False, f"max delta {float(d.max()):.4g}", n_instr


# --- layer_with_rope_matches_norope: reuse test_transformer_layer's setup ---
sys.path.insert(0, _HERE)
import test_transformer_layer as tl  # noqa: E402


def run_layer_rope_invariance_case(hidden, inter, n_heads, n_kv_heads, head_dim,
                                   pos=7, seed=11):
    rng = np.random.default_rng(seed)
    x_row = rng.normal(size=(hidden,)).astype(np.float32) * 0.5
    lw = tl._random_layer_weights(rng, hidden, inter, n_heads, n_kv_heads, head_dim)

    def build_and_run(pos_val):
        x = np.zeros((M, hidden), dtype=np.float32)
        x[0, :] = x_row
        addrs = {}
        base = [0x1000]

        def place(name, arr):
            addrs[name] = base[0]
            base[0] += arr.nbytes + 0x1000
            return addrs[name]

        place("x", x)
        place("gamma1", np.tile(lw["gamma1"], (M, 1)))
        place("gamma2", np.tile(lw["gamma2"], (M, 1)))
        for k in ("wq", "wk", "wv", "wo", "wgate", "wup", "wdown"):
            place(k, lw[k])
        out_addr = base[0]
        base[0] += M * hidden * 4 + 0x1000

        comp = NPUCompiler(HAL)
        comp.dtype = DType.FP32
        comp.instruction_stream = bytearray()
        comp._dram_cursor = base[0] + 0x10000

        tvm_backend._emit_transformer_layer_fp32(
            comp, hidden, inter, n_heads, n_kv_heads, head_dim,
            x_addr=addrs["x"], gamma1_addr=addrs["gamma1"], gamma2_addr=addrs["gamma2"],
            wq_addr=addrs["wq"], wk_addr=addrs["wk"], wv_addr=addrs["wv"], wo_addr=addrs["wo"],
            wgate_addr=addrs["wgate"], wup_addr=addrs["wup"], wdown_addr=addrs["wdown"],
            out_addr=out_addr, pos=pos_val,
        )
        dram_size = max(base[0], comp.dram_bytes_needed) + 0x10000
        dram = bytearray(dram_size)
        dram[addrs["x"]:addrs["x"] + x.nbytes] = x.tobytes()
        for name, arr in (("gamma1", np.tile(lw["gamma1"], (M, 1))),
                          ("gamma2", np.tile(lw["gamma2"], (M, 1))),
                          ("wq", lw["wq"]), ("wk", lw["wk"]), ("wv", lw["wv"]),
                          ("wo", lw["wo"]), ("wgate", lw["wgate"]),
                          ("wup", lw["wup"]), ("wdown", lw["wdown"])):
            a = addrs[name]
            dram[a:a + arr.nbytes] = arr.astype(np.float32).tobytes()
        for addr, data in comp._const_regions.items():
            dram[addr:addr + len(data)] = data
        got = _run_sim(comp.instruction_stream, dram, out_addr, M * hidden * 4, (M, hidden))
        return got, len(comp.instruction_stream) // 16

    got_norope, n_instr_norope = build_and_run(None)
    got_rope, n_instr_rope = build_and_run(pos)

    if n_instr_rope <= n_instr_norope:
        return False, f"rope run emitted no extra instrs ({n_instr_rope} <= {n_instr_norope})", n_instr_rope
    if np.allclose(got_norope, got_rope, atol=1e-4):
        return True, "", n_instr_rope
    d = np.abs(got_norope - got_rope)
    return False, f"rope changed output, max delta {float(d.max()):.4g} (breaks the single-key invariance -- see module docstring)", n_instr_rope


def main():
    cases = [
        ("rope_wiring_2head_hd16", lambda: run_perhead_wiring_case(2, 16, pos=5)),
        ("rope_wiring_4head_hd16", lambda: run_perhead_wiring_case(4, 16, pos=12, seed=7)),
        ("rope_wiring_pos0", lambda: run_perhead_wiring_case(2, 16, pos=0, seed=3)),
        ("layer_rope_invariant_gqa",
         lambda: run_layer_rope_invariance_case(32, 48, 4, 2, 16, pos=7)),
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
