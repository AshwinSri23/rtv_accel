#!/usr/bin/env python3
"""
test_kv_cache.py — tvm_backend._emit_kv_cache_append_fp32 /
_emit_kv_cache_tail_mask_fp32: a genuine, GROWING multi-tile KV-cache,
closing out TODO 1.1's last open item.

Not the same thing as the "genuinely batched multi-row queries" work
(test_attention_batched.py): that's one 16-row TILE of independent rows
attending to each other/themselves in a single call. A KV-cache PERSISTS
and GROWS K/V state across MANY decode steps -- later steps causally
attending to every earlier one, including ones from tiles filled by
earlier calls entirely.

Two real pieces, both driven by the "constant matrix through GEMM" trick
this codebase already uses for broadcast/mask/RoPE (no hardware
broadcast/permute/partial-write primitive exists for any of them):

1. _emit_kv_cache_append_fp32 -- LOAD_TILE/STORE_TILE only move WHOLE
   16x16 tiles (confirmed against pe_array.cpp: no sub-tile partial-row
   write opcode), so "append one token" can't literally write one row of
   an existing tile. Blends the new row in via
   `mask_keep @ old_tile + mask_insert @ new_row_bcast`, two compile-time
   constant matrices selecting "every row except `row`" and "only row
   `row`" respectively.

2. _emit_kv_cache_tail_mask_fp32 -- the newest tile is usually only
   PARTIALLY filled (valid_len tokens out of a full 16-row tile); the
   unfilled tail (and any wholly-future tiles) must not contaminate the
   softmax. Reuses _emit_attention_core_fp32's EXISTING `mask_addr`
   parameter (already built for causal masking) with a per-column
   0/-1e30 mask instead of a new mechanism.

Verified two ways: append in isolation (other 15 rows of a tile must
survive an insert untouched), and a full T-decode-step simulation (T=20,
spanning 2 K/V tiles) checked against a numpy reference that does genuine
incremental single-query attention over all t+1 real cached vectors at
each step -- not a single-shot batch, an actual step-by-step growing
cache, the real point of the feature.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_kv_cache.py
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


def _run_sim(instr_stream, dram, dumps):
    with tempfile.TemporaryDirectory(prefix="rtv_kvcache_") as wd:
        p, d = os.path.join(wd, "p.bin"), os.path.join(wd, "d.img")
        open(p, "wb").write(bytes(instr_stream))
        open(d, "wb").write(bytes(dram))
        args = [SIM, p, HAL, str(max(1, len(dram) // (1024 * 1024) + 2)),
                "--dram-image", d]
        outs = []
        for i, (addr, nbytes) in enumerate(dumps):
            out = os.path.join(wd, f"o{i}.bin")
            args += ["--dump", f"{addr}:{nbytes}:{out}"]
            outs.append(out)
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"simulator failed: {r.stdout[-600:]}{r.stderr[-600:]}")
        return [open(o, "rb").read() for o in outs]


def run_append_case(row, width=16, seed=1):
    """Insert one new row into a pre-populated tile; every OTHER row must
    survive completely untouched."""
    rng = np.random.default_rng(seed)
    old_tile = rng.normal(size=(M, width)).astype(np.float32)
    new_row = rng.normal(size=(width,)).astype(np.float32)
    new_row_bcast = np.tile(new_row, (M, 1)).astype(np.float32)

    expected = old_tile.copy()
    expected[row, :] = new_row

    tile_addr, new_row_addr = 0x1000, 0x100000
    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = new_row_addr + M * width * 4 + 0x10000

    tvm_backend._emit_kv_cache_append_fp32(comp, tile_addr, new_row_addr, row, width)
    n_instr = len(comp.instruction_stream) // 16

    dram_size = max(new_row_addr + M * width * 4, comp.dram_bytes_needed) + 0x10000
    dram = bytearray(dram_size)
    dram[tile_addr:tile_addr + old_tile.nbytes] = old_tile.tobytes()
    dram[new_row_addr:new_row_addr + new_row_bcast.nbytes] = new_row_bcast.tobytes()
    for addr, data in comp._const_regions.items():
        dram[addr:addr + len(data)] = data

    (raw,) = _run_sim(comp.instruction_stream, dram, [(tile_addr, M * width * 4)])
    got = np.frombuffer(raw, dtype=np.float32).reshape(M, width)

    if np.allclose(got, expected, atol=1e-5):
        return True, "", n_instr
    d = np.abs(got - expected)
    return False, f"max delta {float(d.max()):.4g}", n_instr


def run_incremental_decode_case(head_dim, n_steps, seed=7):
    """The real point: T decode steps, each appending one new K/V token and
    attending over the WHOLE cache so far (max_tiles pre-allocated,
    ceil(n_steps/16) of them actually get written), checked step-by-step
    against a genuine incremental numpy reference (no 1/sqrt(head_dim), no
    causal asymmetry beyond "can't see the future" -- matching
    _emit_attention_core_fp32's own no-scale convention)."""
    rng = np.random.default_rng(seed)
    max_tiles = -(-n_steps // M)  # ceil
    w = DType.BYTES[DType.FP32]

    q_rows = rng.normal(size=(n_steps, head_dim)).astype(np.float32)
    k_rows = rng.normal(size=(n_steps, head_dim)).astype(np.float32)
    v_rows = rng.normal(size=(n_steps, head_dim)).astype(np.float32)

    # -- reference: genuine incremental single-query attention --
    expected_ctx = np.zeros((n_steps, head_dim), dtype=np.float64)
    for t in range(n_steps):
        K = k_rows[:t + 1].astype(np.float64)
        V = v_rows[:t + 1].astype(np.float64)
        sc = q_rows[t].astype(np.float64) @ K.T
        e = np.exp(sc - sc.max())
        p = e / e.sum()
        expected_ctx[t] = p @ V

    # -- device side: pre-allocate max_tiles worth of K/V cache tiles --
    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()

    base = [0x1000]

    def alloc(nbytes):
        a = base[0]
        base[0] += nbytes + 0x1000
        return a

    # K's tiles need not be mutually contiguous (each is transposed
    # individually, addressed one at a time -- see the per-step loop
    # below), but V's tiles DO: _emit_attention_core_fp32 takes ONE
    # v_addr and treats it as a single tightly-packed (seq_k, head_dim)
    # block, so all of V's active tiles must sit back-to-back in DRAM.
    k_cache = [alloc(M * head_dim * 4) for _ in range(max_tiles)]
    v_cache_base = alloc(max_tiles * M * head_dim * 4)
    v_cache = [v_cache_base + i * M * head_dim * 4 for i in range(max_tiles)]
    # Two INDEPENDENT bump allocators are in play here: this test's own
    # `alloc()` (k_t_addr/step_ctx, growing throughout the loop below) and
    # compiler.alloc_staging() (every append/transpose/mask/attention call
    # inside the loop allocates several staging buffers of its own, also
    # growing throughout the loop). Syncing them only once, right here,
    # with a SMALL gap silently invites the two to collide once enough
    # iterations have made both grow past that gap -- confirmed directly:
    # the first version of this test used a 0x10000 gap and started
    # corrupting data at step 9. A large, fixed separation (64 MiB) that
    # neither allocator can plausibly reach in one test run sidesteps the
    # problem instead of tuning a gap that would just break again at a
    # different step count.
    comp._dram_cursor = base[0] + 0x400_0000

    # zero-init cache DRAM content up front (built alongside the image below);
    # here we only need the addresses to emit instructions against.
    got_ctx_per_step = []
    n_instr_total = 0

    for t in range(n_steps):
        tile_idx, row = divmod(t, M)
        n_active_tiles = tile_idx + 1
        # Total real tokens seen so far ACROSS ALL active tiles, not just
        # this one's position within the LATEST tile -- `row + 1` was a
        # real bug here: it only happens to equal the true count while
        # tile_idx==0 (the very first tile), and silently masked out
        # every earlier tile's genuinely valid data (e.g. t=16: row=0,
        # row+1=1, but 17 real tokens actually exist by then) the moment
        # a second tile came into play.
        valid_len = t + 1

        # host-side broadcast of this step's new K/V row to all 16 rows,
        # then append into the correct cache tile/row.
        # (constants injected directly -- no on-device broadcast needed
        # since the source is already a fresh host-computed row.)
        #
        # FRESH address every step, not new_k_bcast/new_v_bcast reused:
        # _const_regions is STATIC initial DRAM content, baked in ONCE
        # before the simulator starts, not something that changes mid-run.
        # All n_steps' worth of instructions execute in ONE simulator
        # invocation against ONE static image -- reusing the same address
        # for step t's and step t+1's "new row" would mean only the LAST
        # write to that dict key survives into the actual image, so every
        # step's LOAD_TILE would read the SAME (final) row. Caught by
        # comparing every step's own delta, not just trusting one
        # allclose() over the whole array (every step was wrong, not just
        # a "step 9" cliff np.allclose's argmax happened to name).
        step_k_bcast = alloc(M * head_dim * 4)
        step_v_bcast = alloc(M * head_dim * 4)
        comp._const_regions[step_k_bcast] = np.tile(k_rows[t], (M, 1)).astype(np.float32).tobytes()
        comp._const_regions[step_v_bcast] = np.tile(v_rows[t], (M, 1)).astype(np.float32).tobytes()
        tvm_backend._emit_kv_cache_append_fp32(comp, k_cache[tile_idx], step_k_bcast, row, head_dim)
        tvm_backend._emit_kv_cache_append_fp32(comp, v_cache[tile_idx], step_v_bcast, row, head_dim)

        # de-interleave the active K tiles into one contiguous (n_active*16,
        # head_dim) block, then transpose it whole for matmul(q,k)'s literal
        # semantics -- reuse the same per-tile transpose loop attention's
        # block-level driver already uses for per-head K.
        # k_t_addr is ONE (head_dim, seq_k) row-major buffer -- tile ti's
        # transpose must land in COLUMNS [ti*M:(ti+1)*M] of every one of
        # its head_dim ROWS, not in a separate 1024-byte block; out_stride
        # (row pitch = seq_k, not M) is what makes that column placement
        # correct instead of overlapping every tile at the same offset
        # within its own tightly-packed sub-block -- see
        # _emit_transpose_fp32's own docstring for the bug this caught.
        seq_k = n_active_tiles * M
        k_t_addr = alloc(head_dim * seq_k * 4)
        for ti in range(n_active_tiles):
            tvm_backend._emit_transpose_fp32(comp, in_addr=k_cache[ti],
                                             out_addr=k_t_addr + ti * M * w,
                                             out_stride=seq_k)

        step_q_addr = alloc(M * head_dim * 4)   # fresh per step -- same reasoning as above
        comp._const_regions[step_q_addr] = np.tile(q_rows[t], (M, 1)).astype(np.float32).tobytes()

        mask_addr = tvm_backend._emit_kv_cache_tail_mask_fp32(comp, seq_k, valid_len)
        # V's active tiles are already one contiguous (seq_k, head_dim)
        # block by construction (v_cache_base) -- exactly what
        # _emit_attention_core_fp32 wants, no de-interleaving needed
        # (only K needed the transpose).
        step_ctx = alloc(M * head_dim * 4)
        tvm_backend._emit_attention_core_fp32(
            comp, seq_k=seq_k, head_dim=head_dim,
            q_addr=step_q_addr, k_addr=k_t_addr, v_addr=v_cache[0],
            out_addr=step_ctx, mask_addr=mask_addr,
        )
        got_ctx_per_step.append(step_ctx)

    n_instr_total = len(comp.instruction_stream) // 16

    dram_size = max(comp.dram_bytes_needed, base[0]) + 0x20000
    dram = bytearray(dram_size)
    for tile in k_cache + v_cache:
        dram[tile:tile + M * head_dim * 4] = np.zeros((M, head_dim), dtype=np.float32).tobytes()
    for addr, data in comp._const_regions.items():
        dram[addr:addr + len(data)] = data

    dumps = [(addr, M * head_dim * 4) for addr in got_ctx_per_step]
    raws = _run_sim(comp.instruction_stream, dram, dumps)
    got = np.stack([np.frombuffer(r, dtype=np.float32).reshape(M, head_dim)[0]
                    for r in raws])

    if np.allclose(got, expected_ctx, atol=1e-3):
        return True, "", n_instr_total
    d = np.abs(got - expected_ctx)
    bad = int(np.unravel_index(np.argmax(d), d.shape)[0])
    return False, f"max delta {float(d.max()):.4g} at step {bad}", n_instr_total


def main():
    cases = [
        ("kv_append_row0", lambda: run_append_case(0)),
        ("kv_append_row7", lambda: run_append_case(7, seed=3)),
        ("kv_append_row15", lambda: run_append_case(15, seed=5)),
        ("kv_incremental_decode_20steps", lambda: run_incremental_decode_case(16, 20)),
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
