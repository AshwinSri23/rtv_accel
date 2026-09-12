#!/usr/bin/env python3
"""
test_prefill_batching.py — tvm_backend._emit_model_kvcache_batched_fp32:
a genuine batch of up to compiler.tile_dim (16) real, distinct prompt
tokens processed in ONE call (real per-row RMSNorm, real per-row RoPE at
each token's true position, real causal self-attention within the
batch), writing all of them into the KV-cache in one shot, instead of
test_decode_loop.py's existing one-token-at-a-time path.

Two things checked, not one, because "the batch's own output matches a
reference" alone wouldn't catch a KV-cache-write bug that only shows up
later:

1. batched_prefill_matches_reference: the batch's own per-row outputs
   (rows 0..n_real-1) match a genuine causal-attention numpy reference,
   independently derived (real per-row RMSNorm/RoPE, causal softmax over
   only the real, already-seen tokens).
2. prefill_then_continue_matches_pure_one_at_a_time: the actual point of
   this feature. Run n_real tokens through ONE batched prefill call,
   then continue with the EXISTING one-token-at-a-time
   _emit_model_kvcache_fp32 loop (test_decode_loop.py's own path,
   unmodified) for several more tokens -- and compare EVERY step's
   output against a reference that processes the WHOLE sequence purely
   one token at a time from the start. If the batch's KV-cache write
   were wrong in any way invisible to check #1 (wrong tile, wrong
   layout, a stale padding row), this is what would catch it: the
   one-at-a-time continuation reads directly from what the batch wrote.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_prefill_batching.py
"""

import os
import subprocess
import sys
import tempfile

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


def _run_sim(instr_stream, dram_path_in, dram_path_out, extra_dumps=()):
    with tempfile.TemporaryDirectory(prefix="rtv_prefill_") as wd:
        p = os.path.join(wd, "p.bin")
        open(p, "wb").write(bytes(instr_stream))
        dram_size = os.path.getsize(dram_path_in)
        mb = max(1, dram_size // (1024 * 1024) + 2)
        args = [SIM, p, HAL, str(mb), "--dram-image", dram_path_in,
                "--dump", f"0:{dram_size}:{dram_path_out}"]
        extra_paths = []
        for i, (addr, nbytes, _) in enumerate(extra_dumps):
            ep = os.path.join(wd, f"extra{i}.bin")
            args += ["--dump", f"{addr}:{nbytes}:{ep}"]
            extra_paths.append(ep)
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"simulator failed: {r.stdout[-800:]}{r.stderr[-800:]}")
        return [open(ep, "rb").read() for ep in extra_paths]


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


def _reference_all_steps(layers_np, token_rows, n_heads, n_kv_heads, head_dim):
    """Genuine incremental numpy reference over the WHOLE sequence, one
    token at a time, causal, real growing KV-cache -- same shape as
    test_decode_loop.py's own reference. This is the ground truth both
    device-side paths (pure one-at-a-time, and batched-prefill-then-
    continue) are checked against."""
    rep = n_heads // n_kv_heads
    n_layers = len(layers_np)
    k_cache_ref = [[[] for _ in range(n_kv_heads)] for _ in range(n_layers)]
    v_cache_ref = [[[] for _ in range(n_kv_heads)] for _ in range(n_layers)]
    expected_outputs = []
    for t, token_row in enumerate(token_rows):
        cur = token_row.astype(np.float64)
        for li, lw in enumerate(layers_np):
            norm1 = _rmsnorm_ref(cur, lw["gamma1"])
            q = (norm1 @ lw["wq"]).reshape(n_heads, head_dim)
            k_new = (norm1 @ lw["wk"]).reshape(n_kv_heads, head_dim)
            v_new = (norm1 @ lw["wv"]).reshape(n_kv_heads, head_dim)
            q = np.stack([_rope_ref(q[h], t, head_dim) for h in range(n_heads)])
            k_new = np.stack([_rope_ref(k_new[h], t, head_dim) for h in range(n_kv_heads)])
            for h in range(n_kv_heads):
                k_cache_ref[li][h].append(k_new[h])
                v_cache_ref[li][h].append(v_new[h])
            ctx = np.zeros((n_heads, head_dim), dtype=np.float64)
            for qh in range(n_heads):
                kv_h = qh // rep
                K = np.stack(k_cache_ref[li][kv_h])
                V = np.stack(v_cache_ref[li][kv_h])
                sc = q[qh] @ K.T
                e = np.exp(sc - sc.max())
                p = e / e.sum()
                ctx[qh] = p @ V
            attn = ctx.reshape(n_heads * head_dim) @ lw["wo"]
            resid1 = cur + attn
            norm2 = _rmsnorm_ref(resid1, lw["gamma2"])
            g = norm2 @ lw["wgate"]
            u = norm2 @ lw["wup"]
            mlp = ((g / (1.0 + np.exp(-g))) * u) @ lw["wdown"]
            cur = resid1 + mlp
        expected_outputs.append(cur.astype(np.float32))
    return expected_outputs


def run_batched_prefill_matches_reference_case(hidden, inter, n_heads, n_kv_heads,
                                               head_dim, n_layers, n_real, seed=41):
    layers_np, rng = _make_model(hidden, inter, n_heads, n_kv_heads, head_dim,
                                 n_layers, seed)
    token_rows = [rng.normal(size=(hidden,)).astype(np.float32) * 0.3
                 for _ in range(n_real)]
    expected_outputs = _reference_all_steps(layers_np, token_rows, n_heads, n_kv_heads,
                                            head_dim)

    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    dram, x_addr, layer_weights, scratch_a, scratch_b, max_tiles = \
        tvm_backend.build_kvcache_model_dram_image(
            comp, hidden, n_layers, n_kv_heads, head_dim,
            max_seq_len=n_real, layer_weights_np=layers_np,
        )
    x_batch = np.zeros((M, hidden), dtype=np.float32)
    for r, row in enumerate(token_rows):
        x_batch[r, :] = row
    comp._const_regions[x_addr] = x_batch.tobytes()

    final_addr = tvm_backend._emit_model_kvcache_batched_fp32(
        comp, n_layers, hidden, inter, n_heads, n_kv_heads, head_dim,
        x_addr=x_addr, layer_weights=layer_weights, scratch_a=scratch_a,
        scratch_b=scratch_b, n_real=n_real,
    )
    n_instr = len(comp.instruction_stream) // 16

    needed = max(comp.dram_bytes_needed, len(dram))
    if needed > len(dram):
        dram = dram + bytearray(needed - len(dram) + 0x10000)
    for addr, data in comp._const_regions.items():
        dram[addr:addr + len(data)] = data

    with tempfile.TemporaryDirectory(prefix="rtv_prefill_ref_") as wd:
        p, d, o = (os.path.join(wd, n) for n in ("p.bin", "d.img", "o.bin"))
        open(p, "wb").write(bytes(comp.instruction_stream))
        open(d, "wb").write(bytes(dram))
        mb = max(1, len(dram) // (1024 * 1024) + 2)
        r = subprocess.run([SIM, p, HAL, str(mb), "--dram-image", d,
                            "--dump", f"{final_addr}:{M * hidden * 4}:{o}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"simulator failed: {r.stdout[-500:]}{r.stderr[-500:]}", n_instr
        got = np.frombuffer(open(o, "rb").read(), dtype=np.float32).reshape(M, hidden)

    max_diff = 0.0
    for t in range(n_real):
        d_t = float(np.abs(got[t] - expected_outputs[t]).max())
        max_diff = max(max_diff, d_t)
    if max_diff < 3e-2:
        return True, "", n_instr
    return False, f"max diff over {n_real} real rows: {max_diff:.4g}", n_instr


def run_prefill_then_continue_case(hidden, inter, n_heads, n_kv_heads, head_dim,
                                   n_layers, n_real, n_extra, seed=53):
    layers_np, rng = _make_model(hidden, inter, n_heads, n_kv_heads, head_dim,
                                 n_layers, seed)
    total = n_real + n_extra
    token_rows = [rng.normal(size=(hidden,)).astype(np.float32) * 0.3
                 for _ in range(total)]
    expected_outputs = _reference_all_steps(layers_np, token_rows, n_heads, n_kv_heads,
                                            head_dim)

    comp0 = NPUCompiler(HAL)
    comp0.dtype = DType.FP32
    dram, x_addr, layer_weights, scratch_a, scratch_b, max_tiles = \
        tvm_backend.build_kvcache_model_dram_image(
            comp0, hidden, n_layers, n_kv_heads, head_dim,
            max_seq_len=total, layer_weights_np=layers_np,
        )
    base_dram_cursor = comp0._dram_cursor

    got_outputs = []
    with tempfile.TemporaryDirectory(prefix="rtv_prefill_cont_") as wd:
        cur_path = os.path.join(wd, "state.img")
        open(cur_path, "wb").write(bytes(dram))

        # -- one batched prefill call for the first n_real tokens --
        comp = NPUCompiler(HAL)
        comp.dtype = DType.FP32
        comp.instruction_stream = bytearray()
        comp._dram_cursor = base_dram_cursor
        x_batch = np.zeros((M, hidden), dtype=np.float32)
        for r in range(n_real):
            x_batch[r, :] = token_rows[r]
        comp._const_regions[x_addr] = x_batch.tobytes()
        final_addr = tvm_backend._emit_model_kvcache_batched_fp32(
            comp, n_layers, hidden, inter, n_heads, n_kv_heads, head_dim,
            x_addr=x_addr, layer_weights=layer_weights, scratch_a=scratch_a,
            scratch_b=scratch_b, n_real=n_real,
        )
        step_dram = bytearray(open(cur_path, "rb").read())
        needed = max(comp.dram_bytes_needed, len(step_dram))
        if needed > len(step_dram):
            step_dram += bytearray(needed - len(step_dram) + 0x10000)
        for addr, data in comp._const_regions.items():
            step_dram[addr:addr + len(data)] = data
        open(cur_path, "wb").write(bytes(step_dram))
        next_path = os.path.join(wd, "step_prefill.img")
        (out_raw,) = _run_sim(comp.instruction_stream, cur_path, next_path,
                              extra_dumps=[(final_addr, M * hidden * 4, "out")])
        cur_path = next_path
        batch_out = np.frombuffer(out_raw, dtype=np.float32).reshape(M, hidden)
        for r in range(n_real):
            got_outputs.append(batch_out[r])

        # -- continue purely one token at a time, exactly test_decode_loop.py's path --
        n_instr = 0
        for t in range(n_real, total):
            comp = NPUCompiler(HAL)
            comp.dtype = DType.FP32
            comp.instruction_stream = bytearray()
            comp._dram_cursor = base_dram_cursor
            x_row = np.zeros((M, hidden), dtype=np.float32)
            x_row[0, :] = token_rows[t]
            comp._const_regions[x_addr] = x_row.tobytes()
            final_addr = tvm_backend._emit_model_kvcache_fp32(
                comp, n_layers, hidden, inter, n_heads, n_kv_heads, head_dim,
                x_addr=x_addr, layer_weights=layer_weights, scratch_a=scratch_a,
                scratch_b=scratch_b, pos=t,
            )
            n_instr = len(comp.instruction_stream) // 16
            step_dram = bytearray(open(cur_path, "rb").read())
            needed = max(comp.dram_bytes_needed, len(step_dram))
            if needed > len(step_dram):
                step_dram += bytearray(needed - len(step_dram) + 0x10000)
            for addr, data in comp._const_regions.items():
                step_dram[addr:addr + len(data)] = data
            open(cur_path, "wb").write(bytes(step_dram))
            next_path = os.path.join(wd, f"step_{t}.img")
            (out_raw,) = _run_sim(comp.instruction_stream, cur_path, next_path,
                                  extra_dumps=[(final_addr, M * hidden * 4, "out")])
            cur_path = next_path
            got_outputs.append(
                np.frombuffer(out_raw, dtype=np.float32).reshape(M, hidden)[0])

    max_diff = 0.0
    bad_step = -1
    for t in range(total):
        d_t = float(np.abs(got_outputs[t] - expected_outputs[t]).max())
        if d_t > max_diff:
            max_diff, bad_step = d_t, t
    if max_diff < 3e-2:
        return True, "", n_instr
    return False, f"step {bad_step} wrong, max diff {max_diff:.4g}", n_instr


def main():
    cases = [
        ("batched_prefill_matches_reference_full_tile",
         lambda: run_batched_prefill_matches_reference_case(
             32, 48, 4, 2, 16, n_layers=2, n_real=16)),
        ("batched_prefill_matches_reference_partial_tile",
         lambda: run_batched_prefill_matches_reference_case(
             32, 48, 4, 2, 16, n_layers=2, n_real=5)),
        ("prefill_then_continue_matches_pure_one_at_a_time",
         lambda: run_prefill_then_continue_case(
             32, 48, 4, 2, 16, n_layers=2, n_real=6, n_extra=4)),
        ("prefill_then_continue_full_tile_headdim32",
         lambda: run_prefill_then_continue_case(
             48, 64, 2, 1, 32, n_layers=2, n_real=16, n_extra=3)),
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
