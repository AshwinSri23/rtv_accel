#!/usr/bin/env python3
"""
test_decode_loop.py — a genuine multi-step, multi-layer autoregressive
DECODE LOOP: real Q/K/V/O projections, RoPE at each step's true position,
a growing multi-tile KV-cache that persists ACROSS separate rtv_sim
invocations, and N transformer layers chained with ping-ponged buffers --
tvm_backend._emit_model_kvcache_fp32, the actual per-decode-step operation
a real generation loop needs.

This is qualitatively different from every earlier test in this
directory, which each verify ONE call's worth of work (one attention
block, one layer, one append). Nothing before this actually persisted
state across multiple SEPARATE simulator runs the way real autoregressive
decoding requires: the ISA has no control flow (see codegen.py's own
docstring -- "each decode step is a fresh ~12M-instruction program"), so
there is no single compiled program for "generate N tokens" -- each step
is its own fresh instruction stream, and the only thing connecting step
t to step t+1 is the DRAM IMAGE itself, which this test round-trips
through `rtv_sim --dram-image ... --dump ...` exactly the way a real
driver would: dump the WHOLE DRAM extent after step t, feed that file
back in as step t+1's --dram-image.

Verified against a genuine incremental numpy reference (real distinct
per-step tokens, real growing attention context, real RoPE at each
token's true position, N layers) -- not a single-shot batch pretending to
be a loop.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_decode_loop.py
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
    """Run one decode step: load dram_path_in, execute, dump the WHOLE
    updated DRAM to dram_path_out (so the next step can load it), plus
    any extra_dumps (addr, nbytes, path) for values this test wants to
    inspect. Returns the list of extra_dump results."""
    with tempfile.TemporaryDirectory(prefix="rtv_decode_") as wd:
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


def run_decode_loop_case(hidden, inter, n_heads, n_kv_heads, head_dim,
                         n_layers, n_steps, seed=17):
    rng = np.random.default_rng(seed)
    q_dim, kv_dim = n_heads * head_dim, n_kv_heads * head_dim
    w = DType.BYTES[DType.FP32]

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

    token_rows = [rng.normal(size=(hidden,)).astype(np.float32) * 0.3 for _ in range(n_steps)]

    # -- genuine incremental numpy reference: real growing KV-cache, RoPE
    # at each token's true position, N layers, one token per step --
    rep = n_heads // n_kv_heads
    k_cache_ref = [[[] for _ in range(n_kv_heads)] for _ in range(n_layers)]
    v_cache_ref = [[[] for _ in range(n_kv_heads)] for _ in range(n_layers)]
    expected_outputs = []
    for t in range(n_steps):
        cur = token_rows[t].astype(np.float64)
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
                K = np.stack(k_cache_ref[li][kv_h])   # (t+1, head_dim)
                V = np.stack(v_cache_ref[li][kv_h])
                sc = q[qh] @ K.T
                e = np.exp(sc - sc.max())
                p = e / e.sum()
                ctx[qh] = p @ V
            attn = ctx.reshape(q_dim) @ lw["wo"]
            resid1 = cur + attn
            norm2 = _rmsnorm_ref(resid1, lw["gamma2"])
            g = norm2 @ lw["wgate"]
            u = norm2 @ lw["wup"]
            mlp = ((g / (1.0 + np.exp(-g))) * u) @ lw["wdown"]
            cur = resid1 + mlp
        expected_outputs.append(cur.astype(np.float32))

    # -- device side: build the model once, then loop decode steps,
    # round-tripping the WHOLE DRAM image through separate rtv_sim runs --
    comp0 = NPUCompiler(HAL)
    comp0.dtype = DType.FP32
    dram, x_addr, layer_weights, scratch_a, scratch_b, max_tiles = \
        tvm_backend.build_kvcache_model_dram_image(
            comp0, hidden, n_layers, n_kv_heads, head_dim,
            max_seq_len=n_steps, layer_weights_np=layers_np,
        )
    base_dram_cursor = comp0._dram_cursor

    got_outputs = []
    with tempfile.TemporaryDirectory(prefix="rtv_decode_state_") as wd:
        cur_dram_path = os.path.join(wd, "step_-1.img")
        open(cur_dram_path, "wb").write(bytes(dram))

        for t in range(n_steps):
            comp = NPUCompiler(HAL)
            comp.dtype = DType.FP32
            comp.instruction_stream = bytearray()
            comp._dram_cursor = base_dram_cursor

            x_row = np.zeros((M, hidden), dtype=np.float32)
            x_row[0, :] = token_rows[t]
            comp._const_regions[x_addr] = x_row.tobytes()

            final_addr = tvm_backend._emit_model_kvcache_fp32(
                comp, n_layers, hidden, inter, n_heads, n_kv_heads, head_dim,
                x_addr=x_addr, layer_weights=layer_weights,
                scratch_a=scratch_a, scratch_b=scratch_b, pos=t,
            )
            n_instr = len(comp.instruction_stream) // 16

            # Bake this step's const regions (x_row, RoPE tables, masks,
            # permutation/broadcast matrices) into the DRAM image we're
            # about to feed in -- same finalize_dram_image mechanism every
            # other test uses, just applied to a file-backed image here.
            step_dram = bytearray(open(cur_dram_path, "rb").read())
            needed = max(comp.dram_bytes_needed, len(step_dram))
            if needed > len(step_dram):
                step_dram += bytearray(needed - len(step_dram) + 0x10000)
            for addr, data in comp._const_regions.items():
                step_dram[addr:addr + len(data)] = data
            open(cur_dram_path, "wb").write(bytes(step_dram))

            next_dram_path = os.path.join(wd, f"step_{t}.img")
            (out_raw,) = _run_sim(
                comp.instruction_stream, cur_dram_path, next_dram_path,
                extra_dumps=[(final_addr, M * hidden * 4, "out")],
            )
            got_row = np.frombuffer(out_raw, dtype=np.float32).reshape(M, hidden)[0]
            got_outputs.append(got_row)
            cur_dram_path = next_dram_path

    ok_per_step = []
    max_diff = 0.0
    for t in range(n_steps):
        d = np.abs(got_outputs[t] - expected_outputs[t])
        max_diff = max(max_diff, float(d.max()))
        ok_per_step.append(bool(np.allclose(got_outputs[t], expected_outputs[t], atol=3e-2)))

    if all(ok_per_step):
        return True, "", n_instr
    bad_step = ok_per_step.index(False)
    return False, f"step {bad_step} wrong, max diff over all steps {max_diff:.4g}", n_instr


def main():
    cases = [
        ("decode_1layer_1head_10steps",
         lambda: run_decode_loop_case(32, 48, 1, 1, 16, n_layers=1, n_steps=10)),
        ("decode_2layer_gqa_10steps",
         lambda: run_decode_loop_case(32, 48, 4, 2, 16, n_layers=2, n_steps=10)),
        ("decode_1layer_1head_20steps_spans_2tiles",
         lambda: run_decode_loop_case(32, 48, 1, 1, 16, n_layers=1, n_steps=20)),
        ("decode_2layer_headdim32",
         lambda: run_decode_loop_case(48, 64, 2, 1, 32, n_layers=2, n_steps=8)),
    ]
    n_pass = 0
    for name, fn in cases:
        try:
            ok, msg, n_instr = fn()
        except Exception as e:  # noqa: BLE001
            ok, msg, n_instr = False, f"{type(e).__name__}: {e}", 0
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({n_instr} instrs/step)"
              + (f"  -- {msg}" if msg else ""))
        n_pass += ok
    print(f"\n{n_pass}/{len(cases)} passed")
    return 0 if n_pass == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
