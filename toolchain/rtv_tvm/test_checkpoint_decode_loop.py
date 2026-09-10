#!/usr/bin/env python3
"""
test_checkpoint_decode_loop.py — the two prior checkpoint/decode tests,
combined: a REAL downloaded Qwen3-0.6B checkpoint (`hf download
Qwen/Qwen3-0.6B`), run through several GENUINE, SEPARATE decode steps
(growing multi-tile KV-cache, real distinct tokens, RoPE at each step's
true position, q_norm/k_norm), exactly the way a real generation loop
would drive it.

test_checkpoint_loader.py proved real checkpoint values survive the
loader + compiler at real dimensions, but only for ONE decode step,
attending over "one real cached key, this step's own" (the same
simplification every attention test in this directory uses at the
single-call level). test_decode_loop.py proved the multi-step,
growing-KV-cache decode loop is correct, but only with random weights.
Neither, on its own, proves real Qwen3-0.6B weights survive an actual
multi-token autoregressive decode loop -- attending over a REAL,
multi-token context built from REAL projected/normalized/rotated K/V,
not one key. This does, by combining both: real checkpoint_loader.py
weights (INCLUDING q_norm/k_norm) driven through
tvm_backend._emit_model_kvcache_fp32 across several separate rtv_sim
invocations, round-tripping the whole DRAM image between them the same
way test_decode_loop.py does, checked step-by-step against a genuine
incremental numpy reference built from the SAME loaded tensors.

Kept to 1 layer and a handful of steps (not all 28 layers / a long
sequence) -- real dimensions (hidden=1024, head_dim=128) are slow to
schedule/codegen, and this is a correctness check, not a benchmark.
Expect ~1-2 minutes total.

Skips cleanly (does not fail) if no local Qwen3-0.6B checkpoint is found
-- see test_checkpoint_loader.py's docstring for the download command.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_checkpoint_decode_loop.py
"""

import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from compiler_state import NPUCompiler, DType  # noqa: E402
import tvm_backend                             # noqa: E402
import checkpoint_loader                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")

M = 16
SKIP = "SKIP"


def _find_checkpoint():
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*")
    for cand in sorted(glob.glob(pattern)):
        if os.path.isfile(os.path.join(cand, "config.json")):
            has_weights = os.path.isfile(os.path.join(cand, "model.safetensors")) or \
                any(f.endswith(".safetensors") for f in os.listdir(cand))
            if has_weights:
                return cand
    return None


def _rmsnorm_ref(v, gamma, eps):
    return v / np.sqrt((v.astype(np.float64) ** 2).mean() + eps) * gamma.astype(np.float64)


def _rope_ref(row, pos, head_dim, theta):
    half = head_dim // 2
    inv_freq = theta ** (-np.arange(0, half, dtype=np.float64) * 2.0 / head_dim)
    ang = pos * inv_freq
    cos_full = np.concatenate([np.cos(ang), np.cos(ang)])
    x1, x2 = row[:half], row[half:]
    rotate_half = np.concatenate([-x2, x1])
    sin_full = np.concatenate([np.sin(ang), np.sin(ang)])
    return row * cos_full + rotate_half * sin_full


def _run_sim(instr_stream, dram_path_in, dram_path_out, extra_dumps=()):
    """One decode step: load dram_path_in, execute, dump the WHOLE
    updated DRAM to dram_path_out (so the next step can load it), plus
    any extra_dumps (addr, nbytes, label) this test wants to inspect."""
    with tempfile.TemporaryDirectory(prefix="rtv_ckpt_decode_") as wd:
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


def run_real_checkpoint_decode_case(model_dir, n_layers, token_ids):
    """token_ids: one real vocab id per decode step -- genuinely distinct
    tokens, embedded via the REAL (tied) embedding table, exactly what a
    real generation loop would feed in one at a time."""
    cfg, layers, embed_tokens, lm_head, final_norm = \
        checkpoint_loader.load_qwen3_checkpoint(model_dir, n_layers=n_layers)
    hidden, inter = cfg["hidden_size"], cfg["intermediate_size"]
    n_heads, n_kv_heads, head_dim = cfg["num_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    rope_theta, eps = cfg["rope_theta"], cfg["rms_norm_eps"]
    n_steps = len(token_ids)
    rep = n_heads // n_kv_heads

    # -- genuine incremental numpy reference: real distinct tokens, real
    # growing KV-cache (attending over EVERY real cached key so far, not
    # "one key"), q_norm/k_norm, RoPE at each token's true position --
    k_cache_ref = [[[] for _ in range(n_kv_heads)] for _ in range(n_layers)]
    v_cache_ref = [[[] for _ in range(n_kv_heads)] for _ in range(n_layers)]
    expected_outputs = []
    for t, tok in enumerate(token_ids):
        cur = embed_tokens[tok].astype(np.float64)
        for li, lw in enumerate(layers):
            norm1 = _rmsnorm_ref(cur, lw["gamma1"], eps)
            q = (norm1 @ lw["wq"]).reshape(n_heads, head_dim)
            k_new = (norm1 @ lw["wk"]).reshape(n_kv_heads, head_dim)
            v_new = (norm1 @ lw["wv"]).reshape(n_kv_heads, head_dim)
            q = np.stack([_rmsnorm_ref(q[h], lw["q_norm"], eps) for h in range(n_heads)])
            k_new = np.stack([_rmsnorm_ref(k_new[h], lw["k_norm"], eps)
                              for h in range(n_kv_heads)])
            q = np.stack([_rope_ref(q[h], t, head_dim, rope_theta) for h in range(n_heads)])
            k_new = np.stack([_rope_ref(k_new[h], t, head_dim, rope_theta)
                              for h in range(n_kv_heads)])
            for h in range(n_kv_heads):
                k_cache_ref[li][h].append(k_new[h])
                v_cache_ref[li][h].append(v_new[h])
            ctx = np.zeros((n_heads, head_dim), dtype=np.float64)
            for qh in range(n_heads):
                kv_h = qh // rep
                K = np.stack(k_cache_ref[li][kv_h])   # (t+1, head_dim), REAL growing context
                V = np.stack(v_cache_ref[li][kv_h])
                sc = q[qh] @ K.T
                e = np.exp(sc - sc.max())
                p = e / e.sum()
                ctx[qh] = p @ V
            attn = ctx.reshape(n_heads * head_dim) @ lw["wo"]
            resid1 = cur + attn
            norm2 = _rmsnorm_ref(resid1, lw["gamma2"], eps)
            g = norm2 @ lw["wgate"]
            u = norm2 @ lw["wup"]
            mlp = ((g / (1.0 + np.exp(-g))) * u) @ lw["wdown"]
            cur = resid1 + mlp
        expected_outputs.append(cur.astype(np.float32))

    # -- device side: build the model once, then loop real decode steps,
    # round-tripping the WHOLE DRAM image through separate rtv_sim runs --
    comp0 = NPUCompiler(HAL)
    comp0.dtype = DType.FP32
    dram, x_addr, layer_weights, scratch_a, scratch_b, max_tiles = \
        tvm_backend.build_kvcache_model_dram_image(
            comp0, hidden, n_layers, n_kv_heads, head_dim,
            max_seq_len=n_steps, layer_weights_np=layers,
        )
    base_dram_cursor = comp0._dram_cursor

    got_outputs = []
    n_instr = 0
    with tempfile.TemporaryDirectory(prefix="rtv_ckpt_decode_state_") as wd:
        cur_dram_path = os.path.join(wd, "step_-1.img")
        open(cur_dram_path, "wb").write(bytes(dram))

        for t, tok in enumerate(token_ids):
            comp = NPUCompiler(HAL)
            comp.dtype = DType.FP32
            comp.instruction_stream = bytearray()
            comp._dram_cursor = base_dram_cursor

            x_row = np.zeros((M, hidden), dtype=np.float32)
            x_row[0, :] = embed_tokens[tok].astype(np.float32)
            comp._const_regions[x_addr] = x_row.tobytes()

            final_addr = tvm_backend._emit_model_kvcache_fp32(
                comp, n_layers, hidden, inter, n_heads, n_kv_heads, head_dim,
                x_addr=x_addr, layer_weights=layer_weights, scratch_a=scratch_a,
                scratch_b=scratch_b, pos=t, rope_theta=rope_theta,
            )
            n_instr = len(comp.instruction_stream) // 16

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
    max_relerr = 0.0
    for t in range(n_steps):
        bound = float(np.abs(expected_outputs[t]).max()) or 1.0
        err = float(np.abs(got_outputs[t] - expected_outputs[t]).max()) / bound
        max_relerr = max(max_relerr, err)
        ok_per_step.append(err < 1e-2)

    if all(ok_per_step):
        return True, "", n_instr
    bad_step = ok_per_step.index(False)
    return False, f"step {bad_step} wrong, max relerr over all steps {max_relerr:.3g}", n_instr


def main():
    model_dir = _find_checkpoint()
    if model_dir is None:
        print(f"{SKIP}  no local Qwen3-0.6B checkpoint found "
              f"(run: hf download Qwen/Qwen3-0.6B --include \"*.json\" --include \"*.safetensors\")")
        print("\n0/0 passed (skipped)")
        return 0

    cases = [
        ("real_qwen3_1layer_3token_decode",
         lambda: run_real_checkpoint_decode_case(model_dir, n_layers=1,
                                                  token_ids=[100, 7, 4821])),
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
