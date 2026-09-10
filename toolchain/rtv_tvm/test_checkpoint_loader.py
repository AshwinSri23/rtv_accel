#!/usr/bin/env python3
"""
test_checkpoint_loader.py — checkpoint_loader.py against a REAL downloaded
Qwen3-0.6B checkpoint (`hf download Qwen/Qwen3-0.6B`), run through the FP32
TVM path AT QWEN3's ACTUAL FULL DIMENSIONS (hidden=1024, 16 heads over 8 KV
heads, head_dim=128, intermediate=3072) -- not a synthetic analogue.

Every earlier test in this directory used random weights checked against a
numpy reference built from those SAME random weights -- genuine, honest
verification of the compiler, but it says nothing about whether real
checkpoint VALUES survive the loader's HF-layout transpose correctly. This
does: load real layer 0 (and layer 1, for a real two-layer chain) from the
actual model file, run them through _emit_model_kvcache_fp32 and rtv_sim,
and check against a numpy reference built from the IDENTICAL loaded
tensors (not re-derived, not re-randomized).

This is NOT a claim that the result matches real Qwen3-0.6B's actual
published behavior -- it deliberately is not, for a real, documented
reason: Qwen3 applies a per-head RMSNorm to Q and K (`q_norm`/`k_norm`)
before RoPE, which tvm_backend._emit_attention_block_kvcache_fp32 does not
implement yet (see checkpoint_loader.py's module docstring). Both the
device path and this test's own numpy reference skip that step
IDENTICALLY, so what this proves is "the loader's real weight values,
run through the real compiler pipeline at real dimensions, agree with an
independent computation using those same values" -- loader + compiler
correctness, not architectural fidelity to Qwen3's true forward pass.
That gap is tracked in TODO.md, not hidden here.

Real dimensions are slow: expect this file to take noticeably longer than
anything else in this directory (~10-40s per case, dominated by TVM
schedule/codegen emission at hidden=1024/head_dim=128, not by rtv_sim
itself, which stays well under a second per decode step even here).

Skips cleanly (does not fail) if no local Qwen3-0.6B checkpoint is found
-- this is a ~1.5 GB download
(`hf download Qwen/Qwen3-0.6B --include "*.json" --include "*.safetensors"`),
not something a fresh checkout or CI machine should be required to have.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_checkpoint_loader.py
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
    """Look in the standard `hf download` cache location; None if absent."""
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


def _run_sim(instr_stream, dram, addr, nbytes, shape):
    with tempfile.TemporaryDirectory(prefix="rtv_ckpt_") as wd:
        p, d = os.path.join(wd, "p.bin"), os.path.join(wd, "d.img")
        open(p, "wb").write(bytes(instr_stream))
        open(d, "wb").write(bytes(dram))
        out = os.path.join(wd, "o.bin")
        mb = max(1, len(dram) // (1024 * 1024) + 2)
        r = subprocess.run([SIM, p, HAL, str(mb), "--dram-image", d,
                            "--dump", f"{addr}:{nbytes}:{out}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"simulator failed: {r.stdout[-800:]}{r.stderr[-800:]}")
        return np.frombuffer(open(out, "rb").read(), dtype=np.float32).reshape(shape)


def run_real_checkpoint_case(model_dir, n_layers, token_id, pos=0, seed=0):
    cfg, layers, embed_tokens, lm_head, final_norm = \
        checkpoint_loader.load_qwen3_checkpoint(model_dir, n_layers=n_layers)
    hidden, inter = cfg["hidden_size"], cfg["intermediate_size"]
    n_heads, n_kv_heads, head_dim = cfg["num_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    rope_theta, eps = cfg["rope_theta"], cfg["rms_norm_eps"]

    x_row = embed_tokens[token_id].astype(np.float32)

    # Numpy reference over the SAME loaded real tensors -- no q_norm/k_norm,
    # matching what the device path below also skips (see module docstring).
    cur = x_row.astype(np.float64)
    rep = n_heads // n_kv_heads
    for lw in layers:
        norm1 = _rmsnorm_ref(cur, lw["gamma1"], eps)
        q = (norm1 @ lw["wq"]).reshape(n_heads, head_dim)
        k = (norm1 @ lw["wk"]).reshape(n_kv_heads, head_dim)
        v = (norm1 @ lw["wv"]).reshape(n_kv_heads, head_dim)
        q = np.stack([_rope_ref(q[h], pos, head_dim, rope_theta) for h in range(n_heads)])
        k = np.stack([_rope_ref(k[h], pos, head_dim, rope_theta) for h in range(n_kv_heads)])
        ctx = np.zeros((n_heads, head_dim), dtype=np.float64)
        for qh in range(n_heads):
            ctx[qh] = v[qh // rep]   # one real cached key (this step's own) -> uniform softmax
        attn = ctx.reshape(n_heads * head_dim) @ lw["wo"]
        resid1 = cur + attn
        norm2 = _rmsnorm_ref(resid1, lw["gamma2"], eps)
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
            max_seq_len=pos + 1, layer_weights_np=layers,
        )
    base_cursor = comp0._dram_cursor

    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = base_cursor
    x_buf = np.zeros((M, hidden), dtype=np.float32)
    x_buf[0, :] = x_row
    comp._const_regions[x_addr] = x_buf.tobytes()

    final_addr = tvm_backend._emit_model_kvcache_fp32(
        comp, n_layers, hidden, inter, n_heads, n_kv_heads, head_dim,
        x_addr=x_addr, layer_weights=layer_weights, scratch_a=scratch_a,
        scratch_b=scratch_b, pos=pos, rope_theta=rope_theta,
    )
    n_instr = len(comp.instruction_stream) // 16

    needed = max(comp.dram_bytes_needed, len(dram))
    if needed > len(dram):
        dram = dram + bytearray(needed - len(dram) + 0x10000)
    for addr, data in comp._const_regions.items():
        dram[addr:addr + len(data)] = data

    got = _run_sim(comp.instruction_stream, dram, final_addr, M * hidden * 4, (M, hidden))[0]

    bound = float(np.abs(expected).max()) or 1.0
    err = float(np.abs(got - expected).max()) / bound
    if err < 1e-2:
        return True, "", n_instr
    return False, f"max err/|out| {err:.3g}", n_instr


def main():
    model_dir = _find_checkpoint()
    if model_dir is None:
        print(f"{SKIP}  no local Qwen3-0.6B checkpoint found "
              f"(run: hf download Qwen/Qwen3-0.6B --include \"*.json\" --include \"*.safetensors\")")
        print("\n0/0 passed (skipped)")
        return 0

    cases = [
        ("real_qwen3_layer0_token100",
         lambda: run_real_checkpoint_case(model_dir, n_layers=1, token_id=100)),
        ("real_qwen3_2layers_token7", lambda: run_real_checkpoint_case(
            model_dir, n_layers=2, token_id=7, seed=1)),
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
