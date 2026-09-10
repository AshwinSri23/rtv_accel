#!/usr/bin/env python3
"""
test_dram_builder.py — tvm_backend.build_model_dram_image /
finalize_dram_image, the "given a model's actual weights, produce its DRAM
image" step TODO 1.4 asked for.

Every test before this one (test_transformer_layer.py included) built its
DRAM image by hand with an ad-hoc `place()` closure, bump-allocating
addresses and writing arrays into a bytearray one at a time, with the
caller responsible for knowing that gamma needs broadcasting to (16,
hidden) and x needs to be "row 0 real, rest zero" (see
_emit_rmsnorm_row_fp32's docstring for why). build_model_dram_image
generalizes that into a real, reusable function that takes a model's raw
per-layer weights (whatever a checkpoint loader would hand back) and does
all of that itself -- the TVM-path counterpart to what ingest_model/
export_dram_image do on the npu_compiler.py side.

This test builds a fresh multi-layer model purely through
build_model_dram_image + _emit_model_fp32 + finalize_dram_image (no manual
address placement at all), runs it through rtv_sim, and checks it against
the SAME numpy reference test_transformer_layer.py's model_3L_pingpong case
uses -- proving the builder produces an image at least as correct as the
hand-built one, not a new, looser check.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_dram_builder.py
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


def _rmsnorm_ref(v, gamma):
    return v / np.sqrt((v.astype(np.float64) ** 2).mean() + 1e-6) * gamma.astype(np.float64)


def _layer_ref(x_row, lw, hidden, n_heads, n_kv_heads, head_dim):
    q_dim = n_heads * head_dim
    norm1 = _rmsnorm_ref(x_row, lw["gamma1"])
    q = (norm1 @ lw["wq"].astype(np.float64)).reshape(n_heads, head_dim)
    k = (norm1 @ lw["wk"].astype(np.float64)).reshape(n_kv_heads, head_dim)
    v = (norm1 @ lw["wv"].astype(np.float64)).reshape(n_kv_heads, head_dim)
    rep = n_heads // n_kv_heads
    ctx = np.zeros((n_heads, head_dim), dtype=np.float64)
    for qh in range(n_heads):
        ctx[qh] = v[qh // rep]   # self-attention over one distinct key
    attn = ctx.reshape(q_dim) @ lw["wo"].astype(np.float64)
    resid1 = x_row.astype(np.float64) + attn
    norm2 = _rmsnorm_ref(resid1, lw["gamma2"])
    g = norm2 @ lw["wgate"].astype(np.float64)
    u = norm2 @ lw["wup"].astype(np.float64)
    mlp = ((g / (1.0 + np.exp(-g))) * u) @ lw["wdown"].astype(np.float64)
    return (resid1 + mlp).astype(np.float32)


def _random_layer_weights(rng, hidden, inter, n_heads, n_kv_heads, head_dim):
    q_dim, kv_dim = n_heads * head_dim, n_kv_heads * head_dim
    return dict(
        gamma1=rng.uniform(0.5, 1.5, size=(hidden,)).astype(np.float32),
        gamma2=rng.uniform(0.5, 1.5, size=(hidden,)).astype(np.float32),
        wq=(rng.normal(size=(hidden, q_dim)) * 0.2).astype(np.float32),
        wk=(rng.normal(size=(hidden, kv_dim)) * 0.2).astype(np.float32),
        wv=(rng.normal(size=(hidden, kv_dim)) * 0.2).astype(np.float32),
        wo=(rng.normal(size=(q_dim, hidden)) * 0.2).astype(np.float32),
        wgate=(rng.normal(size=(hidden, inter)) * 0.1).astype(np.float32),
        wup=(rng.normal(size=(hidden, inter)) * 0.1).astype(np.float32),
        wdown=(rng.normal(size=(inter, hidden)) * 0.1).astype(np.float32),
    )


def run_dram_builder_case(n_layers, hidden, inter, n_heads, n_kv_heads, head_dim, seed=31):
    rng = np.random.default_rng(seed)
    x_row = rng.normal(size=(hidden,)).astype(np.float32) * 0.5
    layers_np = [_random_layer_weights(rng, hidden, inter, n_heads, n_kv_heads, head_dim)
                 for _ in range(n_layers)]

    cur = x_row
    for lw in layers_np:
        cur = _layer_ref(cur, lw, hidden, n_heads, n_kv_heads, head_dim)
    expected = np.zeros((M, hidden), dtype=np.float32)
    expected[0, :] = cur

    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()

    dram, x_addr, layer_weight_addrs, scratch_a, scratch_b = tvm_backend.build_model_dram_image(
        comp, hidden, n_layers, x_row, layers_np,
    )

    final_addr = tvm_backend._emit_model_fp32(
        comp, n_layers, hidden, inter, n_heads, n_kv_heads, head_dim,
        x_addr=x_addr, layer_weights=layer_weight_addrs,
        scratch_a=scratch_a, scratch_b=scratch_b,
    )
    n_instr = len(comp.instruction_stream) // 16

    dram = tvm_backend.finalize_dram_image(comp, dram, min_addr=final_addr,
                                           min_len=M * hidden * 4)

    with tempfile.TemporaryDirectory(prefix="rtv_dram_") as wd:
        p, d = os.path.join(wd, "p.bin"), os.path.join(wd, "d.img")
        open(p, "wb").write(bytes(comp.instruction_stream))
        open(d, "wb").write(bytes(dram))
        out = os.path.join(wd, "o.bin")
        mb = max(1, len(dram) // (1024 * 1024) + 2)
        r = subprocess.run(
            [SIM, p, HAL, str(mb), "--dram-image", d,
             "--dump", f"{final_addr}:{M * hidden * 4}:{out}"],
            capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"simulator failed: {r.stdout[-500:]}{r.stderr[-500:]}", 0
        got = np.frombuffer(open(out, "rb").read(), dtype=np.float32).reshape(M, hidden)

    if np.allclose(got, expected, atol=2e-2):
        return True, "", n_instr
    d = np.abs(got - expected)
    return False, f"max delta {float(d.max()):.4g}", n_instr


def main():
    cases = [
        ("dram_builder_1L", lambda: run_dram_builder_case(1, 32, 48, 2, 1, 16)),
        ("dram_builder_3L", lambda: run_dram_builder_case(3, 32, 48, 2, 1, 16, seed=31)),
        ("dram_builder_2L_gqa", lambda: run_dram_builder_case(2, 32, 48, 4, 2, 16, seed=17)),
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
