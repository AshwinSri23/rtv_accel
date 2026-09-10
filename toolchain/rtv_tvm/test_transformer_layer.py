#!/usr/bin/env python3
"""
test_transformer_layer.py — a full transformer layer, and multiple layers
chained with ping-pong DRAM buffers, entirely via TVM schedule+codegen.

Covers what was blocking TODO 1.2's "ping-pong activation DRAM buffers
across transformer layers" and 1.4's "unify DRAM image generation" /
"whole-program execution" items: nothing on the TVM path chained more than
one composite into anything resembling a model before this.
tvm_backend._emit_transformer_layer_fp32 composes RMSNorm -> attention
block (with real Q/K/V/O projections, not just the core) -> residual ->
RMSNorm -> SwiGLU -> residual; _emit_model_fp32 chains N of those,
alternating the activation buffer between two DRAM addresses instead of
allocating one per layer.

x is one real token; self-attention runs over its own (therefore also
16-identical-copy) K/V, which is mathematically exact for that case, not
an approximation -- see _emit_attention_block_fp32's docstring. head_dim
must equal compiler.tile_dim (16) -- the per-head K transpose only handles
one square tile. Genuinely different tokens in one call, RoPE, and a real
multi-token KV-cache are all separate, unimplemented pieces.

Two real conventions collide at the attention block's boundary, and
getting this wrong was the actual bug hunt building it:

    layer-level activations (x, resid1, norm2, ...): "row 0 real, rows
    1-15 ZERO" -- required by _emit_rmsnorm_row_fp32's REDUCE step,
    confirmed directly: feeding it a genuinely REPLICATED 16-row input
    (all rows identical, not zero) summed 16x too much and produced
    exactly 1/4 of the correct RMSNorm output.

    _emit_attention_core_fp32's Q/K/V: "all 16 rows IDENTICAL" -- required
    so its own REDUCE-based softmax isn't contaminated by zero padding
    the way a genuine max/sum would be.

_emit_attention_block_fp32 bridges these at its boundary: broadcasts
Q/K/V projections from "row0-real" to "all-replicated" before the core
(_emit_broadcast_row0_fp32, since there's no hardware broadcast primitive
-- a LOAD_TILE stride of 0 is reinterpreted by the simulator as "not set,"
not a literal zero-stride broadcast, so this uses GEMM against a
compile-time one-hot matrix instead), then masks the core's
all-replicated output back down to row0-real before returning
(_emit_mask_to_row0_fp32), so everything downstream sees the convention
it actually needs.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_transformer_layer.py
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
        # self-attention over ONE distinct key: softmax over identical
        # scores is uniform, so context = that key's own value.
        ctx[qh] = v[qh // rep]
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


def _run_sim(instr_stream, dram, addr, nbytes, shape):
    with tempfile.TemporaryDirectory(prefix="rtv_layer_") as wd:
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


def run_single_layer_case(hidden, inter, n_heads, n_kv_heads, head_dim, seed=11):
    rng = np.random.default_rng(seed)
    x_row = rng.normal(size=(hidden,)).astype(np.float32) * 0.5
    lw = _random_layer_weights(rng, hidden, inter, n_heads, n_kv_heads, head_dim)
    expected_row = _layer_ref(x_row, lw, hidden, n_heads, n_kv_heads, head_dim)
    expected = np.zeros((M, hidden), dtype=np.float32)
    expected[0, :] = expected_row

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
        out_addr=out_addr,
    )
    n_instr = len(comp.instruction_stream) // 16

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

    if np.allclose(got, expected, atol=1e-2):
        return True, "", n_instr
    d = np.abs(got - expected)
    return False, f"max delta {float(d.max()):.4g}", n_instr


def run_multilayer_case(n_layers, hidden, inter, n_heads, n_kv_heads, head_dim, seed=21):
    rng = np.random.default_rng(seed)
    x_row = rng.normal(size=(hidden,)).astype(np.float32) * 0.5
    layers = [_random_layer_weights(rng, hidden, inter, n_heads, n_kv_heads, head_dim)
              for _ in range(n_layers)]

    cur = x_row
    for lw in layers:
        cur = _layer_ref(cur, lw, hidden, n_heads, n_kv_heads, head_dim)
    expected = np.zeros((M, hidden), dtype=np.float32)
    expected[0, :] = cur

    x = np.zeros((M, hidden), dtype=np.float32)
    x[0, :] = x_row

    addrs = {}
    base = [0x1000]

    def place(name, arr):
        addrs[name] = base[0]
        base[0] += arr.nbytes + 0x1000
        return addrs[name]

    place("x", x)
    layer_weights = []
    for li, lw in enumerate(layers):
        entry = {
            "gamma1": place(f"l{li}_gamma1", np.tile(lw["gamma1"], (M, 1))),
            "gamma2": place(f"l{li}_gamma2", np.tile(lw["gamma2"], (M, 1))),
        }
        for k in ("wq", "wk", "wv", "wo", "wgate", "wup", "wdown"):
            entry[k] = place(f"l{li}_{k}", lw[k])
        layer_weights.append(entry)
    scratch_a = base[0]; base[0] += M * hidden * 4 + 0x1000
    scratch_b = base[0]; base[0] += M * hidden * 4 + 0x1000

    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = base[0] + 0x10000

    final_addr = tvm_backend._emit_model_fp32(
        comp, n_layers, hidden, inter, n_heads, n_kv_heads, head_dim,
        x_addr=addrs["x"], layer_weights=layer_weights,
        scratch_a=scratch_a, scratch_b=scratch_b,
    )
    n_instr = len(comp.instruction_stream) // 16
    expected_final = scratch_b if n_layers % 2 == 1 else scratch_a
    if final_addr != expected_final:
        return False, f"ping-pong landed on wrong buffer for {n_layers} layers", n_instr

    dram_size = max(base[0], comp.dram_bytes_needed) + 0x10000
    dram = bytearray(dram_size)
    dram[addrs["x"]:addrs["x"] + x.nbytes] = x.tobytes()
    for li, lw in enumerate(layers):
        for name, arr in ((f"l{li}_gamma1", np.tile(lw["gamma1"], (M, 1))),
                          (f"l{li}_gamma2", np.tile(lw["gamma2"], (M, 1))),
                          (f"l{li}_wq", lw["wq"]), (f"l{li}_wk", lw["wk"]),
                          (f"l{li}_wv", lw["wv"]), (f"l{li}_wo", lw["wo"]),
                          (f"l{li}_wgate", lw["wgate"]), (f"l{li}_wup", lw["wup"]),
                          (f"l{li}_wdown", lw["wdown"])):
            a = layer_weights[li][name.split("_", 1)[1]]
            dram[a:a + arr.nbytes] = arr.astype(np.float32).tobytes()
    for addr, data in comp._const_regions.items():
        dram[addr:addr + len(data)] = data

    got = _run_sim(comp.instruction_stream, dram, final_addr, M * hidden * 4, (M, hidden))

    if np.allclose(got, expected, atol=2e-2):
        return True, "", n_instr
    d = np.abs(got - expected)
    return False, f"max delta {float(d.max()):.4g}", n_instr


def main():
    cases = [
        ("layer_1L", lambda: run_single_layer_case(32, 48, 2, 1, 16)),
        ("layer_1L_gqa", lambda: run_single_layer_case(32, 48, 4, 2, 16, seed=13)),
        ("model_2L", lambda: run_multilayer_case(2, 32, 48, 2, 1, 16)),
        ("model_3L_pingpong", lambda: run_multilayer_case(3, 32, 48, 2, 1, 16, seed=21)),
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
