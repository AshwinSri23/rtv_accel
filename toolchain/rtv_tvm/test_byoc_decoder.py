#!/usr/bin/env python3
"""
test_byoc_decoder.py -- TODO 5.4 step 4's acceptance test: the graph-driven (BYOC) path
produces a BYTE-IDENTICAL instruction stream to the hand-driven decoder.

A Qwen3 decode step is built as a Relax graph (decoder_graph.py): RMSNorm, residual adds
and the SwiGLU FFN are claimed by the RTV-Accel patterns and lowered by emit_composite;
the config-aware attention block (q/k-norm, RoPE, KV-cache, GQA) is an extern call lowered
by the existing emitter. For each of

  * one layer           vs tvm_backend._emit_model_kvcache_fp32 (1 layer)
  * a 3-layer model     vs tvm_backend._emit_model_kvcache_fp32
  * 5 layers relocated  graph-compiled layers 0-1 + address-patched layers vs the hand path

the two instruction streams must match byte for byte (not merely compute the same numbers).
GQA, q_norm/k_norm, RoPE at a nonzero position, the attention scale and RMSNorm epsilon are
all on. Finally one graph-driven step is run on rtv_sim and compared to the hand-driven one.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_byoc_decoder.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from compiler_state import NPUCompiler, DType     # noqa: E402
import tvm_backend                                 # noqa: E402
import decoder_graph                               # noqa: E402
from test_prefill_batching import _make_model, HAL, M   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HID, INTER, NH, NKV, HD, EPS, POS, THETA = 32, 48, 4, 2, 16, 1e-6, 5, 1e6
SCALE = HD ** -0.5
CFG = dict(hidden=HID, inter=INTER, n_heads=NH, n_kv_heads=NKV, head_dim=HD, eps=EPS)


def build(n_layers, seed, max_seq=8):
    layers, rng = _make_model(HID, INTER, NH, NKV, HD, n_layers, seed)
    for lw in layers:
        lw["q_norm"] = rng.uniform(0.5, 1.5, size=(HD,)).astype(np.float32)
        lw["k_norm"] = rng.uniform(0.5, 1.5, size=(HD,)).astype(np.float32)
    c0 = NPUCompiler(HAL)
    c0.dtype = DType.FP32
    dram, x_addr, lws, sa, sb, _ = tvm_backend.build_kvcache_model_dram_image(
        c0, HID, n_layers, NKV, HD, max_seq_len=max_seq, layer_weights_np=layers)
    return dict(dram=dram, x=x_addr, lws=lws, sa=sa, sb=sb, base=c0._dram_cursor, layers=layers)


def fresh(base):
    c = NPUCompiler(HAL)
    c.dtype = DType.FP32
    c.instruction_stream = bytearray()
    c._dram_cursor = base
    c.rms_eps = EPS
    return c


def hand(m, n_layers, relocated=False):
    c = fresh(m["base"])
    fn = (tvm_backend._emit_model_kvcache_fp32_relocated if relocated
          else tvm_backend._emit_model_kvcache_fp32)
    out = fn(c, n_layers, HID, INTER, NH, NKV, HD, x_addr=m["x"], layer_weights=m["lws"],
             scratch_a=m["sa"], scratch_b=m["sb"], pos=POS, rope_theta=THETA, attn_scale=SCALE)
    return c, out


def graph(m, n_layers, relocated=False):
    c = fresh(m["base"])
    if relocated:
        emitter = decoder_graph.make_layer_emitter(CFG, m["lws"], POS, THETA, SCALE)
        out = tvm_backend._emit_model_kvcache_fp32_relocated(
            c, n_layers, HID, INTER, NH, NKV, HD, x_addr=m["x"], layer_weights=m["lws"],
            scratch_a=m["sa"], scratch_b=m["sb"], pos=POS, rope_theta=THETA, attn_scale=SCALE,
            layer_emitter=emitter)
    else:
        out = decoder_graph.emit_model_graph(c, CFG, n_layers, m["x"], m["lws"], m["sa"], m["sb"],
                                             POS, THETA, SCALE)
    return c, out


def hand_prefill(m, n_layers, n_real, tile_idx):
    c = fresh(m["base"])
    out = tvm_backend._emit_model_kvcache_batched_fp32(
        c, n_layers, HID, INTER, NH, NKV, HD, x_addr=m["x"], layer_weights=m["lws"],
        scratch_a=m["sa"], scratch_b=m["sb"], n_real=n_real, rope_theta=THETA, tile_idx=tile_idx,
        attn_scale=SCALE)
    return c, out


def graph_prefill(m, n_layers, n_real, tile_idx):
    c = fresh(m["base"])
    out = decoder_graph.emit_prefill_graph(c, CFG, n_layers, m["x"], m["lws"], m["sa"], m["sb"],
                                           n_real, tile_idx, THETA, SCALE)
    return c, out


def graph_prefill_relocated(m, n_layers, n_real, tile_idx):
    c = fresh(m["base"])
    out = tvm_backend._emit_model_kvcache_fp32_relocated(
        c, n_layers, HID, INTER, NH, NKV, HD, x_addr=m["x"], layer_weights=m["lws"],
        scratch_a=m["sa"], scratch_b=m["sb"], pos=0, rope_theta=THETA, attn_scale=SCALE,
        layer_emitter=decoder_graph.make_prefill_layer_emitter(CFG, n_real, tile_idx, THETA, SCALE))
    return c, out


def compare(name, hc, ho, gc, go):
    same = bytes(hc.instruction_stream) == bytes(gc.instruction_stream) and ho == go
    return (name, same, f"{len(hc.instruction_stream) // 16} instrs"
            + ("" if same else f", hand {len(hc.instruction_stream)} B vs graph "
                               f"{len(gc.instruction_stream)} B, out {ho:#x} vs {go:#x}"))


def run_sim(m, comp, out_addr, x, rows=1):
    img = bytearray(m["dram"])
    need = max(comp.dram_bytes_needed, len(img))
    img += bytearray(max(0, need - len(img)) + 0x10000)
    xt = np.zeros((M, HID), np.float32)
    xt[:rows] = np.asarray(x).reshape(rows, HID)
    img[m["x"]:m["x"] + xt.nbytes] = xt.tobytes()
    for a, d in comp._const_regions.items():
        img[a:a + len(d)] = d
    with tempfile.TemporaryDirectory() as wd:
        p, d, o = (os.path.join(wd, f) for f in ("p.bin", "d.img", "o.bin"))
        open(p, "wb").write(bytes(comp.instruction_stream))
        open(d, "wb").write(bytes(img))
        r = subprocess.run([SIM, p, HAL, str(len(img) // (1 << 20) + 2), "--dram-image", d,
                            "--dump", f"{out_addr}:{M * HID * 4}:{o}"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-300:] + r.stderr[-300:]
        out = np.frombuffer(open(o, "rb").read(), np.float32).reshape(M, HID)
        return out[0].copy() if rows == 1 else out[:rows].copy()


def main():
    results = []
    m1 = build(1, 21)
    results.append(compare("one_layer_byte_identical", *hand(m1, 1), *graph(m1, 1)))
    m3 = build(3, 22)
    results.append(compare("three_layer_model_byte_identical", *hand(m3, 3), *graph(m3, 3)))
    m5 = build(5, 23)
    results.append(compare("five_layers_relocated_byte_identical",
                           *hand(m5, 5, relocated=True), *graph(m5, 5, relocated=True)))
    # and the graph-relocated stream equals the fully recompiled hand stream too
    results.append(compare("graph_relocated_equals_full_hand_recompile",
                           *hand(m5, 5), *graph(m5, 5, relocated=True)))
    # run one graph-driven step on the simulator and compare to the hand-driven step
    x = np.random.default_rng(3).normal(size=HID).astype(np.float32)
    (hc, ho), (gc, go) = hand(m3, 3), graph(m3, 3)
    a, b = run_sim(m3, hc, ho, x), run_sim(m3, gc, go, x)
    results.append(("graph_step_matches_hand_step_on_simulator", bool(np.array_equal(a, b)),
                    f"max |delta| {float(np.abs(a - b).max()):.1e}, |out| {float(np.abs(a).max()):.2f}"))
    # ---- PREFILL: the batched 16-row path, graph-driven (per-row norms and the multi-tile causal
    # attention are externs; the residual adds and SwiGLU are claimed)
    mp = build(3, 24, max_seq=40)
    results.append(compare("prefill_full_tile_byte_identical",
                           *hand_prefill(mp, 3, 16, 0), *graph_prefill(mp, 3, 16, 0)))
    results.append(compare("prefill_partial_tile_byte_identical",
                           *hand_prefill(mp, 3, 10, 0), *graph_prefill(mp, 3, 10, 0)))
    results.append(compare("prefill_second_tile_attends_back_byte_identical",
                           *hand_prefill(mp, 3, 6, 1), *graph_prefill(mp, 3, 6, 1)))
    results.append(compare("prefill_relocated_3layers_byte_identical",
                           *graph_prefill(mp, 3, 16, 0), *graph_prefill_relocated(mp, 3, 16, 0)))
    results.append(compare("prefill_relocated_second_tile_byte_identical",
                           *graph_prefill(mp, 3, 6, 1), *graph_prefill_relocated(mp, 3, 6, 1)))
    xb = np.random.default_rng(8).normal(size=(10, HID)).astype(np.float32)
    (hc, ho), (gc, go) = hand_prefill(mp, 3, 10, 0), graph_prefill(mp, 3, 10, 0)
    a, b = run_sim(mp, hc, ho, xb, rows=10), run_sim(mp, gc, go, xb, rows=10)
    results.append(("graph_prefill_matches_hand_prefill_on_simulator", bool(np.array_equal(a, b)),
                    f"max |delta| {float(np.abs(a - b).max()):.1e}, |out| {float(np.abs(a).max()):.2f}"))
    n = 0
    for name, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  -- {msg}")
        n += ok
    print(f"\n{n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
