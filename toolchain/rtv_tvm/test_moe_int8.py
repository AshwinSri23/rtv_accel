#!/usr/bin/env python3
"""
test_moe_int8.py -- MoE experts stored as INT8 (ISA v1.8 INT8W), both routing modes.

  1. host-segmented: _emit_moe_ffn_fp32 (top-2 experts, route weights) with
     compiler.int8_weights, vs numpy on the DEQUANTISED expert weights
  2. on-device top-1: ONE fixed stream (REDUCE_ARGMAX + INDIRECT_ADDR loads of the
     INT8 weights AND their scale rows) picks different experts for different
     inputs, vs numpy on the dequantised weights
  3. footprint: an expert is K*N + 4N bytes per matrix (~4x smaller than FP32)

The router stays FP32 (it is tiny and decides the routing).

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_moe_int8.py
"""
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import tvm_backend                                             # noqa: E402
from compiler_state import _padded                             # noqa: E402
from test_moe_ffn import (build_model, fresh_comp, silu, run_prog, M,   # noqa: E402
                          X_ADDR, ROUTER_ADDR, EXPERT_BASE, STAGING)
from test_persistent_session import Session                    # noqa: E402

GAMMA_ADDR, EPS = 0x4000, 1e-6
NAMES = ("w_gate", "w_up", "w_down")


def deq(w):
    q, s = tvm_backend.quantize_int8_per_channel(w)
    return q.astype(np.float64) * s.astype(np.float64)


def layout_int8(hidden, inter, router, experts):
    blobs = lambda ex: [tvm_backend.int8_weight_blob(ex[k]) for k in NAMES]
    per = sum(((b.nbytes + 0xFF) & ~0xFF) for b in blobs(experts[0]))
    stride = (per + 0xFFF) & ~0xFFF
    image = bytearray(STAGING + 0x100000)
    image[ROUTER_ADDR:ROUTER_ADDR + router.nbytes] = router.tobytes()
    addrs = []
    for e, ex in enumerate(experts):
        a, cur = {}, EXPERT_BASE + e * stride
        for k, b in zip(NAMES, blobs(ex)):
            a[k] = cur
            image[cur:cur + b.nbytes] = b.tobytes()
            cur += (b.nbytes + 0xFF) & ~0xFF
        addrs.append(a)
    return image, addrs, stride, per


def ffn_ref(x, ex):
    g, u, d = (deq(ex[k]) for k in NAMES)
    return (silu(x @ g) * (x @ u)) @ d


def case_host_segmented():
    hidden, inter, E = 32, 48, 4
    router, experts = build_model(hidden, inter, E, seed=3)
    image, addrs, _, _ = layout_int8(hidden, inter, router, experts)
    x = np.zeros((M, hidden), np.float32)
    x[0] = np.random.default_rng(1).normal(size=hidden).astype(np.float32)
    sel, rw = [2, 0], [0.7, 0.3]
    c = fresh_comp()
    c.int8_weights = True
    out = c.alloc_staging(M * hidden * 4)
    tvm_backend._emit_moe_ffn_fp32(c, hidden, inter, X_ADDR, [addrs[e] for e in sel], rw, out)
    with tempfile.TemporaryDirectory() as wd:
        img = os.path.join(wd, "img")
        open(img, "wb").write(bytes(image))
        sess = Session(dram_mb=len(image) // (1024 * 1024) + 2, image=img)
        try:
            f = os.path.join(wd, "x.bin")
            open(f, "wb").write(x.tobytes())
            assert sess.cmd(f"WRITE {X_ADDR} {f}").startswith("OK")
            raw = run_prog(sess, wd, c, "p", dump=(out, M * hidden * 4))
        finally:
            sess.close()
    got = np.frombuffer(raw, np.float32).reshape(M, hidden)[0]
    ref = sum(w * ffn_ref(x[0].astype(np.float64), experts[e]) for e, w in zip(sel, rw))
    err = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12))
    return err < 1e-3, f"rel err {err:.2e}"


def case_ondevice():
    hidden, inter, E = 32, 48, 4
    router, experts = build_model(hidden, inter, E, seed=11)
    image, addrs, stride, _ = layout_int8(hidden, inter, router, experts)
    gamma = (np.random.default_rng(2).normal(size=hidden) * 0.5 + 1).astype(np.float32)
    gt = np.tile(gamma, (M, 1)).astype(np.float32)
    image[GAMMA_ADDR:GAMMA_ADDR + gt.nbytes] = gt.tobytes()
    c = fresh_comp()
    c.rms_eps = EPS
    c.int8_weights = True
    norm = c.alloc_staging(M * hidden * 4)
    tvm_backend._emit_rmsnorm_row_fp32(
        c, hidden, x_addr=X_ADDR, gamma_addr=GAMMA_ADDR, out_addr=norm,
        scratch_addr=c.alloc_staging(4), stage_addr=c.alloc_staging(M * _padded(hidden) * 4))
    out_addr = c.alloc_staging(M * hidden * 4)
    tvm_backend._emit_moe_top1_ondevice_fp32(c, hidden, inter, X_ADDR, norm, ROUTER_ADDR,
                                             E, addrs[0], stride, out_addr)
    rng = np.random.default_rng(5)
    chosen, errs = [], []
    with tempfile.TemporaryDirectory() as wd:
        img = os.path.join(wd, "img")
        open(img, "wb").write(bytes(image))
        prog = os.path.join(wd, "p.bin")
        open(prog, "wb").write(bytes(c.instruction_stream))
        sess = Session(dram_mb=len(image) // (1024 * 1024) + 2, image=img)
        try:
            for addr, data in c._const_regions.items():
                f = os.path.join(wd, f"c{addr}.bin")
                open(f, "wb").write(bytes(data))
                assert sess.cmd(f"WRITE {addr} {f}").startswith("OK")
            for t in range(8):
                x = np.zeros((M, hidden), np.float32)
                x[0] = rng.normal(size=hidden).astype(np.float32)
                f = os.path.join(wd, f"x{t}.bin")
                open(f, "wb").write(x.tobytes())
                assert sess.cmd(f"WRITE {X_ADDR} {f}").startswith("OK")
                o = os.path.join(wd, f"o{t}.bin")
                r = sess.cmd(f"RUN {prog} {out_addr}:{M * hidden * 4}:{o}")
                assert r.startswith("OK"), r
                got = np.frombuffer(open(o, "rb").read(), np.float32).reshape(M, hidden)[0]
                xr = x[0].astype(np.float64)
                n = xr / np.sqrt(np.mean(xr * xr) + EPS) * gamma
                e = int(np.argmax(n @ router[:, :E].astype(np.float64)))
                ref = xr + ffn_ref(n, experts[e])
                errs.append(float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12)))
                chosen.append(e)
        finally:
            sess.close()
    ok = max(errs) < 1e-3 and len(set(chosen)) >= 2
    return ok, f"max rel err {max(errs):.2e}, experts chosen {sorted(set(chosen))} by one stream"


def case_footprint():
    hidden, inter = 1024, 768          # a typical small-expert shape
    ex = {k: np.zeros(s, np.float32) for k, s in
          zip(NAMES, ((hidden, inter), (hidden, inter), (inter, hidden)))}
    w32 = sum(v.nbytes for v in ex.values())
    w8 = sum(tvm_backend.int8_weight_blob(v).nbytes for v in ex.values())
    return w8 * 3.9 < w32, f"expert {w32} B fp32 -> {w8} B int8+scales ({w32 / w8:.2f}x)"


def main():
    cases = [("host_segmented_top2_int8", case_host_segmented),
             ("ondevice_top1_int8", case_ondevice),
             ("expert_footprint", case_footprint)]
    n = 0
    for name, fn in cases:
        try:
            ok, msg = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok, msg = False, f"{type(e).__name__}: {e}"
        print(f"{'PASS' if ok else 'FAIL'}  {name}  -- {msg}")
        n += ok
    print(f"\n{n}/{len(cases)} passed")
    return 0 if n == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
