#!/usr/bin/env python3
"""
test_moe_topk_ondevice.py -- top-k MoE routing decided entirely on the device with the existing ISA.

One fixed instruction stream: RMSNorm -> router GEMM -> k rounds of REDUCE_ARGMAX with an INDIRECT_ADDR store that
masks each winner -> indirect loads of the chosen logits and per-expert scales -> softmax among the k -> weights
broadcast to tiles by a GEMM -> k expert FFNs fetched with INDIRECT_ADDR weight loads, scaled and summed.
For several random inputs the output must equal what the host computes (float64 softmax, stable top-k,
p[idx] / sum(p[idx]) * scale[idx], then the weighted sum of the dequantised INT4 experts), for k = 8 and k = 3, and the
chosen expert sets must differ between inputs (the stream really routes).

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_moe_topk_ondevice.py
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
from test_moe_ffn import (fresh_comp, M,                       # noqa: E402
                          X_ADDR, ROUTER_ADDR, STAGING)
import test_moe_quant as Q                                     # noqa: E402
from test_persistent_session import Session                    # noqa: E402

GAMMA_ADDR, SCALE_ADDR, EPS = 0x4000, 0x5000, 1e-6   # the 128-expert router occupies 0x8000..0xA000
HIDDEN, INTER = 32, 48


def make_model(n_experts, seed):
    """router [HIDDEN, n_experts] (E_pad == n_experts, as in Gemma) and n_experts random experts."""
    rng = np.random.default_rng(seed)
    router = rng.normal(size=(HIDDEN, n_experts)).astype(np.float32)
    experts = [dict(w_gate=(rng.normal(size=(HIDDEN, INTER)) * 0.3).astype(np.float32),
                    w_up=(rng.normal(size=(HIDDEN, INTER)) * 0.3).astype(np.float32),
                    w_down=(rng.normal(size=(INTER, HIDDEN)) * 0.3).astype(np.float32)) for _ in range(n_experts)]
    return router, experts


def case(top_k, E=16):
    Q.SCHEME["mode"] = "int4"
    router, experts = make_model(E, 21 + top_k)
    image, addrs, stride, _ = Q.layout_int8(HIDDEN, INTER, router, experts)
    gamma = (np.random.default_rng(2).normal(size=HIDDEN) * 0.5 + 1).astype(np.float32)
    gt = np.tile(gamma, (M, 1)).astype(np.float32)
    image[GAMMA_ADDR:GAMMA_ADDR + gt.nbytes] = gt.tobytes()
    scale = np.random.default_rng(3).uniform(0.5, 1.5, size=E).astype(np.float32)
    image[SCALE_ADDR:SCALE_ADDR + scale.nbytes] = scale.tobytes()

    c = fresh_comp()
    c.rms_eps = EPS
    Q.set_flags(c)
    norm = c.alloc_staging(M * HIDDEN * 4)
    tvm_backend._emit_rmsnorm_row_fp32(
        c, HIDDEN, x_addr=X_ADDR, gamma_addr=GAMMA_ADDR, out_addr=norm,
        scratch_addr=c.alloc_staging(4), stage_addr=c.alloc_staging(M * _padded(HIDDEN) * 4))
    logits = c.alloc_staging(M * E * 4)
    tvm_backend._emit_matmul_fp32(c, M, E, HIDDEN, norm, ROUTER_ADDR, logits)
    out_addr = c.alloc_staging(M * HIDDEN * 4)
    tvm_backend._emit_moe_topk_ondevice_fp32(c, HIDDEN, INTER, norm, logits, E, top_k, addrs[0], stride,
                                             SCALE_ADDR, out_addr)

    rng = np.random.default_rng(5)
    sets, errs = [], []
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
            for t in range(10):
                x = np.zeros((M, HIDDEN), np.float32)
                x[0] = rng.normal(size=HIDDEN).astype(np.float32)
                f = os.path.join(wd, f"x{t}.bin")
                open(f, "wb").write(x.tobytes())
                assert sess.cmd(f"WRITE {X_ADDR} {f}").startswith("OK")
                o = os.path.join(wd, f"o{t}.bin")
                r = sess.cmd(f"RUN {prog} {out_addr}:{M * HIDDEN * 4}:{o}")
                assert r.startswith("OK"), r
                got = np.frombuffer(open(o, "rb").read(), np.float32).reshape(M, HIDDEN)[0]
                xr = x[0].astype(np.float64)
                nrm = xr / np.sqrt(np.mean(xr * xr) + EPS) * gamma
                lg = nrm @ router[:, :E].astype(np.float64)
                p = np.exp(lg - lg.max())
                p /= p.sum()
                idx = np.argsort(-p, kind="stable")[:top_k]
                w = p[idx] / p[idx].sum() * scale[idx]
                ref = sum(wj * Q.ffn_ref(nrm, experts[e]) for wj, e in zip(w, idx))
                errs.append(float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12)))
                sets.append(tuple(sorted(int(i) for i in idx)))
        finally:
            sess.close()
    ok = max(errs) < 1e-3 and len(set(sets)) >= 3
    return ok, (f"max rel err {max(errs):.2e}, {len(set(sets))} distinct expert sets in 10 inputs, "
                f"one {len(c.instruction_stream) // 16}-instruction stream")


def main():
    n = 0
    ks = ((8, 16), (3, 16), (8, 128))
    for k, e in ks:
        try:
            ok, msg = case(k, e)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok, msg = False, f"{type(e).__name__}: {e}"
        print(f"{'PASS' if ok else 'FAIL'}  top-{k} of {e} on the device  -- {msg}")
        n += ok
    print(f"\n{n}/{len(ks)} passed")
    return 0 if n == len(ks) else 1


if __name__ == "__main__":
    sys.exit(main())
