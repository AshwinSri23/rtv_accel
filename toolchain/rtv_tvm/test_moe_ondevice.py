#!/usr/bin/env python3
"""
test_moe_ondevice.py -- top-1 MoE routing decided ON THE DEVICE (ISA v1.7).

No host step, no recompilation: ONE instruction stream is compiled once and run on
several inputs; the expert each input uses is chosen by the chip itself:

    n = RMSNorm(x);  logits = n @ W_router
    REDUCE_ARGMAX  -> e * expert_stride  (a uint32 in SRAM)
    expert weights are fetched with INDIRECT_ADDR loads: dram_addr + that word
    out = x + SwiGLU_e(n)

This is the alternative to test_moe_ffn.py's host-segmented routing: it removes the
host round trip at the price of two ISA additions (REDUCE_ARGMAX, INDIRECT_ADDR), still
with no branches and no loops. Checked against a numpy top-1 MoE; the test also
verifies different inputs really do pick different experts using the SAME bytes.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_moe_ondevice.py
"""
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import tvm_backend                                            # noqa: E402
from compiler_state import _padded                            # noqa: E402
from test_moe_ffn import (build_model, layout, fresh_comp, silu, M,   # noqa: E402
                          X_ADDR, ROUTER_ADDR)
from test_persistent_session import Session                    # noqa: E402

GAMMA_ADDR, EPS = 0x4000, 1e-6


def main():
    hidden, inter, n_experts, n_inputs = 32, 48, 4, 8
    router, experts = build_model(hidden, inter, n_experts, seed=11)
    image, addrs = layout(hidden, inter, n_experts, router, experts)
    gamma = (np.random.default_rng(2).normal(size=hidden) * 0.5 + 1).astype(np.float32)
    gt = np.tile(gamma, (M, 1)).astype(np.float32)
    image[GAMMA_ADDR:GAMMA_ADDR + gt.nbytes] = gt.tobytes()
    stride = addrs[1]["w_gate"] - addrs[0]["w_gate"]
    assert all(addrs[e]["w_up"] - addrs[e]["w_gate"] == addrs[0]["w_up"] - addrs[0]["w_gate"]
               and addrs[e]["w_gate"] - addrs[0]["w_gate"] == e * stride for e in range(n_experts))

    # --- compile ONCE
    c = fresh_comp()
    c.rms_eps = EPS
    norm = c.alloc_staging(M * hidden * 4)
    tvm_backend._emit_rmsnorm_row_fp32(
        c, hidden, x_addr=X_ADDR, gamma_addr=GAMMA_ADDR, out_addr=norm,
        scratch_addr=c.alloc_staging(4), stage_addr=c.alloc_staging(M * _padded(hidden) * 4))
    out_addr = c.alloc_staging(M * hidden * 4)
    tvm_backend._emit_moe_top1_ondevice_fp32(c, hidden, inter, X_ADDR, norm, ROUTER_ADDR,
                                             n_experts, addrs[0], stride, out_addr)
    stream = bytes(c.instruction_stream)
    n_instr = len(stream) // 16

    rng = np.random.default_rng(5)
    chosen, errs = [], []
    with tempfile.TemporaryDirectory() as wd:
        img = os.path.join(wd, "img")
        open(img, "wb").write(bytes(image))
        prog = os.path.join(wd, "p.bin")
        open(prog, "wb").write(stream)
        sess = Session(dram_mb=len(image) // (1024 * 1024) + 2, image=img)
        try:
            for addr, data in c._const_regions.items():          # compile-time constants, once
                f = os.path.join(wd, f"c{addr}.bin")
                open(f, "wb").write(bytes(data))
                assert sess.cmd(f"WRITE {addr} {f}").startswith("OK")
            for t in range(n_inputs):
                x = np.zeros((M, hidden), np.float32)
                x[0] = rng.normal(size=hidden).astype(np.float32)
                f = os.path.join(wd, f"x{t}.bin")
                open(f, "wb").write(x.tobytes())
                assert sess.cmd(f"WRITE {X_ADDR} {f}").startswith("OK")
                o = os.path.join(wd, f"o{t}.bin")
                r = sess.cmd(f"RUN {prog} {out_addr}:{M * hidden * 4}:{o}")   # SAME program every time
                assert r.startswith("OK"), r
                got = np.frombuffer(open(o, "rb").read(), np.float32).reshape(M, hidden)[0]

                xr = x[0].astype(np.float64)
                n = xr / np.sqrt(np.mean(xr * xr) + EPS) * gamma
                logits = n @ router[:, :n_experts].astype(np.float64)
                e = int(np.argmax(logits))
                ex = experts[e]
                ref = xr + (silu(n @ ex["w_gate"].astype(np.float64))
                            * (n @ ex["w_up"].astype(np.float64))) @ ex["w_down"].astype(np.float64)
                errs.append(float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12)))
                chosen.append(e)
                print(f"  input {t}: expert {e}  rel err {errs[-1]:.2e}")
        finally:
            sess.close()

    results = [("matches_numpy_top1_moe", max(errs) < 1e-3, f"max rel err {max(errs):.2e}"),
               ("same_stream_picks_different_experts", len(set(chosen)) >= 2,
                f"{sorted(set(chosen))} chosen by one {n_instr}-instruction program")]
    n = 0
    for name, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  -- {msg}")
        n += ok
    print(f"\n{n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
