#!/usr/bin/env python3
"""
test_moe_block.py -- a complete pre-norm MoE transformer FFN block on a persistent
session: out = x + SharedExpert(n) + sum_j w_j * Expert_{e_j}(n), n = RMSNorm(x).

Extends test_moe_ffn.py (host-segmented routing) with the pieces a real MoE layer
has around the experts: the RMSNorm before the router, an always-on SHARED expert,
and the residual connection. Per token:

  device 1: n = RMSNorm(x, gamma);  logits = n @ W_router        (n stays resident)
  host:     softmax, top-k, renormalise
  device 2: out = x + shared(n) + sum_j w_j * expert_{e_j}(n)

Checked against a numpy block (all experts computed, masked to the same top-k).

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_moe_block.py
"""
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import tvm_backend                                            # noqa: E402
from test_moe_ffn import (build_model, layout, fresh_comp, run_prog, silu,  # noqa: E402
                          M, X_ADDR, ROUTER_ADDR)
from test_persistent_session import Session                    # noqa: E402

GAMMA_ADDR, NORM_ADDR, SHARED_BASE = 0x4000, 0x5000, 0x300000


def rmsnorm_ref(v, gamma):
    return v / np.sqrt(np.mean(v * v)) * gamma


def main():
    hidden, inter, n_experts, top_k, n_tokens = 32, 48, 6, 2, 5
    router, experts = build_model(hidden, inter, n_experts, seed=3)
    image, addrs = layout(hidden, inter, n_experts, router, experts)
    rng = np.random.default_rng(9)
    gamma = (rng.normal(size=hidden) * 0.5 + 1.0).astype(np.float32)
    shared = dict(w_gate=(rng.normal(size=(hidden, inter)) * 0.3).astype(np.float32),
                  w_up=(rng.normal(size=(hidden, inter)) * 0.3).astype(np.float32),
                  w_down=(rng.normal(size=(inter, hidden)) * 0.3).astype(np.float32))
    sa = dict(w_gate=SHARED_BASE, w_up=SHARED_BASE + hidden * inter * 4,
              w_down=SHARED_BASE + 2 * hidden * inter * 4)
    for k in ("w_gate", "w_up", "w_down"):
        image[sa[k]:sa[k] + shared[k].nbytes] = shared[k].tobytes()
    gt = np.tile(gamma, (M, 1)).astype(np.float32)
    image[GAMMA_ADDR:GAMMA_ADDR + gt.nbytes] = gt.tobytes()

    errs, sets = [], []
    with tempfile.TemporaryDirectory() as wd:
        img = os.path.join(wd, "img")
        open(img, "wb").write(bytes(image))
        sess = Session(dram_mb=len(image) // (1024 * 1024) + 2, image=img)
        try:
            for t in range(n_tokens):
                x = np.zeros((M, hidden), dtype=np.float32)
                x[0] = rng.normal(size=hidden).astype(np.float32)
                f = os.path.join(wd, f"x{t}.bin")
                open(f, "wb").write(x.tobytes())
                assert sess.cmd(f"WRITE {X_ADDR} {f}").startswith("OK")

                c1 = fresh_comp()
                tvm_backend._emit_rmsnorm_row_fp32(
                    c1, hidden, x_addr=X_ADDR, gamma_addr=GAMMA_ADDR, out_addr=NORM_ADDR,
                    scratch_addr=c1.alloc_staging(4),
                    stage_addr=c1.alloc_staging(M * tvm_backend._padded(hidden) * 4))
                logits_addr = c1.alloc_staging(M * M * 4)
                tvm_backend._emit_matmul_fp32(c1, M, M, hidden, NORM_ADDR, ROUTER_ADDR, logits_addr)
                raw = run_prog(sess, wd, c1, f"r{t}", dump=(logits_addr, M * M * 4))
                logits = np.frombuffer(raw, dtype=np.float32).reshape(M, M)[0, :n_experts]

                p = np.exp(logits - logits.max())
                p /= p.sum()
                idx = np.argsort(-p)[:top_k]
                w = (p[idx] / p[idx].sum()).astype(np.float32)
                sets.append(tuple(sorted(int(i) for i in idx)))

                c2 = fresh_comp()
                out_addr = c2.alloc_staging(M * hidden * 4)
                tvm_backend._emit_moe_block_fp32(
                    c2, hidden, inter, X_ADDR, NORM_ADDR, [addrs[i] for i in idx],
                    [float(v) for v in w], out_addr, shared_expert=sa)
                raw = run_prog(sess, wd, c2, f"m{t}", dump=(out_addr, M * hidden * 4))
                got = np.frombuffer(raw, dtype=np.float32).reshape(M, hidden)[0]

                xr = x[0].astype(np.float64)
                n = rmsnorm_ref(xr, gamma.astype(np.float64))
                ref = xr.copy()

                def ffn(e):
                    h = silu(n @ e["w_gate"].astype(np.float64)) * (n @ e["w_up"].astype(np.float64))
                    return h @ e["w_down"].astype(np.float64)

                ref += ffn(shared)
                for wj, i in zip(w, idx):
                    ref += float(wj) * ffn(experts[i])
                err = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12))
                errs.append(err)
                print(f"  token {t}: experts {sorted(int(i) for i in idx)}  rel err {err:.2e}")
        finally:
            sess.close()

    results = [("matches_numpy_moe_block", max(errs) < 2e-3, f"max rel err {max(errs):.2e}"),
               ("routing_varies", len(set(sets)) >= 2, f"{len(set(sets))} distinct expert sets")]
    n = 0
    for name, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  -- {msg}")
        n += ok
    print(f"\n{n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
