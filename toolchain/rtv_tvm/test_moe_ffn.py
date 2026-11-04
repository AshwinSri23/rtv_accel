#!/usr/bin/env python3
"""
test_moe_ffn.py -- a mixture-of-experts FFN with HOST-SEGMENTED routing, run on
a persistent simulator session (TODO Phase 5.1 option (a), made concrete).

The ISA has no data-dependent addressing, and the expert choice depends on
router logits that the device itself computes, so each token is TWO device
programs with a host step between them:

  1. device: router logits = x @ W_router
  2. host:   softmax, top-k, renormalise  (REDUCE_MAX has no index, same
             limitation that keeps sampling on the host)
  3. device: sum_j w_j * SwiGLU_{expert_j}(x), with the chosen experts'
             weight addresses baked in as literals

All expert weights stay resident in one `rtv_sim --serve` session; nothing is
reloaded between tokens. Checked against a numpy MoE (all experts computed,
masked by the same top-k), which is option (b) used as the oracle, and the test
confirms two different routings give device programs that differ ONLY in
LOAD_TILE/STORE_TILE addresses (so schedule relocation's address patching
applies to expert layers, TODO 5.4 step 3).

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_moe_ffn.py
"""
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from compiler_state import NPUCompiler, DType  # noqa: E402
import tvm_backend                            # noqa: E402
from rtv_isa import Opcode, decode            # noqa: E402
from test_persistent_session import Session   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")
M = 16
X_ADDR, ROUTER_ADDR, EXPERT_BASE, STAGING = 0x1000, 0x8000, 0x20000, 0x400000


def silu(v):
    return v / (1.0 + np.exp(-v))


def build_model(hidden, inter, n_experts, seed=0):
    rng = np.random.default_rng(seed)
    router = np.zeros((hidden, M), dtype=np.float32)       # padded to one 16-wide tile
    router[:, :n_experts] = rng.normal(size=(hidden, n_experts)).astype(np.float32)
    experts = []
    for _ in range(n_experts):
        experts.append(dict(
            w_gate=(rng.normal(size=(hidden, inter)) * 0.3).astype(np.float32),
            w_up=(rng.normal(size=(hidden, inter)) * 0.3).astype(np.float32),
            w_down=(rng.normal(size=(inter, hidden)) * 0.3).astype(np.float32)))
    return router, experts


def layout(hidden, inter, n_experts, router, experts):
    """Place router + experts in a DRAM image; return (image, per-expert addrs)."""
    ex_bytes = (2 * hidden * inter + inter * hidden) * 4
    stride = (ex_bytes + 0xFFF) & ~0xFFF
    image = bytearray(STAGING + 0x100000)
    image[ROUTER_ADDR:ROUTER_ADDR + router.nbytes] = router.tobytes()
    addrs = []
    for e, w in enumerate(experts):
        base = EXPERT_BASE + e * stride
        a = dict(w_gate=base, w_up=base + hidden * inter * 4, w_down=base + 2 * hidden * inter * 4)
        for k in ("w_gate", "w_up", "w_down"):
            image[a[k]:a[k] + w[k].nbytes] = w[k].tobytes()
        addrs.append(a)
    return image, addrs


def fresh_comp():
    c = NPUCompiler(HAL)
    c.dtype = DType.FP32
    c.instruction_stream = bytearray()
    c._dram_cursor = STAGING
    return c


def run_prog(sess, wd, comp, tag, dump=None):
    for i, (addr, data) in enumerate(comp._const_regions.items()):
        f = os.path.join(wd, f"{tag}_c{i}.bin")
        open(f, "wb").write(bytes(data))
        assert sess.cmd(f"WRITE {addr} {f}").startswith("OK")
    prog = os.path.join(wd, f"{tag}.bin")
    open(prog, "wb").write(bytes(comp.instruction_stream))
    spec, out = "", None
    if dump:
        out = os.path.join(wd, f"{tag}.out")
        spec = f"{dump[0]}:{dump[1]}:{out}"
    r = sess.cmd(f"RUN {prog} {spec}")
    assert r.startswith("OK"), r
    return open(out, "rb").read() if out else None


def opcode_skeleton(stream):
    """Every instruction with the DRAM address of LOAD/STORE zeroed out."""
    sk = []
    for i in range(0, len(stream), 16):
        f = decode(int.from_bytes(bytes(stream[i:i + 16]), "big"))
        if f["opcode"] in (Opcode.LOAD_TILE, Opcode.STORE_TILE):
            f["dram_addr"] = 0
        sk.append(tuple(sorted((k, str(v)) for k, v in f.items())))
    return sk


def main():
    hidden, inter, n_experts, top_k, n_tokens = 32, 48, 6, 2, 6
    router, experts = build_model(hidden, inter, n_experts)
    image, addrs = layout(hidden, inter, n_experts, router, experts)
    rng = np.random.default_rng(1)
    chosen_sets, skeletons, errs = [], [], []

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

                # -- device segment 1: router logits
                c1 = fresh_comp()
                logits_addr = c1.alloc_staging(M * M * 4)
                tvm_backend._emit_matmul_fp32(c1, M, M, hidden, X_ADDR, ROUTER_ADDR, logits_addr)
                raw = run_prog(sess, wd, c1, f"r{t}", dump=(logits_addr, M * M * 4))
                logits = np.frombuffer(raw, dtype=np.float32).reshape(M, M)[0, :n_experts]

                # -- host: softmax, top-k, renormalise
                p = np.exp(logits - logits.max())
                p /= p.sum()
                idx = np.argsort(-p)[:top_k]
                w = (p[idx] / p[idx].sum()).astype(np.float32)
                chosen_sets.append(tuple(sorted(int(i) for i in idx)))

                # -- device segment 2: the chosen experts
                c2 = fresh_comp()
                out_addr = c2.alloc_staging(M * hidden * 4)
                tvm_backend._emit_moe_ffn_fp32(c2, hidden, inter, X_ADDR,
                                               [addrs[i] for i in idx], [float(v) for v in w],
                                               out_addr)
                raw = run_prog(sess, wd, c2, f"m{t}", dump=(out_addr, M * hidden * 4))
                got = np.frombuffer(raw, dtype=np.float32).reshape(M, hidden)[0]
                skeletons.append((tuple(int(i) for i in idx), opcode_skeleton(c2.instruction_stream)))

                # -- reference: every expert computed, masked to the top-k
                xr = x[0].astype(np.float64)
                ref = np.zeros(hidden)
                for wj, i in zip(w, idx):
                    e = experts[i]
                    h = silu(xr @ e["w_gate"].astype(np.float64)) * (xr @ e["w_up"].astype(np.float64))
                    ref += float(wj) * (h @ e["w_down"].astype(np.float64))
                err = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12))
                errs.append(err)
                print(f"  token {t}: experts {sorted(int(i) for i in idx)} weights "
                      f"{np.round(np.sort(w)[::-1], 3).tolist()}  rel err {err:.2e}")
        finally:
            sess.close()

    results = []
    results.append(("matches_numpy_moe", max(errs) < 2e-3, f"max rel err {max(errs):.2e}"))
    results.append(("routing_actually_varies", len(set(chosen_sets)) >= 2,
                    f"{len(set(chosen_sets))} distinct expert sets over {n_tokens} tokens"))
    # programs for different routings must differ only in LOAD/STORE addresses
    base_idx, base_sk = skeletons[0]
    other = next(((i, s) for i, s in skeletons if i != base_idx), None)
    ok = other is not None and len(other[1]) == len(base_sk) and other[1] == base_sk
    results.append(("different_routing_differs_only_in_dma_addresses", ok,
                    "no differently-routed token found" if other is None
                    else f"{len(base_sk)} instrs, same skeleton" if ok else "streams differ in more than addresses"))
    n = 0
    for name, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  -- {msg}")
        n += ok
    print(f"\n{n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
