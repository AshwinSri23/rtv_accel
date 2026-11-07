#!/usr/bin/env python3
"""
test_moe_ondevice_planar.py -- on-device top-1 MoE routing (ISA v1.7, INDIRECT_ADDR) with experts stored as
quant_planes weights (Q4_K / Q8_0 / Q6_K style: INT4W planes plus a group-sum correction).

One fixed instruction stream picks the expert on the device (argmax of the router logits) and every per-expert load
-- each plane's packed weights and scales, and the FP32 correction matrix -- is offset by expert * stride. The
group-sum matrix depends only on K, so it is shared and loaded without the offset. For several random inputs the
stream's output must equal x + SwiGLU(expert chosen by numpy) computed on the exactly dequantised weights.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_moe_ondevice_planar.py
"""
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import quant_planes                                            # noqa: E402
import tvm_backend                                             # noqa: E402
from compiler_state import _padded                             # noqa: E402
from test_moe_ffn import (build_model, fresh_comp, silu, M,    # noqa: E402
                          X_ADDR, ROUTER_ADDR, EXPERT_BASE, STAGING)
from test_persistent_session import Session                    # noqa: E402

GAMMA_ADDR, EPS = 0x4000, 1e-6
GSUM_BASE = 0x10000                      # shared group-sum matrices, below EXPERT_BASE
HIDDEN, INTER, E = 32, 64, 4
NAMES = ("w_gate", "w_up", "w_down")
SHAPE = {"w_gate": (HIDDEN, INTER), "w_up": (HIDDEN, INTER), "w_down": (INTER, HIDDEN)}   # (K, N)


def make(kind, rng, K, N):
    if kind == "q4k":
        return quant_planes.asym4(rng.integers(0, 16, size=(K, N)), rng.uniform(0.01, 0.1, size=(K // 32, N)),
                                  rng.uniform(-0.2, 0.3, size=(K // 32, N)), 32)
    if kind == "q8_0":
        return quant_planes.sym8(rng.integers(-128, 128, size=(K, N)), rng.uniform(0.0005, 0.004, size=(K // 32, N)))
    return quant_planes.sym6(rng.integers(0, 64, size=(K, N)), rng.integers(-40, 40, size=(K // 16, N)) * np.float32(0.0006))


def case(kind):
    rng = np.random.default_rng(hash(kind) & 0xFFFF)
    router, _ = build_model(HIDDEN, INTER, E, seed=11)
    specs = [{n: make(kind, rng, *SHAPE[n]) for n in NAMES} for _ in range(E)]

    def blobs(spec, K):                  # an expert's byte arrays for one matrix, in a fixed order
        out = [tvm_backend.int4_weight_blob(v, s).tobytes() for v, s, g in spec["planes"]]
        if spec["corr"] is not None:
            out.append(quant_planes.corr_matrix(spec, K).tobytes())
        return out

    # One layout for every expert: the same component offsets from the expert's base, one stride between experts.
    def align(n):
        return (n + 0xFFF) & ~0xFFF
    comp_sizes = {n: [align(len(b)) for b in blobs(specs[0][n], SHAPE[n][0])] for n in NAMES}
    per = sum(sum(v) for v in comp_sizes.values())
    stride = align(per)
    image = bytearray(STAGING + 0x100000)
    image[ROUTER_ADDR:ROUTER_ADDR + router.nbytes] = router.tobytes()
    gamma = (np.random.default_rng(2).normal(size=HIDDEN) * 0.5 + 1).astype(np.float32)
    gt = np.tile(gamma, (M, 1)).astype(np.float32)
    image[GAMMA_ADDR:GAMMA_ADDR + gt.nbytes] = gt.tobytes()
    comp_addr = {}                                           # name -> [component addresses for expert 0]
    for e in range(E):
        cur = EXPERT_BASE + e * stride
        for n in NAMES:
            addrs = []
            for b, size in zip(blobs(specs[e][n], SHAPE[n][0]), comp_sizes[n]):
                image[cur:cur + len(b)] = b
                addrs.append(cur)
                cur += size
            if e == 0:
                comp_addr[n] = addrs

    c = fresh_comp()
    c.rms_eps = EPS
    c.qweights, c.group_sums = {}, {}
    gcur = GSUM_BASE
    for n in NAMES:
        K = SHAPE[n][0]
        s0 = specs[0][n]
        planes = [(a, g) for a, (_, _, g) in zip(comp_addr[n], s0["planes"])]
        corr = None
        if s0["corr"] is not None:
            g = s0["corr"][1]
            corr = (comp_addr[n][len(planes)], g)
            if (K, g) not in c.group_sums:
                blob, _ = tvm_backend.group_sum_blob(K, g)
                data = blob.tobytes()
                image[gcur:gcur + len(data)] = data
                c.group_sums[(K, g)] = gcur
                gcur = align(gcur + len(data))
        assert gcur < EXPERT_BASE
        c.qweights[planes[0][0]] = dict(planes=planes, corr=corr)
    expert0 = {n: comp_addr[n][0] for n in NAMES}

    norm = c.alloc_staging(M * HIDDEN * 4)
    tvm_backend._emit_rmsnorm_row_fp32(
        c, HIDDEN, x_addr=X_ADDR, gamma_addr=GAMMA_ADDR, out_addr=norm,
        scratch_addr=c.alloc_staging(4), stage_addr=c.alloc_staging(M * _padded(HIDDEN) * 4))
    out_addr = c.alloc_staging(M * HIDDEN * 4)
    tvm_backend._emit_moe_top1_ondevice_fp32(c, HIDDEN, INTER, X_ADDR, norm, ROUTER_ADDR, E, expert0, stride, out_addr)

    deq = [{n: quant_planes.dequant(specs[e][n], SHAPE[n][0]) for n in NAMES} for e in range(E)]
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
            xr_rng = np.random.default_rng(5)
            for t in range(10):
                x = np.zeros((M, HIDDEN), np.float32)
                x[0] = xr_rng.normal(size=HIDDEN).astype(np.float32)
                f = os.path.join(wd, f"x{t}.bin")
                open(f, "wb").write(x.tobytes())
                assert sess.cmd(f"WRITE {X_ADDR} {f}").startswith("OK")
                o = os.path.join(wd, f"o{t}.bin")
                r = sess.cmd(f"RUN {prog} {out_addr}:{M * HIDDEN * 4}:{o}")
                assert r.startswith("OK"), r
                got = np.frombuffer(open(o, "rb").read(), np.float32).reshape(M, HIDDEN)[0]
                xr = x[0].astype(np.float64)
                nrm = xr / np.sqrt(np.mean(xr * xr) + EPS) * gamma
                e = int(np.argmax(nrm @ router[:, :E].astype(np.float64)))
                d = deq[e]
                ref = xr + (silu(nrm @ d["w_gate"]) * (nrm @ d["w_up"])) @ d["w_down"]
                errs.append(float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12)))
                chosen.append(e)
        finally:
            sess.close()
    ok = max(errs) < 1e-3 and len(set(chosen)) >= 2
    return ok, f"max rel err {max(errs):.2e}, experts chosen {sorted(set(chosen))} by one {len(c.instruction_stream) // 16}-instruction stream"


def main():
    n = 0
    kinds = ("q8_0", "q6k", "q4k")
    for kind in kinds:
        try:
            ok, msg = case(kind)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok, msg = False, f"{type(e).__name__}: {e}"
        print(f"{'PASS' if ok else 'FAIL'}  {kind} experts, on-device top-1  -- {msg}")
        n += ok
    print(f"\n{n}/{len(kinds)} passed")
    return 0 if n == len(kinds) else 1


if __name__ == "__main__":
    sys.exit(main())
