#!/usr/bin/env python3
"""
test_gemma4_quant_types.py -- Q4_K / Q8_0 / Q6_K weights through the whole tiny Gemma-4 on the simulated chip.

  1. the GGUF block parsers (gguf_reader.parse_q4_k / parse_q8_0 / parse_q6_k) equal a plain per-element loop written
     from the llama.cpp block layouts, on random blocks, and QMat.spec() planes reproduce QMat.dense() exactly;
  2. the tiny model with EVERY weight matrix (attention, dense MLP, all experts) stored as one scheme, and a mixed one
     (attention Q4_K, MLP Q8_0, experts Q6_K), generates the same ids and top-5 logits on the chip as the same model
     with the dequantised matrices in plain FP32 (relocation, per-signature program templates and all).

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_gemma4_quant_types.py
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(ROOT, "toolchain"))

import gemma4_gguf  # noqa: E402
import gemma4_model  # noqa: E402
import gguf_reader  # noqa: E402
import quant_planes  # noqa: E402

MODEL = os.path.join(_HERE, "fixtures", "gemma4_tiny_model")
SIM, HAL = os.path.join(ROOT, "build", "rtv_sim"), os.path.join(ROOT, "toolchain", "hal_config.json")


# ---- 1. block parsers vs a plain loop ------------------------------------------------------------------------------
def f16(b):
    return float(np.frombuffer(bytes(b), np.float16)[0])


def ref_q8_0(blk):
    d = f16(blk[:2])
    return [d * int(np.int8(x)) for x in blk[2:34]]


def ref_q4_k(blk):
    d, dmin = f16(blk[0:2]), f16(blk[2:4])
    sc12, qs = [int(x) for x in blk[4:16]], blk[16:144]
    out = []
    for j in range(8):                                       # llama.cpp get_scale_min_k4
        if j < 4:
            s, m = sc12[j] & 63, sc12[j + 4] & 63
        else:
            s, m = (sc12[j + 4] & 15) | ((sc12[j - 4] >> 6) << 4), (sc12[j + 4] >> 4) | ((sc12[j] >> 6) << 4)
        chunk, hi = j // 2, j % 2
        for l in range(32):
            q = (int(qs[32 * chunk + l]) >> (4 * hi)) & 15
            out.append(d * s * q - dmin * m)
    return out


def ref_q6_k(blk):
    ql, qh = [int(x) for x in blk[0:128]], [int(x) for x in blk[128:192]]
    sc, d = [int(np.int8(x)) for x in blk[192:208]], f16(blk[208:210])
    out = [0.0] * 256
    for n in range(0, 256, 128):                              # llama.cpp dequantize_row_q6_K
        h = n // 128
        for l in range(32):
            isc = 8 * h + l // 16
            q1 = ((ql[64 * h + l] & 15) | (((qh[32 * h + l] >> 0) & 3) << 4)) - 32
            q2 = ((ql[64 * h + l + 32] & 15) | (((qh[32 * h + l] >> 2) & 3) << 4)) - 32
            q3 = ((ql[64 * h + l] >> 4) | (((qh[32 * h + l] >> 4) & 3) << 4)) - 32
            q4 = ((ql[64 * h + l + 32] >> 4) | (((qh[32 * h + l] >> 6) & 3) << 4)) - 32
            out[n + l] = d * sc[isc] * q1
            out[n + l + 32] = d * sc[isc + 2] * q2
            out[n + l + 64] = d * sc[isc + 4] * q3
            out[n + l + 96] = d * sc[isc + 6] * q4
    return out


def case_parsers(results):
    rng = np.random.default_rng(5)
    for typ, nb, ref, fix in (("Q8_0", 34, ref_q8_0, [(0, 2)]), ("Q4_K", 144, ref_q4_k, [(0, 2), (2, 4)]),
                              ("Q6_K", 210, ref_q6_k, [(208, 210)])):
        N, nblk = 3, 2
        raw = rng.integers(0, 256, size=(N, nblk, nb), dtype=np.uint8)
        for a, b in fix:                                      # finite fp16 scales
            raw[..., a:b] = np.frombuffer(np.array([0.01 * (a + 1)], np.float16).tobytes(), np.uint8)
        elems = gguf_reader.BLOCK[typ][0]
        want = np.array([[sum((ref(raw[n, k]) for k in range(nblk)), []) for n in range(N)]][0])   # (N, K)
        qm = gemma4_gguf.QMat(raw, typ)
        dense = qm.dense()                                    # [K, N]
        e1 = float(np.abs(dense.T - want).max() / np.abs(want).max())
        e2 = float(np.abs(quant_planes.dequant(qm.spec(), qm.K) - dense).max() / np.abs(dense).max())
        results.append((f"{typ}: parser == per-element loop; planes == dense", e1 < 1e-6 and e2 < 1e-6,
                        f"{elems * nblk} inputs x {N}: {e1:.1e}, {e2:.1e}"))


# ---- 2. the whole tiny model ---------------------------------------------------------------------------------------
class SynthQ:
    """A [K, N] FP32 matrix quantised to one scheme; behaves like gemma4_gguf.QMat for the model driver."""

    def __init__(self, w, kind):
        w = np.asarray(w, np.float64)
        self.K, self.N = w.shape
        self.kind = kind
        if kind == "q4k":
            g = w.reshape(self.K // 32, 32, self.N)
            lo, hi = g.min(1), g.max(1)
            s = np.maximum((hi - lo) / 15.0, 1e-9)
            q = np.clip(np.rint((g - lo[:, None]) / s[:, None]), 0, 15).reshape(self.K, self.N)
            self._spec = quant_planes.asym4(q, s, -lo, 32)
        elif kind == "q8_0":
            g = w.reshape(self.K // 32, 32, self.N)
            s = np.maximum(np.abs(g).max(1) / 127.0, 1e-9)
            self._spec = quant_planes.sym8(np.clip(np.rint(g / s[:, None]), -128, 127).reshape(self.K, self.N), s)
        else:
            g = w.reshape(self.K // 16, 16, self.N)
            s = np.maximum(np.abs(g).max(1) / 31.0, 1e-9)
            self._spec = quant_planes.sym6(np.clip(np.rint(g / s[:, None]) + 32, 0, 63).reshape(self.K, self.N), s)

    def spec(self):
        return self._spec

    def dense(self):
        return quant_planes.dequant(self._spec, self.K).astype(np.float32)


def quantise(src, plan):
    cfg, layers, embed, fnorm = src
    qlayers, dlayers = [], []
    for l in layers:
        q, d = dict(l), dict(l)
        for key, kind in plan["dense"].items():
            if l[key] is not None:
                q[key] = SynthQ(l[key], kind)
                d[key] = q[key].dense()
        q["experts"], d["experts"] = [], []
        for e in l["experts"]:
            qe = {k: SynthQ(v, plan["experts"]) for k, v in e.items()}
            q["experts"].append(qe)
            d["experts"].append({k: v.dense() for k, v in qe.items()})
        qlayers.append(q)
        dlayers.append(d)
    return (cfg, qlayers, embed, fnorm), (cfg, dlayers, embed, fnorm)


def case_model(results):
    ref = json.load(open(os.path.join(_HERE, "fixtures", "gemma4_tiny_hf_reference.json")))
    plans = {"q4k": dict(dense={k: "q4k" for k in ("wq", "wk", "wv", "wo", "mlp_gate", "mlp_up", "mlp_down")}, experts="q4k"),
             "q8_0": dict(dense={k: "q8_0" for k in ("wq", "wk", "wv", "wo", "mlp_gate", "mlp_up", "mlp_down")}, experts="q8_0"),
             "q6k": dict(dense={k: "q6k" for k in ("wq", "wk", "wv", "wo", "mlp_gate", "mlp_up", "mlp_down")}, experts="q6k"),
             "mixed": dict(dense=dict(wq="q4k", wk="q4k", wv="q4k", wo="q4k", mlp_gate="q8_0", mlp_up="q8_0", mlp_down="q6k"),
                           experts="q6k")}
    base = gemma4_model.load_gemma4_hf(MODEL)
    for name, plan in plans.items():
        qsrc, dsrc = quantise(base, plan)
        ids_q, st_q = gemma4_model.generate_gemma4(qsrc, ref["prompt"], 3, SIM, HAL)
        ids_d, st_d = gemma4_model.generate_gemma4(dsrc, ref["prompt"], 3, SIM, HAL)
        worst = max(abs(a - b) for x, y in zip(st_q, st_d) for a, b in zip(x[1], y[1]))
        same = ids_q == ids_d and all(x[0] == y[0] for x, y in zip(st_q, st_d))
        results.append((f"tiny model, {name}: ids, top-5 and logits == dequantised FP32 model", same and worst < 1e-4,
                        f"max |delta logit| {worst:.2e}"))


def main():
    results = []
    case_parsers(results)
    case_model(results)
    n = 0
    for name, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {msg}" if msg else ""))
        n += ok
    print(f"\n{n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
