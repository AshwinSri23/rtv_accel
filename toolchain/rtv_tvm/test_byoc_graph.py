#!/usr/bin/env python3
"""
test_byoc_graph.py -- the graph-driven (BYOC) compile path, promoted from "tests only"
to something that lowers a real fragment of a decoder layer.

Graph: out = x + SwiGLU_FFN(RMSNorm(x, gamma))   (the FFN half of a Qwen3 layer).

  1. Coverage: the RTV-Accel patterns claim EVERY op (rmsnorm, swiglu_ffn, add), no
     operator is left over.
  2. Acceptance bar from TODO 5.4 step 4: the instruction stream emitted by
     byoc_graph.emit_graph is BYTE-IDENTICAL to the one the hand-driven emitters
     (_emit_rmsnorm_row_fp32, _emit_swiglu_ffn_fp32, _emit_residual_add_fp32) produce.
  3. Numerics: run on rtv_sim, compare to numpy.
  4. A fuller attention-layer graph (reshape/permute/batched matmul/softmax) is
     partitioned and its coverage MEASURED and printed: what the patterns leave
     unclaimed is exactly what still needs config-aware emitters.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_byoc_graph.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from tvm.script import ir as I, relax as R      # noqa: E402
from compiler_state import NPUCompiler, DType, _padded  # noqa: E402
import tvm_backend                               # noqa: E402
import byoc_graph                                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")
M, HID, INTER, EPS = 16, 32, 48, 1e-5

tvm_backend.register()


@I.ir_module
class FFNHalf:
    @R.function
    def main(x: R.Tensor((1, 32), "float32"), gamma: R.Tensor((32,), "float32"),
             w_gate: R.Tensor((32, 48), "float32"), w_up: R.Tensor((32, 48), "float32"),
             w_down: R.Tensor((48, 32), "float32")) -> R.Tensor((1, 32), "float32"):
        with R.dataflow():
            n = R.nn.rms_norm(x, gamma, axes=[-1], epsilon=1e-5)
            gate = R.matmul(n, w_gate)
            up = R.matmul(n, w_up)
            act = R.nn.silu(gate)
            inter = R.multiply(act, up)
            y = R.matmul(inter, w_down)
            out = R.add(x, y)
            R.output(out)
        return out


@I.ir_module
class AttnLayer:
    """A full decoder layer written with standard Relax ops (no RoPE, no KV-cache)."""
    @R.function
    def main(x: R.Tensor((16, 32), "float32"), g1: R.Tensor((32,), "float32"),
             wq: R.Tensor((32, 32), "float32"), wk: R.Tensor((32, 32), "float32"),
             wv: R.Tensor((32, 32), "float32"), wo: R.Tensor((32, 32), "float32"),
             g2: R.Tensor((32,), "float32"), w_gate: R.Tensor((32, 48), "float32"),
             w_up: R.Tensor((32, 48), "float32"), w_down: R.Tensor((48, 32), "float32")
             ) -> R.Tensor((16, 32), "float32"):
        with R.dataflow():
            n1 = R.nn.rms_norm(x, g1, axes=[-1], epsilon=1e-6)
            q = R.reshape(R.matmul(n1, wq), (16, 2, 16))
            k = R.reshape(R.matmul(n1, wk), (16, 2, 16))
            v = R.reshape(R.matmul(n1, wv), (16, 2, 16))
            qh = R.permute_dims(q, [1, 0, 2])
            kh = R.permute_dims(k, [1, 2, 0])
            vh = R.permute_dims(v, [1, 0, 2])
            scores = R.matmul(qh, kh)
            probs = R.nn.softmax(scores, axis=-1)
            ctx = R.matmul(probs, vh)
            merged = R.reshape(R.permute_dims(ctx, [1, 0, 2]), (16, 32))
            attn = R.matmul(merged, wo)
            r1 = R.add(x, attn)
            n2 = R.nn.rms_norm(r1, g2, axes=[-1], epsilon=1e-6)
            gate = R.matmul(n2, w_gate)
            up = R.matmul(n2, w_up)
            ff = R.matmul(R.multiply(R.nn.silu(gate), up), w_down)
            out = R.add(r1, ff)
            R.output(out)
        return out


def silu(v):
    return v / (1.0 + np.exp(-v))


def new_comp(base):
    c = NPUCompiler(HAL)
    c.dtype = DType.FP32
    c.instruction_stream = bytearray()
    c._dram_cursor = base
    c.rms_eps = EPS
    return c


def case_ffn_half():
    rng = np.random.default_rng(4)
    x = rng.normal(size=HID).astype(np.float32)
    gamma = (rng.normal(size=HID) * 0.5 + 1).astype(np.float32)
    wg = (rng.normal(size=(HID, INTER)) * 0.3).astype(np.float32)
    wu = (rng.normal(size=(HID, INTER)) * 0.3).astype(np.float32)
    wd = (rng.normal(size=(INTER, HID)) * 0.3).astype(np.float32)

    X, G, WG, WU, WD, STAGE = 0x1000, 0x2000, 0x10000, 0x20000, 0x30000, 0x100000
    xt = np.zeros((M, HID), np.float32)
    xt[0] = x
    image = bytearray(STAGE + 0x100000)
    for addr, arr in ((X, xt), (G, np.tile(gamma, (M, 1))), (WG, wg), (WU, wu), (WD, wd)):
        a = np.ascontiguousarray(arr, dtype=np.float32)
        image[addr:addr + a.nbytes] = a.tobytes()

    # --- graph path
    mod = tvm_backend.partition(FFNHalf)
    claimed, unclaimed = byoc_graph.coverage(mod)
    c1 = new_comp(STAGE)
    out_addr, n = byoc_graph.emit_graph(c1, mod, {"x": X, "gamma": G, "w_gate": WG, "w_up": WU,
                                                  "w_down": WD})
    # --- hand-driven path, same allocation order
    c2 = new_comp(STAGE)
    nrm = c2.alloc_staging(M * HID * 4)
    tvm_backend._emit_rmsnorm_row_fp32(c2, HID, x_addr=X, gamma_addr=G, out_addr=nrm,
                                       scratch_addr=c2.alloc_staging(4),
                                       stage_addr=c2.alloc_staging(M * _padded(HID) * 4))
    y = c2.alloc_staging(M * HID * 4)
    tvm_backend._emit_swiglu_ffn_fp32(c2, HID, INTER, nrm, WG, WU, WD, y)
    hand_out = c2.alloc_staging(M * HID * 4)
    tvm_backend._emit_residual_add_fp32(c2, X, y, hand_out, HID)

    identical = bytes(c1.instruction_stream) == bytes(c2.instruction_stream)
    for addr, data in c1._const_regions.items():
        image[addr:addr + len(data)] = data
    with tempfile.TemporaryDirectory() as tmp:
        p, d, o = (os.path.join(tmp, f) for f in ("p.bin", "d.img", "o.bin"))
        open(p, "wb").write(bytes(c1.instruction_stream))
        open(d, "wb").write(bytes(image))
        r = subprocess.run([SIM, p, HAL, str(len(image) // (1 << 20) + 2), "--dram-image", d,
                            "--dump", f"{out_addr}:{M * HID * 4}:{o}"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-300:] + r.stderr[-300:]
        got = np.frombuffer(open(o, "rb").read(), np.float32).reshape(M, HID)[0]
    xr = x.astype(np.float64)
    nr = xr / np.sqrt(np.mean(xr * xr) + EPS) * gamma
    ref = xr + (silu(nr @ wg) * (nr @ wu)) @ wd
    err = float(np.abs(got - ref).max() / np.abs(ref).max())
    res = [("graph_claims_every_op", claimed == ["rmsnorm", "swiglu_ffn", "add"] and not unclaimed,
            f"claimed {claimed}, unclaimed {unclaimed}"),
           ("stream_byte_identical_to_hand_driven", identical,
            f"{len(c1.instruction_stream) // 16} instrs" if identical else "streams differ"),
           ("numerics_match_numpy", err < 1e-4, f"rel err {err:.2e}")]
    return res


def case_attention_layer_coverage():
    mod = tvm_backend.partition(AttnLayer)
    claimed, unclaimed = byoc_graph.coverage(mod)
    from collections import Counter
    print(f"  full-layer graph: {len(claimed)} composites claimed {dict(Counter(claimed))}")
    print(f"  left unclaimed ({len(unclaimed)}): {dict(Counter(unclaimed))}")
    # Everything compute-heavy is claimed; what is left is data-layout glue plus whatever
    # the patterns do not reach. This is the measured scope of the graph path.
    glue = {"relax.reshape", "relax.permute_dims"}
    leftover = set(unclaimed) - glue
    return [("layer_graph_partitions", len(claimed) > 0, f"{len(claimed)} composites"),
            ("unclaimed_is_only_layout_glue", not leftover, f"non-glue unclaimed: {sorted(leftover)}")]


def main():
    results = []
    for name, fn in (("ffn", case_ffn_half), ("attn", case_attention_layer_coverage)):
        try:
            results += fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results.append((f"{name}_case_raised", False, f"{type(e).__name__}: {e}"))
    n = 0
    for name, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  -- {msg}")
        n += ok
    print(f"\n{n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
