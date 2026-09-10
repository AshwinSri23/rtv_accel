#!/usr/bin/env python3
"""
test_composite_dispatch.py — end-to-end check of composite-level BYOC
dispatch, one level up from test_tvm_path.py's bare TensorIR intrinsics.

test_tvm_path.py proves the 11 hardware intrinsics are individually correct.
This proves the pieces TVM actually adds on top of that: real Relax graphs
get partitioned by patterns.py into "rtv_accel"-claimed composite functions,
tvm_backend.offloaded_composites() correctly identifies which pattern
matched each one, and emit_composite() drives the right NPUCompiler method
end to end through the real simulator.

    Relax graph (R.nn.rms_norm / SwiGLU matmul chain / softmax(Q@K)@V)
      -> patterns.partition()          (FuseOpsByPattern + MergeCompositeFunctions)
      -> tvm_backend.offloaded_composites()   (find + classify each composite)
      -> tvm_backend.emit_composite()  (composes rtv_tvm.schedule/codegen directly)
      -> rtv_sim
      -> numpy

Everything runs in FP32 (compiler.dtype = DType.FP32) -- the project's
target moved to native FP32 precision with INT8 quantization out of scope
(see rtv-accel-fp32-only-target in project memory), and emit_composite no
longer calls into npu_compiler.py's compile_* methods for the composite
kinds covered here. See tvm_backend.emit_composite's docstring for which
composite kinds aren't dispatched yet and why, and
_emit_attention_core_fp32's docstring for attention's specific scope limit
(exactly one K/V tile; one real query replicated across all 16 tile rows).

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_composite_dispatch.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import tvm                                  # noqa: E402
from tvm.script import relax as R           # noqa: E402

from compiler_state import NPUCompiler, DType  # noqa: E402
import tvm_backend                           # noqa: E402
from rtv_tvm import patterns                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")

tvm_backend.register()


def _run_sim(instruction_stream, dram, dumps):
    """Run rtv_sim over `dram`, returning {name: np.ndarray} per dump spec
    in `dumps` = {name: (addr, nbytes, np_dtype, shape)}."""
    with tempfile.TemporaryDirectory(prefix="rtv_composite_") as wd:
        p = os.path.join(wd, "p.bin")
        d = os.path.join(wd, "d.img")
        open(p, "wb").write(bytes(instruction_stream))
        open(d, "wb").write(bytes(dram))
        mb = max(1, len(dram) // (1024 * 1024) + 2)
        args = [SIM, p, HAL, str(mb), "--dram-image", d]
        outs = {}
        for name, (addr, nbytes, _, _) in dumps.items():
            outs[name] = os.path.join(wd, f"{name}.bin")
            args += ["--dump", f"{addr}:{nbytes}:{outs[name]}"]
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"simulator failed: {r.stdout[-500:]}{r.stderr[-500:]}")
        results = {}
        for name, (addr, nbytes, np_dtype, shape) in dumps.items():
            results[name] = np.frombuffer(
                open(outs[name], "rb").read(), dtype=np_dtype).reshape(shape)
        return results


# TVMScript's @R.function decorator re-parses the enclosing function's
# SOURCE TEXT and resolves identifiers against the module's globals -- a
# shape parameter captured from an enclosing Python function's closure
# ("Undefined variable: M") doesn't resolve the way a plain global does. So
# each shape used below gets its own module-level, literal-shaped
# ir_module rather than a shared parameterized builder.

@tvm.script.ir_module
class _RMSNormMod1x32:
    @R.function
    def main(x: R.Tensor((1, 32), "float32"),
             gamma: R.Tensor((32,), "float32")) -> R.Tensor((1, 32), "float32"):
        with R.dataflow():
            out = R.nn.rms_norm(x, gamma, axes=[-1])
            R.output(out)
        return out


@tvm.script.ir_module
class _RMSNormMod1x64:
    @R.function
    def main(x: R.Tensor((1, 64), "float32"),
             gamma: R.Tensor((64,), "float32")) -> R.Tensor((1, 64), "float32"):
        with R.dataflow():
            out = R.nn.rms_norm(x, gamma, axes=[-1])
            R.output(out)
        return out


@tvm.script.ir_module
class _MatmulMod16x32x16:
    @R.function
    def main(x: R.Tensor((16, 32), "float32"),
             w: R.Tensor((32, 16), "float32")) -> R.Tensor((16, 16), "float32"):
        with R.dataflow():
            out = R.matmul(x, w)
            R.output(out)
        return out


def run_bare_matmul_case(seed=5):
    """
    Exercises offloaded_matmuls()/emit() -- the ORIGINAL BYOC dispatch path,
    predating offloaded_composites()/emit_composite() above. It was never
    actually run against a real partitioned module before (tvm_backend.py
    had a bad `from toolchain.rtv_tvm import ...` that made it unimportable
    at all), and offloaded_matmuls() turned out to only look at an offloaded
    function's own top-level bindings -- which, after MergeCompositeFunctions,
    never contain the matmul Call itself, only a call to the nested composite
    function that does. It silently returned [] for every real module. Both
    are fixed now; this locks the fix in.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(16, 32)).astype(np.float32)
    w = rng.uniform(-1, 1, size=(32, 16)).astype(np.float32)
    expected = (x.astype(np.float64) @ w.astype(np.float64)).astype(np.float32)

    mod = patterns.partition(_MatmulMod16x32x16)
    shapes = tvm_backend.offloaded_matmuls(mod)
    if shapes != [(16, 32, 16, "float32")]:
        return False, f"unexpected offloaded_matmuls result: {shapes}", 0

    x_ad, w_ad, out_ad = 0x1000, 0x40000, 0x80000
    dram = bytearray(out_ad + 16 * 16 * 4 + 0x1000)
    dram[x_ad:x_ad + x.nbytes] = x.tobytes()
    dram[w_ad:w_ad + w.nbytes] = w.tobytes()

    comp = NPUCompiler(HAL)
    comp.instruction_stream = bytearray()
    tvm_backend.emit(shapes, comp, {"x": x_ad, "w": w_ad, "out": out_ad})
    n_instr = len(comp.instruction_stream) // 16

    got = _run_sim(comp.instruction_stream, dram,
                    {"out": (out_ad, 16 * 16 * 4, np.float32, (16, 16))})["out"]

    if np.allclose(got, expected, atol=1e-2, rtol=1e-2):
        return True, "", n_instr
    d = np.abs(got - expected)
    return False, f"{int((d > 1e-2).sum())}/{d.size} wrong, max delta {float(d.max()):.4g}", n_instr


def _rmsnorm_module(M, H):
    return {(1, 32): _RMSNormMod1x32, (1, 64): _RMSNormMod1x64}[(M, H)]


def run_rmsnorm_case(M, H, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=(M, H)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=(H,)).astype(np.float32)
    expected = x / np.sqrt((x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True) + 1e-6)
    expected = (expected * gamma).astype(np.float32)

    mod = patterns.partition(_rmsnorm_module(M, H))
    composites = tvm_backend.offloaded_composites(mod)
    if len(composites) != 1 or composites[0]["kind"] != "rmsnorm":
        return False, f"expected exactly one rmsnorm composite, got {composites}", 0

    # emit_composite's pure-TVM RMSNorm dispatch multiplies by gamma via a
    # plain tile-shaped ELEMWISE_MUL, not a broadcast load -- so gamma has to
    # be laid out as a genuine (16, H) buffer with the same row repeated,
    # matching every other buffer's fixed 16-row tile convention here. See
    # _emit_rmsnorm_row_fp32's docstring for why (the ORIGINAL
    # compile_rmsnorm_row's attempt at broadcasting a 1-row buffer via stride
    # tricks read garbage for rows 1-15).
    gamma_bcast = np.tile(gamma, (16, 1)).astype(np.float32)

    x_ad, gamma_ad, out_ad = 0x1000, 0x40000, 0x80000
    dram = bytearray(out_ad + M * H * 4 + 0x1000)
    dram[x_ad:x_ad + x.nbytes] = x.tobytes()
    dram[gamma_ad:gamma_ad + gamma_bcast.nbytes] = gamma_bcast.tobytes()

    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    # Staging (the RMS-scaled intermediate + the sumsq scalar) is
    # bump-allocated from address 0 by default -- push it well past every
    # fixed address above so it can't land on top of x/gamma/out.
    comp._dram_cursor = out_ad + M * H * 4 + 0x10000
    tvm_backend.emit_composite(comp, composites[0],
                                {"x": x_ad, "gamma": gamma_ad, "out": out_ad})
    n_instr = len(comp.instruction_stream) // 16

    dram_size = max(len(dram), comp.dram_bytes_needed) + 0x10000
    if dram_size > len(dram):
        dram = dram + bytearray(dram_size - len(dram))
    got = _run_sim(comp.instruction_stream, dram,
                    {"out": (out_ad, M * H * 4, np.float32, (M, H))})["out"]

    if np.allclose(got, expected, atol=1e-3, rtol=1e-3):
        return True, "", n_instr
    d = np.abs(got - expected)
    return False, f"{int((d > 1e-3).sum())}/{d.size} wrong, max delta {float(d.max()):.4g}", n_instr


@tvm.script.ir_module
class _SwigluMod16x32x64:
    @R.function
    def main(x: R.Tensor((16, 32), "float32"),
             w_gate: R.Tensor((32, 64), "float32"),
             w_up: R.Tensor((32, 64), "float32"),
             w_down: R.Tensor((64, 32), "float32")) -> R.Tensor((16, 32), "float32"):
        with R.dataflow():
            gate = R.matmul(x, w_gate)
            up = R.matmul(x, w_up)
            act = R.nn.silu(gate)
            inter = R.multiply(act, up)
            out = R.matmul(inter, w_down)
            R.output(out)
        return out


@tvm.script.ir_module
class _SwigluMod8x64x96:
    @R.function
    def main(x: R.Tensor((8, 64), "float32"),
             w_gate: R.Tensor((64, 96), "float32"),
             w_up: R.Tensor((64, 96), "float32"),
             w_down: R.Tensor((96, 64), "float32")) -> R.Tensor((8, 64), "float32"):
        with R.dataflow():
            gate = R.matmul(x, w_gate)
            up = R.matmul(x, w_up)
            act = R.nn.silu(gate)
            inter = R.multiply(act, up)
            out = R.matmul(inter, w_down)
            R.output(out)
        return out


def _swiglu_module(M, H, I):
    return {(16, 32, 64): _SwigluMod16x32x64,
            (8, 64, 96): _SwigluMod8x64x96}[(M, H, I)]


def run_swiglu_case(M, H, I, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(M, H)).astype(np.float32)
    w_gate = rng.uniform(-0.3, 0.3, size=(H, I)).astype(np.float32)
    w_up = rng.uniform(-0.3, 0.3, size=(H, I)).astype(np.float32)
    w_down = rng.uniform(-0.3, 0.3, size=(I, H)).astype(np.float32)

    gate = x.astype(np.float64) @ w_gate.astype(np.float64)
    up = x.astype(np.float64) @ w_up.astype(np.float64)
    silu = gate / (1.0 + np.exp(-gate))
    inter = silu * up
    expected = (inter @ w_down.astype(np.float64)).astype(np.float32)

    mod = patterns.partition(_swiglu_module(M, H, I))
    composites = tvm_backend.offloaded_composites(mod)
    if len(composites) != 1 or composites[0]["kind"] != "swiglu_ffn":
        return False, f"expected exactly one swiglu_ffn composite, got {composites}", 0

    x_ad, gate_ad, up_ad, down_ad, out_ad = (
        0x1000, 0x40000, 0x80000, 0xC0000, 0x100000)

    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    # Staging buffers (gate/up/act) are bump-allocated above whatever the
    # cursor currently is; point it past every buffer laid out above so
    # staging can't land on top of our own weights/activations.
    comp._dram_cursor = down_ad + w_down.nbytes + 0x10000
    tvm_backend.emit_composite(comp, composites[0], {
        "x": x_ad, "w_gate": gate_ad, "w_up": up_ad, "w_down": down_ad,
        "out": out_ad,
    })
    n_instr = len(comp.instruction_stream) // 16

    dram_size = max(out_ad + M * H * 4, comp.dram_bytes_needed) + 0x10000
    dram = bytearray(dram_size)
    dram[x_ad:x_ad + x.nbytes] = x.tobytes()
    dram[gate_ad:gate_ad + w_gate.nbytes] = w_gate.tobytes()
    dram[up_ad:up_ad + w_up.nbytes] = w_up.tobytes()
    dram[down_ad:down_ad + w_down.nbytes] = w_down.tobytes()

    got = _run_sim(comp.instruction_stream, dram,
                    {"out": (out_ad, M * H * 4, np.float32, (M, H))})["out"]

    if np.allclose(got, expected, atol=1e-2, rtol=1e-2):
        return True, "", n_instr
    d = np.abs(got - expected)
    return False, f"{int((d > 1e-2).sum())}/{d.size} wrong, max delta {float(d.max()):.4g}", n_instr


@tvm.script.ir_module
class _AttentionMod16:
    @R.function
    def main(q: R.Tensor((16, 16), "float32"),
             k: R.Tensor((16, 16), "float32"),
             v: R.Tensor((16, 16), "float32")) -> R.Tensor((16, 16), "float32"):
        with R.dataflow():
            scores = R.matmul(q, k)
            probs = R.nn.softmax(scores, axis=-1)
            out = R.matmul(probs, v)
            R.output(out)
        return out


@tvm.script.ir_module
class _AttentionMaskedMod16:
    @R.function
    def main(q: R.Tensor((16, 16), "float32"),
             k: R.Tensor((16, 16), "float32"),
             mask: R.Tensor((16, 16), "float32"),
             v: R.Tensor((16, 16), "float32")) -> R.Tensor((16, 16), "float32"):
        with R.dataflow():
            scores = R.matmul(q, k)
            masked = R.add(scores, mask)
            probs = R.nn.softmax(masked, axis=-1)
            out = R.matmul(probs, v)
            R.output(out)
        return out


def run_attention_case(masked=False, seed=11):
    """
    One real query attending over 16 real, distinct keys/values.

    Q is REPLICATED across all 16 tile rows rather than being one real row
    padded with zeros -- see _emit_attention_core_fp32's docstring for why:
    REDUCE always collapses the whole 16x16 tile to one scalar, so a
    genuine per-row softmax isn't reachable through this codegen yet.
    Replicating the one real query makes every row compute the identical
    thing, so the whole-tile-shared max/sum is exactly that row's own
    softmax -- not an approximation. K and V don't need this: COMPUTE_GEMM
    is a real per-row systolic multiply with no cross-row reduction, so 16
    genuinely different keys/values come through correctly on their own.
    """
    rng = np.random.default_rng(seed)
    q_row = rng.uniform(-1, 1, size=(16,)).astype(np.float32)
    q = np.tile(q_row, (16, 1)).astype(np.float32)
    k = rng.uniform(-1, 1, size=(16, 16)).astype(np.float32)
    v = rng.uniform(-1, 1, size=(16, 16)).astype(np.float32)

    scores = q_row.astype(np.float64) @ k.astype(np.float64)
    mask = None
    if masked:
        mask_row = np.zeros(16, dtype=np.float32)
        mask_row[10:] = -1e30      # this one query may only see keys 0-9
        mask = np.tile(mask_row, (16, 1)).astype(np.float32)
        scores = scores + mask_row.astype(np.float64)
    m = scores.max()
    e = np.exp(scores - m)
    probs = e / e.sum()
    expected_row = (probs @ v.astype(np.float64)).astype(np.float32)
    expected = np.tile(expected_row, (16, 1)).astype(np.float32)

    mod = patterns.partition(_AttentionMaskedMod16 if masked else _AttentionMod16)
    composites = tvm_backend.offloaded_composites(mod)
    want_kind = "attention_masked" if masked else "attention"
    if len(composites) != 1 or composites[0]["kind"] != want_kind:
        return False, f"expected exactly one {want_kind} composite, got {composites}", 0

    q_ad, k_ad, v_ad, mask_ad, out_ad = (
        0x1000, 0x40000, 0x80000, 0xC0000, 0x100000)
    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = out_ad + 16 * 16 * 4 + 0x10000

    addrs = {"q": q_ad, "k": k_ad, "v": v_ad, "out": out_ad}
    if masked:
        addrs["mask"] = mask_ad
    tvm_backend.emit_composite(comp, composites[0], addrs)
    n_instr = len(comp.instruction_stream) // 16

    dram_size = max(out_ad + 16 * 16 * 4, comp.dram_bytes_needed) + 0x10000
    dram = bytearray(dram_size)
    dram[q_ad:q_ad + q.nbytes] = q.tobytes()
    dram[k_ad:k_ad + k.nbytes] = k.tobytes()
    dram[v_ad:v_ad + v.nbytes] = v.tobytes()
    if masked:
        dram[mask_ad:mask_ad + mask.nbytes] = mask.tobytes()
    # _emit_attention_core_fp32 injects a compile-time constant (the row0
    # mask that keeps the sum step correct under row replication) via
    # compiler._const_regions -- same mechanism the LUT path uses. Copying
    # it into the DRAM image is the caller's job, exactly as
    # test_tvm_path.py's silu test already does for its LUT table.
    for addr, data in comp._const_regions.items():
        dram[addr:addr + len(data)] = data

    got = _run_sim(comp.instruction_stream, dram,
                    {"out": (out_ad, 16 * 16 * 4, np.float32, (16, 16))})["out"]

    if np.allclose(got, expected, atol=1e-2, rtol=1e-2):
        return True, "", n_instr
    d = np.abs(got - expected)
    return False, f"{int((d > 1e-2).sum())}/{d.size} wrong, max delta {float(d.max()):.4g}", n_instr


def main():
    cases = [
        ("bare_matmul_16x32x16", run_bare_matmul_case),
        ("composite_rmsnorm_1x32",  lambda: run_rmsnorm_case(1, 32)),
        ("composite_rmsnorm_1x64",  lambda: run_rmsnorm_case(1, 64, seed=2)),
        ("composite_swiglu_16x32x64", lambda: run_swiglu_case(16, 32, 64)),
        ("composite_swiglu_8x64x96",  lambda: run_swiglu_case(8, 64, 96, seed=3)),
        ("composite_attention_16x16", lambda: run_attention_case(masked=False)),
        ("composite_attention_masked_16x16", lambda: run_attention_case(masked=True, seed=13)),
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
