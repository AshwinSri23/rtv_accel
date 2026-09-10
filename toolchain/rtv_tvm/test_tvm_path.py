#!/usr/bin/env python3
"""
test_tvm_path.py — end-to-end check of the TVM compilation path.

Runs the whole Version-B pipeline and compares against numpy:

    TIR matmul -> schedule (tile, stage to SRAM, tensorize)
               -> codegen (walk TIR, emit 128-bit words)
               -> rtv_sim
               -> numpy

The acceptance criterion is the same one the hand-written compiler is held
to: exact agreement with an independent reference. That matters more here
than usual, because the whole argument for this path is that TVM computes
the addresses -- so the test is really asking whether TVM's addressing agrees
with the device's view of memory.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_tvm_path.py
"""

import os
import subprocess
import sys
import tempfile

import math

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from tvm.script import tirx as T                      # noqa: E402
from npu_compiler import NPUCompiler, SRAMMap         # noqa: E402
from rtv_tvm.schedule import (schedule_matmul, schedule_elemwise,  # noqa: E402
                              schedule_mul, schedule_sumsq,
                              schedule_rms_scale, schedule_rowmax,
                              schedule_rowsum, schedule_transpose,
                              schedule_exp_sub, schedule_scalar_div,
                              schedule_silu)
from rtv_tvm import codegen                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")


def build_matmul(M, N, K):
    """A plain matmul in TIR -- deliberately unscheduled and unaware of the
    hardware. Everything hardware-specific happens in schedule_matmul."""
    @T.prim_func(s_tir=True)
    def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (M, K), "int8")
        B = T.match_buffer(b, (K, N), "int8")
        C = T.match_buffer(c, (M, N), "int32")
        for i, j, k in T.grid(M, N, K):
            with T.sblock("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0
                C[vi, vj] = C[vi, vj] + T.Cast("int32", A[vi, vk]) * T.Cast(
                    "int32", B[vk, vj])
    return matmul


def run_case(M, N, K, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(-8, 9, size=(M, K)).astype(np.int8)
    b = rng.integers(-8, 9, size=(K, N)).astype(np.int8)
    expected = a.astype(np.int32) @ b.astype(np.int32)

    a_ad, b_ad, c_ad = 0x1000, 0x40000, 0x80000
    dram = bytearray(c_ad + M * N * 4 + 0x1000)
    dram[a_ad:a_ad + a.nbytes] = a.tobytes()
    dram[b_ad:b_ad + b.nbytes] = b.tobytes()

    comp = NPUCompiler(HAL)
    comp.instruction_stream = bytearray()
    sch = schedule_matmul(build_matmul(M, N, K))
    codegen.emit(comp, sch, {"A": a_ad, "B": b_ad, "C": c_ad})
    n_instr = len(comp.instruction_stream) // 16

    with tempfile.TemporaryDirectory(prefix="rtv_tvm_") as wd:
        open(os.path.join(wd, "p.bin"), "wb").write(bytes(comp.instruction_stream))
        open(os.path.join(wd, "d.img"), "wb").write(bytes(dram))
        out = os.path.join(wd, "o.bin")
        mb = max(1, len(dram) // (1024 * 1024) + 2)
        r = subprocess.run(
            [SIM, os.path.join(wd, "p.bin"), HAL, str(mb),
             "--dram-image", os.path.join(wd, "d.img"),
             "--dump", f"{c_ad}:{M * N * 4}:{out}"],
            capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"simulator failed: {r.stdout[-300:]}{r.stderr[-300:]}", n_instr
        got = np.frombuffer(open(out, "rb").read(), dtype=np.int32).reshape(M, N)

    if np.array_equal(got, expected):
        return True, "", n_instr
    d = np.abs(got - expected)
    return False, (f"{int((d > 0).sum())}/{d.size} wrong, max delta "
                   f"{int(d.max())}"), n_instr


def build_elemwise(M, N):
    @T.prim_func(s_tir=True)
    def ew(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (M, N), "int8")
        B = T.match_buffer(b, (M, N), "int8")
        C = T.match_buffer(c, (M, N), "int8")
        for i, j in T.grid(M, N):
            with T.sblock("add"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] + B[vi, vj]
    return ew


def run_elemwise(M, N, seed=1):
    """The residual connection through the TVM path.

    Values are chosen to straddle INT8's range so saturation is exercised --
    an elementwise add that never saturates would not distinguish a correct
    ELEMWISE_ADD from a wider one.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(-60, 61, size=(M, N)).astype(np.int8)
    b = rng.integers(-60, 61, size=(M, N)).astype(np.int8)
    expected = np.clip(a.astype(np.int32) + b.astype(np.int32),
                       -128, 127).astype(np.int8)

    a_ad, b_ad, c_ad = 0x1000, 0x20000, 0x40000
    dram = bytearray(c_ad + M * N + 0x1000)
    dram[a_ad:a_ad + a.nbytes] = a.tobytes()
    dram[b_ad:b_ad + b.nbytes] = b.tobytes()

    comp = NPUCompiler(HAL)
    comp.instruction_stream = bytearray()
    sch = schedule_elemwise(build_elemwise(M, N))
    codegen.emit(comp, sch, {"A": a_ad, "B": b_ad, "C": c_ad}, kind="elemwise")
    n_instr = len(comp.instruction_stream) // 16

    with tempfile.TemporaryDirectory(prefix="rtv_tvm_") as wd:
        open(os.path.join(wd, "p.bin"), "wb").write(bytes(comp.instruction_stream))
        open(os.path.join(wd, "d.img"), "wb").write(bytes(dram))
        out = os.path.join(wd, "o.bin")
        r = subprocess.run(
            [SIM, os.path.join(wd, "p.bin"), HAL, "2",
             "--dram-image", os.path.join(wd, "d.img"),
             "--dump", f"{c_ad}:{M * N}:{out}"],
            capture_output=True, text=True)
        if r.returncode != 0:
            return False, "simulator failed", n_instr
        got = np.frombuffer(open(out, "rb").read(), dtype=np.int8).reshape(M, N)

    if np.array_equal(got, expected):
        return True, "", n_instr
    d = np.abs(got.astype(int) - expected.astype(int))
    return False, f"{int((d > 0).sum())}/{d.size} wrong", n_instr


CASES = [
    # square, one tile in each dimension -- the minimal case
    (16, 16, 16),
    # multiple K tiles: exercises accumulate, which the codegen infers from
    # the K loop position rather than being told
    (16, 16, 64),
    # multiple output tiles in both dimensions
    (32, 32, 32),
    # non-square, several tiles everywhere
    (32, 48, 64),
]


def build_mul(M, N):
    @T.prim_func(s_tir=True)
    def mul(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (M, N), "int8")
        B = T.match_buffer(b, (M, N), "int8")
        C = T.match_buffer(c, (M, N), "int8")
        for i, j in T.grid(M, N):
            with T.sblock("mul"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.Cast("int8", T.shift_right(
                    T.Cast("int32", A[vi, vj]) * T.Cast("int32", B[vi, vj]), 7))
    return mul


def run_mul(M, N, seed=2):
    """ELEMWISE_MUL, whose >>7 is the Q0.7 convention rather than a detail.

    Operands span the full INT8 range so the shift actually bites: a test
    with small values would pass even if the shift were wrong.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(-127, 128, size=(M, N)).astype(np.int8)
    b = rng.integers(-127, 128, size=(M, N)).astype(np.int8)
    expected = np.clip((a.astype(np.int32) * b.astype(np.int32)) >> 7,
                       -128, 127).astype(np.int8)
    a_ad, b_ad, c_ad = 0x1000, 0x20000, 0x40000
    dram = bytearray(c_ad + M * N + 0x1000)
    dram[a_ad:a_ad + a.nbytes] = a.tobytes()
    dram[b_ad:b_ad + b.nbytes] = b.tobytes()
    comp = NPUCompiler(HAL); comp.instruction_stream = bytearray()
    codegen.emit(comp, schedule_mul(build_mul(M, N)),
                 {"A": a_ad, "B": b_ad, "C": c_ad}, kind="elemwise")
    n = len(comp.instruction_stream) // 16
    with tempfile.TemporaryDirectory(prefix="rtv_tvm_") as wd:
        open(os.path.join(wd, "p.bin"), "wb").write(bytes(comp.instruction_stream))
        open(os.path.join(wd, "d.img"), "wb").write(bytes(dram))
        out = os.path.join(wd, "o.bin")
        r = subprocess.run([SIM, os.path.join(wd, "p.bin"), HAL, "2",
                            "--dram-image", os.path.join(wd, "d.img"),
                            "--dump", f"{c_ad}:{M * N}:{out}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False, "simulator failed", n
        got = np.frombuffer(open(out, "rb").read(), dtype=np.int8).reshape(M, N)
    return (np.array_equal(got, expected),
            "" if np.array_equal(got, expected) else "mismatch", n)


def build_sumsq(M, N):
    @T.prim_func(s_tir=True)
    def sq(a: T.handle, s: T.handle) -> None:
        A = T.match_buffer(a, (M, N), "int8")
        S = T.match_buffer(s, (1,), "int32")
        for i, j in T.grid(M, N):
            with T.sblock("sumsq"):
                vi, vj = T.axis.remap("RR", [i, j])
                with T.init():
                    S[0] = 0
                S[0] = S[0] + T.Cast("int32", A[vi, vj]) * T.Cast(
                    "int32", A[vi, vj])
    return sq


def run_sumsq(M, N, seed=3):
    """RMSNorm's first pass. N > 16 makes the reduction span tiles, which the
    ISA expresses with accumulate -- inferred from the loop, not told."""
    rng = np.random.default_rng(seed)
    a = rng.integers(-100, 101, size=(M, N)).astype(np.int8)
    expected = int((a.astype(np.int64) ** 2).sum())
    a_ad, s_ad = 0x1000, 0x20000
    dram = bytearray(s_ad + 0x1000)
    dram[a_ad:a_ad + a.nbytes] = a.tobytes()
    comp = NPUCompiler(HAL); comp.instruction_stream = bytearray()
    codegen.emit(comp, schedule_sumsq(build_sumsq(M, N)),
                 {"A": a_ad, "S": s_ad}, kind="reduce")
    comp._emit(comp._pass3_store(sram_addr=SRAMMap.SCALAR_SCRATCH_BASE,
                                 dram_addr=s_ad, dim_m=1, dim_n=4, stride=4))
    n = len(comp.instruction_stream) // 16
    with tempfile.TemporaryDirectory(prefix="rtv_tvm_") as wd:
        open(os.path.join(wd, "p.bin"), "wb").write(bytes(comp.instruction_stream))
        open(os.path.join(wd, "d.img"), "wb").write(bytes(dram))
        out = os.path.join(wd, "o.bin")
        r = subprocess.run([SIM, os.path.join(wd, "p.bin"), HAL, "2",
                            "--dram-image", os.path.join(wd, "d.img"),
                            "--dump", f"{s_ad}:4:{out}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False, "simulator failed", n
        got = int(np.frombuffer(open(out, "rb").read(), dtype=np.int32)[0])
    return got == expected, "" if got == expected else f"{got} != {expected}", n


def run_rmsnorm(M, N, seed=7):
    """Both halves of RMSNorm through TVM: REDUCE_SUMSQ then RMS_SCALE.

    This is the first case where two tensorized operations compose, with the
    scalar produced by one consumed by the other through SRAM -- which is how
    a real layer is built and the thing a single-op test cannot show.

    Only row 0 carries data, matching decode: the reduction sums the whole
    tile, so the padding rows contribute zero and the result is the row's own
    RMS.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((M, N), dtype=np.int8)
    x[0] = rng.integers(-40, 41, size=N)
    ssq = int((x.astype(np.int64) ** 2).sum())
    scale_num = int(round(127.0 * math.sqrt(N)))
    inv = np.float32(scale_num) / np.sqrt(np.float32(ssq))
    xf = x.astype(np.float32)
    expected = np.clip(np.trunc(xf * inv + np.copysign(np.float32(0.5), xf)),
                       -128, 127).astype(np.int8)

    a_ad, s_ad, c_ad = 0x1000, 0x20000, 0x21000
    dram = bytearray(c_ad + M * N + 0x1000)
    dram[a_ad:a_ad + x.nbytes] = x.tobytes()

    comp = NPUCompiler(HAL); comp.instruction_stream = bytearray()
    codegen.emit(comp, schedule_sumsq(build_sumsq(M, N)),
                 {"A": a_ad, "S": s_ad}, kind="reduce")
    codegen.emit(comp, schedule_rms_scale(build_rms(M, N)),
                 {"A": a_ad, "S": s_ad, "C": c_ad}, kind="elemwise",
                 scale_num=scale_num)
    n = len(comp.instruction_stream) // 16

    with tempfile.TemporaryDirectory(prefix="rtv_tvm_") as wd:
        open(os.path.join(wd, "p.bin"), "wb").write(bytes(comp.instruction_stream))
        open(os.path.join(wd, "d.img"), "wb").write(bytes(dram))
        out = os.path.join(wd, "o.bin")
        r = subprocess.run([SIM, os.path.join(wd, "p.bin"), HAL, "2",
                            "--dram-image", os.path.join(wd, "d.img"),
                            "--dump", f"{c_ad}:{M * N}:{out}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False, "simulator failed", n
        got = np.frombuffer(open(out, "rb").read(), dtype=np.int8).reshape(M, N)

    same = np.array_equal(got[0], expected[0])
    return same, "" if same else "row 0 mismatch", n


def build_rms(M, N):
    @T.prim_func(s_tir=True)
    def rms(s: T.handle, a: T.handle, c: T.handle) -> None:
        S = T.match_buffer(s, (1,), "int32")
        A = T.match_buffer(a, (M, N), "int8")
        C = T.match_buffer(c, (M, N), "int8")
        for i, j in T.grid(M, N):
            with T.sblock("rms"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.Cast("int8", T.Cast("float32", A[vi, vj])
                                   / T.sqrt(T.Cast("float32", S[0])))
    return rms


EW_CASES = [(16, 16), (32, 32), (16, 48)]
RMS_CASES = [(16, 32), (16, 64)]
MUL_CASES = [(16, 16), (32, 32)]
SUMSQ_CASES = [(16, 16), (16, 64), (32, 32)]


def build_rowmax(M, N):
    """Softmax pass 1 in plain loop form.

    Written out rather than generated: TVMScript parses the function's SOURCE
    TEXT, so a function built with exec() has none and fails with "could not
    get source code". The duplication with build_rowsum is the price.
    """
    @T.prim_func(s_tir=True)
    def red(a: T.handle, s: T.handle) -> None:
        A = T.match_buffer(a, (M, N), "int8")
        S = T.match_buffer(s, (1,), "int32")
        for i, j in T.grid(M, N):
            with T.sblock("rowmax"):
                vi, vj = T.axis.remap("RR", [i, j])
                with T.init():
                    S[0] = -128
                S[0] = T.max(S[0], T.Cast("int32", A[vi, vj]))
    return red


def build_rowsum(M, N):
    """Softmax pass 2's accumulation half."""
    @T.prim_func(s_tir=True)
    def red(a: T.handle, s: T.handle) -> None:
        A = T.match_buffer(a, (M, N), "int8")
        S = T.match_buffer(s, (1,), "int32")
        for i, j in T.grid(M, N):
            with T.sblock("rowsum"):
                vi, vj = T.axis.remap("RR", [i, j])
                with T.init():
                    S[0] = 0
                S[0] = S[0] + T.Cast("int32", A[vi, vj])
    return red


def run_reduce(M, N, kind, seed=11):
    """REDUCE_MAX and the plain sum, softmax's first two passes.

    N > 16 makes the reduction span tiles, so this also covers the
    per-activation accumulate bookkeeping: each reduction mode needs its own
    "have I emitted yet" state.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(-100, 101, size=(M, N)).astype(np.int8)
    if kind == "rowmax":
        expected = int(a.max())
        sch = schedule_rowmax(build_rowmax(M, N))
    else:
        expected = int(a.astype(np.int64).sum())
        sch = schedule_rowsum(build_rowsum(M, N))

    a_ad, s_ad = 0x1000, 0x20000
    dram = bytearray(s_ad + 0x1000)
    dram[a_ad:a_ad + a.nbytes] = a.tobytes()
    comp = NPUCompiler(HAL); comp.instruction_stream = bytearray()
    cg_slot = (codegen.SCALAR_MAX if kind == "rowmax" else codegen.SCALAR_SUM)
    codegen.emit(comp, sch, {"A": a_ad, "S": s_ad}, kind="reduce")
    comp._emit(comp._pass3_store(sram_addr=cg_slot, dram_addr=s_ad,
                                 dim_m=1, dim_n=4, stride=4))
    n = len(comp.instruction_stream) // 16
    with tempfile.TemporaryDirectory(prefix="rtv_tvm_") as wd:
        open(os.path.join(wd, "p.bin"), "wb").write(bytes(comp.instruction_stream))
        open(os.path.join(wd, "d.img"), "wb").write(bytes(dram))
        out = os.path.join(wd, "o.bin")
        r = subprocess.run([SIM, os.path.join(wd, "p.bin"), HAL, "2",
                            "--dram-image", os.path.join(wd, "d.img"),
                            "--dump", f"{s_ad}:4:{out}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"simulator failed: {r.stdout[-160:]}", n
        got = int(np.frombuffer(open(out, "rb").read(), dtype=np.int32)[0])
    return got == expected, "" if got == expected else f"{got} != {expected}", n


def build_transpose(M, N):
    @T.prim_func(s_tir=True)
    def tp(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (M, N), "int8")
        C = T.match_buffer(c, (M, N), "int8")
        for i, j in T.grid(M, N):
            with T.sblock("transpose"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vj, vi]
    return tp


def run_transpose(N, seed=13):
    """WHOLE-MATRIX transpose: C[i, j] = A[j, i].

    Square only, and that is a real constraint rather than laziness. The TIR
    reads A[vj, vi] out of a buffer declared (M, N), so a non-square case
    indexes past the end -- 16x64 asks for row 63 of a 16-row buffer.

    Note the composition this exercises: at output tile (io, jo) the SOURCE
    tile is (jo, io), and the instruction then transposes within it. The
    codegen gets the swapped tile address for free because it evaluates the
    store's index expressions rather than assuming loop order -- which is the
    whole argument for reading addresses off the TIR. An earlier version of
    this test asserted a per-tile (blockwise) transpose instead and "failed"
    against correct output; the two agree only at 16x16.
    """
    M = N
    rng = np.random.default_rng(seed)
    a = rng.integers(-100, 101, size=(M, N)).astype(np.int8)
    expected = a.T.copy()

    a_ad, c_ad = 0x1000, 0x20000
    dram = bytearray(c_ad + a.nbytes + 0x1000)
    dram[a_ad:a_ad + a.nbytes] = a.tobytes()
    comp = NPUCompiler(HAL); comp.instruction_stream = bytearray()
    codegen.emit(comp, schedule_transpose(build_transpose(M, N)),
                 {"A": a_ad, "C": c_ad}, kind="elemwise")
    n = len(comp.instruction_stream) // 16
    with tempfile.TemporaryDirectory(prefix="rtv_tvm_") as wd:
        open(os.path.join(wd, "p.bin"), "wb").write(bytes(comp.instruction_stream))
        open(os.path.join(wd, "d.img"), "wb").write(bytes(dram))
        out = os.path.join(wd, "o.bin")
        r = subprocess.run([SIM, os.path.join(wd, "p.bin"), HAL, "2",
                            "--dram-image", os.path.join(wd, "d.img"),
                            "--dump", f"{c_ad}:{a.nbytes}:{out}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"simulator failed: {r.stdout[-160:]}", n
        got = np.frombuffer(open(out, "rb").read(), dtype=np.int8).reshape(M, N)
    same = np.array_equal(got, expected)
    return same, "" if same else f"{int((got != expected).sum())} elems differ", n


REDUCE_CASES = [("rowmax", 16, 16), ("rowmax", 16, 64), ("rowmax", 32, 32),
                ("rowsum", 16, 16), ("rowsum", 16, 64), ("rowsum", 32, 32)]
TRANSPOSE_CASES = [16, 32, 64]


def build_expsub(M, N):
    @T.prim_func(s_tir=True)
    def es(s: T.handle, a: T.handle, c: T.handle) -> None:
        S = T.match_buffer(s, (1,), "int32")
        A = T.match_buffer(a, (M, N), "int8")
        C = T.match_buffer(c, (M, N), "int8")
        for i, j in T.grid(M, N):
            with T.sblock("expsub"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.Cast("int8", T.exp(
                    T.Cast("float32", A[vi, vj]) - T.Cast("float32", S[0])))
    return es


def build_sdiv(M, N):
    @T.prim_func(s_tir=True)
    def sd(s: T.handle, a: T.handle, c: T.handle) -> None:
        S = T.match_buffer(s, (1,), "int32")
        A = T.match_buffer(a, (M, N), "int8")
        C = T.match_buffer(c, (M, N), "int8")
        for i, j in T.grid(M, N):
            with T.sblock("sdiv"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.Cast("int8", T.Cast("float32", A[vi, vj])
                                   / T.Cast("float32", S[0]))
    return sd


SOFTMAX_T = 64.0          # SOFTMAX_TEMPERATURE in pe_array.cpp


def run_softmax(M, N, seed=17):
    """All THREE softmax passes composed through TVM.

    The passes cannot be merged or reordered -- each needs the previous one's
    result over the whole matrix before it can start -- and two scalars are
    live at once, because pass 2 reads the max while accumulating the sum.
    That second point is what this test really covers: a single scalar slot
    would have pass 2 overwrite the max it is still reading.

    The reference models the device's INT8 arithmetic exactly (see
    scalar_alu in pe_array.cpp), including the *127 output convention, rather
    than comparing against float softmax -- INT8 softmax is lossy by design,
    so a float reference would fail for reasons that are not bugs.
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(-100, 101, size=(M, N)).astype(np.int8)

    mx = int(x.max())
    e = np.clip(np.round(np.exp((x.astype(np.float32) - mx) / SOFTMAX_T)
                         * 127.0), -128, 127).astype(np.int8)
    tot = int(e.astype(np.int64).sum())
    expected = (np.clip(np.round(e.astype(np.float32) * 127.0 / tot), -128, 127)
                .astype(np.int8) if tot else np.zeros_like(e))

    a_ad, mx_ad, e_ad, c_ad = 0x1000, 0x20000, 0x21000, 0x30000
    dram = bytearray(c_ad + M * N + 0x1000)
    dram[a_ad:a_ad + x.nbytes] = x.tobytes()

    comp = NPUCompiler(HAL); comp.instruction_stream = bytearray()
    # pass 1: row max -> SCALAR_MAX
    codegen.emit(comp, schedule_rowmax(build_rowmax(M, N)),
                 {"A": a_ad, "S": mx_ad}, kind="reduce")
    # pass 2a: exp(x - max) -> e
    codegen.emit(comp, schedule_exp_sub(build_expsub(M, N)),
                 {"S": mx_ad, "A": a_ad, "C": e_ad}, kind="elemwise")
    # pass 2b: sum(e) -> SCALAR_SUM (a different slot; the max is still live)
    codegen.emit(comp, schedule_rowsum(build_rowsum(M, N)),
                 {"A": e_ad, "S": 0x22000}, kind="reduce")
    # pass 3: e / sum
    codegen.emit(comp, schedule_scalar_div(build_sdiv(M, N)),
                 {"S": 0x22000, "A": e_ad, "C": c_ad}, kind="elemwise")
    n = len(comp.instruction_stream) // 16

    with tempfile.TemporaryDirectory(prefix="rtv_tvm_") as wd:
        open(os.path.join(wd, "p.bin"), "wb").write(bytes(comp.instruction_stream))
        open(os.path.join(wd, "d.img"), "wb").write(bytes(dram))
        out = os.path.join(wd, "o.bin")
        r = subprocess.run([SIM, os.path.join(wd, "p.bin"), HAL, "4",
                            "--dram-image", os.path.join(wd, "d.img"),
                            "--dump", f"{c_ad}:{M * N}:{out}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"simulator failed: {r.stdout[-160:]}", n
        got = np.frombuffer(open(out, "rb").read(), dtype=np.int8).reshape(M, N)
    same = np.array_equal(got, expected)
    return same, "" if same else f"{int((got != expected).sum())} elems differ", n


SOFTMAX_CASES = [(16, 16), (16, 64), (32, 32)]


def build_silu(M, N):
    @T.prim_func(s_tir=True)
    def sl(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (M, N), "int8")
        C = T.match_buffer(c, (M, N), "int8")
        for i, j in T.grid(M, N):
            with T.sblock("silu"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.Cast("int8", T.Cast("float32", A[vi, vj]) / (
                    T.float32(1.0)
                    + T.exp(T.float32(0.0) - T.Cast("float32", A[vi, vj]))))
    return sl


def run_silu(M, N, seed=23):
    """SiLU through the LUT instruction.

    The intrinsic's `desc` states the actual arithmetic (x * sigmoid(x))
    while its `impl` is a table lookup -- desc says what the instruction
    COMPUTES and impl says what the device DOES, and those are allowed to
    differ. The reference here is the TABLE, not the closed form: the table
    is what the device reads, so comparing against x*sigmoid(x) directly
    would fail on quantisation error that is by design.
    """
    rng = np.random.default_rng(seed)
    in_scale = 0.05
    x = rng.integers(-100, 101, size=(M, N)).astype(np.int8)

    comp = NPUCompiler(HAL); comp.instruction_stream = bytearray()
    silu = lambda v: v / (1.0 + np.exp(-v))
    lut_ad, _ = comp.lut_dram(silu, in_scale, "silu_test")
    tbl = np.frombuffer(comp._const_regions[lut_ad], dtype=np.int8)
    # The device indexes table[code + 128] -- BIASED into 0..255, not
    # two's-complement wrapped. The two agree on non-negative codes and
    # differ on every negative one, which is most of SiLU's interesting
    # range. The reference is a gather, not an evaluation.
    expected = tbl[x.astype(np.int32) + 128].reshape(M, N)

    a_ad, c_ad = 0x1000, 0x40000
    dram = bytearray(max(c_ad + M * N, lut_ad + 256) + 0x1000)
    dram[a_ad:a_ad + x.nbytes] = x.tobytes()
    dram[lut_ad:lut_ad + 256] = comp._const_regions[lut_ad]

    codegen.emit(comp, schedule_silu(build_silu(M, N)),
                 {"A": a_ad, "C": c_ad}, kind="elemwise", lut_dram=lut_ad)
    n = len(comp.instruction_stream) // 16
    with tempfile.TemporaryDirectory(prefix="rtv_tvm_") as wd:
        open(os.path.join(wd, "p.bin"), "wb").write(bytes(comp.instruction_stream))
        open(os.path.join(wd, "d.img"), "wb").write(bytes(dram))
        out = os.path.join(wd, "o.bin")
        r = subprocess.run([SIM, os.path.join(wd, "p.bin"), HAL, "4",
                            "--dram-image", os.path.join(wd, "d.img"),
                            "--dump", f"{c_ad}:{M * N}:{out}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"simulator failed: {r.stdout[-200:]}", n
        got = np.frombuffer(open(out, "rb").read(), dtype=np.int8).reshape(M, N)
    same = np.array_equal(got, expected)
    return same, "" if same else f"{int((got != expected).sum())} elems differ", n


SILU_CASES = [(16, 16), (32, 32)]


def main():
    ok = total = 0
    for M, N, K in CASES:
        passed, detail, n = run_case(M, N, K)
        print(f"{'PASS' if passed else 'FAIL'}  tvm_matmul_{M}x{N}x{K}  "
              f"({n} instrs){'  ' + detail if detail else ''}")
        ok += passed; total += 1
    for M, N in EW_CASES:
        passed, detail, n = run_elemwise(M, N)
        print(f"{'PASS' if passed else 'FAIL'}  tvm_add_{M}x{N}  "
              f"({n} instrs){'  ' + detail if detail else ''}")
        ok += passed; total += 1
    for M, N in MUL_CASES:
        passed, detail, n = run_mul(M, N)
        print(f"{'PASS' if passed else 'FAIL'}  tvm_mul_{M}x{N}  "
              f"({n} instrs){'  ' + detail if detail else ''}")
        ok += passed; total += 1
    for M, N in SUMSQ_CASES:
        passed, detail, n = run_sumsq(M, N)
        print(f"{'PASS' if passed else 'FAIL'}  tvm_sumsq_{M}x{N}  "
              f"({n} instrs){'  ' + detail if detail else ''}")
        ok += passed; total += 1
    for M, N in RMS_CASES:
        passed, detail, n = run_rmsnorm(M, N)
        print(f"{'PASS' if passed else 'FAIL'}  tvm_rmsnorm_{M}x{N}  "
              f"({n} instrs){'  ' + detail if detail else ''}")
        ok += passed; total += 1
    for kind, M, N in REDUCE_CASES:
        passed, detail, n = run_reduce(M, N, kind)
        print(f"{'PASS' if passed else 'FAIL'}  tvm_{kind}_{M}x{N}  "
              f"({n} instrs){'  ' + detail if detail else ''}")
        ok += passed; total += 1
    for N in TRANSPOSE_CASES:
        passed, detail, n = run_transpose(N)
        print(f"{'PASS' if passed else 'FAIL'}  tvm_transpose_{N}x{N}  "
              f"({n} instrs){'  ' + detail if detail else ''}")
        ok += passed; total += 1
    for M, N in SOFTMAX_CASES:
        passed, detail, n = run_softmax(M, N)
        print(f"{'PASS' if passed else 'FAIL'}  tvm_softmax_{M}x{N}  "
              f"({n} instrs){'  ' + detail if detail else ''}")
        ok += passed; total += 1
    for M, N in SILU_CASES:
        passed, detail, n = run_silu(M, N)
        print(f"{'PASS' if passed else 'FAIL'}  tvm_silu_{M}x{N}  "
              f"({n} instrs){'  ' + detail if detail else ''}")
        ok += passed; total += 1
    print(f"\n{ok}/{total} passed")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
