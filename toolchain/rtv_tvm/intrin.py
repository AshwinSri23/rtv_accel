"""
intrin.py — the RTV-Accel hardware instruction, declared for TVM.

A tensor intrinsic is a pair of functions. `desc` says what one instruction
COMPUTES, in ordinary loop form; `impl` says what the device DOES instead.
Where a scheduled loop nest matches `desc`, `tensorize` substitutes `impl`.

This is the one thing TVM cannot infer. Everything else in the pipeline --
tiling, addressing, strides -- follows from it, which is why it is the first
piece built and the piece that had to work before the approach was viable.

Naming note for this TVM release: `tvm.tir` is `tvm.s_tir`, TVMScript's TIR
dialect is `tvm.script.tirx`, `T.block` is `T.sblock`, and PrimFuncs for
scheduling need `s_tir=True`. No published tutorial matches; the installed
sources under site-packages/tvm/ are the reference.
"""

from tvm import s_tir
from tvm.script import tirx as T

TILE = 16
NAME = "rtv.gemm_16x16"
ADD_NAME = "rtv.add_16x16"
MUL_NAME = "rtv.mul_16x16"
SUMSQ_NAME = "rtv.sumsq_16x16"
RMSSCALE_NAME = "rtv.rms_scale_16x16"


@T.prim_func(s_tir=True)
def gemm_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    """One COMPUTE_GEMM: 16x16x16 INT8 multiply into an INT32 accumulator.

    scope="shared" models SRAM. That is not decoration -- it is why the
    schedule MUST cache operands into SRAM before tensorizing, and it is the
    check that caught the first attempt: tensorize refused a global-scope
    nest with `scope(): global vs shared`.
    """
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    B = T.match_buffer(b, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "int32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(C[0:TILE, 0:TILE], A[0:TILE, 0:TILE], B[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        for i, j, k in T.grid(TILE, TILE, TILE):
            with T.sblock("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + T.Cast("int32", A[vi, vk]) * T.Cast(
                    "int32", B[vk, vj])


@T.prim_func(s_tir=True)
def gemm_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    """What the device does: a single instruction. TVM never looks inside.

    The codegen recognises this extern call and emits a COMPUTE_GEMM word.
    `accumulate` is not modelled here -- it is a property of the K loop, which
    the schedule owns and the codegen reads off the loop structure.
    """
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    B = T.match_buffer(b, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "int32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(C[0:TILE, 0:TILE], A[0:TILE, 0:TILE], B[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("rtv_compute_gemm", C.data, A.data, B.data,
                                 dtype="int32"))


def register():
    s_tir.TensorIntrin.register(NAME, gemm_desc, gemm_impl, override=True)
    return NAME


@T.prim_func(s_tir=True)
def add_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    """One EXEC_ALU ELEMWISE_ADD over a 16x16 tile.

    Elementwise ops matter as much as the GEMM for showing the approach
    generalises: if only matmul could be expressed, TVM would be a frontend
    for one operation rather than a compiler for the ISA.
    """
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    B = T.match_buffer(b, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], B[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("add"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] + B[vi, vj]


@T.prim_func(s_tir=True)
def add_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    B = T.match_buffer(b, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], B[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("rtv_elemwise_add", C.data, A.data, B.data,
                                 dtype="int8"))


def register_add():
    s_tir.TensorIntrin.register(ADD_NAME, add_desc, add_impl, override=True)
    return ADD_NAME


@T.prim_func(s_tir=True)
def mul_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    """EXEC_ALU ELEMWISE_MUL: (a*b) >> 7.

    The shift is part of the instruction's meaning, not an implementation
    detail -- it is the Q0.7 fixed-point convention, and writing it into the
    desc means a loop nest only tensorizes if it actually computes that.
    """
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    B = T.match_buffer(b, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], B[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("mul"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.Cast("int8", T.shift_right(
                    T.Cast("int32", A[vi, vj]) * T.Cast("int32", B[vi, vj]), 7))


@T.prim_func(s_tir=True)
def mul_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    B = T.match_buffer(b, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], B[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("rtv_elemwise_mul", C.data, A.data, B.data,
                                 dtype="int8"))


@T.prim_func(s_tir=True)
def sumsq_desc(a: T.handle, s: T.handle) -> None:
    """REDUCE_SUMSQ: collapse a 16x16 INT8 tile to one INT32 sum of squares.

    Structurally different from the elementwise and GEMM intrinsics: the
    output is a scalar, not a tile, so the whole block is a reduction. This
    is RMSNorm's first pass, and the accumulate flag lets it span the tiles
    of a row -- which the schedule expresses as the outer loop.
    """
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "int32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        # Accumulator FIRST. tensorize compares read sets positionally, and
        # TVM emits the accumulator ahead of the operand -- the gemm desc
        # already had C first, which is why it matched on the first attempt.
        T.reads(S[0:1], A[0:TILE, 0:TILE])
        T.writes(S[0:1])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("sumsq"):
                vi, vj = T.axis.remap("RR", [i, j])
                S[0] = S[0] + T.Cast("int32", A[vi, vj]) * T.Cast("int32", A[vi, vj])


@T.prim_func(s_tir=True)
def sumsq_impl(a: T.handle, s: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "int32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(S[0:1], A[0:TILE, 0:TILE])
        T.writes(S[0:1])
        T.evaluate(T.call_extern("rtv_reduce_sumsq", S.data, A.data, dtype="int32"))


def register_mul():
    s_tir.TensorIntrin.register(MUL_NAME, mul_desc, mul_impl, override=True)
    return MUL_NAME


def register_sumsq():
    s_tir.TensorIntrin.register(SUMSQ_NAME, sumsq_desc, sumsq_impl, override=True)
    return SUMSQ_NAME


@T.prim_func(s_tir=True)
def rms_scale_desc(a: T.handle, s: T.handle, c: T.handle) -> None:
    """SCALAR_ALU RMS_SCALE: divide a tile by the root of a scalar.

    The instruction actually computes `clamp8(round(x * scale_num /
    sqrt(S)))`, but `scale_num` is an immediate and a desc has to be a closed
    function. So the intrinsic matches the SHAPE -- tile divided by the root
    of an accumulated scalar -- and the codegen supplies the constant from the
    scale plan. That split is the same one drawn everywhere else here: TVM
    matches structure, the scale semantics stay with the compiler that
    derived them.
    """
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "int32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    with T.sblock("root"):
        # Operand order follows the BLOCK's read order, which for a
        # non-reduction is the order the buffers appear in the expression --
        # A before S here. The sumsq intrinsic needs the opposite because a
        # read-modify-write accumulator is read first. Neither order is a
        # convention to memorise; it is whatever TVM computes for the block
        # being matched, and the error message names both sides.
        T.reads(A[0:TILE, 0:TILE], S[0:1])
        T.writes(C[0:TILE, 0:TILE])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("rms"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.Cast("int8", T.Cast("float32", A[vi, vj])
                                   / T.sqrt(T.Cast("float32", S[0])))


@T.prim_func(s_tir=True)
def rms_scale_impl(a: T.handle, s: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "int32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], S[0:1])
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("rtv_rms_scale", C.data, A.data, S.data,
                                 dtype="int8"))


def register_rms_scale():
    s_tir.TensorIntrin.register(RMSSCALE_NAME, rms_scale_desc, rms_scale_impl,
                                override=True)
    return RMSSCALE_NAME


# ─────────────────────────────────────────────────────────────────────────────
# Softmax: four instructions across three passes.
#
# The passes cannot be merged or reordered -- each needs the previous one's
# result over the WHOLE row before it can start:
#
#   1. REDUCE_MAX  across the row's tiles              -> row max
#   2. EXP_SUB against that max, while REDUCE sums     -> row sum
#   3. SCALAR_DIV every tile by that sum               -> normalised
#
# Two scalars are live at once, not one: pass 2 reads the max while
# accumulating the sum, so they cannot share a slot. That is a codegen
# concern (which SRAM address each lands at), not a TIR one -- here they are
# simply distinct buffers, and TVM keeps them distinct for free.
# ─────────────────────────────────────────────────────────────────────────────

ROWMAX_NAME = "rtv.rowmax_16x16"
ROWSUM_NAME = "rtv.rowsum_16x16"
EXPSUB_NAME = "rtv.exp_sub_16x16"
SCALARDIV_NAME = "rtv.scalar_div_16x16"
SILU_NAME = "rtv.silu_16x16"
TRANSPOSE_NAME = "rtv.transpose_16x16"


@T.prim_func(s_tir=True)
def rowmax_desc(a: T.handle, s: T.handle) -> None:
    """REDUCE with REDUCE_MAX: collapse a tile to its largest element.

    Same reduction shape as sumsq -- accumulator first in the read set -- and
    the accumulate flag likewise lets one row's max span several tiles.
    """
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "int32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(S[0:1], A[0:TILE, 0:TILE])
        T.writes(S[0:1])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("rowmax"):
                vi, vj = T.axis.remap("RR", [i, j])
                S[0] = T.max(S[0], T.Cast("int32", A[vi, vj]))


@T.prim_func(s_tir=True)
def rowmax_impl(a: T.handle, s: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "int32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(S[0:1], A[0:TILE, 0:TILE])
        T.writes(S[0:1])
        T.evaluate(T.call_extern("rtv_reduce_max", S.data, A.data, dtype="int32"))


@T.prim_func(s_tir=True)
def rowsum_desc(a: T.handle, s: T.handle) -> None:
    """REDUCE with no activation: plain sum of a tile's elements."""
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "int32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(S[0:1], A[0:TILE, 0:TILE])
        T.writes(S[0:1])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("rowsum"):
                vi, vj = T.axis.remap("RR", [i, j])
                S[0] = S[0] + T.Cast("int32", A[vi, vj])


@T.prim_func(s_tir=True)
def rowsum_impl(a: T.handle, s: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "int32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(S[0:1], A[0:TILE, 0:TILE])
        T.writes(S[0:1])
        T.evaluate(T.call_extern("rtv_reduce_sum", S.data, A.data, dtype="int32"))


@T.prim_func(s_tir=True)
def exp_sub_desc(a: T.handle, s: T.handle, c: T.handle) -> None:
    """SCALAR_ALU EXP_SUB: C = exp(A - S), the max subtracted for stability.

    Non-reduction, so the read order is the order the buffers appear in the
    expression: A before S. (The reductions above need the opposite, because
    a read-modify-write accumulator is read first.)
    """
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "int32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], S[0:1])
        T.writes(C[0:TILE, 0:TILE])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("expsub"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.Cast("int8", T.exp(
                    T.Cast("float32", A[vi, vj]) - T.Cast("float32", S[0])))


@T.prim_func(s_tir=True)
def exp_sub_impl(a: T.handle, s: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "int32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], S[0:1])
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("rtv_exp_sub", C.data, A.data, S.data,
                                 dtype="int8"))


@T.prim_func(s_tir=True)
def scalar_div_desc(a: T.handle, s: T.handle, c: T.handle) -> None:
    """SCALAR_ALU SCALAR_DIV: C = A / S, softmax's normalising pass."""
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "int32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], S[0:1])
        T.writes(C[0:TILE, 0:TILE])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("sdiv"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.Cast("int8", T.Cast("float32", A[vi, vj])
                                   / T.Cast("float32", S[0]))


@T.prim_func(s_tir=True)
def scalar_div_impl(a: T.handle, s: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "int32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], S[0:1])
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("rtv_scalar_div", C.data, A.data, S.data,
                                 dtype="int8"))


@T.prim_func(s_tir=True)
def silu_desc(a: T.handle, c: T.handle) -> None:
    """SiLU, x * sigmoid(x), which the device evaluates by table lookup.

    Written here as the actual arithmetic rather than as an opaque shape.
    That is the intended division of labour: `desc` states what the
    instruction COMPUTES and `impl` states what the device DOES, and those
    are allowed to differ -- the device's LUT is an implementation of this
    function, so declaring the function honestly is both truthful and what
    lets a reader check the table is approximating the right thing.
    """
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("silu"):
                vi, vj = T.axis.remap("SS", [i, j])
                # Inlined deliberately: a local binding inside a block is
                # parsed as a BUFFER declaration, not a value, and fails
                # well-formedness with "buffer x is used ... without a prior
                # DeclBuffer". TVMScript is not Python here.
                C[vi, vj] = T.Cast("int8", T.Cast("float32", A[vi, vj]) / (
                    T.float32(1.0)
                    + T.exp(T.float32(0.0) - T.Cast("float32", A[vi, vj]))))


@T.prim_func(s_tir=True)
def silu_impl(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("rtv_lut", C.data, A.data, dtype="int8"))


@T.prim_func(s_tir=True)
def transpose_desc(a: T.handle, c: T.handle) -> None:
    """TRANSPOSE over one square tile: C[i, j] = A[j, i]."""
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("transpose"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vj, vi]


@T.prim_func(s_tir=True)
def transpose_impl(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "int8", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("rtv_transpose", C.data, A.data, dtype="int8"))


def register_rowmax():
    s_tir.TensorIntrin.register(ROWMAX_NAME, rowmax_desc, rowmax_impl, override=True)
    return ROWMAX_NAME


def register_rowsum():
    s_tir.TensorIntrin.register(ROWSUM_NAME, rowsum_desc, rowsum_impl, override=True)
    return ROWSUM_NAME


def register_exp_sub():
    s_tir.TensorIntrin.register(EXPSUB_NAME, exp_sub_desc, exp_sub_impl, override=True)
    return EXPSUB_NAME


def register_scalar_div():
    s_tir.TensorIntrin.register(SCALARDIV_NAME, scalar_div_desc, scalar_div_impl,
                                override=True)
    return SCALARDIV_NAME


def register_silu():
    s_tir.TensorIntrin.register(SILU_NAME, silu_desc, silu_impl, override=True)
    return SILU_NAME


def register_transpose():
    s_tir.TensorIntrin.register(TRANSPOSE_NAME, transpose_desc, transpose_impl,
                                override=True)
    return TRANSPOSE_NAME


# ─────────────────────────────────────────────────────────────────────────────
# FP32 variants.
#
# Everything above is INT8-only -- TVM's `tensorize` requires an exact dtype
# match between the scheduled loop nest and a registered intrinsic's `desc`,
# so none of the INT8 descriptors above can ever match an FP32 PrimFunc.
# These exist because the project's target moved to native FP32 precision
# with INT8 quantization out of scope (see rtv-accel-fp32-only-target in
# project memory) -- these are not a debugging convenience alongside the
# INT8 path, they are now the actually-shipped dtype.
#
# The good news, confirmed directly against pe_array.cpp before writing
# these: the SIMULATOR's FP32 branches are already real and correct (GEMM
# accumulates straight in float with no requantization step, ELEMWISE_MUL is
# a plain `*=` with no Q0.7 shift, RMS_SCALE is `x * sqrt(scale_num / sum)`
# with no fixed-point convention). So these descs are simpler than their
# INT8 counterparts, not just retyped -- there is no scale semantics left to
# encode once dtype is FP32, which is the whole reason retiring
# npu_compiler.py is tractable now (see tvm-replaces-npu-compiler /
# rtv-accel-fp32-only-target).
# ─────────────────────────────────────────────────────────────────────────────

GEMM_FP32_NAME = "rtv.gemm_16x16_fp32"
MUL_FP32_NAME = "rtv.mul_16x16_fp32"
SUMSQ_FP32_NAME = "rtv.sumsq_16x16_fp32"
RMSSCALE_FP32_NAME = "rtv.rms_scale_16x16_fp32"
SILU_FP32_NAME = "rtv.silu_16x16_fp32"


@T.prim_func(s_tir=True)
def gemm_fp32_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    """COMPUTE_GEMM at FP32: straight float accumulate, no requantization."""
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    B = T.match_buffer(b, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(C[0:TILE, 0:TILE], A[0:TILE, 0:TILE], B[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        for i, j, k in T.grid(TILE, TILE, TILE):
            with T.sblock("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func(s_tir=True)
def gemm_fp32_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    B = T.match_buffer(b, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(C[0:TILE, 0:TILE], A[0:TILE, 0:TILE], B[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("rtv_compute_gemm", C.data, A.data, B.data,
                                 dtype="float32"))


def register_gemm_fp32():
    s_tir.TensorIntrin.register(GEMM_FP32_NAME, gemm_fp32_desc, gemm_fp32_impl,
                                override=True)
    return GEMM_FP32_NAME


@T.prim_func(s_tir=True)
def mul_fp32_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    """EXEC_ALU ELEMWISE_MUL at FP32: a plain `a * b`, no Q0.7 shift."""
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    B = T.match_buffer(b, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], B[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("mul"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] * B[vi, vj]


@T.prim_func(s_tir=True)
def mul_fp32_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    B = T.match_buffer(b, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], B[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("rtv_elemwise_mul", C.data, A.data, B.data,
                                 dtype="float32"))


def register_mul_fp32():
    s_tir.TensorIntrin.register(MUL_FP32_NAME, mul_fp32_desc, mul_fp32_impl,
                                override=True)
    return MUL_FP32_NAME


@T.prim_func(s_tir=True)
def sumsq_fp32_desc(a: T.handle, s: T.handle) -> None:
    """REDUCE_SUMSQ at FP32. The simulator sums in double internally (see
    pe_array.cpp) to avoid losing low-order terms over a wide row; that is an
    `impl`-side implementation detail the same way the INT8 desc's simple
    int32 sum doesn't reflect whatever wide accumulation the real hardware
    does either -- `desc` only has to match the SHAPE of a generically
    scheduled reduction for tensorize, not be bit-identical to `impl`."""
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(S[0:1], A[0:TILE, 0:TILE])
        T.writes(S[0:1])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("sumsq"):
                vi, vj = T.axis.remap("RR", [i, j])
                S[0] = S[0] + A[vi, vj] * A[vi, vj]


@T.prim_func(s_tir=True)
def sumsq_fp32_impl(a: T.handle, s: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(S[0:1], A[0:TILE, 0:TILE])
        T.writes(S[0:1])
        T.evaluate(T.call_extern("rtv_reduce_sumsq", S.data, A.data, dtype="float32"))


def register_sumsq_fp32():
    s_tir.TensorIntrin.register(SUMSQ_FP32_NAME, sumsq_fp32_desc, sumsq_fp32_impl,
                                override=True)
    return SUMSQ_FP32_NAME


@T.prim_func(s_tir=True)
def rms_scale_fp32_desc(a: T.handle, s: T.handle, c: T.handle) -> None:
    """SCALAR_ALU RMS_SCALE at FP32: C = A / sqrt(S). No fixed-point scale --
    scale_num=0 at runtime (see pe_array.cpp's FP32 branch) selects the plain
    `1/sqrt(S)` case, which is what this desc matches; codegen.emit's
    scale_num kwarg can still select `sqrt(scale_num/S)` at runtime the same
    way the INT8 path's scale_num does, without changing what the desc
    needs to structurally match."""
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "float32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], S[0:1])
        T.writes(C[0:TILE, 0:TILE])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("rms"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] / T.sqrt(S[0])


@T.prim_func(s_tir=True)
def rms_scale_fp32_impl(a: T.handle, s: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "float32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], S[0:1])
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("rtv_rms_scale", C.data, A.data, S.data,
                                 dtype="float32"))


def register_rms_scale_fp32():
    s_tir.TensorIntrin.register(RMSSCALE_FP32_NAME, rms_scale_fp32_desc,
                                rms_scale_fp32_impl, override=True)
    return RMSSCALE_FP32_NAME


@T.prim_func(s_tir=True)
def silu_fp32_desc(a: T.handle, c: T.handle) -> None:
    """SiLU at FP32: x / (1 + exp(-x)) directly -- no LUT needed in float,
    matching pe_array.cpp's FP32 SILU/LUT branch exactly (it evaluates the
    true function rather than a table lookup once there's no INT8 code range
    to table over)."""
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("silu"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] / (
                    T.float32(1.0) + T.exp(T.float32(0.0) - A[vi, vj]))


@T.prim_func(s_tir=True)
def silu_fp32_impl(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        # Deliberately NOT "rtv_lut": codegen.py's LUT dispatch requires a
        # 256-entry table address (the INT8 device needs one; a table over
        # an 8-bit code range has nothing to mean at FP32). pe_array.cpp's
        # FP32 branch handles ActivationMode.SILU and LUT identically ("no
        # table needed in float"), so this uses the plain SILU opcode
        # instead and skips the table dependency entirely.
        T.evaluate(T.call_extern("rtv_silu_fp32", C.data, A.data, dtype="float32"))


def register_silu_fp32():
    s_tir.TensorIntrin.register(SILU_FP32_NAME, silu_fp32_desc, silu_fp32_impl,
                                override=True)
    return SILU_FP32_NAME


# ─────────────────────────────────────────────────────────────────────────────
# FP32 variants, part 2: the four softmax-pass ops, TRANSPOSE and ADD.
#
# Added for attention's core softmax(Q@K)@V, per composite kind. All five
# confirmed directly against pe_array.cpp before writing: REDUCE (MAX and
# plain sum) and SCALAR_ALU (EXP_SUB, SCALAR_DIV) have real FP32 branches
# with no fixed-point convention, matching everything else in this file's
# FP32 section; TRANSPOSE's FP32 branch requires a SQUARE tile, which one
# 16x16 TensorIntrin invocation always is by construction (TILE==TILE) --
# transposing a matrix wider than one tile is a schedule/addressing concern
# handled the same way schedule_transpose's docstring already describes for
# INT8, not something these two descs need to know about.
# ─────────────────────────────────────────────────────────────────────────────

ROWMAX_FP32_NAME = "rtv.rowmax_16x16_fp32"
ROWSUM_FP32_NAME = "rtv.rowsum_16x16_fp32"
EXPSUB_FP32_NAME = "rtv.exp_sub_16x16_fp32"
SCALARDIV_FP32_NAME = "rtv.scalar_div_16x16_fp32"
TRANSPOSE_FP32_NAME = "rtv.transpose_16x16_fp32"
ADD_FP32_NAME = "rtv.add_16x16_fp32"


@T.prim_func(s_tir=True)
def rowmax_fp32_desc(a: T.handle, s: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(S[0:1], A[0:TILE, 0:TILE])
        T.writes(S[0:1])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("rowmax"):
                vi, vj = T.axis.remap("RR", [i, j])
                S[0] = T.max(S[0], A[vi, vj])


@T.prim_func(s_tir=True)
def rowmax_fp32_impl(a: T.handle, s: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(S[0:1], A[0:TILE, 0:TILE])
        T.writes(S[0:1])
        T.evaluate(T.call_extern("rtv_reduce_max", S.data, A.data, dtype="float32"))


def register_rowmax_fp32():
    s_tir.TensorIntrin.register(ROWMAX_FP32_NAME, rowmax_fp32_desc, rowmax_fp32_impl,
                                override=True)
    return ROWMAX_FP32_NAME


@T.prim_func(s_tir=True)
def rowsum_fp32_desc(a: T.handle, s: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(S[0:1], A[0:TILE, 0:TILE])
        T.writes(S[0:1])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("rowsum"):
                vi, vj = T.axis.remap("RR", [i, j])
                S[0] = S[0] + A[vi, vj]


@T.prim_func(s_tir=True)
def rowsum_fp32_impl(a: T.handle, s: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(S[0:1], A[0:TILE, 0:TILE])
        T.writes(S[0:1])
        T.evaluate(T.call_extern("rtv_reduce_sum", S.data, A.data, dtype="float32"))


def register_rowsum_fp32():
    s_tir.TensorIntrin.register(ROWSUM_FP32_NAME, rowsum_fp32_desc, rowsum_fp32_impl,
                                override=True)
    return ROWSUM_FP32_NAME


@T.prim_func(s_tir=True)
def exp_sub_fp32_desc(a: T.handle, s: T.handle, c: T.handle) -> None:
    """C = exp(A - S). No softmax TEMPERATURE to undo at FP32 -- the scores
    are true values, not quantisation codes (see pe_array.cpp's comment on
    why T existed at all: undoing INT8's score quantisation). scale_num=0
    at runtime keeps the ISA's T=1 default, matching this desc exactly."""
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "float32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], S[0:1])
        T.writes(C[0:TILE, 0:TILE])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("expsub"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.exp(A[vi, vj] - S[0])


@T.prim_func(s_tir=True)
def exp_sub_fp32_impl(a: T.handle, s: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "float32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], S[0:1])
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("rtv_exp_sub", C.data, A.data, S.data,
                                 dtype="float32"))


def register_exp_sub_fp32():
    s_tir.TensorIntrin.register(EXPSUB_FP32_NAME, exp_sub_fp32_desc,
                                exp_sub_fp32_impl, override=True)
    return EXPSUB_FP32_NAME


@T.prim_func(s_tir=True)
def scalar_div_fp32_desc(a: T.handle, s: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "float32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], S[0:1])
        T.writes(C[0:TILE, 0:TILE])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("sdiv"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] / S[0]


@T.prim_func(s_tir=True)
def scalar_div_fp32_impl(a: T.handle, s: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    S = T.match_buffer(s, (1,), "float32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], S[0:1])
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("rtv_scalar_div", C.data, A.data, S.data,
                                 dtype="float32"))


def register_scalar_div_fp32():
    s_tir.TensorIntrin.register(SCALARDIV_FP32_NAME, scalar_div_fp32_desc,
                                scalar_div_fp32_impl, override=True)
    return SCALARDIV_FP32_NAME


@T.prim_func(s_tir=True)
def transpose_fp32_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("transpose"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vj, vi]


@T.prim_func(s_tir=True)
def transpose_fp32_impl(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("rtv_transpose", C.data, A.data, dtype="float32"))


def register_transpose_fp32():
    s_tir.TensorIntrin.register(TRANSPOSE_FP32_NAME, transpose_fp32_desc,
                                transpose_fp32_impl, override=True)
    return TRANSPOSE_FP32_NAME


@T.prim_func(s_tir=True)
def add_fp32_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    """ELEMWISE_ADD at FP32 -- used for attention's masked variant (scores +
    mask before softmax). No clamp, no rescale: a plain sum."""
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    B = T.match_buffer(b, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], B[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        for i, j in T.grid(TILE, TILE):
            with T.sblock("add"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] + B[vi, vj]


@T.prim_func(s_tir=True)
def add_fp32_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    B = T.match_buffer(b, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    C = T.match_buffer(c, (TILE, TILE), "float32", offset_factor=1, scope="shared")
    with T.sblock("root"):
        T.reads(A[0:TILE, 0:TILE], B[0:TILE, 0:TILE])
        T.writes(C[0:TILE, 0:TILE])
        T.evaluate(T.call_extern("rtv_elemwise_add", C.data, A.data, B.data,
                                 dtype="float32"))


def register_add_fp32():
    s_tir.TensorIntrin.register(ADD_FP32_NAME, add_fp32_desc, add_fp32_impl,
                                override=True)
    return ADD_FP32_NAME
