#!/usr/bin/env python3
"""
tvm_backend.py — a TVM BYOC front end for the RTV-Accel ISA.

Why a front end at all. `npu_compiler.py` hand-writes every address, stride
and tile bound, and roughly seven of this project's bugs came from exactly
that: multiplying an element count by a width, by hand, in one of ninety-five
places. In a typed IR the element width lives in the tensor type and the
framework computes strides, which removes that whole class.

Precision note (supersedes the paragraph this replaced): the project's
target moved to native FP32 precision, with INT8 quantization out of scope
(see rtv-accel-fp32-only-target in project memory). The original version of
this module kept `npu_compiler.py` as the actual instruction-emitting
backend specifically to preserve its scale semantics (Q0.7, >>7, RESCALE
shifts) -- knowledge that only exists because INT8 needs it. FP32 needs none
of that (confirmed directly against pe_array.cpp: FP32 GEMM/ELEMWISE_MUL/
RMS_SCALE are all the literal float operation, no fixed-point convention
anywhere), so the reason to keep `npu_compiler.py` in the loop is gone.
`emit_composite` below now composes `rtv_tvm.schedule`/`codegen` primitives
directly against the FP32 `TensorIntrin`s in `rtv_tvm.intrin` -- TVM owns
addressing AND instruction emission, `npu_compiler.py` is not called at all.
(`offloaded_matmuls`/`emit` below still delegate to `NPUCompiler.compile_linear`
for the standalone-matmul path; not yet migrated to the same scheme.)

Design. This is deliberately NOT a `RunCodegen` backend producing a TVM
runtime module -- that needs a C++ runtime and would make TVM own execution,
which is the wrong split when the whole point is our own simulator. Instead:

    Relax graph
      -> FuseOpsByPattern        (TVM matches our hardware's operations)
      -> MergeCompositeFunctions (TVM groups them into offloadable functions)
      -> emit_composite()        (WE walk the result and emit instructions)

TVM does graph handling, pattern matching, fusion AND (for the composites
emit_composite covers) the addressing/scheduling of the instruction stream
itself.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tvm                                          # noqa: E402
from tvm import relax                               # noqa: E402
from tvm.relax.dpl import pattern as dpl            # noqa: E402
from tvm.relax.backend.pattern_registry import register_patterns  # noqa: E402
from tvm.script import tirx as T                    # noqa: E402

from npu_compiler import NPUCompiler, DType, SRAMMap, _padded, _align   # noqa: E402

from rtv_tvm import patterns, schedule, codegen, intrin
from rtv_isa import DType, Opcode, ActivationMode, encode

CODEGEN = patterns.CODEGEN


def register():
    """Tell TVM which subgraphs this accelerator claims."""
    patterns.register()


def partition(mod):
    """Run TVM's BYOC partitioning: match, then group into composites."""
    return patterns.partition(mod)


def offloaded_matmuls(mod):
    """
    Yield (M, K, N, dtype) for every matmul TVM handed to this accelerator.

    MergeCompositeFunctions nests the actual matched ops one function deeper
    than the outer {"Codegen": "rtv_accel"} function -- see _composite_kind's
    docstring, which hit the same structure. The original version here only
    ever looked at the outer function's own top-level bindings, which never
    contain a bare relax.Call to "matmul" (only a call to the nested
    composite function) -- confirmed empirically: it silently returned []
    for every real partitioned module. Fixed by recursing into nested
    Functions, not just SeqExpr blocks.

    Only descends into functions whose OWN composite kind is "matmul" --
    not any Codegen-tagged function, since a composite like swiglu_ffn also
    contains matmul Calls internally, and those are already claimed and
    handled by emit_composite(). Recursing into every Codegen function
    regardless of kind would double-report (and, if both this and
    emit_composite() were run over the same module, double-emit) the
    matmuls inside any bigger composite.
    """
    found = []

    def visit(expr):
        if isinstance(expr, relax.Function):
            visit(expr.body)
            return
        if not isinstance(expr, relax.SeqExpr):
            return
        for blk in expr.blocks:
            for bind in blk.bindings:
                val = bind.value
                if isinstance(val, relax.Function):
                    visit(val)
                    continue
                if not isinstance(val, relax.Call):
                    continue
                op = getattr(val, "op", None)
                name = getattr(op, "name_hint", getattr(op, "name", str(op)))
                if "matmul" not in str(name):
                    continue
                a_shape, a_dtype = patterns._extract_shape_dtype(val.args[0])
                b_shape, _ = patterns._extract_shape_dtype(val.args[1])
                m, k = a_shape
                k2, n = b_shape
                assert k == k2, f"matmul shape mismatch {k} vs {k2}"
                found.append((m, k, n, a_dtype))

    for gv, func in mod.functions.items():
        if isinstance(func, relax.Function):
            attrs = getattr(func, "attrs", None)
            if (attrs and attrs.get("Codegen") == CODEGEN
                    and _composite_kind(func) == "matmul"):
                visit(func)
    return found


def emit(shapes, compiler, addrs):
    """Emit RTV-Accel instructions for each offloaded matmul."""
    for (m, k, n, dtype) in shapes:
        if hasattr(compiler, "compile_linear"):
            compiler.compile_linear(
                dram_weight=addrs["w"], m=m, n=n, k=k,
                out_dram=addrs["out"], dram_activation=addrs["x"],
                dtype=DType.FP32 if dtype == "float32" else DType.INT8,
            )
        else:
            # Direct TVM schedule & codegen emission
            from tvm.script import tirx as T
            @T.prim_func(s_tir=True)
            def matmul_prim(a: T.handle, b: T.handle, c: T.handle) -> None:
                A = T.match_buffer(a, (m, k), "float32" if dtype == "float32" else "int8")
                B = T.match_buffer(b, (k, n), "float32" if dtype == "float32" else "int8")
                C = T.match_buffer(c, (m, n), "float32" if dtype == "float32" else "int32")
                for i, j, k_idx in T.grid(m, n, k):
                    with T.sblock("update"):
                        vi, vj, vk = T.axis.remap("SSR", [i, j, k_idx])
                        with T.init():
                            C[vi, vj] = 0
                        C[vi, vj] = C[vi, vj] + T.Cast(
                            "float32" if dtype == "float32" else "int32", A[vi, vk]
                        ) * T.Cast(
                            "float32" if dtype == "float32" else "int32", B[vk, vj]
                        )
            sch = schedule.schedule_matmul(matmul_prim)
            dram_map = {"A": addrs["x"], "B": addrs["w"], "C": addrs["out"]}
            codegen.emit(compiler, sch, dram_map, dtype=DType.FP32 if dtype == "float32" else DType.INT8)


# ── Generalized composite dispatch ────────────────────────────────────────────
#
# offloaded_matmuls()/emit() above only ever look for bare matmuls. Real
# models offload more than that -- patterns.py also matches SwiGLU FFN and
# RMSNorm (attention is claimed too, but see the note in emit_composite below
# on why it isn't dispatched yet). The functions here generalize the same
# idea: find every subgraph TVM claimed, work out which pattern matched it,
# and hand it to the NPUCompiler method that already knows how to emit that
# operation -- exactly the split the module docstring describes, just no
# longer matmul-only.

def _composite_kind(func: "relax.Function") -> str:
    """
    An outer BYOC function only carries {"Codegen": "rtv_accel"}; the pattern
    name TVM actually matched ({"Composite": "rtv_accel.<kind>"}) lives on a
    nested local function one level inside it, because that's what
    MergeCompositeFunctions produces. Confirmed empirically against TVM
    0.26: partition a single rms_norm op and print the module -- the
    Composite attr is not on the function get_offloaded_functions() returns,
    it's on `local_func` bound inside that function's body.

    Returns the suffix after "rtv_accel." (e.g. "rmsnorm"), or "" if no
    nested Composite-tagged function is found.
    """
    def visit(expr):
        if isinstance(expr, relax.Function):
            attrs = expr.attrs
            if attrs and "Composite" in attrs:
                name = str(attrs["Composite"])
                return name.split(".", 1)[1] if "." in name else name
            return visit(expr.body)
        if isinstance(expr, relax.SeqExpr):
            for blk in expr.blocks:
                for bind in blk.bindings:
                    if isinstance(bind.value, relax.Function):
                        found = visit(bind.value)
                        if found:
                            return found
            return visit(expr.body)
        return ""
    return visit(func)


def offloaded_composites(mod):
    """
    Every offloaded composite in `mod`, tagged with which pattern claimed it.

    Each entry is patterns.get_offloaded_functions()'s dict (name, func,
    params, return) plus a "kind" key -- "rmsnorm", "swiglu_ffn", "matmul",
    "attention", etc. -- and "param_names", the ordered list of param name
    strings for convenient addrs-dict lookups.
    """
    out = []
    for info in patterns.get_offloaded_functions(mod):
        info = dict(info)
        info["kind"] = _composite_kind(info["func"])
        info["param_names"] = [p["name"] for p in info["params"]]
        out.append(info)
    return out


def _rmsnorm_prim_fp32(M, N):
    """One RMSNorm-with-gamma tile, as three chained FP32 PrimFuncs.

    Returns (sumsq_func, rms_scale_func, mul_func). Buffer variable names
    ("A", "S", "B", "C"...) are what dram_map keys must match -- TVM names a
    buffer from the LOCAL VARIABLE it's assigned to in match_buffer, not the
    handle parameter's name (confirmed against how test_tvm_path.py's own
    build_* functions are consumed).
    """
    @T.prim_func(s_tir=True)
    def sq(a: T.handle, s: T.handle) -> None:
        A = T.match_buffer(a, (M, N), "float32")
        S = T.match_buffer(s, (1,), "float32")
        for i, j in T.grid(M, N):
            with T.sblock("sumsq"):
                vi, vj = T.axis.remap("RR", [i, j])
                with T.init():
                    S[0] = T.float32(0)
                S[0] = S[0] + A[vi, vj] * A[vi, vj]

    @T.prim_func(s_tir=True)
    def rms(s: T.handle, a: T.handle, c: T.handle) -> None:
        S = T.match_buffer(s, (1,), "float32")
        A = T.match_buffer(a, (M, N), "float32")
        C = T.match_buffer(c, (M, N), "float32")
        for i, j in T.grid(M, N):
            with T.sblock("rms"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] / T.sqrt(S[0])

    @T.prim_func(s_tir=True)
    def mul(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (M, N), "float32")
        B = T.match_buffer(b, (M, N), "float32")
        C = T.match_buffer(c, (M, N), "float32")
        for i, j in T.grid(M, N):
            with T.sblock("mul"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] * B[vi, vj]

    return sq, rms, mul


def _emit_rmsnorm_row_fp32(compiler, hidden, x_addr, gamma_addr, out_addr,
                           scratch_addr, stage_addr):
    """One row of RMSNorm-with-gamma, entirely via TVM schedule+codegen --
    npu_compiler.py's compile_* methods are not called.

    REDUCE_SUMSQ collapses a WHOLE dim_m x dim_n SRAM tile to one scalar --
    that's the hardware opcode's contract (confirmed directly against
    pe_array.cpp: it sums over both tile axes unconditionally, there is no
    per-row reduction mode), not something this dispatch controls. So this
    processes exactly one real row per call, at the fixed M=16 tile height
    every op here uses, the same convention compile_rmsnorm_row (and
    test_tvm_path.py's own run_rmsnorm) used -- only "row 0" of the 16-row
    tile is real, rows 1-15 are zero padding. A composite whose x has
    multiple rows calls this once per row (see emit_composite below).

    `gamma_addr` must point at a genuine (16, hidden) buffer with gamma
    physically repeated across all 16 rows -- there is no broadcast-on-load
    here. That's a deliberate simplification, not an oversight: the ORIGINAL
    compile_rmsnorm_row tried a stride-based broadcast of a 1-row gamma
    buffer and it read garbage for rows 1-15 (confirmed against rtv_sim
    directly, see git history / prior session notes). Replicating gamma
    once in the DRAM image the caller builds sidesteps that failure mode
    entirely rather than reproducing it.
    """
    M = compiler.tile_dim
    sq, rms, mul = _rmsnorm_prim_fp32(M, hidden)

    codegen.emit(compiler,
                 schedule.schedule_reduce(sq, "sumsq", intrin.register_sumsq_fp32()),
                 {"A": x_addr, "S": scratch_addr},
                 dtype=DType.FP32, kind="reduce")
    # RMSNorm divides by the ROOT MEAN square, not the root SUM -- sumsq only
    # produced sum(x^2). pe_array.cpp's FP32 RMS_SCALE computes
    # sqrt(scale_num / sum) when scale_num is nonzero (else plain 1/sqrt(sum));
    # passing scale_num=hidden turns that into sqrt(N/sum) == 1/sqrt(mean),
    # which is what folds the missing /N into this step.
    codegen.emit(compiler,
                 schedule.schedule_tile_scalar(rms, "rms",
                                               intrin.register_rms_scale_fp32()),
                 {"S": scratch_addr, "A": x_addr, "C": stage_addr},
                 dtype=DType.FP32, kind="elemwise", scale_num=hidden)
    codegen.emit(compiler,
                 schedule.schedule_binary(mul, "mul", intrin.register_mul_fp32()),
                 {"A": stage_addr, "B": gamma_addr, "C": out_addr},
                 dtype=DType.FP32, kind="elemwise")


def _matmul_prim_fp32(M, N, K, a_stride=None, b_stride=None, c_stride=None):
    """One (M,K)x(K,N)->(M,N) FP32 matmul PrimFunc, unscheduled.

    a_stride/b_stride/c_stride (row pitch in ELEMENTS) let an operand be a
    COLUMN SLICE of a wider matrix rather than its own tightly packed
    buffer -- e.g. attention's per-head Q slice out of a (M, n_heads*K)
    projected buffer, where consecutive rows are n_heads*K elements apart,
    not K. Defaults to the tight-packing case (None -> no explicit stride,
    which is what every non-attention caller here still uses).
    """
    @T.prim_func(s_tir=True)
    def mm(a: T.handle, b: T.handle, c: T.handle) -> None:
        # `is not None`, not bare truthiness: TVMScript's ternary handling
        # inside a decorated function body evaluates the condition through
        # its OWN evaluator, which requires an actual Python/TIR bool --
        # passing the raw (possibly-None) stride value as the condition
        # fails with "Expected Python bool or TIR bool, but got NoneType"
        # even though ordinary Python would happily treat None as falsy.
        # Confirmed directly against this TVM build.
        A = T.match_buffer(a, (M, K), "float32",
                           strides=([a_stride, 1] if a_stride is not None else None))
        B = T.match_buffer(b, (K, N), "float32",
                           strides=([b_stride, 1] if b_stride is not None else None))
        C = T.match_buffer(c, (M, N), "float32",
                           strides=([c_stride, 1] if c_stride is not None else None))
        for i, j, k in T.grid(M, N, K):
            with T.sblock("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    return mm


def _emit_matmul_fp32(compiler, M, N, K, a_addr, b_addr, c_addr,
                      a_stride=None, b_stride=None, c_stride=None):
    codegen.emit(compiler,
                 schedule.schedule_matmul(
                     _matmul_prim_fp32(M, N, K, a_stride, b_stride, c_stride),
                     intrin_name=intrin.register_gemm_fp32()),
                 {"A": a_addr, "B": b_addr, "C": c_addr},
                 dtype=DType.FP32, kind="gemm")


def _emit_swiglu_ffn_fp32(compiler, hidden, inter, x_addr, w_gate_addr,
                          w_up_addr, w_down_addr, out_addr):
    """SwiGLU FFN entirely via TVM schedule+codegen -- npu_compiler.py's
    compile_mlp_block is not called.

    Unlike RMSNorm's ALU ops, COMPUTE_GEMM genuinely operates row-wise (it's
    a real systolic multiply) and SiLU/ELEMWISE_MUL are per-element with no
    cross-row reduction, so this handles a real M-row batch correctly up to
    one M-tile (M <= compiler.tile_dim; larger M needs an outer M-tiling
    loop this doesn't have yet -- the same limit the old compile_mlp_block-
    delegating version had, for the same underlying reason).
    """
    M = compiler.tile_dim
    w = DType.BYTES[DType.FP32]   # this whole path is FP32-only; independent
                                  # of whatever compiler.dtype happens to be
    stage_bytes = M * _padded(inter) * w
    gate_stage = compiler.alloc_staging(stage_bytes)
    up_stage = compiler.alloc_staging(stage_bytes)
    act_stage = compiler.alloc_staging(stage_bytes)

    _emit_matmul_fp32(compiler, M, inter, hidden, x_addr, w_gate_addr, gate_stage)
    _emit_matmul_fp32(compiler, M, inter, hidden, x_addr, w_up_addr, up_stage)

    @T.prim_func(s_tir=True)
    def silu(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (M, inter), "float32")
        C = T.match_buffer(c, (M, inter), "float32")
        for i, j in T.grid(M, inter):
            with T.sblock("silu"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] / (T.float32(1.0)
                                        + T.exp(T.float32(0.0) - A[vi, vj]))

    codegen.emit(compiler,
                 schedule.schedule_unary(silu, "silu", intrin.register_silu_fp32()),
                 {"A": gate_stage, "C": gate_stage}, dtype=DType.FP32, kind="elemwise")

    @T.prim_func(s_tir=True)
    def mul(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (M, inter), "float32")
        B = T.match_buffer(b, (M, inter), "float32")
        C = T.match_buffer(c, (M, inter), "float32")
        for i, j in T.grid(M, inter):
            with T.sblock("mul"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] * B[vi, vj]

    codegen.emit(compiler,
                 schedule.schedule_binary(mul, "mul", intrin.register_mul_fp32()),
                 {"A": gate_stage, "B": up_stage, "C": act_stage},
                 dtype=DType.FP32, kind="elemwise")

    _emit_matmul_fp32(compiler, M, hidden, inter, act_stage, w_down_addr, out_addr)


def _emit_matmul_tile_resident_fp32(compiler, K, a_addr, b_addr, c_slot,
                                    a_stride=None, b_stride=None):
    """One (16,K)x(K,16)->(16,16) FP32 GEMM whose C lands directly in SRAM
    output c_slot `c_slot` (0-15, see rtv_isa.py) -- no STORE_TILE, no DRAM
    round trip for the result. A and B still LOAD_TILE from DRAM as usual
    (this only makes the OUTPUT resident); pairs with
    _emit_silu_resident_fp32/_emit_mul_resident_store_fp32 below to chain
    a GEMM straight into elementwise ops without DRAM in between -- see
    _emit_swiglu_ffn_resident_fp32. Does no c_slot bookkeeping of its own;
    the caller must not reuse a c_slot that's still needed elsewhere.

    Legal because COMPUTE_GEMM's C write is genuinely c_slot-addressed in
    hardware (confirmed against src/simulator.cpp's c_slot_addr) -- this
    isn't a software trick layered over a fixed address, it's using the
    ISA's own multi-slot output addressing for what it's for.
    """
    M = compiler.tile_dim
    sch = schedule.schedule_matmul(
        _matmul_prim_fp32(M, M, K, a_stride, b_stride, None),
        intrin_name=intrin.register_gemm_fp32())
    cg = codegen.Codegen(compiler, {"A": a_addr, "B": b_addr, "C": 0},
                         dtype=DType.FP32, kind="gemm")
    cg.skip_tile_io = {"C"}
    cg.c_slot = c_slot
    cg.slot["C"] = SRAMMap.output_slot(c_slot)
    cg.visit(sch.mod["main"].body, {})


def _emit_silu_resident_fp32(compiler, c_slot):
    """SiLU applied IN PLACE to whatever already sits in SRAM output
    c_slot `c_slot` -- no LOAD_TILE/STORE_TILE either side."""
    M = compiler.tile_dim

    @T.prim_func(s_tir=True)
    def silu(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (M, M), "float32")
        C = T.match_buffer(c, (M, M), "float32")
        for i, j in T.grid(M, M):
            with T.sblock("silu"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] / (T.float32(1.0)
                                        + T.exp(T.float32(0.0) - A[vi, vj]))

    sch = schedule.schedule_unary(silu, "silu", intrin.register_silu_fp32())
    cg = codegen.Codegen(compiler, {"A": 0, "C": 0}, dtype=DType.FP32, kind="elemwise")
    addr = SRAMMap.output_slot(c_slot)
    cg.skip_tile_io = {"A", "C"}
    cg.slot["A"] = addr
    cg.slot["C"] = addr
    cg.visit(sch.mod["main"].body, {})


def _emit_mul_resident_store_fp32(compiler, a_c_slot, b_c_slot, out_addr, out_stride=None):
    """ELEMWISE_MUL of two already-resident SRAM output c_slots (result
    left in place at a_c_slot's address, matching _elemwise's own
    in-place-on-A semantics), then STORE_TILE the (16,16) result out to
    DRAM at out_addr -- the one DRAM write in this whole resident chain,
    since the down-projection GEMM that consumes it still goes through
    _emit_matmul_fp32's ordinary DRAM-backed path (see
    _emit_swiglu_ffn_resident_fp32's docstring for why only the
    projection->activation link is made resident, not this one too).
    out_stride lets out_addr be one column-tile of a wider (M, N) buffer.
    """
    M = compiler.tile_dim

    @T.prim_func(s_tir=True)
    def mul(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (M, M), "float32")
        B = T.match_buffer(b, (M, M), "float32")
        C = T.match_buffer(c, (M, M), "float32",
                           strides=([out_stride, 1] if out_stride is not None else None))
        for i, j in T.grid(M, M):
            with T.sblock("mul"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] * B[vi, vj]

    sch = schedule.schedule_binary(mul, "mul", intrin.register_mul_fp32())
    cg = codegen.Codegen(compiler, {"A": 0, "B": 0, "C": out_addr},
                         dtype=DType.FP32, kind="elemwise")
    a_addr = SRAMMap.output_slot(a_c_slot)
    b_addr = SRAMMap.output_slot(b_c_slot)
    cg.skip_tile_io = {"A", "B"}
    cg.slot["A"] = a_addr
    cg.slot["B"] = b_addr
    cg.slot["C"] = a_addr   # EXEC_ALU writes in place at A's slot; STORE_TILE reads from there
    cg.visit(sch.mod["main"].body, {})


def _emit_swiglu_ffn_resident_fp32(compiler, hidden, inter, x_addr, w_gate_addr,
                                   w_up_addr, w_down_addr, out_addr):
    """SwiGLU FFN with the gate/up projections' intermediate activations
    kept RESIDENT IN SRAM across GEMM -> SiLU -> ELEMWISE_MUL, instead of
    _emit_swiglu_ffn_fp32's round trip through DRAM between every op --
    TODO 1.2's "SwiGLU intermediate activations staying resident in SRAM".

    Only that link is made resident, not the whole FFN: the down-projection
    GEMM still reads its input (act_stage) fresh from DRAM via the ordinary
    _emit_matmul_fp32. Going further would mean the down-projection's own
    K-loop pulling tiles directly out of these same c_slots instead of
    LOAD_TILE-ing them from DRAM -- real additional surgery on
    codegen.py's K-loop addressing, not a natural extension of what's
    here, and not needed to close out THIS TODO item (which named the
    projection->activation link specifically).

    Works per 16x16 TILE, not per whole (M, inter) buffer: unlike
    _emit_swiglu_ffn_fp32 (which hands TVM's schedule the whole (M, inter)
    shape and lets it loop internally, always reusing ONE fixed SRAM
    address per N-tile -- fine when every tile round-trips through DRAM
    anyway, since nothing needs more than one tile alive at once), true
    residency needs inter/16 DISTINCT c_slots alive SIMULTANEOUSLY (one
    per gate tile) plus inter/16 more for up, so each tile needs its OWN
    call with its OWN pinned c_slot -- hence the explicit Python loop over
    n_tiles here, in the same spirit as _emit_gqa_attention_fp32's
    explicit per-head loop (also something DFPattern/a single TIR schedule
    can't express).

    Requires inter % compiler.tile_dim == 0 (exact tiles, no padding
    remainder to special-case) and 2*(inter/tile_dim) <= 16 (gate's tiles
    and up's tiles must simultaneously fit in the hardware's 16 c_slots) --
    raises NotImplementedError otherwise; callers past that limit should
    use _emit_swiglu_ffn_fp32 instead.
    """
    M = compiler.tile_dim
    w = DType.BYTES[DType.FP32]
    if inter % M != 0:
        raise NotImplementedError(
            f"_emit_swiglu_ffn_resident_fp32 needs inter ({inter}) to be a "
            f"multiple of compiler.tile_dim ({M})"
        )
    n_tiles = inter // M
    if 2 * n_tiles > 16:
        raise NotImplementedError(
            f"_emit_swiglu_ffn_resident_fp32 needs 2*inter/tile_dim "
            f"({2 * n_tiles}) output c_slots resident simultaneously; "
            f"hardware only has 16. inter={inter} is too large for this "
            f"path -- use _emit_swiglu_ffn_fp32 instead."
        )
    gate_slots = list(range(n_tiles))
    up_slots = list(range(n_tiles, 2 * n_tiles))
    act_stage = compiler.alloc_staging(M * inter * w)

    for t in range(n_tiles):
        _emit_matmul_tile_resident_fp32(compiler, hidden, x_addr,
                                        w_gate_addr + t * M * w, gate_slots[t],
                                        b_stride=inter)
        _emit_matmul_tile_resident_fp32(compiler, hidden, x_addr,
                                        w_up_addr + t * M * w, up_slots[t],
                                        b_stride=inter)
    for t in range(n_tiles):
        _emit_silu_resident_fp32(compiler, gate_slots[t])
        _emit_mul_resident_store_fp32(compiler, gate_slots[t], up_slots[t],
                                      act_stage + t * M * w, out_stride=inter)

    _emit_matmul_fp32(compiler, M, hidden, inter, act_stage, w_down_addr, out_addr)


def _emit_attention_core_fp32(compiler, seq_k, head_dim, q_addr, k_addr, v_addr,
                              out_addr, mask_addr=None,
                              q_stride=None, k_stride=None, v_stride=None,
                              out_stride=None):
    """Scaled dot-product attention's core -- softmax(Q@K [+ mask]) @ V --
    entirely via TVM schedule+codegen. Dispatches patterns.py's
    "attention"/"attention_masked" kinds, which match on already-projected
    Q/K/V (see emit_composite's docstring on why the Q/K/V projections and
    the output projection are NOT part of this -- a granularity decision,
    not an oversight).

    No transpose here, deliberately: _attention_pattern matches the LITERAL
    op sequence matmul(q, k) -> softmax -> matmul(_, v), with no permute
    node inside the pattern at all (patterns.py's get_patterns() has no
    transpose pattern either). So whatever value the "k" wildcard binds to
    in a real graph is already shaped to multiply directly against q -- if
    a real model computes K^T via an explicit permute_dims first, that op
    sits OUTSIDE this composite's boundary (unclaimed by any pattern here,
    so it runs elsewhere) and "k" is its output. Transposing it again here
    would be wrong, not just redundant.

    seq_k and head_dim may each be ANY positive multiple of compiler.tile_dim
    now (checked directly: REDUCE's accumulate flag already lets rowmax/
    rowsum span multiple 16-wide column tiles -- the same mechanism
    test_tvm_path.py's tvm_sumsq_16x64/tvm_rowmax_16x64 already prove at
    N=64 -- and GEMM already tiles M/N/K generically, so scores/probs being
    wider than one tile, or the Q@K/P@V reduction axis spanning more than
    one K-tile, needed no codegen.py changes -- just not hardcoding this
    function's own internal buffers to (M, M)). What's still NOT here: a
    genuine multi-row batch of DIFFERENT real queries. Q is a single real
    query ROW, REPLICATED across all 16 tile rows rather than padded with
    zeros -- not a shortcut for its own sake. REDUCE always collapses a
    WHOLE dim_m x dim_n SRAM tile to one scalar regardless of dtype (the
    same hardware-opcode limit RMSNorm's dispatch hit -- confirmed directly
    against pe_array.cpp: there is no per-row reduction mode) AND
    codegen.py's _reduce()/_tile_scalar() hardcode dim_m=dim_n=TILE in
    every instruction they emit, so a genuine dim_m=1 per-row softmax
    across multiple DIFFERENT rows (what the spec doc describes for the
    hand-written compiler) isn't reachable through this codegen without
    generalizing it further -- not done here. Replicating the one real
    query into all 16 rows instead makes the whole-tile-shared max/sum
    mathematically IDENTICAL to that row's own softmax: every row holds
    the same data, so sharing the reduction changes nothing. That's a
    SEPARATE limitation from seq_k/head_dim size, which this now handles.

    No 1/sqrt(head_dim) scaling: patterns.py's _attention_pattern matches
    exactly matmul->softmax->matmul with no scale step, so a real graph
    either has no scaling or applies it to Q/K before this composite's
    wildcards -- either way it's outside what this composite receives.

    q_stride/k_stride/v_stride/out_stride (row pitch in ELEMENTS) let any
    of Q/K/V/out be a COLUMN SLICE of a wider matrix instead of its own
    tight (M, head_dim) buffer -- exactly what a real per-head Q/K/V/O
    tensor is: head h occupies columns [h*head_dim:(h+1)*head_dim] of a
    (M, n_heads*head_dim) projected buffer, so consecutive rows of that
    ONE head's slice are n_heads*head_dim elements apart, not head_dim.
    See _emit_gqa_attention_fp32, which is what actually sets these when
    driving multiple heads over one shared wide buffer.
    """
    M = compiler.tile_dim
    if seq_k % M != 0 or head_dim % M != 0:
        raise NotImplementedError(
            f"attention dispatch needs seq_k and head_dim as multiples of "
            f"the {M}-wide tile; got seq_k={seq_k}, head_dim={head_dim}"
        )

    scores_stage = compiler.alloc_staging(M * seq_k * 4)
    masked_stage = compiler.alloc_staging(M * seq_k * 4) if mask_addr else scores_stage
    probs_stage = compiler.alloc_staging(M * seq_k * 4)
    sum_input_stage = compiler.alloc_staging(M * seq_k * 4)
    row0_mask_addr = compiler.alloc_staging(M * seq_k * 4)
    rowmax_scratch = compiler.alloc_staging(4)
    rowsum_scratch = compiler.alloc_staging(4)

    # REDUCE_MAX is idempotent under row replication (max of 16 identical
    # rows == that row's own max), but REDUCE's plain sum is NOT: summing 16
    # identical rows gives 16x the true row sum. Confirmed by dumping probs
    # mid-pipeline -- every entry came out at exactly 1/16 of the reference,
    # uniformly. Fixed by masking the exp'd tile down to one real row (row
    # 0) before the sum, the same way every other buffer in this codebase
    # keeps only row 0 real -- rows 1-15 contribute 0 to the sum instead of
    # 15 extra copies of it. row0_mask is a genuine compile-time constant
    # (1.0 on row 0, 0.0 elsewhere), injected via compiler._const_regions --
    # the same mechanism compile_lut's table and RoPE's cos/sin already use;
    # the caller must copy it into the DRAM image (see this dispatch's
    # tests for the pattern).
    row0_mask = np.zeros((M, seq_k), dtype=np.float32)
    row0_mask[0, :] = 1.0
    compiler._const_regions[row0_mask_addr] = row0_mask.tobytes()

    _emit_matmul_fp32(compiler, M, seq_k, head_dim, q_addr, k_addr, scores_stage,
                      a_stride=q_stride, b_stride=k_stride)

    if mask_addr:
        @T.prim_func(s_tir=True)
        def add_mask(a: T.handle, b: T.handle, c: T.handle) -> None:
            A = T.match_buffer(a, (M, seq_k), "float32")
            B = T.match_buffer(b, (M, seq_k), "float32")
            C = T.match_buffer(c, (M, seq_k), "float32")
            for i, j in T.grid(M, seq_k):
                with T.sblock("add"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = A[vi, vj] + B[vi, vj]

        codegen.emit(compiler,
                     schedule.schedule_binary(add_mask, "add",
                                              intrin.register_add_fp32()),
                     {"A": scores_stage, "B": mask_addr, "C": masked_stage},
                     dtype=DType.FP32, kind="elemwise")

    @T.prim_func(s_tir=True)
    def rmax(a: T.handle, s: T.handle) -> None:
        A = T.match_buffer(a, (M, seq_k), "float32")
        S = T.match_buffer(s, (1,), "float32")
        for i, j in T.grid(M, seq_k):
            with T.sblock("rowmax"):
                vi, vj = T.axis.remap("RR", [i, j])
                with T.init():
                    S[0] = T.float32(-3.0e38)
                S[0] = T.max(S[0], A[vi, vj])

    codegen.emit(compiler,
                 schedule.schedule_reduce(rmax, "rowmax",
                                          intrin.register_rowmax_fp32()),
                 {"A": masked_stage, "S": rowmax_scratch},
                 dtype=DType.FP32, kind="reduce")

    @T.prim_func(s_tir=True)
    def esub(s: T.handle, a: T.handle, c: T.handle) -> None:
        S = T.match_buffer(s, (1,), "float32")
        A = T.match_buffer(a, (M, seq_k), "float32")
        C = T.match_buffer(c, (M, seq_k), "float32")
        for i, j in T.grid(M, seq_k):
            with T.sblock("expsub"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.exp(A[vi, vj] - S[0])

    codegen.emit(compiler,
                 schedule.schedule_tile_scalar(esub, "expsub",
                                               intrin.register_exp_sub_fp32()),
                 {"S": rowmax_scratch, "A": masked_stage, "C": probs_stage},
                 dtype=DType.FP32, kind="elemwise")

    @T.prim_func(s_tir=True)
    def mask_to_row0(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (M, seq_k), "float32")
        B = T.match_buffer(b, (M, seq_k), "float32")
        C = T.match_buffer(c, (M, seq_k), "float32")
        for i, j in T.grid(M, seq_k):
            with T.sblock("mul"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] * B[vi, vj]

    codegen.emit(compiler,
                 schedule.schedule_binary(mask_to_row0, "mul",
                                          intrin.register_mul_fp32()),
                 {"A": probs_stage, "B": row0_mask_addr, "C": sum_input_stage},
                 dtype=DType.FP32, kind="elemwise")

    @T.prim_func(s_tir=True)
    def rsum(a: T.handle, s: T.handle) -> None:
        A = T.match_buffer(a, (M, seq_k), "float32")
        S = T.match_buffer(s, (1,), "float32")
        for i, j in T.grid(M, seq_k):
            with T.sblock("rowsum"):
                vi, vj = T.axis.remap("RR", [i, j])
                with T.init():
                    S[0] = T.float32(0)
                S[0] = S[0] + A[vi, vj]

    codegen.emit(compiler,
                 schedule.schedule_reduce(rsum, "rowsum",
                                          intrin.register_rowsum_fp32()),
                 {"A": sum_input_stage, "S": rowsum_scratch},
                 dtype=DType.FP32, kind="reduce")

    @T.prim_func(s_tir=True)
    def sdiv(a: T.handle, s: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (M, seq_k), "float32")
        S = T.match_buffer(s, (1,), "float32")
        C = T.match_buffer(c, (M, seq_k), "float32")
        for i, j in T.grid(M, seq_k):
            with T.sblock("sdiv"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] / S[0]

    codegen.emit(compiler,
                 schedule.schedule_tile_scalar(sdiv, "sdiv",
                                               intrin.register_scalar_div_fp32()),
                 {"A": probs_stage, "S": rowsum_scratch, "C": probs_stage},
                 dtype=DType.FP32, kind="elemwise")

    _emit_matmul_fp32(compiler, M, head_dim, seq_k, probs_stage, v_addr, out_addr,
                      b_stride=v_stride, c_stride=out_stride)


def _emit_softmax_rows_fp32(compiler, tile_addr, n_rows, width):
    """Row-INDEPENDENT softmax over an (n_rows, width) DRAM region, each
    row getting its OWN max/sum -- what genuinely batched attention (real,
    DIFFERENT queries, e.g. Fp32AttentionCase) needs and
    _emit_attention_core_fp32's rowmax/expsub/rowsum/sdiv steps don't
    provide: those rely on every row already being either zero or
    identical to every other row (row0-real-rest-zero / all-replicated),
    so a whole-tile REDUCE happens to equal each row's own reduction too.
    For genuinely different rows that's simply wrong -- REDUCE/SCALAR_ALU
    are hardware ops that collapse/broadcast across a WHOLE dim_m x dim_n
    SRAM region with no per-row mode at all (confirmed directly against
    pe_array.cpp: dim_m/dim_n are real sizes, but there is exactly one
    scalar output / one scalar applied regardless of dim_m).

    Can't go through schedule.py's usual schedule_reduce/
    schedule_tile_scalar either: they unconditionally split every axis
    into TILE(16)-sized chunks (`sch.split(i, factors=[None, TILE])`),
    which cannot express a 1-row axis -- so this drives codegen.py's
    LOAD_TILE/REDUCE/SCALAR_ALU/STORE_TILE primitives directly instead,
    at `dim_m=1`, once per row, the same "hand-drive instructions no TVM
    schedule primitive can express" approach _gemm_k_loop and
    _emit_swiglu_ffn_resident_fp32 already use. A fresh Codegen per row
    keeps each row's REDUCE accumulate-state independent (row 2's max
    must not accumulate onto row 1's).

    Scoped to width == compiler.tile_dim (one column tile, e.g.
    seq_k == tile_dim) -- exactly what Fp32AttentionCase needs; a real
    seq_k > tile_dim batched softmax would additionally need each row's
    reduction to loop column-tiles with REDUCE's accumulate flag, which
    this doesn't attempt.
    """
    M = compiler.tile_dim
    if width != M:
        raise NotImplementedError(
            f"_emit_softmax_rows_fp32 needs width == compiler.tile_dim "
            f"({M}); got {width}"
        )
    w = DType.BYTES[DType.FP32]
    row_slot = SRAMMap.input_slot(0, w)
    max_scalar = SRAMMap.SCALAR_SCRATCH_BASE
    sum_scalar = SRAMMap.SCALAR_SCRATCH_BASE + 4

    for r in range(n_rows):
        row_addr = tile_addr + r * width * w
        cg = codegen.Codegen(compiler, {}, dtype=DType.FP32, kind="elemwise")
        cg.dim_m = 1
        cg.slot["A"] = row_slot
        cg._emit_word(encode(opcode=Opcode.LOAD_TILE, dep_id=codegen.DEP.get("A", 3),
                             dram_addr=row_addr, sram_addr=row_slot,
                             dim_m=1, dim_n=M * w, stride=width * w,
                             dtype=int(DType.FP32)))
        cg._reduce(ActivationMode.REDUCE_MAX, scalar_addr=max_scalar)
        cg._tile_scalar(ActivationMode.EXP_SUB, scalar_addr=max_scalar)
        cg._reduce(ActivationMode.NONE, scalar_addr=sum_scalar)
        cg._tile_scalar(ActivationMode.SCALAR_DIV, scalar_addr=sum_scalar)
        cg._emit_word(encode(opcode=Opcode.STORE_TILE, dram_addr=row_addr,
                             sram_addr=row_slot, dim_m=1, dim_n=M * w,
                             stride=width * w, dtype=int(DType.FP32)))


def _emit_attention_core_batched_fp32(compiler, seq_k, head_dim, q_addr, k_addr,
                                      v_addr, out_addr, mask_addr=None,
                                      q_stride=None, k_stride=None, v_stride=None,
                                      out_stride=None):
    """Like _emit_attention_core_fp32, but for a genuine batch of
    DIFFERENT query rows (real prefill / Fp32AttentionCase), not one real
    query replicated 16x.

    Only the SOFTMAX NORMALIZATION differs -- GEMM (Q@K^T, P@V) is already
    correct for genuinely distinct rows with NO changes: COMPUTE_GEMM is
    inherently row-parallel (each output row depends only on its own input
    row), unlike REDUCE, so the exact same _emit_matmul_fp32 calls the
    non-batched version uses are reused verbatim here. rowmax/expsub/
    rowsum/sdiv go through _emit_softmax_rows_fp32 instead -- see its
    docstring for why a genuine per-row pass is unavoidable here.

    Scoped to seq_k == head_dim == compiler.tile_dim (one tile each,
    exactly what Fp32AttentionCase needs) -- _emit_softmax_rows_fp32
    doesn't handle a wider seq_k, so neither does this.
    """
    M = compiler.tile_dim
    if seq_k != M or head_dim != M:
        raise NotImplementedError(
            f"_emit_attention_core_batched_fp32 needs seq_k == head_dim == "
            f"compiler.tile_dim ({M}); got seq_k={seq_k}, head_dim={head_dim}"
        )

    scores_stage = compiler.alloc_staging(M * seq_k * 4)
    masked_stage = compiler.alloc_staging(M * seq_k * 4) if mask_addr else scores_stage

    _emit_matmul_fp32(compiler, M, seq_k, head_dim, q_addr, k_addr, scores_stage,
                      a_stride=q_stride, b_stride=k_stride)
    if mask_addr:
        _emit_residual_add_fp32(compiler, scores_stage, mask_addr, masked_stage, seq_k)

    _emit_softmax_rows_fp32(compiler, masked_stage, M, seq_k)

    _emit_matmul_fp32(compiler, M, head_dim, seq_k, masked_stage, v_addr, out_addr,
                      b_stride=v_stride, c_stride=out_stride)


def _emit_gqa_attention_fp32(compiler, n_heads, n_kv_heads, seq_k, head_dim,
                             q_addr, k_addr, v_addr, out_addr):
    """
    Multi-head attention with GQA, over already-projected Q/K/V, driven
    directly rather than through patterns.py's pattern-matched dispatch.

    Why this is a separate entry point rather than something
    emit_composite's "attention" kind discovers automatically: the
    granularity decision for that dispatch (see its docstring) is that it
    matches the LITERAL pattern matmul(q,k)->softmax->matmul(_,v) -- one 2D
    op, no head structure. A real per-head-unrolled Relax graph would
    produce n_heads SEPARATE "attention" composites (one per head, each a
    genuine 2D op), which emit_composite already handles correctly one at a
    time. This function is instead for a caller that already KNOWS its
    head/GQA structure from the model config -- e.g. a future whole-model
    driver, informed by hidden_size/num_heads/num_kv_heads the same way
    compile_attention_block is, not by pattern inference. DFPattern isn't
    well-suited to discovering "this is head 3 of 4, sharing KV head 1 of
    2" from graph structure alone.

    Q and out are (M, n_heads*head_dim) buffers -- head h occupies COLUMNS
    [h*head_dim:(h+1)*head_dim], the real transformer layout (a
    projection's natural output / what feeds W_o). Verified directly that
    _tile_copy respects an explicit match_buffer stride for exactly this
    reason (a per-head column slice's rows are n_heads*head_dim elements
    apart in the wide buffer, not head_dim) -- confirmed against rtv_sim
    before wiring this up, since codegen.py silently ignored buffer strides
    before that fix.

    K and V are NOT column-sliced from one wide buffer the same way, and
    that's a real design point, not an inconsistency: _emit_attention_core_
    fp32 computes matmul(q, k) LITERALLY (see its own docstring -- the
    pattern has no transpose node, so "k" must already be oriented for
    direct multiplication against q, i.e. (head_dim, seq_k), not the
    (seq_k, head_dim) a K-projection naturally produces). Getting there
    needs an explicit transpose upstream, outside this composite's
    boundary -- and a transpose materializes a fresh buffer, it doesn't
    hand back a strided view of the original wide K-projection output. So
    K's realistic layout here is n_kv_heads separate CONTIGUOUS
    (head_dim, seq_k) blocks (kv_h's block immediately followed by kv_h+1's),
    addressed by a plain per-head offset -- no stride override needed, same
    as V's separate contiguous (seq_k, head_dim) blocks (V is used as-is,
    with no transpose, in matmul(probs, v)).

    GQA grouping mirrors compile_attention_block's own convention: query
    head h reads KV head h // (n_heads // n_kv_heads), so with n_heads=4,
    n_kv_heads=2 heads 0-1 share KV head 0 and heads 2-3 share KV head 1.

    seq_k/head_dim may each be any multiple of compiler.tile_dim now (see
    _emit_attention_core_fp32's docstring -- this needed no changes here,
    it just passes both through). Inherits that function's remaining limit
    unchanged: one real query row per head (replicated across the 16-row
    tile, not a genuine multi-row batch -- that needs codegen.py's dim_m=1
    generalization, same blocker as RMSNorm's per-row dispatch), and no
    1/sqrt(head_dim) scaling (assumed applied upstream if a real graph
    needs it).
    """
    if n_heads % n_kv_heads != 0:
        raise ValueError(f"n_heads={n_heads} must be a multiple of "
                         f"n_kv_heads={n_kv_heads} for GQA grouping")
    q_dim = n_heads * head_dim
    rep = n_heads // n_kv_heads
    w = DType.BYTES[DType.FP32]
    k_block_bytes = head_dim * seq_k * w   # one KV head's contiguous K block
    v_block_bytes = seq_k * head_dim * w   # one KV head's contiguous V block

    for qh in range(n_heads):
        kv_h = qh // rep
        _emit_attention_core_fp32(
            compiler, seq_k, head_dim,
            q_addr=q_addr + qh * head_dim * w,
            k_addr=k_addr + kv_h * k_block_bytes,
            v_addr=v_addr + kv_h * v_block_bytes,
            out_addr=out_addr + qh * head_dim * w,
            q_stride=q_dim, out_stride=q_dim,
        )


def _emit_gqa_attention_batched_fp32(compiler, n_heads, n_kv_heads, seq_k, head_dim,
                                     q_addr, k_addr, v_addr, out_addr):
    """_emit_gqa_attention_fp32, but for a genuine batch of seq_k DIFFERENT
    query/key/value rows per head (real prefill / Fp32AttentionCase)
    instead of one real query replicated across the tile -- see
    _emit_attention_core_batched_fp32/_emit_softmax_rows_fp32 for why the
    softmax needs a real per-row pass here. Identical to
    _emit_gqa_attention_fp32 in every other respect (same GQA head
    grouping, same Q/out wide-buffer column-slicing, same K/V
    separate-contiguous-per-head-block layout) -- only the per-head call
    swaps to the batched attention core.
    """
    if n_heads % n_kv_heads != 0:
        raise ValueError(f"n_heads={n_heads} must be a multiple of "
                         f"n_kv_heads={n_kv_heads} for GQA grouping")
    q_dim = n_heads * head_dim
    rep = n_heads // n_kv_heads
    w = DType.BYTES[DType.FP32]
    k_block_bytes = head_dim * seq_k * w
    v_block_bytes = seq_k * head_dim * w

    for qh in range(n_heads):
        kv_h = qh // rep
        _emit_attention_core_batched_fp32(
            compiler, seq_k, head_dim,
            q_addr=q_addr + qh * head_dim * w,
            k_addr=k_addr + kv_h * k_block_bytes,
            v_addr=v_addr + kv_h * v_block_bytes,
            out_addr=out_addr + qh * head_dim * w,
            q_stride=q_dim, out_stride=q_dim,
        )


def _emit_broadcast_row0_fp32(compiler, in_addr, out_addr, width):
    """Replicate row 0 of a (16, width) "row0 real, rest zero" buffer into
    all 16 rows -- bridges the layer-level convention (row0 real, rest
    zero, the one _emit_rmsnorm_row_fp32/emit_composite's "rmsnorm" kind
    actually require -- REDUCE collapses the WHOLE tile, so a genuinely
    replicated input gets summed 16x too much, confirmed directly: it
    produced exactly 1/4 of the correct RMSNorm output) into
    _emit_attention_core_fp32's convention (all 16 rows identical, needed
    so REDUCE_MAX/REDUCE's whole-tile collapse gives that row's own
    softmax rather than mixing padding into it).

    No hardware broadcast primitive exists for this (checked: a stride of
    0 on LOAD_TILE is reinterpreted by the simulator as "not set, use the
    natural width," not a literal zero-stride broadcast). Done instead via
    GEMM against a compile-time constant "one-hot" matrix: broadcast[m,0]=1,
    broadcast[m,k>0]=0 for every row m, so
        out[m, n] = sum_k broadcast[m,k] * in[k, n] = in[0, n]
    for every m -- exactly the replication wanted, using only the existing
    GEMM primitive and a constant injected via compiler._const_regions
    (same mechanism the row0 padding-mask inside the attention core uses).
    """
    M = compiler.tile_dim
    bcast_addr = compiler.alloc_staging(M * M * 4)
    bcast = np.zeros((M, M), dtype=np.float32)
    bcast[:, 0] = 1.0
    compiler._const_regions[bcast_addr] = bcast.tobytes()
    _emit_matmul_fp32(compiler, M, width, M, bcast_addr, in_addr, out_addr)


def _emit_mask_to_row0_fp32(compiler, in_addr, out_addr, width):
    """The inverse of _emit_broadcast_row0_fp32: zero out rows 1-15 of a
    (16, width) buffer, keeping only row 0 -- bridges
    _emit_attention_core_fp32's "all rows identical" output convention back
    to the layer-level "row0 real, rest zero" convention downstream RMSNorm
    calls require. Same compile-time-constant-mask technique the attention
    core's own row0_mask already uses for its internal sum step."""
    M = compiler.tile_dim
    mask_addr = compiler.alloc_staging(M * width * 4)
    mask = np.zeros((M, width), dtype=np.float32)
    mask[0, :] = 1.0
    compiler._const_regions[mask_addr] = mask.tobytes()

    @T.prim_func(s_tir=True)
    def mul(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (M, width), "float32")
        B = T.match_buffer(b, (M, width), "float32")
        C = T.match_buffer(c, (M, width), "float32")
        for i, j in T.grid(M, width):
            with T.sblock("mul"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] * B[vi, vj]

    codegen.emit(compiler,
                 schedule.schedule_binary(mul, "mul", intrin.register_mul_fp32()),
                 {"A": in_addr, "B": mask_addr, "C": out_addr},
                 dtype=DType.FP32, kind="elemwise")


def _emit_kv_cache_append_fp32(compiler, cache_tile_addr, new_row_bcast_addr, row, width):
    """Insert one new row (a fresh decode-step token's K or V vector) into
    row `row` of an EXISTING (16, width) cache tile in DRAM, leaving the
    other 15 rows exactly as they were -- the actual "append a token"
    primitive a KV-cache needs.

    LOAD_TILE/STORE_TILE only move WHOLE 16x16 tiles -- there's no
    sub-tile partial-row write opcode (confirmed against pe_array.cpp /
    the ISA doc's LOAD_TILE/STORE_TILE fields) -- so this can't literally
    write one row. Uses the same "constant matrix via GEMM" trick as
    _emit_broadcast_row0_fp32/_emit_mask_to_row0_fp32/_emit_rope_fp32
    instead:

        blended = mask_keep @ old_tile + mask_insert @ new_row_bcast

    mask_keep is IDENTITY with a 0 at (row, row); for a DIAGONAL matrix,
    (mask_keep @ old_tile)[i,:] = mask_keep[i,i] * old_tile[i,:], so this
    reproduces every row of old_tile UNCHANGED except row `row`, which
    becomes exactly zero. mask_insert is ALL ZERO except a 1 at
    (row, row); (mask_insert @ new_row_bcast)[i,:] is zero for every
    i != row and new_row_bcast[row,:] for i == row -- exactly the new
    row's data (since new_row_bcast has it replicated across all 16 rows,
    the way GEMM needs a full operand tile). Adding the two gives row
    `row` = the new data, every other row = the old data, untouched.

    Both mask matrices are compile-time constants (row is a Python int,
    consistent with this whole codebase's per-decode-step recompilation).

    new_row_bcast_addr must already be a (16, width) buffer with the new
    row's real data REPLICATED across all 16 rows -- see
    _emit_broadcast_row0_fp32 if the caller only has it as row 0.
    cache_tile_addr is read AND written (the blended tile replaces the
    old one in place).
    """
    M = compiler.tile_dim
    if not (0 <= row < M):
        raise ValueError(f"row must be in [0, {M}); got {row}")

    mask_keep = np.eye(M, dtype=np.float32)
    mask_keep[row, row] = 0.0
    mask_insert = np.zeros((M, M), dtype=np.float32)
    mask_insert[row, row] = 1.0
    keep_addr = compiler.alloc_staging(M * M * 4)
    insert_addr = compiler.alloc_staging(M * M * 4)
    compiler._const_regions[keep_addr] = mask_keep.tobytes()
    compiler._const_regions[insert_addr] = mask_insert.tobytes()

    kept = compiler.alloc_staging(M * width * 4)
    inserted = compiler.alloc_staging(M * width * 4)
    _emit_matmul_fp32(compiler, M, width, M, keep_addr, cache_tile_addr, kept)
    _emit_matmul_fp32(compiler, M, width, M, insert_addr, new_row_bcast_addr, inserted)
    _emit_residual_add_fp32(compiler, kept, inserted, cache_tile_addr, width)


def _emit_kv_cache_tail_mask_fp32(compiler, seq_k, valid_len):
    """Build (and register as a compile-time constant) an additive
    (16, seq_k) mask for attention over a KV-cache that isn't fully
    populated yet: 0.0 for the first `valid_len` columns (real cached
    tokens), -1e30 for the rest (the current tile's not-yet-written tail,
    or wholly-unwritten future tiles) -- fed to _emit_attention_core_fp32
    as `mask_addr` the same way causal masking already is. -1e30 rather
    than -inf: EXP_SUB's FP32 branch already special-cases non-finite
    inputs to 0 (see pe_array.cpp), but a plain finite very-negative
    number reaches the identical result (exp of a huge negative number
    underflows to 0.0) without relying on that special case, and stays
    finite through the ADD that applies it.

    seq_k must be a multiple of compiler.tile_dim (attention's own
    requirement); valid_len is the number of REAL tokens written so far
    (0 < valid_len <= seq_k) -- i.e. the current decode position + 1.
    Returns the DRAM address of the mask.
    """
    M = compiler.tile_dim
    if seq_k % M != 0:
        raise NotImplementedError(
            f"_emit_kv_cache_tail_mask_fp32 needs seq_k ({seq_k}) as a "
            f"multiple of compiler.tile_dim ({M})"
        )
    if not (0 < valid_len <= seq_k):
        raise ValueError(f"valid_len must be in (0, {seq_k}]; got {valid_len}")

    mask = np.zeros((M, seq_k), dtype=np.float32)
    mask[:, valid_len:] = -1e30
    mask_addr = compiler.alloc_staging(M * seq_k * 4)
    compiler._const_regions[mask_addr] = mask.tobytes()
    return mask_addr


def _emit_transpose_fp32(compiler, in_addr, out_addr, in_stride=None, out_stride=None):
    """One 16x16 FP32 tile transpose: out = in^T.

    Square, single-tile only -- transposing a matrix wider than one tile
    (as a SINGLE call) needs the tile-addressing generalization
    schedule_transpose's own docstring already flags (which tile lands
    where changes, not just its contents), not attempted here; a wider
    transpose is instead built by calling this once per column-tile of
    the input, each writing to its own column-tile of the output (see
    test_kv_cache.py's per-tile transpose loop for a K-cache spanning
    more than one tile) -- which is exactly what out_stride is for.

    in_stride lets `in` be a column slice of a wider matrix (e.g. one
    head's slice of a (M, n_kv_heads*head_dim) K-projection output), the
    same mechanism attention's Q/out already use. out_stride is the
    mirror for the OUTPUT: lets `out` be a column slice of a wider
    (M, N) destination instead of its own tightly-packed (M, M) region --
    e.g. writing tile t's transpose into columns [t*M:(t+1)*M] of a
    (head_dim, seq_k) K-cache buffer with seq_k > M. Missing this was a
    REAL bug (same shape as _emit_residual_add_fp32's missing out_stride,
    caught the same way): without it, STORE_TILE's pitch defaulted to M
    regardless of the wider buffer's actual row length, so tile 1's
    transpose landed at the wrong byte offset within each row instead of
    immediately after tile 0's columns -- confirmed directly building a
    2-tile KV-cache and finding every step past the first tile boundary
    wrong, while every single-tile case (including the transpose used
    inside the attention BLOCK's own per-head loop, always one tile) was
    unaffected -- that's the only reason this had gone unnoticed.
    """
    M = compiler.tile_dim

    @T.prim_func(s_tir=True)
    def transpose(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (M, M), "float32",
                           strides=([in_stride, 1] if in_stride is not None else None))
        C = T.match_buffer(c, (M, M), "float32",
                           strides=([out_stride, 1] if out_stride is not None else None))
        for i, j in T.grid(M, M):
            with T.sblock("transpose"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vj, vi]

    codegen.emit(compiler,
                 schedule.schedule_unary(transpose, "transpose",
                                         intrin.register_transpose_fp32()),
                 {"A": in_addr, "C": out_addr}, dtype=DType.FP32, kind="elemwise")


def _emit_copy_slice_fp32(compiler, in_addr, out_addr, in_stride):
    """Copy a (16, head_dim) column slice of a wider matrix into its own
    tightly-packed buffer -- de-interleaving, not transposing.

    Why this exists: _emit_gqa_attention_fp32 expects K and V as n_kv_heads
    SEPARATE CONTIGUOUS blocks (see its docstring). K gets that layout as a
    side effect of the transpose every head needs anyway (_emit_transpose_
    fp32 reads a column slice and writes a fresh contiguous block). V needs
    no transpose, but a caller whose V projection is one WIDE (M,
    n_kv_heads*head_dim) buffer -- the natural shape straight out of a
    W_v projection -- still has to get it into the same separate-blocks
    layout GQA expects, or GQA's per-head byte offset lands mid-row instead
    of at a head boundary. Found by getting this wrong: passing the wide
    buffer straight through gave every head an IDENTICAL, wrong context
    (worked out to be reading rows of the wide buffer at the wrong pitch,
    not respecting head boundaries).

    Implemented as a GEMM against a compile-time IDENTITY matrix, reading
    `in` with `in_stride` (the wide buffer's true row pitch) -- the same
    "constant matrix via GEMM" trick _emit_broadcast_row0_fp32 uses, just
    with an identity instead of a one-hot broadcast matrix, since here the
    goal is a faithful copy, not a replication.
    """
    M = compiler.tile_dim
    ident_addr = compiler.alloc_staging(M * M * 4)
    ident = np.eye(M, dtype=np.float32)
    compiler._const_regions[ident_addr] = ident.tobytes()
    _emit_matmul_fp32(compiler, M, M, M, ident_addr, in_addr, out_addr, b_stride=in_stride)


def _emit_mul_fp32(compiler, a_addr, b_addr, out_addr, width, a_stride=None):
    """out = a * b, elementwise, over one (compiler.tile_dim, width) FP32
    tile. a_stride lets `a` be a column slice of a wider matrix (the same
    mechanism attention's Q/out already use) -- needed for RoPE applied
    directly to one head's slice of a wide (M, n_heads*head_dim) Q/K
    projection, rather than requiring the caller to de-interleave first.
    """
    M = compiler.tile_dim

    @T.prim_func(s_tir=True)
    def mul(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (M, width), "float32",
                           strides=([a_stride, 1] if a_stride is not None else None))
        B = T.match_buffer(b, (M, width), "float32")
        C = T.match_buffer(c, (M, width), "float32")
        for i, j in T.grid(M, width):
            with T.sblock("mul"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] * B[vi, vj]

    codegen.emit(compiler,
                 schedule.schedule_binary(mul, "mul", intrin.register_mul_fp32()),
                 {"A": a_addr, "B": b_addr, "C": out_addr},
                 dtype=DType.FP32, kind="elemwise")


def _emit_rope_fp32(compiler, head_dim, pos, in_addr, out_addr, theta=1e6, in_stride=None):
    """Apply RoPE to a (16, head_dim) tile at absolute position `pos`:
    out = in*cos + rotate_half(in)*sin, the "half-split" convention Qwen/
    Llama use -- q' = q*cos + rotate_half(q)*sin, rotate_half(x1,x2) =
    (-x2,x1) -- matching `NPUCompiler.rope_tables`'s own docstring and
    `compile_rope`'s algorithm exactly, reused here for the cos/sin
    construction (pure math, no scale semantics -- the same standing as
    reusing `_padded`/`_align` elsewhere in this module).

    `pos` is a Python int, not a runtime value: this whole codebase already
    recompiles one instruction stream per decode step with no position
    register in the ISA (see codegen.py's own docstring: "each decode step
    is a fresh ~12M-instruction program"), so treating position as a
    compile-time constant here is consistent with the existing design, not
    a shortcut specific to this function.

    head_dim must equal compiler.tile_dim (16), unlike compile_rope, which
    requires head_dim to be a multiple of 2*tile_dim (32+) so that a
    column's rotate_half partner is a WHOLE OTHER TILE, reachable with a
    second LOAD_TILE at a shifted address. At head_dim=16 the partner is
    WITHIN the same one tile (columns 0-7 pair with 8-15), which needs an
    actual column permutation, not just a different load address. Done via
    the same "constant matrix through GEMM" trick as
    _emit_broadcast_row0_fp32/_emit_copy_slice_fp32: rotate_half(in) =
    in @ P, where P is a compile-time constant (16,16) matrix swapping the
    two 8-wide halves (P[k, (k+8) mod 16] = 1, else 0) -- confirmed
    directly against numpy that a PLAIN swap (no negation in P itself)
    combined with rope_tables' sign-folded sin table reproduces the
    standard RoPE rotation exactly (the negation rope_tables already bakes
    into the first half of its sin table is exactly what a plain swap
    needs to be correct, since a signed swap would double the negation).

    in_stride lets `in` be a column slice of a wider (M, n_heads*head_dim)
    Q/K projection buffer, so RoPE can be applied per-head directly on the
    natural projection layout, before any broadcast/transpose/de-interleave
    step -- see _emit_attention_block_fp32 for where this fits in the
    pipeline once RoPE is wired into it.
    """
    M = compiler.tile_dim
    if head_dim != M:
        raise NotImplementedError(
            f"_emit_rope_fp32 needs head_dim == compiler.tile_dim ({M}); "
            f"got {head_dim}"
        )
    if head_dim % 2 != 0:
        raise ValueError(f"_emit_rope_fp32 needs an even head_dim; got {head_dim}")

    cos_tab, sin_tab = NPUCompiler.rope_tables(pos + 1, head_dim, base=theta, dtype=DType.FP32)
    cos_bcast = np.tile(cos_tab[pos], (M, 1)).astype(np.float32)
    sin_bcast = np.tile(sin_tab[pos], (M, 1)).astype(np.float32)
    cos_addr = compiler.alloc_staging(M * head_dim * 4)
    sin_addr = compiler.alloc_staging(M * head_dim * 4)
    compiler._const_regions[cos_addr] = cos_bcast.tobytes()
    compiler._const_regions[sin_addr] = sin_bcast.tobytes()

    half = head_dim // 2
    perm = np.zeros((M, M), dtype=np.float32)
    for k in range(head_dim):
        perm[k, (k + half) % head_dim] = 1.0
    perm_addr = compiler.alloc_staging(M * M * 4)
    compiler._const_regions[perm_addr] = perm.tobytes()

    rot_stage = compiler.alloc_staging(M * head_dim * 4)
    term1 = compiler.alloc_staging(M * head_dim * 4)
    term2 = compiler.alloc_staging(M * head_dim * 4)

    _emit_matmul_fp32(compiler, M, head_dim, head_dim, in_addr, perm_addr, rot_stage,
                      a_stride=in_stride)
    _emit_mul_fp32(compiler, in_addr, cos_addr, term1, head_dim, a_stride=in_stride)
    _emit_mul_fp32(compiler, rot_stage, sin_addr, term2, head_dim)
    _emit_residual_add_fp32(compiler, term1, term2, out_addr, head_dim, out_stride=in_stride)


def _emit_rope_batched_fp32(compiler, head_dim, positions, in_addr, out_addr,
                            theta=1e6, in_stride=None):
    """_emit_rope_fp32, but for a genuine batch of 16 DIFFERENT real rows
    (real prefill / Fp32FullLayerCase) each at its OWN position, instead of
    one real token (replicated across the tile) at one shared position.

    `positions` is a length-compiler.tile_dim sequence of Python ints, one
    per tile row (e.g. list(range(16)) for a chunk starting at position 0).
    Row r's cos/sin come from `rope_tables`'s row `positions[r]`, not a
    single row broadcast to all 16 -- the only actual difference from
    _emit_rope_fp32: cos_bcast/sin_bcast are built by INDEXING one table
    row per tile row instead of tiling one row 16 times. The permutation
    matrix (rotate_half's intra-tile swap) and the GEMM/mul/add steps that
    apply it are otherwise IDENTICAL and needed no changes -- they're
    row-parallel, and don't care whether every row's cos/sin agree or not.
    """
    M = compiler.tile_dim
    if head_dim != M:
        raise NotImplementedError(
            f"_emit_rope_batched_fp32 needs head_dim == compiler.tile_dim "
            f"({M}); got {head_dim}"
        )
    if head_dim % 2 != 0:
        raise ValueError(f"_emit_rope_batched_fp32 needs an even head_dim; got {head_dim}")
    if len(positions) != M:
        raise ValueError(f"_emit_rope_batched_fp32 needs exactly {M} positions "
                         f"(one per tile row); got {len(positions)}")

    cos_tab, sin_tab = NPUCompiler.rope_tables(max(positions) + 1, head_dim,
                                               base=theta, dtype=DType.FP32)
    cos_bcast = np.stack([cos_tab[p] for p in positions]).astype(np.float32)
    sin_bcast = np.stack([sin_tab[p] for p in positions]).astype(np.float32)
    cos_addr = compiler.alloc_staging(M * head_dim * 4)
    sin_addr = compiler.alloc_staging(M * head_dim * 4)
    compiler._const_regions[cos_addr] = cos_bcast.tobytes()
    compiler._const_regions[sin_addr] = sin_bcast.tobytes()

    half = head_dim // 2
    perm = np.zeros((M, M), dtype=np.float32)
    for k in range(head_dim):
        perm[k, (k + half) % head_dim] = 1.0
    perm_addr = compiler.alloc_staging(M * M * 4)
    compiler._const_regions[perm_addr] = perm.tobytes()

    rot_stage = compiler.alloc_staging(M * head_dim * 4)
    term1 = compiler.alloc_staging(M * head_dim * 4)
    term2 = compiler.alloc_staging(M * head_dim * 4)

    _emit_matmul_fp32(compiler, M, head_dim, head_dim, in_addr, perm_addr, rot_stage,
                      a_stride=in_stride)
    _emit_mul_fp32(compiler, in_addr, cos_addr, term1, head_dim, a_stride=in_stride)
    _emit_mul_fp32(compiler, rot_stage, sin_addr, term2, head_dim)
    _emit_residual_add_fp32(compiler, term1, term2, out_addr, head_dim, out_stride=in_stride)


def _emit_residual_add_fp32(compiler, a_addr, b_addr, out_addr, width, out_stride=None):
    """out = a + b, over one (compiler.tile_dim, width) FP32 tile. The
    transformer residual connection -- no rescale, no widening: those exist
    in npu_compiler.py specifically to reconcile INT8 scales between the
    residual stream and a sublayer's output, which FP32 has none of.

    out_stride (row pitch in ELEMENTS): lets out_addr be a column SLICE of
    a wider buffer instead of its own tightly-packed (M, width) region --
    e.g. RoPE's in-place write back into one head's slice of a wide
    (M, n_heads*head_dim) Q/K projection. Missing this was a REAL bug,
    caught only when genuinely different data landed in every tile row:
    with no out_stride, STORE_TILE's pitch defaults to `width` (this
    buffer's own declared shape), so row 1's write landed `width` elements
    along instead of the wider buffer's actual row length -- silently
    spilling into the NEXT head's column range. Invisible in every
    row0-real-rest-zero test (rows 1-15 are zero either way, so a
    misplaced write just overwrites zero with zero), only surfaced by
    _emit_rope_batched_fp32's genuinely-different-16-rows case.
    """
    M = compiler.tile_dim

    @T.prim_func(s_tir=True)
    def add(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (M, width), "float32")
        B = T.match_buffer(b, (M, width), "float32")
        C = T.match_buffer(c, (M, width), "float32",
                           strides=([out_stride, 1] if out_stride is not None else None))
        for i, j in T.grid(M, width):
            with T.sblock("add"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj] + B[vi, vj]

    codegen.emit(compiler,
                 schedule.schedule_binary(add, "add", intrin.register_add_fp32()),
                 {"A": a_addr, "B": b_addr, "C": out_addr},
                 dtype=DType.FP32, kind="elemwise")


def _emit_attention_block_fp32(compiler, hidden, n_heads, n_kv_heads, head_dim,
                               x_addr, wq_addr, wk_addr, wv_addr, wo_addr,
                               out_addr, pos=None, rope_theta=1e6):
    """A full attention SUBLAYER -- Q/K/V projections, GQA attention, output
    projection -- unlike _emit_attention_core_fp32/_emit_gqa_attention_fp32,
    which start from already-projected Q/K/V per the granularity decision
    documented in emit_composite. This is the piece that actually needs
    those projections, for building a whole transformer layer.

    pos: if given (a Python int -- see _emit_rope_fp32's docstring on why
    position is compile-time here, consistent with this whole codebase's
    per-decode-step recompilation), applies RoPE to every Q and K head
    right after their projections, before broadcasting/transposing -- the
    natural point, since RoPE is purely elementwise+permute and doesn't
    care about the row0-real-vs-replicated convention either side of it
    (unlike RMSNorm/attention's REDUCE steps, which very much do). Default
    None skips RoPE entirely (e.g. for models/tests that don't use it).

    Constrained to head_dim == compiler.tile_dim (16): K needs transposing
    per head to match _emit_attention_core_fp32's matmul(q,k) semantics
    (see that function's docstring -- no transpose inside the core itself),
    and _emit_transpose_fp32 only handles one square tile -- transposing a
    head_dim > 16 slice needs the same tile-addressing generalization
    attention's OTHER buffers already got, not done for transpose yet.
    (_emit_rope_fp32 has the identical constraint, for the same reason.)

    x is (M, hidden) with M == compiler.tile_dim rows. Real multi-row
    batches inherit _emit_attention_core_fp32's own limit: genuinely
    DIFFERENT query rows need codegen.py's dim_m=1 generalization, not done.
    The degenerate case this supports today -- x has ONE real row,
    REPLICATED across all 16 tile rows, self-attending to its own
    (therefore also 16-identical-copies) K/V -- is mathematically exact,
    not an approximation: attending to 16 identical keys gives the same
    result as attending to the 1 distinct one. A genuine multi-token KV
    context (the actual point of a KV-cache) is a separate piece --
    see _emit_kv_cache_attention_fp32.
    """
    if head_dim != compiler.tile_dim:
        raise NotImplementedError(
            f"_emit_attention_block_fp32 needs head_dim == "
            f"compiler.tile_dim ({compiler.tile_dim}); got {head_dim}"
        )
    M = compiler.tile_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    w = DType.BYTES[DType.FP32]

    q_stage = compiler.alloc_staging(M * q_dim * w)
    k_proj_stage = compiler.alloc_staging(M * kv_dim * w)
    v_stage = compiler.alloc_staging(M * kv_dim * w)
    q_bcast = compiler.alloc_staging(M * q_dim * w)
    k_bcast = compiler.alloc_staging(M * kv_dim * w)
    v_bcast = compiler.alloc_staging(M * kv_dim * w)
    k_t_stage = compiler.alloc_staging(n_kv_heads * head_dim * M * w)
    v_blocks = compiler.alloc_staging(n_kv_heads * M * head_dim * w)
    ctx_stage = compiler.alloc_staging(M * q_dim * w)
    attn_wide = compiler.alloc_staging(M * hidden * w)

    _emit_matmul_fp32(compiler, M, q_dim, hidden, x_addr, wq_addr, q_stage)
    _emit_matmul_fp32(compiler, M, kv_dim, hidden, x_addr, wk_addr, k_proj_stage)
    _emit_matmul_fp32(compiler, M, kv_dim, hidden, x_addr, wv_addr, v_stage)

    if pos is not None:
        # In place: each head's RoPE reads and writes the same column slice,
        # which is safe here the same way RMSNorm's in-place row update is
        # (the whole tile's worth of that head is read before any of it is
        # written back, within one _emit_rope_fp32 call).
        for qh in range(n_heads):
            off = qh * head_dim * w
            _emit_rope_fp32(compiler, head_dim, pos, q_stage + off, q_stage + off,
                            theta=rope_theta, in_stride=q_dim)
        for kv_h in range(n_kv_heads):
            off = kv_h * head_dim * w
            _emit_rope_fp32(compiler, head_dim, pos, k_proj_stage + off, k_proj_stage + off,
                            theta=rope_theta, in_stride=kv_dim)

    # x (and therefore Q/K/V projections, since 0@W=0) follow the LAYER-level
    # "row 0 real, rest zero" convention -- the one _emit_rmsnorm_row_fp32
    # actually needs (see _emit_broadcast_row0_fp32's docstring for why a
    # genuinely replicated input broke it). _emit_attention_core_fp32 needs
    # the OPPOSITE: all 16 rows identical, so its own REDUCE-based softmax
    # isn't contaminated by zero padding. Bridge here, at this block's
    # boundary, rather than changing either convention.
    _emit_broadcast_row0_fp32(compiler, q_stage, q_bcast, q_dim)
    _emit_broadcast_row0_fp32(compiler, k_proj_stage, k_bcast, kv_dim)
    _emit_broadcast_row0_fp32(compiler, v_stage, v_bcast, kv_dim)

    # K-projection's natural output is (M, kv_dim) -- rows are keys, columns
    # are per-head values, same layout as Q. _emit_attention_core_fp32 needs
    # each head's K pre-transposed to (head_dim, seq_k) (see its docstring:
    # no transpose inside the core). Transpose each head's column slice
    # into a contiguous (head_dim, M) block, matching the layout
    # _emit_gqa_attention_fp32 already expects from a caller. Operates on
    # the BROADCAST k_bcast (all 16 rows already identical), so the
    # transposed result has all 16 COLUMNS identical -- exactly the "16
    # key positions, all the same distinct key" this degenerate case needs.
    for kv_h in range(n_kv_heads):
        _emit_transpose_fp32(
            compiler,
            in_addr=k_bcast + kv_h * head_dim * w,
            out_addr=k_t_stage + kv_h * head_dim * M * w,
            in_stride=kv_dim,
        )

    # V needs no transpose, but _emit_gqa_attention_fp32 still expects it as
    # n_kv_heads separate CONTIGUOUS (seq_k, head_dim) blocks, not a column
    # slice of the wide (M, kv_dim) v_bcast -- found by getting this wrong
    # first: passing v_bcast straight through gave every head an IDENTICAL,
    # wrong context, because GQA's per-head byte offset (kv_h * seq_k *
    # head_dim * w) was landing mid-row of the wide buffer instead of at an
    # actual head boundary. De-interleave each head's column slice into its
    # own contiguous block first, the same way K's transpose does as a
    # side effect of its own reshaping.
    for kv_h in range(n_kv_heads):
        _emit_copy_slice_fp32(
            compiler,
            in_addr=v_bcast + kv_h * head_dim * w,
            out_addr=v_blocks + kv_h * M * head_dim * w,
            in_stride=kv_dim,
        )

    _emit_gqa_attention_fp32(
        compiler, n_heads, n_kv_heads, seq_k=M, head_dim=head_dim,
        q_addr=q_bcast, k_addr=k_t_stage, v_addr=v_blocks, out_addr=ctx_stage,
    )

    # ctx_stage comes out "all 16 rows identical" (the attention core's own
    # convention); the o_proj GEMM preserves that (replicated in, replicated
    # out). Mask back down to "row 0 real, rest zero" before handing this
    # block's result to the caller, so downstream RMSNorm calls (which need
    # that convention -- see _emit_broadcast_row0_fp32's docstring) aren't
    # fed 15 rows of non-zero garbage.
    _emit_matmul_fp32(compiler, M, hidden, q_dim, ctx_stage, wo_addr, attn_wide)
    _emit_mask_to_row0_fp32(compiler, attn_wide, out_addr, hidden)


def _emit_attention_block_batched_fp32(compiler, hidden, n_heads, n_kv_heads, head_dim,
                                       x_addr, wq_addr, wk_addr, wv_addr, wo_addr,
                                       out_addr, positions=None, rope_theta=1e6):
    """_emit_attention_block_fp32, but for a genuine batch of 16 DIFFERENT
    real rows in x (real prefill / Fp32AttentionCase) instead of one real
    row replicated across the tile.

    Much SIMPLER than the replicated version, not harder: there is no
    row0-real-vs-all-replicated convention to bridge at all here, because
    every row of x is already genuinely real -- _emit_broadcast_row0_fp32/
    _emit_mask_to_row0_fp32 exist ONLY to reconcile two conventions that
    both come from padding/replication tricks around a single real row;
    neither applies when all 16 rows are independently meaningful. Q/K/V
    projections, K's per-head transpose, and V's per-head de-interleave
    are all ROW-PARALLEL operations (GEMM and TRANSPOSE alike -- see
    _emit_attention_core_batched_fp32's docstring on why GEMM in particular
    needed no changes) and work identically whether fed replicated or
    genuinely distinct rows, so those steps are UNCHANGED from the
    replicated version, just fed x_addr directly with no broadcast step
    first. Only the attention CORE swaps to _emit_gqa_attention_batched_fp32
    for its real per-row softmax.

    positions: optional per-row RoPE. Unlike _emit_attention_block_fp32's
    single `pos` (one token, replicated -- every row shares one position by
    construction), a genuine batch of DIFFERENT rows means DIFFERENT
    positions per row (e.g. a real prefill chunk, token i at position i) --
    a single shared angle would be wrong here. Pass a length-16 sequence of
    per-row positions (e.g. list(range(16)) for a chunk starting at
    position 0), or None (default) to skip RoPE. See
    _emit_rope_batched_fp32 for the actual per-row cos/sin mechanics.

    Same head_dim == compiler.tile_dim constraint as the replicated
    version, for the same reason (K's per-head transpose only handles one
    square tile).
    """
    if head_dim != compiler.tile_dim:
        raise NotImplementedError(
            f"_emit_attention_block_batched_fp32 needs head_dim == "
            f"compiler.tile_dim ({compiler.tile_dim}); got {head_dim}"
        )
    M = compiler.tile_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    w = DType.BYTES[DType.FP32]

    q_stage = compiler.alloc_staging(M * q_dim * w)
    k_proj_stage = compiler.alloc_staging(M * kv_dim * w)
    v_stage = compiler.alloc_staging(M * kv_dim * w)
    k_t_stage = compiler.alloc_staging(n_kv_heads * head_dim * M * w)
    v_blocks = compiler.alloc_staging(n_kv_heads * M * head_dim * w)
    ctx_stage = compiler.alloc_staging(M * q_dim * w)

    _emit_matmul_fp32(compiler, M, q_dim, hidden, x_addr, wq_addr, q_stage)
    _emit_matmul_fp32(compiler, M, kv_dim, hidden, x_addr, wk_addr, k_proj_stage)
    _emit_matmul_fp32(compiler, M, kv_dim, hidden, x_addr, wv_addr, v_stage)

    if positions is not None:
        for qh in range(n_heads):
            off = qh * head_dim * w
            _emit_rope_batched_fp32(compiler, head_dim, positions,
                                    q_stage + off, q_stage + off,
                                    theta=rope_theta, in_stride=q_dim)
        for kv_h in range(n_kv_heads):
            off = kv_h * head_dim * w
            _emit_rope_batched_fp32(compiler, head_dim, positions,
                                    k_proj_stage + off, k_proj_stage + off,
                                    theta=rope_theta, in_stride=kv_dim)

    for kv_h in range(n_kv_heads):
        _emit_transpose_fp32(
            compiler,
            in_addr=k_proj_stage + kv_h * head_dim * w,
            out_addr=k_t_stage + kv_h * head_dim * M * w,
            in_stride=kv_dim,
        )
    for kv_h in range(n_kv_heads):
        _emit_copy_slice_fp32(
            compiler,
            in_addr=v_stage + kv_h * head_dim * w,
            out_addr=v_blocks + kv_h * M * head_dim * w,
            in_stride=kv_dim,
        )

    _emit_gqa_attention_batched_fp32(
        compiler, n_heads, n_kv_heads, seq_k=M, head_dim=head_dim,
        q_addr=q_stage, k_addr=k_t_stage, v_addr=v_blocks, out_addr=ctx_stage,
    )

    _emit_matmul_fp32(compiler, M, hidden, q_dim, ctx_stage, wo_addr, out_addr)


def _emit_transformer_layer_fp32(compiler, hidden, inter, n_heads, n_kv_heads,
                                 head_dim, x_addr, gamma1_addr, gamma2_addr,
                                 wq_addr, wk_addr, wv_addr, wo_addr,
                                 wgate_addr, wup_addr, wdown_addr, out_addr,
                                 pos=None, rope_theta=1e6):
    """One pre-norm transformer layer, entirely via TVM schedule+codegen:

        norm1 = RMSNorm(x, gamma1)
        attn  = AttentionBlock(norm1, Wq, Wk, Wv, Wo)   [+ RoPE if pos given]
        r1    = x + attn
        norm2 = RMSNorm(r1, gamma2)
        mlp   = SwiGLU(norm2, Wgate, Wup, Wdown)
        out   = r1 + mlp

    pos: forwarded to _emit_attention_block_fp32 -- the token's absolute
    position in the sequence (same value for every layer; RoPE only cares
    which token this is, not which layer is processing it). None (default)
    means no RoPE, preserving every existing caller's behavior. No
    1/sqrt(head_dim) scaling (consistent with _emit_attention_core_fp32
    having no scale step -- see its docstring). Inherits
    _emit_attention_block_fp32's constraints: head_dim == compiler.tile_dim,
    and x is one real row replicated across the 16-row tile (self-attention
    over its own replicated K/V, not a genuine multi-token context).

    This intentionally does NOT reproduce a known bug in the hand-written
    pipeline: e2e_test.py's Fp32FullLayerCase documents that
    compile_rmsnorm_row shares ONE scale across all 16 tile rows even when
    they hold real distinct tokens ("exact for decode... a real
    approximation for prefill"). This dispatch's RMSNorm
    (_emit_rmsnorm_row_fp32, via the same per-row loop emit_composite's
    "rmsnorm" kind uses) computes each row's scale independently -- more
    correct, and here the rows are identical anyway (by construction), so
    it makes no numeric difference, but this is NOT a bit-for-bit port of
    that reference and isn't trying to be.
    """
    M = compiler.tile_dim
    w = DType.BYTES[DType.FP32]

    norm1 = compiler.alloc_staging(M * hidden * w)
    attn_out = compiler.alloc_staging(M * hidden * w)
    resid1 = compiler.alloc_staging(M * hidden * w)
    norm2 = compiler.alloc_staging(M * hidden * w)
    mlp_out = compiler.alloc_staging(M * hidden * w)

    # gamma is supplied by the caller as a genuine (M, hidden) broadcast
    # buffer already (same contract _emit_rmsnorm_row_fp32/emit_composite's
    # "rmsnorm" kind documents) -- gamma1_addr/gamma2_addr ARE that buffer,
    # not a raw (hidden,) vector; no rebroadcast needed here.
    _emit_rmsnorm_row_fp32(compiler, hidden, x_addr=x_addr, gamma_addr=gamma1_addr,
                           out_addr=norm1,
                           scratch_addr=compiler.alloc_staging(4),
                           stage_addr=compiler.alloc_staging(M * _padded(hidden) * 4))

    _emit_attention_block_fp32(compiler, hidden, n_heads, n_kv_heads, head_dim,
                               x_addr=norm1, wq_addr=wq_addr, wk_addr=wk_addr,
                               wv_addr=wv_addr, wo_addr=wo_addr, out_addr=attn_out,
                               pos=pos, rope_theta=rope_theta)

    _emit_residual_add_fp32(compiler, x_addr, attn_out, resid1, hidden)

    _emit_rmsnorm_row_fp32(compiler, hidden, x_addr=resid1, gamma_addr=gamma2_addr,
                           out_addr=norm2,
                           scratch_addr=compiler.alloc_staging(4),
                           stage_addr=compiler.alloc_staging(M * _padded(hidden) * 4))

    _emit_swiglu_ffn_fp32(compiler, hidden, inter, x_addr=norm2,
                          w_gate_addr=wgate_addr, w_up_addr=wup_addr,
                          w_down_addr=wdown_addr, out_addr=mlp_out)

    _emit_residual_add_fp32(compiler, resid1, mlp_out, out_addr, hidden)


def _emit_model_fp32(compiler, n_layers, hidden, inter, n_heads, n_kv_heads,
                     head_dim, x_addr, layer_weights, scratch_a, scratch_b,
                     pos=None, rope_theta=1e6):
    """Chain n_layers transformer layers, PING-PONGING the activation buffer
    between scratch_a and scratch_b instead of allocating one buffer per
    layer -- layer 0 reads x_addr and writes scratch_b, layer 1 reads
    scratch_b and writes scratch_a, layer 2 reads scratch_a and writes
    scratch_b, and so on. This is what TODO 1.2's "ping-pong activation
    DRAM buffers across transformer layers" and 1.4's "whole-program
    execution" needed: something that actually chains more than one layer
    at all, which nothing on the TVM path did before this.

    layer_weights[i] is a dict with the 9 addresses
    _emit_transformer_layer_fp32 takes (gamma1/gamma2/wq/wk/wv/wo/
    wgate/wup/wdown), one entry per layer.

    pos: the CURRENT DECODE STEP's absolute token position, forwarded
    unchanged to every layer -- RoPE depends on which token this is, not
    which layer is processing it, so unlike x_addr/scratch buffers this is
    NOT threaded/advanced per layer. None (default) disables RoPE.

    Returns the address holding the final layer's output (either scratch_a
    or scratch_b, whichever was written last).
    """
    cur_in = x_addr
    bufs = [scratch_b, scratch_a]
    for i in range(n_layers):
        cur_out = bufs[i % 2]
        lw = layer_weights[i]
        _emit_transformer_layer_fp32(
            compiler, hidden, inter, n_heads, n_kv_heads, head_dim,
            x_addr=cur_in, gamma1_addr=lw["gamma1"], gamma2_addr=lw["gamma2"],
            wq_addr=lw["wq"], wk_addr=lw["wk"], wv_addr=lw["wv"], wo_addr=lw["wo"],
            wgate_addr=lw["wgate"], wup_addr=lw["wup"], wdown_addr=lw["wdown"],
            out_addr=cur_out, pos=pos, rope_theta=rope_theta,
        )
        cur_in = cur_out
    return cur_in


def _emit_transformer_layer_batched_fp32(compiler, hidden, inter, n_heads, n_kv_heads,
                                         head_dim, x_addr, gamma1_addr, gamma2_addr,
                                         wq_addr, wk_addr, wv_addr, wo_addr,
                                         wgate_addr, wup_addr, wdown_addr, out_addr,
                                         positions=None, rope_theta=1e6):
    """_emit_transformer_layer_fp32, but for a genuine batch of 16
    DIFFERENT real rows (real prefill / Fp32FullLayerCase) instead of one
    real row replicated across the tile.

    RMSNorm and SwiGLU are REUSED VERBATIM, not reimplemented: RMSNorm's
    REDUCE_SUMSQ already collapses the WHOLE (16, hidden) tile to one
    scalar and RMS_SCALE already applies that one scale to all 16 rows,
    regardless of whether those rows are replicated or genuinely
    different -- which happens to be EXACTLY what Fp32FullLayerCase's own
    reference expects (its rms() helper sums squares over the whole
    (TILE, hidden) block too, not per row -- "compile_rmsnorm_row shares
    ONE scale across all 16 tile rows... a real approximation for
    prefill", reproduced here by construction, not chosen deliberately).
    Verified directly against that exact reference with 16 genuinely
    different rows before relying on it. SwiGLU has no cross-row
    reduction at all (GEMM + per-element SiLU/mul), so it was already
    correct for distinct rows with zero changes.

    Only the attention sublayer differs: _emit_attention_block_batched_fp32
    instead of _emit_attention_block_fp32, and `positions` (one absolute
    position per tile row, e.g. list(range(16)) for a chunk starting at
    position 0) instead of a single shared `pos` -- see
    _emit_rope_batched_fp32 for why a single shared angle would be wrong
    for genuinely different rows.
    """
    M = compiler.tile_dim
    w = DType.BYTES[DType.FP32]

    norm1 = compiler.alloc_staging(M * hidden * w)
    attn_out = compiler.alloc_staging(M * hidden * w)
    resid1 = compiler.alloc_staging(M * hidden * w)
    norm2 = compiler.alloc_staging(M * hidden * w)
    mlp_out = compiler.alloc_staging(M * hidden * w)

    _emit_rmsnorm_row_fp32(compiler, hidden, x_addr=x_addr, gamma_addr=gamma1_addr,
                           out_addr=norm1,
                           scratch_addr=compiler.alloc_staging(4),
                           stage_addr=compiler.alloc_staging(M * _padded(hidden) * 4))

    _emit_attention_block_batched_fp32(compiler, hidden, n_heads, n_kv_heads, head_dim,
                                       x_addr=norm1, wq_addr=wq_addr, wk_addr=wk_addr,
                                       wv_addr=wv_addr, wo_addr=wo_addr, out_addr=attn_out,
                                       positions=positions, rope_theta=rope_theta)

    _emit_residual_add_fp32(compiler, x_addr, attn_out, resid1, hidden)

    _emit_rmsnorm_row_fp32(compiler, hidden, x_addr=resid1, gamma_addr=gamma2_addr,
                           out_addr=norm2,
                           scratch_addr=compiler.alloc_staging(4),
                           stage_addr=compiler.alloc_staging(M * _padded(hidden) * 4))

    _emit_swiglu_ffn_fp32(compiler, hidden, inter, x_addr=norm2,
                          w_gate_addr=wgate_addr, w_up_addr=wup_addr,
                          w_down_addr=wdown_addr, out_addr=mlp_out)

    _emit_residual_add_fp32(compiler, resid1, mlp_out, out_addr, hidden)


def _emit_model_batched_fp32(compiler, n_layers, hidden, inter, n_heads, n_kv_heads,
                             head_dim, x_addr, layer_weights, scratch_a, scratch_b,
                             positions=None, rope_theta=1e6):
    """_emit_model_fp32, chaining _emit_transformer_layer_batched_fp32
    instead -- same ping-pong buffering, same layer_weights contract;
    `positions` forwarded unchanged to every layer, same reasoning as
    _emit_model_fp32's `pos` (RoPE depends on which token each row is, not
    which layer is processing it)."""
    cur_in = x_addr
    bufs = [scratch_b, scratch_a]
    for i in range(n_layers):
        cur_out = bufs[i % 2]
        lw = layer_weights[i]
        _emit_transformer_layer_batched_fp32(
            compiler, hidden, inter, n_heads, n_kv_heads, head_dim,
            x_addr=cur_in, gamma1_addr=lw["gamma1"], gamma2_addr=lw["gamma2"],
            wq_addr=lw["wq"], wk_addr=lw["wk"], wv_addr=lw["wv"], wo_addr=lw["wo"],
            wgate_addr=lw["wgate"], wup_addr=lw["wup"], wdown_addr=lw["wdown"],
            out_addr=cur_out, positions=positions, rope_theta=rope_theta,
        )
        cur_in = cur_out
    return cur_in


def build_model_dram_image(compiler, hidden, n_layers, x_row, layer_weights_np):
    """
    Lay out a whole model's weights and initial activation into a single
    DRAM image, ready for _emit_model_fp32 -- the "given a model's actual
    weights, produce its DRAM image" step TODO 1.4 asks for. Generalizes
    the ad-hoc `place()` helper every test before this one built by hand
    (see rtv_tvm/test_transformer_layer.py) into a real, reusable function,
    the TVM-path counterpart to what ingest_model/export_dram_image do on
    the npu_compiler.py side.

    x_row: (hidden,) numpy array, the model's single real input token --
    still one real token (see _emit_attention_block_fp32's docstring on
    why: genuinely batched multi-row queries need codegen.py's dim_m=1
    generalization, not done). layer_weights_np: a list of n_layers dicts,
    each holding numpy arrays for gamma1, gamma2 (both (hidden,)), wq, wk,
    wv, wo, wgate, wup, wdown (matching _emit_transformer_layer_fp32's own
    shapes) -- whatever a real checkpoint loader would hand back per layer.

    Handles both layer-level conventions itself, so the caller doesn't
    have to: x is placed as "row 0 real, rest zero" (_emit_rmsnorm_row_
    fp32's own requirement -- see its docstring on why a genuinely
    replicated x breaks its REDUCE step), and each layer's gamma is
    broadcast to a genuine (16, hidden) buffer (also required).

    Returns (dram, x_addr, layer_weight_addrs, scratch_a, scratch_b) --
    layer_weight_addrs is exactly the list _emit_model_fp32's own
    `layer_weights` parameter expects. `dram` is sized for the weights
    placed here only; call finalize_dram_image AFTER emitting (once
    compiler.dram_bytes_needed reflects the staging emission actually
    used) to grow it and bake in every compile-time constant emission
    injected along the way. Leaves compiler._dram_cursor set so that
    later staging (from _emit_model_fp32 itself) starts after everything
    placed here, not on top of it.
    """
    M = compiler.tile_dim
    cursor = [0x1000]   # headroom so address 0 is never a valid buffer

    placed = []

    def place(arr):
        addr = cursor[0]
        cursor[0] += arr.nbytes + 0x1000   # gap between buffers, same margin every test here already used
        placed.append((addr, arr))
        return addr

    x = np.zeros((M, hidden), dtype=np.float32)
    x[0, :] = x_row
    x_addr = place(x)

    layer_weight_addrs = []
    for lw in layer_weights_np:
        entry = {}
        for name in ("gamma1", "gamma2"):
            bcast = np.tile(np.asarray(lw[name], dtype=np.float32), (M, 1))
            entry[name] = place(bcast)
        for name in ("wq", "wk", "wv", "wo", "wgate", "wup", "wdown"):
            entry[name] = place(np.asarray(lw[name], dtype=np.float32))
        layer_weight_addrs.append(entry)

    scratch_a = place(np.zeros((M, hidden), dtype=np.float32))
    scratch_b = place(np.zeros((M, hidden), dtype=np.float32))

    compiler._dram_cursor = cursor[0] + 0x10000

    dram = bytearray(cursor[0] + 0x20000)
    for addr, arr in placed:
        dram[addr:addr + arr.nbytes] = arr.tobytes()

    return dram, x_addr, layer_weight_addrs, scratch_a, scratch_b


def finalize_dram_image(compiler, dram, min_addr=0, min_len=0):
    """
    Grow `dram` (from build_model_dram_image) to fit whatever staging
    _emit_model_fp32's emission actually bumped compiler._dram_cursor past,
    and bake in every compile-time constant (row0 masks, broadcast/identity
    matrices, ...) injected along the way via compiler._const_regions --
    the same thing every test built by hand before this (`for addr, data
    in comp._const_regions.items(): dram[addr:addr+len(data)] = data`).

    min_addr/min_len: an extra region the caller needs room for (e.g. the
    final output address and its size) that might extend past what
    dram_bytes_needed alone would size for -- pass 0/0 if not needed.
    """
    needed = max(compiler.dram_bytes_needed, min_addr + min_len, len(dram))
    if needed > len(dram):
        dram = dram + bytearray(needed - len(dram) + 0x10000)
    for addr, data in compiler._const_regions.items():
        dram[addr:addr + len(data)] = data
    return dram


def emit_composite(compiler, info, addrs):
    """
    Emit one offloaded composite function directly via TVM schedule+codegen
    against the FP32 TensorIntrins in rtv_tvm.intrin -- NOT via
    npu_compiler.py's compile_* methods. See this module's docstring on why
    that split changed once INT8 quantization went out of scope.

    `addrs` maps every pattern-annotation name in info["param_names"] to a
    DRAM address, plus "out" for the composite's result. For "rmsnorm",
    addrs["gamma"] must point at a genuine (16, hidden) buffer with gamma
    replicated across all 16 rows -- see _emit_rmsnorm_row_fp32's docstring.
    These are supplied by the caller (whoever owns the DRAM image) -- BYOC
    partitioning tells you WHAT was offloaded, never WHERE its tensors live.

    Coverage, and why it stops where it does:

    * "rmsnorm" -- dispatched one real row at a time (see
      _emit_rmsnorm_row_fp32's docstring for why: REDUCE_SUMSQ collapses a
      whole SRAM tile to one scalar regardless of dtype, so there is no way
      to get 16 independent per-row scales out of one call).

    * "swiglu_ffn" -- one call handles a real M-row batch, up to one M-tile
      (M <= compiler.tile_dim). Larger M needs an outer tiling loop not
      implemented yet.

    * "attention"/"attention_masked" -- the GRANULARITY decision is made:
      dispatched at the pattern's own level (already-projected Q/K/V in,
      context out, ONE head per composite match), NOT widened to also
      cover the Q/K/V/O projections or multiple heads at once. DFPattern is
      a poor fit for expressing per-head loops, RoPE or KV-cache state as a
      static graph match, so those stay outside this composite -- the
      projections dispatch separately as ordinary "matmul" composites (see
      below). Multi-head/GQA IS implemented (_emit_gqa_attention_fp32,
      including seq_k/head_dim spanning multiple tiles), but as a SEPARATE
      directly-callable entry point, not reachable through pattern
      inference -- see that function's docstring for why. RoPE and
      KV-cache are still not implemented at all (no position information
      anywhere in this path yet).

    * "matmul" -- a standalone matmul composite, via _emit_matmul_fp32
      directly (same primitive attention's own Q@K and P@V steps use).
      NOTE this is a SEPARATE path from offloaded_matmuls()/emit() above,
      which still calls NPUCompiler.compile_linear -- that one hasn't been
      migrated off npu_compiler.py yet. Prefer THIS path (via
      offloaded_composites()/emit_composite()) for new callers; the other
      one is kept only because something might still depend on its exact
      calling convention (addrs keyed differently: "x"/"w" here too, but
      offloaded_matmuls()'s shape tuples are positional).

    * Everything else -- "linear_bias", "softmax", "silu", "add", "mul" --
      has no dispatch written yet. (Unlike before the FP32-only pivot, this
      is no longer blocked on missing scale data -- FP32 needs none -- it's
      just not built yet.)

    Raises NotImplementedError for any kind not listed above, naming the
    kind, rather than silently emitting nothing.
    """
    kind = info["kind"]

    # Shapes are read off info["params"] POSITIONALLY, not by p["name"] ==
    # <canonical wildcard name>. Confirmed empirically that a composite
    # function's parameter names are inherited from whatever the ORIGINAL
    # graph called those tensors, not from patterns.py's own annotation
    # dict keys ({"x": x, "w_gate": w_gate, ...} names things for the
    # PATTERN's own readability, not for the matched function's signature).
    # Only the ORDER is guaranteed to follow each pattern's own dict
    # (Python dicts preserve insertion order, and FuseOpsByPattern threads
    # that through) -- so name-based lookup silently breaks on any real
    # graph that doesn't happen to name its tensors "x"/"w_gate"/etc, which
    # is not something to rely on. Position is exactly what patterns.py
    # itself defines and is stable regardless of the caller's naming.

    if kind == "rmsnorm":
        x_shape = info["params"][0]["shape"]     # order: x, gamma
        hidden = x_shape[-1]
        rows = 1
        for d in x_shape[:-1]:
            rows *= d
        row_bytes = hidden * DType.BYTES[DType.FP32]
        scratch_addr = compiler.alloc_staging(4)
        stage_addr = compiler.alloc_staging(compiler.tile_dim * _padded(hidden) * 4)
        for row in range(rows):
            _emit_rmsnorm_row_fp32(
                compiler, hidden,
                x_addr=addrs["x"] + row * row_bytes,
                gamma_addr=addrs.get("gamma", 0),
                out_addr=addrs["out"] + row * row_bytes,
                scratch_addr=scratch_addr, stage_addr=stage_addr,
            )
        return

    if kind == "swiglu_ffn":
        # order: x, w_gate, w_up, w_down
        x_shape = info["params"][0]["shape"]
        w_gate_shape = info["params"][1]["shape"]
        hidden, inter = x_shape[-1], w_gate_shape[-1]
        rows = 1
        for d in x_shape[:-1]:
            rows *= d
        if rows > compiler.tile_dim:
            raise NotImplementedError(
                f"emit_composite: swiglu_ffn with {rows} rows exceeds the "
                f"single-M-tile limit ({compiler.tile_dim}); needs an outer "
                "M-tiling loop not yet implemented."
            )
        _emit_swiglu_ffn_fp32(
            compiler, hidden, inter,
            x_addr=addrs["x"], w_gate_addr=addrs["w_gate"],
            w_up_addr=addrs["w_up"], w_down_addr=addrs["w_down"],
            out_addr=addrs["out"],
        )
        return

    if kind in ("attention", "attention_masked"):
        # order: q, k, v (attention) / q, k, mask, v (attention_masked) --
        # q and k are params[0]/[1] in both, which is all this needs.
        k_shape = info["params"][1]["shape"]
        seq_k, head_dim = k_shape[-2], k_shape[-1]
        _emit_attention_core_fp32(
            compiler, seq_k, head_dim,
            q_addr=addrs["q"], k_addr=addrs["k"], v_addr=addrs["v"],
            out_addr=addrs["out"],
            mask_addr=addrs.get("mask") if kind == "attention_masked" else None,
        )
        return

    if kind == "matmul":
        # A plain bare matmul composite -- distinct from offloaded_matmuls()/
        # emit() above, which still calls NPUCompiler.compile_linear. This
        # path exists so a graph that offloads a standalone matmul (not part
        # of a bigger composite) gets the same pure-TVM treatment as
        # rmsnorm/swiglu_ffn/attention above, rather than being the one
        # composite kind still routed through npu_compiler.py.
        x_shape = info["params"][0]["shape"]      # order: x, w
        w_shape = info["params"][1]["shape"]
        m, k = x_shape[-2], x_shape[-1]
        k2, n = w_shape[-2], w_shape[-1]
        assert k == k2, f"matmul shape mismatch: x has K={k}, w has K={k2}"
        _emit_matmul_fp32(compiler, m, n, k, addrs["x"], addrs["w"], addrs["out"])
        return

    raise NotImplementedError(
        f"emit_composite: no dispatch for composite kind {kind!r} yet -- "
        "see this function's docstring for why (linear_bias and the rest "
        "need calibrated quantization scale data this call has no honest "
        "default for; attention IS dispatched, see above)."
    )
