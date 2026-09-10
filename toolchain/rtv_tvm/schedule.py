"""
schedule.py — reshaping a general matmul into RTV-Accel's instruction shape.

These are the decisions TVM does NOT make for you. It supplies `split`,
`cache_read` and `tensorize` as primitives; choosing the 16x16 tile, deciding
that operands stage through SRAM, and placing the staging relative to the K
loop are all hardware knowledge that has to be written down.

The mapping to the ISA is direct, which is the point:

    cache_read   ->  LOAD_TILE      (DRAM -> SRAM)
    tensorize    ->  COMPUTE_GEMM
    cache_write  ->  STORE_TILE     (SRAM -> DRAM)

What TVM computes, and what the hand-written compiler did by hand in ~276
lines: every address, stride and tile bound. Element width lives in the
buffer type, so the byte-vs-element arithmetic that produced seven of this
project's bugs has no place to live.
"""

from tvm import s_tir

from . import intrin

TILE = intrin.TILE


def schedule_matmul(func, block_name="update", intrin_name=None):
    """Tile a matmul to the PE array and tensorize it. Returns the Schedule."""
    intrin_name = intrin_name or intrin.register()
    sch = s_tir.Schedule(func)
    blk = sch.get_sblock(block_name)
    i, j, k = sch.get_loops(blk)

    # Split each axis to the hardware shape and order so the three inner axes
    # are adjacent -- that adjacency is what tensorize matches against.
    _, ii = sch.split(i, factors=[None, TILE])
    jo, ji = sch.split(j, factors=[None, TILE])
    ko, ki = sch.split(k, factors=[None, TILE])
    io = sch.get_loops(blk)[0]
    sch.reorder(io, jo, ko, ii, ji, ki)

    # Stage operands through SRAM. Required, not optional: the intrinsic
    # declares scope="shared", and tensorize rejects a global-scope nest.
    a_s = sch.cache_read(blk, 0, "shared")
    b_s = sch.cache_read(blk, 1, "shared")
    c_s = sch.cache_write(blk, 0, "shared")
    sch.compute_at(a_s, ko)
    sch.compute_at(b_s, ko)
    sch.reverse_compute_at(c_s, jo)

    # Group the inner nest into one block, then hoist the accumulator init out
    # so what remains is a pure accumulate -- which is what `desc` describes.
    # Without decompose_reduction the read set differs (2 buffers vs 3) and
    # tensorize refuses the match.
    inner = sch.blockize(ii)
    sch.decompose_reduction(inner, ko)
    sch.tensorize(inner, intrin_name)
    return sch


def schedule_binary(func, block_name, intrin_name):
    """Shared shape for the two-operand elementwise ops (ADD, MUL)."""
    sch = s_tir.Schedule(func)
    blk = sch.get_sblock(block_name)
    i, j = sch.get_loops(blk)
    io, ii = sch.split(i, factors=[None, TILE])
    jo, ji = sch.split(j, factors=[None, TILE])
    sch.reorder(io, jo, ii, ji)
    a_s = sch.cache_read(blk, 0, "shared")
    b_s = sch.cache_read(blk, 1, "shared")
    c_s = sch.cache_write(blk, 0, "shared")
    sch.compute_at(a_s, jo)
    sch.compute_at(b_s, jo)
    sch.reverse_compute_at(c_s, jo)
    sch.tensorize(sch.blockize(ii), intrin_name)
    return sch


def schedule_mul(func, block_name="mul"):
    return schedule_binary(func, block_name, intrin.register_mul())


def schedule_sumsq(func, block_name="sumsq"):
    """RMSNorm's first pass: sum of squares over a whole row.

    The reduction spans several tiles, and the ISA expresses that with the
    accumulate flag rather than a wider instruction -- so the schedule splits
    the reduction axis and the codegen reads accumulate off the outer loop,
    exactly as it does for the GEMM's K axis.
    """
    name = intrin.register_sumsq()
    sch = s_tir.Schedule(func)
    blk = sch.get_sblock(block_name)
    i, j = sch.get_loops(blk)
    io, ii = sch.split(i, factors=[None, TILE])
    jo, ji = sch.split(j, factors=[None, TILE])
    sch.reorder(io, jo, ii, ji)
    a_s = sch.cache_read(blk, 0, "shared")
    # The scalar accumulator lives in SRAM too -- REDUCE writes it there, and
    # RMS_SCALE reads it from there. tensorize enforces this: without staging
    # it, the match fails with `scope(): global vs shared` on S, the same way
    # the GEMM's C did.
    s_s = sch.cache_write(blk, 0, "shared")
    sch.compute_at(a_s, jo)
    sch.reverse_compute_at(s_s, io)
    # As with the GEMM: T.init() leaves the block reading only A, while the
    # intrinsic describes an accumulate reading both A and S. Hoisting the
    # init out makes what remains a pure accumulate, which is what matches.
    inner = sch.blockize(ii)
    sch.decompose_reduction(inner, io)
    sch.tensorize(inner, name)
    return sch


def schedule_elemwise(func, block_name="add", intrin_name=None):
    """Tile an elementwise op to 16x16 and tensorize it.

    Simpler than the matmul: no reduction axis, so no blockize/decompose
    dance -- the two spatial axes split and the inner pair tensorizes
    directly.
    """
    intrin_name = intrin_name or intrin.register_add()
    sch = s_tir.Schedule(func)
    blk = sch.get_sblock(block_name)
    i, j = sch.get_loops(blk)
    io, ii = sch.split(i, factors=[None, TILE])
    jo, ji = sch.split(j, factors=[None, TILE])
    sch.reorder(io, jo, ii, ji)

    a_s = sch.cache_read(blk, 0, "shared")
    b_s = sch.cache_read(blk, 1, "shared")
    c_s = sch.cache_write(blk, 0, "shared")
    sch.compute_at(a_s, jo)
    sch.compute_at(b_s, jo)
    sch.reverse_compute_at(c_s, jo)

    inner = sch.blockize(ii)
    sch.tensorize(inner, intrin_name)
    return sch


def schedule_rms_scale(func, block_name="rms"):
    """RMSNorm's second pass: scale every tile of a row by one scalar.

    Purely spatial -- the reduction already happened -- so this is the
    elementwise shape with a scalar operand instead of a second tile.
    """
    name = intrin.register_rms_scale()
    sch = s_tir.Schedule(func)
    blk = sch.get_sblock(block_name)
    i, j = sch.get_loops(blk)
    io, ii = sch.split(i, factors=[None, TILE])
    jo, ji = sch.split(j, factors=[None, TILE])
    sch.reorder(io, jo, ii, ji)
    a_s = sch.cache_read(blk, 1, "shared")     # the tile; index 0 is the scalar
    s_s = sch.cache_read(blk, 0, "shared")     # the scalar accumulator
    c_s = sch.cache_write(blk, 0, "shared")
    sch.compute_at(a_s, jo)
    sch.compute_at(s_s, jo)
    sch.reverse_compute_at(c_s, jo)
    sch.tensorize(sch.blockize(ii), name)
    return sch


def schedule_reduce(func, block_name, intrin_name):
    """Shared shape for the tile->scalar reductions (sumsq, rowmax, rowsum).

    Identical to schedule_sumsq: the reduction spans several tiles and the
    ISA expresses that with the accumulate flag, so the reduction axis splits
    and the codegen reads accumulate off the outer loop.
    """
    sch = s_tir.Schedule(func)
    blk = sch.get_sblock(block_name)
    i, j = sch.get_loops(blk)
    io, ii = sch.split(i, factors=[None, TILE])
    jo, ji = sch.split(j, factors=[None, TILE])
    sch.reorder(io, jo, ii, ji)
    a_s = sch.cache_read(blk, 0, "shared")
    s_s = sch.cache_write(blk, 0, "shared")
    sch.compute_at(a_s, jo)
    sch.reverse_compute_at(s_s, io)
    inner = sch.blockize(ii)
    sch.decompose_reduction(inner, io)
    sch.tensorize(inner, intrin_name)
    return sch


def schedule_rowmax(func, block_name="rowmax"):
    """Softmax pass 1."""
    return schedule_reduce(func, block_name, intrin.register_rowmax())


def schedule_rowsum(func, block_name="rowsum"):
    """Softmax pass 2's accumulation half."""
    return schedule_reduce(func, block_name, intrin.register_rowsum())


def schedule_tile_scalar(func, block_name, intrin_name):
    """Shared shape for (tile, scalar) -> tile ops: rms_scale, exp_sub, div.

    Purely spatial -- any reduction already happened in an earlier pass -- so
    this is the elementwise shape with a scalar operand in place of the
    second tile. Operand indices follow the block's read order, which for a
    non-reduction is the order the buffers appear in the expression.
    """
    sch = s_tir.Schedule(func)
    blk = sch.get_sblock(block_name)
    i, j = sch.get_loops(blk)
    io, ii = sch.split(i, factors=[None, TILE])
    jo, ji = sch.split(j, factors=[None, TILE])
    sch.reorder(io, jo, ii, ji)
    a_s = sch.cache_read(blk, 1, "shared")     # the tile; index 0 is the scalar
    s_s = sch.cache_read(blk, 0, "shared")     # the scalar
    c_s = sch.cache_write(blk, 0, "shared")
    sch.compute_at(a_s, jo)
    sch.compute_at(s_s, jo)
    sch.reverse_compute_at(c_s, jo)
    sch.tensorize(sch.blockize(ii), intrin_name)
    return sch


def schedule_exp_sub(func, block_name="expsub"):
    """Softmax pass 2's exponential half."""
    return schedule_tile_scalar(func, block_name, intrin.register_exp_sub())


def schedule_scalar_div(func, block_name="sdiv"):
    """Softmax pass 3."""
    return schedule_tile_scalar(func, block_name, intrin.register_scalar_div())


def schedule_unary(func, block_name, intrin_name):
    """Shared shape for one-in one-out elementwise ops (SiLU, transpose).

    Simplest case in the file: no reduction and no second operand, so the two
    spatial axes split and the inner pair tensorizes directly.
    """
    sch = s_tir.Schedule(func)
    blk = sch.get_sblock(block_name)
    i, j = sch.get_loops(blk)
    io, ii = sch.split(i, factors=[None, TILE])
    jo, ji = sch.split(j, factors=[None, TILE])
    sch.reorder(io, jo, ii, ji)
    a_s = sch.cache_read(blk, 0, "shared")
    c_s = sch.cache_write(blk, 0, "shared")
    sch.compute_at(a_s, jo)
    sch.reverse_compute_at(c_s, jo)
    sch.tensorize(sch.blockize(ii), intrin_name)
    return sch


def schedule_silu(func, block_name="silu"):
    return schedule_unary(func, block_name, intrin.register_silu())


def schedule_transpose(func, block_name="transpose"):
    """TRANSPOSE works on square tiles, so the tiling is the identity here.

    Note this schedules a per-TILE transpose: C[i, j] = A[j, i] within each
    16x16 tile. Transposing a whole MATRIX additionally swaps which tile each
    lands in, which is an addressing change the caller expresses by how it
    declares the output buffer, not something this schedule does.
    """
    return schedule_unary(func, block_name, intrin.register_transpose())
