"""
codegen.py — scheduled TIR to RTV-Accel instructions.

This is the piece TVM cannot supply: no framework knows your encoding, your
SRAM slots, or your dependency tokens. But note what it does NOT do -- it
computes no addresses. Every offset and stride below is read off the TIR,
which TVM derived from the buffer types. That is the ~276 lines of
byte-vs-element arithmetic the hand-written compiler carried, and the source
of roughly seven of this project's bugs.

The codegen FULLY UNROLLS. That is not a simplification, it is forced: the
ISA has no control flow, so a loop in the TIR must become repeated
instructions. It is also why each decode step is a fresh ~12M-instruction
program -- a hardware property this changes nothing about.

Structure produced by rtv_tvm.schedule, which this walks:

    for i_0, j_0:                    output tiles
        update_o_init                zero the accumulator
        for k_0:
            A_shared / B_shared      -> LOAD_TILE
            rtv_compute_gemm         -> COMPUTE_GEMM (accumulate on k_0 != 0)
        C_shared                     -> STORE_TILE
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tvm.ir as tir_ir                                  # noqa: E402
import tvm.tirx as tir                                   # noqa: E402
from rtv_isa import (Opcode, ActivationMode, SRAMMap, DType,  # noqa: E402
                     encode, to_bytes)

# Backward-compatibility aliases
NPUOpcodes = Opcode
NPUActivations = ActivationMode

TILE = 16
# SRAM slots for the three staged tiles. Fixed here rather than allocated:
# the schedule stages exactly three buffers, and giving them stable homes
# keeps this codegen readable. A general version would map TVM's
# sblock_alloc_buffer decls onto an allocator.
# Softmax keeps two scalars live simultaneously (see _reduce), so the scratch
# region is addressed as two 4-byte slots rather than one.
SCALAR_MAX = SRAMMap.SCALAR_SCRATCH_BASE
SCALAR_SUM = SRAMMap.SCALAR_SCRATCH_BASE + 4
DEP = {"A": 1, "B": 2}


def _slot_map(kind: str, width: int) -> dict:
    """
    SRAM addresses for A/B/C, spaced for the ACTUAL element width in use.

    Was two fixed module-level dicts (SLOT, EW_SLOT) computed once at import
    time with the default width=1 (INT8) -- silently wrong for any wider
    dtype, and never caught because every one of the 28 existing tests uses
    the INT8 default. Confirmed the failure mode directly: a 16x16 FP32 tile
    is 1024 bytes, but input_slot(0)/input_slot(1) at width=1 are only 256
    bytes apart, so operand A's FP32 tile would run straight through where
    operand B's tile starts. GEMM's A/B never collided even at width=1
    (different zones -- input vs weight -- both happen to start at their
    zone's byte 0 regardless of spacing), which is exactly why this stayed
    hidden: the one kind that was never wrong made the bug in the other
    kind look untested rather than broken.

    Elementwise ops keep both operands in the input zone: ELEMWISE_ADD/MUL
    read operand B from an SRAM address carried in the stride field.
    """
    if kind == "gemm":
        return {"A": SRAMMap.input_slot(0, width),
                "B": SRAMMap.weight_slot(0, width),
                "C": SRAMMap.output_slot(0)}
    return {"A": SRAMMap.input_slot(0, width),
            "B": SRAMMap.input_slot(1, width),
            "C": SRAMMap.input_slot(0, width)}


def _loop_names(expr, out=None):
    """Names of every Var appearing in an index expression."""
    out = set() if out is None else out
    if isinstance(expr, tir.Var):
        out.add(expr.name)
    for f in ("a", "b"):
        if hasattr(expr, f):
            _loop_names(getattr(expr, f), out)
    return out


def _eval(expr, env):
    """Evaluate a TIR index expression against concrete loop values.

    Only the node types the schedule actually produces are handled; anything
    else raises rather than silently returning a wrong address, because a
    wrong address is exactly the failure mode this file exists to avoid.
    """
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    if isinstance(expr, tir.Var):
        return env[expr.name]
    if isinstance(expr, tir.Add):
        return _eval(expr.a, env) + _eval(expr.b, env)
    if isinstance(expr, tir.Mul):
        return _eval(expr.a, env) * _eval(expr.b, env)
    if isinstance(expr, tir.FloorDiv):
        return _eval(expr.a, env) // _eval(expr.b, env)
    if isinstance(expr, tir.FloorMod):
        return _eval(expr.a, env) % _eval(expr.b, env)
    raise NotImplementedError(f"codegen cannot evaluate {type(expr).__name__}")


class Codegen:
    """Walks scheduled TIR and appends to an NPUCompiler's instruction stream."""

    def __init__(self, compiler, dram, dtype=DType.INT8, kind="gemm"):
        self.c = compiler
        self.dram = dram          # buffer name -> DRAM base address
        self.dtype = dtype
        self.w = DType.BYTES[dtype]
        # `kind` selects the slot map and the output element width. A GEMM
        # writes an INT32 (or, at FP32, an FP32) accumulator; an elementwise
        # op writes at the operand width. Computed from self.w, not a fixed
        # module-level dict -- see _slot_map's docstring.
        self.kind = kind
        self.slot = _slot_map(kind, self.w)
        self.out_w = 4 if kind in ("gemm", "reduce") else DType.BYTES[dtype]
        # Keyed by activation, not a single counter: softmax runs a MAX
        # reduction and a SUM reduction over the same tiles, and each needs
        # its own "have I emitted yet" state or the second would wrongly
        # accumulate onto the first's leftovers.
        self._reduce_count = {}
        # RMS_SCALE's numerator: a compile-time constant from the scale plan,
        # not something TVM can know. Callers override it.
        self.SCALAR_MAX, self.SCALAR_SUM = SCALAR_MAX, SCALAR_SUM
        self.scale_num = 0
        # Q8.8 temperature for EXP_SUB; 0 keeps the ISA's fixed default.
        self.temperature = 0
        # DRAM address of the 256-entry LUT, for nonlinearity kernels.
        self.lut_dram = None
        self._lut_loaded = False
        # Buffer short-names ("A"/"B"/"C") for which _tile_copy should skip
        # the LOAD_TILE/STORE_TILE entirely -- the data is already sitting
        # in SRAM at self.slot[name] (typically pinned by the caller to a
        # fixed c_slot address another op just computed into, or is about
        # to consume from), so there is nothing to move. See
        # tvm_backend._emit_swiglu_ffn_resident_fp32 for the motivating
        # case: chaining a GEMM's C output directly into SiLU/ELEMWISE_MUL
        # without round-tripping DRAM in between. Both COMPUTE_GEMM's C
        # (hardware-fixed to a c_slot address) and EXEC_ALU's operands
        # (genuinely arbitrary SRAM addresses -- confirmed directly against
        # pe_array.cpp, no zone restriction in hardware) support this; only
        # LOAD_TILE/STORE_TILE's own DRAM traffic is what gets skipped.
        self.skip_tile_io = set()
        # Row count for LOAD_TILE/STORE_TILE/EXEC_ALU/REDUCE/SCALAR_ALU's
        # dim_m field. Every one of those was a literal TILE(16) constant
        # until genuinely batched (real, DIFFERENT per-row) softmax/RMSNorm
        # needed it to be 1: REDUCE/SCALAR_ALU are hardware ops that
        # collapse/broadcast across a WHOLE dim_m x dim_n SRAM region with
        # no per-row mode (confirmed against pe_array.cpp -- M, N are real
        # parameters, but there's exactly one output/one applied scalar
        # regardless of M), and TVM's schedule.py primitives always split
        # every axis into TILE-sized chunks (`sch.split(i, factors=[None,
        # TILE])`), which can't express an M=1 axis at all -- so a genuine
        # per-row op has to bypass schedule.py/tensorize entirely and drive
        # these primitives directly with dim_m=1. See
        # tvm_backend._emit_softmax_rows_fp32. COMPUTE_GEMM is NOT threaded
        # through this: it's inherently row-parallel (each output row only
        # depends on its own input row), so a real M=16 GEMM is already
        # correct for 16 genuinely different rows with no change needed --
        # only REDUCE/SCALAR_ALU/the tile-move opcodes collapse rows.
        self.dim_m = TILE
        # COMPUTE_GEMM's hardware C output slot (0-15, see rtv_isa.py's
        # c_slot documentation). 0 is the v1-compatible default, matching
        # _slot_map("gemm")'s own C entry (SRAMMap.output_slot(0)) -- keep
        # this and self.slot["C"] pointing at the SAME slot index/address,
        # or COMPUTE_GEMM will write to a different physical location than
        # whatever the STORE_TILE (or a resident consumer) reads from.
        self.c_slot = 0

    # -- helpers ---------------------------------------------------------
    def _emit_word(self, word: int):
        raw = to_bytes(word)
        if hasattr(self.c, "instruction_stream"):
            self.c.instruction_stream.extend(raw)
        elif hasattr(self.c, "_emit"):
            if hasattr(self.c, "instructions"):
                self.c.instructions.append(word)
        elif isinstance(self.c, bytearray):
            self.c.extend(raw)
        elif isinstance(self.c, list):
            self.c.append(word)

    def _wait(self, dep_id: int):
        word = encode(opcode=Opcode.WAIT_SYNC, dep_id=dep_id)
        self._emit_word(word)

    def _tile_copy(self, realize, env, slot_override=None):
        """A 16x16 staging block: emit one LOAD_TILE or STORE_TILE.

        The block's body copies element by element; the instruction moves the
        whole tile, so only the base index matters. Row pitch comes from the
        source buffer's own shape -- TVM's, not ours.

        slot_override: use this SRAM address instead of the operand's usual
        fixed slot. Only _gemm_k_loop uses this, to alternate the weight
        (B) operand between two slots for double buffering -- everything
        else keeps a fixed slot per operand, unchanged.
        """
        block = realize.block
        store = block.body
        while not isinstance(store, tir.BufferStore):
            store = store.body if hasattr(store, "body") else store.block.body
        dst, src = store.buffer, store.value.buffer

        base = dict(env)
        for it in block.iter_vars:
            if it.var.name.startswith("ax"):
                base[it.var.name] = 0
        for iv, val in zip(block.iter_vars, realize.iter_values):
            base[iv.var.name] = _eval(val, dict(base, **{
                n: 0 for n in _loop_names(val) if n.startswith("ax")}))
        idx_src = [_eval(i, base) for i in store.value.indices]
        idx_dst = [_eval(i, base) for i in store.indices]

        if len(idx_src) < 2 or len(idx_dst) < 2:
            return

        to_sram = str(dst.scope()) == "shared"
        outer = src if to_sram else dst
        key = (dst if to_sram else src).name.split("_")[0]
        if key in self.skip_tile_io:
            # Resident operand: self.slot[key] already holds the right data
            # (pinned by the caller), so there is no DRAM traffic to emit --
            # not even a dram_addr lookup, since a purely-resident buffer
            # need not have a real DRAM entry in self.dram at all.
            return
        # Row pitch defaults to the buffer's own width (a tightly packed
        # (M, N) buffer), but respects an EXPLICIT stride if the caller
        # declared one via match_buffer(..., strides=[...]) -- which is
        # exactly how a column slice of a WIDER matrix is expressed (e.g.
        # attention's per-head slice of a (M, n_heads*head_dim) Q buffer:
        # shape (M, head_dim), stride (q_dim, 1), so consecutive rows of
        # the slice are q_dim elements apart, not head_dim). A default
        # (contiguous) buffer's .strides is an empty tuple, not absent --
        # confirmed directly against this TVM build.
        pitch = (int(outer.strides[-2]) if outer.strides
                 else int(outer.shape[-1])) * self.w
        row, col = (idx_src if to_sram else idx_dst)
        addr = self.dram[outer.name] + row * pitch + col * self.w
        slot = slot_override if slot_override is not None else self.slot[key]

        if to_sram:
            word = encode(opcode=Opcode.LOAD_TILE, dep_id=DEP.get(key, 3),
                          dram_addr=addr, sram_addr=slot, dim_m=self.dim_m,
                          dim_n=TILE * self.w, stride=pitch,
                          dtype=int(self.dtype))
            self._emit_word(word)
        else:
            ew = self.out_w
            # Same fix as the LOAD_TILE branch above, and for the same
            # reason -- this is a SEPARATE pitch computation (STORE_TILE's
            # output element width can differ from the input's, via
            # self.out_w), so it silently ignored an explicit stride too.
            # Caught by attention's multi-head output: writing each head's
            # column slice into a shared (M, n_heads*head_dim) buffer used
            # the wrong (narrow) pitch, so row 1's write landed inside row
            # 0's own head_dim-wide span instead of q_dim further along --
            # every row past 0 silently corrupted by the next head's store.
            pitch = (int(outer.strides[-2]) if outer.strides
                     else int(outer.shape[-1])) * ew
            addr = self.dram[outer.name] + row * pitch + col * ew
            word = encode(opcode=Opcode.STORE_TILE, dram_addr=addr,
                          sram_addr=slot, dim_m=self.dim_m, dim_n=TILE * ew,
                          stride=pitch, dtype=int(self.dtype))
            self._emit_word(word)

    def _elemwise(self, activation):
        """EXEC_ALU binary op: A op= B in place, B addressed via stride."""
        for k in ("A", "B"):
            self._wait(DEP[k])
        word = encode(opcode=Opcode.EXEC_ALU, sram_addr=self.slot["A"],
                      stride=self.slot["B"], dim_m=self.dim_m, dim_n=TILE,
                      activation=int(activation), dtype=int(self.dtype))
        self._emit_word(word)

    def _reduce(self, activation, scalar_addr=None):
        """REDUCE: a tile collapses to one INT32 scalar."""
        self._wait(DEP["A"])
        accumulate = self._reduce_count.get(activation, 0) > 0
        dest_sram = (self.SCALAR_MAX if scalar_addr is None else scalar_addr) & 0xFFFF
        word = encode(opcode=Opcode.REDUCE, dep_id=0, dram_addr=0,
                      sram_addr=dest_sram, stride=self.slot["A"],
                      dim_m=self.dim_m, dim_n=TILE, activation=int(activation),
                      accumulate=accumulate, dtype=int(self.dtype))
        self._emit_word(word)
        self._reduce_count[activation] = self._reduce_count.get(activation, 0) + 1

    def _tile_scalar(self, activation, imm=0, scalar_addr=None):
        """SCALAR_ALU: a tile combined with one scalar held in SRAM."""
        self._wait(DEP["A"])
        dest_sram = (self.SCALAR_MAX if scalar_addr is None else scalar_addr) & 0xFFFF
        word = encode(opcode=Opcode.SCALAR_ALU, dep_id=0, dram_addr=imm,
                      sram_addr=dest_sram, stride=self.slot["A"],
                      dim_m=self.dim_m, dim_n=TILE, activation=int(activation),
                      dtype=int(self.dtype))
        self._emit_word(word)

    def _unary(self, activation):
        """EXEC_ALU with one operand: TRANSPOSE, or a nonlinearity by table."""
        self._wait(DEP["A"])
        if activation == ActivationMode.LUT:
            if self.lut_dram is None:
                raise ValueError("LUT kernel needs codegen.lut_dram set to the "
                                 "DRAM address of the 256-entry table")
            if not self._lut_loaded:
                load_word = encode(opcode=Opcode.LOAD_TILE, dep_id=6,
                                   dram_addr=self.lut_dram,
                                   sram_addr=SRAMMap.CORR_SCRATCH,
                                   dim_m=1, dim_n=256, stride=256,
                                   dtype=int(self.dtype))
                self._emit_word(load_word)
                self._wait(6)
                self._lut_loaded = True
            op_b = SRAMMap.CORR_SCRATCH
        else:
            op_b = 0
        word = encode(opcode=Opcode.EXEC_ALU, sram_addr=self.slot["A"],
                      stride=op_b, dim_m=TILE, dim_n=TILE,
                      activation=int(activation), dtype=int(self.dtype))
        self._emit_word(word)

    def _gemm(self, env, skip_wait=False):
        """The tensorized call. accumulate follows the K loop position.

        skip_wait: _gemm_k_loop already waited for this step's operands
        itself (it needs to, before issuing the next tile's prefetch) --
        WAIT_SYNC on an already-satisfied dep is harmless, but redundant,
        so the double-buffered path skips the second one explicitly rather
        than relying on that.
        """
        if not skip_wait:
            for k in ("A", "B"):
                self._wait(DEP[k])
        accumulate = env.get("__k__", 0) != 0
        word = encode(opcode=Opcode.COMPUTE_GEMM, dep_id=1,
                      sram_addr=self.slot["A"], stride=self.slot["B"],
                      dim_m=TILE, dim_n=TILE, dim_k=TILE,
                      activation=ActivationMode.NONE,
                      accumulate=accumulate, c_slot=self.c_slot,
                      dtype=int(self.dtype))
        self._emit_word(word)

    def _gemm_k_loop(self, node, env):
        """A GEMM's K-reduction loop, with the weight operand (B)
        double-buffered across two SRAM slots.

        Why: the generic tir.For recursion (one full LOAD/WAIT/COMPUTE per
        step, in TIR order) never overlaps a K-tile's weight DMA with
        anything -- confirmed directly against rtv_sim's cycle model
        before writing this: the next tile's LOAD_TILE isn't even ISSUED
        until the current tile's COMPUTE_GEMM instruction has already been
        emitted and executed, so there is nothing for LOAD_TILE's
        non-blocking latency to hide behind. Mirrors compile_linear's
        documented scheme in spirit -- weight tiles alternate
        weight_slot(0)/weight_slot(1), and the NEXT tile's DMA is issued
        before the CURRENT tile's compute so the DMA engine and PE array
        run concurrently -- but pipelines only ONE step ahead rather than
        compile_linear's two, specifically to avoid a real hazard that
        version would hit here: LOAD_TILE's dep_id is a small, REUSED tag
        (DEP["B"] == 2 always), and dep_done[dep_id] is simply overwritten
        on each new LOAD_TILE for that tag (confirmed against
        simulator.cpp). Issuing tile k+2's load before tile k's WAIT would
        overwrite dep_done[2] with k+2's completion time before k's own
        load was ever waited on -- safe in that it can only make a wait
        MORE conservative, never read stale data, but it defeats the point
        for that step. Waiting for step k's B FIRST, then prefetching
        step k+1 into the OTHER slot, then computing, means dep_done[2]
        always reflects exactly the load just waited for, and the
        prefetch itself overlaps with THIS step's COMPUTE_GEMM (the
        thing that actually advances sim_clock), not with a wait.

        The activation operand (A) is NOT double-buffered here either --
        neither is compile_linear's -- correctness first; overlapping A's
        load is a separate, later optimization there too.

        Falls back to the plain per-step recursion if the loop body isn't
        the exact [A_shared load, B_shared load, GEMM] shape schedule_matmul
        produces, rather than guessing at a structure that isn't there.
        """
        name = node.loop_var.name
        k_tiles = int(node.extent)
        body = node.body
        a_realize = b_realize = gemm_node = None
        if isinstance(body, tir.SeqStmt) and len(body.seq) >= 3:
            a_realize = self._staging_block(body.seq[0])
            b_realize = self._staging_block(body.seq[1])
            gemm_node = body.seq[2]
        if a_realize is None or b_realize is None:
            for v in range(k_tiles):
                self.visit(node.body, dict(env, **{name: v, "__k__": v}))
            return

        def load_a(step, slot_addr):
            self._tile_copy(a_realize, dict(env, **{name: step, "__k__": step}),
                            slot_override=slot_addr)

        def load_b(step, slot_addr):
            self._tile_copy(b_realize, dict(env, **{name: step, "__k__": step}),
                            slot_override=slot_addr)

        # A is ALSO double-buffered here, unlike compile_linear's version
        # (which leaves A single-slot, reloaded fresh every step). Measured
        # directly why that matters: with only B double-buffered, A's own
        # un-overlapped reload still takes its full latency every step, and
        # WAIT_SYNC's "stall to the LATEST unmet dependency" means A's fresh
        # wait dominates the critical path regardless of how well B's
        # prefetch overlapped -- B-only double buffering measured as a
        # complete no-op on rtv_sim's own cycle model (identical serial/db
        # numbers whether B was double-buffered or not). Buffering both
        # closes that gap.
        a_slots = [self.slot["A"], SRAMMap.input_slot(1, self.w)]
        b_slots = [self.slot["B"], SRAMMap.weight_slot(1, self.w)]
        load_a(0, a_slots[0])
        load_b(0, b_slots[0])
        for step in range(k_tiles):
            cur = step % 2
            self._wait(DEP["A"])
            self._wait(DEP["B"])
            nxt = step + 1
            if nxt < k_tiles:
                load_a(nxt, a_slots[nxt % 2])
                load_b(nxt, b_slots[nxt % 2])
            saved_a, saved_b = self.slot["A"], self.slot["B"]
            self.slot["A"], self.slot["B"] = a_slots[cur], b_slots[cur]
            self._gemm(dict(env, **{name: step, "__k__": step}), skip_wait=True)
            self.slot["A"], self.slot["B"] = saved_a, saved_b

    # -- walker ----------------------------------------------------------
    @staticmethod
    def _staging_block(node):
        """If this subtree is a whole-tile staging copy, return its block.

        The copy blocks sit INSIDE their `ax0, ax1` grid, so a naive walk
        iterates them 256 times and emits an instruction per ELEMENT. The
        tile move is one instruction, so the loops are recognised here and
        skipped rather than executed.
        """
        n = node
        while isinstance(n, tir.For):
            if not n.loop_var.name.startswith("ax"):
                return None
            n = n.body
        if isinstance(n, tir.SBlockRealize) and \
                n.block.name_hint.endswith("_shared"):
            return n              # the REALIZE: it carries the var bindings
        return None

    def visit(self, node, env):
        staged = self._staging_block(node)
        if staged is not None:
            self._tile_copy(staged, env)
            return
        if isinstance(node, tir.For):
            name = node.loop_var.name
            if name.startswith("k_") and self.kind == "gemm":
                self._gemm_k_loop(node, env)
                return
            for v in range(int(node.extent)):
                e = dict(env, **{name: v})
                if name.startswith("k_"):
                    e["__k__"] = v
                self.visit(node.body, e)
        elif isinstance(node, tir.SeqStmt):
            for s in node.seq:
                self.visit(s, env)
        elif isinstance(node, tir.SBlockRealize):
            # A block's iteration vars are bound HERE, not by the enclosing
            # loops: iter_values gives each block var as an expression of the
            # loop vars. Without binding them the block body's indices
            # reference names the environment has never seen.
            e = dict(env)
            for iv, val in zip(node.block.iter_vars, node.iter_values):
                e[iv.var.name] = _eval(val, env)
            self.visit(node.block, e)
        elif isinstance(node, tir.SBlock):
            nm = node.name_hint
            if nm.endswith("_shared"):
                self._tile_copy(node, env)
            elif "init" in nm:
                pass          # COMPUTE_GEMM with accumulate=False zeroes C
            else:
                self.visit(node.body, env)
        elif isinstance(node, tir.Evaluate):
            call = node.value
            if isinstance(call, tir_ir.Call):
                name = str(call.args[0])
                if "rtv_compute_gemm" in name:
                    self._gemm(env)
                elif "rtv_elemwise_add" in name:
                    self._elemwise(NPUActivations.ELEMWISE_ADD)
                elif "rtv_elemwise_mul" in name:
                    self._elemwise(NPUActivations.ELEMWISE_MUL)
                elif "rtv_reduce_sumsq" in name:
                    self._reduce(NPUActivations.REDUCE_SUMSQ)
                elif "rtv_reduce_max" in name:
                    self._reduce(NPUActivations.REDUCE_MAX, self.SCALAR_MAX)
                elif "rtv_reduce_sum" in name:
                    # Pass 2 accumulates the sum in a SECOND slot while the
                    # max from pass 1 is still live in the first.
                    self._reduce(NPUActivations.NONE, self.SCALAR_SUM)
                elif "rtv_rms_scale" in name:
                    self._tile_scalar(NPUActivations.RMS_SCALE, self.scale_num)
                elif "rtv_exp_sub" in name:
                    self._tile_scalar(NPUActivations.EXP_SUB, self.temperature,
                                      self.SCALAR_MAX)
                elif "rtv_scalar_div" in name:
                    self._tile_scalar(NPUActivations.SCALAR_DIV, 0,
                                      self.SCALAR_SUM)
                elif "rtv_lut" in name:
                    self._unary(NPUActivations.LUT)
                elif "rtv_silu_fp32" in name:
                    # FP32 SiLU: the true function, no table (see
                    # intrin.py's silu_fp32_impl for why this isn't LUT).
                    self._unary(NPUActivations.SILU)
                elif "rtv_transpose" in name:
                    self._unary(NPUActivations.TRANSPOSE)
        elif hasattr(node, "body"):
            self.visit(node.body, env)


def emit(compiler, sch, dram, dtype=DType.INT8, kind="gemm", scale_num=0,
         lut_dram=None, temperature=0):
    """Emit instructions for a scheduled PrimFunc into `compiler`."""
    cg = Codegen(compiler, dram, dtype, kind)
    cg.scale_num = scale_num
    cg.lut_dram = lut_dram
    cg.temperature = temperature
    cg.visit(sch.mod["main"].body, {})
    return compiler.instruction_stream
