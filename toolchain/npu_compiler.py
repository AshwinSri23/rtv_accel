"""
npu_compiler.py — RTV-Accel 4-Pass Optimizing AOT Compiler
Target: Qwen3-0.6B (28 blocks, H=1024, I=3072, V=151936, GQA 16→8 heads, head_dim=128)

Pipeline
========
  Pass 1 — Graph Ingestion    : walk model.named_parameters(), classify
                                 layers, assign static DRAM addresses.
  Pass 2 — Polyhedral Tiling  : slice M×N×K matrices into 16×16 blocks,
                                 zero-pad non-divisible dimensions.
  Pass 3 — List Scheduling    : overlap DMA loads with GEMM compute,
                                 manage SRAM zones, insert WAIT_SYNC fences.
  Pass 4 — Bit-Packing Backend: pack each operation node into a 16-byte
                                 128-bit instruction word and append to the
                                 binary stream.

Binary output is consumed by InstructionDispatcher (src/instruction_dispatcher.cpp)
which calls Decoder::decode() on each 16-byte chunk and routes to the
Simulator's PE array and memory controller.

Qwen3-0.6B key dimensions
--------------------------
  hidden_size       : 1024
  intermediate_size : 3072
  num_hidden_layers : 28
  num_attention_heads: 16   q_dim = 16 × 128 = 2048
  num_key_value_heads: 8    kv_dim =  8 × 128 = 1024
  head_dim          : 128
  vocab_size        : 151936
  rope_theta        : 1_000_000  (pre-computed on host, loaded via LOAD_TILE)
  rms_norm_eps      : 1e-6

Usage (no PyTorch model — standalone):
    compiler = NPUCompiler("toolchain/hal_config.json")
    compiler.compile_sibling_qkv(dram_x=0x001000, dram_wq=0x100000,
                                  dram_wk=0x200000, dram_wv=0x300000,
                                  dram_out=0x400000, m=16, n=1024, k=1024)
    compiler.append_synchronization_barrier(1)
    compiler.export_binary("out/qwen3_layer0_attention.bin")

Usage (with PyTorch model):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    compiler = NPUCompiler("toolchain/hal_config.json")
    compiler.compile_model(model)
    compiler.export_binary("out/qwen3_0.6b.bin")
"""

import struct
import json
import math


def _np_exp(x):
    import numpy as _np
    return _np.exp(x)
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# ── Pass 4: Opcode constants ──────────────────────────────────────────────────
# Bit field [127:124].  Values match Decoder::Opcode in inc/decoder.hpp.
# 0x0 is reserved / unassigned.

class NPUOpcodes:
    LOAD_TILE    = 0x1   # DMA: DRAM → SRAM
    STORE_TILE   = 0x2   # DMA: SRAM → DRAM
    COMPUTE_GEMM = 0x3   # 16×16 systolic matrix multiply
    EXEC_ALU     = 0x4   # Element-wise ALU (activation, quantise, SiLU…)
    WAIT_SYNC    = 0x5   # Pipeline barrier: stall until dep_id clears
    REDUCE       = 0x6   # Row-wise reduction → scalar (RMSNorm step 1-2)
    SCALAR_ALU   = 0x7   # Scalar broadcast to tile (RMSNorm step 3-5)


# ── Pass 4: Activation / ALU mode constants ───────────────────────────────────
# Carried in bits [7:0] of the 80-bit Params field.
# Values match Decoder::ActivationMode in inc/decoder.hpp.

class NPUActivations:
    NONE         = 0x00  # Pass-through (identity)
    RELU         = 0x01  # max(0, x)
    INT8_QUANT   = 0x02  # Scale and narrow INT32 → INT8
    SILU         = 0x03  # sigmoid(x)×x  (SwiGLU gate)
    SQUARE       = 0x04  # x²            (RMSNorm: sum of squares)
    RSQRT        = 0x05  # 1/√x          (RMSNorm: normalise)
    ELEMWISE_MUL = 0x06  # tile_A × tile_B element-wise (SwiGLU merge)
    SCALAR_ADD   = 0x07  # tile + scalar  (RMSNorm: mean(x²) + ε)
    ELEMWISE_ADD = 0x08  # tile_A + tile_B (transformer residual connections)
    TRANSPOSE    = 0x09  # in-place transpose of a square tile (Q·Kᵀ)
    REDUCE_MAX   = 0x0A  # REDUCE mode: max instead of sum (softmax stability)
    EXP_SUB      = 0x0B  # SCALAR_ALU: exp((tile − scalar)/T) (softmax numerator)
    SCALAR_DIV   = 0x0C  # SCALAR_ALU: tile × 127 / scalar (softmax normalise)
    CAUSAL_MASK  = 0x0D  # mask above the diagonal (causal attention)
    REDUCE_SUMSQ = 0x0E  # REDUCE mode: sum of x² in INT32 (RMSNorm)
    RMS_SCALE    = 0x0F  # SCALAR_ALU: tile × C / sqrt(scalar) (RMSNorm)
    PERCHAN_QUANT= 0x10  # EXEC_ALU: INT32 → INT8, per-output-channel mant/shift
    RESCALE      = 0x11  # EXEC_ALU: INT8 → INT8 scale conversion (mant/shift in dram_addr)
    LUT          = 0x12  # EXEC_ALU: 256-entry lookup table -- nonlinearities at true scale


# ── Pass 3: SRAM zone map ─────────────────────────────────────────────────────
# Three 1 KiB zones, each holding up to four 16×16 INT8 tiles (256 B each).
# Matches Component 3 of the spec exactly.

class DType:
    """Element widths, mirroring Decoder::DType."""
    INT8, INT16, INT32, FP32 = 0, 1, 2, 3

    BYTES = {0: 1, 1: 2, 2: 4, 3: 4}


class SRAMMap:
    INPUT_BASE  = 0x0000  # 0x0000–0x03FF  Active input / activation tiles
    WEIGHT_BASE = 0x0400  # 0x0400–0x07FF  Streaming weight buffers
    OUTPUT_BASE = 0x0800  # ISA v1.5 c_slot-addressed output zone (16 slots)

    ZONE_SIZE   = 0x0400  # 1 KiB per zone (input/weight zones only)
    TILE_BYTES  = 0x0100  # 256 bytes per 16×16 INT8 tile (input/weight slots)

    # Output slots hold INT32 accumulators (16*16*4 = 1024 bytes), not INT8
    # tiles -- must match src/simulator.cpp's c_slot_addr()'s SRAM_OUTPUT_SLOT_BYTES
    # exactly, or a COMPUTE_GEMM's c_slot and this helper disagree on where its
    # result actually lands. The old 256-byte spacing here (TILE_BYTES, sized
    # for INT8) predates ISA v1.5's c_slot field and only fit an 8x8 INT32
    # tile -- silently wrong for any real 16x16 GEMM output.
    OUTPUT_TILE_BYTES = 0x0400  # 1024 bytes -- 16*16*sizeof(int32)

    # Scratch SRAM for scalar intermediates (RMSNorm's sum-of-squares, etc.).
    # Must sit ABOVE every c_slot-addressable output slot: c_slot is 4 bits, so
    # slots 0..15 span OUTPUT_BASE .. OUTPUT_BASE + 16*OUTPUT_TILE_BYTES. Placing
    # scalar scratch inside that span (as the old OUTPUT_BASE + 3*TILE_BYTES did,
    # back when output slots were only 256 B apart) puts it on top of slot 0's
    # accumulator.
    SCALAR_SCRATCH_BASE = 0x0800 + 16 * 0x0400   # 0x4800

    # Correction table for EXEC_ALU PERCHAN_QUANT: one uint16 per output
    # column of a tile, so 16 columns * 2 bytes. Sits above the scalar scratch
    # (which holds a single INT32) rather than reusing it -- RMSNorm's
    # sum-of-squares can be live across a projection, and the two would
    # otherwise share an address.
    CORR_SCRATCH = SCALAR_SCRATCH_BASE + 0x0100   # 0x4900, 32 bytes used

    # Wide-element slot zones. A 16x16 tile is 1 KiB at 4 bytes/element, which
    # is the ENTIRE original input or weight zone -- so double-buffering the
    # weights, which compile_linear relies on to overlap DMA with compute,
    # cannot fit there. SRAM above the scratch area is unused (0x4920 to
    # 0x10000, ~46 KiB), so wide tiles get their own bases rather than the
    # narrow zones being resized, which would move every c_slot address and
    # break agreement with simulator.cpp's c_slot_addr().
    WIDE_INPUT_BASE  = 0x5000     # 4 slots x 1 KiB
    WIDE_WEIGHT_BASE = 0x6000     # 4 slots x 1 KiB
    WIDE_SLOT_BYTES  = 0x0400

    @staticmethod
    def input_slot(n: int = 0, width: int = 1) -> int:
        """
        SRAM address of input tile slot n.

        `width` is bytes per element. TILE_BYTES (256) sizes a 16x16 INT8
        tile, so at INT16 a tile is 512 bytes and the default spacing puts
        slot 0 on top of slot 1 -- which shows up as the upper half of every
        result being wrong, since rows 8-15 of slot 0 land where slot 1 begins.
        Slots must therefore be spaced by the ELEMENT WIDTH in use, and the
        1 KiB input zone holds four INT8 slots but only two INT16 ones.
        """
        if width >= 4:
            return SRAMMap.WIDE_INPUT_BASE + n * SRAMMap.WIDE_SLOT_BYTES
        addr = SRAMMap.INPUT_BASE + n * SRAMMap.TILE_BYTES * width
        limit = SRAMMap.INPUT_BASE + SRAMMap.ZONE_SIZE
        if addr + SRAMMap.TILE_BYTES * width > limit:
            raise ValueError(
                f"input_slot({n}, width={width}) needs "
                f"{SRAMMap.TILE_BYTES * width} bytes at 0x{addr:04X}, past the "
                f"input zone end 0x{limit:04X}")
        return addr

    @staticmethod
    def weight_slot(n: int = 0, width: int = 1) -> int:
        """SRAM address of weight tile slot n, spaced by element width.

        Same hazard as input_slot: TILE_BYTES sizes a 16x16 INT8 tile, so at
        4 bytes per element two slots overlap unless the spacing scales."""
        if width >= 4:
            return SRAMMap.WIDE_WEIGHT_BASE + n * SRAMMap.WIDE_SLOT_BYTES
        addr = SRAMMap.WEIGHT_BASE + n * SRAMMap.TILE_BYTES * width
        limit = SRAMMap.WEIGHT_BASE + SRAMMap.ZONE_SIZE
        if addr + SRAMMap.TILE_BYTES * width > limit:
            raise ValueError(
                f"weight_slot({n}, width={width}) overruns the weight zone")
        return addr

    @staticmethod
    def output_slot(n: int = 0) -> int:
        """SRAM address of output/accumulator tile slot n (0–15, ISA v1.5 c_slot).
        n=0 → Q accumulator, n=1 → K accumulator, n=2 → V accumulator (by convention
        for the compile_* callers that use them this way -- the slot itself is generic)."""
        return SRAMMap.OUTPUT_BASE + n * SRAMMap.OUTPUT_TILE_BYTES


# ── Pass 1: Layer classifier keywords ────────────────────────────────────────

PROJECTION_KEYS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "lm_head",
]

NORM_KEYS = ["input_layernorm", "post_attention_layernorm", "norm"]

# Qwen3 RMSNorms each attention head's Q and K over head_dim, before RoPE.
# Unlike the other norms these gains CANNOT be folded into the following
# weight matrix: the RMS divisor is computed from Q itself, so pre-scaling Q
# would change the very quantity being normalised by. They are applied at
# runtime instead -- see compile_attention_block.
QK_NORM_KEYS = ["q_norm", "k_norm"]

# The embedding table. Not a projection: it is never multiplied, only indexed.
EMBEDDING_KEYS = ["embed_tokens"]

# Tensors consumed in their native (out, in) orientation rather than as a
# GEMM B operand, and so exempt from the hf_layout transpose. lm_head is here
# because the vocabulary projection is computed TRANSPOSED -- see
# compile_lm_head for why that is forced by the 16-bit stride field.
NO_TRANSPOSE_KEYS = EMBEDDING_KEYS + ["lm_head"]


# ── Internal data structures ──────────────────────────────────────────────────

@dataclass
class WeightRecord:
    """One weight matrix extracted during Pass 1."""
    name:       str           # full parameter name, e.g. "model.layers.0.self_attn.q_proj.weight"
    key:        str           # classifier key, e.g. "q_proj"
    layer_idx:  int           # transformer block index (-1 for global)
    shape:      tuple
    dram_addr:  int           # assigned DRAM byte address
    n_bytes:    int           # byte size (padded to tile boundary)


@dataclass
class ScheduleNode:
    """Intermediate representation produced by the tiling pass, consumed by the scheduler."""
    opcode:     int
    dep_id:     int
    dram_addr:  int   # 32-bit DRAM address (full 4 GB range)
    sram_addr:  int
    # 72-bit Params sub-fields (passed directly to pack_instruction; no padding field)
    dim_m:      int = 0
    dim_n:      int = 0
    dim_k:      int = 0
    stride:     int = 0
    activation: int = 0
    # ISA v1.5 sub-fields (mirrors inc/decoder.hpp's DecodedInstruction).
    # Zero/False here decodes identically to v1 -- see _emit()'s wire-packing.
    accumulate:  bool = False   # bit 7 of the activation wire byte
    scale_shift: int  = 0       # bits 12:8 of the dim_m wire word (0-31)
    c_slot:      int  = 0       # bits 11:8 of the dim_k wire word (0-15)
    # Element width of the operand tile: 0=INT8, 1=INT16, 2=INT32. Zero is the
    # ISA's original behaviour, so an unset dtype encodes exactly as before.
    # INT16 exists for the residual stream, which accumulates over 28 layers
    # and holds Qwen3's massive activation channels.
    dtype:       int  = 0       # bits 14:13 of the dim_m wire word (0-3)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _n_tiles(dim: int, tile: int = 16) -> int:
    """Number of tiles needed to cover dim elements (ceiling division)."""
    return math.ceil(dim / tile)

def _padded(dim: int, tile: int = 16) -> int:
    """Next multiple of tile that covers dim (zero-padding bound)."""
    return _n_tiles(dim, tile) * tile

def _tile_bytes(rows: int = 16, cols: int = 16, elem_bytes: int = 1) -> int:
    return _padded(rows) * _padded(cols) * elem_bytes

def _pad_rows(a):
    """Pad a 2-D INT8 array's columns out to a tile boundary, matching the
    row pitch every LOAD_TILE addresses it with."""
    import numpy as np
    out = np.zeros((a.shape[0], _padded(a.shape[1])), dtype=a.dtype)
    out[:, :a.shape[1]] = a
    return out


def _align(n: int, align: int = 256) -> int:
    return ((n + align - 1) // align) * align


# ── NPUCompiler ───────────────────────────────────────────────────────────────

class NPUCompiler:
    """
    4-pass AOT compiler: model graph → flat binary instruction stream.

    Each pass is a distinct method group:
      _pass1_*  : Graph ingestion and weight extraction
      _pass2_*  : Polyhedral tiling (loop bound calculation)
      _pass3_*  : List scheduling (SRAM tracking, dep insertion)
      _pass4_*  : Bit-packing backend (pack_instruction)
    High-level compile_* methods orchestrate all four passes per operation.
    """

    def __init__(self, hal_config_path: str) -> None:
        with open(hal_config_path) as f:
            self.config = json.load(f)

        # Hardware constants from HAL profile
        self.tile_dim  : int = self.config.get("pe_array_dim",    16)
        self.sram_size : int = self.config.get("sram_size_bytes", 65536)

        # DRAM allocator cursor — Pass 1 bumps this for each weight tensor
        # Element width for the whole pipeline. INT8 is the accelerator's
        # design point; FP32 exists so the machine can run with NO
        # quantisation, which is what makes a lowering bug distinguishable
        # from a quantisation limit. Held on the compiler rather than threaded
        # through every compile_* call: this pipeline is reached through
        # compile_attention_block, compile_mlp_block, compile_sibling_qkv and
        # compile_model, and a parameter would be four more places to forget.
        self.dtype: int = DType.INT8

        self._dram_cursor: int = 0x0000_0000
        self._weight_map : Dict[str, WeightRecord] = {}

        # Transient staging region — intermediate results (a projection's output
        # feeding the next projection) that live in DRAM between steps. Placed
        # ABOVE the weights by init_staging(), never at a fixed address: the
        # weight region's size depends entirely on the model, so any hardcoded
        # staging address lands inside it for a large enough model.
        self._staging_base:   int = 0
        self._staging_cursor: int = 0
        # High-water mark across every staging generation. reset_staging()
        # rewinds the cursor each layer, so the cursor alone understates how
        # much DRAM the program actually touches; the simulator must be sized
        # from the peak, not from the image length (see dram_bytes_needed).
        self._staging_peak: int = 0

        # Non-weight constants that still must land in the DRAM image --
        # currently the RoPE cos/sin tables, which are computed rather than
        # ingested and so have no WeightRecord.
        self._const_regions: dict = {}

        # Binary output stream — Pass 4 appends 16-byte words here
        self.instruction_stream: bytearray = bytearray()

        # Scheduling state — Pass 3 uses these to track pipeline state
        self._dep_counter: int = 1    # cycles through 1–14 (0xF reserved)
        self._pending_dep: int = 0    # dep_id of the in-flight DMA load, if any

    # =========================================================================
    # Pass 4: Bit-Packing Backend
    # =========================================================================

    def pack_instruction(
        self,
        opcode:    int,
        dep_id:    int,
        dram_addr: int,
        sram_addr: int,
        params_int: int,
    ) -> bytes:
        """
        Pack all fields into a strict 16-byte (128-bit) big-endian word.

        Bit layout:
          [127:124] opcode    (4 b)
          [123:120] dep_id    (4 b)
          [119:88]  dram_addr (32 b)   ← full 32-bit; addresses 4 GB
          [87:72]   sram_addr (16 b)
          [71:0]    params    (72 b)   — no padding field

        params_int is a 72-bit integer; its internal structure is opaque to
        this function (it's just hi/lo-split into the 128-bit word) -- the
        caller (_emit) is responsible for packing it. As of ISA v1.5, that
        packing embeds sub-fields, not raw dim_m/dim_k/activation values:
          bits [71:56] = wire_dim_m  = (dtype << 13) | (scale_shift << 8) | dim_m
          bits [55:40] = dim_n       (unchanged, full 16-bit)
          bits [39:24] = wire_dim_k  = (c_slot << 8) | dim_k
          bits [23:8]  = stride      (unchanged)
          bits [7:0]   = wire_act    = (accumulate << 7) | activation
        See _emit() for the actual packing and inc/decoder.hpp's v1.5 section
        for the full rationale.
        """
        # Upper 64-bit word: opcode[63:60] dep_id[59:56] dram[55:24] sram[23:8] dim_m_hi[7:0]
        word_high  = (opcode    & 0xF)           << 60
        word_high |= (dep_id    & 0xF)           << 56
        word_high |= (dram_addr & 0xFFFF_FFFF)   << 24   # 32-bit field
        word_high |= (sram_addr & 0xFFFF)        <<  8
        word_high |= (params_int >> 64)          & 0xFF  # params bits [71:64] = dim_m upper byte

        # Lower 64-bit word: dim_m_lo[63:56] dim_n[55:40] dim_k[39:24] stride[23:8] act[7:0]
        word_low = params_int & 0xFFFF_FFFF_FFFF_FFFF

        return struct.pack(">QQ", word_high, word_low)

    def _emit(self, node: ScheduleNode) -> None:
        """Assemble one ScheduleNode to bytes and append to the stream."""
        # Pack the 72-bit params field from individual sub-fields.
        # Layout: wire_dim_m[71:56] | dim_n[55:40] | wire_dim_k[39:24] | stride[23:8] | wire_act[7:0]
        # dim_m upper byte lands in hi[7:0]; lower byte in lo[63:56] via pack_instruction.
        #
        # ISA v1.5: wire_dim_m/wire_dim_k/wire_act embed scale_shift/c_slot/
        # accumulate in bit positions that are zero under v1 -- mirrors
        # inc/decoder.hpp's Decoder::encode() and toolchain/rtv_isa.py's
        # encode() exactly. A node with all three v1.5 fields at their
        # dataclass defaults (False/0/0) packs identically to before.
        wire_dim_m = (((node.dtype & 0x3) << 13)
                      | ((node.scale_shift & 0x1F) << 8) | (node.dim_m & 0xFF))
        wire_dim_k = ((node.c_slot      & 0xF)  << 8) | (node.dim_k & 0xFF)
        wire_act   = (0x80 if node.accumulate else 0x00) | (node.activation & 0x7F)

        params = (
            (wire_dim_m  & 0xFFFF) << 56 |
            (node.dim_n  & 0xFFFF) << 40 |
            (wire_dim_k  & 0xFFFF) << 24 |
            (node.stride & 0xFFFF) <<  8 |
            (wire_act    & 0xFF)
        )
        self.instruction_stream.extend(
            self.pack_instruction(
                opcode    = node.opcode,
                dep_id    = node.dep_id,
                dram_addr = node.dram_addr,
                sram_addr = node.sram_addr,
                params_int= params,
            )
        )

    # =========================================================================
    # Pass 1: Graph Ingestion & Weight Extraction
    # =========================================================================

    def _pass1_alloc_dram(self, n_bytes: int) -> int:
        """Bump-allocate from the simulated 4 GB DRAM; returns base address."""
        addr = self._dram_cursor
        self._dram_cursor += _align(n_bytes, 256)   # 256-byte tile alignment
        return addr

    # ── Transient staging allocator ──────────────────────────────────────────
    #
    # Separate from _pass1_alloc_dram's weight allocator, and deliberately so:
    # weights are permanent (allocated once at ingestion, live for the whole
    # program), while staging buffers are transient (a projection's output,
    # dead as soon as the next projection consumes it). Staging therefore
    # bump-allocates from its own cursor that rewinds every layer, so layer N's
    # buffers reuse layer N-1's space -- correct because no staging buffer is
    # live across a layer boundary.

    def init_staging(self) -> None:
        """
        Place the staging region immediately above the last allocated weight.

        Must be called AFTER ingest_model(), since the weight region's extent
        is model-dependent. This is why staging cannot use a fixed address:
        Qwen3-0.6B's INT8 weights alone span ~600 MB from 0x0000, so a
        hardcoded staging base (the old 0xF00000, ~15 MB) lands squarely
        inside the weights and corrupts them on first write.
        """
        self._staging_base   = _align(self._dram_cursor, 256)
        self._staging_cursor = self._staging_base

    def reset_staging(self) -> None:
        """Rewind staging to its base — call once per layer."""
        self._staging_cursor = self._staging_base

    def alloc_staging(self, n_bytes: int) -> int:
        """Bump-allocate one transient staging buffer; returns its DRAM address."""
        if self._staging_base == 0 and self._staging_cursor == 0:
            # Standalone use (compile_* called without compile_model): place
            # staging past whatever the weight allocator has handed out.
            self.init_staging()
        addr = self._staging_cursor
        self._staging_cursor += _align(n_bytes, 256)
        self._staging_peak = max(self._staging_peak, self._staging_cursor)
        return addr

    @property
    def dram_bytes_needed(self) -> int:
        """
        Highest DRAM byte the compiled program can touch.

        This is NOT the length of the exported image. The image ends at the
        last byte with initial CONTENT (weights, tables); staging buffers are
        allocated above that and are written at run time, so they extend past
        the image end and contribute nothing to its length. Sizing the
        simulator's DRAM from the image -- as the probe did -- leaves the
        staging tail hanging off the end of memory, which faults as
        "store_tile: DRAM address out of range" only when the tail happens to
        exceed whatever slack the caller padded on. Size from this instead.
        """
        return max(self._dram_cursor, self._staging_peak, self._staging_cursor)

    def _pass1_classify(self, param_name: str) -> Optional[str]:
        """
        Return the projection key for a named parameter, or None.

        Matching is on dot-separated COMPONENTS, not raw substrings. A plain
        `key in param_name` test made "norm" match "q_norm", so Qwen3's
        per-head q_norm/k_norm were filed as the final model norm -- and since
        both matched the same key in the same layer, one silently overwrote
        the other in _weight_map. Any key that is a substring of another key
        has the same hazard; component matching removes the whole class.
        """
        parts = param_name.split(".")
        for key in PROJECTION_KEYS + NORM_KEYS + QK_NORM_KEYS + EMBEDDING_KEYS:
            if key in parts:
                return key
        return None

    def _pass1_layer_index(self, param_name: str) -> int:
        """Extract the transformer block index from a dotted parameter name."""
        parts = param_name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return -1  # global (embedding, final norm, lm_head)

    def ingest_model(self, model, capture_weights: bool = False,
                     hf_layout: bool = True, per_channel: bool = True):
        """
        Pass 1: walk model.named_parameters(), classify each tensor, and assign
        it a static DRAM base address.

        Returns a dict keyed by parameter name.
        The model argument must support .named_parameters() (PyTorch / HF).

        capture_weights -- also quantize each tensor to INT8 and retain it, so
        export_dram_image() can lay out a DRAM blob the simulator can actually
        execute against. Off by default because Qwen3-0.6B's weights are
        ~440 MB and most callers only need the addresses.

        hf_layout -- transpose 2-D projection weights on capture. PyTorch and
        Hugging Face store nn.Linear weights as (out_features, in_features)
        and compute x @ W.T, but this compiler's GEMM consumes B directly as
        (in, out). Feeding a real checkpoint through without transposing
        silently computes x @ W -- shape-compatible only when in == out, and
        wrong everywhere else. Embedding tables are exempt: embed_tokens is
        already (vocab, hidden), i.e. indexed by row, and must not be flipped.

        Sizing note: each tensor is allocated at its PADDED footprint,
        _padded(rows) * _padded(cols), not _align(rows*cols, 256). The
        addressing in _pass2_dram_stride() strides rows by _padded(cols), so a
        tensor whose column count is not a multiple of the tile size occupies
        more space than its raw element count suggests. Allocating by raw
        element count under-reserved (a 17x17 tensor needs 32*32 = 1024 bytes
        but was given _align(289, 256) = 512), letting the next tensor's base
        land inside the previous one's padded rows.
        """
        self._weight_map.clear()
        self._weight_bytes = {}
        self._unclassified = []
        self._corr_for = {}
        self._corr_addr = {}
        self._weight_peaks = {}
        folds = self._collect_gamma_folds(model) if capture_weights else {}
        self._folded_gammas = set()

        for name, param in model.named_parameters():
            key = self._pass1_classify(name)
            if key is None:
                # Silence here is what let q_norm/k_norm go unnoticed: a
                # checkpoint tensor no key claims is either an architecture
                # feature this compiler does not implement or a naming drift,
                # and both are worth saying out loud.
                self._unclassified.append(name)
                continue

            shape = tuple(param.shape)
            # Embedding rows are indexed, not multiplied -- never transposed.
            flip = (hf_layout and len(shape) == 2
                    and key not in NO_TRANSPOSE_KEYS)
            if len(shape) >= 2:
                rows, cols = (shape[1], shape[0]) if flip else (shape[0], shape[1])
            else:
                rows, cols = 1, (shape[0] if shape else 1)

            n_bytes = _padded(rows) * _padded(cols) * DType.BYTES[self.dtype]
            dram    = self._pass1_alloc_dram(n_bytes)

            self._weight_map[name] = WeightRecord(
                name      = name,
                key       = key,
                layer_idx = self._pass1_layer_index(name),
                shape     = shape,
                dram_addr = dram,
                n_bytes   = n_bytes,
            )

            if capture_weights:
                src = param
                gamma = folds.get(name)
                if gamma is not None:
                    import numpy as np
                    raw = (param.detach().cpu().numpy()
                           if hasattr(param, "detach") else param)
                    raw = np.asarray(raw, dtype=np.float32)
                    # HF stores nn.Linear as (out_features, in_features), and
                    # gamma indexes in_features -- axis 1 -- whether or not
                    # this tensor is later transposed into (in, out).
                    src = raw * gamma[None, :]
                    self._folded_gammas.add(name)
                if self.dtype == DType.FP32:
                    # No quantisation at all: the weights go to DRAM as the
                    # checkpoint stores them. Nothing to scale, so no
                    # correction table and no calibration.
                    import numpy as _np
                    raw = (src.detach().cpu().numpy()
                           if hasattr(src, "detach") else src)
                    arr = _np.asarray(raw, dtype=_np.float32)
                    arr = arr.T if flip else arr
                    arr = arr.reshape(rows, cols)
                    padded = _np.zeros((_padded(rows), _padded(cols)),
                                       dtype=_np.float32)
                    padded[:rows, :cols] = arr
                    self._weight_bytes[name] = padded
                    continue
                # Projections quantise per output channel; everything else
                # (embeddings, norms) keeps the per-tensor path -- embed_tokens
                # is indexed rather than multiplied, so it has no GEMM to carry
                # a correction, and the norms are folded or applied directly.
                if per_channel and key in PROJECTION_KEYS and len(shape) >= 2:
                    q, tab, peaks = self._quantize_per_channel(
                        src, rows, cols, transpose=flip)
                    self._weight_bytes[name] = q
                    # Retained so the table can be REBUILT once calibration
                    # says what scale this projection's output must land in.
                    # Ingest happens before scales are known, and a table built
                    # here normalises to the widest column -- an accidental
                    # output scale that no later RESCALE constant can assume.
                    self._weight_peaks[name] = peaks
                    tb = tab.astype("<u2").tobytes()
                    taddr = self._pass1_alloc_dram(len(tb))
                    self._const_regions[taddr] = tb
                    self._corr_for[dram] = taddr
                    self._corr_addr[name] = taddr
                else:
                    self._weight_bytes[name] = self._quantize_int8(
                        src, rows, cols, transpose=flip)

        if self._unclassified:
            import re as _re
            gen = sorted({_re.sub(r"\.\d+\.", ".N.", n) for n in self._unclassified})
            print(f"[NPUCompiler] warning: {len(self._unclassified)} checkpoint "
                  f"tensor(s) matched no key and were NOT compiled:")
            for g in gen:
                print(f"    {g}")
        return self._weight_map

    def lut_table(self, fn, in_scale: float, out_scale: float = None):
        """
        Build a 256-entry INT8 lookup table for `fn` at a known input scale.

        Returns (table bytes, out_scale). Entry i corresponds to code i-128,
        whose true value is (i-128)*in_scale, so `fn` is evaluated on the
        VALUE and the result stored back as a code in out_scale.

        This is why the mode exists. Applying a nonlinearity to the raw code
        computes a different function: EXEC_ALU SILU evaluates
        x/(1+exp(-x)) on the INT8 code, and SiLU's curvature lives near
        |x| < 6, so at integer granularity the result is indistinguishable
        from ReLU for ALL 256 codes -- Qwen3's SwiGLU gate was a ReLU gate.
        A table sidesteps it entirely and costs one SRAM load.

        out_scale defaults to the table's own peak, which keeps the output
        using the full INT8 range.
        """
        import numpy as np
        codes = np.arange(-128, 128, dtype=np.float64)
        vals = np.asarray(fn(codes * float(in_scale)), dtype=np.float64)
        if out_scale is None:
            pk = float(np.abs(vals).max())
            out_scale = pk / 127.0 if pk > 0 else 1.0
        q = np.clip(np.round(vals / out_scale), -128, 127).astype(np.int8)
        return q.tobytes(), float(out_scale)

    def lut_dram(self, fn, in_scale: float, name: str, out_scale: float = None):
        """Place a LUT in DRAM as a const region; cached by name."""
        cache = getattr(self, "_lut_regions", None)
        if cache is None:
            cache = self._lut_regions = {}
        if name in cache:
            return cache[name]
        tbl, s_out = self.lut_table(fn, in_scale, out_scale)
        addr = self.alloc_staging(len(tbl))
        self._const_regions[addr] = tbl
        cache[name] = (addr, s_out)
        return addr, s_out

    def compile_lut(self, sram_addr: int, dim_m: int, dim_n: int,
                    dram_table: int, dep_id: int = 6) -> None:
        """Load a 256-byte LUT into SRAM and apply it to a tile in place."""
        self._emit(self._pass3_load(
            dram_addr = dram_table,
            sram_addr = SRAMMap.CORR_SCRATCH,
            dim_m = 1, dim_n = 256, dep_id = dep_id, stride = 256,
        ))
        self._emit(self._pass3_wait(dep_id))
        self.compile_exec_alu(
            sram_addr      = sram_addr,
            dim_m = dim_m, dim_n = dim_n,
            activation     = NPUActivations.LUT,
            operand_b_sram = SRAMMap.CORR_SCRATCH,
        )

    def gamma_tile_dram(self, gamma, name: str) -> int:
        """
        Place a Q0.7 gamma vector in DRAM, broadcast to a full tile of rows.

        ELEMWISE_MUL reads a (tile_dim x width) operand, so a per-channel
        vector has to be replicated down the tile's rows; every row of a
        staged activation then multiplies the same gains. Registered as a
        const region and cached by name, so 28 layers sharing a shape do not
        each re-emit an identical table.

        Quantised per-tensor to Q0.7 (127 == max|gamma|). ELEMWISE_MUL's
        fixed (a*b)>>7 makes the output carry an implicit 127/max|gamma|
        scale, which the downstream INT8_QUANT shift absorbs -- the same
        treatment every other activation scale in this compiler gets.
        """
        import numpy as np
        cache = getattr(self, "_gamma_regions", None)
        if cache is None:
            cache = self._gamma_regions = {}
        if name in cache:
            return cache[name]

        g = np.asarray(gamma, dtype=np.float32).reshape(-1)
        n = g.size
        if self.dtype == DType.FP32:
            # No Q0.7 convention in float: ELEMWISE_MUL is a plain multiply,
            # so the gain goes in as itself. Quantising it here would put an
            # INT8 tile into a float pipeline.
            tile = np.zeros((self.tile_dim, _padded(n)), dtype=np.float32)
            tile[:, :n] = g[None, :]
        else:
            peak = float(np.max(np.abs(g))) if g.size else 0.0
            q = (np.clip(np.round(g * (127.0 / peak)), -127, 127).astype(np.int8)
                 if peak > 0 else np.zeros(n, dtype=np.int8))
            tile = np.zeros((self.tile_dim, _padded(n)), dtype=np.int8)
            tile[:, :n] = q[None, :]
        addr = self.alloc_staging(tile.nbytes)
        self._const_regions[addr] = tile.tobytes()
        cache[name] = addr
        return addr

    # Which projection each RMSNorm's learned gain folds into. RMSNorm is
    # immediately followed by GEMMs against these weights, and
    #     (x * gamma) @ W  ==  x @ (gamma[:, None] * W)
    # because gamma is diagonal. Folding at compile time is exact, costs no
    # instructions, and -- crucially -- avoids quantising gamma itself:
    # Qwen3's norm gains span 0.017..192 within a single tensor, a 2757:1
    # dynamic range that INT8 cannot hold, so a runtime ELEMWISE_MUL against
    # an INT8 gamma would flush every small channel to zero. Folded into W
    # before quantisation, the range is absorbed by W's own per-tensor scale.
    GAMMA_FOLD = {
        "input_layernorm":          ("q_proj", "k_proj", "v_proj"),
        "post_attention_layernorm": ("gate_proj", "up_proj"),
        "norm":                     ("lm_head",),
    }

    def _collect_gamma_folds(self, model) -> dict:
        """
        Map each projection parameter name -> the gamma vector to fold into it.

        Keyed by (layer_idx, key) so a layer's norm only reaches that layer's
        projections; the final `model.norm` sits at layer -1 alongside lm_head.
        """
        import numpy as np

        norms, projs = {}, {}
        for name, param in model.named_parameters():
            key = self._pass1_classify(name)
            if key is None:
                continue
            slot = (self._pass1_layer_index(name), key)
            if key in self.GAMMA_FOLD:
                arr = (param.detach().cpu().numpy()
                       if hasattr(param, "detach") else param)
                norms[slot] = np.asarray(arr, dtype=np.float32).reshape(-1)
            else:
                projs[slot] = name

        folds = {}
        for (layer, nkey), gamma in norms.items():
            for target in self.GAMMA_FOLD[nkey]:
                tname = projs.get((layer, target))
                if tname is not None:
                    folds[tname] = gamma
        return folds

    @staticmethod
    def encode_ratio(ratio: float):
        """
        Encode a positive real multiplier as (mantissa, shift) with
        multiplier == mant / 2**shift, mant in [128, 255].

        The ISA's scale_shift is a power of two, which cannot express the
        ratios that actually arise: aligning the residual stream with a
        sublayer output at layer 0 of Qwen3 needs 4.26. An 8-bit mantissa with
        its own shift represents any ratio to ~0.4% -- measured sufficient,
        where rounding to a power of two is not (rank 6223 vs 0).

        Returns (0, 255) for a non-positive ratio, which multiplies to ~0.
        """
        import math
        if not (ratio > 0) or not math.isfinite(ratio):
            return 0, 255
        e = math.ceil(math.log2(128.0 / ratio))
        e = max(0, min(255, int(e)))
        mant = int(round(ratio * (2.0 ** e)))
        if mant > 255:                      # ratio >= 2: shift down one more
            e = max(0, e - 1)
            mant = int(round(ratio * (2.0 ** e)))
        return max(0, min(255, mant)), e

    def calibrate_residual_scales(self, model, token_ids, ref=None):
        """
        Peak magnitude of each layer's residual buffers, from the FLOAT model.

        Returns [s_0, s_1, ... s_L] where s_i is the scale (real value of one
        INT8 step) that layer i's residual arithmetic uses.

        Every activation scale in this compiler was previously implicit -- a
        side effect of whatever chain of >>7 requantisations produced the
        buffer -- so two operands meeting at an ELEMWISE_ADD had unrelated
        units. Calibration makes them explicit: the scale is chosen from the
        magnitudes the model actually produces, so both operands can be
        targeted at the same one.

        Per LAYER rather than one global scale because the residual grows
        ~600x across the stack (max|x| is 0.07 at layer 0 and 291 at layer
        27); a single INT8 scale large enough for the last layer rounds the
        first few to zero.
        """
        import numpy as np
        if ref is None:
            from ref_model import RefQwen3
            ref = RefQwen3(model)
        states = ref.forward(list(token_ids))
        peaks = [float(np.abs(np.asarray(x)).max()) for x in states]
        # Layer i reads states[i] and writes states[i+1]; both, plus the
        # intermediate x+attn, share the layer's scale, so size it by the
        # larger endpoint.
        scales = []
        for i in range(len(peaks) - 1):
            pk = max(peaks[i], peaks[i + 1])
            scales.append((pk / 127.0) if pk > 0 else 1.0)
        scales.append((peaks[-1] / 127.0) if peaks[-1] > 0 else 1.0)
        self.residual_scales = scales
        return scales

    @staticmethod
    def _quantize_per_channel(param, rows: int, cols: int, transpose: bool = False,
                              in_scale: float = None, out_scale: float = None):
        """
        Per-OUTPUT-CHANNEL symmetric INT8 quantisation.

        Returns (int8 array padded to tile bounds, correction table).

        Weights land in the GEMM as (in, out), so an output channel is a
        COLUMN of the stored matrix and each column gets its own scale
        max|w[:, c]| / 127. Per-tensor quantisation -- one scale for the whole
        matrix -- is not adequate for a real LLM: measured on Qwen3-0.6B it
        drives the hidden state to anti-correlation with ground truth by layer
        27 (cosine -0.23, ' Paris' at rank 110766), and no choice of clipping
        percentile rescues it. Per-channel is near-lossless over the same 28
        layers (cosine 0.9992, rank 0).

        The correction table restores the relative magnitudes the per-column
        scales removed. Each entry is mant / 2^shift with mant in [128, 255],
        packed as one uint16 per column (low byte mant, high byte shift) for
        EXEC_ALU PERCHAN_QUANT. A mantissa is needed because rounding each
        scale to a power of two -- which the existing scale_shift field could
        have expressed for free -- still fails (rank 6223).
        """
        import numpy as np

        arr = np.asarray(
            param.detach().cpu().numpy() if hasattr(param, "detach") else param,
            dtype=np.float32,
        )
        arr = arr.T if transpose else arr
        arr = arr.reshape(rows, cols)

        peak = np.abs(arr).max(axis=0)                     # one per output column
        safe = np.where(peak > 0, peak, 1.0).astype(np.float64)
        q = np.clip(np.round(arr * (127.0 / safe)[None, :]), -127, 127)

        # Column c's accumulator is in units of (in_scale * safe[c]/127).
        #
        # With in_scale/out_scale given, the correction targets an EXPLICIT
        # output scale: the consumer states what one INT8 step of its input
        # must be worth, and this lands the result there. That is what makes
        # two buffers meeting at an ELEMWISE_ADD commensurable -- without it
        # each buffer's scale is an accident of its own arithmetic and the add
        # mixes units (see the scale-bookkeeping section of the spec).
        #
        # Without them, fall back to normalising against the widest column,
        # which keeps every multiplier <= 1 but leaves the output scale
        # implicit -- correct only when nothing later adds to this result.
        if in_scale is not None and out_scale is not None and out_scale > 0:
            rel = (float(in_scale) * safe) / (127.0 * float(out_scale))
        else:
            rel = safe / safe.max()
        e = np.ceil(np.log2(128.0 / rel))                  # mant = rel*2^e in [128,255]
        e = np.clip(e, 0, 255)
        mant = np.clip(np.round(rel * (2.0 ** e)), 128, 255)
        table = ((e.astype(np.uint16) << 8) | mant.astype(np.uint16)).astype(np.uint16)

        out = np.zeros((_padded(rows), _padded(cols)), dtype=np.int8)
        out[:rows, :cols] = q.astype(np.int8)
        tab = np.zeros(_padded(cols), dtype=np.uint16)
        tab[:cols] = table
        # A padded column must not scale a real one: mant=128, shift=255 gives
        # a multiplier of ~0, and the padded accumulator is zero anyway.
        tab[cols:] = (255 << 8) | 128
        return out, tab, safe

    @staticmethod
    def _quantize_int8(param, rows: int, cols: int, transpose: bool = False):
        """
        Per-tensor symmetric INT8 quantization, laid out row-major with rows
        padded to the tile boundary that _pass2_dram_stride assumes.

        scale = max|w| / 127, q = round(w / scale) clipped to [-127, 127].
        Returns a numpy int8 array of shape (_padded(rows), _padded(cols)).

        The scale is NOT recorded in the instruction stream: the ISA's
        scale_shift field is a power-of-two right shift, so an arbitrary
        float scale cannot be represented. Requantization between stages
        therefore uses INT8_QUANT's shift, and this scale exists only to get
        weights into INT8 range sensibly.
        """
        import numpy as np

        arr = np.asarray(
            param.detach().cpu().numpy() if hasattr(param, "detach") else param,
            dtype=np.float32,
        )
        arr = arr.T if transpose else arr
        arr = arr.reshape(rows, cols)

        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        if peak > 0:
            q = np.clip(np.round(arr * (127.0 / peak)), -127, 127)
        else:
            q = np.zeros_like(arr)

        out = np.zeros((_padded(rows), _padded(cols)), dtype=np.int8)
        out[:rows, :cols] = q.astype(np.int8)
        return out

    def export_dram_image(self, output_filepath: str, extra=None) -> int:
        """
        Write a flat DRAM image with every captured weight at the address Pass 1
        assigned it. Requires ingest_model(..., capture_weights=True).

        extra -- optional {dram_addr: bytes-like} of additional regions to place
        (e.g. the model's input activations), merged after the weights.

        Returns the image size in bytes. Feed it to the simulator with
        `rtv_sim program.bin --dram-image <file>`; without it DRAM is
        zero-initialised and every result is trivially zero.
        """
        if not getattr(self, "_weight_bytes", None) and not self._const_regions:
            raise RuntimeError(
                "export_dram_image: no captured weights — call "
                "ingest_model(model, capture_weights=True) first."
            )

        regions = {rec.dram_addr: self._weight_bytes[name].tobytes()
                   for name, rec in self._weight_map.items()
                   if name in self._weight_bytes}
        regions.update({a: bytes(b) for a, b in self._const_regions.items()})
        for addr, blob in (extra or {}).items():
            regions[addr] = bytes(blob)

        size = max(addr + len(blob) for addr, blob in regions.items())
        image = bytearray(size)
        for addr, blob in regions.items():
            image[addr:addr + len(blob)] = blob

        os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)
        with open(output_filepath, "wb") as f:
            f.write(image)
        print(f"[NPUCompiler] Written DRAM image ({size} bytes) → {output_filepath}")
        return size

    # =========================================================================
    # Pass 2: Polyhedral Tiling
    # =========================================================================

    def _pass2_tile_bounds(
        self, M: int, N: int, K: int
    ) -> Tuple[int, int, int]:
        """
        Return (m_tiles, n_tiles, k_tiles).

        Zero-padding guard: if any dimension is not divisible by tile_dim,
        the tiling engine pads the DRAM allocation to the next multiple so
        the C++ simulator always fetches uniform 16×16 blocks.
        """
        t = self.tile_dim
        m_tiles = math.ceil(M / t)
        n_tiles = math.ceil(N / t)
        k_tiles = math.ceil(K / t)
        return m_tiles, n_tiles, k_tiles

    def _pass2_dram_stride(
        self, row: int, col: int, full_cols: int, elem_bytes: int = 1
    ) -> int:
        """
        Byte offset into a row-major matrix for tile (row, col).

          offset = row * tile_dim * padded(full_cols) * elem_bytes
                 + col * tile_dim * elem_bytes
        """
        t = self.tile_dim
        return (row * t * _padded(full_cols) + col * t) * elem_bytes

    # =========================================================================
    # Pass 3: List Scheduling
    # =========================================================================

    def _pass3_next_dep(self) -> int:
        """Allocate the next dependency vector tag (cycles 1–14; 0 = no dep)."""
        dep = self._dep_counter
        self._dep_counter = (self._dep_counter % 14) + 1
        return dep

    def _pass3_load(
        self,
        dram_addr: int,
        sram_addr: int,
        dim_m: int = 16,
        dim_n: int = 16,
        dep_id: int = 0,
        stride: int = 0,
    ) -> ScheduleNode:
        """
        Emit LOAD_TILE for one tile.

        stride — byte distance between consecutive ROWS of the tile in DRAM,
        i.e. the full row width of the matrix the tile is being cut out of
        (NOT the tile's own width). MemoryController::load_tile copies
        `dim_n` bytes per row, advancing `stride` bytes per row; extracting a
        16x16 tile from a 1024-wide matrix therefore needs stride=1024.

        Leaving this 0 makes the simulator fall back to `dim_n` (16), which
        reads 256 CONTIGUOUS bytes instead of gathering 16 strided rows --
        silently loading the wrong data for any matrix wider than one tile.
        stride=16 is only correct for block-linear (swizzled) layout; this
        compiler addresses DRAM row-major (see _pass2_dram_stride), so it
        must pass the real row width.
        """
        if stride > 0xFFFF:
            raise ValueError(
                f"_pass3_load: stride {stride} exceeds the 16-bit stride field. "
                "A matrix wider than 65535 columns cannot have its row pitch "
                "encoded; the operand must be laid out so the pitch fits "
                "(e.g. lm_head is consumed transposed, pitch=hidden, rather "
                "than as a vocab-wide B operand)."
            )
        return ScheduleNode(
            opcode    = NPUOpcodes.LOAD_TILE,
            dep_id    = dep_id,
            dram_addr = dram_addr & 0xFFFF_FFFF,  # 32-bit
            sram_addr = sram_addr & 0xFFFF,
            dim_m     = dim_m,
            dim_n     = dim_n,
            stride    = stride & 0xFFFF,
        )

    def _pass3_gemm(
        self,
        sram_addr:   int,
        dim_m:       int = 16,
        dim_n:       int = 16,
        dim_k:       int = 16,
        dep_id:      int = 1,
        activation:  int = NPUActivations.NONE,
        b_sram:      int = SRAMMap.WEIGHT_BASE,
        accumulate:  bool = False,
        scale_shift: int  = 0,
        c_slot:      int  = 0,
        dtype:       int = 0,
    ) -> ScheduleNode:
        # b_sram encodes which weight slot to use as matrix B.
        # Stored in the stride field; simulator reads it as:
        #   b_sram = stride if stride != 0 else WEIGHT_BASE
        # Default WEIGHT_BASE (0x0400) makes existing callers without b_sram work unchanged.
        #
        # accumulate (ISA v1.5): False zeroes the C accumulator before this
        # GEMM (v1 behavior, correct for the first tile of a K-reduction);
        # True adds into whatever is already there. A K-tiled caller MUST
        # pass accumulate=True for every step after the first, or only the
        # last K-step's partial product survives -- see compile_linear's and
        # compile_sibling_qkv's fix, and the ISA v1.5 section of the spec doc.
        return ScheduleNode(
            opcode      = NPUOpcodes.COMPUTE_GEMM,
            dep_id      = dep_id,
            dram_addr   = 0,
            sram_addr   = sram_addr & 0xFFFF,
            dim_m       = dim_m,
            dim_n       = dim_n,
            dim_k       = dim_k,
            stride      = b_sram & 0xFFFF,
            activation  = activation,
            accumulate  = accumulate,
            scale_shift = scale_shift,
            dtype       = dtype,
            c_slot      = c_slot,
        )

    def _pass3_wait(self, dep_id: int) -> ScheduleNode:
        return ScheduleNode(
            opcode    = NPUOpcodes.WAIT_SYNC,
            dep_id    = dep_id,
            dram_addr = 0,
            sram_addr = 0,
        )

    def _pass3_exec_alu(
        self,
        sram_addr:  int,
        dim_m:      int = 16,
        dim_n:      int = 16,
        activation: int = NPUActivations.NONE,
        stride:     int = 0,
        params:     int = 0,
        dtype:      int = 0,
    ) -> ScheduleNode:
        return ScheduleNode(
            opcode     = NPUOpcodes.EXEC_ALU,
            dep_id     = 0,
            # dram_addr carries immediate operands for modes that need a
            # constant rather than an address: RESCALE's mantissa/shift,
            # ELEMWISE_MUL's shift, the widen's left shift.
            dram_addr  = params,
            sram_addr  = sram_addr & 0xFFFF,
            dtype      = dtype,
            dim_m      = dim_m,
            dim_n      = dim_n,
            stride     = stride & 0xFFFF,
            activation = activation,
        )

    def _pass3_store(
        self,
        sram_addr: int,
        dram_addr: int,
        dim_m:     int = 16,
        dim_n:     int = 16,
        stride:    int = 0,
    ) -> ScheduleNode:
        """
        Emit STORE_TILE for one tile.

        dim_n and stride are both in BYTES, not elements — store_tile is
        dtype-agnostic and just memcpy's `dim_n` bytes per row, advancing
        `stride` bytes per destination row. A 16x16 tile of INT32
        accumulators is therefore dim_n = 16*4 = 64 bytes per row, and
        stride = (full output row width) * 4. Passing element counts here
        stores a quarter of each row, scrambled across the wrong offsets.
        """
        if stride > 0xFFFF:
            raise ValueError(
                f"_pass3_store: stride {stride} exceeds the 16-bit stride field. "
                "A matrix wider than 65535 columns cannot have its row pitch "
                "encoded; the operand must be laid out so the pitch fits "
                "(e.g. lm_head is consumed transposed, pitch=hidden, rather "
                "than as a vocab-wide B operand)."
            )
        return ScheduleNode(
            opcode    = NPUOpcodes.STORE_TILE,
            dep_id    = 0,
            dram_addr = dram_addr & 0xFFFF_FFFF,  # 32-bit
            sram_addr = sram_addr & 0xFFFF,
            dim_m     = dim_m,
            dim_n     = dim_n,
            stride    = stride & 0xFFFF,
        )

    # =========================================================================
    # High-level compile operations (all 4 passes combined)
    # =========================================================================

    def compile_sibling_qkv(
        self,
        dram_x:   int,
        dram_wq:  int,
        dram_wk:  int,
        dram_wv:  int,
        dram_out: int,
        m:        int,
        n:        int,
        k:        int,
    ) -> None:
        """
        Pass 2+3+4: Fused sibling loop for Q, K, V projections.

        Key optimisation (spec §2 Component 2 — Fused Sibling Projections):
          The input activation tile X is loaded once per (r, step) pair and
          reused across all three weight projections, cutting DMA traffic by 3×
          compared to three independent GEMM loops.

        Latency-masked pipelining (spec §2 Component 3):
          Each weight tile load for projection p_idx+1 is issued immediately
          after the COMPUTE_GEMM for p_idx starts, overlapping DMA with compute.

        SRAM slot assignment:
          input_slot(0)   — shared X tile
          weight_slot(0)  — Q weight tile (reused each iteration)
          weight_slot(1)  — K weight tile
          weight_slot(2)  — V weight tile
          output_slot(0)  — Q accumulator
          output_slot(1)  — K accumulator
          output_slot(2)  — V accumulator

        Arguments
        ---------
        dram_x    — DRAM address of the input activation matrix X
        dram_wq/k/v — DRAM addresses of the three weight matrices
        dram_out  — DRAM address of the output buffer (unused here; caller stores)
        m, n, k   — logical matrix dimensions; tiling engine pads to tile boundary
        """
        t = self.tile_dim
        m_tiles, n_tiles, k_tiles = self._pass2_tile_bounds(m, n, k)

        for r in range(m_tiles):
            for c in range(n_tiles):
                for step in range(k_tiles):

                    # ── Step 1: Load shared input activation tile ─────────────
                    # Issued once; all three sibling GEMMs read from input_slot(0).
                    act_offset = self._pass2_dram_stride(r, step, k)
                    dep_x      = self._pass3_next_dep()

                    # stride = X's full row width (K elements, INT8) -- see
                    # _pass3_load's docstring on why this can't be left at 0.
                    self._emit(self._pass3_load(
                        dram_addr = dram_x + act_offset,
                        sram_addr = SRAMMap.input_slot(0),
                        dep_id    = dep_x,
                        stride    = _padded(k),
                    ))

                    # ── Step 2: Sibling loop over W_Q, W_K, W_V ───────────────
                    # Each projection loads its own weight tile into a dedicated
                    # weight slot, waits for the shared X load if needed, then GEMMs.
                    weight_bases = [dram_wq, dram_wk, dram_wv]

                    for p_idx, w_base in enumerate(weight_bases):
                        w_offset = self._pass2_dram_stride(step, c, n)
                        dep_w    = self._pass3_next_dep()

                        # Issue weight tile DMA — overlaps with X load / prior GEMM.
                        # stride = weight matrix's full row width (N elements, INT8).
                        self._emit(self._pass3_load(
                            dram_addr = w_base + w_offset,
                            sram_addr = SRAMMap.weight_slot(p_idx),
                            dep_id    = dep_w,
                            stride    = _padded(n),
                        ))

                        # Barrier: wait for X tile (only once, before first GEMM)
                        if p_idx == 0:
                            self._emit(self._pass3_wait(dep_x))

                        # Barrier: wait for this projection's weight tile
                        self._emit(self._pass3_wait(dep_w))

                        # Parameters: [M:16b][N:16b][K:16b] packed per COMPUTE_GEMM encoding
                        #
                        # b_sram=SRAMMap.weight_slot(p_idx): this was missing
                        # entirely before -- _pass3_gemm's default (WEIGHT_BASE,
                        # i.e. weight_slot(0)) meant every sibling's GEMM read
                        # from Q's weight slot regardless of p_idx, so K and V
                        # silently computed against Q's weights instead of
                        # their own. Each sibling must read the slot its own
                        # LOAD_TILE just filled, two lines above.
                        #
                        # accumulate=(step != 0): same K-accumulation fix as
                        # compile_linear -- without it, only the last K-tile's
                        # partial product survives for any k_tiles > 1.
                        # c_slot=p_idx: each sibling needs its own output slot
                        # (SRAMMap.output_slot(p_idx)) -- without this, all
                        # three GEMMs defaulted to c_slot=0, so K's and V's
                        # results each silently overwrote the previous
                        # sibling's before it was ever read out.
                        self._emit(self._pass3_gemm(
                            sram_addr  = SRAMMap.input_slot(0),
                            b_sram     = SRAMMap.weight_slot(p_idx),
                            dim_m      = t,
                            dim_n      = t,
                            dim_k      = t,
                            dep_id     = 1,
                            activation = NPUActivations.NONE,
                            accumulate = (step != 0),
                            c_slot     = p_idx,
                        ))

    def compile_linear(
        self,
        dram_weight: int,
        m:           int,
        n:           int,
        k:           int,
        activation:      int = NPUActivations.NONE,
        out_dram:        int = 0,
        c_slot:          int = 0,
        a_sram_slot:     int = None,
        dram_activation: int = None,
        a_row_stride:    int = None,
        b_row_stride:    int = None,
        out_row_stride:  int = None,
        scale_shift:     int = 0,
        dram_corr:       int = None,
        dtype:           int = None,
    ) -> None:
        """
        Tiled (M×K)×(K×N) GEMM for a single linear projection.

        True double-buffering: weight tiles alternate between weight_slot(0) and
        weight_slot(1).  The next tile's DMA is issued before WAIT+COMPUTE for
        the current tile so both engines (DMA and PE array) run concurrently.

        Instruction pattern per tile pair (slots 0 and 1):
            LOAD W[i]   → slot(i%2),  dep = (i%2)+1
            LOAD W[i+1] → slot(1-i%2), dep = 2-(i%2)   ← pre-issue next
            WAIT(dep for W[i])
            [LOAD W[i+2] → slot(i%2), dep = (i%2)+1]   ← refill slot i%2
            COMPUTE_GEMM b_sram=slot(i%2)
            WAIT(dep for W[i+1])
            COMPUTE_GEMM b_sram=slot(1-i%2)
            ...

        a_sram_slot — SRAM address of the input activation tile. Defaults to
        SRAMMap.input_slot(0) (the common case: activation freshly loaded from
        DRAM). Callers chaining off a PREVIOUS projection's output (e.g. o_proj
        reading Q's result, down_proj reading gate*up's result) MUST pass the
        actual SRAMMap.output_slot(N) that result landed at -- there is no
        SRAM-to-SRAM copy instruction in the ISA, so compile_linear cannot
        "fetch" its input from anywhere it isn't explicitly told to read.

        c_slot (ISA v1.5) — which of the 16 output slots (SRAMMap.output_slot(c_slot))
        this projection's C accumulator writes to. Defaults to 0. Callers that
        need more than one live projection result at once (e.g. gate_proj and
        up_proj in the same MLP block) MUST use distinct c_slots, or the second
        call's GEMM silently overwrites the first's result before it's read.

        dram_activation — DRAM base of the A matrix. When given, A's K-slice is
        re-loaded from DRAM before every K-step, exactly as the weight tiles
        already are. REQUIRED whenever k_tiles > 1: a K-reduction computes
        C = sum_i A[:, i*16:(i+1)*16] @ B[i*16:(i+1)*16, :], so each step needs
        a different K-slice of BOTH operands. Advancing only B (and leaving A
        pinned at a_sram_slot) computes sum_i A_fixed @ B[i] instead, which is
        mathematically wrong regardless of the accumulate flag being correct.
        Left None for k_tiles == 1, where there is only one slice and A can
        simply be resident at a_sram_slot.

        a_row_stride / b_row_stride / out_row_stride -- row pitch in ELEMENTS for
        each operand, overriding the default (that operand's own width). These
        exist for operating on a column SLICE of a wider matrix without
        copying it out first: multi-head attention slices Q, K and V per head,
        so head h's operand is `base + h*head_dim` with the row pitch of the
        FULL matrix, not of the slice. Without the override the tile offsets
        would be computed as though the slice were a standalone matrix and
        would walk into the wrong rows.

        scale_shift (ISA v1.5) -- right-shift applied when INT8_QUANT narrows
        the INT32 accumulator back to INT8. 0 means "unset", which the
        simulator maps to the v1 default of 7.

        That default is a poor fit for real layers and the mismatch is not
        subtle: a K=1024 projection of Qwen3's embeddings accumulates to
        ~38,000, and >>7 leaves ~299 -- more than double INT8's range, so
        roughly the top third of every activation saturates at 127. The
        correct shift for that layer is 9. Since the accumulator magnitude
        scales with K and with the operand distributions, the shift belongs
        per projection, which is why the ISA carries a field for it rather
        than a constant.

        The output is accumulated at SRAMMap.output_slot(c_slot) and optionally
        stored back to dram_out if out_dram != 0.
        """
        t = self.tile_dim
        dtype = self.dtype if dtype is None else dtype
        w = DType.BYTES[dtype]            # bytes per element
        if a_sram_slot is None:
            # Must match the width, or the load writes to the wide zone while
            # the GEMM reads the narrow one and every result is zero.
            a_sram_slot = SRAMMap.input_slot(0, w)
        a_pitch = (_padded(k) if a_row_stride is None else a_row_stride) * w
        # A projection quantised per output channel registers its correction
        # table against its own DRAM address at ingest, so every existing
        # caller picks it up without a signature change -- compile_linear is
        # reached through compile_attention_block, compile_mlp_block and
        # compile_sibling_qkv, and threading a parameter through all of them
        # would have been three more places to forget. Pass dram_corr=0
        # explicitly to force the per-tensor path.
        if dram_corr is None:
            dram_corr = getattr(self, "_corr_for", {}).get(dram_weight, 0)
        # FP32 carries no scale of its own, so the per-channel correction that
        # exists to reconcile INT8 scales has nothing to do.
        if dtype == DType.FP32:
            dram_corr = 0

        b_pitch = (_padded(n) if b_row_stride is None else b_row_stride) * w
        m_tiles, n_tiles, k_tiles = self._pass2_tile_bounds(m, n, k)

        if k_tiles > 1 and dram_activation is None:
            raise ValueError(
                f"compile_linear: k={k} needs {k_tiles} K-tiles, which requires "
                "dram_activation so the A operand can advance per step. Without "
                "it every step re-reads the same slice of A and the result is "
                "silently wrong (see the ISA v1.5 section of the spec doc)."
            )

        for r in range(m_tiles):
            for c in range(n_tiles):
                if k_tiles == 0:
                    continue

                # ── Pre-issue first two weight tile loads back-to-back ────────
                # Both DMAs start simultaneously; no WAIT yet.
                # stride = weight matrix's full row width (N elements, INT8) --
                # see _pass3_load's docstring on why this can't be left at 0.
                self._emit(self._pass3_load(
                    dram_addr = dram_weight + 0 * t * b_pitch + c * t * w,
                    sram_addr = SRAMMap.weight_slot(0, w),
                    dep_id    = 1,
                    dim_n     = t * w,
                    stride    = b_pitch,
                ))
                if k_tiles > 1:
                    self._emit(self._pass3_load(
                        dram_addr = dram_weight + 1 * t * b_pitch + c * t * w,
                        sram_addr = SRAMMap.weight_slot(1, w),
                        dep_id    = 2,
                        dim_n     = t * w,
                        stride    = b_pitch,
                    ))

                # ── Main K-reduction loop ─────────────────────────────────────
                for step in range(k_tiles):
                    cur_slot = step % 2          # which SRAM slot holds this tile
                    cur_dep  = cur_slot + 1      # dep_id assigned to this slot (1 or 2)
                    nxt_slot = 1 - cur_slot
                    nxt_dep  = nxt_slot + 1

                    # Wait for the current weight tile to arrive in SRAM
                    self._emit(self._pass3_wait(cur_dep))

                    # Advance the A operand to this step's K-slice. Uses a
                    # dedicated dep_id (3) so it can't be confused with the
                    # weight double-buffer's deps (1 and 2). Not double-buffered
                    # itself -- correctness first; overlapping A's load with the
                    # previous step's compute is a later optimisation.
                    if dram_activation is not None:
                        self._emit(self._pass3_load(
                            dram_addr = dram_activation
                                      + r * t * a_pitch + step * t * w,
                            sram_addr = a_sram_slot,
                            dep_id    = 3,
                            dim_n     = t * w,
                            stride    = a_pitch,
                        ))
                        self._emit(self._pass3_wait(3))

                    # Compute with the current tile (b_sram = cur_slot).
                    #
                    # accumulate=(step != 0): the first K-step must zero the
                    # accumulator (v1 behavior); every step after that must
                    # add into it. Without this, each step's COMPUTE_GEMM
                    # would zero C before writing, so only the LAST K-tile's
                    # partial product would survive -- silently wrong for
                    # any k_tiles > 1 (i.e. any real K > 16). See the ISA
                    # v1.5 section of the spec doc for the full writeup.
                    # With a per-channel correction the GEMM must NOT quantise:
                    # PERCHAN_QUANT needs the INT32 accumulator, and applying a
                    # shared shift first would clamp the channels the
                    # correction exists to rescale.
                    final_act = (NPUActivations.NONE if dram_corr else activation)
                    act = final_act if step == k_tiles - 1 else NPUActivations.NONE
                    self._emit(self._pass3_gemm(
                        sram_addr  = a_sram_slot,
                        b_sram     = SRAMMap.weight_slot(cur_slot, w),
                        dim_m      = t,
                        dim_n      = t,
                        dim_k      = t,
                        dep_id     = 1,
                        activation = act,
                        accumulate = (step != 0),
                        c_slot     = c_slot,
                        scale_shift = scale_shift,
                        dtype      = dtype,
                    ))

                    # Refill cur_slot for the tile needed two steps later --
                    # AFTER the GEMM above, never before it. With only two
                    # weight slots, step i uses slot i%2 and step i+1 uses the
                    # other, so slot i%2 is not free until step i's GEMM has
                    # consumed it. Issuing this prefetch before the GEMM (as
                    # this originally did) DMAs over the exact tile the GEMM is
                    # about to read -- a write-after-read hazard on real
                    # hardware, and immediately wrong here. It only stayed
                    # hidden because k_tiles <= 2 never reaches the prefetch.
                    #
                    # LOAD_TILE is non-blocking, so issuing it after the GEMM
                    # still overlaps the DMA with subsequent compute.
                    prefetch_step = step + 2
                    if prefetch_step < k_tiles:
                        self._emit(self._pass3_load(
                            dram_addr = dram_weight
                                      + prefetch_step * t * b_pitch + c * t * w,
                            sram_addr = SRAMMap.weight_slot(cur_slot, w),
                            dep_id    = cur_dep,
                            dim_n     = t * w,
                            stride    = b_pitch,
                        ))

                # ── Per-channel requantisation ───────────────────────────
                # Load this tile's slice of the correction table and narrow the
                # INT32 accumulator to INT8 with one mantissa/shift per column.
                if dram_corr:
                    self._emit(self._pass3_load(
                        dram_addr = dram_corr + c * t * 2,   # uint16 per column
                        sram_addr = SRAMMap.CORR_SCRATCH,
                        dim_m     = 1,
                        dim_n     = t * 2,                   # bytes
                        dep_id    = 4,
                        stride    = t * 2,
                    ))
                    self._emit(self._pass3_wait(4))
                    self.compile_exec_alu(
                        sram_addr      = SRAMMap.output_slot(c_slot),
                        dim_m          = t,
                        dim_n          = t,
                        activation     = NPUActivations.PERCHAN_QUANT,
                        operand_b_sram = SRAMMap.CORR_SCRATCH,
                    )

                if out_dram:
                    # store_tile counts dim_n and stride in BYTES, so the
                    # element width has to follow what actually ended up in
                    # the output slot: INT8_QUANT narrows the INT32
                    # accumulator to INT8 in place on the final K-step, so a
                    # quantized result is 1 byte/element and an unquantized
                    # one is 4. Hardcoding *4 here stored a quantized result
                    # at 4x its real width, which is exactly what a chained
                    # projection (o_proj reading Q, down_proj reading gate*up)
                    # needs to get right. See _pass3_store's docstring.
                    elem = (4 if dtype == DType.FP32
                            else (1 if (dram_corr or
                                        activation == NPUActivations.INT8_QUANT)
                                  else 4))
                    # In ELEMENTS, like a_row_stride/b_row_stride, and scaled
                    # here. Taking this override in bytes while the other two
                    # were in elements meant a caller had to know the output
                    # width -- and got it wrong the moment the width was not 1.
                    out_pitch = (_padded(n) if out_row_stride is None
                                 else out_row_stride) * elem
                    self._emit(self._pass3_store(
                        sram_addr = SRAMMap.output_slot(c_slot),
                        dram_addr = out_dram + r * t * out_pitch + c * t * elem,
                        dim_m     = t,
                        dim_n     = t * elem,
                        stride    = out_pitch,
                    ))

    def compile_exec_alu(
        self,
        sram_addr:  int,
        dim_m:      int,
        dim_n:      int,
        activation: int,
        operand_b_sram: int = 0,
        params:     int = 0,
        dtype:      int = None,
    ) -> None:
        """
        Emit EXEC_ALU for element-wise ops (SiLU, ReLU, ELEMWISE_MUL, …).

        For binary ops (ELEMWISE_MUL), operand_b_sram carries the second tile
        address via the stride field, exactly as the ISA encodes it.
        """
        self._emit(self._pass3_exec_alu(
            sram_addr  = sram_addr,
            dim_m      = dim_m,
            dim_n      = dim_n,
            activation = activation,
            stride     = operand_b_sram,
            params     = params,
            dtype      = self.dtype if dtype is None else dtype,
        ))

    def append_synchronization_barrier(self, dependency_vector_id: int) -> None:
        """
        Insert an explicit WAIT_SYNC fence.

        Called at dependency boundaries — e.g. before o_proj to ensure Q, K, V
        stores have completed, or before down_proj to ensure gate×up is ready.
        """
        self._emit(self._pass3_wait(dependency_vector_id))

    # ── Full layer / model compilation ───────────────────────────────────────

    def compile_residual_add(
        self,
        dram_a:  int,
        dram_b:  int,
        dram_out: int,
        n:       int,
        dtype:   int = None,
        rescale_b: tuple = None,
        b_dtype: int = None,
        widen_shift: int = 0,
        rescale_a: tuple = None,
    ) -> None:
        """
        out = a + b, element-wise over one M-tile of `n` values.

        This is the transformer residual connection (x = x + sublayer(x)).

        `rescale_b` is (mantissa, shift) and is NOT optional in practice. An
        INT8 buffer's value is `code * scale`, and the two operands here reach
        this point with unrelated scales -- at layer 0 of Qwen3 one step of the
        residual is worth 0.00250677 and one step of o_proj's output is worth
        0.000587858, a ratio of 4.26. Adding the raw codes computes
        `a*s_a + b*s_a`, weighting the sublayer output 4.26x too heavily, and
        it happens 56 times over the stack. `rescale_b` converts b into a's
        scale first; use NPUCompiler.encode_ratio(s_b / s_a) to build it. The
        ratio is not a power of two, which is why EXEC_ALU RESCALE exists and
        the scale_shift field could not do this.

        `dtype` selects the element width. The residual stream wants INT16:
        it accumulates across 28 layers and carries the massive activation
        channels, and an INT8 residual measurably costs accuracy.

        Both operands must be resident at once, so they go to two different
        input slots; ELEMWISE_ADD reads operand B's address from the stride
        field, exactly as ELEMWISE_MUL does.
        """
        dtype = self.dtype if dtype is None else dtype
        width = DType.BYTES[dtype]
        # b_dtype defaults to the accumulator width. In the real pipeline the
        # residual stream is INT16 while a sublayer output arrives as INT8, so
        # B is loaded narrow and widened in place before the add -- INT8_QUANT
        # under dtype=INT16 is that widen.
        if b_dtype is None:
            b_dtype = dtype
        w_b = DType.BYTES[b_dtype]
        t = self.tile_dim
        pitch = _padded(n) * width
        for c in range(_n_tiles(n, t)):
            off = c * t * width
            self._emit(self._pass3_load(
                dram_addr = dram_a + off,
                sram_addr = SRAMMap.input_slot(0, width),
                dim_m = t, dim_n = t * width, dep_id = 10, stride = pitch,
            ))
            # B keeps its own element width and row pitch; it is a different
            # buffer, not a slice of A.
            self._emit(self._pass3_load(
                dram_addr = dram_b + c * t * w_b,
                sram_addr = SRAMMap.input_slot(1, width),
                dim_m = t, dim_n = t * w_b, dep_id = 11,
                stride = _padded(n) * w_b,
            ))
            self._emit(self._pass3_wait(10))
            self._emit(self._pass3_wait(11))

            if w_b < width:
                # Widen B in place. Read backwards inside the PE so the wider
                # destination does not clobber source elements still unread.
                self._emit(ScheduleNode(
                    opcode     = NPUOpcodes.EXEC_ALU,
                    dep_id     = 0,
                    dram_addr  = widen_shift & 0xFF,   # left shift during widen
                    sram_addr  = SRAMMap.input_slot(1, width),
                    dim_m      = t,
                    dim_n      = t,
                    activation = NPUActivations.INT8_QUANT,
                    dtype      = dtype,
                ))

            # The residual stream's own scale changes between layers -- the
            # peak grows ~600x across the stack, so one scale for all 28 would
            # round the early layers to zero. A carries its incoming scale, so
            # it is converted too whenever the layer it is entering uses a
            # different one.
            if rescale_a is not None:
                mant_a, shift_a = rescale_a
                self._emit(ScheduleNode(
                    opcode     = NPUOpcodes.EXEC_ALU,
                    dep_id     = 0,
                    dram_addr  = ((shift_a & 0xFF) << 8) | (mant_a & 0xFF),
                    sram_addr  = SRAMMap.input_slot(0, width),
                    dim_m      = t,
                    dim_n      = t,
                    activation = NPUActivations.RESCALE,
                    dtype      = dtype,
                ))

            if rescale_b is not None:
                mant, shift = rescale_b
                self._emit(ScheduleNode(
                    opcode     = NPUOpcodes.EXEC_ALU,
                    dep_id     = 0,
                    dram_addr  = ((shift & 0xFF) << 8) | (mant & 0xFF),
                    sram_addr  = SRAMMap.input_slot(1, width),
                    dim_m      = t,
                    dim_n      = t,
                    activation = NPUActivations.RESCALE,
                    dtype      = dtype,
                ))

            self._emit(ScheduleNode(
                opcode     = NPUOpcodes.EXEC_ALU,
                dep_id     = 0,
                dram_addr  = 0,
                sram_addr  = SRAMMap.input_slot(0, width),
                dim_m      = t,
                dim_n      = t,
                stride     = SRAMMap.input_slot(1, width),
                activation = NPUActivations.ELEMWISE_ADD,
                dtype      = dtype,
            ))
            self._emit(self._pass3_store(
                sram_addr = SRAMMap.input_slot(0, width),
                dram_addr = dram_out + off,
                dim_m = t, dim_n = t * width, stride = pitch,
            ))

    @staticmethod
    def rope_tables(max_pos: int, head_dim: int, base: float = 1e6,
                    dtype: int = 0):
        """
        Host-side RoPE tables: (cos, signed_sin), each max_pos x head_dim INT8.

        theta_i = base ** (-2i / head_dim) for i in [0, head_dim/2), and the
        angle at position p is p * theta_i. Both halves of the vector share a
        row of angles, matching the half-split convention Qwen/Llama use:

            q' = q * cos + rotate_half(q) * sin,
            rotate_half(x1, x2) = (-x2, x1)

        The negation in rotate_half is folded into the SIN table -- the first
        half is stored negated -- so the device never needs a negate or a
        shuffle instruction. That is what keeps RoPE expressible with only
        ELEMWISE_MUL and ELEMWISE_ADD.

        Values are INT8 with 127 standing for 1.0, which is the scale
        ELEMWISE_MUL's (a*b) >> 7 already assumes.
        """
        import numpy as np

        half = head_dim // 2
        inv = base ** (-np.arange(0, half, dtype=np.float64) * 2.0 / head_dim)
        ang = np.outer(np.arange(max_pos, dtype=np.float64), inv)   # max_pos x half

        cos = np.concatenate([np.cos(ang), np.cos(ang)], axis=1)
        sin = np.concatenate([-np.sin(ang), np.sin(ang)], axis=1)   # sign folded in
        if dtype == DType.FP32:
            # Float tables: the rotation is exact, and ELEMWISE_MUL has no
            # >>7 to undo. The Q0.7 form below exists only because INT8
            # needs a fixed-point convention for values in [-1, 1].
            return cos.astype(np.float32), sin.astype(np.float32)
        q = lambda a: np.clip(np.round(a * 127.0), -127, 127).astype(np.int8)
        return q(cos), q(sin)

    # ── Host-side responsibilities ───────────────────────────────────────────
    #
    # Two steps of generation are NOT expressible on this ISA and belong to the
    # host. Naming them as host work is the honest design, not a shortfall:
    #
    #   Embedding lookup is a GATHER -- embed_tokens[token_id] -- and LOAD_TILE
    #   takes a compile-time constant address. There is no indirect or computed
    #   addressing anywhere in the instruction word, so a runtime token id
    #   cannot select a row on-device. The host already builds DRAM, so it
    #   writes the row for the current token into the layer-0 input buffer.
    #
    #   Sampling needs the ARGMAX of the logits. REDUCE_MAX yields the maximum
    #   VALUE, never its index, and nothing carries an index. Beyond argmax,
    #   temperature/top-k/top-p are control flow, which this ISA has none of.
    #   The host reads the logits back and chooses.

    def embedding_row(self, token_id: int, embed_name: str = None):
        """
        INT8 embedding row for `token_id`, for the host to place in DRAM.

        Requires ingest_model(..., capture_weights=True).
        """
        if not getattr(self, "_weight_bytes", None):
            raise RuntimeError("embedding_row: call ingest_model(capture_weights=True) first")
        if embed_name is None:
            embed_name = next((n for n, r in self._weight_map.items()
                               if r.key in EMBEDDING_KEYS), None)
        if embed_name is None:
            raise RuntimeError("embedding_row: no embed_tokens tensor was ingested")
        table = self._weight_bytes[embed_name]
        if not (0 <= token_id < table.shape[0]):
            raise IndexError(f"token_id {token_id} outside vocab {table.shape[0]}")
        return table[token_id]

    def calibrate_scale_shifts(self, sample_x, percentile: float = 99.9):
        """
        Choose a per-projection INT8_QUANT shift from real accumulator
        magnitudes, rather than using the fixed default of 7.

        Requires ingest_model(capture_weights=True). `sample_x` is a batch of
        INT8 activations (rows x hidden) -- embedding rows for a handful of
        real tokens are a reasonable calibration set.

        Why this is needed: the accumulator's magnitude scales with K and with
        the operand distributions, so no single shift fits every projection. A
        K=1024 layer of Qwen3 accumulates to ~38,000, and the default >>7
        leaves ~299 -- past INT8's range, so roughly the top third of every
        activation clips to 127. Clipping, not attenuation, is the failure
        mode, and it compounds across 28 layers.

        The shift is picked so a high percentile of observed accumulators
        lands just inside INT8 rather than the maximum: calibrating to the
        absolute max wastes range on one outlier, which is standard practice
        in INT8 quantisation.

        Returns {weight_name: shift}, clamped to the field's 5-bit range.
        Approximate by construction -- it propagates each projection's output
        forward as the next one's input without modelling the intervening
        norms and activations, which is enough to get the shift within a bit
        or so, and far closer than a constant.
        """
        import numpy as np

        if not getattr(self, "_weight_bytes", None):
            raise RuntimeError("calibrate_scale_shifts: ingest with capture_weights=True")

        x = np.asarray(sample_x, dtype=np.int32)
        shifts, act = {}, x

        def pick(acc):
            mag = float(np.percentile(np.abs(acc), percentile))
            if mag <= 127.0:
                return 0                      # already in range; leave unset
            return int(min(31, max(1, math.ceil(math.log2(mag / 127.0)))))

        by_layer = {}
        for name, rec in self._weight_map.items():
            if rec.key in EMBEDDING_KEYS or rec.key not in PROJECTION_KEYS:
                continue
            by_layer.setdefault(rec.layer_idx, {})[rec.key] = name

        for li in sorted(k for k in by_layer if k >= 0):
            layer, cur = by_layer[li], act
            for key in ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"):
                nm = layer.get(key)
                if nm is None:
                    continue
                w = self._weight_bytes[nm].astype(np.int32)
                src = cur if cur.shape[1] == w.shape[0] else None
                if src is None:      # shape does not chain; probe with full-range input
                    src = np.full((1, w.shape[0]), 96, dtype=np.int32)
                acc = src @ w
                sh = pick(acc)
                shifts[nm] = sh
                out = np.clip(acc >> (sh if sh else 7), -128, 127)
                if key == "down_proj":
                    cur = out
            act = cur
        return shifts

    def compile_lm_head(
        self,
        dram_in:   int,
        dram_w:    int,
        dram_out:  int,
        m:         int,
        hidden:    int,
        vocab:     int,
    ) -> None:
        """
        Vocabulary projection, computed TRANSPOSED: logits^T = W_lm @ x^T.

        The obvious formulation -- logits = x @ W_lm with W_lm as (hidden,
        vocab) -- is not encodable at real vocabulary sizes. That B operand's
        row pitch is `vocab`, and the ISA's stride field is 16 bits: Qwen3's
        151,936 does not fit, and would silently mask to 20,864. Every small
        test passes because a toy vocabulary fits; only the real model breaks,
        and it breaks quietly.

        Transposing fixes it without a wider field. Taking W_lm in its native
        Hugging Face (vocab, hidden) orientation as the A operand gives a row
        pitch of `hidden` (1024), and x^T is a column, so every stride is
        small. The result is logits^T, shape (vocab, m) -- which argmax on the
        host reads just as happily.

        x^T is produced by transposing x padded to a full tile: TRANSPOSE
        works on square tiles, so a 1 x hidden row is padded to 16 x hidden,
        transposed to hidden x 16, and column 0 taken as the operand.

        Output stays INT32: these are logits to be compared, not operands for
        a further GEMM, and requantising would discard exactly the resolution
        argmax needs.
        """
        t = self.tile_dim
        xt = self.alloc_staging(_padded(hidden) * t * DType.BYTES[self.dtype])
        self.compile_transpose(
            dram_in = dram_in, dram_out = xt,
            rows = t, cols = hidden,
            in_stride = _padded(hidden), out_stride = t,
        )
        self.compile_linear(
            dram_weight     = xt,          # B = x^T, a column (pitch = tile)
            m = vocab, n = m, k = hidden,
            out_dram        = dram_out,
            c_slot          = 0,
            dram_activation = dram_w,      # A = W_lm, native (vocab, hidden)
            a_row_stride    = _padded(hidden),
            b_row_stride    = t,
            out_row_stride  = _padded(m),   # elements; scaled by the output width
        )

    def compile_rope(
        self,
        dram_in:    int,
        dram_out:   int,
        dram_cos:   int,
        dram_sin:   int,
        rows:       int,
        head_dim:   int,
        row_stride: int = None,
        pos_offset: int = 0,
    ) -> None:
        """
        Apply RoPE to a rows x head_dim slice: out = in*cos + rotate_half(in)*sin.

        Without this the model has no notion of token order at all -- RoPE is
        the only thing carrying position in this architecture, so its absence
        makes results wrong rather than merely limited.

        rotate_half pairs column j with j + head_dim/2 (or j - head_dim/2 for
        the upper half). When head_dim/2 is a multiple of the tile width, that
        partner is a whole OTHER TILE, so the rotation is just a second tile
        load -- no shuffle instruction needed. Hence the constraint below;
        real head_dims (64, 128, 256) satisfy it.

        Writes to a separate output buffer rather than in place: tile c's
        partner is read after tile c would have been overwritten.

        pos_offset shifts which rows of the tables are used, for decode steps
        where the query sits at absolute position cache_len rather than 0.
        """
        t = self.tile_dim
        if head_dim % (2 * t) != 0:
            raise ValueError(
                f"compile_rope: head_dim={head_dim} must be a multiple of "
                f"{2 * t} so that head_dim/2 is tile-aligned and a column's "
                "rotate_half partner is a whole tile rather than an "
                "intra-tile shuffle."
            )
        ew         = DType.BYTES[self.dtype]
        pitch      = (_padded(head_dim) if row_stride is None else row_stride) * ew
        tbl_pitch  = _padded(head_dim) * ew
        half_tiles = (head_dim // 2) // t
        n_col      = _n_tiles(head_dim, t)

        for r in range(_n_tiles(rows, t)):
            for c in range(n_col):
                partner = c + half_tiles if c < half_tiles else c - half_tiles
                row_off = r * t * pitch
                tbl_off = (pos_offset + r * t) * tbl_pitch

                # Column offsets are BYTE offsets: a tile is t elements wide,
                # so the step between tiles scales with the element width. The
                # rotate_half partner is `half_tiles` tiles away, which is the
                # one that has to move by the same amount -- at FP32 it landed
                # 16 bytes away instead of 64 and the rotation paired each
                # column with the wrong partner.
                for slot, addr, dep in (
                    (0, dram_in  + row_off + c * t * ew,       4),
                    (1, dram_in  + row_off + partner * t * ew, 5),
                ):
                    self._emit(self._pass3_load(
                        dram_addr = addr, sram_addr = SRAMMap.input_slot(slot, ew),
                        dim_m = t, dim_n = t * ew, dep_id = dep, stride = pitch,
                    ))
                for slot, addr, dep in (
                    (2, dram_cos + tbl_off + c * t * ew, 6),
                    (3, dram_sin + tbl_off + c * t * ew, 7),
                ):
                    self._emit(self._pass3_load(
                        dram_addr = addr, sram_addr = SRAMMap.input_slot(slot, ew),
                        dim_m = t, dim_n = t * ew, dep_id = dep, stride = tbl_pitch,
                    ))
                for dep in (4, 5, 6, 7):
                    self._emit(self._pass3_wait(dep))

                # in * cos
                self.compile_exec_alu(
                    sram_addr = SRAMMap.input_slot(0, ew), dim_m = t, dim_n = t,
                    activation = NPUActivations.ELEMWISE_MUL,
                    operand_b_sram = SRAMMap.input_slot(2, ew))
                # rotate_half(in) * sin  -- the negation is already in the table
                self.compile_exec_alu(
                    sram_addr = SRAMMap.input_slot(1, ew), dim_m = t, dim_n = t,
                    activation = NPUActivations.ELEMWISE_MUL,
                    operand_b_sram = SRAMMap.input_slot(3, ew))
                # sum
                self.compile_exec_alu(
                    sram_addr = SRAMMap.input_slot(0, ew), dim_m = t, dim_n = t,
                    activation = NPUActivations.ELEMWISE_ADD,
                    operand_b_sram = SRAMMap.input_slot(1, ew))

                self._emit(self._pass3_store(
                    sram_addr = SRAMMap.input_slot(0, ew),
                    dram_addr = dram_out + row_off + c * t * ew,
                    dim_m = t, dim_n = t * ew, stride = pitch,
                ))

    def compile_softmax_rows(
        self,
        dram_in:     int,
        dram_out:    int,
        m:           int,
        n:           int,
        sram_scalar: int,
        temperature: float = 0.0,
    ) -> None:
        """
        Row-wise softmax over an m x n INT8 score matrix, for ANY n.

        Three passes per row, and they cannot be merged or reordered -- each
        needs the previous one's result over the WHOLE row before it can start:

          1. REDUCE_MAX  accumulating across the row's tiles   -> row max
          2. EXP_SUB against that max, storing the exponentials,
             while REDUCE accumulates their sum               -> row sum
          3. SCALAR_DIV every tile by that sum                -> normalised

        This is the same accumulate-across-tiles shape as compile_rmsnorm_row,
        for the same reason: a reduction whose operands span more tiles than
        one. The earlier version required n <= 16, which capped attention at a
        16-token context.

        Rows are processed one at a time (dim_m = 1) because REDUCE collapses a
        whole tile to a single scalar, while softmax needs a PER-ROW max and
        sum. A 1 x n slice is what makes that expressible without a per-row
        reduction opcode -- at the cost of m times the instructions.

        Two scalars are needed, not one: pass 2 reads the max while
        accumulating the sum, so they cannot share a slot.

        Partial tiles matter here. Where n is not a multiple of the tile width,
        the final tile is issued at its true width rather than a full 16. A
        full-width op would read the row's zero padding, and for REDUCE_MAX
        that is not harmless -- a padding zero becomes the maximum whenever
        every real score in the row is negative, which silently flattens the
        distribution.
        """
        t = self.tile_dim
        tiles = _n_tiles(n, t)
        scalar_max = sram_scalar
        scalar_sum = sram_scalar + 4
        # ew = bytes per element. LOAD/STORE_TILE count dim_n and stride in
        # BYTES, while EXEC_ALU and SCALAR_ALU count dim_n in ELEMENTS -- the
        # distinction that has produced several bugs in this file, so the two
        # are spelled differently here.
        ew = DType.BYTES[self.dtype]
        pitch = _padded(n) * ew

        def width(c):
            return min(t, n - c * t)

        for r in range(m):
            row = r * pitch

            # ── pass 1: row maximum ──────────────────────────────────────────
            for c in range(tiles):
                w = width(c)
                self._emit(self._pass3_load(
                    dram_addr = dram_in + row + c * t * ew,
                    sram_addr = SRAMMap.input_slot(0, ew),
                    dim_m = 1, dim_n = w * ew, dep_id = 14, stride = pitch,
                ))
                self._emit(self._pass3_wait(14))
                self._emit(ScheduleNode(
                    opcode = NPUOpcodes.REDUCE, dep_id = 0, dram_addr = 0,
                    sram_addr = scalar_max & 0xFFFF, dim_m = 1, dim_n = w,
                    stride = SRAMMap.input_slot(0, ew),
                    activation = NPUActivations.REDUCE_MAX, dtype = self.dtype,
                    accumulate = (c != 0),
                ))

            # ── pass 2: exp(x - max), stored; sum accumulated ────────────────
            for c in range(tiles):
                w = width(c)
                self._emit(self._pass3_load(
                    dram_addr = dram_in + row + c * t * ew,
                    sram_addr = SRAMMap.input_slot(0, ew),
                    dim_m = 1, dim_n = w * ew, dep_id = 14, stride = pitch,
                ))
                self._emit(self._pass3_wait(14))
                self._emit(ScheduleNode(
                    opcode = NPUOpcodes.SCALAR_ALU, dep_id = 0,
                    # Q8.8 temperature; 0 keeps the ISA's fixed default.
                    dram_addr = int(round(temperature * 256.0)) & 0xFFFFFFFF,
                    sram_addr = scalar_max & 0xFFFF,
                    stride = SRAMMap.input_slot(0, ew), dim_m = 1, dim_n = w,
                    activation = NPUActivations.EXP_SUB, dtype = self.dtype,
                ))
                self._emit(ScheduleNode(
                    opcode = NPUOpcodes.REDUCE, dep_id = 0, dram_addr = 0,
                    sram_addr = scalar_sum & 0xFFFF, dim_m = 1, dim_n = w,
                    stride = SRAMMap.input_slot(0, ew),
                    activation = NPUActivations.NONE, dtype = self.dtype,
                    accumulate = (c != 0),
                ))
                self._emit(self._pass3_store(
                    sram_addr = SRAMMap.input_slot(0, ew),
                    dram_addr = dram_out + row + c * t * ew,
                    dim_m = 1, dim_n = w * ew, stride = pitch,
                ))

            # ── pass 3: normalise by the row sum ─────────────────────────────
            for c in range(tiles):
                w = width(c)
                self._emit(self._pass3_load(
                    dram_addr = dram_out + row + c * t * ew,
                    sram_addr = SRAMMap.input_slot(0, ew),
                    dim_m = 1, dim_n = w * ew, dep_id = 14, stride = pitch,
                ))
                self._emit(self._pass3_wait(14))
                self._emit(ScheduleNode(
                    opcode = NPUOpcodes.SCALAR_ALU, dep_id = 0, dram_addr = 0,
                    sram_addr = scalar_sum & 0xFFFF,
                    stride = SRAMMap.input_slot(0, ew), dim_m = 1, dim_n = w,
                    activation = NPUActivations.SCALAR_DIV, dtype = self.dtype,
                ))
                self._emit(self._pass3_store(
                    sram_addr = SRAMMap.input_slot(0, ew),
                    dram_addr = dram_out + row + c * t * ew,
                    dim_m = 1, dim_n = w * ew, stride = pitch,
                ))

    def compile_transpose(
        self,
        dram_in:    int,
        dram_out:   int,
        rows:       int,
        cols:       int,
        in_stride:  int = None,
        out_stride: int = None,
    ) -> None:
        """
        Transpose a rows x cols INT8 matrix: dram_out[j, i] = dram_in[i, j].

        A block transpose, not just a tile one: each 16x16 sub-tile is
        transposed in place by EXEC_ALU TRANSPOSE, and the tile POSITIONS are
        swapped too -- tile (tr, tc) of the input becomes tile (tc, tr) of the
        output. That combination is what lets a non-square matrix transpose
        using an instruction that only handles square tiles.

        in_stride/out_stride override the row pitch, in ELEMENTS, for
        transposing a column SLICE of a wider matrix (multi-head attention
        slices K per head). Default to the matrix's own width. They are
        converted to bytes here, so a caller never has to know the width.
        """
        t = self.tile_dim
        w = DType.BYTES[self.dtype]
        in_stride  = (_padded(cols) if in_stride  is None else in_stride) * w
        out_stride = (_padded(rows) if out_stride is None else out_stride) * w

        for tr in range(_n_tiles(rows, t)):
            for tc in range(_n_tiles(cols, t)):
                self._emit(self._pass3_load(
                    dram_addr = dram_in + tr * t * in_stride + tc * t * w,
                    sram_addr = SRAMMap.input_slot(0, w),
                    dim_m = t, dim_n = t * w, dep_id = 15, stride = in_stride,
                ))
                self._emit(self._pass3_wait(15))
                self.compile_exec_alu(
                    sram_addr = SRAMMap.input_slot(0, w), dim_m = t, dim_n = t,
                    activation = NPUActivations.TRANSPOSE,
                )
                self._emit(self._pass3_store(
                    sram_addr = SRAMMap.input_slot(0, w),
                    dram_addr = dram_out + tc * t * out_stride + tr * t * w,
                    dim_m = t, dim_n = t * w, stride = out_stride,
                ))

    def compile_rmsnorm_row(
        self,
        dram_in:     int,
        dram_out:    int,
        n:           int,
        sram_scalar: int,
        norm_eps:    float = 1e-6,
        dram_gamma:  int = 0,
        row_stride:  int = None,
        dtype:       int = None,
        out_peak:    float = 1.0,
    ) -> None:
        """
        RMSNorm over a full row:  y = x / sqrt(mean(x^2)) * 127 * gamma.

        Three defects made the previous version emit an identically zero
        vector on real activations -- and since every projection then
        multiplied zero, the residual carried the embedding untouched through
        all 28 layers and the model reproduced its own input:

          1. It divided by sqrt(SUM(x^2)) rather than sqrt(MEAN(x^2)) --
             off by sqrt(n), a factor of 32 at hidden=1024.
          2. It emitted the raw quotient. RMSNorm's output is ~1.0 by
             construction, and with no fixed-point convention that rounds to
             0 in INT8. Even given an exact sum of squares the peak value was
             0.117. A normalised activation needs an agreed scale; this uses
             Q0.7, so 1.0 is 127.
          3. It never applied the learned gain gamma, despite
             input_layernorm.weight being ingested and resident.

        Both (1) and (2) fold into RMS_SCALE's compile-time constant, since
        x / sqrt(sum/n) * 127 == x * (127 * sqrt(n)) / sqrt(sum).

        The squaring also no longer round-trips through INT8: REDUCE_SUMSQ
        squares inside the reduction, where EXEC_ALU SQUARE used to clamp
        x^2 to 127 and saturate 28% of real elements before summing.

        Two passes -- every tile must be in the sum before any tile is
        scaled by it.

        norm_eps is accepted for interface symmetry and ignored; the
        `scalar > 0` guard in scalar_alu serves its numerical-stability role.
        """
        t = self.tile_dim
        tiles = _n_tiles(n, t)
        # dtype is the INPUT width. RMSNorm is the boundary where the wide
        # residual stream narrows: it reads INT16 and always writes Q0.7 INT8,
        # because its output is normalised by construction and needs no range.
        # The scale cancels -- x/rms(x) is the same whatever one step is worth
        # -- so scale_num is unchanged.
        dtype = self.dtype if dtype is None else dtype
        w_in = DType.BYTES[dtype]
        # Output width. In the INT8 pipeline RMSNorm always writes Q0.7 INT8
        # regardless of what it read -- it is the point where the wide
        # residual narrows. In float there is nothing to narrow to, so the
        # output keeps the input's width.
        w_out = w_in if dtype == DType.FP32 else 1
        # row_stride lets this normalise a head_dim-wide COLUMN SLICE of a
        # wider matrix (Qwen3's per-head q_norm/k_norm run over 128 columns of
        # a 2048-wide Q), where the row pitch is the full matrix's, not the
        # slice's. Defaults to the contiguous case.
        pitch = (_padded(n) if row_stride is None else row_stride) * w_in
        # x / sqrt(sum/n) * 127  ==  x * (127*sqrt(n)) / sqrt(sum)
        #
        # out_peak is the normalised activation's peak in units of its own RMS,
        # and dividing by it is what stops the output clamping. Q0.7 -- mapping
        # 1.0 to 127 -- silently assumes that peak is about 1. Measured on
        # Qwen3 it ranges from 1.18 to 506 across the 56 norms, median 8.67,
        # because RMSNorm does not bound its output: massive activation
        # channels stay massive after normalisation. At the median the output
        # was using 11.5% of the INT8 range and everything above 1 RMS
        # saturated at 127.
        #
        # The output's scale becomes out_peak/127 rather than 1/127, which the
        # consuming projection's correction table has to agree with.
        if dtype == DType.FP32:
            # Float RMS_SCALE takes the row length and forms sqrt(n/sum)
            # itself; there is no output scale to fold in.
            scale_num = n
        else:
            scale_num = max(1, int(round(
                127.0 * math.sqrt(n) / max(out_peak, 1e-6))))

        # In-place (dram_in == dram_out) is safe: pass 1 completes the whole
        # sum before pass 2 begins, and pass 2 reads each tile immediately
        # before rewriting that same tile.

        # ── pass 1: sum of squares over the whole row ────────────────────────
        for c in range(tiles):
            w = min(t, n - c * t)
            self._emit(self._pass3_load(
                dram_addr = dram_in + c * t * w_in,
                sram_addr = SRAMMap.input_slot(1, w_in),
                dim_m = t, dim_n = w * w_in, dep_id = 12, stride = pitch,
            ))
            self._emit(self._pass3_wait(12))
            self._emit(ScheduleNode(
                opcode = NPUOpcodes.REDUCE, dep_id = 0, dram_addr = 0,
                sram_addr = sram_scalar & 0xFFFF, dim_m = t, dim_n = w,
                stride = SRAMMap.input_slot(1, w_in),
                activation = NPUActivations.REDUCE_SUMSQ,
                accumulate = (c != 0),
                dtype = dtype,
            ))

        # ── pass 2: scale, then apply gamma ──────────────────────────────────
        for c in range(tiles):
            w = min(t, n - c * t)
            self._emit(self._pass3_load(
                dram_addr = dram_in + c * t * w_in,
                sram_addr = SRAMMap.input_slot(0, w_in),
                dim_m = t, dim_n = w * w_in, dep_id = 13, stride = pitch,
            ))
            self._emit(self._pass3_wait(13))
            self._emit(ScheduleNode(
                opcode     = NPUOpcodes.SCALAR_ALU,
                dep_id     = 0,
                dram_addr  = scale_num,        # RMS_SCALE's numerator
                sram_addr  = sram_scalar & 0xFFFF,
                stride     = SRAMMap.input_slot(0, w_in),
                dim_m      = t,
                dim_n      = w,
                activation = NPUActivations.RMS_SCALE,
                dtype      = dtype,
            ))
            if dram_gamma:
                # gamma is Q0.7 (1.0 -> 127), broadcast across the tile's rows
                # by the host, so ELEMWISE_MUL's (a*b)>>7 leaves y unchanged
                # where gamma == 1.0.
                self._emit(self._pass3_load(
                    dram_addr = dram_gamma + c * t * w_in,
                    sram_addr = SRAMMap.input_slot(1, w_in),
                    dim_m = t, dim_n = w * w_in, dep_id = 14,
                    stride = _padded(n) * w_in,
                ))
                self._emit(self._pass3_wait(14))
                self.compile_exec_alu(
                    sram_addr      = SRAMMap.input_slot(0, w_in),
                    dim_m = t, dim_n = w,
                    activation     = NPUActivations.ELEMWISE_MUL,
                    operand_b_sram = SRAMMap.input_slot(1, w_in),
                )
            self._emit(self._pass3_store(
                sram_addr = SRAMMap.input_slot(0, w_in),
                dram_addr = dram_out + c * t * w_out,
                dim_m = t, dim_n = w * w_out,
                stride = (_padded(n) if row_stride is None else row_stride) * w_out,
            ))

    def compile_rmsnorm_block(
        self,
        sram_tile:   int,
        sram_scalar: int,
        sram_square: int,
        dim_m:       int = 16,
        dim_n:       int = 16,
        norm_eps:    float = 1e-6,
    ) -> None:
        """
        Emit RMSNorm for one tile: y = x / sqrt(sum(x^2)).

        CONTRACT: sram_tile and sram_square must both already hold the SAME
        input data. The ISA has no SRAM-to-SRAM copy, so the caller loads it
        twice -- sram_square is squared (destroying it) to compute the scale,
        while sram_tile keeps the original values the scale is applied to.

        Sequence:
          1. EXEC_ALU  SQUARE  on sram_square      -- x^2, in place
          2. REDUCE            sram_square -> sram_scalar  -- sum(x^2)
          3. SCALAR_ALU RSQRT  scalar=sram_scalar, tile=sram_tile
                               -- tile[i] = round(tile[i] / sqrt(scalar))

        What this replaces, and why each step went:

        * SQUARE used to run in place on sram_tile itself, so by the time the
          scale was applied the tile held x^2, not x -- it normalised the
          squares instead of the values. Hence the separate sram_square.

        * The scale used to be applied with EXEC_ALU + ELEMWISE_MUL passing
          the scalar's address in the stride field. But EXEC_ALU's
          ELEMWISE_MUL reads operand B as a full dim_m x dim_n tile of INT8,
          so it consumed 256 bytes starting at the 4-byte scalar -- 252 bytes
          of whatever happened to sit after it. SCALAR_ALU is the opcode that
          broadcasts a scalar across a tile.

        * There used to be a SCALAR_ADD step adding epsilon as Q0.15
          (`int(norm_eps * 32768)`). At the default eps=1e-6 that is int(0.033)
          = 0, so it added nothing -- Q0.15's smallest non-zero value is ~3e-5.
          It is dropped rather than rescaled: sum(x^2) here is a non-negative
          integer, so epsilon's numerical-stability role is served by
          scalar_alu's own `scalar > 0` guard, which yields 0 instead of
          dividing by zero. norm_eps is kept in the signature for interface
          stability and is documented as ignored.

        * A separate SCALAR_ALU RSQRT used to convert the scalar to Q0.15
          (x32768) before a multiply that shifted by 7 (/128) -- a 256x scale
          mismatch. scalar_alu's RSQRT applies 1/sqrt directly to the tile in
          one instruction, so no fixed-point round trip happens at all.
        """
        # 1. x^2 into the scratch copy (destroys sram_square, not sram_tile).
        self._emit(self._pass3_exec_alu(
            sram_addr  = sram_square,
            dim_m      = dim_m,
            dim_n      = dim_n,
            activation = NPUActivations.SQUARE,
        ))

        # 2. sum(x^2) -> scalar. Tile source travels in the stride field.
        self._emit(ScheduleNode(
            opcode    = NPUOpcodes.REDUCE,
            dep_id    = 0,
            dram_addr = 0,
            sram_addr = sram_scalar & 0xFFFF,
            dim_m     = dim_m,
            dim_n     = dim_n,
            stride    = sram_square & 0xFFFF,
        ))

        # 3. tile /= sqrt(scalar). For SCALAR_ALU the roles are the reverse of
        # EXEC_ALU's: sram_addr is the SCALAR and stride is the TILE.
        self._emit(ScheduleNode(
            opcode     = NPUOpcodes.SCALAR_ALU,
            dep_id     = 0,
            dram_addr  = 0,
            sram_addr  = sram_scalar & 0xFFFF,
            stride     = sram_tile & 0xFFFF,
            dim_m      = dim_m,
            dim_n      = dim_n,
            activation = NPUActivations.RSQRT,
        ))

    def compile_attention_block(
        self,
        dram_x:   int,
        dram_wq:  int,
        dram_wk:  int,
        dram_wv:  int,
        dram_wo:  int,
        dram_out: int,
        hidden:    int,
        q_dim:     int,
        kv_dim:    int,
        head_dim:  int = None,
        n_heads:   int = 1,
        n_kv_heads: int = 1,
        quantize_out: bool = False,
        seq_len:   int = None,
        cache_len: int = 0,
        causal:    bool = False,
        dram_cos:  int = 0,
        dram_sin:  int = 0,
        shifts:    dict = None,
        dram_k_cache: int = 0,
        dram_v_cache: int = 0,
        dram_q_norm: int = 0,
        dram_k_norm: int = 0,
        softmax_T: float = 0.0,
    ) -> None:
        """
        Emit the attention block's projections.

        quantize_out -- requantize the output projection to INT8. Required
        when the result feeds another stage (the next GEMM takes INT8
        operands); left False when the caller wants the raw INT32
        accumulator, e.g. as a final result.

        Qwen3-0.6B GQA dimensions:
          q_dim  = num_heads    x head_dim = 16 x 128 = 2048
          kv_dim = num_kv_heads x head_dim =  8 x 128 = 1024

        Sequence:
          1. Q = x @ W_q   (H -> q_dim),  requantized to INT8, staged
          2. K = x @ W_k   (H -> kv_dim), requantized to INT8, staged
          3. V = x @ W_v   (H -> kv_dim), requantized to INT8, staged
          4. WAIT_SYNC barrier
          5. out = Q @ W_o (q_dim -> H)

        Full sequence:
          1. Q = x @ W_q,  K = x @ W_k,  V = x @ W_v   (INT8, staged)
          2. K^T           (EXEC_ALU TRANSPOSE)
          3. S = Q @ K^T   (scores)
          4. P = softmax(S) row-wise
          5. O = P @ V
          6. out = O @ W_o

        Steps 2-5 required four ISA additions -- TRANSPOSE, REDUCE_MAX,
        EXP_SUB and SCALAR_DIV -- because COMPUTE_GEMM computes A @ B (no
        transpose), REDUCE collapses a tile to one scalar (no per-row max),
        and nothing exponentiated or divided. See compile_softmax_rows.

        Multi-head with GQA: the loop runs once per query head over a
        head_dim-wide slice of Q, and head h reads K/V head
        h // (n_heads // n_kv_heads) -- so with n_heads=16 and n_kv_heads=8
        each K/V head serves two query heads, which is what "grouped-query"
        means. Each head's context is written into its own head_dim-wide slice
        of the context buffer, so the concatenation across heads is implicit
        in the addressing rather than a separate step.

        Heads are SLICES of the staged Q/K/V, never copied out: the per-head
        operands are addressed with the FULL matrix's row pitch via
        compile_linear's a_row_stride / b_row_stride / out_row_stride.

        KV cache (autoregressive decode). Pass seq_len=1 with a non-zero
        cache_len and dram_k_cache/dram_v_cache to attend one new query
        position over cache_len previously-computed key/value positions. The
        freshly-projected K/V for this step are appended to the cache at
        offset cache_len before the attention runs, so the effective key
        length is cache_len + seq_len.

        This is the shape generation actually needs: one query against a
        growing cache, giving scores of 1 x (cache_len+1) -- a single wide
        row, which is exactly what compile_softmax_rows was generalised to
        handle. Note that causal masking is unnecessary in this mode: the
        cache only ever holds positions at or before the current one, so
        there is nothing to mask out.

        causal=True applies a causal mask, so query position i cannot attend
        to key position j > i. Needed for parallel prefill over several
        positions at once; unnecessary (and left off) in cache_len>0 decode,
        where the cache holds only positions at or before the current one.

        Masking is applied per score TILE, which makes most of it free: for
        tile (ti, tj), tiles with tj > ti are entirely masked and are simply
        NOT COMPUTED -- skipping the GEMM outright rather than computing and
        then discarding it -- while tiles below the diagonal are fully
        visible. Only the diagonal tile needs the triangular CAUSAL_MASK.

        What this replaces: the previous version computed Q with one
        compile_linear call and then called compile_sibling_qkv with
        dram_wq=dram_wk, dram_wk=dram_wv, dram_wv=dram_wv -- computing V
        twice, discarding K, and (because every projection then landed in the
        same output slot) overwriting Q before o_proj could read it. Its own
        comment flagged the V argument as a placeholder.

        Q/K/V share the input X, so they are a genuine fusion opportunity
        (load X once, three GEMMs against it). compile_sibling_qkv does that,
        but takes a single N for all three siblings, and here Q is q_dim wide
        while K and V are kv_dim -- so it does not fit without generalising
        that function to a variable sibling count. Left as three independent
        projections: correct first, fused later.
        """
        t = self.tile_dim
        # ew = bytes per element. Every DRAM address and row pitch below is a
        # BYTE quantity derived from an element count; the multiplications are
        # explicit rather than implicit-by-1 so the FP32 path works.
        ew = DType.BYTES[self.dtype]
        q_stage  = self.alloc_staging(t * _padded(q_dim) * ew)
        k_stage  = self.alloc_staging(t * _padded(kv_dim) * ew)
        v_stage  = self.alloc_staging(t * _padded(kv_dim) * ew)
        self.attn_kv_stage = (k_stage, v_stage)   # exposed for tests/debugging

        shifts = shifts or {}
        for weight, out_stage, n_dim, slot, key in (
            (dram_wq, q_stage, q_dim,  0, "q_proj"),
            (dram_wk, k_stage, kv_dim, 1, "k_proj"),
            (dram_wv, v_stage, kv_dim, 2, "v_proj"),
        ):
            self.compile_linear(
                dram_weight     = weight,
                m = t, n = n_dim, k = hidden,
                activation      = NPUActivations.INT8_QUANT,
                out_dram        = out_stage,
                c_slot          = slot,
                dram_activation = dram_x,
                scale_shift     = shifts.get(key, 0),
            )

        # Barrier before the attention math consumes Q/K/V.
        self.append_synchronization_barrier(1)

        # ── Attention: O = softmax(Q @ K^T) @ V, per head ────────────────────
        sq       = t if seq_len is None else seq_len       # query positions
        group    = max(1, n_heads // max(1, n_kv_heads))   # Q heads per KV head
        use_cache = cache_len > 0 or dram_k_cache != 0
        keys     = cache_len + sq                          # total key positions

        head_dim = (q_dim // n_heads) if head_dim is None else head_dim

        # ── Per-head Q/K RMSNorm (Qwen3 q_norm / k_norm) ─────────────────────
        # Runs on each head's head_dim-wide slice, in place, BEFORE RoPE --
        # the order the architecture specifies. Omitting this entirely was
        # enough to reduce the model to noise even in full float precision
        # (see ref_model.py): Q and K are dot-producted against each other, so
        # an unnormalised head rescales every score in its attention row, and
        # the softmax that follows is not scale-invariant.
        #
        # Normalising a head means normalising a column slice of a wider
        # matrix, which is what row_stride is for. In-place is safe -- see
        # compile_rmsnorm_row.
        for dram_g, stage, count, width in (
            (dram_q_norm, q_stage, n_heads,    q_dim),
            (dram_k_norm, k_stage, n_kv_heads, kv_dim),
        ):
            if not dram_g:
                continue
            for h in range(count):
                self.compile_rmsnorm_row(
                    dram_in     = stage + h * head_dim * ew,
                    dram_out    = stage + h * head_dim * ew,
                    n           = head_dim,
                    sram_scalar = SRAMMap.SCALAR_SCRATCH_BASE,
                    dram_gamma  = dram_g,
                    row_stride  = _padded(width),
                )

        # ── RoPE on Q and K (never V) ────────────────────────────────────────
        # Applied per head, before the cache append, so cached keys are stored
        # already-rotated -- earlier positions must not be re-rotated on later
        # steps. Query position is cache_len + row, hence the offset.
        if dram_cos and dram_sin:
            # Row count is _padded(sq), not sq. A tile is always 16 rows
            # tall, so a buffer sized for the logical row count is overrun
            # whenever sq < 16 -- the store writes 16 rows into space for one
            # and lands in the next staging buffer. At sq == 16 the two agree,
            # which is why every test passed and only decode (sq == 1) broke.
            q_rot = self.alloc_staging(_padded(sq) * _padded(q_dim) * ew)
            k_rot = self.alloc_staging(_padded(sq) * _padded(kv_dim) * ew)
            for h in range(n_heads):
                self.compile_rope(
                    dram_in = q_stage + h * head_dim * ew,
                    dram_out = q_rot + h * head_dim * ew,
                    dram_cos = dram_cos, dram_sin = dram_sin,
                    rows = sq, head_dim = head_dim,
                    row_stride = _padded(q_dim), pos_offset = cache_len,
                )
            for kh in range(n_kv_heads):
                self.compile_rope(
                    dram_in = k_stage + kh * head_dim * ew,
                    dram_out = k_rot + kh * head_dim * ew,
                    dram_cos = dram_cos, dram_sin = dram_sin,
                    rows = sq, head_dim = head_dim,
                    row_stride = _padded(kv_dim), pos_offset = cache_len,
                )
            q_stage, k_stage = q_rot, k_rot

        # With a cache, K/V for this step are appended to it and attention
        # reads the whole cache; without one they are read straight from the
        # freshly-projected staging buffers.
        if use_cache:
            k_src, v_src   = dram_k_cache, dram_v_cache
            k_pitch        = _padded(kv_dim)   # elements
            # Append this step's K/V at row `cache_len` of each cache.
            for src, dst in ((k_stage, dram_k_cache), (v_stage, dram_v_cache)):
                for c in range(_n_tiles(kv_dim, t)):
                    self._emit(self._pass3_load(
                        dram_addr = src + c * t * ew,
                        sram_addr = SRAMMap.input_slot(0, ew),
                        dim_m = sq, dim_n = t * ew, dep_id = 9,
                        stride = _padded(kv_dim) * ew,
                    ))
                    self._emit(self._pass3_wait(9))
                    self._emit(self._pass3_store(
                        sram_addr = SRAMMap.input_slot(0, ew),
                        # cache_len indexes POSITIONS; the row pitch is a byte
                        # quantity, so both factors scale with the width.
                        dram_addr = dst + (cache_len * _padded(kv_dim) + c * t) * ew,
                        dim_m = sq, dim_n = t * ew, stride = _padded(kv_dim) * ew,
                    ))
        else:
            k_src, v_src = k_stage, v_stage
            k_pitch      = _padded(kv_dim)     # elements

        kt_stage     = self.alloc_staging(_padded(head_dim) * _padded(keys) * ew)
        scores_stage = self.alloc_staging(_padded(sq) * _padded(keys) * ew)
        probs_stage  = self.alloc_staging(_padded(sq) * _padded(keys) * ew)
        ctx_stage    = self.alloc_staging(_padded(sq) * _padded(q_dim) * ew)

        for h in range(n_heads):
            kv_h   = h // group          # GQA: several Q heads share one KV head
            q_off  = h * head_dim * ew   # byte offset of this head within a Q row
            kv_off = kv_h * head_dim * ew  # ... and within a K/V row

            # K_h^T -- transpose the keys x head_dim slice into head_dim x keys.
            self.compile_transpose(
                dram_in  = k_src + kv_off, dram_out = kt_stage,
                rows = keys, cols = head_dim,
                in_stride = k_pitch, out_stride = _padded(keys),
            )

            # scores = Q_h @ K_h^T  ->  sq x keys
            self.compile_linear(
                dram_weight     = kt_stage,
                m = sq, n = keys, k = head_dim,
                activation      = NPUActivations.INT8_QUANT,
                out_dram        = scores_stage,
                c_slot          = 3,
                dram_activation = q_stage + q_off,
                a_row_stride    = _padded(q_dim),   # Q_h is a slice of Q
                b_row_stride    = _padded(keys),
            )

            # Causal mask. Query row i sits at absolute position
            # cache_len + i, so with a cache the diagonal is offset: tile
            # (ti, tj) is fully visible when the whole tile lies at or below
            # it, fully masked when it lies entirely above.
            if causal:
                for ti in range(_n_tiles(sq, t)):
                    q_lo = cache_len + ti * t          # first query position
                    for tj in range(_n_tiles(keys, t)):
                        k_lo = tj * t                  # first key position
                        if k_lo > q_lo + t - 1:
                            continue                   # fully masked; skipped above
                        if k_lo + t - 1 <= q_lo:
                            continue                   # fully visible
                        self._emit(self._pass3_load(
                            dram_addr = scores_stage + ti * t * _padded(keys) + tj * t,
                            sram_addr = SRAMMap.input_slot(0),
                            dim_m = t, dim_n = t, dep_id = 9,
                            stride = _padded(keys),
                        ))
                        self._emit(self._pass3_wait(9))
                        self.compile_exec_alu(
                            sram_addr = SRAMMap.input_slot(0, ew), dim_m = t, dim_n = t,
                            activation = NPUActivations.CAUSAL_MASK,
                        )
                        self._emit(self._pass3_store(
                            sram_addr = SRAMMap.input_slot(0, ew),
                            dram_addr = scores_stage
                                      + (ti * t * _padded(keys) + tj * t) * ew,
                            dim_m = t, dim_n = t * ew,
                            stride = _padded(keys) * ew,
                        ))

            # probs = softmax(scores), row-wise over the full key length
            self.compile_softmax_rows(
                dram_in = scores_stage, dram_out = probs_stage,
                m = sq, n = keys, sram_scalar = SRAMMap.SCALAR_SCRATCH_BASE,
                temperature = softmax_T,
            )

            # ctx_h = probs @ V_h  ->  sq x head_dim, into this head's slice
            self.compile_linear(
                dram_weight     = v_src + kv_off,
                m = sq, n = head_dim, k = keys,
                activation      = NPUActivations.INT8_QUANT,
                out_dram        = ctx_stage + q_off,
                c_slot          = 4,
                dram_activation = probs_stage,
                a_row_stride    = _padded(keys),
                b_row_stride    = k_pitch,          # V_h is a slice of V
                out_row_stride  = _padded(q_dim),   # ctx_h is a slice of ctx
            )

        self.append_synchronization_barrier(2)

        # Output projection consumes the concatenated per-head context.
        self.compile_linear(
            dram_weight     = dram_wo,
            m = (t if seq_len is None else seq_len), n = hidden, k = q_dim,
            activation      = (NPUActivations.INT8_QUANT if quantize_out
                               else NPUActivations.NONE),
            out_dram        = dram_out,
            c_slot          = 0,
            dram_activation = ctx_stage,
            scale_shift     = shifts.get("o_proj", 0),
        )

    def compile_mlp_block(
        self,
        dram_x:    int,
        dram_gate: int,
        dram_up:   int,
        dram_down: int,
        dram_out:  int,
        hidden:    int,
        inter:     int,
        quantize_out: bool = False,
        shifts:    dict = None,
        dram_silu_lut: int = 0,
        mul_shift: int = 0,
    ) -> None:
        """
        Emit the SwiGLU MLP block:

          gate = x @ W_gate            (H -> I)
          up   = x @ W_up              (H -> I)
          act  = SiLU(gate) * up       (element-wise, I wide)
          out  = act @ W_down          (I -> H)

        Two things force this to be a staged, tile-wise loop rather than the
        four straight-line calls it used to be:

        1. dtype. The GEMMs produce INT32 accumulators, but EXEC_ALU's SILU
           and ELEMWISE_MUL operate on INT8 tiles (PEArray reads them as
           int8_t*). gate and up must therefore be requantized (INT8_QUANT)
           before the element-wise stage, and the down projection needs an
           INT8 activation anyway.

        2. size. gate and up are each m x I -- at Qwen3's I=3072 that is 192
           N-tiles, far past what fits in SRAM at once, and the element-wise
           ops process one 16x16 tile per instruction. So gate and up are
           staged to DRAM, combined a tile at a time, and the combined result
           staged again for the down projection to stream back in.

        The previous version applied SiLU to output_slot(0) after up_proj had
        already overwritten gate's result there, multiplied by an
        output_slot(1) nothing had ever written, and processed only the first
        of I/16 tiles.
        """
        t = self.tile_dim
        shifts = shifts or {}
        w = DType.BYTES[self.dtype]
        stage_bytes = t * _padded(inter) * w
        gate_stage = self.alloc_staging(stage_bytes)
        up_stage   = self.alloc_staging(stage_bytes)
        act_stage  = self.alloc_staging(stage_bytes)

        # gate/up requantize to INT8 so the element-wise stage below can read
        # them at that width -- but that's only true in the INT8 pipeline.
        # FP32 has no such boundary: SILU and ELEMWISE_MUL already operate on
        # the native width (see pe_array.cpp's FP32 branches), so quantizing
        # here would silently truncate real float accumulators into int8
        # codes before anything asked for INT8. NONE leaves the GEMM's raw
        # accumulator in place, which compile_linear already stores at its
        # correct width either way (elem=4 for FP32, regardless of activation).
        proj_act = (NPUActivations.NONE if self.dtype == DType.FP32
                    else NPUActivations.INT8_QUANT)

        # ── 1. gate = x @ W_gate, requantized to INT8, staged to DRAM ────────
        self.compile_linear(
            dram_weight     = dram_gate,
            m = t, n = inter, k = hidden,
            activation      = proj_act,
            out_dram        = gate_stage,
            c_slot          = 0,
            dram_activation = dram_x,
            scale_shift     = shifts.get("gate_proj", 0),
        )

        # ── 2. up = x @ W_up, likewise ───────────────────────────────────────
        self.compile_linear(
            dram_weight     = dram_up,
            m = t, n = inter, k = hidden,
            activation      = proj_act,
            out_dram        = up_stage,
            c_slot          = 1,
            dram_activation = dram_x,
            scale_shift     = shifts.get("up_proj", 0),
        )

        # ── 3. act = SiLU(gate) * up, one tile at a time ─────────────────────
        # Both operands must be resident simultaneously, so they go to two
        # different input slots; ELEMWISE_MUL carries operand B's address in
        # the stride field.
        for c in range(_n_tiles(inter, t)):
            off = c * t * w
            self._emit(self._pass3_load(
                dram_addr = gate_stage + off,
                sram_addr = SRAMMap.input_slot(0, w),
                dim_m = t, dim_n = t * w, dep_id = 4, stride = _padded(inter) * w,
            ))
            self._emit(self._pass3_load(
                dram_addr = up_stage + off,
                sram_addr = SRAMMap.input_slot(1, w),
                dim_m = t, dim_n = t * w, dep_id = 5, stride = _padded(inter) * w,
            ))
            self._emit(self._pass3_wait(4))
            self._emit(self._pass3_wait(5))

            if dram_silu_lut:
                # A table built for the gate's calibrated scale, so SiLU is
                # evaluated at the TRUE value. EXEC_ALU SILU applies
                # x/(1+exp(-x)) to the raw INT8 code, which at integer
                # granularity equals ReLU for all 256 codes -- the SwiGLU gate
                # degenerates into a ReLU gate.
                self.compile_lut(
                    sram_addr = SRAMMap.input_slot(0, w),
                    dim_m = t, dim_n = t,
                    dram_table = dram_silu_lut,
                )
            else:
                self.compile_exec_alu(
                    sram_addr  = SRAMMap.input_slot(0, w),
                    dim_m = t, dim_n = t,
                    activation = NPUActivations.SILU,
                )
            self.compile_exec_alu(
                sram_addr      = SRAMMap.input_slot(0, w),
                dim_m = t, dim_n = t,
                activation     = NPUActivations.ELEMWISE_MUL,
                operand_b_sram = SRAMMap.input_slot(1, w),
                params         = mul_shift & 0xFF,
            )
            self._emit(self._pass3_store(
                sram_addr = SRAMMap.input_slot(0, w),
                dram_addr = act_stage + off,
                dim_m = t, dim_n = t * w, stride = _padded(inter) * w,
            ))

        # ── 4. out = act @ W_down ────────────────────────────────────────────
        # quantize_out: required when this feeds the next layer, whose GEMMs
        # take INT8 operands. Without it the output is INT32 and the next
        # stage's LOAD_TILE reads it at the wrong element width.
        self.compile_linear(
            dram_weight     = dram_down,
            m = t, n = hidden, k = inter,
            activation      = (NPUActivations.INT8_QUANT if quantize_out
                               else NPUActivations.NONE),
            out_dram        = dram_out,
            c_slot          = 0,
            dram_activation = act_stage,
            scale_shift     = shifts.get("down_proj", 0),
        )

    def retarget_corr(self, name: str, in_scale: float, out_scale: float) -> bool:
        """
        Rebuild a projection's per-channel correction so its output lands in
        `out_scale`, overwriting the table placed at ingest.

        Column c's accumulator is in units of in_scale * peak[c]/127, so the
        multiplier that lands it in out_scale is in_scale*peak[c]/(127*out_scale).
        Without this the output scale is whatever normalising to the widest
        column produced, and every RESCALE constant computed against the
        calibrated scale is wrong by that unknown factor.
        """
        import numpy as np
        peaks = getattr(self, "_weight_peaks", {}).get(name)
        addr = getattr(self, "_corr_addr", {}).get(name)
        if peaks is None or addr is None or not (out_scale > 0):
            return False
        rel = (float(in_scale) * np.asarray(peaks, dtype=np.float64)) \
              / (127.0 * float(out_scale))
        rel = np.where(rel > 0, rel, 1e-12)
        e = np.clip(np.ceil(np.log2(128.0 / rel)), 0, 255)
        mant = np.clip(np.round(rel * (2.0 ** e)), 1, 255)
        tab = np.zeros(_padded(len(rel)), dtype=np.uint16)
        tab[:len(rel)] = ((e.astype(np.uint16) << 8) | mant.astype(np.uint16))
        tab[len(rel):] = (255 << 8) | 128
        self._const_regions[addr] = tab.astype("<u2").tobytes()
        return True

    def plan_scales(self, calib: dict, resid_bits: int = 16):
        """
        Turn calibration magnitudes into the constants the instruction stream
        carries: one residual scale per layer, and the (mantissa, shift) pairs
        that bring each sublayer output into it.

        One scale PER LAYER rather than one for the model: the residual peak
        grows ~600x across 28 layers (0.07 at layer 0, 291 at layer 27), so a
        single INT8-or-INT16 scale large enough for the last layer rounds the
        first few to zero. Each layer therefore gets its own, and the boundary
        between them carries a conversion of the stream itself -- which is
        what `rescale_a` on the first add of each layer is for.

        `widen_shift` is fixed at 8. An INT8 sublayer output is widened into
        the INT16 stream anyway, and shifting left by 8 during that widen
        absorbs a factor of 256 -- without it the leftover ratio reaches 348,
        past what RESCALE's 8-bit mantissa can express.
        """
        import numpy as np

        def _s(peak):
            return (peak / 127.0) if peak > 0 else 1.0

        lim_r = (1 << (resid_bits - 1)) - 1
        layers = calib["layers"]
        plan = {"widen_shift": 8, "resid_bits": resid_bits, "layers": []}

        s_res = []
        for L in layers:
            peak = max(L["x_in"], L["x_attn"], L["x_out"])
            s_res.append((peak / lim_r) if peak > 0 else 1.0)

        # The embedding is written straight into layer 0's residual buffer, so
        # it must be quantised at that layer's scale, not one of its own.
        plan["embed_scale"] = s_res[0]

        for i, L in enumerate(layers):
            s_x = s_res[i]
            s_o = (L["o"] / 127.0) if L["o"] > 0 else 1.0
            s_d = (L["d"] / 127.0) if L["d"] > 0 else 1.0
            prev = s_res[i - 1] if i > 0 else s_res[0]
            # Intermediate buffers produced by FIXED-CONVENTION ops do not get
            # to have a calibrated scale: their scale is whatever the
            # convention yields, and the compiler must use that number rather
            # than the one calibration measured.
            #
            #   ELEMWISE_MUL is (a*b)>>7, so the SwiGLU product lands at
            #   s_silu * s_up * 128 -- which corresponds to the product of the
            #   two PEAKS, not the peak of the product. Those differ whenever
            #   the peaks do not coincide: measured at layer 0, 5.265 against
            #   2.803, so down_proj was being fed a buffer 1.89x off.
            #
            #   ctx = probs @ V then INT8_QUANT >>7, with probs Q0.7, so it
            #   lands at (1/127) * s_v * 128.
            s_gate = _s(L["gate"])
            s_up   = _s(L["up"])
            # The LUT's own output scale: lut_table sizes it by the table's
            # peak, which is silu evaluated at the largest representable code.
            _t = np.arange(-128, 128, dtype=np.float64) * s_gate
            s_silu = float(np.abs(_t / (1.0 + np.exp(-_t))).max()) / 127.0
            # ELEMWISE_MUL is (a*b)>>k. Choose k so the calibrated product
            # peak lands near 127 instead of assuming both operands fill the
            # INT8 range, which a fixed k=7 does.
            prod_peak = L["act"] / max(s_silu * s_up, 1e-30)
            mul_shift = int(max(0, min(31, math.ceil(
                math.log2(max(prod_peak, 1.0) / 127.0)))))
            s_act = s_silu * s_up * (2.0 ** mul_shift)
            s_ctx = (1.0 / 127.0) * _s(L["v"]) * 128.0

            plan["layers"].append({
                "s_res":  s_x,
                "s_o":    s_o,
                "s_d":    s_d,
                # first add of the layer also converts the incoming stream
                "rescale_a": (None if abs(prev - s_x) < 1e-30
                              else self.encode_ratio(prev / s_x)),
                "rescale_o": self.encode_ratio((s_o / s_x) / 256.0),
                "rescale_d": self.encode_ratio((s_d / s_x) / 256.0),
                # Peak of each normalised activation in units of its own RMS.
                "y_peak":  max(L["y"], 1e-6),
                "y2_peak": max(L["y2"], 1e-6),
                "s_gate": s_gate,
                "s_act":  s_act,
                "mul_shift": mul_shift,
                # exp() must see the TRUE score difference, and a score code is
                # worth s_scores, so T = 1/s_scores. The ISA's fixed T=64 is
                # right only if one score step happens to be 1/64.
                "softmax_T": 1.0 / max(_s(L["scores"]), 1e-30),
                "s_ctx":  s_ctx,
                # Input and output scales for every projection, so its
                # correction table can be retargeted. The inputs of q/k/v and
                # gate/up are RMSNorm outputs, which are Q0.7 by construction
                # (1.0 -> 127) whatever their input was; o_proj and down_proj
                # take calibrated buffers.
                "proj": {
                    "q_proj":    (L["y"] / 127.0, _s(L["q"])),
                    "k_proj":    (L["y"] / 127.0, _s(L["k"])),
                    "v_proj":    (L["y"] / 127.0, _s(L["v"])),
                    "o_proj":    (s_ctx, s_o),
                    "gate_proj": (L["y2"] / 127.0, _s(L["gate"])),
                    "up_proj":   (L["y2"] / 127.0, _s(L["up"])),
                    "down_proj": (s_act, s_d),
                },
            })
        return plan

    def compile_model(self, model, capture_weights: bool = False,
                      seq_len: int = None, causal: bool = True,
                      cache_len: int = 0, max_cache: int = 0,
                      scale_shifts: dict = None,
                      per_channel: bool = True,
                      scale_plan: dict = None) -> None:
        """
        Full model compilation pipeline for Qwen3-0.6B (Qwen3ForCausalLM).

        Requires a PyTorch / Hugging Face model instance.
        Calls ingest_model() then emits, for each of the 28 transformer layers:
          1. Pre-attention RMSNorm  (input_layernorm)
          2. Attention block        (Q/K/V + o_proj, GQA 16→8 heads)
          3. Post-attention RMSNorm (post_attention_layernorm)
          4. MLP block              (gate_proj / up_proj / down_proj, SwiGLU)

        RoPE rotation matrices are pre-computed on the host CPU and loaded via
        LOAD_TILE before each attention block; this compiler does not emit the
        rotation matrices themselves.

        Example:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
            compiler = NPUCompiler("toolchain/hal_config.json")
            compiler.compile_model(model)
            compiler.export_binary("out/qwen3_0.6b.bin")
        """
        weights  = self.ingest_model(model, capture_weights=capture_weights,
                                     per_channel=per_channel)
        cfg      = self.config.get("model", {})
        hidden   = cfg.get("hidden_size",       1024)
        inter    = cfg.get("intermediate_size", 3072)
        head_dim = cfg.get("head_dim",          128)
        q_dim    = cfg.get("num_heads",         16) * head_dim   # 2048
        kv_dim   = cfg.get("num_kv_heads",      8)  * head_dim   # 1024
        n_layers = cfg.get("num_layers",        28)
        norm_eps = cfg.get("rms_norm_eps",      1e-6)

        # Staging region sits just past the weights — see init_staging().
        self.init_staging()
        self.layer_buffers = {}

        # Scratch SRAM for the RMSNorm scalar intermediary. Sits above every
        # c_slot-addressable output slot; see SRAMMap.SCALAR_SCRATCH_BASE.
        SCALAR_SCRATCH = SRAMMap.SCALAR_SCRATCH_BASE

        # Group weight records by layer index and key
        by_layer: Dict[int, Dict[str, WeightRecord]] = {}
        for rec in weights.values():
            by_layer.setdefault(rec.layer_idx, {})[rec.key] = rec

        # Layer activations live in DRAM and are handed from one layer to the
        # next. Allocated OUTSIDE the per-layer staging rewind, since the
        # buffer produced by layer N is the input to layer N+1 -- the one
        # thing that IS live across a layer boundary.
        # With a scale plan the residual stream is INT16: it accumulates over
        # 28 layers and carries the massive activation channels, and an INT8
        # residual measurably costs accuracy. Sublayer outputs stay INT8 and
        # are widened into it at each add.
        w = DType.BYTES[self.dtype]
        # At FP32 there is one width everywhere -- no narrow sublayer output
        # widened into a wider stream, because there is nothing to widen.
        res_w = w if self.dtype != DType.INT8 else (2 if scale_plan else 1)
        act_bytes = self.tile_dim * _padded(hidden) * w
        res_bytes = self.tile_dim * _padded(hidden) * res_w
        layer_io = [self.alloc_staging(res_bytes), self.alloc_staging(res_bytes)]
        self.model_input_dram = layer_io[0]

        # Per-layer KV caches. Allocated outside the rewind and at a FIXED
        # address across steps: the caller rebuilds the DRAM image each step
        # but leaves these regions intact, so previously-computed keys and
        # values survive. max_cache sizes them for the longest context the
        # caller intends to reach.
        self.kv_cache_dram = []
        if max_cache:
            for _ in range(n_layers):
                kb = max_cache * _padded(kv_dim) * w
                self.kv_cache_dram.append(
                    (self.alloc_staging(kb), self.alloc_staging(kb)))

        # RoPE tables. Allocated outside the per-layer rewind because they are
        # read by every layer, and registered as const regions so
        # export_dram_image places them alongside the weights -- they are
        # computed on the host, not ingested, so they have no WeightRecord.
        sq = self.tile_dim if seq_len is None else seq_len
        cos_tbl, sin_tbl = self.rope_tables(sq, head_dim,
                                            cfg.get("rope_theta", 1e6),
                                            dtype = self.dtype)
        cos_dram = self.alloc_staging(cos_tbl.shape[0] * _padded(head_dim) * w)
        sin_dram = self.alloc_staging(sin_tbl.shape[0] * _padded(head_dim) * w)
        self._const_regions[cos_dram] = _pad_rows(cos_tbl).tobytes()
        self._const_regions[sin_dram] = _pad_rows(sin_tbl).tobytes()

        # Per-layer scratch starts HERE -- after every buffer that outlives a
        # layer. Capturing the base earlier (right after layer_io) put it at
        # the KV caches and RoPE tables, so reset_staging() handed layer N+1's
        # attn_out_dram the address of the cos table: layer 0's attention
        # output overwrote it, and every subsequent layer applied RoPE with
        # its own stale activations as the rotation. Layer 0 still read an
        # intact table, so a single-layer model was bit-exact and only a
        # 2+-layer model diverged -- which is what made this look like float
        # drift rather than a collision.
        # Broadcast q_norm/k_norm gamma tiles. Allocated HERE, before the
        # staging base is captured, because they are read by every layer and
        # so must outlive the per-layer rewind -- gamma_tile_dram calls
        # alloc_staging, and calling it lazily from inside the layer loop
        # would place each tile in per-layer scratch, where the next
        # allocation in the same layer overwrites it.
        raw_params = {}
        if capture_weights:
            for _n, _p in model.named_parameters():
                if self._pass1_classify(_n) in QK_NORM_KEYS:
                    raw_params[_n] = (_p.detach().cpu().numpy()
                                      if hasattr(_p, "detach") else _p)
            for _li in range(n_layers):
                for _k in QK_NORM_KEYS:
                    _rec = by_layer.get(_li, {}).get(_k)
                    if _rec is not None and _rec.name in raw_params:
                        self.gamma_tile_dram(raw_params[_rec.name], _rec.name)

        # SiLU tables, one per layer (each layer's gate has its own scale).
        # Allocated HERE for the same reason as the RoPE tables and gamma
        # tiles: they are read by every step of a layer and must outlive the
        # per-layer rewind. Allocating them lazily inside the loop would place
        # each table in per-layer scratch, where the next allocation in that
        # same layer overwrites it -- the identical failure this file has now
        # hit three times.
        silu_luts = {}
        if scale_plan:
            for _li in range(n_layers):
                _pl = scale_plan["layers"][_li]
                silu_luts[_li], _ = self.lut_dram(
                    lambda z: z / (1.0 + _np_exp(-z)),
                    in_scale = _pl["s_gate"],
                    name = f"silu_l{_li}",
                )

        self._staging_base = self._staging_cursor
        # After n_layers of ping-ponging, the final output sits in this one.
        self.model_output_dram = layer_io[n_layers % 2]

        def qk_norm_addr(layer_idx: int, key: str) -> int:
            """
            DRAM address of a layer's broadcast q_norm/k_norm gamma tile, or 0.

            Reads the raw float vector back from the model rather than the
            ingested INT8 copy: gamma_tile_dram needs the true values to pick
            its own scale, and the ingested copy is a (1, head_dim) row while
            ELEMWISE_MUL wants it broadcast down a full tile.
            """
            rec = by_layer.get(layer_idx, {}).get(key)
            if rec is None or not capture_weights:
                return 0
            raw = raw_params.get(rec.name)
            if raw is None:
                return 0
            return self.gamma_tile_dram(raw, rec.name)

        # Retarget every projection's correction table to the scale the plan
        # says its consumer expects. Must happen before any instruction is
        # emitted: the RESCALE constants below are computed against these
        # scales, and a table left at its ingest-time normalisation makes them
        # wrong by an unknown factor.
        if scale_plan and capture_weights:
            for li in range(n_layers):
                pl = scale_plan["layers"][li].get("proj", {})
                for key, (s_in, s_out) in pl.items():
                    rec_ = by_layer.get(li, {}).get(key)
                    if rec_ is not None:
                        self.retarget_corr(rec_.name, s_in, s_out)

        for layer_idx in range(n_layers):
            layer = by_layer.get(layer_idx, {})

            def addr(key: str) -> int:
                rec = layer.get(key)
                return rec.dram_addr if rec else 0

            # Per-projection INT8_QUANT shifts for this layer. Without them
            # every GEMM uses the fixed default of 7, which saturates most
            # real layers -- see calibrate_scale_shifts.
            lshift = {}
            for k_, rec_ in layer.items():
                if scale_shifts and rec_.name in scale_shifts:
                    lshift[k_] = scale_shifts[rec_.name]

            # Ping-pong the layer activation buffers: read one, write the
            # other, swap. Reusing a single buffer would have the block
            # overwrite its own input mid-computation.
            x_dram   = layer_io[layer_idx % 2]
            out_dram = layer_io[(layer_idx + 1) % 2]

            # Rewind per-layer scratch: nothing inside a layer outlives it.
            self.reset_staging()
            attn_out_dram = self.alloc_staging(act_bytes)   # attn sublayer output (INT8)
            attn_res_dram = self.alloc_staging(res_bytes)   # x + attn(norm(x))
            mlp_out_dram  = self.alloc_staging(act_bytes)   # mlp sublayer output
            norm_dram     = self.alloc_staging(act_bytes)   # norm(x), fed to the sublayer

            # Recorded so a probe can dump the INTERMEDIATE buffers, not only
            # the layer's output. Knowing "layer 0 is wrong" is not actionable;
            # knowing which of these five buffers first disagrees is. Each
            # carries its own width and scale, which is also what lets a probe
            # distinguish a wrong constant (output correlated but uniformly
            # scaled) from a lowering bug (output decorrelated).
            self.layer_buffers[layer_idx] = {
                "norm":     (norm_dram,     1, "q07"),
                "attn_out": (attn_out_dram, 1, "s_o"),
                "attn_res": (attn_res_dram, res_w, "s_res"),
                "mlp_out":  (mlp_out_dram,  1, "s_d"),
                "layer_out": (out_dram,     res_w, "s_res"),
            }

            # ── Pre-attention RMSNorm (input_layernorm) ───────────────────────
            # Row-wide: the scale is a sum over all `hidden` elements, so the
            # single-tile block would apply a different scale per tile.
            plan_l = scale_plan["layers"][layer_idx] if scale_plan else None
            self.compile_rmsnorm_row(
                dram_in = x_dram, dram_out = norm_dram,
                n = hidden, sram_scalar = SCALAR_SCRATCH, norm_eps = norm_eps,
                dtype = (self.dtype if self.dtype != DType.INT8
                         else (1 if scale_plan else 0)),
                out_peak = plan_l["y_peak"] if plan_l else 1.0,
            )

            # ── Attention projections ────────────────────────────────────────
            self.compile_attention_block(
                dram_x   = norm_dram,   # sublayer sees norm(x)
                dram_wq  = addr("q_proj"),
                dram_wk  = addr("k_proj"),
                dram_wv  = addr("v_proj"),
                dram_wo  = addr("o_proj"),
                dram_out = attn_out_dram,
                hidden   = hidden,
                q_dim    = q_dim,
                kv_dim   = kv_dim,
                head_dim = head_dim,
                n_heads  = cfg.get("num_heads", 1),
                n_kv_heads = cfg.get("num_kv_heads", 1),
                seq_len  = sq,
                causal   = causal,     # prefill is causal; decode needs no mask
                cache_len = cache_len,
                dram_k_cache = (self.kv_cache_dram[layer_idx][0]
                                if self.kv_cache_dram else 0),
                dram_v_cache = (self.kv_cache_dram[layer_idx][1]
                                if self.kv_cache_dram else 0),
                dram_cos = cos_dram,   # RoPE -- the only carrier of position
                dram_sin = sin_dram,
                dram_q_norm = qk_norm_addr(layer_idx, "q_norm"),
                dram_k_norm = qk_norm_addr(layer_idx, "k_norm"),
                # In float the only scaling attention needs is its own
                # 1/sqrt(head_dim); in INT8 the temperature also has to undo
                # the score quantisation, which is what plan_l carries.
                softmax_T   = (0.0 if self.dtype == DType.FP32
                               else (plan_l["softmax_T"] if plan_l else 0.0)),
                shifts   = lshift,
                quantize_out = True,   # feeds the post-attention RMSNorm
            )

            # ── Residual: x + attn(norm(x)) ──────────────────────────────────
            # Pre-norm transformer: the residual adds the layer INPUT, not the
            # normalised copy, which is why x_dram is kept live across the
            # whole block rather than being normalised in place.
            # The two operands arrive with unrelated scales -- measured, they
            # differ by up to 348x -- so the sublayer output is widened into
            # the stream and converted into its units before the add. rescale_a
            # converts the stream itself when the layer boundary changes its
            # scale. Without this the add mixes two different units, 56 times
            # across the stack.
            self.compile_residual_add(
                dram_a = x_dram, dram_b = attn_out_dram,
                dram_out = attn_res_dram, n = hidden,
                dtype = (self.dtype if self.dtype != DType.INT8
                         else (1 if scale_plan else 0)),
                b_dtype = 0 if (scale_plan and self.dtype == DType.INT8) else None,
                widen_shift = (scale_plan["widen_shift"]
                               if (scale_plan and self.dtype == DType.INT8) else 0),
                rescale_b = plan_l["rescale_o"] if plan_l else None,
                rescale_a = plan_l["rescale_a"] if plan_l else None,
            )

            # ── Post-attention RMSNorm (post_attention_layernorm) ─────────────
            self.compile_rmsnorm_row(
                dram_in = attn_res_dram, dram_out = norm_dram,
                n = hidden, sram_scalar = SCALAR_SCRATCH, norm_eps = norm_eps,
                dtype = (self.dtype if self.dtype != DType.INT8
                         else (1 if scale_plan else 0)),
                out_peak = plan_l["y2_peak"] if plan_l else 1.0,
            )

            # ── MLP block (SwiGLU) ───────────────────────────────────────────
            silu_lut = silu_luts.get(layer_idx, 0)
            self.compile_mlp_block(
                dram_x    = norm_dram,   # sublayer sees norm(x + attn)
                dram_gate = addr("gate_proj"),
                dram_up   = addr("up_proj"),
                dram_down = addr("down_proj"),
                dram_out  = mlp_out_dram,
                dram_silu_lut = silu_lut,
                mul_shift = plan_l["mul_shift"] if plan_l else 0,
                hidden    = hidden,
                inter     = inter,
                shifts    = lshift,
                quantize_out = True,   # feeds the residual add below
            )

            # ── Residual: (x + attn) + mlp(norm(x + attn)) ────────────────────
            self.compile_residual_add(
                dram_a = attn_res_dram, dram_b = mlp_out_dram,
                dram_out = out_dram, n = hidden,
                dtype = (self.dtype if self.dtype != DType.INT8
                         else (1 if scale_plan else 0)),
                b_dtype = 0 if (scale_plan and self.dtype == DType.INT8) else None,
                widen_shift = (scale_plan["widen_shift"]
                               if (scale_plan and self.dtype == DType.INT8) else 0),
                rescale_b = plan_l["rescale_d"] if plan_l else None,
                # No rescale_a: attn_res_dram is already in this layer's scale,
                # converted by the first add.
            )

    # ── Output ────────────────────────────────────────────────────────────────

    def export_binary(self, output_filepath: str) -> None:
        """
        Flush the assembled instruction stream to a binary file.

        The file format is a flat array of 128-bit big-endian instruction words
        (16 bytes each).  The C++ InstructionDispatcher reads this file, calls
        __builtin_bswap64 on each 64-bit word to convert from big-endian to
        native little-endian, then calls Decoder::decode() to dispatch.
        """
        os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)
        with open(output_filepath, "wb") as f:
            f.write(self.instruction_stream)
        n_instr = len(self.instruction_stream) // 16
        print(
            f"[NPUCompiler] Written {n_instr} instructions "
            f"({len(self.instruction_stream)} bytes) → {output_filepath}"
        )

    @property
    def instruction_count(self) -> int:
        return len(self.instruction_stream) // 16

    def disassemble(self, limit: int = 0) -> str:
        """
        Human-readable listing of the compiled instruction stream.
        If limit > 0, only the first `limit` instructions are shown.
        """
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from toolchain.rtv_isa import disassemble as isa_disasm
        lines = []
        total = len(self.instruction_stream) // 16
        show  = total if limit == 0 else min(limit, total)
        for i in range(show):
            chunk = self.instruction_stream[i * 16 : (i + 1) * 16]
            hi, lo = struct.unpack(">QQ", chunk)
            word   = (hi << 64) | lo
            lines.append(f"{i:05d}  {isa_disasm(word)}")
        if show < total:
            lines.append(f"       … ({total - show} more instructions)")
        return "\n".join(lines)


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, os

    hal_path = os.path.join(os.path.dirname(__file__), "hal_config.json")

    compiler = NPUCompiler(hal_path)

    # Qwen3-0.6B layer-0 attention block example
    # M=16 (one context step), N=1024 (hidden), K=1024
    # Q output dim = 2048 (16 heads × 128), KV output dim = 1024 (8 heads × 128)
    compiler.compile_rmsnorm_block(
        sram_tile   = SRAMMap.input_slot(0),
        sram_scalar = SRAMMap.SCALAR_SCRATCH_BASE,
        sram_square = SRAMMap.input_slot(3),
        norm_eps    = 1e-6,
    )

    compiler.compile_sibling_qkv(
        dram_x   = 0x001000,
        dram_wq  = 0x100000,
        dram_wk  = 0x200000,
        dram_wv  = 0x300000,
        dram_out = 0x400000,
        m        = 16,
        n        = 1024,
        k        = 1024,
    )

    # O-proj dependency barrier
    compiler.append_synchronization_barrier(dependency_vector_id=1)

    out_path = "/tmp/qwen3_layer0_attention.bin"
    compiler.export_binary(out_path)

    print(f"\nFirst 20 instructions:")
    print(compiler.disassemble(limit=20))
