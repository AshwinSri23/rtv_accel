"""
compiler_state.py — shared device/compiler state for the RTV-Accel toolchain.

Extracted from npu_compiler.py when that file's ~3000 lines of INT8-era
Pass 1/2 graph-ingestion and compile_* orchestration logic were retired
(see TODO.md Phase 1.5, and rtv-accel-fp32-only-target in project memory
for the FP32 pivot that made them moot). What's here is the part every
caller — the TVM path (tvm_backend.py, rtv_tvm/*) and the surviving
lower-level INT8 intrinsic tests (rtv_tvm/test_tvm_path.py) alike — always
needed regardless of which dtype or which compiler front end is driving
it: HAL config loading, the DRAM staging allocator, the compile-time
constant region table, RoPE's host-side table math, and the two raw
instruction-emission primitives (`_pass3_store`, `_emit`) test_tvm_path.py
drives directly to build a STORE_TILE by hand.

DType/SRAMMap/Opcode/ActivationMode/encode are NOT redefined here — they
live in rtv_isa.py, the single source of truth for the ISA's wire format
(npu_compiler.py used to carry its own duplicate NPUOpcodes/NPUActivations/
SRAMMap definitions, confirmed value-identical to rtv_isa.py's; retired
along with the rest of it rather than carried forward as a second copy).
"""

import json
from dataclasses import dataclass

from rtv_isa import DType, SRAMMap, Opcode, ActivationMode, encode

# Backward-compatibility aliases: several call sites (ported over from
# npu_compiler.py) still spell these NPUOpcodes/NPUActivations.
NPUOpcodes = Opcode
NPUActivations = ActivationMode


def _n_tiles(dim: int, tile: int = 16) -> int:
    """Number of tiles needed to cover dim elements (ceiling division)."""
    return -(-dim // tile)


def _padded(dim: int, tile: int = 16) -> int:
    """Next multiple of tile that covers dim (zero-padding bound)."""
    return _n_tiles(dim, tile) * tile


def _align(n: int, align: int = 256) -> int:
    return ((n + align - 1) // align) * align


@dataclass
class ScheduleNode:
    """One instruction's fields, pre-packing. See NPUCompiler._emit for how
    these map onto the 128-bit wire format (mirrors rtv_isa.encode's own
    field layout, kept as a separate dataclass+pack path here because
    test_tvm_path.py calls _pass3_store/_emit directly rather than going
    through encode())."""
    opcode:     int
    dep_id:     int
    dram_addr:  int
    sram_addr:  int
    dim_m:      int = 0
    dim_n:      int = 0
    dim_k:      int = 0
    stride:     int = 0
    activation: int = 0
    accumulate:  bool = False
    scale_shift: int  = 0
    c_slot:      int  = 0
    dtype:       int  = 0


class NPUCompiler:
    """Device/compiler state shared across the toolchain: HAL config, the
    DRAM staging allocator, compile-time constant regions, and the raw
    instruction stream. No graph ingestion or high-level compile_* methods
    live here any more — those were npu_compiler.py's INT8-era Pass 1/2
    orchestration, retired with it."""

    def __init__(self, hal_config_path: str) -> None:
        with open(hal_config_path) as f:
            self.config = json.load(f)

        self.tile_dim: int = self.config.get("pe_array_dim", 16)
        self.sram_size: int = self.config.get("sram_size_bytes", 65536)

        self.dtype: int = DType.INT8

        self._dram_cursor: int = 0x0000_0000

        # Transient staging region -- see alloc_staging's own docstring.
        self._staging_base: int = 0
        self._staging_cursor: int = 0
        self._staging_peak: int = 0

        # Compile-time constant regions (RoPE cos/sin tables, LUTs, masks,
        # broadcast/permutation matrices, ...) that must land in the DRAM
        # image but have no weight/graph origin -- see tvm_backend.py's
        # extensive use of this for exactly that.
        self._const_regions: dict = {}

        # Memoization for those same constants -- see tvm_backend.py's
        # _get_or_build_const. Many of them (RoPE's rotate_half permutation
        # matrix, the row0<->broadcast masks, the KV-cache tail mask) are
        # bit-identical across every head and every layer within ONE decode
        # step (they depend only on head_dim/seq_k/pos/theta -- decode-step-
        # global values, never on which head or layer is asking), so
        # rebuilding a fresh copy per call was pure waste: at Qwen3-0.6B's
        # real dimensions, one 64KB RoPE permutation matrix alone was being
        # rebuilt 672 times (16 Q heads + 8 K heads, x28 layers) per decode
        # step for content that never changes within that step. Keyed by
        # whatever the caller considers this constant's identity (e.g.
        # ("rope_perm", head_dim)); a fresh NPUCompiler per decode step
        # means this cache naturally resets exactly when it should -- new
        # position, new step, new cache -- without any explicit reset call.
        self._const_cache: dict = {}

        self.instruction_stream: bytearray = bytearray()

    # ── Staging allocator ────────────────────────────────────────────────

    def init_staging(self) -> None:
        """Place the staging region immediately above whatever the caller
        has already bump-allocated from _dram_cursor (weights, a DRAM
        image's placed tensors, ...). Must be called AFTER any such
        allocation, for the same reason npu_compiler.py's version was:
        a fixed staging base risks landing inside real data for a large
        enough caller."""
        self._staging_base = _align(self._dram_cursor, 256)
        self._staging_cursor = self._staging_base

    def reset_staging(self) -> None:
        """Rewind staging to its base -- call once per layer/generation if
        staging buffers don't need to stay live across that boundary."""
        self._staging_cursor = self._staging_base

    def alloc_staging(self, n_bytes: int) -> int:
        """Bump-allocate one transient staging buffer; returns its DRAM
        address."""
        if self._staging_base == 0 and self._staging_cursor == 0:
            # Standalone use (no explicit init_staging() call): place
            # staging past whatever _dram_cursor already reflects.
            self.init_staging()
        addr = self._staging_cursor
        self._staging_cursor += _align(n_bytes, 256)
        self._staging_peak = max(self._staging_peak, self._staging_cursor)
        return addr

    @property
    def dram_bytes_needed(self) -> int:
        """Highest DRAM byte the compiled program can touch -- see
        alloc_staging: staging extends past any exported image length, so
        sizing a simulator run from this (not from image length) avoids a
        staging buffer's writes faulting as out-of-range."""
        return max(self._dram_cursor, self._staging_peak, self._staging_cursor)

    # ── RoPE host-side tables ────────────────────────────────────────────

    @staticmethod
    def rope_tables(max_pos: int, head_dim: int, base: float = 1e6, dtype: int = 0):
        """(cos, signed_sin) tables, each (max_pos, head_dim). theta_i =
        base ** (-2i/head_dim); angle at position p is p*theta_i. Both
        halves of each row share the same angles (the half-split
        convention Qwen/Llama use: q' = q*cos + rotate_half(q)*sin,
        rotate_half(x1,x2) = (-x2,x1)) -- the negation is folded into the
        FIRST half of the sin table, so the device only ever needs
        ELEMWISE_MUL/ELEMWISE_ADD, never a negate or shuffle instruction.

        dtype selects the output convention: DType.FP32 returns the exact
        float values; anything else returns Q0.7 INT8 (127 == 1.0, the
        scale ELEMWISE_MUL's (a*b)>>7 already assumes) -- FP32's rotation
        is exact and has no >>7 to undo, unlike the fixed-point form.
        """
        import numpy as np

        half = head_dim // 2
        inv = base ** (-np.arange(0, half, dtype=np.float64) * 2.0 / head_dim)
        ang = np.outer(np.arange(max_pos, dtype=np.float64), inv)

        cos = np.concatenate([np.cos(ang), np.cos(ang)], axis=1)
        sin = np.concatenate([-np.sin(ang), np.sin(ang)], axis=1)
        if dtype == DType.FP32:
            return cos.astype(np.float32), sin.astype(np.float32)
        q = lambda a: np.clip(np.round(a * 127.0), -127, 127).astype(np.int8)
        return q(cos), q(sin)

    # ── LUT tables (INT8-only; test_tvm_path.py's LUT intrinsic test) ────

    def lut_table(self, fn, in_scale: float, out_scale: float = None):
        """Build a 256-entry INT8 lookup table for `fn` at a known input
        scale. Entry i corresponds to code i-128 (true value
        (i-128)*in_scale); `fn` is evaluated on the VALUE, then requantized
        to out_scale (default: the table's own peak, using the full INT8
        range). Returns (table bytes, out_scale)."""
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

    # ── Raw instruction emission (test_tvm_path.py drives these directly) ─

    def pack_instruction(self, opcode: int, dep_id: int, dram_addr: int,
                         sram_addr: int, params_int: int) -> bytes:
        """Pack fields into a 16-byte (128-bit) big-endian word. See
        rtv_isa.encode for the canonical field layout this mirrors;
        kept as a separate byte-packer here since ScheduleNode/_emit
        is the interface test_tvm_path.py actually calls."""
        import struct
        word_high = (opcode & 0xF) << 60
        word_high |= (dep_id & 0xF) << 56
        word_high |= (dram_addr & 0xFFFF_FFFF) << 24
        word_high |= (sram_addr & 0xFFFF) << 8
        word_high |= (params_int >> 64) & 0xFF
        word_low = params_int & 0xFFFF_FFFF_FFFF_FFFF
        return struct.pack(">QQ", word_high, word_low)

    def _emit(self, node: ScheduleNode) -> None:
        """Assemble one ScheduleNode to bytes and append to the stream."""
        wire_dim_m = (((node.dtype & 0x3) << 13)
                      | ((node.scale_shift & 0x1F) << 8) | (node.dim_m & 0xFF))
        wire_dim_k = ((node.c_slot & 0xF) << 8) | (node.dim_k & 0xFF)
        wire_act = (0x80 if node.accumulate else 0x00) | (node.activation & 0x7F)

        params = (
            (wire_dim_m & 0xFFFF) << 56 |
            (node.dim_n & 0xFFFF) << 40 |
            (wire_dim_k & 0xFFFF) << 24 |
            (node.stride & 0xFFFF) << 8 |
            (wire_act & 0xFF)
        )
        self.instruction_stream.extend(
            self.pack_instruction(
                opcode=node.opcode, dep_id=node.dep_id,
                dram_addr=node.dram_addr, sram_addr=node.sram_addr,
                params_int=params,
            )
        )

    def _pass3_store(self, sram_addr: int, dram_addr: int, dim_m: int = 16,
                     dim_n: int = 16, stride: int = 0) -> ScheduleNode:
        """Build a STORE_TILE ScheduleNode for one tile. dim_n and stride
        are both in BYTES, not elements -- STORE_TILE is dtype-agnostic and
        just memcpy's dim_n bytes per row, advancing stride bytes per
        destination row."""
        if stride > 0xFFFF:
            raise ValueError(
                f"_pass3_store: stride {stride} exceeds the 16-bit stride "
                "field. A matrix wider than 65535 columns cannot have its "
                "row pitch encoded; lay the operand out so the pitch fits."
            )
        return ScheduleNode(
            opcode=NPUOpcodes.STORE_TILE, dep_id=0,
            dram_addr=dram_addr & 0xFFFF_FFFF, sram_addr=sram_addr & 0xFFFF,
            dim_m=dim_m, dim_n=dim_n, stride=stride & 0xFFFF,
        )
