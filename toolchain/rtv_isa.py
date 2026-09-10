"""
rtv_isa.py — RTV-Accel ISA definitions and 128-bit instruction encoding.

Mirrors inc/decoder.hpp exactly.  Every opcode value and activation mode
value must stay in sync with the C++ enums.

Bit layout (128 bits = hi:uint64 | lo:uint64):

  Field          Bits       Width   Position in {hi, lo}
  ────────────── ────────── ─────   ──────────────────────────────────────────
  opcode         127:124     4 b    hi[63:60]
  dep_id         123:120     4 b    hi[59:56]
  dram_addr      119:88     32 b    hi[55:24]   ← full 32-bit; addresses 4 GB
  sram_addr       87:72     16 b    hi[23:8]
  dim_m[15:8]     71:64      8 b    hi[7:0]     ← upper byte (straddles hi/lo)
  dim_m[7:0]      63:56      8 b    lo[63:56]   ← lower byte
  dim_n           55:40     16 b    lo[55:40]
  dim_k           39:24     16 b    lo[39:24]
  stride          23:8      16 b    lo[23:8]
  activation       7:0       8 b    lo[7:0]

  padding field removed — 8 bits reclaimed for the 32-bit dram_addr.

ISA v1.5 (backward-compatible; mirrors decoder.hpp's v1.5 section exactly):
no field moves, no field changes width. Three fields gain sub-fields in bit
positions that are always zero under v1:

  activation [7:0]  bit 7      = accumulate flag (COMPUTE_GEMM adds into C
                                  instead of zeroing it first)
                     bits 6:0  = mode enum (unchanged)
  dim_m wire word    bits 7:0  = tile rows (unchanged meaning)
                     bits 12:8 = scale_shift — INT32→INT8 requantization
                                  shift for INT8_QUANT (0-31)
  dim_k wire word    bits 7:0  = tile depth (unchanged meaning)
                     bits 11:8 = c_slot — destination C-buffer slot (0-15)

A v1-encoded word leaves all three zero, decoding to accumulate=False,
scale_shift=0, c_slot=0 — precisely v1 semantics (see src/simulator.cpp's
COMPUTE_GEMM handler for why scale_shift=0 is treated as "use the v1 default
of 7" at execution time, not as a literal >>0 request).
"""

from enum import IntEnum
import struct


# ── Opcodes ───────────────────────────────────────────────────────────────────

class Opcode(IntEnum):
    # Values match Decoder::Opcode in decoder.hpp (and compiler_state.py's
    # NPUOpcodes alias, kept identical on purpose).
    # 0x0 is reserved / unassigned.
    LOAD_TILE    = 0x1   # DMA: DRAM → SRAM
    STORE_TILE   = 0x2   # DMA: SRAM → DRAM
    COMPUTE_GEMM = 0x3   # 16×16 systolic matrix multiply
    EXEC_ALU     = 0x4   # element-wise op selected by ActivationMode
    WAIT_SYNC    = 0x5   # stall until dep_id is cleared
    REDUCE       = 0x6   # row-wise sum/sum-of-squares → scalar in SRAM
    SCALAR_ALU   = 0x7   # scalar from SRAM applied element-wise to tile


# ── Activation / operation modes ─────────────────────────────────────────────

class DType(IntEnum):
    """Element width of the tile an instruction operates on (dim_m bits 14:13).

    INT8 is 0, so every previously-encoded program decodes unchanged. INT16
    exists for the residual stream: it accumulates across 28 layers and holds
    Qwen3's massive activation channels, and an INT8 residual measurably costs
    accuracy against a 16-bit one.
    """
    INT8  = 0
    INT16 = 1
    INT32 = 2
    FP32  = 3   # no quantisation: the baseline that separates lowering bugs from quantisation loss

DType.BYTES = {0: 1, 1: 2, 2: 4, 3: 4, DType.INT8: 1, DType.INT16: 2, DType.INT32: 4, DType.FP32: 4}



class ActivationMode(IntEnum):
    NONE         = 0x00  # pass-through
    RELU         = 0x01  # max(0, x)
    INT8_QUANT   = 0x02  # scale and narrow INT32 → INT8
    SILU         = 0x03  # sigmoid(x)×x — SwiGLU gate
    SQUARE       = 0x04  # x² — RMSNorm step 1
    RSQRT        = 0x05  # 1/√x — RMSNorm step 3
    ELEMWISE_MUL = 0x06  # tile_A × tile_B — SwiGLU: SiLU(gate) × up
    SCALAR_ADD   = 0x07  # add scalar to tile — RMSNorm: mean(x²) + ε
    ELEMWISE_ADD = 0x08  # tile_A + tile_B — transformer residual connections
    TRANSPOSE    = 0x09  # in-place transpose of a square tile — enables Q·Kᵀ
    REDUCE_MAX   = 0x0A  # REDUCE mode: max instead of sum — softmax stability
    EXP_SUB      = 0x0B  # SCALAR_ALU: exp((tile − scalar)/T) — softmax numerator
    SCALAR_DIV   = 0x0C  # SCALAR_ALU: tile × 127 / scalar — softmax normalise
    CAUSAL_MASK  = 0x0D  # mask above the diagonal — causal attention
    REDUCE_SUMSQ = 0x0E  # REDUCE mode: sum of x² in INT32 — RMSNorm
    RMS_SCALE    = 0x0F  # SCALAR_ALU: tile × C / sqrt(scalar) — RMSNorm
    PERCHAN_QUANT= 0x10  # EXEC_ALU: INT32 → INT8 with a per-output-channel mantissa/shift
    RESCALE      = 0x11  # EXEC_ALU: INT8 → INT8 scale conversion, mant/shift in dram_addr
    LUT          = 0x12  # EXEC_ALU: out[i] = table[tile[i]+128], 256-byte table at stride


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode(
    opcode:      int,
    dep_id:      int = 0,
    dram_addr:   int = 0,
    sram_addr:   int = 0,
    dim_m:       int = 0,
    dim_n:       int = 0,
    dim_k:       int = 0,
    stride:      int = 0,
    activation:  int = 0,
    accumulate:  bool = False,
    scale_shift: int = 0,
    c_slot:      int = 0,
    dtype:       int = 0,
) -> int:
    """
    Pack all fields into a single 128-bit Python int.
    Returns the instruction word as a Python int (fits in 128 bits).
    Raises ValueError if any field exceeds its declared width.

    dim_m straddles hi/lo: upper byte in hi[7:0], lower byte in lo[63:56].
    dram_addr is now 32 bits (was 24); padding field has been removed.

    ISA v1.5: dim_m/dim_k are now 8-bit tile dimensions (0-255, bounded by
    the 16x16 PE array) -- the wire bits above that hold scale_shift/c_slot,
    packed here from the accumulate/scale_shift/c_slot parameters, not from
    dim_m/dim_k directly. accumulate packs into activation's bit 7.
    """
    if opcode      & ~0xF:  raise ValueError(f"opcode {opcode:#x} exceeds 4 bits")
    if dep_id      & ~0xF:  raise ValueError(f"dep_id {dep_id:#x} exceeds 4 bits")
    if dram_addr   & ~0xFFFF_FFFF: raise ValueError(f"dram_addr {dram_addr:#x} exceeds 32 bits")
    if sram_addr   & ~0xFFFF:      raise ValueError(f"sram_addr {sram_addr:#x} exceeds 16 bits")
    if dim_m       & ~0xFF:        raise ValueError(f"dim_m {dim_m} exceeds 8 bits (v1.5 tile rows)")
    if dim_n       & ~0xFFFF:      raise ValueError(f"dim_n {dim_n} exceeds 16 bits")
    if dim_k       & ~0xFF:        raise ValueError(f"dim_k {dim_k} exceeds 8 bits (v1.5 tile depth)")
    if stride      & ~0xFFFF:      raise ValueError(f"stride {stride} exceeds 16 bits")
    if activation  & ~0x7F:        raise ValueError(f"activation {activation:#x} exceeds 7 bits (bit 7 is accumulate)")
    if scale_shift & ~0x1F:        raise ValueError(f"scale_shift {scale_shift} exceeds 5 bits (0-31)")
    if c_slot      & ~0xF:         raise ValueError(f"c_slot {c_slot} exceeds 4 bits (0-15)")
    if dtype       & ~0x3:         raise ValueError(f"dtype {dtype} exceeds 2 bits (0-3)")

    wire_dim_m = (((dtype & 0x3) << 13) | ((scale_shift & 0x1F) << 8)
                  | (dim_m & 0xFF))
    wire_dim_k = ((c_slot      & 0xF)  << 8) | (dim_k & 0xFF)
    wire_act   = (0x80 if accumulate else 0x00) | (activation & 0x7F)

    hi = (
        ((opcode      & 0xF)         << 60) |
        ((dep_id      & 0xF)         << 56) |
        ((dram_addr   & 0xFFFF_FFFF) << 24) |   # 32-bit field at [55:24]
        ((sram_addr   & 0xFFFF)      <<  8) |
        ((wire_dim_m >> 8)           & 0xFF)     # upper byte of wire_dim_m in hi[7:0]
    )
    lo = (
        ((wire_dim_m  & 0xFF)        << 56) |   # lower byte of wire_dim_m in lo[63:56]
        ((dim_n       & 0xFFFF)      << 40) |
        ((wire_dim_k  & 0xFFFF)      << 24) |
        ((stride      & 0xFFFF)      <<  8) |
        (wire_act     & 0xFF)
    )
    return (hi << 64) | lo


def decode(word: int) -> dict:
    """
    Unpack a 128-bit instruction word back into a field dict.
    Useful for debugging and disassembly.

    dim_m straddles hi/lo: hi[7:0] holds the upper byte, lo[63:56] the lower.
    dram_addr is 32 bits; padding field no longer exists.

    ISA v1.5: dim_m/dim_k are the 8-bit tile dimensions (wire bits 7:0 of
    each field); scale_shift/c_slot are extracted from the bits above that;
    accumulate is bit 7 of the activation wire byte, mode enum is bits 6:0.
    A v1-encoded word leaves all three sub-fields zero.
    """
    hi = (word >> 64) & 0xFFFF_FFFF_FFFF_FFFF
    lo =  word        & 0xFFFF_FFFF_FFFF_FFFF

    wire_dim_m = ((hi & 0xFF) << 8) | ((lo >> 56) & 0xFF)
    wire_dim_k = (lo >> 24) & 0xFFFF
    wire_act   = lo & 0xFF

    return {
        "opcode":      Opcode((hi >> 60) & 0xF),
        "dep_id":              (hi >> 56) & 0xF,
        "dram_addr":           (hi >> 24) & 0xFFFF_FFFF,   # 32-bit
        "sram_addr":           (hi >>  8) & 0xFFFF,
        "dim_m":               wire_dim_m & 0xFF,
        "dim_n":               (lo >> 40) & 0xFFFF,
        "dim_k":               wire_dim_k & 0xFF,
        "stride":              (lo >>  8) & 0xFFFF,
        "activation":  ActivationMode(wire_act & 0x7F),
        "accumulate":  bool((wire_act >> 7) & 1),
        "scale_shift": (wire_dim_m >> 8) & 0x1F,
        "c_slot":      (wire_dim_k >> 8) & 0xF,
        "dtype":       DType((wire_dim_m >> 13) & 0x3),
    }


def to_bytes(word: int) -> bytes:
    """
    Serialise a 128-bit instruction word to 16 bytes, big-endian
    (hi word first — matches the struct RawInstruction { hi, lo } layout
    when read as a big-endian pair of uint64_t values).
    """
    hi = (word >> 64) & 0xFFFF_FFFF_FFFF_FFFF
    lo =  word        & 0xFFFF_FFFF_FFFF_FFFF
    return struct.pack(">QQ", hi, lo)


def disassemble(word: int) -> str:
    """
    Return a one-line human-readable string for logging.

    ISA v1.5 sub-fields (accumulate, scale_shift, c_slot) are appended only
    when nonzero -- a v1-encoded word disassembles identically to before.
    A v1 disassembler reading a v1.5 word without this would print
    misleading dimensions (e.g. dim_m=0x0710 rendering as M=1808); decoding
    dim_m/dim_k to their 8-bit tile values here avoids that.
    """
    f = decode(word)
    line = (
        f"{f['opcode'].name:<12} "
        f"dep={f['dep_id']} "
        f"dram=0x{f['dram_addr']:08X} "   # 8 hex digits for 32-bit address
        f"sram=0x{f['sram_addr']:04X} "
        f"M={f['dim_m']} N={f['dim_n']} K={f['dim_k']} "
        f"stride={f['stride']} "
        f"act={f['activation'].name}"
    )
    if f["accumulate"]:
        line += " acc=1"
    if f["scale_shift"]:
        line += f" sshift={f['scale_shift']}"
    if f["c_slot"]:
        line += f" cslot={f['c_slot']}"
    if f["dtype"]:
        line += f" dtype={f['dtype'].name}"
    return line


# ── SRAM Memory Map ───────────────────────────────────────────────────────────

class SRAMMap:
    """Hardware SRAM memory map and slot allocation helpers."""
    INPUT_BASE  = 0x0000  # 0x0000–0x03FF  Active input / activation tiles
    WEIGHT_BASE = 0x0400  # 0x0400–0x07FF  Streaming weight buffers
    OUTPUT_BASE = 0x0800  # ISA v1.5 c_slot-addressed output zone (16 slots)

    ZONE_SIZE   = 0x0400  # 1 KiB per zone (input/weight zones only)
    TILE_BYTES  = 0x0100  # 256 bytes per 16×16 INT8 tile (input/weight slots)
    OUTPUT_TILE_BYTES = 0x0400  # 1024 bytes -- 16*16*sizeof(int32)

    SCALAR_SCRATCH_BASE = 0x0800 + 16 * 0x0400   # 0x4800
    CORR_SCRATCH = SCALAR_SCRATCH_BASE + 0x0100   # 0x4900, 32 bytes used

    WIDE_INPUT_BASE  = 0x5000     # 4 slots x 1 KiB
    WIDE_WEIGHT_BASE = 0x6000     # 4 slots x 1 KiB
    WIDE_SLOT_BYTES  = 0x0400

    @staticmethod
    def input_slot(n: int = 0, width: int = 1) -> int:
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
        return SRAMMap.OUTPUT_BASE + n * SRAMMap.OUTPUT_TILE_BYTES

