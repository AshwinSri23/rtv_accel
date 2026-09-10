#pragma once
#include <cstdint>
#include <string>

// ─── Bit layout (128-bit fixed-length instruction) ───────────────────────────
//
//  Field        Bits       Width   Location in {hi:uint64_t, lo:uint64_t}
//  ──────────── ────────── ─────   ────────────────────────────────────────────
//  opcode       127:124     4 b    hi[63:60]
//  dep_id       123:120     4 b    hi[59:56]
//  dram_addr    119:88     32 b    hi[55:24]   ← full 32-bit; addresses 4 GB
//  sram_addr     87:72     16 b    hi[23:8]
//  dim_m[15:8]   71:64      8 b    hi[7:0]     ─╮  dim_m straddles hi/lo
//  dim_m[7:0]    63:56      8 b    lo[63:56]   ─╯
//  dim_n         55:40     16 b    lo[55:40]    │  Params (72 b, no padding field)
//  dim_k         39:24     16 b    lo[39:24]    │
//  stride        23:8      16 b    lo[23:8]     │
//  activation     7:0       8 b    lo[7:0]     ─╯
//
//  padding field removed (was lo[15:8]) to make room for the 32-bit dram_addr.
//
// ─────────────────────────────────────────────────────────────────────────────

// Decoder owns all ISA types as nested members so callers never need to import
// a separate namespace; everything is accessed as Decoder::Opcode, etc.
class Decoder {
public:
    // 4-bit opcode field — 16 slots available, 7 defined.
    // Values match the Python NPUOpcodes class in toolchain/npu_compiler.py.
    // Value 0x0 is intentionally unassigned (reserved / NOP).
    enum class Opcode : uint8_t {
        LOAD_TILE    = 0x1,  // DMA: DRAM → SRAM
        STORE_TILE   = 0x2,  // DMA: SRAM → DRAM
        COMPUTE_GEMM = 0x3,  // 16×16 systolic matrix multiply
        EXEC_ALU     = 0x4,  // element-wise ops; activation selected by ActivationMode
        WAIT_SYNC    = 0x5,  // stall until dep_id is cleared
        REDUCE       = 0x6,  // row-wise sum/sum-of-squares → scalar written to SRAM (RMSNorm)
        SCALAR_ALU   = 0x7,  // read scalar from SRAM, apply element-wise to tile (RMSNorm scale, attn 1/√d)
    };

    // 8-bit activation / operation mode carried in bits [7:0] of Params.
    // Selects the operation for EXEC_ALU and SCALAR_ALU. 256 slots, 8 defined.
    //
    // For binary EXEC_ALU modes (ELEMWISE_MUL): tile A is at sram_addr;
    // tile B is at the address carried in the stride field.
    enum class ActivationMode : uint8_t {
        NONE        = 0x00,  // pass-through
        RELU        = 0x01,  // max(0, x) — post-GEMM clamp
        INT8_QUANT  = 0x02,  // scale and narrow INT32 accumulator to INT8
        SILU        = 0x03,  // sigmoid(x)×x (Swish) — SwiGLU gate activation
        SQUARE      = 0x04,  // x² element-wise — RMSNorm step 1
        RSQRT       = 0x05,  // 1/√x element-wise — RMSNorm step 3
        ELEMWISE_MUL= 0x06,  // tile_A × tile_B element-wise — SwiGLU: SiLU(gate)×up
        SCALAR_ADD  = 0x07,  // add scalar from SRAM to every element — RMSNorm: mean(x²)+ε
        ELEMWISE_ADD= 0x08,  // tile_A + tile_B element-wise — transformer residuals
        TRANSPOSE   = 0x09,  // in-place transpose of a SQUARE tile — enables Q·Kᵀ
        REDUCE_MAX  = 0x0A,  // REDUCE mode: max instead of sum — softmax stability
        EXP_SUB     = 0x0B,  // SCALAR_ALU: exp((tile − scalar)/T) — softmax numerator
        SCALAR_DIV  = 0x0C,  // SCALAR_ALU: tile × 127 / scalar — softmax normalise
        CAUSAL_MASK = 0x0D,  // write MASKED sentinel above the diagonal — causal attention
        REDUCE_SUMSQ= 0x0E,  // REDUCE mode: sum of x² accumulated in INT32 — RMSNorm
        RMS_SCALE   = 0x0F,  // SCALAR_ALU: tile × C / sqrt(scalar), C in dram_addr — RMSNorm
        RESCALE     = 0x11,  // EXEC_ALU: convert an INT8 tile from one scale to another.
                             // out[i] = clamp8((tile[i] * mant) >> shift), with mant in
                             // bits 7:0 of dram_addr and shift in bits 15:8. Needed because
                             // an INT8 buffer's scale is a real quantity, not a power of
                             // two: aligning the residual stream with a sublayer output at
                             // layer 0 of Qwen3 means multiplying by 4.26, which
                             // scale_shift's power-of-two field cannot express. Without
                             // this, ELEMWISE_ADD silently adds two differently-scaled
                             // operands -- see the scale-bookkeeping section of the spec.
        LUT         = 0x12,  // EXEC_ALU: out[i] = table[tile[i] + 128], with a 256-byte
                             // INT8 table at b_sram. The way a fixed-function unit
                             // evaluates a nonlinearity. Necessary because applying a
                             // nonlinear function to the CODE is not the same function:
                             // SILU computes x/(1+exp(-x)) on the raw INT8 code, and at
                             // integer granularity that is indistinguishable from ReLU
                             // for all 256 inputs -- SiLU's curvature lives near |x|<6.
                             // The table is built at compile time for the buffer's known
                             // scale, so the function is evaluated at the true value.
        PERCHAN_QUANT=0x10,  // EXEC_ALU: INT32 accumulator → INT8 with a PER-OUTPUT-CHANNEL
                             // scale. Per-tensor INT8 weights are not sufficient for a real
                             // LLM: measured on Qwen3-0.6B, one scale per tensor drives the
                             // hidden state to ANTI-correlation with ground truth by layer 27
                             // (cos -0.23), while one scale per output channel is
                             // near-lossless (cos 0.9992). A power-of-two shift cannot carry
                             // those scales -- rounding each to a power of two still fails --
                             // so each channel gets an 8-bit mantissa and its own shift:
                             //     out[r][c] = clamp8((acc[r][c] * mant[c]) >> shift[c])
                             // The correction table lives in SRAM at the address in `stride`,
                             // one INT16 per column: low byte mant (128..255), high byte shift.
    };

    // Element width of the tile an instruction operates on.
    // Encoded in bits 14:13 of the dim_m wire word; 0 == INT8 == v1 behaviour.
    enum class DType : uint8_t {
        INT8  = 0,
        INT16 = 1,
        INT32 = 2,
        // Single-precision float. Present so the machine can run a model with
        // NO quantisation at all, which makes it possible to tell a lowering
        // bug from a quantisation limit: if the float path is correct, every
        // address, tiling decision, K-accumulation and scheduling choice in
        // the compiler is proven, and anything still wrong in the INT8 path is
        // quantisation. Floats carry their own exponent, so the scale-
        // bookkeeping mistakes the INT8 path is prone to cannot occur here.
        FP32  = 3,
    };

    // Bytes per element for a given width.
    static constexpr uint32_t dtype_bytes(DType d) {
        return d == DType::INT8 ? 1u : (d == DType::INT16 ? 2u : 4u);
    }

    static constexpr bool dtype_is_float(DType d) { return d == DType::FP32; }

    // Raw 128-bit instruction word stored as two native 64-bit halves.
    // hi = bits [127:64],  lo = bits [63:0].
    // Callers construct this from a memory-mapped instruction buffer.
    struct RawInstruction {
        uint64_t hi;
        uint64_t lo;
    };

    // All fields extracted from a RawInstruction in a named, typed form.
    // dram_addr is a full 32-bit value, addressing up to 4 GB of system DRAM.
    // dim_m/n/k are tile row/col/depth counts bounded by the 16×16 PE array.
    //
    // v1.5 sub-fields (zero in a v1-encoded instruction → identical semantics):
    //   accumulate  — bit 7 of the activation wire byte; when true, COMPUTE_GEMM
    //                 adds into the existing C accumulator instead of zeroing it first
    //   scale_shift — bits 12:8 of the dim_m wire word; INT32→INT8 right-shift (0–31)
    //   dtype       — bits 14:13 of the dim_m wire word; element width for the
    //                 tile an instruction operates on. Zero decodes as INT8, so
    //                 every previously-encoded program keeps its meaning.
    //   c_slot      — bits 11:8 of the dim_k wire word; destination C-buffer slot (0–15)
    struct DecodedInstruction {
        Opcode         opcode;
        uint8_t        dep_id;      // 4-bit dependency tag for WAIT_SYNC barriers
        uint32_t       dram_addr;   // 32-bit system DRAM pointer (full 4 GB range)
        uint16_t       sram_addr;   // 16-bit NPU scratchpad offset
        // ── Params sub-fields ────────────────────────────────────────────────
        uint8_t        dim_m;       // tile rows   (wire bits 7:0 of dim_m field)
        uint16_t       dim_n;       // tile columns (full 16-bit wire field)
        uint8_t        dim_k;       // tile depth  (wire bits 7:0 of dim_k field)
        uint16_t       stride;      // memory stride between rows
        ActivationMode activation;
        // ── ISA v1.5 sub-fields (zero → v1 semantics) ────────────────────────
        bool           accumulate;  // do not zero C before GEMM; add into existing
        uint8_t        scale_shift; // INT32→INT8 shift amount (replaces fixed >>7)
        uint8_t        c_slot;      // C-buffer destination slot index
        // Element width of the operand tile. INT8 is the default and the only
        // width the ISA originally had; the residual stream needs INT16
        // because it accumulates across 28 layers and carries Qwen3's massive
        // activation channels (2610 against a median of 0.065 at layer 2),
        // which an INT8 buffer cannot hold -- measured, an INT8 residual costs
        // 2 of 8 prompts against a 16-bit one.
        DType          dtype;
    };

    // Decode a raw 128-bit word into named fields.
    // Throws std::invalid_argument on an unrecognised opcode or activation mode.
    static DecodedInstruction decode(const RawInstruction& raw);

    // Re-encode a decoded instruction back to a raw word.
    // Guarantees: decode(encode(d)) == d for any valid DecodedInstruction.
    static RawInstruction encode(const DecodedInstruction& instr);

    // Returns a one-line human-readable string, e.g.:
    //   COMPUTE_GEMM dep=3 dram=0x001000 sram=0x0200 M=16 N=16 K=16 stride=16 pad=0 act=RELU
    static std::string to_string(const DecodedInstruction& instr);

private:
    // Validate and convert a 4-bit opcode value; throw on unknown values.
    static Opcode         parse_opcode(uint8_t bits);
    // Validate and convert an 8-bit activation value; throw on unknown values.
    static ActivationMode parse_activation(uint8_t bits);
};
