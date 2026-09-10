#pragma once
#include "decoder.hpp"
#include "memory_controller.hpp"
#include <cstdint>

// ─── PEArray ──────────────────────────────────────────────────────────────────
//
// Behavioral model of the 16×16 processing-element systolic array.
//
// All operands live in SRAM; the PE array reads and writes them via
// MemoryController::sram_ptr(). No DRAM access occurs here.
//
// Tiles are stored row-major, contiguously in SRAM (stride == cols after
// load_tile has been called). INT8 inputs accumulate into INT32.
//
// Neon path (ARM64 + FEAT_DotProd): compute_gemm uses vdotq_s32 when
// CHIP_SIM_USE_NEON is defined (set by CMake when compiling for AArch64).
//
// Scalar fallback: always available; used on non-ARM hosts and in unit tests.
//
// ─────────────────────────────────────────────────────────────────────────────

class PEArray {
public:
    // ── COMPUTE_GEMM ─────────────────────────────────────────────────────────
    //
    // C[M×N] += A[M×K] · B[K×N]  (INT8 → INT32 accumulation)
    //
    // a_sram     — SRAM offset of the A tile (activations), INT8, row-major, M×K bytes
    // b_sram     — SRAM offset of the B tile (weights),      INT8, row-major, K×N bytes
    // c_sram     — SRAM offset of the C tile (accumulators), INT32, row-major, M×N×4 bytes
    // activation — applied to each INT32 accumulator after the multiply-add:
    //              NONE: leave as INT32
    //              RELU: clamp negative INT32 to 0
    //              INT8_QUANT: divide by scale_shift, saturate to INT8, store back as INT8
    // accumulate — ISA v1.5: when true, add into existing C values instead of zeroing first.
    //              Required for K-tiled GEMM where C = A0·B0 + A1·B1 + … across K tiles.
    // scale_shift— ISA v1.5: right-shift for INT8_QUANT narrowing (default 7 → divide by 128).
    //
    // Throws std::out_of_range if any SRAM window falls outside [0, SRAM_BYTES).
    void compute_gemm(MemoryController& mem,
                      uint16_t a_sram, uint16_t b_sram, uint16_t c_sram,
                      uint16_t M, uint16_t N, uint16_t K,
                      Decoder::ActivationMode activation = Decoder::ActivationMode::NONE,
                      bool accumulate = false,
                      uint8_t scale_shift = 7,
                      Decoder::DType dtype = Decoder::DType::INT8);

    // ── EXEC_ALU ─────────────────────────────────────────────────────────────
    //
    // Element-wise operation on a tile of M×N INT8 values at sram_addr.
    // Results are written back to the same address (in-place).
    //
    // Supported modes:
    //   RELU        — max(0, x)
    //   SILU        — sigmoid(x)×x  (computed in float, rounded back to INT8)
    //   SQUARE      — x²            (widens to INT16 then narrows; saturates)
    //   RSQRT       — 1/√x          (float; writes INT8 approximation)
    //   ELEMWISE_MUL— tile_A × tile_B → INT8  (tile_B at b_sram)
    //   ELEMWISE_ADD— tile_A + tile_B → INT8  (tile_B at b_sram); residuals
    //   SCALAR_ADD  — add INT8 scalar (at b_sram, single byte) to every element
    //   INT8_QUANT  — divide every element by 128 and saturate to INT8
    //
    //   PERCHAN_QUANT — INT32 accumulator → INT8, one mantissa/shift per column
    //                   (table of uint16 at b_sram, low byte mant, high byte shift)
    //   RESCALE     — INT8 → INT8 scale conversion: (tile[i] * mant) >> shift,
    //                 with mant in params[7:0] and shift in params[15:8]
    //
    // b_sram is ignored for unary modes (RELU, SILU, SQUARE, RSQRT, INT8_QUANT).
    // params carries immediate operands that do not fit an SRAM address; only
    // RESCALE uses it.
    //
    // Throws std::out_of_range if any SRAM window falls outside [0, SRAM_BYTES).
    void exec_alu(MemoryController& mem,
                  uint16_t sram_addr, uint16_t M, uint16_t N,
                  Decoder::ActivationMode mode,
                  uint16_t b_sram = 0,
                  uint32_t params = 0,
                  Decoder::DType dtype = Decoder::DType::INT8);

    // ── REDUCE ───────────────────────────────────────────────────────────────
    //
    // Row-wise sum of an M×N INT8 tile, then sum across rows → single INT32 scalar.
    // The scalar is written as a 4-byte little-endian INT32 at scalar_sram.
    //
    // Used by RMSNorm: after EXEC_ALU(SQUARE), REDUCE computes sum(x²).
    //
    // Throws std::out_of_range on any out-of-bounds SRAM access.
    // accumulate (ISA v1.5): when true, ADD this tile's sum into the existing
    // scalar instead of overwriting it. Required to reduce across a row wider
    // than one tile -- RMSNorm's scale is a sum over the whole row, so each
    // tile's contribution must accumulate, exactly as K-tiled GEMM accumulates
    // partial products. Same accumulate bit, same reason.
    // mode: NONE (default) sums the tile; REDUCE_MAX takes its maximum.
    // dtype selects the element width of the tile read. The residual stream
    // is INT16, so the RMSNorm that consumes it must sum squares over INT16.
    void reduce(MemoryController& mem,
                uint16_t tile_sram, uint16_t scalar_sram,
                uint16_t M, uint16_t N,
                bool accumulate = false,
                Decoder::ActivationMode mode = Decoder::ActivationMode::NONE,
                Decoder::DType dtype = Decoder::DType::INT8);

    // ── SCALAR_ALU ───────────────────────────────────────────────────────────
    //
    // Read a 4-byte little-endian INT32 scalar from scalar_sram, then apply
    // the given operation element-wise to the M×N INT8 tile at tile_sram.
    //
    // Supported modes:
    //   SCALAR_ADD  — tile[i] = clamp(tile[i] + scalar, -128, 127)
    //   RSQRT       — scalar = 1/√scalar (float); tile[i] = clamp(tile[i] × scalar, -128, 127)
    //   ELEMWISE_MUL— tile[i] = clamp(tile[i] × scalar / 128, -128, 127)
    //
    // Throws std::out_of_range on any out-of-bounds SRAM access.
    void scalar_alu(MemoryController& mem,
                    uint16_t scalar_sram, uint16_t tile_sram,
                    uint16_t M, uint16_t N,
                    Decoder::ActivationMode mode,
                    uint32_t scale_num = 0,
                    Decoder::DType dtype = Decoder::DType::INT8);
};
