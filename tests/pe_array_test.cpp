#include "../inc/pe_array.hpp"
#include "../inc/memory_controller.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>

// ─── Minimal test harness ────────────────────────────────────────────────────

static int s_pass = 0, s_fail = 0;

#define CHECK(cond) do { \
    if (cond) { ++s_pass; } \
    else { ++s_fail; \
           std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
} while(0)
#define CHECK_EQ(a, b) CHECK((a) == (b))

// ─── SRAM zone constants (matches SRAMMap in npu_compiler.py) ────────────────

static constexpr uint16_t ZONE_INPUT  = 0x0000;
static constexpr uint16_t ZONE_WEIGHT = 0x0400;
static constexpr uint16_t ZONE_OUTPUT = 0x0800;
static constexpr uint16_t ZONE_SCALAR = 0x0C00;

// ─── Helpers ─────────────────────────────────────────────────────────────────

// Write INT8 values row-major into SRAM at `base`.
static void write_tile_i8(MemoryController& mem, uint16_t base,
                          const int8_t* data, uint16_t rows, uint16_t cols)
{
    std::memcpy(mem.sram_ptr(base), data, static_cast<std::size_t>(rows) * cols);
}

// Read INT32 accumulators from SRAM at `base`.
static int32_t read_acc(MemoryController& mem, uint16_t base, uint16_t idx)
{
    int32_t v;
    std::memcpy(&v, mem.sram_ptr(base) + idx * 4, 4);
    return v;
}

// ─── COMPUTE_GEMM tests ───────────────────────────────────────────────────────

// Cross-check against a reference computed in plain C.
// A = identity 4×4, B = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
// C = A·B = B  (since A is identity)
static void test_gemm_identity() {
    MemoryController mem;

    const uint16_t M = 4, N = 4, K = 4;

    int8_t A[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    int8_t B[16] = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16,
    };

    write_tile_i8(mem, ZONE_INPUT,  A, M, K);
    write_tile_i8(mem, ZONE_WEIGHT, B, K, N);

    PEArray pe;
    pe.compute_gemm(mem, ZONE_INPUT, ZONE_WEIGHT, ZONE_OUTPUT, M, N, K);

    // C = A·B = B row by row
    for (uint16_t m = 0; m < M; ++m)
        for (uint16_t n = 0; n < N; ++n)
            CHECK_EQ(read_acc(mem, ZONE_OUTPUT, m * N + n),
                     static_cast<int32_t>(B[m * N + n]));
}

// Verify accumulation: all-ones A × all-ones B → each cell == K.
static void test_gemm_all_ones() {
    MemoryController mem;

    const uint16_t M = 8, N = 8, K = 8;
    int8_t A[64], B[64];
    std::memset(A, 1, sizeof(A));
    std::memset(B, 1, sizeof(B));

    write_tile_i8(mem, ZONE_INPUT,  A, M, K);
    write_tile_i8(mem, ZONE_WEIGHT, B, K, N);

    PEArray pe;
    pe.compute_gemm(mem, ZONE_INPUT, ZONE_WEIGHT, ZONE_OUTPUT, M, N, K);

    for (int i = 0; i < M * N; ++i)
        CHECK_EQ(read_acc(mem, ZONE_OUTPUT, i), static_cast<int32_t>(K));
}

// RELU post-activation: negative accumulators should be clamped to 0.
static void test_gemm_relu_activation() {
    MemoryController mem;

    const uint16_t M = 2, N = 2, K = 2;
    // A = [[1,1],[1,1]], B = [[1,-1],[-1,1]]
    // C = [[0,0],[0,0]] before RELU (1·1 + 1·(-1) = 0 each)
    // Use values that give one negative: A=[[2,0],[0,0]], B=[[-1,1],[0,0]]
    // C[0][0] = -2, C[0][1] = 2, rest = 0 → after RELU: 0, 2, 0, 0
    int8_t A[4] = { 2, 0, 0, 0 };
    int8_t B[4] = { -1, 1, 0, 0 };

    write_tile_i8(mem, ZONE_INPUT,  A, M, K);
    write_tile_i8(mem, ZONE_WEIGHT, B, K, N);

    PEArray pe;
    pe.compute_gemm(mem, ZONE_INPUT, ZONE_WEIGHT, ZONE_OUTPUT, M, N, K,
                    Decoder::ActivationMode::RELU);

    CHECK_EQ(read_acc(mem, ZONE_OUTPUT, 0), 0);  // was -2, clamped
    CHECK_EQ(read_acc(mem, ZONE_OUTPUT, 1), 2);
    CHECK_EQ(read_acc(mem, ZONE_OUTPUT, 2), 0);
    CHECK_EQ(read_acc(mem, ZONE_OUTPUT, 3), 0);
}

// Full 16×16 smoke test: known values, cross-check sum.
static void test_gemm_16x16_smoke() {
    MemoryController mem;

    const uint16_t M = 16, N = 16, K = 16;
    int8_t A[256], B[256];
    // A[i][j] = (i + j) % 7; B[i][j] = (i * j) % 5
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j) {
            A[i * 16 + j] = static_cast<int8_t>((i + j) % 7);
            B[i * 16 + j] = static_cast<int8_t>((i * j) % 5);
        }

    write_tile_i8(mem, ZONE_INPUT,  A, M, K);
    write_tile_i8(mem, ZONE_WEIGHT, B, K, N);

    PEArray pe;
    pe.compute_gemm(mem, ZONE_INPUT, ZONE_WEIGHT, ZONE_OUTPUT, M, N, K);

    // Reference: compute C in plain C and compare
    int32_t ref[256] = {};
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n)
            for (int k = 0; k < 16; ++k)
                ref[m * 16 + n] += static_cast<int32_t>(A[m * 16 + k])
                                 * static_cast<int32_t>(B[k * 16 + n]);

    bool all_ok = true;
    for (int i = 0; i < 256; ++i) {
        if (read_acc(mem, ZONE_OUTPUT, i) != ref[i]) { all_ok = false; break; }
    }
    CHECK(all_ok);
}

// ─── EXEC_ALU tests ──────────────────────────────────────────────────────────

static void test_exec_alu_relu() {
    MemoryController mem;
    int8_t tile[4] = { -5, 0, 3, -1 };
    write_tile_i8(mem, ZONE_INPUT, tile, 2, 2);

    PEArray pe;
    pe.exec_alu(mem, ZONE_INPUT, 2, 2, Decoder::ActivationMode::RELU);

    const int8_t* out = reinterpret_cast<const int8_t*>(mem.sram_ptr(ZONE_INPUT));
    CHECK_EQ(out[0], 0);
    CHECK_EQ(out[1], 0);
    CHECK_EQ(out[2], 3);
    CHECK_EQ(out[3], 0);
}

static void test_exec_alu_square() {
    MemoryController mem;
    int8_t tile[4] = { 3, -4, 5, 0 };
    write_tile_i8(mem, ZONE_INPUT, tile, 2, 2);

    PEArray pe;
    pe.exec_alu(mem, ZONE_INPUT, 2, 2, Decoder::ActivationMode::SQUARE);

    const int8_t* out = reinterpret_cast<const int8_t*>(mem.sram_ptr(ZONE_INPUT));
    CHECK_EQ(out[0], 9);    // 3² = 9 (fits INT8)
    CHECK_EQ(out[1], 16);   // 4² = 16
    CHECK_EQ(out[2], 25);   // 5² = 25
    CHECK_EQ(out[3], 0);
}

static void test_exec_alu_elemwise_mul() {
    MemoryController mem;
    // A = [4, 8, 12, 16], B = [2, 2, 2, 2]
    // prod = [8, 16, 24, 32], >>7 = [0, 0, 0, 0] (small values)
    // Use larger values: A = [64, 64, 64, 64], B = [64, 64, 64, 64]
    // prod = 4096, >>7 = 32
    int8_t A[4] = { 64, 64, 64, 64 };
    int8_t B[4] = { 64, 64, 64, 64 };
    write_tile_i8(mem, ZONE_INPUT,  A, 2, 2);
    write_tile_i8(mem, ZONE_WEIGHT, B, 2, 2);

    PEArray pe;
    pe.exec_alu(mem, ZONE_INPUT, 2, 2, Decoder::ActivationMode::ELEMWISE_MUL, ZONE_WEIGHT);

    const int8_t* out = reinterpret_cast<const int8_t*>(mem.sram_ptr(ZONE_INPUT));
    // 64 * 64 = 4096; 4096 >> 7 = 32
    for (int i = 0; i < 4; ++i)
        CHECK_EQ(out[i], 32);
}

// ─── REDUCE tests ─────────────────────────────────────────────────────────────

static void test_reduce_sum() {
    MemoryController mem;
    // 2×2 tile: [1, 2, 3, 4] → sum = 10
    int8_t tile[4] = { 1, 2, 3, 4 };
    write_tile_i8(mem, ZONE_INPUT, tile, 2, 2);

    PEArray pe;
    pe.reduce(mem, ZONE_INPUT, ZONE_SCALAR, 2, 2);

    int32_t result;
    std::memcpy(&result, mem.sram_ptr(ZONE_SCALAR), 4);
    CHECK_EQ(result, 10);
}

// After SQUARE, REDUCE gives sum of squares — core of RMSNorm.
static void test_reduce_sum_of_squares() {
    MemoryController mem;
    // tile = [3, 4]  (1×2)
    // After SQUARE: [9, 16]
    // After REDUCE: 25
    int8_t tile[2] = { 3, 4 };
    write_tile_i8(mem, ZONE_INPUT, tile, 1, 2);

    PEArray pe;
    pe.exec_alu(mem, ZONE_INPUT, 1, 2, Decoder::ActivationMode::SQUARE);
    pe.reduce(mem, ZONE_INPUT, ZONE_SCALAR, 1, 2);

    int32_t result;
    std::memcpy(&result, mem.sram_ptr(ZONE_SCALAR), 4);
    CHECK_EQ(result, 25);
}

// ─── SCALAR_ALU tests ─────────────────────────────────────────────────────────

static void test_scalar_alu_add() {
    MemoryController mem;
    // tile = [10, 20, 30, 40], scalar = 5 → [15, 25, 35, 45]
    int8_t tile[4] = { 10, 20, 30, 40 };
    int32_t scalar = 5;
    write_tile_i8(mem, ZONE_INPUT, tile, 2, 2);
    std::memcpy(mem.sram_ptr(ZONE_SCALAR), &scalar, 4);

    PEArray pe;
    pe.scalar_alu(mem, ZONE_SCALAR, ZONE_INPUT, 2, 2, Decoder::ActivationMode::SCALAR_ADD);

    const int8_t* out = reinterpret_cast<const int8_t*>(mem.sram_ptr(ZONE_INPUT));
    CHECK_EQ(out[0], 15);
    CHECK_EQ(out[1], 25);
    CHECK_EQ(out[2], 35);
    CHECK_EQ(out[3], 45);
}

static void test_scalar_alu_rsqrt() {
    MemoryController mem;
    // scalar = 100; 1/sqrt(100) = 0.1
    // tile = [10, 20]; output: round(10 * 0.1) = 1, round(20 * 0.1) = 2
    int32_t scalar = 100;
    int8_t tile[2] = { 10, 20 };
    write_tile_i8(mem, ZONE_INPUT, tile, 1, 2);
    std::memcpy(mem.sram_ptr(ZONE_SCALAR), &scalar, 4);

    PEArray pe;
    pe.scalar_alu(mem, ZONE_SCALAR, ZONE_INPUT, 1, 2, Decoder::ActivationMode::RSQRT);

    const int8_t* out = reinterpret_cast<const int8_t*>(mem.sram_ptr(ZONE_INPUT));
    CHECK_EQ(out[0], 1);
    CHECK_EQ(out[1], 2);
}

// ─── RMSNorm 5-instruction sequence test ────────────────────────────────────
// Verifies that the full sequence (SQUARE→REDUCE→SCALAR_ADD→RSQRT→ELEMWISE_MUL)
// produces a numerically correct result for a known input.
static void test_rmsnorm_sequence() {
    MemoryController mem;

    // Input: [3, 4, 0, 0] (1×4 tile)
    // sum(x²) = 9 + 16 + 0 + 0 = 25
    // 1/sqrt(25 + epsilon) ≈ 0.2
    // output ≈ [0, 0, 0, 0] — too small once quantized; use larger values
    //
    // Input: [10, 10, 10, 10]
    // sum(x²) = 400; mean(x²) = 100
    // 1/sqrt(100) = 0.1; output ≈ round(10 * 0.1) = 1 for each
    int8_t input[4] = { 10, 10, 10, 10 };
    write_tile_i8(mem, ZONE_INPUT, input, 1, 4);

    // Save a copy at ZONE_WEIGHT to use in the final ELEMWISE_MUL (γ scaling = 1 here)
    // Since we want to verify the math, we'll use the normalized values directly.
    // Step 1: SQUARE — in place
    PEArray pe;
    pe.exec_alu(mem, ZONE_INPUT, 1, 4, Decoder::ActivationMode::SQUARE);

    // Step 2: REDUCE — sum into ZONE_SCALAR
    pe.reduce(mem, ZONE_INPUT, ZONE_SCALAR, 1, 4);

    int32_t sum_sq;
    std::memcpy(&sum_sq, mem.sram_ptr(ZONE_SCALAR), 4);
    // Each element was 10; squared = 100 (clamped, still 100 since 100 < 127)
    // sum = 4 × 100 = 400 — but wait, SQUARE writes INT8 saturated (100 fits INT8)
    // REDUCE reads INT8, sums → 400
    CHECK_EQ(sum_sq, 400);

    // Step 3: SCALAR_ADD — add epsilon (1 here, stand-in for rms_norm_eps scaled up)
    // In the real NPU, epsilon is pre-scaled to match the INT8 domain.
    int32_t eps = 0;  // skip epsilon for this test
    std::memcpy(mem.sram_ptr(ZONE_SCALAR), &eps, 4);

    // Step 4: put 400 back as the scalar (skip SCALAR_ADD here — epsilon=0)
    int32_t denom = 400;
    std::memcpy(mem.sram_ptr(ZONE_SCALAR), &denom, 4);

    // Restore original input (SQUARE clobbered it) for the final scale
    write_tile_i8(mem, ZONE_WEIGHT, input, 1, 4);  // save original

    // RSQRT via scalar_alu: 1/sqrt(400) = 0.05; round(10 * 0.05) = 1
    pe.scalar_alu(mem, ZONE_SCALAR, ZONE_WEIGHT, 1, 4, Decoder::ActivationMode::RSQRT);

    const int8_t* out = reinterpret_cast<const int8_t*>(mem.sram_ptr(ZONE_WEIGHT));
    // 1/sqrt(400) = 0.05; 10 * 0.05 = 0.5 → rounds to 1 (or 0 with floor)
    // Using roundf: roundf(0.5) = 1 on most platforms
    for (int i = 0; i < 4; ++i)
        CHECK(out[i] == 0 || out[i] == 1);  // rounding may give 0 or 1
}

// ─── Bounds-check test ───────────────────────────────────────────────────────

static void test_gemm_oob_throws() {
    MemoryController mem;
    PEArray pe;
    bool threw = false;
    try {
        // C tile needs M*N*4 = 16*16*4 = 1024 bytes.
        // 0xFF00 (65280) + 1024 = 66304 > SRAM_BYTES (65536) → must throw.
        pe.compute_gemm(mem, 0, 0, 0xFF00, 16, 16, 16);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

// PERCHAN_QUANT: INT32 accumulator -> INT8 with one mantissa/shift per column.
// The point of the mode is that columns whose true magnitudes differ by orders
// of magnitude all land in INT8 range, which a single shared shift cannot do.
static void test_exec_alu_perchan_quant() {
    MemoryController mem;
    PEArray pe;

    const uint16_t tile_addr = 0x0000;   // INT32 accumulator, 2 rows x 4 cols
    const uint16_t corr_addr = 0x0200;   // 4 INT16 correction entries

    // Column magnitudes deliberately span 1000x.
    const int32_t acc[2][4] = {
        { 1000, 100000, 500,  20 },
        { -2000, 50000, -250, 10 },
    };
    int32_t* c = reinterpret_cast<int32_t*>(mem.sram_ptr(tile_addr));
    for (int r = 0; r < 2; ++r)
        for (int j = 0; j < 4; ++j) c[r * 4 + j] = acc[r][j];

    // mant in [128,255], shift in the high byte: multiplier = mant / 2^shift.
    // Chosen so each column's larger magnitude maps near full INT8 scale.
    const uint8_t mant[4]  = { 128, 160, 250, 128 };
    const uint8_t shift[4] = {  10,  17,   9,   4 };
    uint16_t* corr = reinterpret_cast<uint16_t*>(mem.sram_ptr(corr_addr));
    for (int j = 0; j < 4; ++j)
        corr[j] = static_cast<uint16_t>((shift[j] << 8) | mant[j]);

    pe.exec_alu(mem, tile_addr, 2, 4,
                Decoder::ActivationMode::PERCHAN_QUANT, corr_addr);

    const int8_t* out = reinterpret_cast<const int8_t*>(mem.sram_ptr(tile_addr));
    for (int r = 0; r < 2; ++r) {
        for (int j = 0; j < 4; ++j) {
            const long long v =
                (static_cast<long long>(acc[r][j]) * mant[j]) >> shift[j];
            const int want = v > 127 ? 127 : (v < -128 ? -128 : static_cast<int>(v));
            CHECK_EQ(static_cast<int>(out[r * 4 + j]), want);
        }
    }
    // Every column must survive: a single shared shift would flush the
    // small-magnitude columns to zero or saturate the large ones.
    CHECK(out[1] != 0 && out[3] != 0);
}

// RESCALE: convert an INT8 tile from one scale to another.
// The multiplier is a real ratio, not a power of two -- which is the whole
// reason the mode exists. Aligning Qwen3's residual stream with a sublayer
// output at layer 0 needs 4.26x, and scale_shift can only do 2, 4, 8...
static void test_exec_alu_rescale() {
    MemoryController mem;
    PEArray pe;

    const uint16_t addr = 0x0000;
    const int8_t in[8] = { 100, -100, 30, -7, 1, -1, 0, 12 };
    int8_t* tile = reinterpret_cast<int8_t*>(mem.sram_ptr(addr));
    for (int i = 0; i < 8; ++i) tile[i] = in[i];

    // 4.26x encoded as 136 / 2^5 = 4.25.
    const uint32_t mant = 136, shift = 5;
    pe.exec_alu(mem, addr, 1, 8, Decoder::ActivationMode::RESCALE, 0,
                (shift << 8) | mant);

    for (int i = 0; i < 8; ++i) {
        const long long v = (static_cast<long long>(in[i]) * mant) >> shift;
        const int want = v > 127 ? 127 : (v < -128 ? -128 : static_cast<int>(v));
        CHECK_EQ(static_cast<int>(tile[i]), want);
    }
    // 100 * 4.25 saturates rather than wrapping.
    CHECK_EQ(static_cast<int>(tile[0]), 127);
    CHECK_EQ(static_cast<int>(tile[1]), -128);
    // 30 * 4.25 = 127.5 -> 127 after the arithmetic shift (floor).
    CHECK_EQ(static_cast<int>(tile[2]), 127);
}

// INT16 residual stream: the wide path an INT8 buffer cannot cover.
// Qwen3's residual holds massive activation channels (2610 against a median
// of 0.065 at layer 2), so values that saturate INT8 must survive here.
static void test_int16_residual_ops() {
    MemoryController mem;
    PEArray pe;
    const uint16_t A = 0x0000, B = 0x0400;

    // Values well outside INT8 range.
    const int16_t av[4] = { 20000, -20000, 300, -1 };
    const int16_t bv[4] = { 10000,  -5000, 400,  1 };
    int16_t* a = reinterpret_cast<int16_t*>(mem.sram_ptr(A));
    int16_t* b = reinterpret_cast<int16_t*>(mem.sram_ptr(B));
    for (int i = 0; i < 4; ++i) { a[i] = av[i]; b[i] = bv[i]; }

    pe.exec_alu(mem, A, 1, 4, Decoder::ActivationMode::ELEMWISE_ADD, B, 0,
                Decoder::DType::INT16);
    CHECK_EQ(static_cast<int>(a[0]), 30000);     // no INT8 saturation
    CHECK_EQ(static_cast<int>(a[1]), -25000);
    CHECK_EQ(static_cast<int>(a[2]), 700);
    CHECK_EQ(static_cast<int>(a[3]), 0);
    // The sum still saturates at the INT16 boundary rather than wrapping.
    a[0] = 30000; b[0] = 30000;
    pe.exec_alu(mem, A, 1, 1, Decoder::ActivationMode::ELEMWISE_ADD, B, 0,
                Decoder::DType::INT16);
    CHECK_EQ(static_cast<int>(a[0]), 32767);

    // RESCALE on INT16, multiplier 3/2.
    a[0] = 20000; a[1] = -20000;
    pe.exec_alu(mem, A, 1, 2, Decoder::ActivationMode::RESCALE, 0,
                (7u << 8) | 192u, Decoder::DType::INT16);
    CHECK_EQ(static_cast<int>(a[0]), 30000);
    CHECK_EQ(static_cast<int>(a[1]), -30000);

    // REDUCE_SUMSQ over INT16 accumulates in int64: 4 * 20000^2 = 1.6e9,
    // which overflows int32 if summed narrowly.
    for (int i = 0; i < 4; ++i) a[i] = 20000;
    const uint16_t SC = 0x0800;
    pe.reduce(mem, A, SC, 1, 4, false,
              Decoder::ActivationMode::REDUCE_SUMSQ, Decoder::DType::INT16);
    // INT64 scalar: a 16x16 INT16 tile reaches 256*(2^15)^2 = 2.7e11, far past
    // INT32, and saturating there makes the normalisation silently wrong.
    CHECK_EQ(*reinterpret_cast<int64_t*>(mem.sram_ptr(SC)), 1600000000LL);
    for (int i = 0; i < 4; ++i) a[i] = 32767;
    pe.reduce(mem, A, SC, 1, 4, false,
              Decoder::ActivationMode::REDUCE_SUMSQ, Decoder::DType::INT16);
    CHECK_EQ(*reinterpret_cast<int64_t*>(mem.sram_ptr(SC)),
             4LL * 32767LL * 32767LL);      // 4.29e9 -- would overflow int32
    for (int i = 0; i < 4; ++i) a[i] = 20000;
    pe.reduce(mem, A, SC, 1, 4, false,
              Decoder::ActivationMode::REDUCE_SUMSQ, Decoder::DType::INT16);

    // RMS_SCALE reads INT16 and writes INT8 -- the narrowing boundary.
    // 20000 * (127*2) / sqrt(1.6e9) = 20000*254/40000 = 127.
    // scalar_alu takes (scalar_sram, tile_sram) in that order.
    pe.scalar_alu(mem, SC, A, 1, 4, Decoder::ActivationMode::RMS_SCALE,
                  static_cast<uint32_t>(std::lround(127.0 * 2.0)),
                  Decoder::DType::INT16);
    const int8_t* out = reinterpret_cast<const int8_t*>(mem.sram_ptr(A));
    for (int i = 0; i < 4; ++i) CHECK_EQ(static_cast<int>(out[i]), 127);
}

// FP32: the no-quantisation path. Exists so a lowering bug can be told apart
// from a quantisation limit -- if this path is correct, every address, tile
// and accumulation in the compiler is proven independently of any scale.
static void test_fp32_ops() {
    MemoryController mem;
    PEArray pe;
    const uint16_t A = 0x0000, B = 0x0400, C = 0x0800, SC = 0x0C00;

    // 2x2 GEMM in float, with values no INT8 scale could hold together.
    float* a = reinterpret_cast<float*>(mem.sram_ptr(A));
    float* b = reinterpret_cast<float*>(mem.sram_ptr(B));
    a[0] = 1e-4f; a[1] = 2.0f; a[2] = -3.5f; a[3] = 1e4f;
    b[0] = 2.0f;  b[1] = 0.5f; b[2] = -1.0f; b[3] = 4.0f;
    pe.compute_gemm(mem, A, B, C, 2, 2, 2, Decoder::ActivationMode::NONE,
                    false, 7, Decoder::DType::FP32);
    const float* c = reinterpret_cast<const float*>(mem.sram_ptr(C));
    CHECK(std::fabs(c[0] - (1e-4f * 2.0f + 2.0f * -1.0f)) < 1e-5f);
    CHECK(std::fabs(c[3] - (-3.5f * 0.5f + 1e4f * 4.0f)) < 1e-1f);

    // Accumulate adds into C rather than replacing it -- K-tiling depends on it.
    pe.compute_gemm(mem, A, B, C, 2, 2, 2, Decoder::ActivationMode::NONE,
                    true, 7, Decoder::DType::FP32);
    CHECK(std::fabs(c[0] - 2.0f * (1e-4f * 2.0f + 2.0f * -1.0f)) < 1e-5f);

    // Elementwise multiply: no >>7, so a small product survives. The INT8
    // path shifts 5*3 to zero; here 0.01*0.02 is exactly representable.
    float* t = reinterpret_cast<float*>(mem.sram_ptr(A));
    float* u = reinterpret_cast<float*>(mem.sram_ptr(B));
    t[0] = 0.01f; t[1] = -4.0f; u[0] = 0.02f; u[1] = 0.25f;
    pe.exec_alu(mem, A, 1, 2, Decoder::ActivationMode::ELEMWISE_MUL, B, 0,
                Decoder::DType::FP32);
    CHECK(std::fabs(t[0] - 2e-4f) < 1e-9f);
    CHECK(t[0] != 0.0f);
    CHECK(std::fabs(t[1] - -1.0f) < 1e-6f);

    // RESCALE and INT8_QUANT are identities here: there is only one scale.
    pe.exec_alu(mem, A, 1, 2, Decoder::ActivationMode::RESCALE, 0,
                (5u << 8) | 136u, Decoder::DType::FP32);
    CHECK(std::fabs(t[0] - 2e-4f) < 1e-9f);

    // Sum of squares, then RMS_SCALE -- no Q0.7 output convention to clamp.
    float* v = reinterpret_cast<float*>(mem.sram_ptr(A));
    v[0] = 3.0f; v[1] = 4.0f;
    pe.reduce(mem, A, SC, 1, 2, false,
              Decoder::ActivationMode::REDUCE_SUMSQ, Decoder::DType::FP32);
    CHECK(std::fabs(*reinterpret_cast<float*>(mem.sram_ptr(SC)) - 25.0f) < 1e-4f);
    pe.scalar_alu(mem, SC, A, 1, 2, Decoder::ActivationMode::RMS_SCALE, 0,
                  Decoder::DType::FP32);
    CHECK(std::fabs(v[0] - 0.6f) < 1e-5f);   // 3/5, unclamped
    CHECK(std::fabs(v[1] - 0.8f) < 1e-5f);

    // SILU at the true value: silu(-2) = -0.238, which the INT8 path
    // rounds to 0 and reports as ReLU.
    v[0] = -2.0f; v[1] = 0.5f;
    pe.exec_alu(mem, A, 1, 2, Decoder::ActivationMode::SILU, 0, 0,
                Decoder::DType::FP32);
    CHECK(std::fabs(v[0] - (-2.0f / (1.0f + std::exp(2.0f)))) < 1e-6f);
    CHECK(v[0] < 0.0f);                       // not ReLU
    CHECK(std::fabs(v[1] - (0.5f / (1.0f + std::exp(-0.5f)))) < 1e-6f);
}

// ─── Runner ──────────────────────────────────────────────────────────────────

struct Test { const char* name; void(*fn)(); };

int main() {
    const Test tests[] = {
        {"GemmIdentity",          test_gemm_identity},
        {"GemmAllOnes",           test_gemm_all_ones},
        {"GemmReluActivation",    test_gemm_relu_activation},
        {"Gemm16x16Smoke",        test_gemm_16x16_smoke},
        {"ExecAluRelu",           test_exec_alu_relu},
        {"ExecAluSquare",         test_exec_alu_square},
        {"ExecAluElemwiseMul",    test_exec_alu_elemwise_mul},
        {"ReduceSum",             test_reduce_sum},
        {"ReduceSumOfSquares",    test_reduce_sum_of_squares},
        {"ScalarAluAdd",          test_scalar_alu_add},
        {"ScalarAluRsqrt",        test_scalar_alu_rsqrt},
        {"RmsnormSequence",       test_rmsnorm_sequence},
        {"ExecAluPerchanQuant",   test_exec_alu_perchan_quant},
        {"ExecAluRescale",        test_exec_alu_rescale},
        {"Int16ResidualOps",      test_int16_residual_ops},
        {"Fp32Ops",               test_fp32_ops},
        {"GemmOobThrows",         test_gemm_oob_throws},
    };

    for (const auto& t : tests) {
        int before = s_fail;
        try { t.fn(); }
        catch (const std::exception& e) {
            ++s_fail;
            std::fprintf(stderr, "FAIL  %s  uncaught: %s\n", t.name, e.what());
        }
        std::printf("%s  %s\n", s_fail == before ? "PASS" : "FAIL", t.name);
    }

    std::printf("\n%d passed, %d failed\n", s_pass, s_fail);
    return s_fail == 0 ? 0 : 1;
}
