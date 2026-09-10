#include "../inc/pe_array.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <cstring>
#include <stdexcept>

#ifdef CHIP_SIM_USE_NEON
#include <arm_neon.h>
#endif
#ifdef CHIP_SIM_USE_AVX2
#include <immintrin.h>
#endif

// ─── Bounds helpers ───────────────────────────────────────────────────────────

static void check_sram(uint16_t base, uint32_t len, const char* who) {
    if (static_cast<uint32_t>(base) + len > SRAM_BYTES)
        throw std::out_of_range(std::string(who) + ": SRAM window out of range");
}

// Softmax exponent temperature -- see EXP_SUB in scalar_alu().
static constexpr float SOFTMAX_TEMPERATURE = 64.0f;

// Sentinel marking a masked score. EXP_SUB maps it to exactly 0 probability.
//
// A sentinel rather than an additive -inf penalty: adding -128 to a score
// only reaches this value when the score is already <= 0 (clamp8 saturates),
// and even where it lands, exp((-128 - max)/T) at T=64 is roughly 13% of the
// peak -- masked positions would leak substantial attention weight. An exact
// mask needs a value EXP_SUB special-cases, not a large negative number.
static constexpr int8_t MASKED_SENTINEL = -128;

static int8_t clamp8(int32_t v) {
    return static_cast<int8_t>(std::max(-128, std::min(127, v)));
}

// ─── COMPUTE_GEMM ─────────────────────────────────────────────────────────────

void PEArray::compute_gemm(MemoryController& mem,
                           uint16_t a_sram, uint16_t b_sram, uint16_t c_sram,
                           uint16_t M, uint16_t N, uint16_t K,
                           Decoder::ActivationMode activation,
                           bool accumulate,
                           uint8_t scale_shift,
                           Decoder::DType dtype)
{
    if (dtype == Decoder::DType::FP32) {
        // No quantisation: A, B and C are all float, so there is no
        // accumulator width to choose, no requantisation step, and no scale
        // to track. This is the path that establishes whether the compiler's
        // addressing and tiling are correct independently of any of that.
        const uint32_t na = static_cast<uint32_t>(M) * K;
        const uint32_t nb = static_cast<uint32_t>(K) * N;
        const uint32_t nc = static_cast<uint32_t>(M) * N;
        check_sram(a_sram, na * 4, "compute_gemm A (fp32)");
        check_sram(b_sram, nb * 4, "compute_gemm B (fp32)");
        check_sram(c_sram, nc * 4, "compute_gemm C (fp32)");
        const float* Af = reinterpret_cast<const float*>(mem.sram_ptr(a_sram));
        const float* Bf = reinterpret_cast<const float*>(mem.sram_ptr(b_sram));
        float*       Cf = reinterpret_cast<float*>(mem.sram_ptr(c_sram));
        if (!accumulate)
            std::memset(Cf, 0, static_cast<std::size_t>(nc) * sizeof(float));
        for (uint16_t i = 0; i < M; ++i)
            for (uint16_t k = 0; k < K; ++k) {
                const float a = Af[i * K + k];
                if (a == 0.0f) continue;
                for (uint16_t j = 0; j < N; ++j)
                    Cf[i * N + j] += a * Bf[k * N + j];
            }
        return;
    }

    check_sram(a_sram, static_cast<uint32_t>(M) * K,     "compute_gemm A");
    check_sram(b_sram, static_cast<uint32_t>(K) * N,     "compute_gemm B");
    check_sram(c_sram, static_cast<uint32_t>(M) * N * 4, "compute_gemm C");

    const int8_t*  A = reinterpret_cast<const int8_t*>(mem.sram_ptr(a_sram));
    const int8_t*  B = reinterpret_cast<const int8_t*>(mem.sram_ptr(b_sram));
          int32_t* C = reinterpret_cast<      int32_t*>(mem.sram_ptr(c_sram));

    // Zero accumulators only when not accumulating into an existing C tile.
    if (!accumulate)
        std::memset(C, 0, static_cast<std::size_t>(M) * N * sizeof(int32_t));

#if defined(CHIP_SIM_USE_NEON)
    // Neon path using vdotq_s32 (FEAT_DotProd, available on Cortex-A76).
    //
    // vdotq_s32(acc, a, b) computes four independent 4-element dot products:
    //   acc[i] += a[4i]*b[4i] + a[4i+1]*b[4i+1] + a[4i+2]*b[4i+2] + a[4i+3]*b[4i+3]
    //
    // To compute four output columns (n4..n4+3) of row m simultaneously, the
    // data layout inside the 16-lane vectors must be:
    //
    //   a_buf: {A[m][k], A[m][k+1], A[m][k+2], A[m][k+3]} repeated 4× (one per output col)
    //   b_buf: {B[k][n4+0], B[k+1][n4+0], B[k+2][n4+0], B[k+3][n4+0],   ← col 0
    //           B[k][n4+1], B[k+1][n4+1], B[k+2][n4+1], B[k+3][n4+1],   ← col 1
    //           B[k][n4+2], ...                                             ← col 2
    //           B[k][n4+3], ...}                                            ← col 3
    //
    // This gives: acc[j] += sum_{ki=0..3} A[m][k+ki] * B[k+ki][n4+j] = partial C[m][n4+j]
    for (uint16_t m = 0; m < M; ++m) {
        for (uint16_t n4 = 0; n4 + 4 <= N; n4 += 4) {
            int32x4_t acc = vdupq_n_s32(0);
            uint16_t k = 0;
            for (; k + 4 <= K; k += 4) {
                // a: broadcast A[m][k..k+3] to all 4 lane-groups
                int8_t a_buf[16];
                for (int j = 0; j < 4; ++j)
                    for (int ki = 0; ki < 4; ++ki)
                        a_buf[j * 4 + ki] = A[m * K + k + ki];

                // b: for each output column j, store B[k..k+3][n4+j] contiguously
                int8_t b_buf[16];
                for (int j = 0; j < 4; ++j)
                    for (int ki = 0; ki < 4; ++ki)
                        b_buf[j * 4 + ki] = B[(k + ki) * N + (n4 + j)];

                int8x16_t a_vec = vld1q_s8(a_buf);
                int8x16_t b_vec = vld1q_s8(b_buf);
                acc = vdotq_s32(acc, a_vec, b_vec);
            }
            int32_t acc_arr[4];
            vst1q_s32(acc_arr, acc);
            for (int c = 0; c < 4; ++c)
                C[m * N + n4 + c] += acc_arr[c];
            // Scalar tail for K values not divisible by 4
            for (; k < K; ++k)
                for (int c = 0; c < 4; ++c)
                    C[m * N + n4 + c] += static_cast<int32_t>(A[m * K + k])
                                       * static_cast<int32_t>(B[k * N + n4 + c]);
        }
        // Scalar tail for N columns not covered by the n4 loop
        for (uint16_t n = (N / 4) * 4; n < N; ++n)
            for (uint16_t k = 0; k < K; ++k)
                C[m * N + n] += static_cast<int32_t>(A[m * K + k])
                              * static_cast<int32_t>(B[k * N + n]);
    }
#elif defined(CHIP_SIM_USE_AVX2)
    // AVX2 path. Eight output columns at a time: broadcast one A element and
    // multiply it across eight widened B values, accumulating into eight
    // INT32 lanes.
    //
    // Widen-then-multiply (cvtepi8_epi32 + mullo_epi32) rather than one of the
    // fused INT8 dot-product instructions: _mm256_maddubs_epi16 treats its
    // first operand as UNSIGNED and saturates its INT16 intermediate, so it
    // would neither accept signed activations nor stay exact; VNNI's
    // dpbusd_epi32 is not available on this target. Widening keeps the
    // arithmetic bit-identical to the scalar path, which matters because the
    // whole E2E suite validates against scalar semantics.
    //
    // Integer addition is associative and these products cannot overflow
    // INT32 (|int8 * int8| <= 16384, and K <= 255), so lane-wise accumulation
    // reorders the sum without changing it.
    for (uint16_t m = 0; m < M; ++m) {
        uint16_t n = 0;
        for (; n + 8 <= N; n += 8) {
            __m256i acc = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&C[m * N + n]));
            for (uint16_t k = 0; k < K; ++k) {
                const __m128i b8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(&B[k * N + n]));
                const __m256i b  = _mm256_cvtepi8_epi32(b8);
                const __m256i a  = _mm256_set1_epi32(
                    static_cast<int32_t>(A[m * K + k]));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(a, b));
            }
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&C[m * N + n]), acc);
        }
        // Scalar tail for the columns the 8-wide loop could not cover.
        for (; n < N; ++n)
            for (uint16_t k = 0; k < K; ++k)
                C[m * N + n] += static_cast<int32_t>(A[m * K + k])
                              * static_cast<int32_t>(B[k * N + n]);
    }
#else
    // Scalar reference: straightforward O(MNK) loop.
    for (uint16_t m = 0; m < M; ++m)
        for (uint16_t n = 0; n < N; ++n)
            for (uint16_t k = 0; k < K; ++k)
                C[m * N + n] += static_cast<int32_t>(A[m * K + k])
                              * static_cast<int32_t>(B[k * N + n]);
#endif

    // Post-GEMM activation applied element-wise to INT32 accumulators.
    const uint32_t elems = static_cast<uint32_t>(M) * N;
    switch (activation) {
    case Decoder::ActivationMode::NONE:
        break;
    case Decoder::ActivationMode::RELU:
        for (uint32_t i = 0; i < elems; ++i)
            if (C[i] < 0) C[i] = 0;
        break;
    case Decoder::ActivationMode::INT8_QUANT: {
        // Narrow INT32 → INT8 by right-shifting scale_shift bits and saturating.
        // v1.5: scale_shift is caller-supplied; v1 default is 7 (divide by 128).
        int8_t* out = reinterpret_cast<int8_t*>(C);
        for (uint32_t i = 0; i < elems; ++i)
            out[i] = clamp8(C[i] >> scale_shift);
        break;
    }
    default:
        // Other activation modes are handled by a subsequent EXEC_ALU instruction.
        break;
    }
}

// ─── EXEC_ALU ─────────────────────────────────────────────────────────────────

void PEArray::exec_alu(MemoryController& mem,
                       uint16_t sram_addr, uint16_t M, uint16_t N,
                       Decoder::ActivationMode mode,
                       uint16_t b_sram,
                       uint32_t params,
                       Decoder::DType dtype)
{
    const uint32_t elems = static_cast<uint32_t>(M) * N;

    // FP32 tiles. Several modes exist only to manage quantisation and have
    // nothing to do here: RESCALE converts between two scales, INT8_QUANT
    // narrows an accumulator, and LUT precomputes a nonlinearity for a known
    // scale. In float there is one scale -- the exponent each value carries --
    // so all three are identity or a direct evaluation.
    if (dtype == Decoder::DType::FP32) {
        check_sram(sram_addr, elems * 4, "exec_alu tile (fp32)");
        float* t = reinterpret_cast<float*>(mem.sram_ptr(sram_addr));
        switch (mode) {
        case Decoder::ActivationMode::RELU:
            for (uint32_t i = 0; i < elems; ++i) t[i] = std::max(0.0f, t[i]);
            break;
        case Decoder::ActivationMode::SILU:
        case Decoder::ActivationMode::LUT:      // no table needed in float
            for (uint32_t i = 0; i < elems; ++i)
                t[i] = t[i] / (1.0f + std::exp(-t[i]));
            break;
        case Decoder::ActivationMode::ELEMWISE_MUL: {
            check_sram(b_sram, elems * 4, "exec_alu ELEMWISE_MUL B (fp32)");
            const float* B = reinterpret_cast<const float*>(mem.sram_ptr(b_sram));
            for (uint32_t i = 0; i < elems; ++i) t[i] *= B[i];
            break;
        }
        case Decoder::ActivationMode::ELEMWISE_ADD: {
            check_sram(b_sram, elems * 4, "exec_alu ELEMWISE_ADD B (fp32)");
            const float* B = reinterpret_cast<const float*>(mem.sram_ptr(b_sram));
            for (uint32_t i = 0; i < elems; ++i) t[i] += B[i];
            break;
        }
        case Decoder::ActivationMode::TRANSPOSE: {
            if (M != N)
                throw std::invalid_argument("exec_alu TRANSPOSE needs a square tile");
            for (uint16_t r = 0; r < M; ++r)
                for (uint16_t c = r + 1; c < N; ++c)
                    std::swap(t[r * N + c], t[c * N + r]);
            break;
        }
        case Decoder::ActivationMode::CAUSAL_MASK:
            for (uint16_t r = 0; r < M; ++r)
                for (uint16_t c = r + 1; c < N; ++c)
                    t[r * N + c] = -std::numeric_limits<float>::infinity();
            break;
        case Decoder::ActivationMode::NONE:
        case Decoder::ActivationMode::RESCALE:      // identity: one scale only
        case Decoder::ActivationMode::INT8_QUANT:   // identity: nothing to narrow
            break;
        default:
            throw std::invalid_argument(
                "exec_alu: activation mode not supported for FP32 tiles");
        }
        return;
    }

    // INT16 tiles: only the modes the residual stream needs are widened.
    // ELEMWISE_ADD accumulates the residual itself, and RESCALE is how an
    // INT8 sublayer output enters it -- those two plus the INT16 REDUCE that
    // feeds RMSNorm are the whole INT16 surface. Everything else stays INT8,
    // which is what the rest of the pipeline still uses.
    if (dtype == Decoder::DType::INT16) {
        check_sram(sram_addr, elems * 2, "exec_alu tile (int16)");
        int16_t* t16 = reinterpret_cast<int16_t*>(mem.sram_ptr(sram_addr));
        auto clamp16 = [](int64_t v) -> int16_t {
            return static_cast<int16_t>(std::max<int64_t>(-32768,
                                        std::min<int64_t>(32767, v)));
        };
        switch (mode) {
        case Decoder::ActivationMode::ELEMWISE_ADD: {
            check_sram(b_sram, elems * 2, "exec_alu ELEMWISE_ADD b (int16)");
            const int16_t* b = reinterpret_cast<const int16_t*>(mem.sram_ptr(b_sram));
            for (uint32_t i = 0; i < elems; ++i)
                t16[i] = clamp16(static_cast<int64_t>(t16[i]) + b[i]);
            break;
        }
        case Decoder::ActivationMode::RESCALE: {
            const int64_t mant  = params & 0xFF;
            const int64_t shift = (params >> 8) & 0xFF;
            for (uint32_t i = 0; i < elems; ++i)
                t16[i] = clamp16((static_cast<int64_t>(t16[i]) * mant) >> shift);
            break;
        }
        case Decoder::ActivationMode::INT8_QUANT: {
            // Widen an INT8 tile in place into INT16, so a sublayer output can
            // join the INT16 residual. Read backwards: the destination is
            // twice the width of the source and overlaps it.
            //
            // params[7:0] is a LEFT shift applied during the widen. Without it
            // a widened value keeps its INT8 magnitude (max 127) and wastes
            // the INT16 range, and the follow-up RESCALE then needs a
            // multiplier that can exceed 255 -- more than its 8-bit mantissa
            // can express. Calibration puts these ratios as high as 348.
            // Shifting left by 8 during the widen absorbs a factor of 256 and
            // brings every measured ratio back inside RESCALE's range.
            const uint32_t lsh = params & 0xFF;
            const int8_t* src = reinterpret_cast<const int8_t*>(mem.sram_ptr(sram_addr));
            for (uint32_t i = elems; i-- > 0; ) {
                const int32_t v = static_cast<int32_t>(src[i]) << lsh;
                t16[i] = static_cast<int16_t>(std::max(-32768, std::min(32767, v)));
            }
            break;
        }
        default:
            throw std::invalid_argument(
                "exec_alu: activation mode not supported for INT16 tiles");
        }
        return;
    }

    check_sram(sram_addr, elems, "exec_alu tile");

    int8_t* tile = reinterpret_cast<int8_t*>(mem.sram_ptr(sram_addr));

    switch (mode) {
    case Decoder::ActivationMode::RELU:
        for (uint32_t i = 0; i < elems; ++i)
            if (tile[i] < 0) tile[i] = 0;
        break;

    case Decoder::ActivationMode::SILU:
        for (uint32_t i = 0; i < elems; ++i) {
            float x = static_cast<float>(tile[i]);
            float s = x / (1.0f + std::exp(-x));  // sigmoid(x)*x
            tile[i] = clamp8(static_cast<int32_t>(std::roundf(s)));
        }
        break;

    case Decoder::ActivationMode::SQUARE:
        for (uint32_t i = 0; i < elems; ++i) {
            int32_t sq = static_cast<int32_t>(tile[i]) * static_cast<int32_t>(tile[i]);
            tile[i] = clamp8(sq);
        }
        break;

    case Decoder::ActivationMode::RSQRT:
        for (uint32_t i = 0; i < elems; ++i) {
            float x = static_cast<float>(tile[i]);
            float r = (x > 0.0f) ? (1.0f / std::sqrt(x)) : 0.0f;
            tile[i] = clamp8(static_cast<int32_t>(std::roundf(r)));
        }
        break;

    case Decoder::ActivationMode::CAUSAL_MASK:
        // Mark the strictly-upper triangle as masked: query row r may not
        // attend to key column c > r. Applied to the DIAGONAL score tile
        // only -- tiles entirely above the diagonal are skipped by the
        // compiler rather than computed and masked, and tiles below it are
        // fully visible.
        for (uint16_t r = 0; r < M; ++r)
            for (uint16_t c = r + 1; c < N; ++c)
                tile[r * N + c] = MASKED_SENTINEL;
        break;

    case Decoder::ActivationMode::TRANSPOSE: {
        // In-place transpose of a SQUARE tile. Only square is supported: a
        // non-square transpose changes the tile's shape, and the instruction
        // carries no way to express the new one. Attention always transposes
        // 16x16 tiles, so this is sufficient.
        //
        // Needed because COMPUTE_GEMM computes A @ B, but attention scores are
        // Q @ K^T. Neither LOAD_TILE nor STORE_TILE can transpose (both copy
        // contiguous rows), so without this the ISA cannot express Q·K^T at all.
        if (M != N)
            throw std::invalid_argument("exec_alu TRANSPOSE: tile must be square");
        for (uint16_t r = 0; r < M; ++r)
            for (uint16_t c = r + 1; c < N; ++c)
                std::swap(tile[r * N + c], tile[c * N + r]);
        break;
    }

    case Decoder::ActivationMode::ELEMWISE_ADD: {
        // tile_A += tile_B, element-wise. B's address travels in the stride
        // field, same as ELEMWISE_MUL. No shift: unlike the multiply, a sum of
        // two INT8 values needs no rescaling to stay in range -- only the
        // saturating clamp. This is what a transformer residual connection
        // (x = x + sublayer(x)) lowers to; without it the ISA could multiply
        // two tiles but not add them.
        check_sram(b_sram, elems, "exec_alu ELEMWISE_ADD B");
        const int8_t* B = reinterpret_cast<const int8_t*>(mem.sram_ptr(b_sram));
        for (uint32_t i = 0; i < elems; ++i)
            tile[i] = clamp8(static_cast<int32_t>(tile[i]) +
                             static_cast<int32_t>(B[i]));
        break;
    }

    case Decoder::ActivationMode::ELEMWISE_MUL: {
        check_sram(b_sram, elems, "exec_alu ELEMWISE_MUL B");
        const int8_t* B = reinterpret_cast<const int8_t*>(mem.sram_ptr(b_sram));
        // params[7:0] is the right shift, defaulting to 7 when unset so a
        // v1-encoded instruction is unchanged. A fixed >>7 assumes both
        // operands occupy the full INT8 range: when they do not, small
        // products vanish -- 5 * 3 = 15 shifts to zero. Measured on Qwen3's
        // SwiGLU product that left the MLP output with 52% median relative
        // error against the same computation done in full precision. The
        // compiler picks the shift from the product's calibrated peak.
        const uint32_t emul_sh = (params & 0xFF) ? (params & 0xFF) : 7u;
        for (uint32_t i = 0; i < elems; ++i) {
            int32_t prod = static_cast<int32_t>(tile[i]) * static_cast<int32_t>(B[i]);
            tile[i] = clamp8(prod >> emul_sh);
        }
        break;
    }

    case Decoder::ActivationMode::LUT: {
        // 256-byte table at b_sram, indexed by the code biased into [0,255].
        // Built at compile time for this buffer's scale, so the function is
        // applied to the true value rather than to the integer code.
        check_sram(b_sram, 256, "exec_alu LUT table");
        const int8_t* table = reinterpret_cast<const int8_t*>(mem.sram_ptr(b_sram));
        for (uint32_t i = 0; i < elems; ++i)
            tile[i] = table[static_cast<int32_t>(tile[i]) + 128];
        break;
    }

    case Decoder::ActivationMode::RESCALE: {
        // Convert an INT8 tile from one scale to another: the multiplier is
        // mant/2^shift, a real ratio rather than a power of two.
        //
        // This exists because an INT8 buffer's scale -- what one integer step
        // is worth -- is a genuine quantity, and two buffers that meet at an
        // ELEMWISE_ADD must agree on it. At layer 0 of Qwen3 the residual
        // stream and o_proj's output differ by 4.26x, which scale_shift's
        // power-of-two field cannot express, so the add silently mixed two
        // different units. See the scale-bookkeeping section of the spec.
        const int32_t mant  = static_cast<int32_t>(params & 0xFF);
        const int32_t shift = static_cast<int32_t>((params >> 8) & 0xFF);
        for (uint32_t i = 0; i < elems; ++i) {
            const int64_t v = static_cast<int64_t>(tile[i]) * mant;
            tile[i] = clamp8(static_cast<int32_t>(v >> shift));
        }
        break;
    }

    case Decoder::ActivationMode::PERCHAN_QUANT: {
        // INT32 accumulator -> INT8 with one scale per OUTPUT CHANNEL.
        //
        // The tile at sram_addr is INT32 (a GEMM accumulator left unquantised
        // by activation=NONE), laid out M rows x N columns. b_sram points at
        // N INT16 correction entries, one per column: low byte is an 8-bit
        // mantissa in [128,255], high byte is a right-shift count.
        //
        //     out[r][c] = clamp8((acc[r][c] * mant[c]) >> shift[c])
        //
        // Mantissa and shift together because neither alone suffices: a
        // per-tensor scale collapses the model (cos -0.23 at layer 27) and
        // rounding each channel's scale to a power of two -- the cheap
        // alternative -- still fails (rank 6223 vs 0). 8 bits of mantissa
        // with a per-channel shift matches exact float.
        //
        // The product is taken in int64: acc can reach ~2^27 for K=1024 and
        // mant up to 255 adds 8 more bits, which overflows int32.
        const int32_t* acc = reinterpret_cast<const int32_t*>(mem.sram_ptr(sram_addr));
        check_sram(sram_addr, elems * 4, "exec_alu PERCHAN_QUANT tile");
        check_sram(b_sram,    static_cast<uint32_t>(N) * 2, "exec_alu PERCHAN_QUANT table");
        const uint16_t* corr = reinterpret_cast<const uint16_t*>(mem.sram_ptr(b_sram));

        // Read the whole accumulator before writing, since the INT8 result is
        // written back over the INT32 source in place and would otherwise
        // overwrite entries this loop has not read yet.
        std::vector<int8_t> tmp(elems);
        for (uint32_t r = 0; r < M; ++r) {
            for (uint32_t c = 0; c < N; ++c) {
                const uint16_t e     = corr[c];
                const int32_t  mant  = static_cast<int32_t>(e & 0xFF);
                const int32_t  shift = static_cast<int32_t>((e >> 8) & 0xFF);
                const int64_t  v     = static_cast<int64_t>(acc[r * N + c]) * mant;
                tmp[r * N + c] = clamp8(static_cast<int32_t>(v >> shift));
            }
        }
        int8_t* out = reinterpret_cast<int8_t*>(mem.sram_ptr(sram_addr));
        for (uint32_t i = 0; i < elems; ++i)
            out[i] = tmp[i];
        break;
    }

    case Decoder::ActivationMode::SCALAR_ADD: {
        // Single INT8 scalar at b_sram
        check_sram(b_sram, 1, "exec_alu SCALAR_ADD scalar");
        int8_t scalar = *reinterpret_cast<const int8_t*>(mem.sram_ptr(b_sram));
        for (uint32_t i = 0; i < elems; ++i)
            tile[i] = clamp8(static_cast<int32_t>(tile[i]) + static_cast<int32_t>(scalar));
        break;
    }

    case Decoder::ActivationMode::INT8_QUANT:
        for (uint32_t i = 0; i < elems; ++i)
            tile[i] = clamp8(static_cast<int32_t>(tile[i]) >> 7);
        break;

    default:
        break;
    }
}

// ─── REDUCE ───────────────────────────────────────────────────────────────────

void PEArray::reduce(MemoryController& mem,
                     uint16_t tile_sram, uint16_t scalar_sram,
                     uint16_t M, uint16_t N,
                     bool accumulate,
                     Decoder::ActivationMode mode,
                     Decoder::DType dtype)
{
    if (dtype == Decoder::DType::FP32) {
        const uint32_t n = static_cast<uint32_t>(M) * N;
        check_sram(tile_sram, n * 4, "reduce tile (fp32)");
        check_sram(scalar_sram, 4,   "reduce scalar (fp32)");
        const float* t = reinterpret_cast<const float*>(mem.sram_ptr(tile_sram));
        float* out = reinterpret_cast<float*>(mem.sram_ptr(scalar_sram));
        if (mode == Decoder::ActivationMode::REDUCE_MAX) {
            float best = t[0];
            for (uint32_t i = 1; i < n; ++i) best = std::max(best, t[i]);
            *out = accumulate ? std::max(*out, best) : best;
            return;
        }
        // Sum in double: a 1024-wide row of squares loses low-order terms in
        // float32, and RMSNorm's divisor is exactly that sum.
        double total = 0.0;
        for (uint32_t i = 0; i < n; ++i)
            total += (mode == Decoder::ActivationMode::REDUCE_SUMSQ)
                   ? static_cast<double>(t[i]) * t[i] : t[i];
        *out = static_cast<float>((accumulate ? *out : 0.0f) + total);
        return;
    }

    if (dtype == Decoder::DType::INT16) {
        // The residual stream is INT16, so RMSNorm's sum of squares has to be
        // taken over INT16. int64 accumulation: 1024 elements of (2^15)^2 is
        // 2^40, well past int32.
        const uint32_t n = static_cast<uint32_t>(M) * N;
        check_sram(tile_sram,   n * 2, "reduce tile (int16)");
        // The scalar is INT64 here, not INT32. A 16x16 tile of INT16 values
        // sums squares up to 256 * (2^15)^2 = 2.7e11, and a full row spans
        // several tiles -- INT32 saturates at 2.1e9 and the normalisation
        // silently comes out wrong. Wide accumulators are what real hardware
        // uses for exactly this reason.
        check_sram(scalar_sram, 8, "reduce scalar (int64)");
        const int16_t* t = reinterpret_cast<const int16_t*>(mem.sram_ptr(tile_sram));
        int64_t* out = reinterpret_cast<int64_t*>(mem.sram_ptr(scalar_sram));
        int64_t total = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const int64_t x = t[i];
            total += (mode == Decoder::ActivationMode::REDUCE_SUMSQ) ? x * x : x;
        }
        *out = (accumulate ? *out : 0) + total;
        return;
    }

    check_sram(tile_sram,   static_cast<uint32_t>(M) * N, "reduce tile");
    check_sram(scalar_sram, 4,                             "reduce scalar");

    const int8_t* tile = reinterpret_cast<const int8_t*>(mem.sram_ptr(tile_sram));

    const uint32_t elems = static_cast<uint32_t>(M) * N;
    int32_t* scalar = reinterpret_cast<int32_t*>(mem.sram_ptr(scalar_sram));

    if (mode == Decoder::ActivationMode::REDUCE_MAX) {
        // Max instead of sum. Softmax subtracts the row maximum before
        // exponentiating, so this is the reduction that makes EXP_SUB stable.
        // Accumulate here means "max with what is already there", which is the
        // natural extension for spanning more tiles than one.
        int32_t best = static_cast<int32_t>(tile[0]);
        for (uint32_t i = 1; i < elems; ++i)
            best = std::max(best, static_cast<int32_t>(tile[i]));
        *scalar = accumulate ? std::max(*scalar, best) : best;
        return;
    }

    // REDUCE_SUMSQ squares in INT32 rather than reading a tile that
    // EXEC_ALU SQUARE already clamped to INT8. That clamp saturates for
    // |x| >= 12 -- on real Qwen3 activations 28% of elements hit the
    // ceiling -- so the sum of squares it produced bore little relation to
    // the true one. Squaring inside the reduction removes the round trip.
    int32_t total = 0;
    if (mode == Decoder::ActivationMode::REDUCE_SUMSQ) {
        for (uint32_t i = 0; i < elems; ++i) {
            const int32_t x = static_cast<int32_t>(tile[i]);
            total += x * x;
        }
    } else {
        for (uint32_t i = 0; i < elems; ++i)
            total += static_cast<int32_t>(tile[i]);
    }
    *scalar = accumulate ? (*scalar + total) : total;
}

// ─── SCALAR_ALU ───────────────────────────────────────────────────────────────

void PEArray::scalar_alu(MemoryController& mem,
                         uint16_t scalar_sram, uint16_t tile_sram,
                         uint16_t M, uint16_t N,
                         Decoder::ActivationMode mode,
                         uint32_t scale_num,
                         Decoder::DType dtype)
{
    if (dtype == Decoder::DType::FP32) {
        const uint32_t n = static_cast<uint32_t>(M) * N;
        check_sram(tile_sram, n * 4, "scalar_alu tile (fp32)");
        check_sram(scalar_sram, 4,   "scalar_alu scalar (fp32)");
        float* t = reinterpret_cast<float*>(mem.sram_ptr(tile_sram));
        const float sc = *reinterpret_cast<const float*>(mem.sram_ptr(scalar_sram));
        switch (mode) {
        case Decoder::ActivationMode::RMS_SCALE:
            // RMSNorm divides by the ROOT MEAN square, so the row length is
            // needed as well as the sum of squares: x/sqrt(sum/N), not
            // x/sqrt(sum). scale_num carries N here (the INT8 path instead
            // folds N and the Q0.7 output scale into one constant, which it
            // can do because its output scale is fixed).
            //
            // No Q0.7 convention to violate either: the result keeps its own
            // magnitude, which is exactly what the INT8 path could not do --
            // its normalised output clamped whenever the peak exceeded 1 RMS.
            if (sc > 0.0f) {
                const float inv = scale_num
                    ? std::sqrt(static_cast<float>(scale_num) / sc)
                    : 1.0f / std::sqrt(sc);
                for (uint32_t i = 0; i < n; ++i) t[i] *= inv;
            }
            break;
        case Decoder::ActivationMode::EXP_SUB: {
            // The scores are true values here, so there is no quantisation
            // temperature to undo -- but attention's own 1/sqrt(head_dim)
            // still has to be applied, and this is where it goes. scale_num
            // carries it as Q8.8; unset means 1.
            const float T = scale_num ? (static_cast<float>(scale_num) / 256.0f)
                                      : 1.0f;
            for (uint32_t i = 0; i < n; ++i)
                t[i] = std::isfinite(t[i]) ? std::exp((t[i] - sc) / T) : 0.0f;
            break;
        }
        case Decoder::ActivationMode::SCALAR_DIV:
            if (sc != 0.0f)
                for (uint32_t i = 0; i < n; ++i) t[i] /= sc;
            break;
        case Decoder::ActivationMode::SCALAR_ADD:
            for (uint32_t i = 0; i < n; ++i) t[i] += sc;
            break;
        default:
            throw std::invalid_argument(
                "scalar_alu: mode not supported for FP32 tiles");
        }
        return;
    }

    if (dtype == Decoder::DType::INT16 &&
        mode == Decoder::ActivationMode::RMS_SCALE) {
        // Reads the INT16 residual, writes the INT8 Q0.7 normalised result --
        // RMSNorm is the boundary where the wide stream narrows again.
        const uint32_t n = static_cast<uint32_t>(M) * N;
        check_sram(tile_sram, n * 2, "scalar_alu tile (int16)");
        check_sram(scalar_sram, 8, "scalar_alu scalar (int64)");
        const int16_t* src = reinterpret_cast<const int16_t*>(mem.sram_ptr(tile_sram));
        // Matches the INT64 scalar REDUCE_SUMSQ writes in the INT16 path.
        const int64_t scalar = *reinterpret_cast<const int64_t*>(mem.sram_ptr(scalar_sram));
        std::vector<int8_t> tmp(n, 0);
        if (scalar > 0) {
            const float inv = static_cast<float>(scale_num)
                            / std::sqrt(static_cast<double>(scalar));
            for (uint32_t i = 0; i < n; ++i)
                tmp[i] = clamp8(static_cast<int32_t>(
                    std::roundf(static_cast<float>(src[i]) * inv)));
        }
        int8_t* dst = reinterpret_cast<int8_t*>(mem.sram_ptr(tile_sram));
        for (uint32_t i = 0; i < n; ++i) dst[i] = tmp[i];
        return;
    }

    const uint32_t elems = static_cast<uint32_t>(M) * N;
    check_sram(scalar_sram, 4,     "scalar_alu scalar");
    check_sram(tile_sram,   elems, "scalar_alu tile");

    int32_t  scalar = *reinterpret_cast<const int32_t*>(mem.sram_ptr(scalar_sram));
    int8_t*  tile   = reinterpret_cast<int8_t*>(mem.sram_ptr(tile_sram));

    switch (mode) {
    case Decoder::ActivationMode::SCALAR_ADD:
        for (uint32_t i = 0; i < elems; ++i)
            tile[i] = clamp8(static_cast<int32_t>(tile[i]) + scalar);
        break;

    case Decoder::ActivationMode::RSQRT: {
        // Compute 1/√scalar (float), then scale each tile element.
        float inv_sqrt = (scalar > 0) ? (1.0f / std::sqrt(static_cast<float>(scalar))) : 0.0f;
        for (uint32_t i = 0; i < elems; ++i)
            tile[i] = clamp8(static_cast<int32_t>(
                std::roundf(static_cast<float>(tile[i]) * inv_sqrt)));
        break;
    }

    case Decoder::ActivationMode::ELEMWISE_MUL:
        for (uint32_t i = 0; i < elems; ++i)
            tile[i] = clamp8((static_cast<int32_t>(tile[i]) * scalar) >> 7);
        break;

    case Decoder::ActivationMode::EXP_SUB: {
        // exp((x - scalar) / T), scaled to INT8. `scalar` is the row maximum,
        // so the largest input maps to exp(0) = 1 -> 127 and everything else
        // decays from there. Subtracting the max is what keeps this from
        // overflowing, and is why REDUCE_MAX exists.
        //
        // T = SOFTMAX_TEMPERATURE is a fixed design choice, not a derived
        // constant: INT8 scores span [-128, 127], so after subtracting the max
        // the input range is [-255, 0]. T must be large enough that a
        // moderately-below-max score does not flush straight to zero. T = 64
        // keeps roughly the top ~4x of the range distinguishable. INT8 softmax
        // is lossy regardless; this trades tail resolution for peak resolution.
        for (uint32_t i = 0; i < elems; ++i) {
            if (tile[i] == MASKED_SENTINEL) {
                tile[i] = 0;          // exactly zero probability, no leakage
                continue;
            }
            // Temperature from scale_num as Q8.8, falling back to the fixed
            // SOFTMAX_TEMPERATURE when unset. T is not a free constant: the
            // score codes mean `code * s_scores`, so exponentiating the true
            // score requires T = 1/s_scores. A hardcoded 64 is correct only
            // if one score step happens to be worth 1/64, which nothing
            // establishes -- the same unstated-magnitude assumption that made
            // Q0.7 wrong for RMSNorm output and >>7 wrong for the SwiGLU
            // product.
            const float T = scale_num ? (static_cast<float>(scale_num) / 256.0f)
                                      : SOFTMAX_TEMPERATURE;
            const float d = (static_cast<float>(tile[i]) - static_cast<float>(scalar))
                          / T;
            tile[i] = clamp8(static_cast<int32_t>(std::roundf(std::exp(d) * 127.0f)));
        }
        break;
    }

    case Decoder::ActivationMode::RMS_SCALE:
        // tile[i] = clamp8(round(tile[i] * scale_num / sqrt(scalar)))
        //
        // RMSNorm normalises by the ROOT MEAN square, and its result is ~1.0
        // by construction -- which is why it needs an explicit fixed-point
        // convention to survive INT8 at all. Both are folded into scale_num,
        // a compile-time constant the caller sets to round(127 * sqrt(N)):
        //
        //   x / sqrt(sum/N) * 127  ==  x * (127 * sqrt(N)) / sqrt(sum)
        //
        // so the mean (rather than the sum) and the Q0.7 output scale cost
        // nothing at run time. Dividing by the SUM and emitting the raw
        // quotient -- as this previously did -- peaks at ~0.117 on real
        // activations and rounds to zero for every element.
        //
        // scale_num arrives in dram_addr, unused by SCALAR_ALU and wide
        // enough for the constant; the ISA already overloads stride three
        // ways by opcode.
        if (scalar > 0) {
            const float inv = static_cast<float>(scale_num)
                            / std::sqrt(static_cast<float>(scalar));
            for (uint32_t i = 0; i < elems; ++i)
                tile[i] = clamp8(static_cast<int32_t>(
                    std::roundf(static_cast<float>(tile[i]) * inv)));
        }
        break;

    case Decoder::ActivationMode::SCALAR_DIV:
        // tile * 127 / scalar. Normalises the exponentials so the row sums to
        // ~127, i.e. 127 is the INT8 stand-in for probability 1.0.
        for (uint32_t i = 0; i < elems; ++i)
            tile[i] = (scalar != 0)
                ? clamp8(static_cast<int32_t>(std::roundf(
                      static_cast<float>(tile[i]) * 127.0f / static_cast<float>(scalar))))
                : 0;
        break;

    default:
        break;
    }
}
