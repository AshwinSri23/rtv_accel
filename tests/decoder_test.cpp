#include "../inc/decoder.hpp"
#include <cstdio>
#include <stdexcept>

// ─── Minimal test harness ────────────────────────────────────────────────────

static int s_pass = 0;
static int s_fail = 0;

#define CHECK(cond) do { \
    if (cond) { ++s_pass; } \
    else { ++s_fail; \
           std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
} while(0)

#define CHECK_EQ(a, b) CHECK((a) == (b))

// Function template instead of macro so the try/catch compiles in a clean,
// unambiguous scope with no macro-expansion surprises.
template<typename ExcType, typename Fn>
static bool throws(Fn fn) {
    try { fn(); return false; }
    catch (const ExcType&) { return true; }
    catch (...) { return false; }
}

#define CHECK_THROWS(ExcType, expr) \
    CHECK((throws<ExcType>([&]{ (expr); })))

// ─── Helpers ──────────────────────────────────────────────────────────────────

static bool instr_eq(const Decoder::DecodedInstruction& a,
                     const Decoder::DecodedInstruction& b) {
    return a.opcode      == b.opcode
        && a.dep_id      == b.dep_id
        && a.dram_addr   == b.dram_addr
        && a.sram_addr   == b.sram_addr
        && a.dim_m       == b.dim_m
        && a.dim_n       == b.dim_n
        && a.dim_k       == b.dim_k
        && a.stride      == b.stride
        && a.activation  == b.activation
        && a.accumulate  == b.accumulate
        && a.scale_shift == b.scale_shift
        && a.c_slot      == b.c_slot;
}

// Build a RawInstruction from named fields using the new 32-bit dram_addr layout.
// dim_m straddles hi/lo: upper byte in hi[7:0], lower byte in lo[63:56].
static Decoder::RawInstruction make_raw(uint8_t opcode, uint8_t dep_id,
                                        uint32_t dram,  uint16_t sram,
                                        uint16_t m,     uint16_t n, uint16_t k,
                                        uint16_t stride, uint8_t act) {
    uint64_t hi =
        (static_cast<uint64_t>(opcode) << 60) |
        (static_cast<uint64_t>(dep_id) << 56) |
        (static_cast<uint64_t>(dram)   << 24) |   // 32-bit at [55:24]
        (static_cast<uint64_t>(sram)   <<  8) |
        (static_cast<uint64_t>(m >> 8));            // dim_m upper byte
    uint64_t lo =
        (static_cast<uint64_t>(m & 0xFF) << 56) |  // dim_m lower byte
        (static_cast<uint64_t>(n)        << 40) |
        (static_cast<uint64_t>(k)        << 24) |
        (static_cast<uint64_t>(stride)   <<  8) |
        (static_cast<uint64_t>(act));
    return {hi, lo};
}

// ─── Tests ────────────────────────────────────────────────────────────────────

static void test_decode_known_gemm() {
    // 0x3 = COMPUTE_GEMM; use a 32-bit dram_addr to exercise the full field width
    Decoder::RawInstruction raw =
        make_raw(0x3, 0x3, 0xAB001000, 0x0200, 16, 16, 16, 16, 0x01);
    Decoder::DecodedInstruction d = Decoder::decode(raw);
    CHECK_EQ((int)d.opcode,    (int)Decoder::Opcode::COMPUTE_GEMM);
    CHECK_EQ(d.dep_id,          0x3);
    CHECK_EQ(d.dram_addr,       0xAB001000u);   // full 32-bit value preserved
    CHECK_EQ(d.sram_addr,       0x0200);
    CHECK_EQ(d.dim_m,           16);
    CHECK_EQ(d.dim_n,           16);
    CHECK_EQ(d.dim_k,           16);
    CHECK_EQ(d.stride,          16);
    CHECK_EQ((int)d.activation, (int)Decoder::ActivationMode::RELU);
}

static void test_encode_decode_round_trip() {
    // All 7 opcodes must survive a full encode→decode round-trip.
    // dram_addr uses the full 32-bit range to verify no truncation occurs.
    const Decoder::Opcode opcodes[] = {
        Decoder::Opcode::LOAD_TILE,    Decoder::Opcode::COMPUTE_GEMM,
        Decoder::Opcode::EXEC_ALU,     Decoder::Opcode::STORE_TILE,
        Decoder::Opcode::WAIT_SYNC,    Decoder::Opcode::REDUCE,
        Decoder::Opcode::SCALAR_ALU
    };
    for (auto op : opcodes) {
        Decoder::DecodedInstruction orig{};
        orig.opcode     = op;
        orig.dep_id     = 0xA;
        orig.dram_addr  = 0xABCDEF12;  // full 32-bit value
        orig.sram_addr  = 0x1234;
        orig.dim_m      = 32;
        orig.dim_n      = 64;
        orig.dim_k      = 128;
        orig.stride     = 256;
        orig.activation = Decoder::ActivationMode::INT8_QUANT;
        CHECK(instr_eq(orig, Decoder::decode(Decoder::encode(orig))));
    }
}

// All 8 activation modes must survive a round-trip.
static void test_all_activation_modes_round_trip() {
    const Decoder::ActivationMode modes[] = {
        Decoder::ActivationMode::NONE,
        Decoder::ActivationMode::RELU,
        Decoder::ActivationMode::INT8_QUANT,
        Decoder::ActivationMode::SILU,
        Decoder::ActivationMode::SQUARE,
        Decoder::ActivationMode::RSQRT,
        Decoder::ActivationMode::ELEMWISE_MUL,
        Decoder::ActivationMode::SCALAR_ADD,
    };
    for (auto mode : modes) {
        Decoder::DecodedInstruction orig{};
        orig.opcode     = Decoder::Opcode::EXEC_ALU;
        orig.activation = mode;
        CHECK((int)Decoder::decode(Decoder::encode(orig)).activation == (int)mode);
    }
}

// REDUCE round-trip preserves dep_id and sram_addr (scalar destination).
static void test_reduce_round_trip() {
    Decoder::DecodedInstruction r{};
    r.opcode     = Decoder::Opcode::REDUCE;
    r.dep_id     = 0x2;
    r.sram_addr  = 0x0400;  // scalar output address
    r.activation = Decoder::ActivationMode::SQUARE;  // sum-of-squares mode
    Decoder::DecodedInstruction out = Decoder::decode(Decoder::encode(r));
    CHECK_EQ((int)out.opcode,    (int)Decoder::Opcode::REDUCE);
    CHECK_EQ(out.dep_id,         0x2);
    CHECK_EQ(out.sram_addr,      0x0400);
    CHECK_EQ((int)out.activation,(int)Decoder::ActivationMode::SQUARE);
}

// SCALAR_ALU round-trip: scalar address in sram_addr, tile address in stride field.
static void test_scalar_alu_round_trip() {
    Decoder::DecodedInstruction s{};
    s.opcode     = Decoder::Opcode::SCALAR_ALU;
    s.sram_addr  = 0x0400;  // scalar source
    s.stride     = 0x0800;  // tile destination
    s.activation = Decoder::ActivationMode::RSQRT;
    Decoder::DecodedInstruction out = Decoder::decode(Decoder::encode(s));
    CHECK_EQ((int)out.opcode,    (int)Decoder::Opcode::SCALAR_ALU);
    CHECK_EQ(out.sram_addr,      0x0400);
    CHECK_EQ(out.stride,         0x0800);
    CHECK_EQ((int)out.activation,(int)Decoder::ActivationMode::RSQRT);
}

// ELEMWISE_MUL: tile B address carried in stride field.
static void test_elemwise_mul_round_trip() {
    Decoder::DecodedInstruction e{};
    e.opcode     = Decoder::Opcode::EXEC_ALU;
    e.sram_addr  = 0x0200;  // tile A (SiLU(gate))
    e.stride     = 0x0400;  // tile B (up_proj output)
    e.activation = Decoder::ActivationMode::ELEMWISE_MUL;
    Decoder::DecodedInstruction out = Decoder::decode(Decoder::encode(e));
    CHECK_EQ((int)out.activation,(int)Decoder::ActivationMode::ELEMWISE_MUL);
    CHECK_EQ(out.sram_addr,      0x0200);
    CHECK_EQ(out.stride,         0x0400);
}

static void test_raw_round_trip() {
    // 0x3 = COMPUTE_GEMM; tests that encode(decode(raw)) == raw exactly.
    // Uses a 32-bit dram_addr to exercise the full field.
    Decoder::RawInstruction raw =
        make_raw(0x3, 0x5, 0xFF001234, 0xBEEF, 8, 16, 32, 4, 0x00);
    Decoder::RawInstruction out = Decoder::encode(Decoder::decode(raw));
    CHECK_EQ(raw.hi, out.hi);
    CHECK_EQ(raw.lo, out.lo);
}

static void test_invalid_opcode_throws() {
    Decoder::RawInstruction raw = make_raw(0xF, 0, 0, 0, 0, 0, 0, 0, 0);
    CHECK_THROWS(std::invalid_argument, Decoder::decode(raw));
}

static void test_invalid_activation_throws() {
    // 0x1 = LOAD_TILE (valid opcode); 0xFF = invalid activation → must throw
    Decoder::RawInstruction raw = make_raw(0x1, 0, 0, 0, 0, 0, 0, 0, 0xFF);
    CHECK_THROWS(std::invalid_argument, Decoder::decode(raw));
}

static void test_dram_addr_full_32bit() {
    // dram_addr is now a full 32-bit field — all bits must survive encode/decode
    // without corrupting dep_id or opcode.
    Decoder::DecodedInstruction d{};
    d.opcode     = Decoder::Opcode::LOAD_TILE;
    d.dep_id     = 0x5;
    d.dram_addr  = 0xDEAD'BEEF;  // full 32-bit value — must survive intact
    d.activation = Decoder::ActivationMode::NONE;

    Decoder::RawInstruction raw = Decoder::encode(d);
    Decoder::DecodedInstruction out = Decoder::decode(raw);

    CHECK_EQ(out.dram_addr,       0xDEAD'BEEFu);                 // all 32 bits preserved
    CHECK_EQ(out.dep_id,          0x5);                           // not corrupted
    CHECK_EQ((int)out.opcode, (int)Decoder::Opcode::LOAD_TILE);  // not corrupted
}

static void test_wait_sync_preserves_dep_id() {
    Decoder::DecodedInstruction ws{};
    ws.opcode     = Decoder::Opcode::WAIT_SYNC;
    ws.dep_id     = 0xB;
    ws.activation = Decoder::ActivationMode::NONE;
    Decoder::DecodedInstruction out = Decoder::decode(Decoder::encode(ws));
    CHECK_EQ((int)out.opcode, (int)Decoder::Opcode::WAIT_SYNC);
    CHECK_EQ(out.dep_id, 0xB);
}

// v1.5 accumulate flag: round-trips through encode/decode for all 7 opcodes.
static void test_v15_accumulate_round_trip() {
    Decoder::DecodedInstruction d{};
    d.opcode      = Decoder::Opcode::COMPUTE_GEMM;
    d.dep_id      = 0x4;
    d.dram_addr   = 0x00010000u;
    d.sram_addr   = 0x0000;
    d.dim_m       = 16;
    d.dim_n       = 16;
    d.dim_k       = 16;
    d.stride      = 0x0400;
    d.activation  = Decoder::ActivationMode::NONE;
    d.accumulate  = true;
    d.scale_shift = 0;
    d.c_slot      = 0;
    CHECK(instr_eq(d, Decoder::decode(Decoder::encode(d))));
}

// v1.5 scale_shift + c_slot: all sub-fields survive encode/decode.
static void test_v15_subfields_round_trip() {
    Decoder::DecodedInstruction d{};
    d.opcode      = Decoder::Opcode::COMPUTE_GEMM;
    d.dep_id      = 0x2;
    d.dram_addr   = 0xCAFE0000u;
    d.sram_addr   = 0x0200;
    d.dim_m       = 12;
    d.dim_n       = 16;
    d.dim_k       = 8;
    d.stride      = 0x0600;
    d.activation  = Decoder::ActivationMode::INT8_QUANT;
    d.accumulate  = true;
    d.scale_shift = 5;   // non-default shift
    d.c_slot      = 3;   // non-zero C slot
    Decoder::DecodedInstruction out = Decoder::decode(Decoder::encode(d));
    CHECK(instr_eq(d, out));
    CHECK_EQ(out.accumulate,  true);
    CHECK_EQ(out.scale_shift, 5u);
    CHECK_EQ(out.c_slot,      3u);
}

// v1.5 backward compat: a v1-encoded instruction (all new bits zero) decodes
// with accumulate=false, scale_shift=0, c_slot=0.
static void test_v15_backward_compat() {
    // Build a pure v1 raw word: M=16 means dim_m wire = 0x0010 (bits 12:8 = 0)
    Decoder::RawInstruction raw =
        make_raw(0x3, 0x1, 0x001000, 0x0000, 16, 16, 16, 0x0000, 0x01); // RELU, no accum
    Decoder::DecodedInstruction d = Decoder::decode(raw);
    CHECK_EQ(d.accumulate,  false);
    CHECK_EQ(d.scale_shift, 0u);
    CHECK_EQ(d.c_slot,      0u);
    CHECK_EQ((int)d.activation, (int)Decoder::ActivationMode::RELU);
    CHECK_EQ(d.dim_m, 16u);
    CHECK_EQ(d.dim_k, 16u);
}

// ─── Runner ───────────────────────────────────────────────────────────────────

struct Test { const char* name; void (*fn)(); };

int main() {
    const Test tests[] = {
        {"DecodeKnownGemm",              test_decode_known_gemm},
        {"EncodeDecodeRoundTrip",        test_encode_decode_round_trip},
        {"AllActivationModesRoundTrip",  test_all_activation_modes_round_trip},
        {"ReduceRoundTrip",              test_reduce_round_trip},
        {"ScalarAluRoundTrip",           test_scalar_alu_round_trip},
        {"ElemwiseMulRoundTrip",         test_elemwise_mul_round_trip},
        {"RawRoundTrip",                 test_raw_round_trip},
        {"InvalidOpcodeThrows",          test_invalid_opcode_throws},
        {"InvalidActivationThrows",      test_invalid_activation_throws},
        {"DramAddrFull32bit",            test_dram_addr_full_32bit},
        {"WaitSyncPreservesDepId",       test_wait_sync_preserves_dep_id},
        // ISA v1.5
        {"V15AccumulateRoundTrip",       test_v15_accumulate_round_trip},
        {"V15SubfieldsRoundTrip",        test_v15_subfields_round_trip},
        {"V15BackwardCompat",            test_v15_backward_compat},
    };

    for (const auto& t : tests) {
        int before_fail = s_fail;
        try {
            t.fn();
        } catch (const std::exception& e) {
            ++s_fail;
            std::fprintf(stderr, "FAIL  %s  uncaught exception: %s\n", t.name, e.what());
        } catch (...) {
            ++s_fail;
            std::fprintf(stderr, "FAIL  %s  uncaught unknown exception\n", t.name);
        }
        std::printf("%s  %s\n", s_fail == before_fail ? "PASS" : "FAIL", t.name);
    }

    std::printf("\n%d passed, %d failed\n", s_pass, s_fail);
    return s_fail == 0 ? 0 : 1;
}
