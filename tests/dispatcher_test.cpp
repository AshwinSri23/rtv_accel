#include "../inc/instruction_dispatcher.hpp"
#include "../inc/decoder.hpp"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

// ─── Minimal test harness (same as decoder_test) ─────────────────────────────

static int s_pass = 0, s_fail = 0;

#define CHECK(cond) do { \
    if (cond) { ++s_pass; } \
    else { ++s_fail; \
           std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
} while(0)
#define CHECK_EQ(a,b) CHECK((a)==(b))

// ─── Helper: build a big-endian 16-byte instruction word ─────────────────────

// Build a big-endian 16-byte instruction word matching the new 32-bit dram_addr layout.
// dim_m straddles hi/lo: upper byte in hi[7:0], lower byte in lo[63:56].
// padding parameter removed from ISA.
static void append_be_instruction(
    std::vector<uint8_t>& buf,
    uint8_t  opcode,  uint8_t  dep_id,
    uint32_t dram,    uint16_t sram,
    uint16_t dim_m,   uint16_t dim_n,   uint16_t dim_k,
    uint16_t stride,  uint8_t  act)
{
    uint64_t hi =
        (static_cast<uint64_t>(opcode)       << 60) |
        (static_cast<uint64_t>(dep_id)       << 56) |
        (static_cast<uint64_t>(dram)         << 24) |   // 32-bit at [55:24]
        (static_cast<uint64_t>(sram)         <<  8) |
        (static_cast<uint64_t>(dim_m >> 8));             // dim_m upper byte
    uint64_t lo =
        (static_cast<uint64_t>(dim_m & 0xFF) << 56) |   // dim_m lower byte
        (static_cast<uint64_t>(dim_n)        << 40) |
        (static_cast<uint64_t>(dim_k)        << 24) |
        (static_cast<uint64_t>(stride)       <<  8) |
        (static_cast<uint64_t>(act));

    // Write big-endian (Python compiler's output format)
    auto push64 = [&](uint64_t v) {
        for (int b = 7; b >= 0; --b)
            buf.push_back(static_cast<uint8_t>((v >> (b * 8)) & 0xFF));
    };
    push64(hi);
    push64(lo);
}

// ─── Tests ───────────────────────────────────────────────────────────────────

static void test_load_buffer_and_decode() {
    std::vector<uint8_t> buf;
    // LOAD_TILE = 0x1,  dep=2,  dram=0xAB001000 (32-bit), sram=0x0000, M=16, N=16
    append_be_instruction(buf, 0x1, 2, 0xAB001000, 0x0000, 16, 16, 0, 0, 0x00);

    InstructionDispatcher disp;
    disp.load_buffer(buf.data(), buf.size());

    CHECK_EQ(disp.instruction_count(), 1u);

    Decoder::DecodedInstruction d = disp.decode_at(0);
    CHECK_EQ((int)d.opcode,   (int)Decoder::Opcode::LOAD_TILE);
    CHECK_EQ(d.dep_id,         2);
    CHECK_EQ(d.dram_addr,      0xAB001000u);   // full 32-bit address preserved
    CHECK_EQ(d.sram_addr,      0x0000u);
    CHECK_EQ(d.dim_m,          16u);
    CHECK_EQ(d.dim_n,          16u);
}

static void test_dispatch_routes_to_handler() {
    std::vector<uint8_t> buf;
    // COMPUTE_GEMM = 0x3, dep=1, sram=0x0400, M=16, N=16, K=16
    append_be_instruction(buf, 0x3, 1, 0x000000, 0x0400, 16, 16, 16, 0, 0x00);
    // EXEC_ALU     = 0x4, activation=SILU=0x03
    append_be_instruction(buf, 0x4, 0, 0x000000, 0x0800, 16, 16, 0,  0, 0x03);

    int gemm_calls = 0, alu_calls  = 0;

    InstructionDispatcher disp;
    disp.load_buffer(buf.data(), buf.size());

    disp.on(Decoder::Opcode::COMPUTE_GEMM, [&](const Decoder::DecodedInstruction& d) {
        ++gemm_calls;
        CHECK_EQ(d.dim_k, 16u);
    });
    disp.on(Decoder::Opcode::EXEC_ALU, [&](const Decoder::DecodedInstruction& d) {
        ++alu_calls;
        CHECK_EQ((int)d.activation, (int)Decoder::ActivationMode::SILU);
    });

    disp.dispatch_all();

    CHECK_EQ(gemm_calls, 1);
    CHECK_EQ(alu_calls,  1);
}

static void test_wait_sync_dispatched() {
    std::vector<uint8_t> buf;
    // WAIT_SYNC = 0x5, dep=3
    append_be_instruction(buf, 0x5, 3, 0, 0, 0, 0, 0, 0, 0);

    int sync_calls = 0;
    InstructionDispatcher disp;
    disp.load_buffer(buf.data(), buf.size());
    disp.on(Decoder::Opcode::WAIT_SYNC, [&](const Decoder::DecodedInstruction& d) {
        ++sync_calls;
        CHECK_EQ(d.dep_id, 3);
    });
    disp.dispatch_all();
    CHECK_EQ(sync_calls, 1);
}

static void test_unhandled_fallback() {
    std::vector<uint8_t> buf;
    // STORE_TILE = 0x2; no specific handler registered → should hit unhandled
    append_be_instruction(buf, 0x2, 0, 0x003000, 0x0800, 16, 16, 0, 0, 0);

    int fallback_calls = 0;
    InstructionDispatcher disp;
    disp.load_buffer(buf.data(), buf.size());
    disp.on_unhandled([&](const Decoder::DecodedInstruction&) { ++fallback_calls; });
    disp.dispatch_all();
    CHECK_EQ(fallback_calls, 1);
}

static void test_multi_instruction_sequence() {
    // Replicate the canonical single-tile GEMM sequence emitted by the Python compiler:
    //   LOAD_TILE (act)  dep=1
    //   LOAD_TILE (wgt)  dep=2
    //   WAIT_SYNC        dep=1
    //   WAIT_SYNC        dep=2
    //   COMPUTE_GEMM
    //   STORE_TILE
    std::vector<uint8_t> buf;
    append_be_instruction(buf, 0x1, 1, 0x001000, 0x0000, 16, 16, 0,  0, 0x00);
    append_be_instruction(buf, 0x1, 2, 0x100000, 0x0400, 16, 16, 0,  0, 0x00);
    append_be_instruction(buf, 0x5, 1, 0,        0,      0,  0,  0,  0, 0x00);
    append_be_instruction(buf, 0x5, 2, 0,        0,      0,  0,  0,  0, 0x00);
    append_be_instruction(buf, 0x3, 1, 0,        0x0000, 16, 16, 16, 0, 0x01);
    append_be_instruction(buf, 0x2, 0, 0x003000, 0x0800, 16, 16, 0,  0, 0x00);

    InstructionDispatcher disp;
    disp.load_buffer(buf.data(), buf.size());
    CHECK_EQ(disp.instruction_count(), 6u);

    // Check opcode sequence
    const Decoder::Opcode expected[] = {
        Decoder::Opcode::LOAD_TILE,
        Decoder::Opcode::LOAD_TILE,
        Decoder::Opcode::WAIT_SYNC,
        Decoder::Opcode::WAIT_SYNC,
        Decoder::Opcode::COMPUTE_GEMM,
        Decoder::Opcode::STORE_TILE,
    };
    for (int i = 0; i < 6; ++i)
        CHECK_EQ((int)disp.decode_at(i).opcode, (int)expected[i]);

    // GEMM at index 4 should have RELU activation (0x01)
    CHECK_EQ((int)disp.decode_at(4).activation, (int)Decoder::ActivationMode::RELU);
}

static void test_bad_buffer_size_throws() {
    std::vector<uint8_t> buf(17, 0);   // 17 bytes — not a multiple of 16
    InstructionDispatcher disp;
    bool threw = false;
    try { disp.load_buffer(buf.data(), buf.size()); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// ─── Runner ───────────────────────────────────────────────────────────────────

struct Test { const char* name; void(*fn)(); };

int main() {
    const Test tests[] = {
        {"LoadBufferAndDecode",       test_load_buffer_and_decode},
        {"DispatchRoutesToHandler",   test_dispatch_routes_to_handler},
        {"WaitSyncDispatched",        test_wait_sync_dispatched},
        {"UnhandledFallback",         test_unhandled_fallback},
        {"MultiInstructionSequence",  test_multi_instruction_sequence},
        {"BadBufferSizeThrows",       test_bad_buffer_size_throws},
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
