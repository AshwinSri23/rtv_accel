#include "../inc/instruction_dispatcher.hpp"
#include "../inc/memory_controller.hpp"
#include "../inc/pe_array.hpp"
#include "../inc/decoder.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// ── SRAM zone addresses (mirror SRAMMap in toolchain/npu_compiler.py) ─────────
static constexpr uint16_t SRAM_INPUT_BASE  = 0x0000;
static constexpr uint16_t SRAM_WEIGHT_BASE = 0x0400;
static constexpr uint16_t SRAM_OUTPUT_BASE = 0x0800;

// ISA v1.5 c_slot addressing: c_slot (4 bits, 0-15) selects one of 16 fixed-size
// output slots starting at SRAM_OUTPUT_BASE. Each slot is sized to hold the
// largest possible C tile the 16x16 PE array can produce: 16*16*sizeof(int32_t)
// = 1024 bytes. This supersedes the older "4 x 256B slots" convention in
// memory_controller.hpp's zone-layout comment, which predates c_slot and was
// too small to hold a full-size INT32 accumulator tile (256B only fits 8x8).
// c_slot=0 (the v1/zero-initialized default) resolves to exactly
// SRAM_OUTPUT_BASE, so this is backward compatible with every v1-encoded
// COMPUTE_GEMM and with tools/compiler_main.cpp / tests/compiler_test.cpp's
// local run_program() helpers, which hardcode 0x0800 for the same reason.
static constexpr uint16_t SRAM_OUTPUT_SLOT_BYTES = 16 * 16 * sizeof(int32_t);  // 1024

static uint16_t c_slot_addr(uint8_t c_slot) {
    return static_cast<uint16_t>(SRAM_OUTPUT_BASE + c_slot * SRAM_OUTPUT_SLOT_BYTES);
}

// ── Minimal JSON key search ────────────────────────────────────────────────────
// Reads one integer value from a JSON file without a library dependency.
static uint32_t json_read_uint(const std::string& path, const std::string& key,
                               uint32_t default_val)
{
    std::ifstream f(path);
    if (!f.is_open()) return default_val;

    std::string content((std::istreambuf_iterator<char>(f)), {});
    const std::string needle = "\"" + key + "\"";
    auto pos = content.find(needle);
    if (pos == std::string::npos) return default_val;

    pos = content.find(':', pos + needle.size());
    if (pos == std::string::npos) return default_val;

    while (++pos < content.size() &&
           (content[pos] == ' ' || content[pos] == '\t' ||
            content[pos] == '\n' || content[pos] == '\r')) {}

    try { return static_cast<uint32_t>(std::stoul(content.substr(pos))); }
    catch (...) { return default_val; }
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <program.bin> [hal_config.json] [dram_mb] [options]\n"
            "\n"
            "  program.bin     — binary instruction stream from npu_compiler.py\n"
            "  hal_config.json — hardware profile (default: toolchain/hal_config.json)\n"
            "  dram_mb         — simulated DRAM size in MiB (default: 32)\n"
            "\n"
            "Options (required for end-to-end numeric verification — without them\n"
            "DRAM is zero-initialised and every result is trivially zero):\n"
            "  --dram-image FILE      preload DRAM from FILE, starting at address 0\n"
            "  --dump ADDR:LEN:FILE   after execution, write DRAM[ADDR, ADDR+LEN) to FILE\n"
            "                         (ADDR/LEN accept 0x-prefixed hex; repeatable)\n",
            argv[0]);
        return 1;
    }

    const std::string bin_path    = argv[1];
    std::string       config_path = "toolchain/hal_config.json";
    std::size_t       dram_mb     = 32;
    std::string       dram_image;
    struct DumpSpec { uint32_t addr; std::size_t len; std::string path; };
    std::vector<DumpSpec> dumps;

    // Positional args 2 and 3 stay supported for backward compatibility; any
    // argument starting with "--" switches to flag parsing.
    {
        int i = 2;
        for (; i < argc && std::string(argv[i]).rfind("--", 0) != 0; ++i) {
            if      (i == 2) config_path = argv[i];
            else if (i == 3) dram_mb     = std::stoul(argv[i]);
        }
        for (; i < argc; ++i) {
            const std::string flag = argv[i];
            auto next = [&](const char* what) -> std::string {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "error: %s needs a value\n", what);
                    std::exit(1);
                }
                return argv[++i];
            };
            if (flag == "--dram-image") {
                dram_image = next("--dram-image");
            } else if (flag == "--dump") {
                // ADDR:LEN:FILE
                const std::string spec = next("--dump");
                const auto c1 = spec.find(':');
                const auto c2 = spec.find(':', c1 == std::string::npos ? c1 : c1 + 1);
                if (c1 == std::string::npos || c2 == std::string::npos) {
                    std::fprintf(stderr, "error: --dump wants ADDR:LEN:FILE, got '%s'\n",
                                 spec.c_str());
                    return 1;
                }
                dumps.push_back({
                    static_cast<uint32_t>(std::stoul(spec.substr(0, c1), nullptr, 0)),
                    static_cast<std::size_t>(std::stoul(spec.substr(c1 + 1, c2 - c1 - 1),
                                                        nullptr, 0)),
                    spec.substr(c2 + 1),
                });
            } else {
                std::fprintf(stderr, "error: unknown option '%s'\n", flag.c_str());
                return 1;
            }
        }
    }

    // ── HAL config ────────────────────────────────────────────────────────────
    const uint32_t dram_latency =
        json_read_uint(config_path, "dram_latency_cycles", 120);

    // ── Load instruction stream ───────────────────────────────────────────────
    InstructionDispatcher disp;
    try {
        disp.load_binary(bin_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "load error: %s\n", e.what());
        return 1;
    }

    const std::size_t n_instr = disp.instruction_count();
    std::printf("program  : %s\n",   bin_path.c_str());
    std::printf("config   : %s\n",   config_path.c_str());
    std::printf("instrs   : %zu\n",  n_instr);
    std::printf("dram     : %zu MiB  (latency %u cycles)\n",
                dram_mb, dram_latency);

    // ── Hardware models ───────────────────────────────────────────────────────
    MemoryController mem(dram_latency);
    mem.resize_dram(dram_mb * 1024 * 1024);

    // ── Optional DRAM preload ────────────────────────────────────────────────
    // Without this every LOAD_TILE reads zeros, so any numeric result is
    // trivially zero and proves nothing. The image is a flat blob written at
    // address 0 — the compiler's weight/activation addresses index into it.
    if (!dram_image.empty()) {
        std::ifstream img(dram_image, std::ios::binary);
        if (!img) {
            std::fprintf(stderr, "error: cannot open --dram-image %s\n", dram_image.c_str());
            return 1;
        }
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(img)),
                                          std::istreambuf_iterator<char>());
        if (bytes.size() > dram_mb * 1024 * 1024) {
            std::fprintf(stderr,
                "error: --dram-image is %zu bytes but DRAM is only %zu MiB\n",
                bytes.size(), dram_mb);
            return 1;
        }
        try {
            mem.write_dram(0, bytes.data(), bytes.size());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "dram-image error: %s\n", e.what());
            return 1;
        }
        std::printf("dram-img : %s (%zu bytes)\n", dram_image.c_str(), bytes.size());
    }
    PEArray pe;

    // ── Cycle-accurate timing model ───────────────────────────────────────────
    //
    // sim_clock  — current simulated time; advanced by blocking operations and
    //              by WAIT_SYNC when the in-flight DMA hasn't finished yet.
    //
    // dep_done[] — completion cycle for each DMA dep_id (4-bit tag → 0-15).
    //              LOAD_TILE sets dep_done[dep_id] = sim_clock + latency
    //              without advancing sim_clock (DMA is non-blocking).
    //              WAIT_SYNC advances sim_clock to dep_done[dep_id] if needed.
    //
    // serial_cycles — hypothetical cost with zero overlap (LOAD always blocks,
    //                 no DMA/compute concurrency). Used to compute speedup.

    uint64_t sim_clock    = 0;
    uint64_t dep_done[16] = {};
    uint64_t serial_cycles = 0;

    // ── Opcode counters ───────────────────────────────────────────────────────
    uint64_t cnt[8] = {};   // indexed by Decoder::Opcode value (1–7)

    // ── Handlers ─────────────────────────────────────────────────────────────

    // LOAD_TILE: DRAM → SRAM.
    // DMA is non-blocking: dep_done[dep_id] records when it will finish, but
    // sim_clock is NOT advanced here.  A subsequent WAIT_SYNC stalls the clock.
    // stride=0 in compiler output means contiguous tile; default to dim_n.
    disp.on(Decoder::Opcode::LOAD_TILE, [&](const Decoder::DecodedInstruction& d) {
        const uint16_t stride = d.stride ? d.stride : d.dim_n;
        mem.load_tile(d.dram_addr, d.sram_addr, d.dim_m, d.dim_n, stride);

        const uint64_t lat = dram_latency
                           + static_cast<uint64_t>(d.dim_m) * d.dim_n;
        if (d.dep_id < 16)
            dep_done[d.dep_id] = sim_clock + lat;

        serial_cycles += lat;
        ++cnt[1];
    });

    // STORE_TILE: SRAM → DRAM.
    // Treated as blocking: the CPU cannot proceed until the write completes.
    disp.on(Decoder::Opcode::STORE_TILE, [&](const Decoder::DecodedInstruction& d) {
        const uint16_t stride = d.stride ? d.stride : d.dim_n;
        mem.store_tile(d.sram_addr, d.dram_addr, d.dim_m, d.dim_n, stride);

        const uint64_t lat = dram_latency
                           + static_cast<uint64_t>(d.dim_m) * d.dim_n;
        sim_clock    += lat;
        serial_cycles += lat;
        ++cnt[2];
    });

    // COMPUTE_GEMM: A at sram_addr, B at b_sram, C at c_slot_addr(d.c_slot).
    //
    // b_sram is encoded in the stride field (stride != 0 → use stride as B
    // address; stride == 0 → fall back to WEIGHT_BASE for backward compat).
    // This lets double-buffered code alternate between weight_slot(0) and (1).
    //
    // ISA v1.5: d.c_slot selects the C output slot (see c_slot_addr above);
    // d.accumulate is passed straight through -- false is both PEArray's
    // default and what a zero-initialized/v1-decoded instruction carries,
    // so there's no ambiguity there.
    //
    // scale_shift needs a fallback, not a straight pass-through: a v1-encoded
    // instruction decodes with scale_shift=0 (the spec's backward-compat
    // guarantee), but v1's actual hardcoded INT8_QUANT behavior was >>7, not
    // >>0 -- passing 0 straight through would silently change quantization
    // for every pre-v1.5 binary that uses INT8_QUANT. Treat 0 as "not set,
    // use the v1 default of 7"; any nonzero value is an explicit v1.5
    // override and passes through unchanged.
    //
    // Cost model: dim_m * dim_n cycles.
    // A 16×16 PE array with 16 columns processes one output row (all N cols)
    // per K-reduction step, so for K=16 steps and M output rows:
    //   throughput ≈ M × N cycles (= M rows × 1 row/cycle, 16 columns wide)
    disp.on(Decoder::Opcode::COMPUTE_GEMM, [&](const Decoder::DecodedInstruction& d) {
        const uint16_t b_sram = d.stride ? static_cast<uint16_t>(d.stride)
                                         : SRAM_WEIGHT_BASE;
        const uint16_t c_sram = c_slot_addr(d.c_slot);
        const uint8_t  scale_shift = d.scale_shift ? d.scale_shift : 7;
        pe.compute_gemm(mem,
                        d.sram_addr,       // A: input zone
                        b_sram,            // B: weight slot (from stride field)
                        c_sram,            // C: ISA v1.5 c_slot-addressed output
                        d.dim_m, d.dim_n, d.dim_k,
                        d.activation,
                        d.accumulate,
                        scale_shift,
                        d.dtype);

        const uint64_t cost = static_cast<uint64_t>(d.dim_m) * d.dim_n;
        sim_clock    += cost;
        serial_cycles += cost;
        ++cnt[3];
    });

    // EXEC_ALU: element-wise op on tile at sram_addr.
    // stride doubles as b_sram for binary modes (ELEMWISE_MUL, SCALAR_ADD).
    disp.on(Decoder::Opcode::EXEC_ALU, [&](const Decoder::DecodedInstruction& d) {
        // dram_addr doubles as an immediate-operand field for EXEC_ALU modes
        // that need a constant rather than an SRAM address (RESCALE's
        // mantissa/shift), the same way SCALAR_ALU RMS_SCALE already uses it.
        pe.exec_alu(mem, d.sram_addr, d.dim_m, d.dim_n, d.activation,
                    static_cast<uint16_t>(d.stride), d.dram_addr, d.dtype);

        const uint64_t cost = static_cast<uint64_t>(d.dim_m) * d.dim_n;
        sim_clock    += cost;
        serial_cycles += cost;
        ++cnt[4];
    });

    // WAIT_SYNC: stall sim_clock until dep_id's DMA load has completed.
    // In the double-buffering scheme this is where the non-blocking LOADs
    // "manifest": if the DMA finished during a prior COMPUTE_GEMM, the stall
    // is zero; otherwise we pay the remaining latency.
    disp.on(Decoder::Opcode::WAIT_SYNC, [&](const Decoder::DecodedInstruction& d) {
        if (d.dep_id < 16 && dep_done[d.dep_id] > sim_clock)
            sim_clock = dep_done[d.dep_id];
        ++cnt[5];
    });

    // REDUCE: sum all elements of a tile into a 4-byte INT32 scalar written
    // to sram_addr.
    //
    // Tile source is carried in the stride field (otherwise unused by
    // REDUCE) -- stride=0 falls back to SRAM_INPUT_BASE for backward
    // compatibility with any caller that predates this field's use, same
    // pattern as LOAD_TILE/STORE_TILE's stride=0 fallback and COMPUTE_GEMM's
    // b_sram=0 fallback. Previously this was unconditionally hardcoded to
    // SRAM_INPUT_BASE regardless of where the tile actually was -- correct
    // only when the tile happened to already be at 0x0000, silently wrong
    // otherwise (e.g. RMSNorm on a tile sitting in the output zone).
    disp.on(Decoder::Opcode::REDUCE, [&](const Decoder::DecodedInstruction& d) {
        const uint16_t tile_sram = d.stride ? static_cast<uint16_t>(d.stride)
                                            : SRAM_INPUT_BASE;
        // d.accumulate lets a reduction span more tiles than one: RMSNorm's
        // scale is a sum over a whole row, which is wider than a single tile.
        pe.reduce(mem, tile_sram, d.sram_addr, d.dim_m, d.dim_n,
                  d.accumulate, d.activation, d.dtype);

        const uint64_t cost = static_cast<uint64_t>(d.dim_m) * d.dim_n;
        sim_clock    += cost;
        serial_cycles += cost;
        ++cnt[6];
    });

    // SCALAR_ALU: in-place scalar operations used by RMSNorm.
    // sram_addr = address of the 4-byte INT32 scalar.
    // SCALAR_ADD: scalar += stride  (stride encodes ε as Q0.15 fixed-point)
    // RSQRT:      scalar  = 1/√scalar  (result stored as Q0.15)
    // Other modes delegate to pe_array for tile-based scalar operations.
    disp.on(Decoder::Opcode::SCALAR_ALU, [&](const Decoder::DecodedInstruction& d) {
        // Delegate every mode to PEArray, which is the reference executor.
        //
        // This handler used to special-case SCALAR_ADD and RSQRT, transforming
        // the SCALAR in place (scalar += stride; scalar = rsqrt(scalar) as
        // Q0.15) and never touching the tile. PEArray::scalar_alu does the
        // opposite for those same modes -- it applies the scalar TO the tile
        // (tile[i] += scalar; tile[i] = round(tile[i] / sqrt(scalar))) -- which
        // is what the ISA actually specifies: "reads a scalar value from a
        // nominated SRAM address and applies it element-wise to a tile".
        //
        // So the simulator and the reference executor disagreed on the meaning
        // of two of SCALAR_ALU's modes, and the simulator's version was the one
        // that contradicted the spec. Reimplementing opcode semantics here
        // rather than dispatching to PEArray is how that divergence happened.
        //
        // Field roles for SCALAR_ALU (note these are the reverse of EXEC_ALU's):
        //   sram_addr = the 4-byte INT32 scalar
        //   stride    = the tile the scalar is applied to
        // dram_addr carries RMS_SCALE's numerator constant; unused otherwise.
        pe.scalar_alu(mem, d.sram_addr,
                      static_cast<uint16_t>(d.stride),
                      d.dim_m, d.dim_n,
                      d.activation, d.dram_addr, d.dtype);

        sim_clock    += 4;
        serial_cycles += 4;
        ++cnt[7];
    });

    // ── Execute ───────────────────────────────────────────────────────────────
    std::printf("\nexecuting...\n");
    try {
        disp.dispatch_all();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nruntime error: %s\n", e.what());
        return 1;
    }

    // ── Optional DRAM dumps ──────────────────────────────────────────────────
    // Writes back regions the compiler stored results into, so a caller can
    // compare them against an independently-computed reference.
    for (const auto& d : dumps) {
        std::vector<uint8_t> buf(d.len);
        try {
            mem.read_dram(d.addr, buf.data(), buf.size());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "dump error: %s\n", e.what());
            return 1;
        }
        std::ofstream out(d.path, std::ios::binary | std::ios::trunc);
        if (!out) {
            std::fprintf(stderr, "error: cannot write --dump file %s\n", d.path.c_str());
            return 1;
        }
        out.write(reinterpret_cast<const char*>(buf.data()),
                  static_cast<std::streamsize>(buf.size()));
        std::printf("dumped   : 0x%08X +%zu -> %s\n", d.addr, d.len, d.path.c_str());
    }

    // ── Results ───────────────────────────────────────────────────────────────
    const uint64_t total = cnt[1]+cnt[2]+cnt[3]+cnt[4]+cnt[5]+cnt[6]+cnt[7];

    std::printf("\n── results ──────────────────────────────────────────────────\n");
    std::printf("  LOAD_TILE    %6llu\n", (unsigned long long)cnt[1]);
    std::printf("  STORE_TILE   %6llu\n", (unsigned long long)cnt[2]);
    std::printf("  COMPUTE_GEMM %6llu\n", (unsigned long long)cnt[3]);
    std::printf("  EXEC_ALU     %6llu\n", (unsigned long long)cnt[4]);
    std::printf("  WAIT_SYNC    %6llu\n", (unsigned long long)cnt[5]);
    std::printf("  REDUCE       %6llu\n", (unsigned long long)cnt[6]);
    std::printf("  SCALAR_ALU   %6llu\n", (unsigned long long)cnt[7]);
    std::printf("  ──────────────────────────────────────────────────────────\n");
    std::printf("  total instrs %6llu\n", (unsigned long long)total);
    std::printf("\n── cycle estimates ──────────────────────────────────────────\n");
    std::printf("  serial      %8llu  (no overlap; DMA always blocks)\n",
                (unsigned long long)serial_cycles);
    std::printf("  db          %8llu  (double-buffered; DMA overlaps GEMM)\n",
                (unsigned long long)sim_clock);
    if (sim_clock > 0 && serial_cycles >= sim_clock) {
        const double speedup = static_cast<double>(serial_cycles) /
                               static_cast<double>(sim_clock);
        std::printf("  speedup     %8.2fx\n", speedup);
    }

    return 0;
}
