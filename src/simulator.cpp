#include <algorithm>
#include "../inc/instruction_dispatcher.hpp"
#include "../inc/memory_controller.hpp"
#include "../inc/pe_array.hpp"
#include "../inc/decoder.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
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

static constexpr std::size_t SRAM_BYTES_LIMIT = 64 * 1024;   // simulated SRAM size

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <program.bin|--serve> [hal_config.json] [dram_mb] [options]\n"
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
    // --serve: a persistent session. DRAM (and the loaded weights in it) stays
    // resident while programs are submitted over stdin; see the serve loop below.
    const bool        serve       = (bin_path == "--serve");
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
    // Bytes the single shared DMA engine moves per cycle: an explicit
    // "dram_bytes_per_cycle" if the config sets one, else the bus width the
    // config already declares (bus_width_bits / 8; 128 bits = 16 B/cycle).
    const uint32_t bus_bytes = std::max<uint32_t>(
        1, json_read_uint(config_path, "bus_width_bits", 128) / 8);
    const uint32_t dma_bpc = std::max<uint32_t>(
        1, json_read_uint(config_path, "dram_bytes_per_cycle", bus_bytes));

    // ── Load instruction stream ───────────────────────────────────────────────
    InstructionDispatcher disp;
    try {
        if (!serve) disp.load_binary(bin_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "load error: %s\n", e.what());
        return 1;
    }

    const std::size_t n_instr = disp.instruction_count();
    std::printf("program  : %s\n",   bin_path.c_str());
    std::printf("config   : %s\n",   config_path.c_str());
    std::printf("instrs   : %zu\n",  n_instr);
    std::printf("dram     : %zu MiB  (latency %u cycles, %u B/cycle shared DMA)\n",
                dram_mb, dram_latency, dma_bpc);

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
    // BF16W is a COMPUTE_GEMM-only dtype; fail loudly rather than misread data.
    auto reject_bf16w = [](const Decoder::DecodedInstruction& d) {
        if (d.dtype == Decoder::DType::BF16W)
            throw std::runtime_error("dtype BF16W is only valid on COMPUTE_GEMM");
    };
    // LOAD/STORE address: 32 low bits from dram_addr, bits 35:32 from dram_hi.
    auto dram_addr36 = [&](const Decoder::DecodedInstruction& d) -> uint64_t {
        if (d.activation == Decoder::ActivationMode::INDIRECT_ADDR) {
            // ISA v1.7: dram_addr + a uint32 read from SRAM. The pointer's SRAM address is
            // the whole dim_k wire word (dim_k | c_slot<<8 | dram_hi<<12), so dram_hi is
            // spent on the pointer and indirect transfers address below 4 GB.
            const uint16_t ptr = static_cast<uint16_t>(d.dim_k | (d.c_slot << 8) | (d.dram_hi << 12));
            uint32_t off;
            std::memcpy(&off, mem.sram_ptr(ptr), sizeof(off));
            return static_cast<uint64_t>(d.dram_addr) + off;
        }
        return (static_cast<uint64_t>(d.dram_hi) << 32) | d.dram_addr;
    };

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
    // One DMA engine serves every LOAD and STORE: transfers queue behind each
    // other (dma_free = cycle the engine can start the next one), and only the
    // fixed DRAM latency overlaps between requests. Without this, any number
    // of in-flight loads completed independently, which no real engine does
    // and which made deep prefetch look free.
    uint64_t dma_free = 0;
    auto xfer_cycles = [&](uint64_t bytes) { return (bytes + dma_bpc - 1) / dma_bpc; };
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
        mem.load_tile(dram_addr36(d), d.sram_addr, d.dim_m, d.dim_n, stride);

        const uint64_t xfer  = xfer_cycles(static_cast<uint64_t>(d.dim_m) * d.dim_n);
        const uint64_t start = std::max(sim_clock, dma_free);
        dma_free = start + xfer;
        if (d.dep_id < 16)
            dep_done[d.dep_id] = start + xfer + dram_latency;

        serial_cycles += dram_latency + xfer;
        ++cnt[1];
    });

    // STORE_TILE: SRAM → DRAM.
    // Treated as blocking: the CPU cannot proceed until the write completes.
    disp.on(Decoder::Opcode::STORE_TILE, [&](const Decoder::DecodedInstruction& d) {
        const uint16_t stride = d.stride ? d.stride : d.dim_n;
        mem.store_tile(d.sram_addr, dram_addr36(d), d.dim_m, d.dim_n, stride);

        const uint64_t xfer  = xfer_cycles(static_cast<uint64_t>(d.dim_m) * d.dim_n);
        const uint64_t start = std::max(sim_clock, dma_free);
        dma_free  = start + xfer;
        sim_clock = start + xfer + dram_latency;   // blocking; waits its turn
        serial_cycles += dram_latency + xfer;
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
        reject_bf16w(d);
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
        reject_bf16w(d);
        const uint16_t tile_sram = d.stride ? static_cast<uint16_t>(d.stride)
                                            : SRAM_INPUT_BASE;
        if (d.activation == Decoder::ActivationMode::REDUCE_ARGMAX) {
            // ISA v1.7: index of the FIRST maximum over a dim_m x dim_n FP32 tile (row-major),
            // times an immediate byte multiplier (dram_addr, 1 if 0), as a uint32 at sram_addr.
            // This is the missing half of data-dependent addressing: REDUCE_MAX yields the value
            // but not where it was; INDIRECT_ADDR then uses this word as a DRAM offset.
            if (d.dtype != Decoder::DType::FP32)
                throw std::runtime_error("REDUCE_ARGMAX is FP32-only");
            const std::size_t n = static_cast<std::size_t>(d.dim_m) * d.dim_n;
            if (n == 0 || static_cast<std::size_t>(tile_sram) + n * 4 > SRAM_BYTES_LIMIT)
                throw std::runtime_error("REDUCE_ARGMAX: tile out of SRAM range");
            std::vector<float> t(n);
            std::memcpy(t.data(), mem.sram_ptr(tile_sram), n * 4);
            std::size_t best = 0;
            for (std::size_t i = 1; i < n; ++i)
                if (t[i] > t[best]) best = i;
            const uint32_t out = static_cast<uint32_t>(best) * (d.dram_addr ? d.dram_addr : 1u);
            std::memcpy(mem.sram_ptr(d.sram_addr), &out, sizeof(out));
            const uint64_t cost = static_cast<uint64_t>(d.dim_m) * d.dim_n;
            sim_clock += cost;
            serial_cycles += cost;
            ++cnt[6];
            return;
        }
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
        reject_bf16w(d);
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

    // ── Persistent session (--serve) ──────────────────────────────────────────
    // One process holds DRAM across many programs, so weights are loaded once
    // and each program is just submitted, like a command buffer to a device.
    // Commands, one per line on stdin; every reply is one line on stdout:
    //   RUN <prog.bin> [ADDR:LEN:FILE ...]  run a program, then dump regions
    //   WRITE <addr> <file>                 copy a file into DRAM at addr
    //   QUIT
    // Replies: "OK instrs=N serial=S db=D" or "ERR <message>". Timing state is
    // reset per RUN; DRAM and SRAM contents persist.
    if (serve) {
        std::setvbuf(stdout, nullptr, _IOLBF, 0);
        std::printf("READY\n");
        std::string line;
        while (std::getline(std::cin, line)) {
            std::istringstream in(line);
            std::string cmd;
            in >> cmd;
            if (cmd.empty()) continue;
            if (cmd == "QUIT") break;
            try {
                if (cmd == "WRITE") {
                    std::string addr_s, file;
                    in >> addr_s >> file;
                    std::ifstream f(file, std::ios::binary);
                    if (!f) throw std::runtime_error("cannot open " + file);
                    const std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)),
                                                  std::istreambuf_iterator<char>());
                    mem.write_dram(static_cast<uint32_t>(std::stoull(addr_s, nullptr, 0)),
                                   b.data(), b.size());
                    std::printf("OK wrote=%zu\n", b.size());
                } else if (cmd == "RUN") {
                    std::string prog;
                    in >> prog;
                    std::vector<DumpSpec> ds;
                    for (std::string spec; in >> spec;) {
                        const auto c1 = spec.find(':'), c2 = spec.find(':', c1 + 1);
                        if (c1 == std::string::npos || c2 == std::string::npos)
                            throw std::runtime_error("bad dump spec " + spec);
                        ds.push_back({static_cast<uint32_t>(std::stoull(spec.substr(0, c1), nullptr, 0)),
                                      static_cast<std::size_t>(std::stoull(spec.substr(c1 + 1, c2 - c1 - 1), nullptr, 0)),
                                      spec.substr(c2 + 1)});
                    }
                    sim_clock = 0; serial_cycles = 0; dma_free = 0;
                    std::fill(std::begin(dep_done), std::end(dep_done), 0);
                    std::fill(std::begin(cnt), std::end(cnt), 0);
                    disp.load_binary(prog);
                    disp.dispatch_all();
                    for (const auto& d : ds) {
                        std::vector<uint8_t> buf(d.len);
                        mem.read_dram(d.addr, buf.data(), buf.size());
                        std::ofstream out(d.path, std::ios::binary | std::ios::trunc);
                        out.write(reinterpret_cast<const char*>(buf.data()),
                                  static_cast<std::streamsize>(buf.size()));
                        if (!out) throw std::runtime_error("write failed: " + d.path);
                    }
                    std::printf("OK instrs=%zu serial=%llu db=%llu\n", disp.instruction_count(),
                                (unsigned long long)serial_cycles, (unsigned long long)sim_clock);
                } else {
                    throw std::runtime_error("unknown command " + cmd);
                }
            } catch (const std::exception& e) {
                std::printf("ERR %s\n", e.what());
            }
        }
        return 0;
    }

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
        // write() doesn't throw on failure -- it just sets the stream's
        // failbit, which nothing here checked before. A full disk/tmpfs
        // (this project's own callers round-trip DRAM images through
        // /tmp, which can be a size-capped RAM-backed tmpfs) then produced
        // a silently truncated or empty file with exit code 0 -- a caller
        // reading it back saw "0 bytes", not a clear error, which is
        // exactly as confusing as it sounds (confirmed hitting this
        // directly running toolchain/generate.py at Qwen3-0.6B's real
        // 28-layer scale: /tmp filled up mid-run and this line is the
        // only place that would have caught it before the caller did).
        if (!out) {
            std::fprintf(stderr,
                "error: write to --dump file %s failed (disk/tmpfs full?)\n",
                d.path.c_str());
            return 1;
        }
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
