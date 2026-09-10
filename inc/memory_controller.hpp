#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

// ─── MemoryController ─────────────────────────────────────────────────────────
//
// Cycle-approximate behavioral model of the NPU memory subsystem.
//
// Two tiers:
//   DRAM  — std::vector<uint8_t> backed by host RAM.  Variable size; call
//            resize_dram() before loading weights.  Access costs
//            _dram_latency + transfer_bytes cycles per tile op.
//   SRAM  — 64 KiB std::array<uint8_t, SRAM_BYTES> scratchpad.  Fixed size.
//            The PE array reads and writes it directly via sram_ptr().
//            Access costs 1 cycle.
//
// Zone layout (matches SRAMMap in toolchain/npu_compiler.py):
//   0x0000–0x03FF  Input tiles     (4 × 256 B slots)
//   0x0400–0x07FF  Weight tiles    (4 × 256 B slots)
//   0x0800–0x0BFF  Output / accum  (4 × 256 B slots)
//   0x0C00–0xFFFF  Scratch / free
//
// Tile transfer — stride semantics:
//   load_tile / store_tile copy `rows` rows of `cols` bytes each.
//   In DRAM the source is embedded in a larger row-major matrix; `stride`
//   is the byte distance between the start of consecutive rows in DRAM
//   (e.g. 1024 for a 1024-column INT8 matrix).
//   In SRAM the tile is always stored contiguously (stride = cols).
//
// Throws std::out_of_range if any address is out of bounds.
//
// ─────────────────────────────────────────────────────────────────────────────

static constexpr std::size_t SRAM_BYTES = 65536;  // 64 KiB, matches hal_config.json

class MemoryController {
public:
    // dram_latency_cycles matches hal_config.json "dram_latency_cycles" (default 120).
    explicit MemoryController(uint32_t dram_latency_cycles = 120);

    // ── DRAM management ──────────────────────────────────────────────────────

    // Allocate (or grow) the simulated DRAM buffer.
    // Safe to call multiple times; existing data below the new size is preserved.
    void resize_dram(std::size_t bytes);

    // Copy `len` bytes from `src` into DRAM at `addr`.
    // Used by the weight loader to populate DRAM before running the simulator.
    // Throws std::out_of_range if addr + len exceeds the allocated DRAM size.
    void write_dram(uint32_t addr, const uint8_t* src, std::size_t len);

    // Read `len` bytes from DRAM at `addr` into `dst`.
    // Throws std::out_of_range if addr + len exceeds the allocated DRAM size.
    void read_dram(uint32_t addr, uint8_t* dst, std::size_t len) const;

    // ── Tile transfers (handlers for LOAD_TILE / STORE_TILE) ─────────────────

    // Copy a rows×cols tile from DRAM into SRAM.
    //   dram_addr  — byte address in DRAM of the top-left element
    //   sram_addr  — byte offset in SRAM to write the tile (stored contiguously)
    //   rows, cols — tile dimensions in elements (typically 16×16)
    //   stride     — byte distance between row starts in DRAM
    //                (equals the full matrix column count for INT8)
    // Adds dram_latency + rows*cols to the cycle counter.
    // Throws std::out_of_range on any out-of-bounds access.
    void load_tile(uint32_t dram_addr, uint16_t sram_addr,
                   uint16_t rows, uint16_t cols, uint16_t stride);

    // Copy a rows×cols tile from SRAM back to DRAM.
    // Inverse of load_tile; stride has the same meaning.
    // Adds dram_latency + rows*cols to the cycle counter.
    // Throws std::out_of_range on any out-of-bounds access.
    void store_tile(uint16_t sram_addr, uint32_t dram_addr,
                    uint16_t rows, uint16_t cols, uint16_t stride);

    // ── Direct SRAM access for the PE array ──────────────────────────────────

    // Return a raw pointer into SRAM at `offset`.
    // The PE array uses this to read inputs and write outputs directly,
    // avoiding a copy through the tile-transfer path.
    // Throws std::out_of_range if offset >= SRAM_BYTES.
          uint8_t* sram_ptr(uint16_t offset);
    const uint8_t* sram_ptr(uint16_t offset) const;

    // ── Cycle counter ─────────────────────────────────────────────────────────

    // Running total of modeled cycles accumulated across all tile ops.
    uint64_t cycle_count() const;

    // Reset to zero — call between benchmark runs.
    void reset_cycles();

private:
    std::vector<uint8_t>            _dram;
    std::array<uint8_t, SRAM_BYTES> _sram{};
    uint64_t                        _cycles      = 0;
    uint32_t                        _dram_latency;
};
