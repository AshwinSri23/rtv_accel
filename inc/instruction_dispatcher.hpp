#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include "decoder.hpp"

// ─── InstructionDispatcher ────────────────────────────────────────────────────
//
// Reads a binary instruction file produced by toolchain/npu_compiler.py
// and dispatches each decoded instruction to a registered per-opcode handler.
//
// File format: flat array of 128-bit big-endian instruction words (16 bytes each),
// matching the output of NPUCompiler.export_binary().
//
// The dispatcher byte-swaps each 64-bit word from big-endian (file order) to
// native little-endian (ARM64 runtime order) before passing to Decoder::decode().
//
// Typical usage:
//
//   InstructionDispatcher disp;
//
//   disp.on(Decoder::Opcode::COMPUTE_GEMM, [](const Decoder::DecodedInstruction& d) {
//       run_neon_systolic_array(d.sram_addr, d.dim_m, d.dim_n, d.dim_k);
//   });
//   disp.on(Decoder::Opcode::LOAD_TILE, [&mem](const Decoder::DecodedInstruction& d) {
//       mem.load_tile(d.dram_addr, d.sram_addr, d.dim_m, d.dim_n, d.stride);
//   });
//
//   disp.load_binary("smollm2_layer0_attention.bin");
//   disp.dispatch_all();
//   std::printf("Dispatched %zu instructions\n", disp.instruction_count());

class InstructionDispatcher {
public:
    using Handler = std::function<void(const Decoder::DecodedInstruction&)>;

    InstructionDispatcher() = default;

    // ── Handler registration ─────────────────────────────────────────────────

    // Register a handler for a specific opcode.
    // Only one handler per opcode; calling on() again replaces the previous one.
    void on(Decoder::Opcode opcode, Handler handler);

    // Register a catch-all handler invoked for opcodes without a specific handler.
    void on_unhandled(Handler handler);

    // ── Binary loading ───────────────────────────────────────────────────────

    // Load and parse the binary file produced by NPUCompiler.export_binary().
    // Each 16-byte chunk is byte-swapped from big-endian to native endian and
    // stored as a RawInstruction.
    // Throws std::runtime_error if the file cannot be opened or has a bad size.
    void load_binary(const std::string& filepath);

    // Alternatively, load instructions from an in-memory buffer (e.g. mmap).
    // buf must be a multiple of 16 bytes; big-endian byte order assumed.
    void load_buffer(const uint8_t* buf, std::size_t byte_len);

    // ── Dispatch ─────────────────────────────────────────────────────────────

    // Decode and dispatch every loaded instruction in order.
    // Calls Decoder::decode() on each RawInstruction then routes to the
    // matching handler (or the unhandled handler if none is registered).
    void dispatch_all();

    // Decode and dispatch a single instruction by index (0-based).
    void dispatch_one(std::size_t index);

    // ── Introspection ────────────────────────────────────────────────────────

    std::size_t instruction_count() const { return _instructions.size(); }

    // Return the raw instruction word at index (native endian, post-bswap).
    const Decoder::RawInstruction& raw_at(std::size_t index) const;

    // Decode the instruction at index without dispatching it.
    Decoder::DecodedInstruction decode_at(std::size_t index) const;

private:
    std::vector<Decoder::RawInstruction> _instructions;

    // Handler table indexed by opcode value (0–15).
    // Slot 0 is unused (reserved opcode); valid opcodes are 1–7.
    Handler _handlers[16];
    Handler _unhandled;

    // Byte-swap a big-endian 64-bit word to native (little-endian on ARM64).
    static uint64_t bswap64(uint64_t v);
};
