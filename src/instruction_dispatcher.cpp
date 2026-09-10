#include "../inc/instruction_dispatcher.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

// ── Handler registration ──────────────────────────────────────────────────────

void InstructionDispatcher::on(Decoder::Opcode opcode, Handler handler) {
    _handlers[static_cast<uint8_t>(opcode)] = std::move(handler);
}

void InstructionDispatcher::on_unhandled(Handler handler) {
    _unhandled = std::move(handler);
}

// ── Binary loading ────────────────────────────────────────────────────────────

// The Python compiler writes 128-bit words as big-endian pairs of uint64_t
// (struct.pack(">QQ", hi, lo)).  On ARM64 (little-endian) we must bswap each
// 64-bit word before storing it in native-endian RawInstruction fields.

uint64_t InstructionDispatcher::bswap64(uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap64(v);
#else
    // Portable fallback
    return ((v & 0xFF00000000000000ULL) >> 56)
         | ((v & 0x00FF000000000000ULL) >> 40)
         | ((v & 0x0000FF0000000000ULL) >> 24)
         | ((v & 0x000000FF00000000ULL) >>  8)
         | ((v & 0x00000000FF000000ULL) <<  8)
         | ((v & 0x0000000000FF0000ULL) << 24)
         | ((v & 0x000000000000FF00ULL) << 40)
         | ((v & 0x00000000000000FFULL) << 56);
#endif
}

void InstructionDispatcher::load_binary(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("InstructionDispatcher: cannot open '" + filepath + "'");

    const std::streamsize size = file.tellg();
    if (size % 16 != 0)
        throw std::runtime_error(
            "InstructionDispatcher: file size " + std::to_string(size) +
            " is not a multiple of 16 bytes");

    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> raw(static_cast<std::size_t>(size));
    file.read(reinterpret_cast<char*>(raw.data()), size);

    load_buffer(raw.data(), static_cast<std::size_t>(size));
}

void InstructionDispatcher::load_buffer(const uint8_t* buf, std::size_t byte_len) {
    if (byte_len % 16 != 0)
        throw std::runtime_error(
            "InstructionDispatcher: buffer size is not a multiple of 16 bytes");

    _instructions.clear();
    _instructions.reserve(byte_len / 16);

    for (std::size_t offset = 0; offset < byte_len; offset += 16) {
        // Read two 64-bit values from the buffer in their raw byte order
        uint64_t hi_be, lo_be;
        std::memcpy(&hi_be, buf + offset,     8);
        std::memcpy(&lo_be, buf + offset + 8, 8);

        // Byte-swap from big-endian (file/Python) to native (ARM64 little-endian)
        _instructions.push_back({ bswap64(hi_be), bswap64(lo_be) });
    }
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

void InstructionDispatcher::dispatch_all() {
    for (std::size_t i = 0; i < _instructions.size(); ++i)
        dispatch_one(i);
}

void InstructionDispatcher::dispatch_one(std::size_t index) {
    const Decoder::DecodedInstruction d = Decoder::decode(_instructions[index]);
    const uint8_t key = static_cast<uint8_t>(d.opcode);

    if (_handlers[key]) {
        _handlers[key](d);
        return;
    }
    if (_unhandled) {
        _unhandled(d);
    }
}

// ── Introspection ─────────────────────────────────────────────────────────────

const Decoder::RawInstruction& InstructionDispatcher::raw_at(std::size_t index) const {
    if (index >= _instructions.size())
        throw std::out_of_range("InstructionDispatcher::raw_at: index out of range");
    return _instructions[index];
}

Decoder::DecodedInstruction InstructionDispatcher::decode_at(std::size_t index) const {
    return Decoder::decode(raw_at(index));
}
