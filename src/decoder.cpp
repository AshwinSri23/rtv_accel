#include "../inc/decoder.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>

// ─── Private helpers ──────────────────────────────────────────────────────────

// Convert a raw 4-bit opcode value to the typed enum.
// Unknown values produce a descriptive exception that includes the decimal
// value so debug output stays readable alongside hex instruction dumps.
Decoder::Opcode Decoder::parse_opcode(uint8_t bits) {
    // 0x0 is intentionally unassigned (reserved).  Values 0x1–0x7 are valid.
    switch (bits) {
        case 0x1: return Opcode::LOAD_TILE;
        case 0x2: return Opcode::STORE_TILE;
        case 0x3: return Opcode::COMPUTE_GEMM;
        case 0x4: return Opcode::EXEC_ALU;
        case 0x5: return Opcode::WAIT_SYNC;
        case 0x6: return Opcode::REDUCE;
        case 0x7: return Opcode::SCALAR_ALU;
        default:
            throw std::invalid_argument(
                std::string("Unknown opcode: ") +
                std::to_string(static_cast<unsigned>(bits)));
    }
}

// Convert a raw activation value to the typed enum.
// Caller must strip the v1.5 accumulate flag (bit 7) before calling — only
// bits [6:0] are presented here.
Decoder::ActivationMode Decoder::parse_activation(uint8_t bits) {
    switch (bits) {
        case 0x00: return ActivationMode::NONE;
        case 0x01: return ActivationMode::RELU;
        case 0x02: return ActivationMode::INT8_QUANT;
        case 0x03: return ActivationMode::SILU;
        case 0x04: return ActivationMode::SQUARE;
        case 0x05: return ActivationMode::RSQRT;
        case 0x06: return ActivationMode::ELEMWISE_MUL;
        case 0x07: return ActivationMode::SCALAR_ADD;
        case 0x08: return ActivationMode::ELEMWISE_ADD;
        case 0x09: return ActivationMode::TRANSPOSE;
        case 0x0A: return ActivationMode::REDUCE_MAX;
        case 0x0B: return ActivationMode::EXP_SUB;
        case 0x0C: return ActivationMode::SCALAR_DIV;
        case 0x0D: return ActivationMode::CAUSAL_MASK;
        case 0x0E: return ActivationMode::REDUCE_SUMSQ;
        case 0x0F: return ActivationMode::RMS_SCALE;
        case 0x10: return ActivationMode::PERCHAN_QUANT;
        case 0x11: return ActivationMode::RESCALE;
        case 0x12: return ActivationMode::LUT;
        default:
            throw std::invalid_argument(
                std::string("Unknown activation mode: ") +
                std::to_string(static_cast<unsigned>(bits)));
    }
}

// ─── decode() ────────────────────────────────────────────────────────────────

// Each field is extracted by right-shifting to position 0 and masking to its
// declared width.  Opcode and activation are validated immediately via their
// parse_* helpers so that an invalid instruction never produces a
// partially-initialised DecodedInstruction.
//
// dim_m straddles the hi/lo boundary: upper byte in hi[7:0], lower byte in lo[63:56].
Decoder::DecodedInstruction Decoder::decode(const RawInstruction& raw) {
    const uint64_t hi = raw.hi;
    const uint64_t lo = raw.lo;

    DecodedInstruction d{};

    // hi word: opcode[63:60] | dep_id[59:56] | dram_addr[55:24] | sram_addr[23:8] | dim_m_wire[15:8]
    d.opcode    = parse_opcode(static_cast<uint8_t>((hi >> 60) & 0xF));
    d.dep_id    = static_cast<uint8_t>((hi >> 56) & 0xF);
    d.dram_addr = static_cast<uint32_t>((hi >> 24) & 0xFFFF'FFFF);
    d.sram_addr = static_cast<uint16_t>((hi >>  8) & 0xFFFF);

    // dim_m straddles hi/lo: upper byte in hi[7:0], lower byte in lo[63:56].
    // v1.5: wire bits [7:0] = tile rows; bits [12:8] = scale_shift; [15:13] reserved.
    const uint16_t wire_dim_m =
        static_cast<uint16_t>(((hi & 0xFF) << 8) | ((lo >> 56) & 0xFF));
    d.dim_m       = static_cast<uint8_t>(wire_dim_m & 0xFF);
    d.scale_shift = static_cast<uint8_t>((wire_dim_m >> 8) & 0x1F);
    // bits 14:13 -- 0 decodes as INT8, so a v1/v1.5 word is unchanged.
    d.dtype       = static_cast<DType>((wire_dim_m >> 13) & 0x3);

    // lo word: dim_m_lo[63:56] | dim_n[55:40] | dim_k_wire[39:24] | stride[23:8] | activation_wire[7:0]
    d.dim_n = static_cast<uint16_t>((lo >> 40) & 0xFFFF);

    // v1.5: dim_k wire bits [7:0] = tile depth; bits [11:8] = c_slot; [15:12] reserved.
    const uint16_t wire_dim_k = static_cast<uint16_t>((lo >> 24) & 0xFFFF);
    d.dim_k  = static_cast<uint8_t>(wire_dim_k & 0xFF);
    d.c_slot = static_cast<uint8_t>((wire_dim_k >> 8) & 0x0F);

    d.stride = static_cast<uint16_t>((lo >> 8) & 0xFFFF);

    // v1.5: activation wire bit 7 = accumulate flag; bits [6:0] = mode enum.
    const uint8_t act_wire = static_cast<uint8_t>(lo & 0xFF);
    d.accumulate = (act_wire >> 7) & 1;
    d.activation = parse_activation(act_wire & 0x7F);

    return d;
}

// ─── encode() ────────────────────────────────────────────────────────────────

// Pack a DecodedInstruction back into a RawInstruction by shifting each field
// into its designated bit position and ORing the pieces together.
// dram_addr is a full 32-bit value; it fits exactly in hi[55:24] without masking.
// dim_m is split: upper byte goes into hi[7:0], lower byte into lo[63:56].
Decoder::RawInstruction Decoder::encode(const DecodedInstruction& d) {
    // v1.5: pack sub-fields back into the wire words before building hi/lo.
    // wire_dim_m: bits [12:8] = scale_shift, bits [7:0] = tile rows
    const uint16_t wire_dim_m =
        (static_cast<uint16_t>(static_cast<uint8_t>(d.dtype) & 0x3) << 13) |
        (static_cast<uint16_t>(d.scale_shift & 0x1F) << 8) | d.dim_m;
    // wire_dim_k: bits [11:8] = c_slot, bits [7:0] = tile depth
    const uint16_t wire_dim_k =
        (static_cast<uint16_t>(d.c_slot & 0x0F) << 8) | d.dim_k;
    // wire_activation: bit 7 = accumulate, bits [6:0] = mode enum
    const uint8_t wire_act =
        (static_cast<uint8_t>(d.accumulate ? 0x80 : 0x00)) |
        (static_cast<uint8_t>(d.activation) & 0x7F);

    // Reconstruct hi: opcode[63:60] dep_id[59:56] dram[55:24] sram[23:8] wire_dim_m_hi[7:0]
    const uint64_t hi =
        (static_cast<uint64_t>(d.opcode)              << 60) |
        (static_cast<uint64_t>(d.dep_id)              << 56) |
        (static_cast<uint64_t>(d.dram_addr)           << 24) |
        (static_cast<uint64_t>(d.sram_addr)           <<  8) |
        (static_cast<uint64_t>(wire_dim_m >> 8));               // upper byte of wire_dim_m

    // Reconstruct lo: wire_dim_m_lo[63:56] dim_n[55:40] wire_dim_k[39:24] stride[23:8] wire_act[7:0]
    const uint64_t lo =
        (static_cast<uint64_t>(wire_dim_m & 0xFF)     << 56) |
        (static_cast<uint64_t>(d.dim_n)               << 40) |
        (static_cast<uint64_t>(wire_dim_k)            << 24) |
        (static_cast<uint64_t>(d.stride)              <<  8) |
        (static_cast<uint64_t>(wire_act));

    return RawInstruction{hi, lo};
}

// ─── to_string() ─────────────────────────────────────────────────────────────

// File-local helpers map enums back to their mnemonic strings.
static const char* opcode_name(Decoder::Opcode op) {
    switch (op) {
        case Decoder::Opcode::LOAD_TILE:    return "LOAD_TILE";
        case Decoder::Opcode::STORE_TILE:   return "STORE_TILE";
        case Decoder::Opcode::COMPUTE_GEMM: return "COMPUTE_GEMM";
        case Decoder::Opcode::EXEC_ALU:     return "EXEC_ALU";
        case Decoder::Opcode::WAIT_SYNC:    return "WAIT_SYNC";
        case Decoder::Opcode::REDUCE:       return "REDUCE";
        case Decoder::Opcode::SCALAR_ALU:   return "SCALAR_ALU";
        default:                            return "UNKNOWN";
    }
}

static const char* activation_name(Decoder::ActivationMode am) {
    switch (am) {
        case Decoder::ActivationMode::NONE:         return "NONE";
        case Decoder::ActivationMode::RELU:         return "RELU";
        case Decoder::ActivationMode::INT8_QUANT:   return "INT8_QUANT";
        case Decoder::ActivationMode::SILU:         return "SILU";
        case Decoder::ActivationMode::SQUARE:       return "SQUARE";
        case Decoder::ActivationMode::RSQRT:        return "RSQRT";
        case Decoder::ActivationMode::ELEMWISE_MUL: return "ELEMWISE_MUL";
        case Decoder::ActivationMode::SCALAR_ADD:   return "SCALAR_ADD";
        case Decoder::ActivationMode::ELEMWISE_ADD: return "ELEMWISE_ADD";
        case Decoder::ActivationMode::TRANSPOSE:    return "TRANSPOSE";
        case Decoder::ActivationMode::REDUCE_MAX:   return "REDUCE_MAX";
        case Decoder::ActivationMode::EXP_SUB:      return "EXP_SUB";
        case Decoder::ActivationMode::SCALAR_DIV:   return "SCALAR_DIV";
        case Decoder::ActivationMode::CAUSAL_MASK:  return "CAUSAL_MASK";
        case Decoder::ActivationMode::REDUCE_SUMSQ: return "REDUCE_SUMSQ";
        case Decoder::ActivationMode::RMS_SCALE:    return "RMS_SCALE";
        case Decoder::ActivationMode::PERCHAN_QUANT: return "PERCHAN_QUANT";
        case Decoder::ActivationMode::RESCALE:       return "RESCALE";
        case Decoder::ActivationMode::LUT:           return "LUT";
        default:                                    return "UNKNOWN";
    }
}

// Format mirrors the ISA field order for easy comparison against a hex dump.
// dram_addr is printed as 8 hex digits to reflect the full 32-bit range.
// v1.5 sub-fields (accumulate, scale_shift, c_slot) are printed only when nonzero.
std::string Decoder::to_string(const DecodedInstruction& d) {
    char buf[192];
    int n = std::snprintf(buf, sizeof(buf),
        "%s dep=%u dram=0x%08X sram=0x%04X "
        "M=%u N=%u K=%u stride=%u act=%s",
        opcode_name(d.opcode),
        d.dep_id,
        d.dram_addr,
        d.sram_addr,
        d.dim_m, d.dim_n, d.dim_k,
        d.stride,
        activation_name(d.activation));
    if (d.accumulate)
        n += std::snprintf(buf + n, sizeof(buf) - static_cast<std::size_t>(n), " acc=1");
    if (d.scale_shift)
        n += std::snprintf(buf + n, sizeof(buf) - static_cast<std::size_t>(n), " sshift=%u", d.scale_shift);
    if (d.c_slot)
        std::snprintf(buf + n, sizeof(buf) - static_cast<std::size_t>(n), " cslot=%u", d.c_slot);
    return std::string(buf);
}
