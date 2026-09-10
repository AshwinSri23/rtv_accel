#include "../inc/memory_controller.hpp"
#include <cstring>
#include <stdexcept>

MemoryController::MemoryController(uint32_t dram_latency_cycles)
    : _dram_latency(dram_latency_cycles) {}

void MemoryController::resize_dram(std::size_t bytes) {
    _dram.resize(bytes);
}

void MemoryController::write_dram(uint32_t addr, const uint8_t* src, std::size_t len) {
    if (static_cast<std::size_t>(addr) + len > _dram.size())
        throw std::out_of_range("write_dram: address out of range");
    std::memcpy(_dram.data() + addr, src, len);
}

void MemoryController::read_dram(uint32_t addr, uint8_t* dst, std::size_t len) const {
    if (static_cast<std::size_t>(addr) + len > _dram.size())
        throw std::out_of_range("read_dram: address out of range");
    std::memcpy(dst, _dram.data() + addr, len);
}

void MemoryController::load_tile(uint32_t dram_addr, uint16_t sram_addr,
                                 uint16_t rows, uint16_t cols, uint16_t stride)
{
    // Bounds check: last DRAM byte read is at dram_addr + (rows-1)*stride + (cols-1)
    if (rows > 0 && cols > 0) {
        std::size_t last_dram = static_cast<std::size_t>(dram_addr)
                              + static_cast<std::size_t>(rows - 1) * stride
                              + cols;
        if (last_dram > _dram.size())
            throw std::out_of_range("load_tile: DRAM address out of range");
    }
    std::size_t sram_end = static_cast<std::size_t>(sram_addr)
                         + static_cast<std::size_t>(rows) * cols;
    if (sram_end > SRAM_BYTES)
        throw std::out_of_range("load_tile: SRAM address out of range");

    for (uint16_t r = 0; r < rows; ++r) {
        std::memcpy(_sram.data() + sram_addr + r * cols,
                    _dram.data() + dram_addr + static_cast<std::size_t>(r) * stride,
                    cols);
    }
    _cycles += _dram_latency + static_cast<uint64_t>(rows) * cols;
}

void MemoryController::store_tile(uint16_t sram_addr, uint32_t dram_addr,
                                  uint16_t rows, uint16_t cols, uint16_t stride)
{
    std::size_t sram_end = static_cast<std::size_t>(sram_addr)
                         + static_cast<std::size_t>(rows) * cols;
    if (sram_end > SRAM_BYTES)
        throw std::out_of_range("store_tile: SRAM address out of range");

    if (rows > 0 && cols > 0) {
        std::size_t last_dram = static_cast<std::size_t>(dram_addr)
                              + static_cast<std::size_t>(rows - 1) * stride
                              + cols;
        if (last_dram > _dram.size())
            throw std::out_of_range("store_tile: DRAM address out of range");
    }

    for (uint16_t r = 0; r < rows; ++r) {
        std::memcpy(_dram.data() + dram_addr + static_cast<std::size_t>(r) * stride,
                    _sram.data() + sram_addr + r * cols,
                    cols);
    }
    _cycles += _dram_latency + static_cast<uint64_t>(rows) * cols;
}

uint8_t* MemoryController::sram_ptr(uint16_t offset) {
    if (offset >= SRAM_BYTES)
        throw std::out_of_range("sram_ptr: offset out of range");
    return _sram.data() + offset;
}

const uint8_t* MemoryController::sram_ptr(uint16_t offset) const {
    if (offset >= SRAM_BYTES)
        throw std::out_of_range("sram_ptr: offset out of range");
    return _sram.data() + offset;
}

uint64_t MemoryController::cycle_count() const { return _cycles; }
void     MemoryController::reset_cycles()       { _cycles = 0; }
