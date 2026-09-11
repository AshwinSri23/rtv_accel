cmake_minimum_required(VERSION 3.16)
project(rtv_accel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Decoder static library ───────────────────────────────────────────────────
# Bit-exact ISA encoder/decoder.  All other targets link this.

add_library(rtv_decoder STATIC src/decoder.cpp)
target_include_directories(rtv_decoder PUBLIC inc)

# ─── Dispatcher static library ────────────────────────────────────────────────
# Reads binary payloads produced by the compiler (see the "Compiler" section
# below), byte-swaps from big-endian to native, and routes decoded
# instructions to registered per-opcode handlers via InstructionDispatcher.

add_library(rtv_dispatcher STATIC src/instruction_dispatcher.cpp)
target_include_directories(rtv_dispatcher PUBLIC inc)
target_link_libraries(rtv_dispatcher PUBLIC rtv_decoder)

# ─── Test executables ─────────────────────────────────────────────────────────

enable_testing()

add_executable(decoder_test tests/decoder_test.cpp)
target_include_directories(decoder_test PRIVATE inc)
target_link_libraries(decoder_test PRIVATE rtv_decoder)
add_test(NAME decoder_test COMMAND decoder_test)

add_executable(dispatcher_test tests/dispatcher_test.cpp)
target_include_directories(dispatcher_test PRIVATE inc)
target_link_libraries(dispatcher_test PRIVATE rtv_dispatcher)
add_test(NAME dispatcher_test COMMAND dispatcher_test)

# ─── MemoryController static library ─────────────────────────────────────────

add_library(rtv_memory STATIC src/memory_controller.cpp)
target_include_directories(rtv_memory PUBLIC inc)

# ─── PEArray static library ───────────────────────────────────────────────────

add_library(rtv_pe STATIC src/pe_array.cpp)
target_include_directories(rtv_pe PUBLIC inc)
target_link_libraries(rtv_pe PUBLIC rtv_memory)

# Enable Neon vdotq_s32 on AArch64 (Raspberry Pi 5 / Cortex-A76)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_definitions(rtv_pe PRIVATE CHIP_SIM_USE_NEON)
    target_compile_options(rtv_pe PRIVATE -march=armv8.2-a+dotprod)
else()
    # x86-64: enable the AVX2 GEMM path when the BUILD MACHINE can actually
    # run it. check_cxx_source_runs (not _compiles) so a toolchain that
    # accepts -mavx2 on a CPU lacking it does not produce a binary that
    # SIGILLs at runtime.
    include(CheckCXXSourceRuns)
    set(CMAKE_REQUIRED_FLAGS "-mavx2")
    check_cxx_source_runs("
        #include <immintrin.h>
        int main() {
            __m256i a = _mm256_set1_epi32(3), b = _mm256_set1_epi32(4);
            return _mm256_extract_epi32(_mm256_mullo_epi32(a, b), 0) == 12 ? 0 : 1;
        }" RTV_HAVE_AVX2)
    unset(CMAKE_REQUIRED_FLAGS)
    if(RTV_HAVE_AVX2)
        target_compile_definitions(rtv_pe PRIVATE CHIP_SIM_USE_AVX2)
        target_compile_options(rtv_pe PRIVATE -mavx2)
        message(STATUS "PE array: AVX2 GEMM path enabled")
    else()
        message(STATUS "PE array: scalar GEMM path (no AVX2)")
    endif()
endif()

add_executable(pe_array_test tests/pe_array_test.cpp)
target_include_directories(pe_array_test PRIVATE inc)
target_link_libraries(pe_array_test PRIVATE rtv_pe rtv_memory)
add_test(NAME pe_array_test COMMAND pe_array_test)

# ─── Compiler ────────────────────────────────────────────────────────────────
# The compiler is the TVM path (toolchain/tvm_backend.py, composing
# toolchain/rtv_tvm/{patterns,schedule,codegen,intrin}.py): a Relax graph is
# pattern-matched, scheduled and tensorized against this ISA's intrinsics,
# then walked straight to instructions -- TVM owns addressing, this project
# owns only the ISA and the simulator that executes it. A hand-written
# Python compiler (npu_compiler.py) and, earlier still, a C++ frontend
# (SLMAllocator, ir.hpp, Emitter, compiler.cpp, compiler_main) both preceded
# this and have been removed in turn, each for the same reason: a second
# lowering path meant every ISA change had to be made twice, and TVM's typed
# IR computes the addressing arithmetic that produced several of this
# project's bugs when done by hand. toolchain/compiler_state.py carries what
# both the old and new paths needed regardless of which was doing the
# lowering (HAL config loading, the DRAM staging allocator, RoPE's host-side
# table math) -- not a compiler itself, just shared device/runtime state.

# ─── Top-level simulator ──────────────────────────────────────────────────────

add_executable(rtv_sim src/simulator.cpp)
target_include_directories(rtv_sim PRIVATE inc)
target_link_libraries(rtv_sim PRIVATE rtv_dispatcher rtv_pe rtv_memory)

# ─── TVM compilation path ─────────────────────────────────────────────────────
# Each test below compiles operations through the TVM path (TIR scheduled
# and tensorized against this ISA's intrinsics, lowered to instructions by
# toolchain/rtv_tvm/codegen.py), executes them on rtv_sim against a real
# DRAM image, and compares the result to a numpy reference -- the loop that
# makes a lowering bug visible as a wrong number on the very first run.
#
# Needs TVM, which lives in its own venv because the wheel's API differs
# substantially from the system Python's other packages. Registered only
# when that venv exists, so a checkout without it still configures and
# tests cleanly -- the TVM path is additive and nothing else depends on it.

set(RTV_TVM_PY ${CMAKE_CURRENT_SOURCE_DIR}/.venv-tvm/bin/python)
if(EXISTS ${RTV_TVM_PY})
    set(RTV_TVM_TESTS
        test_tvm_path            # 11 hardware intrinsics, INT8 and FP32
        test_composite_dispatch  # BYOC pattern -> composite -> emit_composite
        e2e_migrated             # e2e_test.py's FP32 cases, re-verified here
        test_gqa_attention       # multi-head / GQA, one real query replicated
        test_gemm_double_buffer  # K-tile double buffering, real cycle-count win
        test_attention_multitile # seq_k/head_dim spanning >1 tile
        test_transformer_layer   # full layer(s), ping-ponged DRAM buffers
        test_dram_builder        # build_model_dram_image/finalize_dram_image
        test_rope_attention      # RoPE wired into the attention block
        test_swiglu_resident     # SwiGLU intermediates kept resident in SRAM
        test_attention_batched   # genuinely different rows, real per-row softmax
        test_kv_cache            # growing multi-tile KV-cache, append + tail mask
        test_attention_wide_headdim  # head_dim > tile_dim, incl. Qwen3's real hd=128
        test_decode_loop         # multi-step decode loop, state persists across rtv_sim runs
        test_checkpoint_loader   # real Qwen3-0.6B weights; skips cleanly if not downloaded locally
        test_checkpoint_decode_loop  # real Qwen3-0.6B weights through a genuine multi-step decode loop
        test_const_memoization   # repeated constant matrices (RoPE, masks) collapse to one cached copy
    )
    foreach(RTV_TVM_TEST ${RTV_TVM_TESTS})
        add_test(NAME ${RTV_TVM_TEST}
                 COMMAND ${RTV_TVM_PY}
                         ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/rtv_tvm/${RTV_TVM_TEST}.py)
    endforeach()
    message(STATUS "TVM path tests enabled (.venv-tvm): ${RTV_TVM_TESTS}")
else()
    message(STATUS "TVM path tests skipped: .venv-tvm not present")
endif()
