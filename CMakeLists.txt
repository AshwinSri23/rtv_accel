cmake_minimum_required(VERSION 3.16)
project(rtv_accel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Decoder static library ───────────────────────────────────────────────────
# Bit-exact ISA encoder/decoder.  All other targets link this.

add_library(rtv_decoder STATIC src/decoder.cpp)
target_include_directories(rtv_decoder PUBLIC inc)

# ─── Dispatcher static library ────────────────────────────────────────────────
# Reads binary payloads produced by toolchain/npu_compiler.py,
# byte-swaps from big-endian to native, and routes decoded instructions
# to registered per-opcode handlers via InstructionDispatcher.

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
# The compiler is toolchain/npu_compiler.py. A C++ frontend (SLMAllocator,
# ir.hpp, Emitter, compiler.cpp, compiler_main) was built here in parallel and
# has been removed: it duplicated the Python compiler's job while lagging it in
# capability, and having two lowering paths meant every ISA change had to be
# made twice. The Python compiler is the single source of truth for lowering;
# what remains in C++ is the ISA and the simulator that executes it.
#
# The e2e_test target below is the compiler's test suite -- it drives
# npu_compiler.py, runs the result on rtv_sim, and checks it against numpy.

# ─── Top-level simulator ──────────────────────────────────────────────────────

add_executable(rtv_sim src/simulator.cpp)
target_include_directories(rtv_sim PRIVATE inc)
target_link_libraries(rtv_sim PRIVATE rtv_dispatcher rtv_pe rtv_memory)

# ─── End-to-end numeric verification ──────────────────────────────────────────
# Compiles operations with toolchain/npu_compiler.py, executes them on rtv_sim
# against a real DRAM image, and compares the stored result to a numpy
# reference. This is the loop the Python compiler previously lacked -- every
# correctness bug found by hand-auditing it produces visibly wrong numbers
# here on the first run. Needs python3 + numpy; skipped if absent.

# ─── TVM compilation path ─────────────────────────────────────────────────────
# The Version-B pipeline: a matmul in TVM's TIR, scheduled and tensorized
# against the rtv.gemm_16x16 intrinsic, lowered to instructions by
# toolchain/rtv_tvm/codegen.py, executed on rtv_sim and checked against numpy.
#
# Needs TVM, which lives in its own venv because the wheel's API differs
# substantially from the system Python's other packages. Registered only when
# that venv exists, so a checkout without it still configures and tests
# cleanly -- the TVM path is additive and nothing else depends on it.

set(RTV_TVM_PY ${CMAKE_CURRENT_SOURCE_DIR}/.venv-tvm/bin/python)
if(EXISTS ${RTV_TVM_PY})
    add_test(NAME tvm_path
             COMMAND ${RTV_TVM_PY}
                     ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/rtv_tvm/test_tvm_path.py)
    message(STATUS "tvm_path test enabled (.venv-tvm)")
else()
    message(STATUS "tvm_path test skipped: .venv-tvm not present")
endif()

find_program(RTV_PYTHON3 python3)
if(RTV_PYTHON3)
    execute_process(
        COMMAND ${RTV_PYTHON3} -c "import numpy"
        RESULT_VARIABLE RTV_NUMPY_CHECK
        OUTPUT_QUIET ERROR_QUIET)
    if(RTV_NUMPY_CHECK EQUAL 0)
        add_test(NAME e2e_test
                 COMMAND ${RTV_PYTHON3}
                         ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/e2e_test.py
                         --sim $<TARGET_FILE:rtv_sim>
                         --hal ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/hal_config.json)
        message(STATUS "e2e_test enabled (${RTV_PYTHON3})")
    else()
        message(STATUS "e2e_test skipped: numpy not available for ${RTV_PYTHON3}")
    endif()
else()
    message(STATUS "e2e_test skipped: python3 not found")
endif()
