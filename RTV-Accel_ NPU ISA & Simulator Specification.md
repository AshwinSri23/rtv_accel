```markdown
# **RTV-Accel: NPU ISA & Simulator Specification**

This document outlines the design and implementation of a custom Neural Processing Unit (NPU) through a high-performance, behavioral C++ simulator. The project encompasses Instruction Set Architecture (ISA) design, virtual silicon modeling, and an end-to-end compiler infrastructure built on Apache TVM (BYOC and TensorIR), replacing custom standalone Python compiler scripts (`.py` files representing the compiler).

> **Scope note.** This project also includes a Linux kernel driver layer (`/dev/rtv_accel`, DMA API, `mmap` zero-copy, `ioctl` control), deferred rather than dropped: three of five planned parts are implemented and verified (character device registration, an `ioctl`-based control-plane handshake with real kernel wait-queue blocking, all tested inside a `virtme-ng` VM), but work on it is paused so the ISA/simulator/compiler core can be brought to full correctness first — the driver doesn't affect that correctness in either direction. See `driver/` for the current implementation.

> **Platform note.** "NPU" and "silicon" throughout this document describe what the ISA/simulator MODEL, not hardware this project has ever run on. Every build, test, and benchmark to date has run on x86-64 (the AVX2-vectorized PE array path, `src/pe_array.cpp`) — there is no physical RTV-Accel silicon, and the ARM/NEON path referenced in §2 compiles but has never actually executed, on real AArch64 hardware or otherwise (this repo has no cross-compiler or emulator for it either). Where the text below says "target Raspberry Pi 5 platform" or similar, read it as the intended eventual deployment target this project is designed toward, not a claim about where anything has actually run — see TODO.md's Phase 4.1 for that gap tracked as open work.

---

## **1. ISA Definition**

The RTV-Accel architecture utilizes a 128-bit fixed-length ISA. This format provides sufficient width to accommodate large memory addresses and specific tiling parameters within a single instruction word.

### **Instruction Bit-Layout — v1**

| Bits | Field | Width | Description |
| :--- | :--- | :--- | :--- |
| 127:124 | **Opcode** | 4 b | Instruction identifier (LOAD\_TILE, COMPUTE\_GEMM, …). |
| 123:120 | **Dep\_ID** | 4 b | Dependency tag for hardware-level synchronization. |
| 119:88 | **DRAM\_Addr** | 32 b | Pointer to system DRAM (full 4 GB range). |
| 87:72 | **SRAM\_Addr** | 16 b | Offset in the NPU scratchpad. |
| 71:56 | **dim\_m** | 16 b | Tile rows. |
| 55:40 | **dim\_n** | 16 b | Tile columns. |
| 39:24 | **dim\_k** | 16 b | Tile reduction depth. |
| 23:8 | **stride** | 16 b | DRAM row stride for LOAD/STORE; overloaded as the operand-B SRAM address by COMPUTE\_GEMM and EXEC\_ALU, and as ε by SCALAR\_ALU. |
| 7:0 | **activation** | 8 b | Operation / activation mode selector. |

> **Note.** An earlier draft of this document specified a 24-bit `DRAM_Addr` at 119:96 with a `padding` byte inside `Params`. The implementation widened the address to 32 bits and reclaimed the padding byte. The table above reflects the shipped encoding in `inc/decoder.hpp` and `toolchain/rtv_isa.py`.

### **Instruction Bit-Layout — v1.5**

v1.5 is a **backward-compatible revision** of the encoding above. No field moves and no field changes width. Three fields gain internal sub-fields in bit positions whose values are always zero under v1, making three previously inexpressible operations encodable.

| Field | v1 meaning | v1.5 sub-fields |
| :--- | :--- | :--- |
| `activation` [7:0] | Mode enum, values 0x00–0x07 | **bit 7** — accumulate flag<br>**bits 6:0** — mode enum (unchanged) |
| `dim_m` [71:56] | Tile rows | **bits 7:0** — tile rows<br>**bits 12:8** — requantization scale shift (0–31)<br>bits 15:13 reserved |
| `dim_k` [39:24] | Tile reduction depth | **bits 7:0** — tile depth<br>**bits 11:8** — destination C slot index (0–15)<br>bits 15:12 reserved |

The tile dimension fields are safe to subdivide because they are bounded by the $16 \times 16$ PE array; an 8-bit width admits values up to 255, a 16× margin over any legal tile.

**Motivation.** v1 cannot express three operations the compiler requires:

1. **Accumulation.** A K-tiled GEMM computes $C = A_0B_0 + A_1B_1 + \dots$, so every K-step must add into C. v1 carries no bit distinguishing accumulate from overwrite, so `PEArray::compute_gemm` zeroes its accumulator unconditionally and a tiled reduction returns only its final term.
2. **A destination address.** `COMPUTE_GEMM` needs three operand addresses (A, B, C) but v1 supplies one. A comes from `SRAM_Addr`, B is carried in `stride`, and C is hardcoded to the output-zone base in the simulator — so every GEMM in a program writes to the same 256 bytes.
3. **A quantization scale.** Narrowing an INT32 accumulator back to INT8 requires a scale derived from how the operands were quantized. v1 has no field for it; the implementation applies a fixed `>> 7`.

**Backward compatibility.** A v1-encoded instruction leaves all three new sub-fields zero, decoding to `accumulate = 0`, `c_slot = 0`, `scale_shift = 0` — precisely v1 semantics. Existing binaries decode identically under a v1.5 decoder, and the v1 test suites pass unmodified.

**Disassembly.** A v1.5 word read by a v1 disassembler prints misleading dimensions — `dim_m = 0x0710` renders as `M=1808` — so the disassemblers in both `src/decoder.cpp` and `toolchain/rtv_isa.py` must print the sub-fields explicitly.

> [!NOTE]
> **Case study: why standalone `.py` compiler scripts were abandoned, and why their INT8-era debugging history is gone from this document.** `toolchain/npu_compiler.py` (~3400 lines) was a hand-written Python compiler predating TVM: independent wire-packing, manual DRAM/SRAM address arithmetic, untyped buffers. Bringing up Qwen3-0.6B against it surfaced a long series of silent-corruption bugs across GEMM accumulation, SRAM slot addressing, staging buffer sizing, RoPE, attention, and INT8 requantization calibration — cascading, hard-to-find defects that motivated abandoning hand-written `.py` compiler scripts entirely in favor of Apache TVM (BYOC + TensorIR), where a typed tensor IR derives addresses and strides from buffer types instead of trusting hand-written arithmetic to agree across ~95 call sites.
>
> That debugging history no longer belongs in this document for two independent reasons, not one: `npu_compiler.py` itself is now **deleted** (see "Superseded & Retired Components" below and `toolchain/compiler_state.py`, which carries forward only the dtype-agnostic shared state — HAL config, the staging allocator, RoPE's host-side table math — every caller actually needed), and separately, the project no longer targets INT8 at all: the NPU runs models in their **native FP32 precision**, with quantization out of scope entirely, not merely deprioritized. Both the specific bugs (all in code that no longer exists) and the INT8 calibration narrative built on top of them (scale bookkeeping, per-channel outlier handling, `device_probe.py`/`quant_sim.py`/`scale_sim.py` comparisons, 8-prompt accuracy targets) are gone rather than summarized — none of it describes anything this project still does. The full narrative remains in this file's git history for anyone tracing why a specific ISA field (`scale_shift`, `c_slot`, `accumulate`) exists.

### **FP32 is the target precision, not a debug aid**

`Decoder::DType::FP32` was originally added as a diagnostic mode running
alongside INT8 — a float run that matched the reference would prove every
address, tile, K-accumulation and schedule correct, isolating any remaining
INT8 disagreement to quantisation rather than lowering. That framing is now
backwards: **FP32 is the target precision the NPU runs models in.**
Quantization is out of scope entirely, not merely deprioritized — there is
no INT8 path left to compare against, and none of the modes below exist to
serve one.

Several ISA modes are consequently identities or unused at FP32: `RESCALE`
(converts between two quantisation scales — there is one scale, the
float's own exponent), `INT8_QUANT` (narrows an accumulator — nothing
narrows), `LUT` (precomputes a nonlinearity for a known fixed-point scale
— FP32 SiLU evaluates the true function directly, confirmed against
`pe_array.cpp`: "no table needed in float"). These aren't dead code to
prune from the ISA — real INT8 hardware use is still a legitimate
capability of the simulator/decoder — they're simply not what the
compiler emits for.

**Sum of squares accumulates in double** in `pe_array.cpp`'s FP32 REDUCE_SUMSQ
path: a 1024-wide row of squares loses low-order terms in float32, and that
sum is exactly RMSNorm's divisor.

**FP32's 4-byte element width needed its own SRAM zone**, not a resize of
the INT8-sized ones. A 16×16 FP32 tile is 1024 bytes — the entire narrow
input or weight zone at INT8 spacing — so wide dtypes (FP32, and INT16 for
the residual stream) use dedicated `WIDE_INPUT_BASE`/`WIDE_WEIGHT_BASE`
zones in the ~46 KiB of SRAM above the scratch area, rather than resizing
the narrow zones (which would move every `c_slot` address and break
agreement with `simulator.cpp`'s `c_slot_addr()`).

**Error is measured against the magnitude of the terms actually summed**
(`|a| @ |b|`), not against the result — a dot product with inputs spanning
several orders of magnitude can cancel to near zero by construction, and
any float32 accumulation order then leaves order-1 relative error on a
residue that was never the point of the test.

### **The TVM path: complete and the sole compiler (Phase 1)**

`toolchain/rtv_tvm/` (`intrin.py`, `schedule.py`, `codegen.py`, `patterns.py`) plus `toolchain/tvm_backend.py` now form the entire compiler. `toolchain/npu_compiler.py` is deleted, not superseded-but-kept — see "Superseded & Retired Components" below. This section used to track that migration stage by stage (an early 4-intrinsic proof of concept, then 11 intrinsics with composition still hand-written, then a staging-buffer bug found while proving the FP32 path matched a float reference at real model scale); all of those stages are complete now, and TODO.md is the living, detailed record of exactly what was built, what broke, and how it was fixed — this document summarizes the architecture that resulted rather than re-narrating how it got there.

**The mapping from TVM's scheduling model to the ISA came out direct**, the substantive finding that made this whole approach viable:

    cache_read   ->  LOAD_TILE      (DRAM -> SRAM)
    tensorize    ->  COMPUTE_GEMM / EXEC_ALU / REDUCE / SCALAR_ALU
    cache_write  ->  STORE_TILE     (SRAM -> DRAM)

**The codegen computes no addresses.** Every offset and stride is read off the TIR, which TVM derives from typed buffer definitions — the ~276 lines of byte-vs-element arithmetic the hand-written compiler carried, and the source of most of its bugs, has no place to live in this design.

**All 11 hardware intrinsics have FP32 `TensorIntrin` coverage** (`gemm`/`mul`/`sumsq`/`rms_scale`/`silu`/`rowmax`/`rowsum`/`exp_sub`/`scalar_div`/`transpose`/`add`), alongside the original INT8 versions kept for hardware-capability coverage. Composed into RMSNorm, SwiGLU FFN, multi-head/GQA attention (including RoPE and a real, growing multi-tile KV-cache), and whole transformer layers chained with ping-ponged DRAM buffers — verified against `rtv_sim` and numpy across 73 tests in 12 files, registered individually in `CMakeLists.txt` and run through `ctest`.

**It fully unrolls, and must.** The ISA has no control flow, so a TIR loop becomes repeated instructions. The same property forces a fresh program per decode step, which no compiler change fixes — see "What TVM does not fix" below.

**Still hand-driven, by design, not by gap:** `dep_id`/`WAIT_SYNC` assignment (TVM has no model of DMA dependency tokens) and every per-head/per-tile loop a static Relax graph match can't express (GQA head grouping, K-tile double buffering, per-row batched softmax, KV-cache append) — these are driven directly by callers that already know the model's configuration, the same role `compile_attention_block` used to play, not a limitation of the TVM approach.

### **Core Instructions**
* **LOAD\_TILE:** Transfers data (weights or feature maps) from system DRAM to the NPU's internal SRAM.
* **COMPUTE\_GEMM:** Executes a Matrix-Matrix multiplication task on a $16 \times 16$ tile residing in SRAM.
* **EXEC\_ALU:** Handles post-processing operations including element-wise multiplication, addition, and activation functions (RELU, SILU, SQUARE, RSQRT) and INT8 quantisation. The specific operation is selected by the `activation` sub-field of the Params word.
* **STORE\_TILE:** Transfers computed output tiles from SRAM back to system DRAM.
* **WAIT\_SYNC:** Implements a hardware barrier, stalling the pipeline until a specific `Dep_ID` is cleared by an asynchronous engine.
* **REDUCE:** Computes a row-wise sum or sum-of-squares across a tile and writes the resulting scalar to a nominated SRAM address. Required for the RMSNorm reduction step.
* **SCALAR\_ALU:** Reads a scalar value from a nominated SRAM address and applies it element-wise to a tile (multiply, add, or reciprocal-sqrt). Required for the RMSNorm scale step and the $1/\sqrt{d}$ attention scaling factor.

---

## **2. Simulator Architecture**

The simulator is implemented as a cycle-approximate behavioral model in C++, optimized via vectorization to simulate modern neural workloads without processing latency.

* **Memory Controller:** Manages tiered memory simulation, modeling the latency gap between DRAM and local SRAM. Implements **SRAM Partitioning** to support concurrent execution.
* **Instruction Dispatcher:** Parses instructions and routes them to dual asynchronous execution queues (Load/Store Engine vs. Compute Engine).
* **PE Array:** A functional model of a $16 \times 16$ systolic array engine, vectorized on whichever SIMD ISA the build machine actually has: **Arm Neon** on AArch64, **AVX2** on x86-64 (`check_cxx_source_runs` selects it only when the build machine can actually execute it, not just accept the compiler flag — see `CMakeLists.txt`). Every development, build, and test run of this project to date has been on an x86-64 machine (this one included: an Intel Core Ultra 7 258V, 8 threads, AVX2 present), via the AVX2 path — that is the ONLY path with any real execution history behind it. The NEON path compiles (cross-compilation isn't even attempted from this machine — no `aarch64-linux-gnu-gcc` or QEMU user-mode emulation is installed here) but has never actually been *run*, on real ARM hardware or otherwise; "the target Raspberry Pi 5 platform" describes an intended eventual deployment target, not the platform this simulator has ever executed on. Closing that gap is TODO.md's Phase 4.1 (Physical ARM NEON Validation), open and unstarted as of this writing.

---

## **3. Software Stack & Compiler**

The compiler is **Apache TVM with a BYOC (Bring Your Own Codegen) backend and TensorIR scheduling**, rather than a hand-crafted toolchain composed of standalone Python compiler scripts. TVM is now the sole compiler architecture for RTV-Accel: the legacy prototype (`toolchain/npu_compiler.py`, ~3400 lines) is **deleted**, not merely superseded-and-kept. The decision to use TVM is based on what building and debugging that prototype taught.

### **Why the change: eliminating hand-written `.py` compiler scripts**

Writing a compiler as ad-hoc Python scripts (`.py` files representing the compiler) proved structurally fragile across the entire development cycle. Categorising the defects found while bringing up Qwen3-0.6B:

| class | count | would a typed tensor IR have prevented it? |
|---|---|---|
| byte-vs-element / SRAM slot width | ~7 | **Yes** — element width lives in the type, not in hand-written arithmetic |
| DRAM stride & layout mismatches | ~3 | **Yes** — buffer strides and shapes are derived directly from the tensor types |
| staging lifetime & memory collisions | ~4 | **Yes** — buffer scopes and lifetimes are tracked systematically in IR |
| bit-packing wire divergence | ~2 | **Yes** — unified codegen eliminates duplicate hand-rolled packing logic |
| scale semantics | ~5 | **No** — domain knowledge no framework has (managed in codegen passes) |

The defects demonstrate why hand-written `.py` compiler scripts fail:
1. **Manual byte-vs-element arithmetic:** In `npu_compiler.py`, multiplying an element count by a width by hand across ~95 different call sites caused repeated, silent memory corruption. INT16 and FP32 slot overlaps, GEMM operand slot mismatches, storing INT32 element counts instead of byte widths, and mismatched row pitches were all variations of this exact trap. In TVM, buffer extents and strides are derived automatically from typed tensor definitions.
2. **Duplicate wire packing:** Standalone Python scripts duplicate instruction encoding (`ScheduleNode` / `pack_instruction`), which drifted from the hardware C++ decoders over ISA revisions (e.g. missing `accumulate` bits).
3. **Staging buffer overruns:** Manual allocation in Python scripts sized staging buffers from logical sequence lengths rather than physical hardware tile geometries, causing 16x overruns during single-token decode. TVM's TensorIR models physical allocation scopes and bounds explicitly.

### **Why TVM specifically, over MLIR or an LLVM backend**

**Not LLVM.** An LLVM backend targets a scalar register machine — virtual
registers, scalar loads and stores, branches. RTV-Accel has no register file,
no control flow, and 128-bit instructions performing whole 16x16 GEMMs with
DMA dependencies tracked by `dep_id`. There is no sensible lowering from LLVM
IR to `COMPUTE_GEMM`, and LLVM's register allocator and scheduler would have
nothing to allocate or schedule. (LLVM *is* the right target for a scalar ISA
— see the sibling RTV-Core project.)

**TVM over MLIR.** TVM provides the fastest, cleanest path to end-to-end correctness for RTV-Accel:

  1. **First-class Tensor Intrinsics (`tensorize`):** TVM natively matches fixed-size hardware primitives (such as the 16x16 systolic GEMM) against loop nests using `tensorize`, allowing the scheduler to handle loop splitting, reordering, and tiling automatically rather than through hand-coded nested Python loops.
  2. **VTA architectural alignment:** TVM's VTA (Versatile Tensor Accelerator) reference architecture features decoupled load/compute/store queues and stage dependency tokens (`dep_id`/`WAIT_SYNC`), providing a proven, production-grade template for RTV-Accel BYOC integration.
  3. **Python-accessible lowering passes:** Compile-time constant construction — RoPE's host-side cos/sin tables, and the broadcast/mask/permutation matrices `tvm_backend.py` builds wherever the ISA has no matching primitive (no hardware broadcast, no partial-tile write, no intra-tile permute) — stays in high-level Python passes without needing to write complex C++ MLIR dialects.

### **TVM Compiler Architecture & Pipeline**

The compilation pipeline is implemented in `toolchain/rtv_tvm/` and `toolchain/tvm_backend.py`:

```
[Hugging Face / PyTorch model]
        |
        v
[TVM frontend import -> Relax IR]
        |
        v
[FuseOpsByPattern]        Claim NPU subgraphs (toolchain/rtv_tvm/patterns.py)
        |                 Linear, Matmul, SiLU/SwiGLU, RMSNorm, Attention
        v
[MergeCompositeFunctions] Group into "Codegen": "rtv_accel" composite functions
        |
        v
[TensorIR Scheduling]     Schedule with hardware intrinsics (rtv_tvm/schedule.py)
        |                 - cache_read / cache_write for SRAM staging
        |                 - split / reorder for 16x16 PE array tiling
        |                 - cache_read/cache_write pairs hand-driven for
        |                   K-tile double buffering (no double_buffer
        |                   primitive exists in this TVM build)
        |                 - tensorize to target RTV-Accel intrinsics
        v
[TVM BYOC Codegen]  ----> rtv_tvm/codegen.py:
        |                 - Read type-derived buffer addresses & strides
        |                 - Assign hardware dependency tokens (dep_id / WAIT_SYNC)
        |                 - Emit compile-time constant regions (RoPE tables,
        |                   broadcast/mask/permutation matrices)
        v
[128-bit instruction stream] -> rtv_sim / Hardware Driver
```

### **Hardware Tensor Intrinsics (`toolchain/rtv_tvm/intrin.py`)**

Every hardware operation on the 16x16 PE array and vector ALU is registered as a formal TVM TensorIR intrinsic, with a distinct `TensorIntrin` per dtype (`tensorize` requires an exact dtype match). All 11 have both an FP32 variant (what the compiler actually emits for) and the original INT8 variant (kept for hardware-capability coverage, exercised by `test_tvm_path.py`):

| Intrinsic (FP32 name) | ISA Instruction | Semantics |
|---|---|---|
| `rtv.gemm_16x16_fp32` | `COMPUTE_GEMM` | 16x16 matrix multiply with reduction into a float accumulator (INT32 for the INT8 variant) |
| `rtv.add_16x16_fp32` | `EXEC_ALU ELEMWISE_ADD` | Elementwise addition, `a + b` directly in float — no rescale (the INT8 variant clamps to 8 bits) |
| `rtv.mul_16x16_fp32` | `EXEC_ALU ELEMWISE_MUL` | Elementwise multiplication, `a * b` directly — no fixed-point shift (the INT8 variant applies one) |
| `rtv.sumsq_16x16_fp32` | `REDUCE REDUCE_SUMSQ` | Tile sum-of-squares reduction to a scalar for RMSNorm, accumulated in double to avoid losing low-order terms |
| `rtv.rms_scale_16x16_fp32` | `SCALAR_ALU RMS_SCALE` | `x * sqrt(N / sum)` applied elementwise — the true reciprocal-RMS scale, no fixed-point convention |
| `rtv.rowmax_16x16_fp32` | `REDUCE REDUCE_MAX` | Row maximum reduction for numerical stability in softmax |
| `rtv.exp_sub_16x16_fp32` | `SCALAR_ALU EXP_SUB` | `exp((x - max) / T)` in true float (the INT8 variant scales to a fixed-point range) |
| `rtv.rowsum_16x16_fp32` | `REDUCE` (sum) | Row summation for softmax probability normalization |
| `rtv.scalar_div_16x16_fp32` | `SCALAR_ALU SCALAR_DIV` | `x / sum`, normalizing to true probabilities (the INT8 variant produces an 8-bit probability code) |
| `rtv.silu_16x16_fp32` | `EXEC_ALU SILU` | SiLU evaluated as the true function directly — no lookup table needed in float (the INT8 variant needs a calibrated 256-byte SRAM table, `EXEC_ALU LUT`, since SiLU's curvature is invisible at 8-bit granularity) |
| `rtv.transpose_16x16_fp32` | `EXEC_ALU TRANSPOSE` | In-place square tile transposition in SRAM |

### **What each side owns**

**TVM provides:**
* Frontend model ingestion from PyTorch, ONNX, and Hugging Face safetensors.
* Graph partitioning via Relax BYOC pattern matching.
* Automatic CPU fallback for non-accelerated ops (runtime token embedding gather, argmax sampling).
* Operator fusion and composite function formation.
* Typed memory buffer management, stride computation, and memory layout validation.
* Loop transformation, tiling, and schedule search space.

**The RTV-Accel project defines:**
* Hardware tensor intrinsics (`toolchain/rtv_tvm/intrin.py`).
* SRAM memory hierarchy scopes (`shared` scratchpad zones, `WIDE_INPUT_BASE` / `WIDE_WEIGHT_BASE`).
* Target schedule recipes (16x16 systolic tiling, weight double-buffering with safe prefetch ordering).
* Synchronization pass (`dep_id` tracking and `WAIT_SYNC` emission).
* Compile-time constant construction where the ISA has no matching primitive: broadcast/mask/permutation matrices applied via GEMM (no hardware broadcast, partial-tile write, or intra-tile permute instruction exists), and RoPE's host-side cos/sin tables.

### **What does not change**

The hardware simulator and datapath remain the core deliverable:
`inc/decoder.hpp` and `src/decoder.cpp` (the ISA), `src/pe_array.cpp` and
`src/memory_controller.cpp` (the datapath, 157 tests), and `src/simulator.cpp`.

**Numerical validation** now happens directly in the TVM test suite (`toolchain/rtv_tvm/test_*.py`, `e2e_migrated.py`): each op or composite is checked against a numpy reference and, where applicable, against `e2e_test.py`'s original test vectors and tolerances for parity. The separate INT8-era validation stack this used to describe (`ref_model.py`, `quant_sim.py`, `scale_sim.py`, `quant_eval.py`, `device_probe.py`) is retired along with INT8 itself — most of it existed specifically to distinguish a quantisation limit from a lowering bug, a distinction that no longer applies once there is no quantisation.

### **What TVM does not fix**

The ISA has no control flow, so KV-cache length is baked into addresses and
**each decode step is a freshly compiled ~12M-instruction program**. TVM emits
those instructions rather than `npu_compiler.py`, but the structure is a
hardware property. The remedy is a scalar control core, not a change of
compiler.

There is also a structural mismatch to resolve: TVM's runtime invokes
*kernels*, while this device runs one flat program per token with everything
statically sequenced. Either the whole-program model is kept (TVM as compiler
only, orchestration stays here) or the design moves to per-op kernels and
`rtv_sim` gains a per-kernel entry point.

### **Superseded & Retired Components**

* **Standalone Python Compiler (`toolchain/npu_compiler.py`, ~3400 lines):** **Deleted**, not merely superseded-and-kept. Its dtype-agnostic shared state (HAL config loading, the DRAM staging allocator, RoPE's host-side table math, and the raw instruction-emission primitives `test_tvm_path.py`'s INT8-intrinsic tests still drive directly) was extracted into `toolchain/compiler_state.py` first; everything else — every `compile_*` method, `ingest_model`, the four-pass INT8 orchestration this section used to describe — is gone, not relocated.
* **INT8 quantization, end to end:** Out of scope, not deprioritized. The NPU runs models in native FP32; there is no calibration, scale planning, or requantization step anywhere in the current compiler. `toolchain/e2e_test.py` (22 INT8/FP32 test classes; the 4 FP32 ones are superseded by `rtv_tvm/e2e_migrated.py` with verified parity) and the INT8-era validation stack (`generate.py`, `device_probe.py`, `quant_sim.py`, `scale_sim.py`, `quant_eval.py`, `hf_loader.py`) are retired alongside it — `generate.py`/`device_probe.py` carry an explicit RETIRED notice in place; the rest are simply unused by anything still running.
* **JIT Swizzle Engine (`toolchain/swizzle.py`):** Withdrawn. Tensor layout transformation and strided memory access are managed by TVM's typed buffer system.
* **Hand-Written Polyhedral Tiling & List Scheduling:** Withdrawn. Tiling and double-buffering are expressed via TVM TensorIR schedule primitives (`split`, `reorder`, `tensorize`, and a hand-driven K-tile double-buffer in `codegen.py` where no TVM schedule primitive could express it — see `Codegen._gemm_k_loop`).
* **Hardware Abstraction Layer (`toolchain/hal_config.json`):** Retained. Hardware dimensions (PE size, SRAM zone bounds) feed directly into TVM memory scopes and intrinsic declarations.

---

## **4. High-Performance Targets: Workload Evolution**

Rather than executing abstract synthetic workloads, the software toolchain is architected to parse and execute cutting-edge AI architectures, run in their native FP32 precision (no quantization anywhere in this pipeline):
* **Computer Vision:** End-to-end execution of a **ResNet-18** convolution pipeline.
* **Edge Large Language Models:** FP32 deployment of **[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)**'s architecture (Qwen3ForCausalLM, ~0.6B parameters, hidden\_size=1024, 28 layers, GQA 16→8 heads, head\_dim=128, SiLU/SwiGLU FFN, RMSNorm ε=1e-6, RoPE θ=1,000,000). The pipeline utilizes Apache TVM (with BYOC subgraph partitioning and TensorIR schedules) to orchestrate the model's attention and FFN blocks through the simulated NPU hardware without overflowing local SRAM buffers. RoPE rotation tables are pre-computed on the host CPU and loaded via LOAD\_TILE. Verified so far at up to 3-layer chained models with real Q/K/V/O projections, GQA, RoPE, and a growing KV-cache (`head_dim == 16`); the full 28-layer/head_dim=128 configuration is not yet run end to end (`head_dim > 16` needs the attention path's per-head transpose and RoPE's intra-tile permutation generalized to span multiple tiles first).

---

## **5. Roadmap (revised)**

Months 1-3 of the original plan are complete: the ISA, decoder, simulator,
PE array and Linux character driver (parts 1-3) exist and are tested. What
follows replaced Month 4, which specified writing fusion and scheduling
passes by hand — Stages A through C below are now complete, in full;
TODO.md's Phase 1 is the living, detailed record of exactly what was built,
what broke, and how it was fixed. This section keeps only the outcome.

### **Stages A-C — migrated to TVM, `npu_compiler.py` retired — ✅ complete**

The FP32 divergence at model scale was resolved (a staging buffer sized by
logical sequence length rather than physical tile count — see TODO.md for
the specific bug). TVM BYOC graph partitioning claims every supported
operation (linear/matmul, SwiGLU FFN, RMSNorm, attention's core, with
multi-head/GQA, RoPE, and a real growing KV-cache driven directly by
callers that know the model configuration, since DFPattern can't discover
per-head loops from graph structure). Host CPU fallback for embedding
gather and argmax sampling works via BYOC's ordinary "don't match it"
mechanism — no dispatch code needed. Whole-model TensorIR scheduling is
complete: every composed block (attention block with real projections,
transformer layers chained with ping-ponged buffers, SwiGLU with its
intermediate kept resident in SRAM) goes through `tensorize` against the
11 FP32 hardware intrinsics, and `toolchain/npu_compiler.py` (~3400 lines)
is deleted — not decommissioned-and-kept, actually gone, with its
dtype-agnostic shared state extracted first into `toolchain/compiler_state.py`.
Verified: 73 tests across 12 files in `toolchain/rtv_tvm/`, registered
individually in `CMakeLists.txt` and passing under `ctest`.

**What's still open, tracked as TODO.md Phase 1 gaps, not new roadmap items:**
`head_dim > compiler.tile_dim` (e.g. 128, Qwen3's real value) is
unsupported throughout the attention path — the per-head K transpose and
RoPE's intra-tile permutation both need generalizing to span multiple
tiles. The KV-cache append/tail-mask mechanism is verified at the
attention-core level but not yet wired into a full per-layer/per-model
decode loop with real projections and RoPE at the current position —
assembly of pieces that already exist, not an open design question.
TVM's auto-scheduling search space (exploring alternate loop orderings
and tile configurations beyond the one hand-chosen schedule per op) is
unexplored.

### **Dropped: INT8 completion**

INT8 quantization is out of scope entirely, not deprioritized — the NPU
targets native FP32 precision (see "FP32 is the target precision, not a
debug aid," above). Calibrating `q_norm`/`k_norm` gain scales, Q0.7 RoPE
tables, `INT8_QUANT`'s default shift, and in-row activation-outlier
handling were all steps toward INT8 accuracy parity; none of them apply
once there is no quantisation step to calibrate. Left here for history,
same treatment as TODO.md's own Phase 2.

### **Deferred**

Driver parts 4-5 (DMA/`mmap`, daemon — TODO.md Phase 3, deliberately last),
physical ARM NEON validation (TODO.md Phase 4.1 — needs real AArch64
hardware; every build and test run to date has been x86-64/AVX2, see the
Platform note at the top of this document), and a scalar control core to
remove the recompile-per-token (TODO.md Phase 4.2) — the last of which is
the single largest structural inefficiency and is tracked in the sibling
RTV-Core project.

```