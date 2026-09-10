```markdown
# **RTV-Accel: NPU ISA & Simulator Specification**

This document outlines the design and implementation of a custom Neural Processing Unit (NPU) through a high-performance, behavioral C++ simulator. The project encompasses Instruction Set Architecture (ISA) design, virtual silicon modeling, and an end-to-end compiler infrastructure built on Apache TVM (BYOC and TensorIR), replacing custom standalone Python compiler scripts (`.py` files representing the compiler).

> **Scope note.** This project also includes a Linux kernel driver layer (`/dev/rtv_accel`, DMA API, `mmap` zero-copy, `ioctl` control), deferred rather than dropped: three of five planned parts are implemented and verified (character device registration, an `ioctl`-based control-plane handshake with real kernel wait-queue blocking, all tested inside a `virtme-ng` VM), but work on it is paused so the ISA/simulator/compiler core can be brought to full correctness first — the driver doesn't affect that correctness in either direction. See `driver/` for the current implementation.

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
> **Case Study: Why standalone `.py` compiler scripts were abandoned.** The bugs documented below in the prototype `npu_compiler.py` demonstrate why building standalone Python files representing the compiler is unmaintainable. An independent wire-packing implementation, manual DRAM/SRAM address arithmetic, and untyped buffers led to cascading silent errors that motivated replacing custom `.py` compiler scripts entirely with Apache TVM (BYOC and TensorIR).

**Bug: `npu_compiler.py`'s K-tiled GEMM never actually accumulated.** Adding the `accumulate` field to the ISA (`decoder.hpp`, `pe_array.hpp`) and to `rtv_isa.py` did not, by itself, fix K-tiled GEMM correctness — `npu_compiler.py`'s own `ScheduleNode`/`pack_instruction()` is a *third*, independent bit-packing implementation (it does not import `rtv_isa.py`), and it had no `accumulate` field at all. `compile_linear()`'s and `compile_sibling_qkv()`'s K-reduction loops each emitted one `COMPUTE_GEMM` per K-tile with no way to request accumulation, so every step's `COMPUTE_GEMM` zeroed the C accumulator before writing — only the *last* K-tile's partial product ever survived. This affected every linear projection in the actual target model: `hidden_size=1024` means 64 K-tiles per projection, all silently discarded but the last.

Found while auditing whether `npu_compiler.py` needed updating alongside the C++/`rtv_isa.py` v1.5 work. Fixed by adding `accumulate`/`scale_shift`/`c_slot` to `ScheduleNode`, threading them through `_emit()`'s wire-packing (mirroring `Decoder::encode()` exactly) and `_pass3_gemm()`, and passing `accumulate=(step != 0)` from both K-reduction loops. Verified by decoding the compiler's own output through the real C++ `Decoder::decode()` and confirming the accumulate bit round-trips correctly, and by confirming the instruction sequence for a `k_tiles=4` linear layer has exactly one `accumulate=False` GEMM (the first tile) followed by three `accumulate=True` GEMMs.

**A second, unrelated bug found in the same audit:** `compile_sibling_qkv()`'s `COMPUTE_GEMM` call never passed `b_sram`, so every sibling's GEMM defaulted to `SRAMMap.WEIGHT_BASE` (== `weight_slot(0)`, Q's slot) regardless of which projection (Q, K, or V) it was computing. K and V's `LOAD_TILE`s correctly loaded their own weight tiles into `weight_slot(1)`/`weight_slot(2)`, but their `COMPUTE_GEMM`s read from `weight_slot(0)` anyway — K and V would have silently computed using Q's weights. Not a v1.5/accumulate issue; a plain missing-parameter bug, predating this ISA revision entirely. Fixed by passing `b_sram=SRAMMap.weight_slot(p_idx)` explicitly.

**A third round of bugs, found auditing `compile_rmsnorm_block()`, `compile_attention_block()`, and `compile_mlp_block()`:**

1. **REDUCE had no way to specify its tile source at all.** `simulator.cpp`'s REDUCE handler unconditionally hardcoded the tile it sums to `SRAM_INPUT_BASE` (`0x0000`), ignoring wherever the tile the caller actually wants reduced might be. This was silently correct only when that tile happened to already sit at `0x0000` — and `compile_model()`'s post-attention RMSNorm call passes `sram_tile=SRAMMap.output_slot(0)` (`0x0800`), so that call was actively reading garbage, not a hypothetical risk. Fixed by repurposing REDUCE's otherwise-unused `stride` field to carry the tile source address (`stride=0` falls back to `SRAM_INPUT_BASE` for backward compatibility, same pattern as every other 0-means-default field in this ISA), in both `simulator.cpp` and `compile_rmsnorm_block()`.

2. **Every projection's output silently collided at the same fixed address.** `compile_linear()`'s `COMPUTE_GEMM` never specified a `c_slot`, so every call wrote to the same address (`c_slot=0`). Two consecutive calls to `compile_linear()` — e.g. `compile_mlp_block()`'s `gate_proj` and `up_proj` — would have the second silently overwrite the first's result before anything read it. Fixed by adding a `c_slot` parameter to `compile_linear()`, threaded through to both the GEMM and the optional `STORE_TILE`; `compile_sibling_qkv()` similarly now passes `c_slot=p_idx` per sibling instead of implicitly colliding all three into slot 0.

3. **`SRAMMap.output_slot()`'s byte spacing didn't match `c_slot_addr()`'s.** `output_slot(n)` used 256-byte spacing (sized for an INT8 tile); the simulator's `c_slot_addr(c_slot)` uses 1024-byte spacing (sized for a full 16×16 INT32 accumulator — the actual thing living in the output zone). Fixed by giving `SRAMMap` a dedicated `OUTPUT_TILE_BYTES = 0x0400` distinct from the 256-byte `TILE_BYTES` used for input/weight slots.

4. **`compile_linear()` had no way to read its input from anywhere but a hardcoded default.** Its A operand was always `SRAMMap.input_slot(0)`, undocumented as a hard requirement beyond a docstring note. `compile_attention_block()`'s o_proj step and `compile_mlp_block()`'s down_proj step both call `compile_linear()` immediately after a *previous* projection's result lands in an output slot, with nothing moving it into `input_slot(0)` first (the ISA has no SRAM-to-SRAM copy instruction at all). Fixed by adding an `a_sram_slot` parameter, defaulting to `input_slot(0)` for the common case, overridable for chained projections.

**A fourth round: three address-collision bugs, all found by sizing the DRAM/SRAM maps against real Qwen3-0.6B dimensions rather than reading code.**

1. **Staging DRAM was hardcoded inside the weight region.** `compile_model()` staged each layer's intermediate output at a fixed `0xF00000` (~15.7 MB). But weights are bump-allocated from `0x0000` upward by `_pass1_alloc_dram`, and Qwen3-0.6B's INT8 weights span ~440 MB — so that staging address sits *inside* the weights, and the first intermediate write corrupts roughly layer 0/1's parameters, with every subsequent layer then computing against garbage. The bug is invisible on a small test model (where 15 MB genuinely is past the weights) and only appears once the model is large enough, which the real target is by ~28×. Fixed by deriving the staging base from `self._dram_cursor` after `ingest_model()` (`init_staging()`), so it always lands just past whatever the weights actually consumed, for any model size.

2. **Attention output and MLP output shared one staging address.** Both `compile_attention_block`'s and `compile_mlp_block`'s `dram_out` were the same hardcoded `0xF00000`, so the MLP's result overwrote the attention block's. Fixed by giving the staging region its own bump allocator (`alloc_staging()`) that hands out distinct buffers, rewound per layer (`reset_staging()`) since no staging buffer is live across a layer boundary — layers reuse the same space, which is correct and keeps the region bounded.

3. **RMSNorm's scalar scratch landed inside output slot 0.** `compile_model` computed `SCALAR_SCRATCH = OUTPUT_BASE + 3 * TILE_BYTES` (`0x0B00`) — correct back when output slots were 256 B apart, but once `output_slot()` was corrected to the 1024-byte spacing that `c_slot_addr()` actually uses (bug #3 of the previous round), slot 0 alone spans `0x0800`–`0x0BFF`, swallowing `0x0B00`. RMSNorm's scalar would have been written on top of the accumulator it was meant to normalize. Fixed with a derived `SRAMMap.SCALAR_SCRATCH_BASE` placed above all 16 c_slot-addressable slots (`0x4800`), rather than a hand-computed offset that silently drifts whenever slot geometry changes.

Note the pattern across all three: each was a *hardcoded address* that was correct under assumptions that later changed (small model, single staging buffer, 256-byte slots). All three are now derived from the quantity they actually depend on.

**A fifth round: every tile transfer in the Python compiler loaded the wrong bytes.**

`_pass3_load()` and `_pass3_store()` never accepted or emitted a `stride`, so every `LOAD_TILE`/`STORE_TILE` the compiler produced carried `stride=0`. The simulator falls back to `dim_n` (16) in that case — which means each load read **256 contiguous bytes** instead of gathering 16 rows of 16 bytes strided by the source matrix's width. Extracting the (1,1) tile of a 32-wide matrix returned rows `[16..31], [32..47], [48..63]` where the correct tile is `[16..31], [48..63], [80..95]`.

The root cause is that the compiler was **mixing two incompatible DRAM layouts**. There are two self-consistent conventions, and `toolchain/swizzle.py` implements the second:

| | Address formula | Correct stride |
| :--- | :--- | :--- |
| Row-major (unswizzled) | `row·16·padded(full_cols) + col·16` | full matrix row width |
| Block-linear (swizzled) | `(tile_row·ntc + tile_col)·16·16` | 16 |

`_pass2_dram_stride()` uses the **row-major** address formula, but the emitted stride of 16 is the **block-linear** one — exactly `swizzle.py`'s `swizzled_stride()`. Neither half matched the other, and nothing ever called the swizzler in the first place: `npu_compiler.py` does not import `swizzle.py`, and `ingest_model()` only assigns DRAM addresses, never reformatting tensor bytes. So weights sat in DRAM row-major while every transfer strode through them as though they had been swizzled.

Fixed by staying row-major and emitting the real row width (`_padded(n)` for weights, `_padded(k)` for activations), matching the address math already in use. Wiring up the swizzler instead was considered and deferred: it requires physically reordering weight data, and there is currently **no path that loads weight data into simulated DRAM at all** (`export_binary()` emits instructions only; `rtv_sim` zero-initializes DRAM). Swizzling is best done as a transform inside that weight-export pipeline once it exists, rather than bolted on before it. Note also that the simulator's cost model charges `dram_latency + rows*cols` per load regardless of stride, so the burst-efficiency advantage of swizzling is currently invisible to every number this project produces.

**A sixth bug, same family, found in the same fix:** `_pass3_store()` passed `dim_n` as an *element* count. `MemoryController::store_tile` is dtype-agnostic and counts `dim_n`/`stride` in **bytes**, so storing a 16×16 tile of INT32 accumulators needs `dim_n = 16*4 = 64` and `stride = row_width*4` — with element counts it wrote a quarter of each row, scrambled across the wrong offsets. This is the identical byte-vs-element trap that hit the C++ `Compiler::compile()`'s `STORE_TILE` earlier (there the symptom was one output value landing correctly by coincidence while the rest were garbage); the INT8 load path is immune only because 1 byte per element makes the two counts coincide.

**A seventh issue, and the end-to-end loop that finally caught things automatically:** `compile_linear()`'s K-reduction loop never advances the A (activation) operand — every K-step's `COMPUTE_GEMM` reads the *same* fixed `a_sram_slot`, while only the B (weight) operand correctly rotates through DMA'd tiles per step. Correct K-tiled GEMM requires `C = Σᵢ A[:, i·16:(i+1)·16] @ B[i·16:(i+1)·16, :]` — a different K-slice of *both* operands per term. As written, this computes `Σᵢ A_fixed @ B[i]`, which is mathematically wrong for any `k_tiles > 1`, independent of the (already-fixed) `accumulate` bug — accumulate correctly sums the partial products, but the thing being summed is itself incorrect. Fixed by re-loading A's K-slice from DRAM per step, exactly as B already is (`compile_linear`'s new `dram_activation` parameter). The alternative — pre-loading the full K-wide row once and reading per-step offsets into it — was rejected because `PEArray::compute_gemm` assumes A is packed contiguously (`A[m*K + k]` with `K` = that instruction's own `dim_k`), so slicing a wider resident row would need either a stride-aware `compute_gemm` or a pre-blocked layout via the swizzler. Re-loading reuses a mechanism already proven correct for B with no new ISA surface. A guard now raises if `k_tiles > 1` without `dram_activation`, so this cannot silently regress.

### **End-to-end verification (`toolchain/e2e_test.py`)**

Every bug above was found by reading code — and each one produces *visibly wrong numbers on the first execution*. None would have survived a single end-to-end run. The compiler had never had one, because there was no way to get data into simulated DRAM: `export_binary()` emits instructions only, and `rtv_sim` zero-initialised DRAM, so any result was trivially zero.

`rtv_sim` now takes `--dram-image FILE` (preload DRAM from a flat blob at address 0) and `--dump ADDR:LEN:FILE` (write a DRAM region back out after execution). `toolchain/e2e_test.py` uses these to close the loop: build INT8 operands with numpy, lay them out in a DRAM image at the addresses the compiler will emit, compile the operation, execute it on `rtv_sim`, read the stored result back, and compare against `A.astype(int32) @ B.astype(int32)`. It runs as part of `ctest`.

Its first run immediately reproduced the A-operand bug above with no human involvement — every case with `k_tiles == 1` passed, every case with `k_tiles > 1` failed. That correlation *is* the diagnosis.

**An eighth bug, found by that loop rather than by reading:** with the A-operand issue fixed, cases with `k_tiles ∈ {3, 4, 8}` still failed while `k_tiles ∈ {1, 2}` passed. The weight double-buffer's prefetch was emitted *before* the `COMPUTE_GEMM` that consumes the slot it refills — at step 0 it issued `LOAD W[2] → weight_slot(0)` and then ran `GEMM(b_sram=weight_slot(0))`, DMAing over the exact tile the next instruction reads. With only two weight slots, step *i* uses slot *i*%2 and step *i+1* uses the other, so slot *i*%2 is not free until step *i*'s GEMM has consumed it; the refill must follow the GEMM, not precede it. This is a genuine write-after-read hazard on real hardware, not a simulator artifact. It stayed hidden because the prefetch only fires when `k_tiles > 2`, so no previously-exercised case ever reached it. `LOAD_TILE` is non-blocking, so moving the prefetch after the GEMM keeps the DMA/compute overlap the double-buffering exists for.

**A ninth bug, found by extending the loop to chained projections:** `compile_linear`'s `STORE_TILE` hardcoded INT32 element widths (`dim_n = t*4`, `stride = padded(n)*4`) regardless of the requested activation. But `INT8_QUANT` narrows the INT32 accumulator to INT8 *in place* on the final K-step, so a quantized result is 1 byte per element — storing it at 4× its real width scatters it across the wrong offsets. This matters specifically for chaining: feeding one projection's output into the next requires requantizing INT32 down to INT8 first (the next GEMM takes INT8 operands), so every chained pair — `o_proj` reading Q, `down_proj` reading gate×up — depends on this path. Fixed by deriving the element width from the activation.

The loop now covers, at 17 cases:

* `compile_linear` — single and multi K-tile reductions, multi-N-tile stores, deep (8-step) accumulations, dimensions that are not multiples of the 16-wide tile, and non-zero `c_slot`s (which is what actually exercises `output_slot()`'s spacing agreeing with `c_slot_addr()`; at slot 0 the old and corrected spacings both resolve to `OUTPUT_BASE`, so a mismatch is invisible there)
* `compile_sibling_qkv` — three projections against three distinct weight matrices landing in three distinct output slots, which is what verifies K and V read their own weights rather than Q's
* `REDUCE` — reducing tiles resident at several different SRAM addresses, including non-zero ones, verifying the tile source is genuinely carried in the instruction rather than hardcoded
* chained projections — one projection requantized to INT8, staged to DRAM, and re-read as the next projection's activation

### **Block-level rewrites (`compile_mlp_block`, `compile_attention_block`, `compile_model`)**

Both block functions were structurally broken rather than merely buggy, and both are now rewritten and covered end-to-end.

**`compile_mlp_block`** previously issued four straight-line calls: `gate_proj`, `up_proj`, `SiLU` on `output_slot(0)`, then `ELEMWISE_MUL` against `output_slot(1)`. Every step of that was wrong. `gate` and `up` both landed in the same output slot, so `up` overwrote `gate` before `SiLU` ran — meaning `SiLU` was applied to `up`; `output_slot(1)` was never written by anything, so the multiply used stale memory; and only the first of `I/16` tiles was ever processed (`I = 3072` is 192 tiles). Two constraints force the correct shape:

* **dtype** — the GEMMs produce INT32 accumulators, but `EXEC_ALU`'s `SILU`/`ELEMWISE_MUL` read the tile as `int8_t*`, so `gate` and `up` must be requantized (`INT8_QUANT`) before the element-wise stage.
* **size** — `gate` and `up` are each far too wide to hold in SRAM, so they are staged to DRAM, combined one tile at a time, and the result staged again for the down projection to stream back in.

**`compile_attention_block`** computed Q with one `compile_linear` call, then called `compile_sibling_qkv(dram_wq=dram_wk, dram_wk=dram_wv, dram_wv=dram_wv, …)` — computing V twice, discarding K, and (since every projection then targeted the same output slot) overwriting Q before `o_proj` could read it. Its own source comment flagged the duplicated V argument as a placeholder. Now Q, K and V are three independent projections, each requantized and staged to its own buffer, with `o_proj` reading Q's staged result. Q/K/V do share the input X and are a real fusion opportunity, but `compile_sibling_qkv` takes a single N for all three siblings while Q is `q_dim` wide and K/V are `kv_dim` — fusing them needs that function generalised to a variable sibling count, deferred as an optimisation.

**`compile_model`** passed `dram_x = 0x000000` to both blocks — inert while neither used it, but a live read-from-address-zero (i.e. from the first weight) the moment they did. Layer activations now live in two ping-ponged DRAM buffers allocated outside the per-layer staging rewind, since the buffer produced by layer *N* is the input to layer *N+1* and is the one thing genuinely live across a layer boundary; reusing a single buffer would have each block overwrite its own input mid-computation.

**`compile_rmsnorm_block` had four defects at once**, none of which the earlier REDUCE-only check could see:

1. **`SQUARE` ran in place on the tile itself**, so by the time the scale was applied the tile held `x²`, not `x` — it normalised the squares instead of the values. The rewrite takes a separate `sram_square` scratch copy (the ISA has no SRAM-to-SRAM copy, so the caller loads the input twice).
2. **The scale was applied with `EXEC_ALU` + `ELEMWISE_MUL`**, passing the scalar's address in `stride`. But `EXEC_ALU`'s `ELEMWISE_MUL` reads operand B as a full `dim_m × dim_n` INT8 tile — so it consumed 256 bytes starting at a 4-byte scalar, using 252 bytes of adjacent SRAM as multiplicands. `SCALAR_ALU` is the opcode that broadcasts.
3. **Epsilon was always exactly zero.** `int(norm_eps * (1 << 15))` at `eps = 1e-6` is `int(0.033) = 0`; Q0.15's smallest non-zero value is ~3e-5. Dropped rather than rescaled — `sum(x²)` is a non-negative integer here, so epsilon's numerical-stability role is served by `scalar_alu`'s own `scalar > 0` guard.
4. **A 256× scale mismatch.** `RSQRT` produced Q0.15 (×32768) feeding a multiply that shifted by 7 (÷128). `PEArray::scalar_alu`'s `RSQRT` applies `1/sqrt` to the tile directly in one instruction, so the rewrite has no fixed-point round trip at all.

**And a simulator bug underneath them, exposed by the rewrite:** `simulator.cpp`'s `SCALAR_ALU` handler special-cased `SCALAR_ADD` and `RSQRT`, transforming the *scalar* in place and never touching the tile — while `PEArray::scalar_alu` applies the scalar *to the tile* for those same modes, which is what this document specifies ("reads a scalar value from a nominated SRAM address and applies it element-wise to a tile"). The simulator and the reference executor disagreed on the meaning of two opcode modes, and the simulator's version was the one contradicting the spec. Reimplementing opcode semantics in the dispatch handler instead of delegating to `PEArray` is how that divergence arose; the handler now delegates unconditionally.

**Still missing, and not fixable as bugs — these are absent capabilities:**

* ~~**Residual connections.**~~ **Done.** `ELEMWISE_ADD = 0x08` was added to `ActivationMode` and mirrored across `decoder.hpp`, `decoder.cpp`, `pe_array.{hpp,cpp}`, `rtv_isa.py` and `npu_compiler.py`. It is `clamp8(a + b)` with operand B's address in the stride field, exactly like `ELEMWISE_MUL` but without the `>> 7` — a sum of two INT8 values needs no rescaling, only the saturating clamp. `compile_residual_add()` lowers a residual a tile at a time, and `compile_model` now applies both of a pre-norm layer's residuals.

  **Two bugs surfaced while wiring it up, and the first is the more instructive:**

  * **The earlier `full_layer` test was passing for the wrong reason.** This integer RMSNorm divides by `sqrt(sum of 256 clamped squares)`; at the magnitudes the test used (`|x| <= 8`), `inv ≈ 0.0136` and every normalised value rounded to zero. Both the reference and the hardware then collapsed to all-zeros, and the comparison succeeded without exercising a single downstream stage. Only adding residuals — which carry `x` past the collapse — made the output non-zero and exposed it. The test now uses a sparse mix of large and small inputs so `norm(x)` is genuinely non-degenerate (32 of 256 elements non-zero). **A test that passes because everything is zero is indistinguishable from one that passes because everything is right**, and nothing in the earlier run would have revealed the difference.
  * **The RMSNorm result was computed and then discarded.** `compile_model` normalised into SRAM `input_slot(0)`, but the sublayer immediately followed with `dram_activation = x_dram`, reloading the *raw* input from DRAM per K-step and overwriting the normalised tile. The attention block was provably operating on un-normalised input. Fixed by storing the normalised tile back to a `norm_dram` staging buffer and pointing the sublayer at that — which is also the correct pre-norm structure, since the residual must add the raw input while the sublayer sees the normalised one.
* ~~**Attention math.**~~ **Done.** `compile_attention_block` now computes `softmax(Q·Kᵀ)·V` rather than passing Q straight to the output projection. Four ISA additions were required, each closing a specific gap:

  | Mode | Why it was needed |
  | :--- | :--- |
  | `TRANSPOSE` (0x09) | `COMPUTE_GEMM` computes `A @ B`; attention needs `Q @ Kᵀ`. Neither `LOAD_TILE` nor `STORE_TILE` can transpose — both copy contiguous rows — so `Kᵀ` was inexpressible. In-place, square tiles only (a non-square transpose changes the tile's shape and the instruction carries no way to state the new one). |
  | `REDUCE_MAX` (0x0A) | Softmax subtracts the row maximum before exponentiating; `REDUCE` could only sum. |
  | `EXP_SUB` (0x0B) | `exp((x − max)/T)` scaled to INT8, as a `SCALAR_ALU` mode so the max arrives as the scalar operand. |
  | `SCALAR_DIV` (0x0C) | Normalising the exponentials — `SCALAR_ALU` could multiply by a scalar but not divide. |

  **Per-row processing, without a new opcode.** `REDUCE` collapses an entire tile to one scalar, but softmax needs a *per-row* max and sum. Issuing the reduction over a `1 × n` tile (`dim_m = 1`) makes that expressible with the existing opcode, at the cost of *m*× the instructions. The alternative was a per-row reduction opcode purely to save instruction count.

  **Precision is genuinely lossy, and the temperature is a design choice, not a derivation.** INT8 scores span [−128, 127], so after subtracting the max the exponent's input range is [−255, 0]. `SOFTMAX_TEMPERATURE = 64` (in `pe_array.cpp`, mirrored by `SOFTMAX_T` in the test) keeps roughly the top quarter of that range distinguishable; a smaller *T* flushes near-max scores to zero, a larger one flattens the distribution. Output uses 127 as the INT8 stand-in for probability 1.0, so each row sums to ≈127 and small probabilities quantise to zero. That is inherent to softmax in 8 bits, not an artefact of this lowering — real accelerators generally keep softmax in higher precision.

  **Multi-head and GQA are wired up.** The attention loop runs once per query head over a `head_dim`-wide slice, and head *h* reads K/V head `h // (n_heads // n_kv_heads)` — so at Qwen3's 16 query heads over 8 K/V heads, each K/V head serves two query heads, which is what grouped-query attention means. Each head's context is written into its own `head_dim`-wide slice of the context buffer, making the concatenation across heads implicit in the addressing rather than a separate pass.

  Two things had to generalise for this:

  * **Non-square transpose.** `TRANSPOSE` handles square tiles only, but a head's K slice is `m × head_dim` and its transpose is `head_dim × m`. `compile_transpose()` composes the two halves: each 16×16 sub-tile is transposed in place *and* tile position (tr, tc) is written to (tc, tr). A square-tile instruction is sufficient to transpose a non-square matrix once the block positions also swap.
  * **Strided operands.** Each head is a column *slice* of the staged Q/K/V, so its tile offsets must be computed with the **full** matrix's row pitch, not the slice's own width. `compile_linear` gained `a_row_stride` / `b_row_stride` / `out_row_stride` for exactly this. Without them the slice would be addressed as though it were a standalone matrix and would walk into the wrong rows — and heads would have had to be physically copied out first.

  **Sequence length, KV cache and causal masking** were then closed in that order, because each unblocked the next:

  * **Wide-row softmax.** The score row's max and sum must span the whole key length, so a row wider than one tile needed the accumulate-across-tiles treatment — three passes (max, then exp-and-sum, then normalise), each depending on the previous completing over the entire row. No ISA additions: `REDUCE`'s `accumulate` bit already did what was needed. Two scalars rather than one, since pass 2 reads the max while accumulating the sum. Partial tiles are issued at their true width, not padded to 16 — a padding zero read into `REDUCE_MAX` becomes the row maximum whenever every real score is negative, which silently flattens the distribution.
  * **KV cache.** `seq_len`/`cache_len` separate query positions from key positions. This is the shape generation actually needs: one query against a growing cache gives scores of `1 × (cache_len+1)` — a single wide row, which is precisely what the softmax work above made expressible. This step's K/V are appended to the cache before attention runs.
  * **Causal masking** (`CAUSAL_MASK`, 0x0D). Needed only for parallel prefill — in cache-based decode the cache holds only positions at or before the current one, so there is nothing to mask.

  **Masking is a sentinel, not an additive penalty, and that distinction is load-bearing.** The obvious approach — add a −128 mask tile via `ELEMWISE_ADD` — does not work twice over: `clamp8` saturation only reaches −128 when the score is already ≤ 0, and even where it lands, `exp((−128 − max)/T)` at `T = 64` is roughly 13% of the peak, so masked positions would leak substantial attention weight. Masked scores instead carry a sentinel that `EXP_SUB` maps to exactly zero.

  Most of the masking is also free: it is applied per score *tile*, and tiles lying entirely above the diagonal are **not computed at all** rather than computed and discarded. Only the diagonal tile needs the triangular mask; with a cache the diagonal is offset by `cache_len`.

### **RoPE**

Implemented (`compile_rope`, `rope_tables`) and applied to Q and K — never V — per head inside `compile_attention_block`. This was the last remaining *correctness* gap rather than a scope limit: RoPE is the only thing carrying token position in this architecture, so without it the model is order-blind and results are wrong, not merely narrow.

**It needed no new ISA mode**, which is worth explaining because the obvious reading says it should. Qwen/Llama use the half-split convention, `q' = q·cos + rotate_half(q)·sin` with `rotate_half(x₁, x₂) = (−x₂, x₁)`. Two observations collapse that to existing instructions:

* A tile's rotate-half partner is **another whole tile**, offset by `head_dim/2` — not an intra-tile shuffle — provided `head_dim/2` is tile-aligned. `compile_rope` enforces `head_dim % 32 == 0`; real head dimensions (64, 128, 256) satisfy it.
* The negation folds into the **sin table** on the host (first half stored negated), so the device needs no negate instruction either.

What remains is two `ELEMWISE_MUL`s and one `ELEMWISE_ADD` per tile against host-precomputed tables — exactly the design this document originally described ("rotation matrices are pre-computed on the host CPU and loaded via `LOAD_TILE`").

Rotation is applied **before** the KV-cache append, so cached keys are stored already-rotated and earlier positions are never re-rotated on later decode steps. `pos_offset` shifts which table rows are used, so a decode step's query is rotated at absolute position `cache_len` rather than 0.

Verified at `head_dim` 32, 64 and 128 (Qwen3's actual value), with and without a position offset, and end-to-end through a full attention block.

### **Embedding lookup, `lm_head`, and sampling**

**Two of these are not expressible on this ISA, and naming them as host work is the correct design rather than a shortfall.**

* **Embedding lookup is a gather.** `embed_tokens[token_id]` selects a row by runtime value, but `LOAD_TILE`'s address is a compile-time constant in the instruction word — there is no indirect or computed addressing anywhere in the encoding. A runtime token id therefore cannot select a row on-device. The host already builds DRAM, so it writes the row for the current token into the layer-0 input buffer (`embedding_row()`). Adding an indirect-load opcode would be the alternative, and it would be a large change to serve one instruction per token.
* **Sampling needs an argmax.** `REDUCE_MAX` yields the maximum *value*; nothing in the ISA carries an index. Beyond argmax, temperature/top-k/top-p are control flow, which this ISA has none of. The host reads the logits back and chooses.
* **`lm_head` is expressible** — it is a very wide linear. At Qwen3's vocabulary of 151,936 that is 9,496 N-tiles, so this one call dominates a decode step's instruction count. Its output stays INT32: these are logits to be compared, not operands for a further GEMM, and requantising would discard exactly the resolution argmax depends on.

`embed_tokens` was also simply **not being ingested** — `_pass1_classify` matched only `PROJECTION_KEYS` and `NORM_KEYS`, so the embedding table was silently dropped. It now has its own `EMBEDDING_KEYS`, kept separate because it is indexed rather than multiplied and must never be transposed.

**A layout bug found while wiring this up, and it would have silently corrupted every real checkpoint.** PyTorch and Hugging Face store `nn.Linear` weights as `(out_features, in_features)` and compute `x @ W.T`, but this compiler's GEMM consumes B directly as `(in, out)`. Nothing transposed. Feeding a genuine Qwen3 checkpoint through would have computed `x @ W` — shape-compatible only where `in == out`, and wrong everywhere else, with no error raised. `ingest_model` now transposes 2-D projection weights on capture (`hf_layout=True`, on by default, embeddings exempt), and the full-layer tests were switched to generate weights in HF layout so they exercise the transpose rather than hiding it.

### **Generation, and what real scale exposed**

`toolchain/generate.py` closes the loop: encode → per-token decode → sample. The host performs the embedding gather and the argmax (neither is expressible, as above); the transformer layers and vocabulary projection run on the simulated NPU. Because the ISA has no control flow, each step **recompiles** with `cache_len` set to the number of positions already seen — the cache length is baked into addresses and instruction counts, so a fresh program per step is the design, not a workaround. The cache regions keep fixed addresses and the DRAM image carries them forward, so earlier keys and values persist. Verified contextual: identical first tokens with different second tokens produce divergent continuations.

**`compile_model` had been emitting attention with neither RoPE nor causal masking.** Both were implemented and individually verified, but the top-level model compiler never passed `dram_cos`/`dram_sin` or `causal=True` — so the full-model path was order-blind and bidirectional, and the `full_layer_*` cases passed because their references modelled what the compiler emitted rather than what a transformer requires. A capability that exists but is not reached from the top level is not a capability.

**Compiling Qwen3-0.6B at its real configuration** (28 layers, `hidden=1024`, `head_dim=128`, 16 heads over 8 KV heads, `vocab=151,936`) produces **8.8 million instructions** for the layer stack alone — a 141 MB binary over 752 MB of weight DRAM — in about 7 seconds.

**And it surfaced a bug no small test could have found.** The vocabulary projection, computed the obvious way as `logits = x @ W_lm`, needs a B operand of shape `(hidden, vocab)` whose row pitch is `vocab`. The ISA's stride field is **16 bits**. Qwen3's 151,936 does not fit and was silently masking to 20,864 — no error, just the wrong rows loaded. Every test passed because a toy vocabulary fits in 16 bits; only the real model breaks, and it breaks quietly.

Two changes: `_pass3_load`/`_pass3_store` now **raise** rather than mask an out-of-range stride, and `compile_lm_head` computes the projection **transposed** — `logitsᵀ = W_lm @ xᵀ`, taking `W_lm` in its native Hugging Face `(vocab, hidden)` orientation as the A operand, so the pitch is `hidden` (1024) and every stride is small. That also makes `lm_head` exempt from the `hf_layout` transpose, alongside embeddings. At real scale the projection is 3.0 million instructions — a third of the whole program for one token.

### **x86 SIMD**

The NEON path was only ever compiled on AArch64, so on an x86 development machine every test exercised the scalar loop — the ARM path remains unverified. An AVX2 path now sits alongside it: eight output columns at a time, broadcasting one A element across eight widened B values.

Widen-then-multiply (`cvtepi8_epi32` + `mullo_epi32`) rather than a fused INT8 dot product: `_mm256_maddubs_epi16` treats its first operand as *unsigned* and saturates its INT16 intermediate, so it would neither accept signed activations nor stay exact, and VNNI's `dpbusd_epi32` is unavailable on this target. Widening keeps the arithmetic bit-identical to the scalar path, which matters because the entire E2E suite validates against scalar semantics. CMake selects it with `check_cxx_source_runs` rather than `_compiles`, so a toolchain that accepts `-mavx2` on a CPU lacking it cannot produce a binary that faults at runtime.

**~9× faster** on a 256×256×256 GEMM (0.18s → 0.02s), with both builds passing all 48 numeric cases — so each agrees with the reference, and therefore with the other.

### **Running the real Qwen3-0.6B**

The checkpoint is read through `toolchain/hf_loader.py`, which parses safetensors directly rather than pulling in torch (~2 GB) to read a file. The one obstacle is dtype: Qwen3 ships bfloat16, which numpy has no native type for — but bfloat16 is exactly the upper 16 bits of an IEEE float32, so widening is a shift, not a decode.

**It runs.** 311 tensors ingest and quantise in ~10 s; 28 layers plus the vocabulary projection compile to **11.9 million instructions**; one full forward pass executes in **~2 s** (4.86 M instr/s) over 753 MB of INT8 weights, producing logits across all 151,936 tokens. Reported cycle model: 2.41 G serial vs 1.54 G double-buffered, a 1.57× overlap speedup.

**But the output is wrong, and informatively so.** Every input token produces itself as the top prediction. That is not the model predicting repetition — computing `lm_head(embedding)` in numpy reproduces the device's top-5 *exactly*, for every token tried. **The 28 transformer layers contribute nothing measurable**: the residual carries the embedding straight through to the projection, and Qwen3's tied embeddings then make the input token its own argmax.

Calibrating the requantisation shift was the obvious suspect and turned out not to be the cause, though it did surface a real defect. `compile_linear` never set ISA v1.5's `scale_shift`, so every GEMM used the fixed default of 7 — and `calibrate_scale_shifts()` shows that default is correct for only **12 of 196 projections**; the rest need 8–11, meaning ~90% of activations were clipping at 127 rather than scaling. That is now measured per projection from real accumulator magnitudes and threaded through. The emitted instructions carry the calibrated shifts (verified by decoding the stream), and the output is **bit-identical** either way — which is what rules the shift out as the explanation.

The real cause is structural: this quantisation scheme re-quantises to a fixed power-of-two shift at every stage and **tracks no activation scale between stages**, so the relative magnitude of the residual path and the sublayer path is not preserved. The sublayer output arrives small in INT8 next to a full-range residual, and 28 layers of that leave the embedding untouched. Genuine INT8 inference uses calibrated per-tensor activation scales carried through the network, usually with per-channel weight scales — that is a substantive piece of work, and it is the honest next project rather than a bug to patch.

**What this does and does not establish.** The ISA, compiler and simulator are structurally correct and verified: 48 end-to-end numeric cases, every stage checked against an independent reference, and a real 0.6B-parameter checkpoint compiling and executing at the right shapes, addresses and instruction counts. What is *not* established is that the numerics are adequate for inference — and at real scale they demonstrably are not yet.
* ~~**RMSNorm over a full row.**~~ **Done.** `compile_rmsnorm_row()` reduces the sum of squares across every tile of a row before scaling any of them. It needed one ISA extension: `REDUCE` now honours the v1.5 `accumulate` bit, so a reduction can span more tiles than one — the same bit, for the same reason, as K-tiled GEMM accumulating partial products. The two passes cannot be merged, since every tile's contribution must be in the total before any tile is scaled by it.

  This lifted the cap on the full-layer test. `compile_rmsnorm_block` normalised each 16×16 tile by *its own* sum, so at Qwen3's `hidden=1024` it would have applied 64 different scale factors where there should be one; the earlier `full_layer` case pinned `hidden=16` precisely to hide that. Coverage now includes `hidden=48` (three tiles), and the row-norm cases use deliberately uneven per-tile magnitudes so a per-tile scale and a row-wide scale give visibly different answers.
* ~~**Weight loading.**~~ **Done.** `ingest_model(model, capture_weights=True)` now quantizes each tensor to INT8 (per-tensor symmetric, `scale = max|w| / 127`) and lays it out row-major with rows padded to the tile boundary `_pass2_dram_stride()` addresses with; `export_dram_image()` writes those at their assigned addresses as a flat blob for `rtv_sim --dram-image`. `compile_model` is consequently verifiable now, and a full transformer layer executes against real weights and matches a reference exactly (`full_layer_1L`/`full_layer_2L`, the latter confirming the inter-layer activation ping-pong).

  Two bugs fell out of building it:

  * **`ingest_model` under-allocated every ragged tensor.** It sized each weight at `_align(rows*cols, 256)` from the raw element count, but `_pass2_dram_stride()` strides rows by `_padded(cols)` — so a 17×17 tensor needs 32×32 = 1024 bytes while only 512 were reserved, letting the next tensor's base land inside the previous one's padded rows. Inert for Qwen3 (every dimension is a multiple of 16) but wrong for anything ragged. Now allocated at the padded footprint.
  * **Block outputs feeding another stage were INT32 where INT8 was required.** `o_proj` fed the post-attention RMSNorm and `down_proj` fed the next layer, but both emitted raw INT32 accumulators while the consuming `LOAD_TILE` read them at INT8 element width. Both blocks now take `quantize_out`, which `compile_model` sets; standalone callers keep the INT32 accumulator by default.

  The quantization scale is deliberately not recorded in the instruction stream: the ISA's `scale_shift` is a power-of-two right shift and cannot represent an arbitrary float scale, so requantization between stages uses `INT8_QUANT`'s shift and this scale exists only to get weights into INT8 range sensibly.

### **The golden model, and what it found**

Every test in `e2e_test.py` compares the device against a reference that
*mirrors the device's own integer semantics*. That verifies the compiler
lowers what it intended to lower, but it structurally cannot detect that the
intent itself is wrong: if a step is missing from the compiler, it is missing
from the reference too, and both agree on the wrong answer.

`toolchain/ref_model.py` closes that hole. It implements Qwen3's forward pass
as the *architecture* defines it -- float32, read straight from the
checkpoint, no reference to the ISA at all. Greedy decoding from it produces
`"The capital of France is Paris. The capital of Italy"`, so it is trustworthy
ground truth.

It immediately found a defect no ISA-mirroring test could have:

  * **`q_norm` / `k_norm` are absent from the compiler entirely.** Qwen3
    applies RMSNorm to each attention head's Q and K, over `head_dim`, before
    RoPE. `_pass1_classify` matches keys by **substring**, and `"norm"` is a
    substring of `"q_norm"` -- so rather than being dropped, both tensors are
    *misclassified as the final `model.norm`* and filed under layer N. They
    collide with each other in `_weight_map` (one silently overwrites the
    other), consume DRAM, are quantised, and are never read by anything. A
    key that is a substring of another key is the underlying hazard;
    `NORM_KEYS` ordering happens to save `input_layernorm` and
    `post_attention_layernorm`, which is luck, not design.

    Ablating just these two tensors in the float reference -- **no
    quantisation anywhere** -- degrades the output to
    `"The capital of France is Question Question Instructions Instructions"`.
    Q and K are dot-producted against each other, so an unnormalised Q head
    scales every logit in its attention row; the softmax that follows is not
    scale-invariant, and the resulting distribution collapses.

    This is worth stating plainly because it contradicts the earlier working
    theory that INT8 quantisation was the limiting factor. It is not: the
    architecture is incompletely implemented, and that alone is sufficient to
    explain garbage output. Quantisation loss cannot be measured until the
    architecture is complete.

  * **The learned RMSNorm gain is ingested but never applied.** `NORM_KEYS`
    captures `input_layernorm` / `post_attention_layernorm` / `norm` and
    assigns them DRAM addresses, but `compile_model` never passes a
    `dram_gamma` to `compile_rmsnorm_row`, so every gamma is resident and unused.

**Process note.** Both defects are invisible to a self-consistent test suite,
and the suite was at 48/48 while the model emitted noise. A reference derived
from the specification, not from the implementation, is the only kind that can
catch an omission -- and it should have been built before the E2E loop, not
after.

### **Closing the two architecture gaps**

**Learned RMSNorm gains are folded into the following weight matrix at
compile time.** RMSNorm is always immediately followed by GEMMs, and gamma is
diagonal, so

    (x * gamma) @ W  ==  x @ (gamma[:, None] * W)

`GAMMA_FOLD` maps each norm to its consumers -- `input_layernorm` into
q/k/v_proj, `post_attention_layernorm` into gate/up_proj, `model.norm` into
lm_head (141 projections in Qwen3-0.6B). This is exact, emits no
instructions, and sidesteps the real obstacle: Qwen3's gains span 0.017..192
*within one tensor*, a 2757:1 dynamic range. A runtime `ELEMWISE_MUL` against
an INT8 gamma would flush every small channel to zero. Folded into W before
quantisation, that range is absorbed by W's own per-tensor scale.

**q_norm / k_norm cannot be folded and are applied at runtime.** The RMS
divisor is computed *from Q*, so pre-scaling Q by gamma would change the very
quantity being normalised by -- the one place the algebra above does not hold.
These run as a per-head `compile_rmsnorm_row` over each head's head_dim-wide
column slice, in place, before RoPE. Two supporting changes:

  * `compile_rmsnorm_row` gained `row_stride`, so it can normalise a
    128-column slice of a 2048-wide Q using the full matrix's row pitch.
  * `gamma_tile_dram` places a Q0.7 gamma broadcast down a full tile of rows,
    since `ELEMWISE_MUL` reads a tile-shaped operand, not a vector. Here the
    dynamic range is survivable: q_norm/k_norm are one head_dim-wide vector
    shared by all heads, spanning ~16:1.

In-place normalisation is safe because pass 1 completes the entire sum before
pass 2 begins, and pass 2 reads each tile immediately before rewriting it.

**A near-miss worth recording.** `gamma_tile_dram` calls `alloc_staging`, and
it was first called lazily from inside the per-layer loop -- placing each tile
in per-layer scratch, which the next allocation in that same layer overwrites.
That is the identical failure mode as the `_staging_base` collision above,
reintroduced within the hour. The tiles are now allocated before
`_staging_base` is captured, alongside the RoPE tables. **Any buffer read by
more than one layer must be allocated before that line**; the per-layer rewind
makes every address after it transient, and nothing enforces this
structurally.

### **Measurement: what the INT8 design can and cannot do**

Three models now exist, and the middle one is what made the question
answerable at all:

| model | what it is | answers |
|---|---|---|
| `ref_model.py` | float32, from the architecture | what *should* happen |
| `quant_sim.py` | integer, mirroring the ISA | what the *design* can do |
| `rtv_sim` | the device, running compiled code | what *does* happen |

`layer_probe.py` reports per-layer cosine against `ref_model`, so a gradual
decay (precision loss) is distinguishable from a cliff (a bug). Cosine, not
RMSE, because every INT8 stage carries an arbitrary scale and only direction
has to survive.

**Result — weight quantisation granularity is the dominant factor.** Prompt
`"The capital of France is"`, measuring where ` Paris` (12095) ranks:

| weights | cos @ L0 | cos @ L13 | cos @ L27 | argmax | rank |
|---|---|---|---|---|---|
| per-tensor INT8 | 0.7487 | 0.1838 | -0.2345 | `'lobals'` | 110766 |
| per-channel INT8 | 0.9998 | 0.9993 | 0.9992 | `' Paris'` | **0** |

Per-tensor INT8 weights alone destroy the model -- the hidden state is
*anti-correlated* with ground truth by layer 27. Per-channel (one scale per
output channel) is near-lossless end to end. This is the single most
important number in the project: it says the ISA and the pipeline are
adequate and the **quantisation granularity was the blocker**, which no
amount of compiler debugging would have fixed.

**How much precision the per-channel correction needs.** The correction is a
per-output-channel multiply after the GEMM, which `ELEMWISE_MUL` against a
broadcast tile already expresses -- but only at Q0.7:

| correction format | cos @ L27 | rank |
|---|---|---|
| Q0.7 (today's `ELEMWISE_MUL`) | -0.0204 | 88984 |
| per-channel power of two (a per-channel shift) | 0.7319 | 6223 |
| Q0.7 mantissa x per-channel exponent (~13 bits) | 0.9992 | **0** |
| exact float | 0.9992 | **0** |

So neither of the two cheap options works. Q0.7 cannot span the range of
per-channel scales, and a pure power-of-two rounds each scale by up to 1.41x,
which compounds over 28 layers. **An 8-bit mantissa with a per-channel
exponent is sufficient and matches exact float**, and that is the change to
specify: a per-channel scale vector with mantissa and exponent, applied
between the INT32 accumulator and the INT8 requantisation.

**Also settled: gamma must not be folded into the weights.** Folding is exact
in float and destructive in INT8 -- gamma's 2757:1 within-tensor range
multiplies into W's columns and per-tensor quantisation then flushes 44% of
the matrix to zero (`q_proj`: 7.6% -> 43.9% zeroed, 2.50 -> 0.82 effective
bits). With per-channel weight scales the fold is no longer needed to avoid
a runtime multiply, and it should be reverted.

**Not yet measured.** These results hold activations, RoPE, softmax and the
residual at full precision -- the dequantised path keeps activations in float,
so the `int8_acts` flag does not currently engage. Quantised *activations*
therefore remain unmeasured, and the softmax temperature (T=64) and INT8
attention probabilities are untested against a correct baseline. Measuring
them needs `quant_sim` to carry an explicit scale with each INT8 tensor rather
than dequantising to true units -- which is the same scale-bookkeeping the
compiler itself lacks.

**Process note, again.** Getting to these numbers required fixing six bugs in
`quant_sim` itself: a dropped correction on `embed_tokens` and `lm_head`
(both bypass the GEMM helper), a dequantisation scale normalised by `pk.max()`
instead of `pk/127`, INT8 clamps firing on values held in true units rather
than INT8 codes, float probabilities cast to `int32` (flooring every one to
zero), and `q_norm`/`k_norm` gains quantised unconditionally in `forward`
while the trace helper honoured the flag. Every one produced a plausible
wrong answer rather than an error, and the first four rounds of ablation
numbers were pure noise reported as findings. The lesson is the one the
`float_mode` check exists to enforce: **an instrument must be validated
against known ground truth before any measurement it produces is quoted.**

### **Why a pre-quantised INT8 checkpoint does not avoid this**

The obvious shortcut is to stop quantising Qwen3 ourselves and consume an
INT8 checkpoint directly, on the theory that a model already in INT8 removes
the problem. It does not, for two measured reasons.

**Per-tensor INT8 fails regardless of who chooses the scale.** Clipping the
outliers -- what any real quantiser does, rather than scaling to the absolute
maximum -- does not rescue it:

| per-tensor scale | cos @ L27 | rank |
|---|---|---|
| max (100%) | -0.2345 | 110766 |
| 99.99% | -0.1179 | 128212 |
| 99.9% | -0.2924 | 112058 |
| 99.5% | -0.1062 | 67624 |
| 99.0% | -0.1028 | 28106 |

Every one is garbage. The limit is the *granularity* of the scale, not its
value, and no choice of a single number per tensor recovers the model. A
checkpoint quantised by someone else is still subject to this, because it is
a property of Qwen3's weight distribution.

**Real INT8 checkpoints are finer-grained, not coarser.**
`Qwen/Qwen3-0.6B-GPTQ-Int8` -- the official one -- declares:

    "bits": 8, "group_size": 128, "sym": true, "quant_method": "gptq"

That is one scale per **128 weights** along the input dimension, strictly more
scale machinery than the per-channel scheme measured above. Consuming it would
mean implementing per-group scales plus GPTQ's packing (INT32-packed weights,
`g_idx` ordering) -- more work than the per-channel fix, for no numerical gain,
since our own per-channel quantisation already reaches rank 0.

**What the instinct gets right.** The ISA does not need to change to handle
INT8 weights; it is already an INT8 machine, and the weights were never the
issue. What has to change is the **scale representation** -- a per-tensor
power-of-two shift cannot express what INT8 inference requires. That is a
compiler and requantisation-path change, and it is required whether the INT8
weights come from us or from a checkpoint.

### **The scale-bookkeeping bug**

**An INT8 activation buffer in this compiler has no recorded scale, and
`ELEMWISE_ADD` requires both of its operands to have the same one.**

A value is stored as `code x scale`, where the scale is the real-world size of
one integer step. The code alone is meaningless: 9 steps of 0.0025 and 9 steps
of 0.0006 are different quantities. The compiler tracks the codes and discards
the scales -- `grep` for any per-buffer scale field finds nothing.

The residual connection is where this becomes a defect. `compile_residual_add`
emits `ELEMWISE_ADD` over `x` (the residual stream) and the sublayer output,
which is a raw integer add of the two codes. At layer 0 of Qwen3-0.6B:

| buffer | one INT8 code is worth |
|---|---|
| `x`, the residual stream (embedding, per-tensor scale) | 0.00250677 |
| `o`, `o_proj`'s output (after a chain of `>>7` requantisations) | 0.000587858 |
| **ratio** | **4.26x** |

Those two numbers were produced by unrelated paths and nothing compares them.
The device computes `x_code + o_code`, i.e. `x_code*s_x + o_code*s_x`, when
the correct sum is `x_code*s_x + o_code*s_o` -- so **the sublayer output is
weighted ~4.26x too heavily against the skip path.** Adding 3 metres to 5 feet
and recording 8.

The following RMSNorm does not rescue it. RMSNorm removes the *absolute*
scale, but what is corrupted here is the *mixture* -- how much of the
sublayer's contribution survives relative to the residual -- and that
survives normalisation. It happens **56 times**: two residual adds per layer,
28 layers, each with its own mismatch.

**Why every test passed.** `e2e_test.py`'s references are written to mirror
the device's integer semantics, so each reference performs the same
mismatched add and agrees exactly. 51/51 green while the model emitted
`"The capital of France is 对他们uration"`. A test compares two things; when
both are wrong identically it is blind. This is the same defect that drove
`quant_sim`'s layer-0 cosine to 0.0079 -- the embedding held in INT8 units
while sublayer outputs were in true units -- which was fixed there and left
live here.

**Why it is fixable without new hardware.** Every one of these scales is
computable before the chip runs: `s_out = s_in * s_weight * 2^shift`
propagates deterministically from the weights. No runtime reduction is
needed. This is narrower than the "dynamic per-channel activation scale"
problem described elsewhere in this document -- that remains open, but it is
not what breaks the model today.

Aligning the two scales above means multiplying by 4.26, which is **not a
power of two** -- precisely why the `scale_shift` field could never express
it. `PERCHAN_QUANT`'s mantissa-and-shift encoding can represent an arbitrary
ratio, so the conversion folds into a requantisation the compiler already
performs.

**The fix.** Give every activation buffer a scale, chosen at compile time:

  1. Calibrate. Run `ref_model` over a prompt and record the peak magnitude
     of each residual buffer per layer. `s_i = max|x_i| / 127`.
  2. Anchor. RMSNorm's output is Q0.7 by construction (1.0 -> 127), so its
     scale is always `1/127` regardless of its input -- every layer's
     projections start from a known scale.
  3. Target. A projection's correction table is chosen so its output lands
     in the scale its consumer expects, rather than in whatever scale falls
     out of the arithmetic.
  4. Rescale at layer boundaries. The residual grows ~600x across 28 layers
     (max|x|: 0.07 at layer 0, 291 at layer 27), so one global INT8 scale
     would round early layers to zero. Each layer gets its own scale and the
     boundary carries one conversion.

### **The same root cause elsewhere: SILU is exactly ReLU**

`PEArray::exec_alu`'s SILU case computes `x / (1 + exp(-x))` on the raw INT8
**code**, as though the code were the value. SiLU is nonlinear, so it is only
the intended function when one code equals 1.0.

The consequence is measurable and total:

    INT8 codes where SILU is indistinguishable from ReLU: 256/256

SiLU's curvature lives in roughly |x| < 6, and it approaches ReLU outside
that. At integer granularity the largest gap is at x = +-1, where
`silu(1) = 0.731` against `relu(1) = 1` -- a difference of 0.269, which
rounds to the same INT8. **There is no input in the entire INT8 domain for
which this SILU differs from ReLU.** Qwen3's SwiGLU gate is currently a
ReLU gate.

The same reasoning applies to the softmax temperature (`T = 64`), which is a
constant chosen against an assumed score scale that nothing establishes, and
to `ELEMWISE_MUL`'s fixed `>>7`, which asserts both operands are Q0.7.

This is the residual-add bug in a different costume. Linear operations are
forgiving of an unknown scale -- the error is a constant factor. Nonlinear
ones are not: applied at the wrong scale they compute a different function
entirely, and no downstream normalisation recovers it. Any scale-bookkeeping
design has to give every nonlinearity a defined input scale, not only the
operations that add two buffers together.

### **What INT8 can actually do (measured over 8 prompts)**

An earlier version of this section reported ` Paris` at rank 0 and called the
INT8 design solved. That number was from **one prompt**, and it did not hold:
across eight prompts the same configuration agrees with the float model on
5 of 8. The corrected numbers are below. `toolchain/quant_eval.py` reproduces
them, so this claim is checkable rather than remembered.

Two metrics against `ref_model`: top-1 agreement, and the mean rank of the
float model's chosen token (graded, so "close everywhere" beats "perfect on
half").

| configuration | top-1 agree | mean rank |
|---|---|---|
| as designed — W8 A8 R8, per-token | 4/8 | 2.38 |
| wider residual — W8 A8 R16, per-token | 6/8 | 1.62 |
| per-channel residual — W8 A8 R16 | 5/8 | 3.75 |
| + wider activations — W8 A12 R16, per-channel | 6/8 | 0.38 |
| + wider activations — W8 A16 R16, per-channel | 6/8 | **0.25** |

**Weights are not the problem.** Every row above uses INT8 weights. What buys
accuracy is width and granularity on the ACTIVATION side -- the opposite of
where this project spent most of its effort.

**The limit is in-row dynamic range.** At layer 2, `down_proj` channel 35
reaches 2610 against a median channel of 0.065 -- 40000:1 *within a single
row*. Per-token scaling cannot help, because the outlier is inside the row it
is scaling. Even 24-bit activations leave typical channels ~0.25% precision,
and over 28 layers that compounds to a layer-27 cosine of 0.90. This is why
top-1 agreement plateaus at 6/8 no matter how many bits are added: the
remaining error is structural, and the fix is outlier handling (mixed
precision for the handful of massive channels, or SmoothQuant-style migration
into the weights), not width.

**Validation.** `quant_eval.py --validate` bypasses all quantisation and
requires `scale_sim` to reproduce `ref_model` exactly (worst per-layer cosine
1.000000). This gate did not exist when the rank-0 result was reported, and
building it surfaced two real bugs: `_rmsnorm` regularised with `eps` on the
CODE rather than the value -- eps is not scale-invariant, so this added a
different regulariser each layer -- and `QT` stored integer codes in float32,
whose 24-bit mantissa silently capped every wide-residual experiment.

### **Porting the recipe: ISA v1.6 (dtype) and the scaled residual**

The first half of the port from `scale_sim` to the compiler is done.

**A dtype field.** `dim_m` wire bits 14:13 now select the element width of an
instruction's operand tile: `0 = INT8`, `1 = INT16`, `2 = INT32`. Zero is the
ISA's original behaviour, so every previously-encoded program decodes
unchanged. The field cost nothing -- those bits were already reserved.

**INT16 where it is needed, nowhere else.** Only the residual path is widened:
`ELEMWISE_ADD` (the residual accumulation itself), `RESCALE` (how an INT8
sublayer output enters it), `INT8_QUANT` as an in-place widen, and the
`REDUCE_SUMSQ` / `RMS_SCALE` pair that lets RMSNorm read the wide stream and
narrow it back to Q0.7 INT8. `REDUCE_SUMSQ` accumulates in int64 -- 1024
elements of (2^15)^2 is 2^40, which overflows the int32 the INT8 path used.

**`compile_residual_add` now takes `rescale_b`,** a (mantissa, shift) pair
built by `encode_ratio(s_b / s_a)`, and emits an `EXEC_ALU RESCALE` that
converts operand B into A's scale before the add. This is the fix for the
defect documented above: at layer 0 of Qwen3 those scales differ by 4.26x,
which is not a power of two, and the add previously mixed the two units.

**A collision the port walked straight into.** `SRAMMap.TILE_BYTES` is 256 --
a 16x16 **INT8** tile. An INT16 tile is 512 bytes, so `input_slot(0)` and
`input_slot(1)` overlapped, and exactly half of every result was wrong from
row 8 onward. `input_slot` now takes a `width` and validates against the zone
end. The 1 KiB input zone holds four INT8 slots but only two INT16 ones.
Worth recording as the byte-vs-element hazard appearing for the fourth time
in this project: every SRAM helper sized in bytes for one element width is a
latent bug the moment a second width exists.

Verified by three E2E cases (`resid_scaled_*`) that store operands with
genuinely different scales -- including the measured 4.26x -- and by
`Int16ResidualOps` in the PE-array suite, which checks INT16 saturation,
int64 sum-of-squares, and the INT16-to-INT8 narrowing in `RMS_SCALE`.

**Remaining in the port:** static calibrated scales threaded through
`compile_model`, a LUT activation mode so SiLU stops being ReLU, and the
softmax temperature derived from the score scale.

### **Port, second instalment: LUT and the INT16 RMSNorm boundary**

**`LUT` (0x12).** `EXEC_ALU` mode: `out[i] = table[tile[i] + 128]`, with a
256-byte INT8 table at `b_sram`. The compiler builds the table for the
buffer's known input scale (`lut_table` / `lut_dram`), so the function is
evaluated at the **true value** rather than at the integer code. Measured at
Qwen3's layer-2 gate scale, the table reproduces SiLU to 0.157 -- exactly half
an output quantisation step, i.e. as close as INT8 output allows -- against
0.278 for the ReLU that `SILU` currently computes. This is how fixed-function
units evaluate nonlinearities, and it costs one SRAM load.

**RMSNorm now reads the INT16 residual.** `compile_rmsnorm_row` takes a
`dtype` for its INPUT; the output is always Q0.7 INT8, because a normalised
value needs no range and the scale cancels (`x/rms(x)` is unchanged by what
one step is worth). This is the boundary where the wide stream narrows.

**The reduce scalar had to widen to INT64.** A 16x16 tile of INT16 sums
squares up to 256 * (2^15)^2 = 2.7e11, and a full row spans several tiles.
The INT32 scalar saturated at 2.1e9 and RMSNorm silently normalised by the
wrong quantity -- caught by an E2E case whose input carries one massive
channel (30000) alongside ordinary ones, which is the real distribution.
Wide accumulators exist in real hardware for exactly this reason.

Verified by `RMSNormInt16Case` (E2E) and an extended `Int16ResidualOps`
covering the INT64 sum-of-squares at values that would overflow INT32.
**144 PE-array tests, 58/58 E2E, 4/4 ctest.**

**Remaining in the port:** calibrated scales threaded through `compile_model`
(the pass that fills in every `RESCALE` constant and LUT table for all 28
layers), and the softmax temperature derived from the score scale.

### **Port, third instalment: the calibration pass**

**`ScaleQwen3.collect_scales`** runs the pipeline once and records what one
integer step is worth in every buffer the compiler emits instructions for --
`x_in`, `y`, `q/k/v`, `q_rope/k_rope`, `scores`, `probs`, `ctx`, `o`,
`x_attn`, `y2`, `gate/up`, `silu`, `act`, `d`, `x_out`, per layer. This is
the pass the device cannot do for itself: there is no reduction over a buffer
and no way to feed a computed value into a later instruction's immediate
field, so every scale must be a compile-time constant.

Measured on Qwen3-0.6B, the ratios that a residual add has to reconcile:

| layer | o -> x | d -> x |
|---|---|---|
| 0 | 32.9 | 347.8 |
| 1 | 34.2 | 142.3 |
| 13 | 0.10 | 0.26 |
| 27 | 0.94 | 222.8 |

Nearly four orders of magnitude apart, and none of them a power of two. This
is the quantitative form of the scale-bookkeeping bug: `ELEMWISE_ADD` was
adding these operands raw, as though every one of those ratios were 1.

**A limit this exposed.** `RESCALE`'s multiplier is `mant / 2^shift` with an
8-bit mantissa, so it tops out at 255 -- and 347.8 does not fit. The fix
falls out of the pipeline's own shape: an INT8 sublayer output has to be
widened into the INT16 stream anyway, and a plain widen leaves it at INT8
magnitude (max 127), wasting the range it was just given. The widen now takes
a **left shift** (`params[7:0]`), so shifting by 8 absorbs a factor of 256 and
brings every measured ratio back inside RESCALE's mantissa. One instruction,
no new opcode, and the INT16 buffer actually uses its range.

`compile_residual_add` now takes `b_dtype` and `widen_shift`, so the real
shape -- INT16 residual, INT8 sublayer output widened and scale-aligned into
it -- is expressible and tested (`resid_mixed_*`, `resid_bigratio`, the last
using the measured 348).

### **Port status: wired end to end, output still wrong**

`compile_model` now consumes a scale plan. What the instruction stream
carries, per layer, is no longer implicit:

  * the residual stream is INT16 (`layer_io` and `attn_res_dram` doubled)
  * both RMSNorms read INT16 and write Q0.7 INT8
  * both residual adds widen the INT8 sublayer output (left shift 8),
    `RESCALE` it into the stream's units, and convert the stream itself at
    the layer boundary
  * every projection's correction table is **retargeted** so its output lands
    in the scale the plan assumed -- without this the RESCALE constants were
    computed against a scale nothing enforced, which is a subtle way to be
    exactly wrong
  * SiLU goes through a per-layer lookup table built for that layer's gate
    scale

Instruction mix on Qwen3-0.6B confirms the shape: 3584 INT16 `ELEMWISE_ADD`
(56 residual adds x 64 tiles), 3584 widens, 5248 `RESCALE` (3584 operand-B
plus 1664 boundary conversions), 3584 INT16 `REDUCE_SUMSQ`/`RMS_SCALE` pairs.
Every ratio encodes to within 0.33% across a 0.099-223 range.

**And the model still emits noise.** `"The capital of France is endetems{o"`.

Two things are genuinely not ported: the **attention interior** -- q_norm/
k_norm gain scales, RoPE's Q0.7 tables, and the softmax temperature, which is
still the hardcoded `T = 64` rather than derived from the score scale -- and
any lowering bug in what *was* ported would look identical from here.

**The next step is not another run.** Each one costs two minutes and returns
one bit. The three-model structure exists precisely for this: `scale_sim`
reaches rank 0-1 on this prompt with these settings, so comparing the DEVICE's
per-layer hidden state against `scale_sim`'s isolates lowering bugs from
design gaps, the same way `layer_probe.py` did for `quant_sim` against
`ref_model`. That comparison should have been built before this run, not
after it.

**A hazard that recurred for the third time.** `lut_dram` calls
`alloc_staging`, and calling it inside the per-layer loop would have placed
each table in per-layer scratch for the next allocation to overwrite --
identical to the `_staging_base` collision and the gamma-tile near-miss. Caught
by inspection this time. Any buffer read across a layer must be allocated
before `_staging_base` is captured, and nothing in the allocator enforces it.

### **`device_probe.py`: the third comparison**

    ref_model  vs  quant_sim   what quantisation costs
    ref_model  vs  scale_sim   whether the design is sound
    scale_sim  vs  DEVICE      whether the compiler lowers it correctly

The third one was missing. `device_probe.py` compiles the model truncated to
L layers, runs it, dumps the INT16 residual, converts codes back to values
using the scale the compiler itself planned, and reports cosine against
`scale_sim` at the same depth.

**A property of the probe worth stating.** It runs a SINGLE token, which
makes the attention interior inert: softmax over one element is 1 whatever
the temperature, and RoPE at position 0 is the identity. So q_norm/k_norm,
the Q0.7 rotation tables and `T = 64` -- the parts not yet ported -- cannot
influence the result. Any divergence it reports is in the ported path.

**First finding: fixed-convention ops do not get a calibrated scale.**
`ELEMWISE_MUL` is `(a*b)>>7`, so the SwiGLU product lands at
`s_silu * s_up * 128` -- the product of the two PEAKS. The peak of the
product is a different number whenever the peaks do not coincide: measured at
layer 0, 5.265 against 2.803, so `down_proj` was being fed a buffer **1.89x
off**. The plan now derives these intermediates from the device's own
arithmetic rather than from calibration. Cosine at 3 layers went 0.089 ->
0.489.

| layers | before | after |
|---|---|---|
| 1 | 0.824 | 0.838 |
| 2 | 0.474 | 0.557 |
| 3 | 0.089 | 0.489 |

The general rule: **calibration may only set the scale of a buffer whose
scale the compiler actually controls.** Where a fixed shift or a Q0.7
convention decides it, the compiler must compute what the convention yields
and use that.

**Second finding, open.** Layer 1 barely moved, so a distinct difference
remains inside layer 0. The leading candidate is that `scale_sim` applies the
RMSNorm gain to the ACTIVATION while `ingest_model` folds it into the weight
(`GAMMA_FOLD`). Those are equivalent in float and not under quantisation --
folding measurably worsens weight quantisation (1.26% -> 2.40% of weights
flushed to zero even with per-channel scales). The two models are therefore
computing different things, and the probe is right to report it.

### **The stage dump, and the Q0.7 RMSNorm bug it found**

`device_probe.py --stages N` dumps all five per-layer buffers -- `norm`,
`attn_out`, `attn_res`, `mlp_out`, `layer_out` -- and compares each against
`scale_sim`. Because every buffer carries its own width and scale, the report
includes a magnitude ratio, and the two failure modes look different: a wrong
CONSTANT gives output that is correlated but uniformly scaled, a lowering bug
gives output that is decorrelated. The single-value probe could not tell them
apart.

The first dump read:

| stage | cosine | dev/ref magnitude |
|---|---|---|
| norm | 0.732 | **0.073** |
| attn_out | 0.957 | 0.679 |
| attn_res | 0.956 | 0.686 |
| mlp_out | 0.659 | 0.447 |
| layer_out | 0.835 | 0.590 |

**The bug: RMSNorm's Q0.7 output convention is wrong for this model.**
`RMS_SCALE` maps 1.0 to 127, which silently assumes the normalised
activation peaks near 1.0. RMSNorm does not bound its output -- it divides by
the RMS, and a massive activation channel stays massive afterwards. Measured
across all 56 norms in Qwen3-0.6B:

    peak / RMS:  min 1.18   median 8.67   max 506.04

At the median the norm output was using **11.5% of the INT8 range**, and
everything above 1 RMS saturated at 127. At layer 27 the peak is 506x the
RMS, so the convention discarded almost the entire signal.

The numerator is now calibrated: `scale_num = 127*sqrt(n) / peak_ratio`, which
makes the output scale `peak_ratio/127` instead of `1/127`, and the consuming
projections' correction tables are retargeted to agree. Effect:

| stage | before | after | | depth | before | after |
|---|---|---|---|---|---|---|
| norm | 0.732 | **0.985** | | 1 layer | 0.835 | **0.957** |
| attn_out | 0.957 | **0.995** | | 2 layers | 0.561 | **0.871** |
| attn_res | 0.956 | **0.995** | | 3 layers | 0.489 | **0.981** |
| mlp_out | 0.659 | **0.900** | | 4 layers | 0.487 | **0.980** |
| layer_out | 0.835 | **0.957** | | | | |

Magnitudes now agree too (4.07 against 4.22 at one layer, where the device was
previously 1.70x low).

**A false alarm worth recording.** The first dump showed `norm` at cosine
0.09, which looked like a catastrophic failure in RMSNorm. It was the probe's
error: `compile_model` REUSES `norm_dram` for both of a layer's RMSNorms, so
what survives to the end of a run is the post-attention one, and the probe was
comparing it against the pre-attention value. The buffer was fine. A probe
that does not model buffer lifetimes will invent bugs.

### **A class of bug: fixed conventions that assume a magnitude**

Three separate defects turned out to be the same mistake, and naming the
pattern is more useful than listing them. Each is an ISA constant chosen as
though the data had a known size, when nothing established that size:

| convention | assumed | actually measured on Qwen3 |
|---|---|---|
| `RMS_SCALE` maps 1.0 to 127 (Q0.7) | normalised activation peaks near 1.0 | peak/RMS **1.18 to 506**, median 8.67 |
| `ELEMWISE_MUL` is `(a*b)>>7` | both operands fill the INT8 range | small products vanish: 5*3 = 15 shifts to **0** |
| `EXP_SUB` uses `T = 64` | one score step is worth 1/64 | required T is **2.91 to 23.51**, median 7.66 |

All three now take the value from calibration:
`scale_num = 127*sqrt(n)/peak_ratio`, a per-instruction shift on
`ELEMWISE_MUL`, and `T = 1/s_scores` passed as Q8.8. All three default to the
old constant when unset, so previously-encoded programs are unchanged.

**Effect, measured with the stage dump at layer 0:**

| stage | initial | after RMSNorm fix | after product shift |
|---|---|---|---|
| `norm` cos / med_rel | 0.732 / — | 0.985 / 0.147 | 0.985 / 0.147 |
| `attn_out` | 0.957 | 0.995 / 0.109 | 0.995 / 0.109 |
| `mlp_out` | 0.659 | 0.900 / 0.519 | **0.966 / 0.282** |
| `layer_out` | 0.835 | 0.957 / 0.456 | **0.985 / 0.244** |

**A metric lesson alongside it.** Plain cosine was nearly useless here: the
residual is dominated by massive activation channels (channel 35 at 2610
against a median of 0.065), so the inner product mostly reports whether that
one channel is right. The probe now also reports cosine with the largest
channels trimmed and the MEDIAN per-channel relative error, and it was the
latter that exposed the SwiGLU product -- `mlp_out` read cosine 0.900 while
its median channel was **52% wrong**.

**The searchable form of this bug:** any constant in the ISA that encodes a
scale -- a fixed shift, a fixed temperature, a fixed fixed-point format --
is an assumption about magnitude that must either be justified from the data
or become a parameter. `SOFTMAX_TEMPERATURE`, the Q0.7 RoPE tables, and
`INT8_QUANT`'s default shift of 7 are the remaining instances.

### **FP32: running with no quantisation at all**

`Decoder::DType::FP32` is a fourth element width, and its purpose is
diagnostic rather than architectural. Every hard question in this project has
been "is this a quantisation limit or a lowering bug?", and a float run
answers it outright: if the float path is correct, then every address,
tiling decision, K-accumulation, double-buffer and schedule in the compiler
is proven, and anything still wrong in the INT8 path is quantisation.

**Several modes disappear in float**, which is itself informative about what
they exist for. `RESCALE` converts between two scales and `INT8_QUANT`
narrows an accumulator -- both are identities when there is one scale.
`LUT` precomputes a nonlinearity for a known scale; in float it is a direct
SiLU. Floats carry their own exponent, so the scale-bookkeeping bugs
documented above are not merely absent but *unrepresentable*. That quantifies
how much of this project's difficulty is inherent to INT8 rather than
incidental.

**Sum of squares accumulates in double** on the FP32 path: a 1024-wide row of
squares loses low-order terms in float32, and that sum is exactly RMSNorm's
divisor.

**Two SRAM bugs the float path exposed, both the byte-vs-element hazard.**
An FP32 16x16 tile is 1024 bytes -- the *entire* input or weight zone -- so
`compile_linear`'s weight double-buffering could not fit two slots.
Wide element widths now use `WIDE_INPUT_BASE`/`WIDE_WEIGHT_BASE` in the
~46 KiB of unused SRAM above the scratch area, rather than resizing the
narrow zones, which would move every `c_slot` address and break agreement
with `simulator.cpp`'s `c_slot_addr()`. Then the loads went to the wide zone
while `COMPUTE_GEMM` still read the narrow one, and every result was zero.
Both operand slots now take the width. That is the fourth and fifth
appearance of this hazard.

**Element width lives on the compiler** (`self.dtype`), not on each call.
`compile_linear` is reached through `compile_attention_block`,
`compile_mlp_block`, `compile_sibling_qkv` and `compile_model`; a threaded
parameter would have been four more places to forget, and forgetting one has
already happened twice in this file.

**A test-methodology note.** The first FP32 GEMM test failed at relative
error 2.3e-4, which was the TEST being wrong. Its inputs span 1e8 by
construction, so a dot product can cancel to near zero and any float32
accumulation order leaves order-1 relative error on the residue. Error is now
measured against `|a| @ |b|` -- the magnitude actually summed -- which is what
float32 precision applies to.

Covered by `Fp32Ops` (PE array: GEMM with accumulate, elementwise, reduce,
RMS_SCALE unclamped, SiLU that is demonstrably not ReLU) and by
`fp32_linear_*` and `fp32_mlp_*` end to end.

**Remaining:** `compile_attention_block` and `compile_model` still assume
1 byte per element in their staging and per-head addressing.

### **The TVM path, working (Version B)**

`toolchain/rtv_tvm/` compiles through TVM rather than by hand, verified end
to end against the same simulator and the same numpy references as
everything else.

| file | role |
|---|---|
| `intrin.py` | `rtv.gemm_16x16` and `rtv.add_16x16` -- what one instruction computes |
| `schedule.py` | tile to the PE array, stage through SRAM, tensorize |
| `codegen.py` | walk scheduled TIR, emit 128-bit words |
| `test_tvm_path.py` | 7 cases, exact against numpy, wired into ctest |

**The mapping came out direct**, which is the substantive finding -- the ISA
and TVM's scheduling model line up rather than needing to be forced:

    cache_read   ->  LOAD_TILE      (DRAM -> SRAM)
    tensorize    ->  COMPUTE_GEMM / EXEC_ALU
    cache_write  ->  STORE_TILE     (SRAM -> DRAM)

**Independent convergence on instruction count.** Given only the intrinsic
and told to stage through SRAM, TVM's scheduler produces exactly the loop
structure that was hand-derived:

| shape | hand-written | TVM |
|---|---|---|
| 16x16x16 | 6 | 6 |
| 16x16x64 | 21 | 21 |
| 32x32x32 | 44 | 44 |
| 32x48x64 | 126 | 126 |

That is evidence the intrinsic describes the hardware correctly rather than
coincidentally.

**The codegen computes no addresses.** Every offset and stride is read off the
TIR, which TVM derived from buffer types. That is the ~276 lines of
byte-vs-element arithmetic the hand-written compiler carries, and the source
of roughly seven of this project's bugs.

**It fully unrolls, and must.** The ISA has no control flow, so a TIR loop
becomes repeated instructions. The same property forces a fresh program per
token, which no compiler change fixes.

**Three things learned about TensorIR's shape**, each a bug while writing the
walker:

  * Block iteration variables are bound by `SBlockRealize.iter_values`, not by
    the enclosing loops.
  * Staging copies sit INSIDE their `ax0, ax1` grid, so a naive walk emits one
    `LOAD_TILE` per ELEMENT -- 5144 instructions where 44 were correct.
  * `scope="shared"` on the intrinsic is load-bearing: `tensorize` refuses a
    global-scope nest with `scope(): global vs shared`, which is TVM enforcing
    a hardware fact rather than trusting the author to remember it.
  * `blockize` + `decompose_reduction` are required before `tensorize`, because
    `T.init()` leaves the block reading two buffers where the intrinsic
    describes three.

**Naming in this release**: `tvm.tir` is `tvm.s_tir`, TVMScript's dialect is
`tvm.script.tirx`, `T.block` is `T.sblock`, `get_block` is `get_sblock`,
`Call` lives in `tvm.ir`, and PrimFuncs need `s_tir=True`. No published
tutorial matches; installed sources are the reference.

**Four operations now compile through TVM**, covering three structurally
different shapes:

| intrinsic | ISA | shape |
|---|---|---|
| `rtv.gemm_16x16` | `COMPUTE_GEMM` | reduction over K into an INT32 accumulator |
| `rtv.add_16x16` | `EXEC_ALU ELEMWISE_ADD` | two INT8 operands, INT8 out |
| `rtv.mul_16x16` | `EXEC_ALU ELEMWISE_MUL` | as above, with the Q0.7 `>>7` in the desc |
| `rtv.sumsq_16x16` | `REDUCE REDUCE_SUMSQ` | tile collapses to one INT32 scalar |

12 cases pass end to end against numpy.

**Two rules that tensorize enforces**, both discovered by being refused:

  * **Every buffer the intrinsic touches must be staged into its scope.**
    Not just the tiles -- the reduction's scalar accumulator too, which failed
    with `scope(): global vs shared` on S until `cache_write` staged it. TVM
    is checking a hardware fact rather than trusting the author.
  * **Read sets are compared positionally, accumulator first.** `sumsq`
    declared `T.reads(A, S)` and was refused; `T.reads(S, A)` matched. The
    gemm desc happened to have `C` first, which is why it matched immediately
    and hid the rule.

**A bug worth recording, because the shape of it recurs.** `accumulate` was
inferred from a single loop variable, which is correct when a reduction is
split across one axis and wrong when it is split across two. A 32x32 sum of
squares splits into `i_0` and `j_0`; resetting whenever `j_0 == 0` discarded
the previous `i_0` pass and returned **half** the true sum -- 1732699 against
3374244. The reduction writes one scalar for the whole program, so the
correct rule is "first emission resets", independent of how many axes the
schedule chose. Reading a semantic property off one syntactic feature of the
schedule is the mistake; the property belongs to the reduction, not the loop.

**Still hand-written**: `dep_id`/`WAIT_SYNC` assignment (TVM has no model of
DMA dependency tokens), every scale decision, and the operations that remain
before a whole layer can take this path -- `RMS_SCALE`, softmax's three
passes, and the LUT nonlinearity.

### **TVM migration: the kernel set is complete**

The architectural mandate is that the TVM compiler stack (`toolchain/rtv_tvm/`) REPLACES `toolchain/npu_compiler.py` and any custom `.py` compiler scripts entirely -- not wraps them, not runs beside them. The kernel half of that is complete: every operation the ISA can perform on a tile is expressible as a TVM tensor intrinsic, scheduled by TVM, with addresses read off the TIR rather than computed by hand.

Eleven intrinsics, 28 passing tests:

| Instruction | Intrinsic | Replaces |
|---|---|---|
| `COMPUTE_GEMM` | `rtv.gemm_16x16` | `compile_linear`, `compile_lm_head` |
| `EXEC_ALU ELEMWISE_ADD` | `rtv.add_16x16` | `compile_residual_add` |
| `EXEC_ALU ELEMWISE_MUL` | `rtv.mul_16x16` | part of `compile_mlp_block` |
| `REDUCE REDUCE_SUMSQ` | `rtv.sumsq_16x16` | `compile_rmsnorm_row` pass 1 |
| `SCALAR_ALU RMS_SCALE` | `rtv.rms_scale_16x16` | `compile_rmsnorm_row` pass 2 |
| `REDUCE REDUCE_MAX` | `rtv.rowmax_16x16` | `compile_softmax_rows` pass 1 |
| `SCALAR_ALU EXP_SUB` | `rtv.exp_sub_16x16` | `compile_softmax_rows` pass 2 |
| `REDUCE` (sum) | `rtv.rowsum_16x16` | `compile_softmax_rows` pass 2 |
| `SCALAR_ALU SCALAR_DIV` | `rtv.scalar_div_16x16` | `compile_softmax_rows` pass 3 |
| `EXEC_ALU LUT` | `rtv.silu_16x16` | `compile_lut` |
| `EXEC_ALU TRANSPOSE` | `rtv.transpose_16x16` | `compile_transpose` |

Two compositions are verified end to end against the device: **RMSNorm**
(sumsq then rms_scale, the scalar passing between them through SRAM) and
**softmax** (all three passes, exact against the INT8 semantics in
`pe_array.cpp`). Softmax is the harder one -- two scalars are live at once,
because pass 2 reads the row max while accumulating the row sum, so a single
scratch slot would have pass 2 destroy the value it is still reading.

**What the tests taught, all of it about the tests rather than the compiler:**

  * A **transpose** case "failed" at 32x32 while passing at 16x16. The
    compiler was right: the TIR says whole-matrix transpose, so at output
    tile `(io, jo)` the source tile is `(jo, io)`, and the codegen gets that
    swapped address for free by evaluating the store's index expressions.
    The REFERENCE was a blockwise transpose, which agrees only at one tile.
    A 16x64 case in the same batch was not merely wrong but ill-posed --
    `A[vj, vi]` asks for row 63 of a 16-row buffer.
  * The **LUT** reference indexed `code & 0xFF` where the device indexes
    `table[code + 128]`. Biased, not two's-complement wrapped; the two agree
    on non-negative codes and differ on every negative one, which is most of
    SiLU's interesting range.
  * TVMScript is **not Python**. A local binding inside a block
    (`x = T.Cast(...)`) is parsed as a BUFFER declaration and fails
    well-formedness; and TVMScript parses a function's SOURCE TEXT, so a
    PrimFunc built with `exec()` dies with "could not get source code". Both
    force explicit, duplicated spelling over factoring.

**What remains, and it is the larger half.** Kernels were the tractable part.
Still hand-written: `compile_attention_block`, `compile_mlp_block`,
`compile_sibling_qkv`, `compile_rope`, and `compile_model` -- roughly 900
lines that place staging buffers and compute addresses. *That is where both
bugs above lived*, so porting kernels alone does not retire the bug class;
porting composition does. Also outstanding: `dep_id`/`WAIT_SYNC`, which has
no TVM analogue, and weight ingestion, which TVM cannot do at all (it does
not read safetensors) and which will survive as a shrunken adapter.

### **Resolved: the FP32 divergence was a staging overrun**

Layer 0 of the real model in float sat at cosine 0.128 against `ref_model`
across several sessions. It is now **1.0000**, and the compiler's lowering is
proven correct at model scale: every address, tile, stride, K-accumulation and
schedule in `compile_model` reproduces the float reference exactly. Anything
still wrong in the INT8 path is quantisation, not lowering.

**The bug.** Five staging buffers in `compile_attention_block` were sized by
the logical sequence length:

    ctx_stage = self.alloc_staging(sq * _padded(q_dim) * ew)

`sq` is how many query positions exist; the compiler emits **tiles**, and a
tile is always 16 rows tall. `compile_linear` stores `dim_m = t` regardless of
how many rows carry data. At Qwen3's decode shape -- `sq = 1`,
`q_dim = 2048`, FP32 -- that allocates 8,192 bytes for 131,072 bytes of
writes, a **16x overrun**. `alloc_staging` is a bump allocator, so
`scores_stage` overran into `probs_stage` into `ctx_stage`. Silent, no fault.

**Why nothing caught it.** Every test used `seq_len = 16`, where `sq` equals
`tile_dim` and the expression is exactly right. The bug is arithmetically
invisible at the only size anything exercised, and appears only at
`seq_len < 16` -- which is decode, the mode `generate.py` actually runs. The
fix is `_padded(sq)`: size by what is written, not by what is logically
present.

**Two lessons from how it was found**, both uncomfortable:

  * **Six isolated component tests all came back exact** -- RMSNorm at every
    configuration, RoPE at both head dimensions, attention at both sequence
    lengths, MLP, linear. The bug was in none of them, because it was not in a
    component: it was in how a component was SIZED by its caller. Testing
    components in isolation cannot find that, and the method should have been
    abandoned after the second clean result.
  * **The lead that produced it was a bad test.** A reported "exact at
    seq_len=16, wrong at seq_len=1" result was the harness comparing
    incompatible computations -- a 16-position reference against a
    1-position run. The conclusion was right and the reasoning was wrong;
    reading the staging allocations while investigating it is what found the
    overrun.

**The class is familiar.** A size computed from a logical quantity rather than
the physical one, correct at a single configuration and wrong elsewhere --
the same shape as the seven byte-vs-element defects. It is also precisely what
a typed IR removes: TVM derives buffer extents from the tiling instead of
trusting two hand-written numbers to agree.

### **Audit coverage**

Audited and end-to-end verified: `compile_linear`, `compile_sibling_qkv`, `compile_rmsnorm_block`, `compile_attention_block`, `compile_mlp_block`, and the `_pass2_`/`_pass3_` helpers they rest on.

Audited, correct as written, not separately covered: `export_binary` (flat write of the instruction stream), `compile_exec_alu` and `append_synchronization_barrier` (thin wrappers), `disassemble`.

`toolchain/swizzle.py` was audited and is internally correct — `swizzle_matrix` round-trips against `tile_dram_addr` for both tile-aligned and ragged dimensions, and pads as documented. It is, however, **dead code**: nothing calls it, and it assumes an *unpadded* row-major source (row stride `N`) while `_pass2_dram_stride` assumes rows padded to tile boundaries (stride `_padded(N)`). Wiring the swizzler up would have to reconcile that difference first.

`compile_model` is verified only structurally (it emits a well-formed stream of the expected shape); it cannot be numerically verified until weight loading exists. The three staging-address fixes therefore remain assertion-verified rather than proven by execution.

One clarification on an earlier note: `npu_compiler.py` *does* import `rtv_isa` — lazily, inside `disassemble()`. It is only the **instruction-packing** path (`ScheduleNode` → `pack_instruction`) that is independent of it, which is what allowed the two to drift over ISA v1.5 while disassembly still worked.

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
* **PE Array:** A functional model of a $16 \times 16$ systolic array engine accelerated with **Arm Neon intrinsics** on the target Raspberry Pi 5 platform.

---

## **3. Software Stack & Compiler**

The compiler is **Apache TVM with a BYOC (Bring Your Own Codegen) backend and TensorIR scheduling**, rather than a hand-crafted toolchain composed of standalone Python compiler scripts. This establishes TVM as the sole official compiler architecture for RTV-Accel, permanently replacing the legacy prototype approach of writing `.py` file(s) representing the compiler (`toolchain/npu_compiler.py`, ~3000 lines). The decision to use TVM is based on what building and debugging the prototype compiler taught.

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
  3. **Python-accessible lowering passes:** Custom quantization calibration, scale propagation, and LUT generation remain in high-level passes without needing to write complex C++ MLIR dialects.

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
        |                 - double_buffer for DMA / compute latency hiding
        |                 - tensorize to target RTV-Accel intrinsics
        v
[TVM BYOC Codegen]  ----> rtv_tvm/codegen.py:
        |                 - Read type-derived buffer addresses & strides
        |                 - Assign hardware dependency tokens (dep_id / WAIT_SYNC)
        |                 - Lower calibrated scale plans & LUT tables
        v
[128-bit instruction stream] -> rtv_sim / Hardware Driver
```

### **Hardware Tensor Intrinsics (`toolchain/rtv_tvm/intrin.py`)**

Every hardware operation on the 16x16 PE array and vector ALU is registered as a formal TVM TensorIR intrinsic:

| Intrinsic | ISA Instruction | Hardware Semantics |
|---|---|---|
| `rtv.gemm_16x16` | `COMPUTE_GEMM` | 16x16 matrix multiply with reduction into INT32 accumulator |
| `rtv.add_16x16` | `EXEC_ALU ELEMWISE_ADD` | Saturated INT8 / INT16 elementwise addition |
| `rtv.mul_16x16` | `EXEC_ALU ELEMWISE_MUL` | Elementwise multiplication with configurable fixed-point scale shift |
| `rtv.sumsq_16x16` | `REDUCE REDUCE_SUMSQ` | Tile sum-of-squares reduction to 64-bit scalar for RMSNorm |
| `rtv.rms_scale_16x16` | `SCALAR_ALU RMS_SCALE` | Reciprocal RMS scale applied elementwise to tile |
| `rtv.rowmax_16x16` | `REDUCE REDUCE_MAX` | Row maximum reduction for numerical stability in softmax |
| `rtv.exp_sub_16x16` | `SCALAR_ALU EXP_SUB` | Fixed-point `exp((x - max) / T)` calculation |
| `rtv.rowsum_16x16` | `REDUCE` (sum) | Row summation for softmax probability normalization |
| `rtv.scalar_div_16x16` | `SCALAR_ALU SCALAR_DIV` | Normalizing exponential sums to INT8 probabilities |
| `rtv.silu_16x16` | `EXEC_ALU LUT` | Non-linear activation evaluated via 256-byte calibrated SRAM table |
| `rtv.transpose_16x16` | `EXEC_ALU TRANSPOSE` | In-place square tile transposition in SRAM |

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
* Quantization scale planning and calibration (per-channel weights, dynamic range alignment).

### **What does not change**

The hardware simulator and datapath remain the core deliverable:
`inc/decoder.hpp` and `src/decoder.cpp` (the ISA), `src/pe_array.cpp` and
`src/memory_controller.cpp` (the datapath, 157 tests), and `src/simulator.cpp`.

Neither does the **numerical validation stack**, which validates TVM compiler outputs:
* `ref_model.py` — float golden model, derived from the architecture
* `quant_sim.py` — the design as built
* `scale_sim.py` — the design as proposed, with explicit scale tracking
* `quant_eval.py` — accuracy over 8 prompts, not one
* `device_probe.py` — device-vs-model, per layer and per stage

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

* **Standalone Python Compiler (`toolchain/npu_compiler.py`, ~3000 lines):** Formally superseded and retired. Creating standalone `.py` file(s) representing the compiler is abandoned in favor of the Apache TVM BYOC & TensorIR compiler stack (`toolchain/rtv_tvm/`).
* **JIT Swizzle Engine (`toolchain/swizzle.py`):** Withdrawn. Tensor layout transformation and strided memory access are managed by TVM's typed buffer system.
* **Hand-Written Polyhedral Tiling & List Scheduling:** Withdrawn. Tiling and double-buffering are expressed via TVM TensorIR schedule primitives (`split`, `reorder`, `double_buffer`, `tensorize`).
* **Hardware Abstraction Layer (`toolchain/hal_config.json`):** Retained. Hardware dimensions (PE size, SRAM zone bounds) feed directly into TVM memory scopes and intrinsic declarations.

---

## **4. High-Performance Targets: Workload Evolution**

Rather than executing abstract synthetic workloads, the software toolchain is architected to parse and execute cutting-edge, quantized AI architectures:
* **Computer Vision:** End-to-end execution of a **ResNet-18** convolution pipeline.
* **Edge Large Language Models:** Quantized INT8 deployment of **[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)** (Qwen3ForCausalLM, ~0.6B parameters, hidden\_size=1024, 28 layers, GQA 16→8 heads, head\_dim=128, SiLU/SwiGLU FFN, RMSNorm ε=1e-6, RoPE θ=1,000,000). The pipeline utilizes Apache TVM (with BYOC subgraph partitioning and TensorIR schedules) to orchestrate the model's attention and FFN blocks through the simulated NPU hardware without overflowing local SRAM buffers. RoPE rotation matrices are pre-computed on the host CPU and loaded via LOAD\_TILE.

---

## **5. Roadmap (revised)**

Months 1-3 of the original plan are largely complete: the ISA, decoder,
simulator, PE array and Linux character driver (parts 1-3) exist and are
tested. What follows replaces Month 4, which specified writing fusion and
scheduling passes by hand.

### **Stage A — correctness before migration**

Migrating while a lowering bug is open means debugging an unfamiliar framework
and an unresolved defect simultaneously, and the defect does not disappear —
it relocates into `rtv/codegen.py`.

* Resolve the FP32 divergence at layer 0 of the real model (cosine 0.128
  against `ref_model` while every isolated component — RMSNorm at all
  configurations, RoPE at head_dim 32 and 128, attention at seq_len 1 and 16,
  MLP, linear — tests exact). Hypothesis space is narrowed to many-head GQA at
  Qwen3's dimensions; the method is to shrink the real configuration toward
  the synthetic one (force `num_heads` to 1, then 2, then 8) rather than test
  further components in isolation.
* Land the float model run. If float is correct, every address, tile,
  K-accumulation and schedule in the compiler is proven, and anything still
  wrong in the INT8 path is quantisation rather than lowering.

### **Stage B — TVM BYOC graph partitioning and codegen**

* **B1.** Relax graph pattern matching (`toolchain/rtv_tvm/patterns.py`): Expand `FuseOpsByPattern` to claim the complete set of supported operations (linear GEMM projections, SwiGLU FFN, RMSNorm rows, attention score generation), mapping them to composite functions annotated for the `"rtv_accel"` codegen target.
* **B2.** Runtime module integration: Finalize the whole-program vs per-kernel TVM execution path to feed instruction streams and DRAM images directly into `rtv_sim` and the Linux kernel driver.
* **B3.** Host CPU fallback handling: Route unsupported operations (runtime token embedding lookup gather and argmax sampling) through TVM's standard CPU execution path.

### **Stage C — TensorIR schedule completion & retiring `npu_compiler.py`**

* **C1.** Complete whole-model TensorIR scheduling: Transition all composed blocks (attention multi-head projection, GQA concatenation, SwiGLU intermediate staging) to TensorIR schedules using `tensorize` with the 11 hardware intrinsics (`toolchain/rtv_tvm/intrin.py`).
* **C2.** TVM SRAM staging and double-buffering: Rely on TVM's `cache_read`, `cache_write`, and `double_buffer` primitives for scratchpad allocation, eliminating hand-calculated byte offsets and preventing prefetch/compute hazards.
* **C3.** Retiring standalone `.py` compiler files: Fully decommission and delete `toolchain/npu_compiler.py` (~3000 lines), ensuring that all end-to-end tests, layer probes, and model execution workflows target the TVM compiler pipeline (`toolchain/rtv_tvm/`).
* **C4.** Exploration of schedule space: Leverage TVM's auto-scheduling and search spaces to explore alternate loop orderings and tile configurations.

### **Stage D — INT8 completion**

Independent of the migration; the semantics live in the codegen either way.

* Port the remaining attention interior: q_norm/k_norm gain scales and the
  Q0.7 RoPE tables — the last two instances of the "fixed convention that
  assumes a magnitude" bug class, alongside `INT8_QUANT`'s default shift of 7.
* Outlier handling. `scale_sim` plateaus at 6/8 prompts because of in-row
  dynamic range (layer 2, `down_proj` channel 35 reaches 2610 against a median
  of 0.065); additional bits do not fix this.

### **Deferred**

Driver parts 4-5 (DMA/`mmap`, daemon), ARM NEON verification, and a scalar
control core to remove the recompile-per-token — the last of which is the
single largest structural inefficiency and is tracked in the sibling RTV-Core
project.

```