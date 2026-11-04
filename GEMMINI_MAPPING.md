# RTV-Accel vs Gemmini: ISA mapping (draft, 2026-11-03)

Gemmini (UC Berkeley, `ucb-bar/gemmini`) is the reference design for evolving
this ISA. We borrow concepts, not binary compatibility: Gemmini is a RISC-V
RoCC coprocessor driven by a host core that supplies addresses and loop
bounds; RTV-Accel is fixed-function with no control flow. This table says what
to adopt, adapt or skip.

Sources: the Gemmini README (ISA section), the Buddy Compiler Gemmini dialect
docs, and (verified 2026-11-04) `include/gemmini.h` from
`ucb-bar/gemmini-rocc-tests`.

**Verified from `gemmini.h`:** the loop unroller is triggered by a single RoCC
instruction, `k_LOOP_WS = 8`: rs1 = `(B << 32) | A` (scratchpad base addresses),
rs2 = `(bias << 48) | (K << 32) | (J << 16) | I` (16-bit bounds). The software
fallback `sp_tiled_matmul_ws` issues config + mvin instructions, then per
(j, k, i) a `preload` (funct 6) and `compute_preloaded` (funct 4) or
`compute_accumulated` (funct 5). The header also defines CISC-style config
instructions: `config_addr_ab` (11), `config_addr_cd` (12), `config_size0` (13),
`config_size1` (14), `config_repeating_bias` (15). The `loop_ws_config_*` names
in the Buddy docs do not appear in this header (a different version or
wrapper). The header has **no BF16** (only an `ELEM_T_IS_FLOAT` path). Note the
`gemmini_loop_ws` macro is commented out in this header, with the software loop
used instead, so which version a given Gemmini build uses is not settled by
this file alone.

| Gemmini | RTV-Accel analogue | Verdict |
|---|---|---|
| `mvin` / `mvin2` / `mvin3` | `LOAD_TILE` | **Adapt.** Gemmini's three load channels have independent config so A, B and D loads overlap. Ours assigns `dep_id` by operand role, which is the source of the 1.20x (vs ~2x) double-buffering shortfall. Give loads independent channels/tags instead. |
| `mvout` | `STORE_TILE` | **Adapt.** Gemmini stores are a separate decoupled controller. Ours blocks the clock. Make `STORE` non-blocking with a dep tag. Skip max-pooling-on-store. |
| `config_mvin` / `config_mvout` (stride, scale held in state) | stride/size inside each instruction | **Skip.** Our instructions are self-describing, which is what makes relocation and patching work. Stateful config would break that. |
| `config_ex` (dataflow, ReLU, transpose, scale) | `activation`, `dtype`, `scale_shift` fields per instruction | **Skip.** Same reason. |
| `matmul.preload` + `compute_preloaded` / `compute_accumulated` | `COMPUTE_GEMM` with `accumulate` flag | **Defer.** Weight-stationary preload amortizes one weight tile over many A tiles. It matters for batched prefill, not 1-row decode. |
| Scratchpad + accumulator address space (bit 31 acc, bit 30 accumulate, bit 29 raw) | SRAM zones + `c_slot` + `accumulate` | **Skip.** Equivalent. |
| `loop_ws` (with `loop_ws_config_bounds/addrs/strides`) | none | **Deferred (decision 2026-11-03: compiler keeps scheduling).** Gemmini's unroller schedules internally. We keep the schedule in the compiler: fusion across ops (SwiGLU gate/up resident in SRAM), shape-specific tuning, and a fully visible stream for relocation and byte-identical tests. If stream size later forces it, add a repetition-only loop (strides and bounds, with slots and tags named by the compiler), not an internal scheduler. Profile: 98.4% of a layer's 329k instructions are LOAD/WAIT/GEMM, so the size cost is real but is not what we are fixing first. |
| `loop_conv_ws` | none | **Skip** unless the ResNet-18 target in the spec is revived. |
| ROB hazard detection | explicit `dep_id` / `WAIT_SYNC` | **Keep ours.** Static programs can carry explicit tags. Borrow only the decoupled load/execute/store controller idea. |
| `flush` (TLB), virtual addresses | none | **Skip.** No virtual memory. |
| ReLU / ReLU6 only; `config_norm` (experimental, I-BERT integer GELU/LayerNorm/softmax) | `REDUCE`, `SCALAR_ALU`, `EXEC_ALU` (RMSNorm, softmax, SiLU, RoPE helpers) | **Skip.** Ours is a superset. Gemmini offloads these to the CPU. |
| Datatype fixed at elaboration (int8 default, float via generator params) | runtime `dtype` field | **Keep ours,** and add BF16 ourselves: it is not a Gemmini feature. |

## What this implies for ISA v1.6 (backward compatible, uses reserved bits)

1. **36-bit DRAM address** via `dim_k[15:12]` as high bits on LOAD/STORE (4 GB
   limit blocks real MoE).
2. **3-bit `dtype`** via `dim_m[15]`, adding BF16 weights with FP32 accumulate.
3. **Async `STORE`** plus a deeper dependency model: more than one outstanding tag per operand, so the compiler can prefetch further than one step (the double-buffering fix). Profile: one `WAIT_SYNC` per `LOAD_TILE` today (130,320 each).
4. *(Deferred)* repetition-only loop opcode; see the `loop_ws` row.

Not from Gemmini, still open: MoE routing (host-segmented vs argmax-index +
indirect `LOAD_TILE`), see `TODO.md` Phase 5.1.

## Already found independently of Gemmini

`LOAD_TILE`/`STORE_TILE` are byte-granular (rows, cols in bytes, stride), so
KV append is one-row DMA, not two GEMMs
(`tvm_backend._emit_kv_cache_append_dma_fp32`). The "no hardware
broadcast/permute/partial-write" claims elsewhere in the docs should be
rechecked against this.
