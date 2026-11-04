# Weight-only quantization (INT8W, ISA v1.8)

Scope: the 7 per-layer weight matrices (Q/K/V/O, SwiGLU gate/up/down). Activations, norms, embeddings, lm_head and
the KV-cache stay FP32; accumulation is FP32. Because activations are FP32 there is no activation-outlier problem
(the reason the old Phase 2 INT8 plan was dropped).

## 1. Which scheme (`toolchain/quant_study.py`, Qwen3-0.6B, WikiText-2 test, first 2048 tokens)

| scheme | bits/weight | perplexity | vs FP32 |
|---|---|---|---|
| FP32 | 32 | 29.39 | |
| BF16 | 16 | 29.39 | 0 |
| INT8 per output channel | 8 | 29.69 | +1.0% |
| INT8 group 32 / 128 | 9 / 8.25 | 29.32 / 29.32 | about 0 |
| INT4 group 128 (sym) | 4.25 | 40.11 | +36% |
| INT4 group 32 (sym) | 5.0 | 34.88 | +19% |
| INT4 group 32 (asym) | 5.1 | 33.25 | +13% |
| INT4 group 16 (asym) | 6.25 | 32.04 | +9% |

Greedy-token agreement is a noisy metric (one near-tie flips everything after it), so perplexity decides.
Decision: **per-output-channel INT8**. Group scales add scale traffic and per-K-group handling for no measurable
gain at 8 bits. Round-to-nearest INT4 is not usable on a 0.6B model; it needs calibrated rounding (GPTQ/AWQ).
Attention-only INT8 is nearly free (+0.36%); the MLP carries most of the loss.

## 2. Design

- Per matrix W[K, N]: `q = round(W / s)`, `s[n] = max|W[:, n]| / 127`, stored as K*N int8 bytes then N FP32 scales.
- `COMPUTE_GEMM` with `DType.INT8W`: FP32 A x raw INT8 B -> FP32 (simulator `pe_array.cpp`). No scale in the instruction.
- Compiler (`_emit_matmul_int8w_fp32`): the usual prefetched K-loop, then `C *= scale` as an elementwise multiply (the
  scale row is broadcast to 16 rows by DMA). Scale is constant along K, so applying it once after accumulation is exact.
- Relocation: the role byte-ranges include the scale tail, so relocated layers stay byte-identical to a full recompile.

## 3. Measured

- Weight bytes, full Qwen3-0.6B (28 layers): FP32 1761.6 MB, BF16 880.8 MB, **INT8 441.8 MB** (3.99x).
- Chip vs Hugging Face with the **same** INT8 weights fake-quantized (`hf_compare.py --ref-int8`): identical tokens for
  "The capital of France is" and for the 26-token two-tile prompt (the latter through `--graph`).
  Chip vs *unquantized* HF is not expected to be identical (see perplexity above).
- Cycles, one 16x1024 @ 1024x3072 matmul (double-buffered model): FP32 3,228,672, BF16 3,181,272, **INT8 3,328,448**.
  INT8 is about 3% *slower* than FP32 in the simulator: the model is compute-bound, so the 4x smaller weights buy
  footprint, not speed, and the dequant multiply (DRAM round trip) costs a little. On bandwidth-bound hardware the
  smaller weights would help; the simulator cannot show that.

## 3b. MoE experts

Experts take the same path: `_emit_swiglu_ffn_fp32` already goes through the weight-matmul dispatch, so
`compiler.int8_weights` covers host-segmented routing unchanged. On-device top-1 routing needed one addition: the
INDIRECT_ADDR offset applies to the INT8 weight tiles *and* to the scale-row load (`_emit_broadcast_row0_fp32(...,
indirect_ptr=)`), so an expert's blob (weights then scales) must sit at the fixed routing stride. The router, norms,
embeddings and lm_head stay FP32. Tests: `test_moe_int8` (both routing modes vs numpy on dequantised weights, one
stream picking all 4 experts on-device; expert 9.44 MB -> 2.37 MB) and `test_moe_model` (whole tiny Qwen3-MoE on INT8
attention + experts vs an INT8-fake-quantized Hugging Face reference, logits within 2e-7; the INT8 and FP32 references
differ by 3e-3, so the INT8 path is genuinely exercised). Not measured: accuracy of INT8 on a *real* trained MoE.

## 4. Not done

INT4 / group-wise (needs calibration); BF16 with on-device routed loads; applying the scale in SRAM
before the store to remove the round trip. Tracked in `TODO.md` Phase 2b.
