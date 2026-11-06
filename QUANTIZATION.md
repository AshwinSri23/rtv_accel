# Weight-only quantization (INT8W ISA v1.8, INT4W ISA v1.9)

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
- Compiler (`_emit_matmul_int8w_fp32`): the usual prefetched K-loop. The scale row is broadcast to 16 rows by DMA once
  per matmul; then, per output tile, a 16x16 scale tile is prefetched into a spare SRAM output slot under the last K
  step and multiplied into C in place (`EXEC_ALU ELEMWISE_MUL`) before the store. Scale is constant along K, so
  applying it once after accumulation is exact, and doing it in SRAM avoids a DRAM round trip of C.
- Relocation: the role byte-ranges include the scale tail, so relocated layers stay byte-identical to a full recompile.

## 2b. INT4 (ISA v1.9)

Plain round-to-nearest INT4 is unusable (+13 to +36% perplexity), so the INT4 path uses **GPTQ**: each weight column is
rounded in turn and its rounding error is pushed onto the not-yet-rounded columns through the inverse Hessian of the
layer's calibration inputs (`quant_study.py gptq`, Hessians from 32x512 WikiText-2 *train* tokens; evaluated on the
*test* split):

| scheme (symmetric unless noted) | bits/weight | RTN ppl | GPTQ ppl | GPTQ vs FP32 (29.39) |
|---|---|---|---|---|
| INT4 g128 | 4.25 | 40.11 | 34.09 | +16% |
| INT4 g64 | 4.5 | n/a | 32.97 | +12% |
| **INT4 g32** | **5.0** | 34.88 | **30.77** | **+4.7%** |
| INT4 g64 asymmetric | 4.56 | n/a | 31.01 | +5.5% |
| INT4 g32 asymmetric | 5.1 | 33.25 | 30.83 | +4.9% |

Decision: **symmetric INT4, group 32** (no zero point; asymmetric buys nothing at g32). Group scales vary along K, so they
cannot be applied after accumulation the way INT8's per-channel scale is: `COMPUTE_GEMM` with `DType.INT4W` takes the
SRAM address of the 16-float scale row in its `dram_addr` field and computes `C += A . (q * scale)`; one scale row is
loaded per K-group per output tile and shared by `group/16` K-tile GEMMs. B is packed two nibbles per byte and loaded
with ordinary byte-granular LOADs (16 rows x 8 bytes). Offline: `quant_gptq.py` writes the codes and scales
(`generate.py --int4-weights file.npz`). Not committed: the 495 MB export.

Measured: weights **275 MB** for the 28 layers (FP32 1761.6; INT8 441.8), 6.4x smaller. Chip vs Hugging Face with the same
dequantised INT4 weights (`hf_compare.py --ref-int4`): identical tokens on both prompts (the 26-token one through
`--graph`). INT4 output differs from FP32 ("...Rome is the capital of Italy. The capital" vs "...Italy. Which of"),
as expected at +4.7% perplexity. Cycles (same 16x1024 @ 1024x3072 matmul, double-buffered): **3,181,252**, equal to BF16
and 1.5% under FP32; it issues 20% more instructions (scale-row loads) but moves the fewest bytes.

## 3. Measured

- Weight bytes, full Qwen3-0.6B (28 layers): FP32 1761.6 MB, BF16 880.8 MB, **INT8 441.8 MB** (3.99x).
- Chip vs Hugging Face with the **same** INT8 weights fake-quantized (`hf_compare.py --ref-int8`): identical tokens for
  "The capital of France is" and for the 26-token two-tile prompt (the latter through `--graph`).
  Chip vs *unquantized* HF is not expected to be identical (see perplexity above).
- Cycles, one 16x1024 @ 1024x3072 matmul (double-buffered model): FP32 3,228,672, BF16 3,181,272, **INT8 3,247,032**
  (3,328,448 before the dequant moved into SRAM). INT8 is about 0.6% *slower* than FP32 in the simulator: the model
  is compute-bound, so the 4x smaller weights buy footprint, not speed. On bandwidth-bound hardware the
  smaller weights would help; the simulator cannot show that.

## 3b. MoE experts

Experts take the same path: `_emit_swiglu_ffn_fp32` already goes through the weight-matmul dispatch, so
`compiler.int8_weights` covers host-segmented routing unchanged. On-device top-1 routing needed one addition: the
INDIRECT_ADDR offset applies to the INT8 weight tiles *and* to the scale-row load (`_emit_broadcast_row0_fp32(...,
indirect_ptr=)`), so an expert's blob (weights then scales) must sit at the fixed routing stride. The router, norms,
embeddings and lm_head stay FP32. Tests: `test_moe_int8` (both routing modes vs numpy on dequantised weights, one
stream picking all 4 experts on-device; expert 9.44 MB -> 2.37 MB) and `test_moe_model` (removed 2026-11-05 with the Qwen3-MoE driver; whole tiny Qwen3-MoE on INT8
attention + experts vs an INT8-fake-quantized Hugging Face reference, logits within 2e-7; the INT8 and FP32 references
differ by 3e-3, so the INT8 path is genuinely exercised). Not measured: accuracy of INT8 on a *real* trained MoE.

BF16 and INT4 experts work the same way, in both routing modes. The indirect (on-device routed) path now covers
`_emit_matmul_bf16w_fp32` and `_emit_matmul_int4w_fp32` through `indirect_b=`; for INT4 the offset applies to the packed
weight loads and to every scale-row load, so an expert's blob (packed nibbles, then scale rows) sits at the routing
stride. Tests: `test_moe_quant` (4 cases, errors ~1e-7 vs numpy on dequantised weights; one stream picks all 4 experts)
and `test_moe_model`'s INT4 case (removed 2026-11-05; tiny Qwen3-MoE, INT4 g16 round-to-nearest attention + experts vs
`make_moe_int4_fixture.py`'s Hugging Face reference, top-5 logits within 3.6e-7, tokens differ from FP32 and INT8).

## 3c. Perplexity on the chip

`chip_ppl.py` scores WikiText-2 test tokens through the chip's graph-driven prefill (every 16-row tile's final hidden
rows via `generate(..., prefill_hook=)`, then the trailing norm and lm_head on the host) and through Hugging Face with the
same dequantised weights. INT4 g32 GPTQ, first 64 tokens: chip ppl 45.5234, PyTorch 45.5214, max per-token |dNLL|
1.1e-4. So the chip computes the model the study measured. The absolute number is for 63 tokens and is not comparable to
the 2048-token study figures (29.39 FP32 / 30.77 INT4); a 64-token run takes ~7 minutes on the simulator, so the full
2048 tokens was not run (after the compile-time work a 64-token run takes under a minute, so a longer run is now practical).

Same 63 tokens, BF16 weights: chip ppl 49.8571 vs PyTorch 49.8549 (max |dNLL| 1.0e-4). INT8 weights (per-channel): chip 49.2227 vs PyTorch 49.2205 (max |dNLL| 1.05e-4). (FP32 reference on these tokens is not shown here; compare formats against each other only through the 2048-token study.)

## 3d. Embedding / lm_head table (`quant_embed_study.py`)

The tied table (151936 x 1024, 622 MB FP32) is the biggest tensor in Qwen3-0.6B and was left FP32. Fake-quantized per
vocab row (INT4: groups along the hidden dim), WikiText-2 test, first 2048 tokens, layers FP32:

| table | MB | ppl, shared by lookup + lm_head | ppl, lm_head only |
|---|---|---|---|
| FP32 | 622 | 29.39 | 29.39 |
| BF16 | 311 | 29.39 | 29.39 |
| **INT8 per row** | **156** | **29.38** | 29.38 |
| INT4 g128 RTN | 83 | 31.72 (+7.9%) | 31.63 |
| INT4 g32 RTN | 97 | 30.74 (+4.6%) | 30.38 (+3.4%) |

On top of INT8 layers (29.69): INT8 table 29.68, INT4 g32 table 31.08 (the errors add roughly, nothing compounds).
Conclusion: **INT8 per-row costs nothing measurable and cuts the table 4x**; INT4 g32 RTN costs 3-5% and would want
GPTQ-style calibration (not tried; the lm_head Hessian is the final-norm output, cheap to collect). Greedy-token
agreement drops for INT8 (77/98) because near-ties flip; perplexity is the metric that decides, as in section 1.
Caveat: today the embedding lookup and lm_head run on the host (the chip returns the final hidden rows), so this only
shrinks host memory. If lm_head moves on-device the INT8W GEMM already supports it (K=1024, N=151936, per-channel
scale) with no new ISA.

## 3f. Other GGUF quantisation types on the same hardware (2026-11-06)

The chip multiplies a signed 4-bit value by one FP32 scale per (group of inputs, output column). Q4_K, Q8_0 and Q6_K are
of the form `w = scale * (integer + offset)`, which is a sum of such 4-bit planes plus a constant per group
(`toolchain/quant_planes.py`):

| GGUF type | form | planes (INT4W) | correction per group |
|---|---|---|---|
| Q4_K | `s*q - m`, q 0..15, g32 | 1: (q-8, s) | `8s - m` |
| Q8_0 | `s*q`, q -128..127, g32 | 2: (q>>4, 16s), (q&15 - 8, s) | `8s` |
| Q6_K | `s*(q-32)`, q 0..63, g16 | 2: (q>>4 - 2, 16s), (q&15 - 8, s) | `8s` |

The correction is `y[n] += sum_g xsum_g * c_g[n]` with `xsum = x @ G` (G the 0/1 group-membership matrix, itself an
INT4W blob) and `c` an FP32 `[groups, N]` matrix: two small extra GEMMs. Everything is existing instructions;
`_emit_weight_matmul_fp32` dispatches on a registry (`compiler.qweights`), relocation roles cover every plane and the
correction, and layers whose matrices differ in scheme get their own program templates. Cost per matrix relative to
Q4_0: Q4_K about +25% weight traffic (the correction matrix) and an extra GEMM chain; Q8_0 and Q6_K two planes, so about
the traffic of INT8 (they are 8 and 6.5 bits per weight to begin with). Verified: exact to 3e-7 against the dequantised
matrix on the chip for 1-row and full-tile shapes (`test_qweights`), block parsers equal a per-element loop written
from the llama.cpp layouts and bit-identical to the `gguf` package, and the whole tiny Gemma-4 with every matrix in one
scheme or a mix matches the same model with dequantised FP32 weights (`test_gemma4_quant_types`). Not done: on-device
(indirect) MoE routing with planar weights, Q5_K / Q5_0 / Q4_1 (same method, not needed), and a run on a real K-quant
GGUF (none is available locally).

## 4. Not done

INT4 on a real trained MoE (only the tiny random one is tested); chip perplexity beyond 64 tokens; AWQ/outlier handling and group sizes other than a multiple of 16. Tracked in `TODO.md` Phase 2b.

### 3e. Gemma-4 embedding / lm_head table as INT8 per row (2026-11-05)

`load_gemma4_gguf(embed_int8=True)` keeps the 262144 x 2816 tied table as INT8 with one FP32 scale per row (0.74 GB
instead of 2.95 GB; host-side only). On the real 26B model, 47 wikitext tokens: perplexity 994.15 (FP32 table) ->
1010.13 (+1.6%), largest per-token log-prob change 0.68 nats. Unlike Qwen3-0.6B (section 3d, free), Gemma pays for it;
off by default. Group-scaled INT8 (a scale per 32 inputs) is the obvious next try and is untested.
