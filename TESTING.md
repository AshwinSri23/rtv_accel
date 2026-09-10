# RTV-Accel Testing Guide

How this project is verified, and why each layer of tests exists. See
`TODO.md` for what's been built and when; this document is about *how the
test suite proves it's correct*, not what's left to do.

## Running everything

```
cmake -B build && cmake --build build
cd build && ctest
```

`ctest` registers two independent layers, described below. The TVM-path
layer only registers itself when `.venv-tvm/` exists (`cmake` prints
`TVM path tests skipped: .venv-tvm not present` otherwise) — a checkout
without that venv still configures and builds cleanly, it just doesn't
get compiler coverage. As of this writing: 3 C++ tests + 16 TVM-path
tests = 19 total, all passing.

Any individual TVM-path file is also directly runnable without `ctest`,
which is faster when iterating on one thing:

```
.venv-tvm/bin/python toolchain/rtv_tvm/test_kv_cache.py
```

---

## Layer 1: C++ unit tests (the hardware/ISA layer)

Three targets, each a small hand-rolled `CHECK`/`CHECK_EQ` harness (no
external test framework — this predates any Python tooling in the repo):

- **`decoder_test`** (`tests/decoder_test.cpp`) — the 128-bit instruction
  word's encode/decode round-trip: every opcode, every field, big-endian
  byte order, malformed-word rejection.
- **`dispatcher_test`** (`tests/dispatcher_test.cpp`) — `InstructionDispatcher`
  routes a decoded instruction to the right per-opcode handler; builds raw
  big-endian instruction buffers by hand and checks dispatch, not compute.
- **`pe_array_test`** (`tests/pe_array_test.cpp`) — the PE array itself:
  GEMM, elementwise ops, reductions, RMS_SCALE, activation modes, directly
  against `MemoryController`-backed SRAM, independent of anything
  instruction-stream-shaped. This is where a hardware-semantics bug (wrong
  accumulate behavior, wrong INT8 rounding, wrong AVX2/NEON path) would
  actually be caught — everything above this layer trusts `pe_array.cpp`
  to be correct and only re-proves *addressing*, not arithmetic.

These three answer "is the simulator a faithful model of the ISA
spec?" — independent of any compiler. If a TVM-path test fails, the first
useful question is whether these three still pass; if they don't, the bug
is in the simulator, not the compiler.

---

## Layer 2: the TVM compiler path (`toolchain/rtv_tvm/`)

### The shared methodology

Every file in this layer follows the same loop, stated plainly in
`test_tvm_path.py`'s own docstring:

```
TIR op -> schedule (tile to 16x16, stage through SRAM, tensorize)
       -> codegen (walk scheduled TIR, emit real 128-bit instruction words)
       -> rtv_sim (the actual C++ simulator, via subprocess, on a real DRAM image)
       -> compare to an independent numpy reference
```

Nothing here is "does this look right" or "does TVM's Python-level result
match" — every test executes the *real simulator binary* on a *real byte
image* and checks the output against numpy computing the same math a
different way. That matters more than it would for an ordinary compiler
test suite: the whole argument for using TVM at all is that it computes
addresses/strides so hand-written Python doesn't have to, so a test that
only checked "the schedule looks correct" would miss exactly the class of
bug (wrong stride, wrong byte offset, off-by-one tile) that motivated the
migration away from `npu_compiler.py` in the first place.

A test failing here means one of: the TVM schedule is wrong, the codegen
mis-lowered scheduled TIR into instruction words, or the numpy reference
itself doesn't match what the hardware primitive actually computes (this
has happened — see `test_gqa_attention.py`'s note on omitting
1/sqrt(head_dim) because the attention-core primitive has no scale step).

### Reading the files in build-up order

The files aren't alphabetical-order dependent, but they *were written* in
roughly this order, each removing one specific limitation of the one
before it. Reading them in this order tells the actual story of how the
attention path got from "one hardcoded tile" to "a real multi-layer
decode loop on real weights":

1. **`test_tvm_path.py`** — the 11 hardware intrinsics (GEMM, elementwise
   add/mul, SiLU, transpose, reductions, RMS_SCALE, etc.) individually,
   INT8 and FP32, bare TensorIR with no BYOC/pattern-matching involved at
   all. The foundation everything else assumes is correct.

2. **`test_composite_dispatch.py`** — one level up: real Relax graphs
   (`R.nn.rms_norm`, a SwiGLU matmul chain, `softmax(Q@K)@V`) get
   pattern-matched by `patterns.py`, classified by
   `tvm_backend.offloaded_composites()`, and lowered by
   `tvm_backend.emit_composite()`. Proves the BYOC plumbing, not just the
   intrinsics.

3. **`e2e_migrated.py`** — the 4 FP32 case classes salvaged from the
   deleted hand-written-compiler test suite (`e2e_test.py`), re-verified
   against the TVM path at their ORIGINAL parameters (not a downgraded
   substitute). Existence proof that this path is at least as capable as
   what it replaced.

4. **`test_gqa_attention.py`** — multi-head attention with grouped query
   attention (GQA), driven by calling `_emit_gqa_attention_fp32` directly
   rather than through pattern matching (there's no DFPattern for "a whole
   attention block, informed by a model config" — see the file's own note
   on why). One real query row, replicated across the 16-row tile.

5. **`test_gemm_double_buffer.py`** — not a new capability, a performance
   regression test: SRAM double-buffering both GEMM operands (not just
   one) produces a measurably lower cycle count via `rtv_sim`'s own cycle
   model, checked as a real number, not just "didn't crash."

6. **`test_attention_multitile.py`** — `seq_k`/`head_dim` spanning more
   than one 16-wide tile, at the attention-core level. Needed no
   `codegen.py` changes — REDUCE's accumulate flag and GEMM's generic M/N/K
   tiling already handled it; only the attention core's own internal
   buffers were hardcoded to one tile.

7. **`test_transformer_layer.py`** — the first time anything chained a
   full pre-norm transformer layer (RMSNorm → attention block with real
   projections → residual → RMSNorm → SwiGLU → residual) and multiple
   layers with ping-ponged DRAM buffers. Self-attention here is still over
   one real token's own replicated K/V — mathematically exact for that
   degenerate case, not yet a real multi-token context.

8. **`test_dram_builder.py`** — `build_model_dram_image`/
   `finalize_dram_image`: turning "a model's raw per-layer weight arrays"
   into a DRAM image stopped being hand-rolled per test file. Checked
   against the *same* numpy reference `test_transformer_layer.py` uses, to
   prove the builder is at least as correct as manual placement.

9. **`test_rope_attention.py`** — RoPE wired into the attention block via
   a `pos=` parameter. Two deliberately different checks: one exercises
   the actual per-head stride arithmetic directly; the other confirms a
   full layer's output is bit-identical with `pos=None` vs `pos=7`, which
   sounds like a null test but is really verifying a structural fact
   (self-attention with one real key can't be affected by a rotation
   applied identically to that key and its own query).

10. **`test_swiglu_resident.py`** — SwiGLU's gate/up GEMM outputs kept
    resident in SRAM across SiLU/multiply instead of round-tripping
    through DRAM between every op. Like double-buffering, checked for both
    correctness and a genuine cycle-count win — and only after confirming
    directly against `pe_array.cpp` that pointing an elementwise op at a
    GEMM's raw output slot is real hardware behavior, not a simulator
    shortcut.

11. **`test_attention_batched.py`** — the first *genuinely* batched
    attention: all 16 tile rows hold different real queries/keys/values,
    not one row replicated. Needed a real `codegen.py` change (a `dim_m`
    instance attribute, previously a hardcoded constant) plus a new
    per-row softmax driver, because the existing schedule primitives
    can't express a 1-row axis at all.

12. **`test_kv_cache.py`** — a KV-cache that actually persists and GROWS
    across calls (not just "many rows in one call"), via the "constant
    matrix through GEMM" trick used everywhere in this codebase where no
    hardware broadcast/permute/partial-write primitive exists: inserting
    one new row into an existing 16-row tile is `mask_keep @ old_tile +
    mask_insert @ new_row`, two compile-time one-hot/identity-minus-one-hot
    matrices.

13. **`test_attention_wide_headdim.py`** — generalizing the whole attention
    block past `head_dim == compiler.tile_dim`, up to and including
    Qwen3-0.6B's real `head_dim=128`. Three real bugs found and fixed here
    (RoPE's permutation matrix sizing, a hardcoded copy width, a new
    `_emit_transpose_wide_fp32` helper for per-head-column transposes) —
    all shape/stride mistakes that head_dim==tile_dim had been silently
    hiding, not conceptual gaps.

14. **`test_decode_loop.py`** — the first test that is qualitatively
    different from everything before it: a genuine multi-step,
    multi-layer autoregressive decode loop. Because the ISA has no control
    flow, there is no single compiled program for "generate N tokens" —
    each step is its own fresh instruction stream, and the DRAM image
    itself is the only thing connecting step *t* to step *t+1*. This test
    round-trips the WHOLE DRAM extent through separate `rtv_sim`
    invocations (`--dram-image` in, `--dump` the whole thing out, feed
    that back in as the next step's input) exactly the way a real driver
    has to. One case also exercises Qwen3's q_norm/k_norm with random
    weights, independent of any checkpoint download.

15. **`test_checkpoint_loader.py`** — swaps random weights for a REAL,
    locally-downloaded Qwen3-0.6B checkpoint (`hf download Qwen/Qwen3-0.6B
    --include "*.json" --include "*.safetensors"`, ~1.5 GB), run at
    Qwen3's actual full dimensions (hidden=1024, 16 heads over 8 KV heads,
    head_dim=128). Checked against a numpy reference built from the
    IDENTICAL loaded tensors — this is a check that the loader's
    HF-layout transpose and the compiler agree with each other on real
    values, not a claim of matching Qwen3's own published outputs (that
    would additionally require a real multi-token context, which this
    file's reference deliberately doesn't build — see #16).

16. **`test_checkpoint_decode_loop.py`** — combines #14 and #15: real
    checkpoint weights, driven through several genuinely separate decode
    steps with a real growing multi-token KV-cache (every query head
    attends over every real cached key so far, not "one key"). This is
    the closest thing in the suite to "does the actual model, run for
    several real tokens, compute what an independent implementation says
    it should" — everything needed for it already existed in #14 and #15;
    this file only needed to compose them.

### The checkpoint-dependent tests skip, they don't fail, without a download

`test_checkpoint_loader.py` and `test_checkpoint_decode_loop.py` both
search `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*` for
a local checkpoint and print a `SKIP` line + exit 0 if nothing is there —
mirroring `CMakeLists.txt`'s own top-level skip when `.venv-tvm/` is
absent. Neither is a required dependency for the rest of the suite to mean
anything; they're the two tests that specifically need real weights on
disk to say anything about real-world fidelity.

### What's real-checked vs. deliberately not

A few limitations are documented, not accidental, and worth knowing before
reading a "PASS" as more than it claims:

- **Every attention test attends over at most one real cached key per new
  token except #16**, which genuinely grows the context — but even there,
  the model itself is only ever run 1-2 layers deep in the checkpoint
  tests (real dimensions are slow to schedule/codegen; see below).
- **Genuinely batched (prefill-style) multi-row processing** exists at the
  attention-core level (`test_attention_batched.py`) but was never wired
  into the decode loop — decode is inherently single-token, so it was
  never on that loop's critical path. `toolchain/generate.py` prefills a
  prompt one token at a time for this reason, not because batching is
  unimplemented everywhere.
- **`toolchain/generate.py` itself is not part of this test suite.** It's
  a demo/driver script with no numpy oracle of its own to check against —
  its correctness rests entirely on the fact that it calls the exact same
  `_emit_model_kvcache_fp32` function #16 already verifies, with no
  independent logic of its own in between except host-side embedding
  lookup, final RMSNorm, `lm_head`, and argmax (all standard, un-tested-here
  numpy).

### A note on runtime

Real dimensions (hidden=1024, head_dim=128) are slow to schedule/codegen
through TVM — not to simulate; `rtv_sim` itself stays well under a second
per decode step even at full size. Expect `test_checkpoint_loader.py`/
`test_checkpoint_decode_loop.py` to take noticeably longer than anything
else in the suite (tens of seconds, not the ~1-5s of most other files),
and a full 28-layer `generate.py` run to take minutes per generated token
— this is architectural (each decode step really is a fresh,
multi-million-instruction program with no code reuse across steps, per
`codegen.py`'s own docstring), not a test-suite inefficiency.
