# RTV-Accel: Project Overview (for new contributors)

This document explains what this project *is*, in plain language, and
what each part of it does. It's meant to be read once, before diving into
`TODO.md` (the roadmap/history) or `TESTING.md` (how the test suite
works) or the spec doc (the exact hardware details).

## The one-sentence version

This project is two things bolted together: a **simulated chip** (a
pretend NPU — a "neural processing unit," a chip specialized for the
matrix math neural networks need) and a **compiler** that takes a real,
pretrained language model and translates it into the raw instructions
that pretend chip understands — so the whole pipeline can be run and
checked, even though the physical chip doesn't exist. Two real models
have been taken through it: **Qwen3-0.6B**, a small dense model that
the project started with, and **Gemma-4-26B-A4B**, a 26-billion-parameter
mixture-of-experts model (128 experts per layer, 8 used per token) that
is now the main target. `README.md` has the short version of how each
runs, how to run them, and what has and has not been verified. This
document is the longer tour; its earlier parts were written around
Qwen3 and describe the dense path, which the Gemma path reuses.

Think of it like building a video game console emulator (the simulator)
and a game compiler (the toolchain) for a console that was designed on
paper but never manufactured. You can't buy the chip, but you can prove
the design works by running real software on a faithful software model of
it.

## The big picture

```
 A real, pretrained language model            The chip's own instruction
 (Qwen3-0.6B's downloaded weight file)         set — 128-bit fixed-width
              │                                words, ~8 opcodes total
              ▼                                          │
    ┌───────────────────┐                                │
    │  checkpoint_loader │  reads real weights off disk   │
    │  + hf_loader        │  into plain numpy arrays       │
    └─────────┬───────────┘                               │
              ▼                                            │
    ┌────────────────────┐        translates the           │
    │   THE COMPILER      │───────  model's math into ─────►│
    │  (tvm_backend.py +  │        a stream of those
    │   toolchain/rtv_tvm)│        raw instructions
    └─────────┬────────────┘
              │  writes out: an instruction stream (a .bin file)
              │              + a DRAM image (the chip's starting memory)
              ▼
    ┌────────────────────┐
    │   THE SIMULATOR      │   executes those exact instructions,
    │   (rtv_sim, C++)     │   byte for byte, as if it were real silicon
    └─────────┬────────────┘
              ▼
        the model's actual output
     (read back out of the DRAM image)
```

Everything in this repo exists to make that pipeline real, correct, and
provable — not just "looks plausible," but "produces the exact same
numbers as an independent calculation done a totally different way,"
checked automatically by the test suite (see `TESTING.md`).

## Part 1: the simulated chip (`inc/`, `src/`)

This is a from-scratch C++ program (`rtv_sim`) that behaves like a real
chip would, at the level of individual instructions and memory bytes.
Four pieces:

- **`decoder.hpp`/`decoder.cpp`** — defines the instruction format: every
  instruction is exactly 128 bits (16 bytes), with fixed fields for
  "which operation," "where in memory," "how big a chunk of data." This
  is the chip's actual machine language — nothing fancier than that
  fixed layout exists.
- **`instruction_dispatcher.hpp`/`.cpp`** — reads a file full of those
  128-bit instructions and, for each one, calls the right handler
  function for its opcode. The "traffic cop" that turns a flat binary
  file into a sequence of actions.
- **`memory_controller.hpp`/`.cpp`** — models the chip's two kinds of
  memory: a large, slow **DRAM** (like a computer's main RAM) and a
  small, fast 64 KiB **SRAM** scratchpad (like a CPU's cache, but
  explicitly managed by software instead of automatic). Every access
  costs simulated cycles, so this also produces a rough performance
  estimate, not just a correctness check.
- **`pe_array.hpp`/`.cpp`** — the actual math engine: a 16×16 grid of
  "processing elements" (PEs) that do the multiply-and-add work a matrix
  multiplication needs, the way a real NPU's core would. This is where
  numbers actually get computed; everything else just moves data around
  it.
- **`simulator.cpp`** — the `main()` that wires all of the above
  together into the `rtv_sim` executable: load an instruction file and a
  starting memory image, run every instruction in order, dump out
  whatever memory the caller asked for at the end.

**Why this matters:** nothing about this layer knows or cares that it's
running a language model. It only knows about instructions, bytes, and
cycles — the same way a real CPU doesn't know it's running a web browser
versus a game. This separation is what makes the correctness argument
possible: if the compiler and the simulator agree independently, that's
real evidence, not circular reasoning.

## Part 2: the compiler (`toolchain/tvm_backend.py`, `toolchain/rtv_tvm/`)

This is the translator: it takes "compute a transformer layer" (or "run
one decode step of a whole model") and produces the actual stream of
128-bit instructions from Part 1 that does that computation on the
simulated chip.

It's built on **Apache TVM**, an existing, real open-source compiler
framework for machine learning models — this project didn't write a
graph optimizer or a scheduler from scratch. TVM provides three things
this project plugs its own hardware knowledge into:

- **`patterns.py`** — tells TVM "this shape of computation graph (e.g. a
  matmul followed by RMS normalization) should be handed to *our* chip,
  not run generically."
- **`schedule.py`** — makes the hardware-specific decisions TVM can't
  make on its own: split every computation into 16×16 tiles (the PE
  array's native size), stage data through SRAM before computing, and so
  on. This is where "how do I use *this specific chip's* strengths"
  lives.
- **`codegen.py`** — walks the scheduled computation and emits the actual
  128-bit instruction words — the last step before it's real bytes on
  disk.
- **`intrin.py`** — the glue that tells TVM "this exact small pattern of
  math corresponds to this exact hardware instruction" (a "tensor
  intrinsic").

**`tvm_backend.py`** is the top-level file that composes all of the
above into things people actually call: "emit one whole transformer
layer," "emit one whole model's decode step," etc. If you want to
understand what the compiler can currently do, this is the file to read
first.

**`compiler_state.py`** holds the bookkeeping the compiler needs that
isn't really "compiling" per se: where in DRAM things get placed, what
the chip's configuration is (tile size, SRAM zones), and small
host-side math (like RoPE's rotation tables) that's cheaper to compute
once on the CPU than to re-derive on-device every time.

**Why it's built this way:** the chip's instruction set has **no control
flow at all** — no loops, no branches, no indirect memory addressing.
Every single instruction's address and size has to be known before the
program runs. This is the single fact that explains most of this
project's odd-looking design decisions (see the callout below).

## Part 3: getting a real model onto the chip (`hf_loader.py`, `checkpoint_loader.py`)

A real language model ships as a folder of files (a "checkpoint"): a
config describing its shape, and a large binary file of the actual
learned numbers ("weights"), usually in Hugging Face's `.safetensors`
format.

- **`hf_loader.py`** reads that binary format directly (no PyTorch
  required — PyTorch alone is a ~2 GB dependency just to read a file,
  and this project only ever needs plain arrays of numbers).
- **`checkpoint_loader.py`** knows specifically about Qwen3's
  architecture: it pulls out each layer's weights and rearranges them
  into exactly the shape and orientation the compiler expects (Hugging
  Face and this project's compiler disagree on one convention — whether
  a weight matrix is stored "transposed" or not — so this is where that
  gets fixed, once, on load).

Together these turn "a ~1.5 GB folder downloaded from the internet" into
"plain numpy arrays the compiler can hand straight to `tvm_backend.py`."

## Part 4: the demo (`toolchain/generate.py`)

This is the part a non-engineer could actually watch work: type in a
sentence, get more sentence back. It's a small Python script that ties
everything above together into an actual "prompt in, tokens out" loop.

Because the chip has no control flow (see below), a few small things
*can't* run on it and have to happen on the regular host computer
instead:

- **Turning words into numbers** (tokenizing the prompt, and looking up
  each word's numeric "embedding" — this is just picking one row out of
  a giant table, which needs a runtime-computed row number the chip
  can't do).
- **Turning the model's final numbers back into a word** (picking the
  single most likely next word is an "argmax" — find the *index* of the
  biggest number — which this chip's math primitives don't produce; they
  only produce the *value* of the biggest number, not where it was).

Everything in between — every actual transformer layer, all the real
"thinking" — runs on the simulated chip, unmodified from what the test
suite already proves is correct.

## The one fact that explains almost everything else

**The chip's instruction set has no loops and no branches, and (apart
from one narrow exception added later, see below) no computed
addresses.** Every instruction's exact memory address has to be baked
in at compile time. The exception is ISA v1.7's `INDIRECT_ADDR`, which
adds one SRAM-held offset to a load or store address (used for on-device
top-1 expert routing); it reaches at most about 8 GiB and does not
change the rest of this section. This sounds like a strange
limitation, but it's realistic for a certain style of real accelerator
(some real NPUs work exactly this way, trading flexibility for
predictability and speed) — and it's the reason for almost every
unusual-looking piece of this codebase:

- **A whole new instruction program has to be generated for every single
  generated word** ("decode step"), because a real generation loop's
  length isn't known ahead of time and a loop can't be expressed in the
  instruction set at all. There's no such thing as "the program that
  generates N words" — there's only "the program for word 1," "the
  program for word 2," etc., each one freshly compiled.
- **The only thing connecting one word's program to the next word's
  program is the chip's memory itself** (the "KV-cache," where the model
  remembers earlier words) — so a real multi-step run works by dumping
  the entire simulated chip memory out to a file after each word, and
  handing that whole file back in as the starting point for the next
  word's program.
- **Picking which row of a table to read (an embedding lookup) or which
  index has the biggest value (sampling) can't happen on-device at all**
  — both need a number that's only known *while the program is running*,
  which this instruction set has no way to express. Both are done on the
  ordinary host computer instead (see Part 4).

If something in this codebase looks over-engineered or unusually
indirect, it's very likely a direct consequence of this one constraint,
not accidental complexity.

## Historical / retired pieces (not on the current path)

The project originally targeted INT8 (8-bit integer) arithmetic with
manually-tracked numeric scale factors, before pivoting to plain FP32
(32-bit float, the same precision the original model itself uses) —
simpler, and floats don't need any of that scale bookkeeping.

A cluster of INT8-era files survived that pivot as dead code — nothing
imported them except each other — and have now been **deleted**:
`scale_sim.py`, `quant_sim.py`, `quant_eval.py`, `ref_model.py`,
`layer_probe.py`, `device_probe.py` and `swizzle.py` in `toolchain/`,
plus `find_ops.py` at the repository root (a scratch script that
inspected SmolLM2, a model this project never targeted). Together they
were ~1,870 lines, about 10% of the repository. The hand-written Python
compiler `npu_compiler.py` (~3,400 lines) was deleted earlier, when the
TVM path replaced it — see `TODO.md`'s Phase 1 for that story.

None of this is needed to work on the current pipeline. It's recorded
here only so the names remain findable: all of it is recoverable from
git history if the INT8 work is ever revisited.

## ISA coverage: one instruction set for the dense and the MoE model

The hardware is fixed and the compiler adapts: Qwen3 (dense, SwiGLU) and Gemma-4-26B-A4B (dense GELU MLP in parallel
with 128 experts, sandwich norms, sliding-window attention) run on the **same ISA**. Moving from Qwen3 to Gemma-4
added no instruction and no hardware mode; what was new was composed from existing operations, done in the compiler,
or done on the host.

| Model feature | Implemented by | Qwen3 (dense) | Gemma-4 MoE |
|---|---|---|---|
| Linear layers | `COMPUTE_GEMM`, FP32 activations x FP32 / BF16W / INT8W / INT4W weights | yes | yes (INT4W g32, straight from Q4_0) |
| RMSNorm | `REDUCE_SUMSQ`, `SCALAR_ALU` RMS_SCALE, `ELEMWISE_MUL` by the gain | 2 per layer + q/k norms | up to 7 per layer + q/k norms + a weightless V norm |
| Softmax | `REDUCE_MAX`, `SCALAR_ALU` EXP_SUB, `REDUCE`, `SCALAR_DIV` | yes | yes |
| SwiGLU | SiLU (`EXEC_ALU` SILU / LUT) x `ELEMWISE_MUL` | yes | not used |
| GELU(tanh) | **composed**: mul, add, RELU, exp (`EXP_SUB`), Newton reciprocal, ~310 instructions per tile column | no | yes (dense MLP and every expert) |
| RoPE | rotation tables (compile-time constants) + `ELEMWISE_MUL`/`ADD` | full | full (sliding layers) or partial "proportional" (full layers) |
| Causal / tail mask | additive mask tile (`ELEMWISE_ADD`) | yes | yes |
| Sliding window | **compiler only**: skip old KV tiles + a mask tile | no | yes (25 of 30 layers, 1024 keys) |
| KV cache | `LOAD_TILE` / `STORE_TILE` append, contiguous V block | yes | yes (K=V on full layers) |
| Addresses above 4 GiB | 36-bit `LOAD_TILE`/`STORE_TILE` (ISA v1.6) | not needed | needed (16.5 GB image) |
| Routing, top-k of 128 | **host**, between two device programs per layer (softmax, top-k, renormalise) | no | yes |
| Routing, top-1 | `REDUCE_ARGMAX` + `INDIRECT_ADDR` loads (ISA v1.7), no host trip | no | not used (Gemma needs top-8) |
| Router logits, layer scalar, route weights | GEMM / constant tiles | -- | yes |
| Embedding lookup, lm_head, softcap, sampling | host (FP32) | yes | yes |

### What one decode layer costs (instructions)

Measured by `toolchain/isa_coverage.py` on the real-sized per-layer programs (position 20, INT4W g32 weights):

| Program | Instructions | `COMPUTE_GEMM` (INT4W) | `LOAD_TILE` | `EXEC_ALU` | `SCALAR_ALU` | `REDUCE` |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B, one dense layer | 384,448 | 61,440 | 158,491 | 1,664 | 384 | 410 |
| Gemma-4 sliding layer: A (attention + dense MLP + router) | 1,312,570 | 204,864 | 539,369 | 10,096 | 1,864 | 1,640 |
| Gemma-4 full layer: A (head_dim 512, 2 KV heads) | 1,658,126 | 261,184 | 682,757 | 10,736 | 1,992 | 1,756 |
| Gemma-4: B (8 chosen experts + norms + residual), any layer | 1,252,154 | 185,856 | 509,022 | 21,824 | 1,056 | 370 |

A sliding Gemma layer is A + B = 2.56 M instructions and a full layer 2.91 M, so one token through 25 sliding + 5 full
layers is about **79 M instructions** (Qwen3-0.6B: 28 x 0.38 M = 10.8 M). Every `LOAD_TILE` is paired with a
`WAIT_SYNC`, and `LOAD_TILE` + `WAIT_SYNC` + `COMPUTE_GEMM` make up about 98% of every program (that includes
activation tile loads as well as weight tiles): the work is data-movement- and GEMM-shaped, not ALU-shaped. See the
traffic table below for how little of that movement is weights. These are instruction counts, not cycles or seconds.

### DRAM traffic per decode layer (bytes moved by `LOAD_TILE` / `STORE_TILE`)

Decoded from the compiled programs (`isa_coverage.py` for Qwen3, the same decoder on a Gemma run for Gemma). Weights =
loads from the weight region; "other" = activations, KV cache, constants. Decode has one real row, so GEMMs load a
**single activation row** (64 B, `compiler.gemm_a_rows = 1`, rows 1-15 of the SRAM slot kept zero) instead of a 16-row
tile (1 KB, 15/16 zeros) per output tile. Before -> after:

| Program | Weight loads | Other loads | Stores |
|---|---:|---:|---:|
| Qwen3-0.6B dense layer | 9.8 MB | 68.1 -> **9.1 MB** | 3.4 MB |
| Gemma-4 sliding layer, A | 32.8 MB | 238 -> **41.4 MB** | 14.4 MB |
| Gemma-4 full layer, A | 41.8 MB | 299 -> **47.8 MB** | 15.9 MB |
| Gemma-4, B (8 experts) | 29.7 MB | 236 -> **57.8 MB** | 26.0 MB |

Per token: Gemma-4 **1.92 GB of weights** (12.0 M INT4 weight tiles x 128 B + one 64 B FP32 scale row per two K tiles;
Q4_0's own fp16 scales would make it 1.73 GB) and now ~3.0 GB of other loads (was 14.5 GB) plus 1.2 GB of stores, so
weights went from ~11% to ~31% of the DRAM traffic. Qwen3-0.6B: 275 MB of weights, ~255 MB of other loads (was 1.9 GB).
Numerics are unchanged (real-model per-layer error vs Hugging Face identical to before, 3.6e-6). What is left in "other"
is element-wise work (norms, residuals, the composed GELU, RoPE, masks) that still loads/stores whole 16-row tiles;
making those row-0-only is a separate, riskier change (later ops such as the attention core and K transposes do read
all 16 rows) and has not been done. Not covered by the one-row load: the FP32-weight GEMM (Gemma's small router) and
prefill/batched paths, which have real rows.

### Weak spots (what this table does not hide)

- **GELU is composed, not native.** All elementwise and scalar work of the whole expert program B is 22,880
  instructions out of 1.25 M (1.8%), and GELU is most of it, so by instruction count it is cheap. The caveat is that
  instruction count is not time: its ~60 passes over a tile each pay a DMA round trip through DRAM. If a cycle-level
  profile ever shows it matters, a general element-wise function unit is the one hardware change that would pay off.
  Until then it stays composed.
- **Top-k routing is on the host.** Each layer is two device programs with a host round trip between them (softmax,
  top-k over 128 experts, renormalise), 60 program launches per token. The on-device path in the tree exists only for
  top-1 (`REDUCE_ARGMAX` + `INDIRECT_ADDR`). On-device top-8 was prototyped with the existing instructions (repeated
  argmax with an indirect-store mask) and worked, but it cannot reach Gemma's experts: see the note on indirect
  addressing in "Where the simulator spends its time" below.
- **Sliding window and K=V are compiler constructs**, so they cost nothing in hardware, but the stream's shape
  changes at KV-tile boundaries (a derived-step cache has to restart there).
- **Embedding table and lm_head stay on the host** in FP32 (2.9 GB). They are not part of the ISA story at all.

## Where the simulator spends its time (measured 2026-11-06)

`perf stat` on `rtv_sim --serve` while it ran the real Gemma-4-26B (20 s window, one P-core):

| Counter | Value | Reading |
|---|---|---|
| instructions per cycle | 5.3 | the core is busy with dense arithmetic, not stalled on memory |
| dTLB load misses | 0.06% of loads | huge pages would gain nothing |
| cache misses | ~0.35 G (about 1 GB/s of traffic) | far below memory bandwidth |
| task-clock | 13.5 s of 20 s | the simulator idles about a third of the time, waiting for the Python driver |

So the simulator is **compute-bound on one core**, not memory-bound and not (as first assumed) dominated by
instruction decode. It uses one of the machine's 8 cores.

**The INT4W GEMM was scalar code.** `PEArray::compute_gemm` had AVX2 (and NEON) only for the INT8 path; the FP32-activation
x INT4-weight path that the real model runs, unpacking one nibble per iteration, was a plain loop. The INT4W path now
processes eight output columns per step (`_mm256_srlv_epi32` to pull eight nibbles out of four bytes, sign-extend by
xor/sub, convert, multiply by the scale row, multiply by the activation, add into C).

Why the result is bit-identical rather than merely close: every output element still accumulates over K in the same
order, and each lane does exactly the scalar steps, `a * (float(q) * scale)` then `+=`, as separately rounded
operations. No FMA is enabled for `rtv_pe` (only `-mavx2`); enabling `-mfma` would let the compiler fuse them and change
the rounding, so it must stay off here. Checked against the original loop on 32 cases (shapes with N not a multiple of 8,
with and without accumulate, with zero rows): 0 mismatches.

Microbenchmark, the common decode GEMM (16x64x16, one live row, the other fifteen zero and skipped): 0.98 us -> 0.26 us
per call, **3.8x**. That is the kernel only; the effect on whole-token time depends on the share of time spent in these
GEMMs and has to be measured on the model (see TODO.md).

Remaining ideas, in order of expected value: (1) the same vectorisation for the BF16W, INT8W and FP32 paths (they may
already be auto-vectorised at -O3; not checked); (2) threading the GEMM across output columns, since 7 of 8 cores are
idle (tiles are 16x16..., so the handoff cost has to be small); (3) an integrated-GPU kernel (shared memory means no copy
cost, but launch overhead dominates unless many tiles are batched per launch). The Intel NPU is INT8-oriented and does not
fit the FP32 x INT4-with-group-scale semantics. Measured after the changes above (6-layer / 40-position real-model run, 107 s -> 84 s overall): skipping all-zero
activation rows (the decode path loads one live row of 16) took the kernel from 0.26 to 0.13 us and the run to 84.5 s;
one memcpy per contiguous tile load/store changed nothing measurable; link-time optimisation across the decoder,
dispatcher and simulator (the per-instruction decode and dispatch is about 20% of simulator time) gave 83.4 s -> 82.4 s,
which is inside run-to-run noise, so it was not adopted. What remains of the simulator is `compute_gemm` (~35%), tile
copies (~25%) and instruction decode/dispatch (~20%), and the simulator is only about a third of wall time: the rest is
the Python compiler. Column-parallel threading of the GEMM is therefore bounded by roughly 35% x 2/3 of the simulator
share, i.e. a few percent of wall time at best, and each GEMM is so small (0.13 us) that thread handoff would cost more
than it saves unless many GEMMs were batched per handoff. Not built. The larger remaining lever is the Python side
(a position-independent attention program would remove the per-tile compile).

On-device top-8 routing was prototyped and then dropped (2026-11-06; the code is in commit 574e341 and reverted in
4a99261). It works with the existing ISA: k rounds of REDUCE_ARGMAX, each followed by an INDIRECT_ADDR store of -1e30
that masks the winner, indirect loads of the chosen logits and per-expert scales, a softmax among the k, and each
weight broadcast to a tile by a GEMM against a ones row; top-8 of 128 matched the host to 2e-6 on one fixed stream
(`test_moe_topk_ondevice`). It cannot reach Gemma-4-26B's experts: an indirect transfer adds a 32-bit SRAM offset to a
32-bit instruction address (the `dram_hi` bits hold the pointer's SRAM address), so it addresses at most 8 GiB, while
the experts are about 14 GB of the 16.5 GB image. Using it for the 26B model would need an ISA change (pointer address
narrowed to 12 bits so `dram_hi` extends the DRAM address again); nothing a compiler layout can fix.

Two things that were considered and are NOT worth doing: huge pages (TLB
misses are negligible) and reducing memory traffic (cache-miss traffic is about 1 GB/s).

## Where to go next

- **`TODO.md`** — the project's running history and roadmap: what's been
  built, in what order, and why, including bugs found and fixed along
  the way.
- **`TESTING.md`** — how the test suite actually proves correctness, and
  a guided tour of every test file in the order they were written.
- **`RTV-Accel_ NPU ISA & Simulator Specification.md`** — the precise,
  detailed hardware spec: exact instruction encodings, memory layout,
  every opcode's semantics.
