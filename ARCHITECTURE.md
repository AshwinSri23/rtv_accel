# RTV-Accel: Project Overview (for new contributors)

This document explains what this project *is*, in plain language, and
what each part of it does. It's meant to be read once, before diving into
`TODO.md` (the roadmap/history) or `TESTING.md` (how the test suite
works) or the spec doc (the exact hardware details).

## The one-sentence version

This project is two things bolted together: a **simulated chip** (a
pretend NPU — a "neural processing unit," a chip specialized for the
matrix math neural networks need) and a **compiler** that takes a real,
pretrained language model (Qwen3-0.6B, a small but real open-weight LLM)
and translates it into the raw instructions that pretend chip understands
— so the whole pipeline can be run and checked, even though the physical
chip doesn't exist.

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

**The chip's instruction set has no loops, no branches, and no
computed/indirect addresses.** Every instruction's exact memory address
has to be baked in at compile time. This sounds like a strange
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
simpler, and floats don't need any of that scale bookkeeping. A cluster
of files in `toolchain/` are leftovers from that earlier INT8 effort and
are **not used by the current pipeline** (nothing imports them except
each other): `scale_sim.py`, `quant_sim.py`, `quant_eval.py`,
`ref_model.py`, `layer_probe.py`, `device_probe.py` (explicitly marked
`RETIRED` in its own file header), and `swizzle.py`. There's also a
now-deleted hand-written Python compiler, `npu_compiler.py`, that this
whole TVM-based compiler replaced — see `TODO.md`'s Phase 1 for that
story. None of this needs to be understood to work on the current
pipeline; it's mentioned here only so it doesn't cause confusion when you
see it in the file listing.

## Where to go next

- **`TODO.md`** — the project's running history and roadmap: what's been
  built, in what order, and why, including bugs found and fixed along
  the way.
- **`TESTING.md`** — how the test suite actually proves correctness, and
  a guided tour of every test file in the order they were written.
- **`RTV-Accel_ NPU ISA & Simulator Specification.md`** — the precise,
  detailed hardware spec: exact instruction encodings, memory layout,
  every opcode's semantics.
