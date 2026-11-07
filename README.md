# RTV-Accel

A **pretend AI chip** and the **compiler** that lets a real language model run on it.

The chip does not exist as silicon. It exists as a C++ program (a *simulator*) that behaves exactly like the chip would,
instruction by instruction. The compiler translates a real, downloaded language model into the chip's own machine
language. Together they let us show that the chip design works by running real software on it. The largest model run
so far is **Gemma-4-26B-A4B**, a 26-billion-parameter mixture-of-experts model, chatting across several turns.

If you are new to this area, read "The ideas you need" first. Everything else builds on it.

## Contents

1. [The ideas you need](#the-ideas-you-need)
2. [How the project fits together](#how-the-project-fits-together)
3. [How a model runs: dense and mixture-of-experts](#how-a-model-runs-dense-and-mixture-of-experts)
4. [Making the weights small: quantisation](#making-the-weights-small-quantisation)
5. [Try it](#try-it)
6. [What has been verified, and what has not](#what-has-been-verified-and-what-has-not)
7. [Speed](#speed)
8. [Where things are](#where-things-are)

## The ideas you need

**A language model is mostly big tables of numbers and one operation.** A model such as Gemma or Qwen is a stack of
*layers*. Each layer mostly multiplies the current token's vector of numbers by large matrices of learned numbers (the
*weights*). Matrix multiplication (a *GEMM*) is nearly all the work. A chip that does it fast and cheaply is an *NPU*
(neural processing unit).

**Generating text happens one token at a time.** A *token* is a piece of a word. To produce each new token the model
runs every layer once for that token. This step is called *decode*. Each token also looks back at all earlier tokens
through *attention*, using stored results called the *KV cache* (K and V stand for the "keys" and "values" kept for
every earlier token).

**Two kinds of model show up here.**
- A **dense** model uses all of its weights for every token. Qwen3-0.6B is the small dense model the project started with.
- A **mixture-of-experts (MoE)** model has many separate mini-networks ("experts") in each layer. A small *router*
  looks at each token and picks only a few experts for it. Gemma-4-26B-A4B has 128 experts per layer and uses 8 per
  token, so only about 4 billion of its 26 billion weights do work for any one token. Choosing the experts depends on
  the data, which makes MoE harder to compile than dense, because the program can't be fixed ahead of time.

**Weights are stored in few bits to save memory.** The Gemma file here keeps each weight in 4 bits instead of 32.
That is called *quantisation* (more below). The chip multiplies directly with the 4-bit weights.

**A simulator versus real hardware.** The simulator counts the cycles an instruction would take, so it also gives a
rough performance estimate. It runs on an ordinary computer, much slower than a real chip would.

## How the project fits together

```
  A downloaded model            THE COMPILER                  THE SIMULATED CHIP
  (weights on disk)             (Python, uses Apache TVM)     (C++ program: rtv_sim)

  Qwen3-0.6B  ──┐                                             ┌──────────────────────┐
  Gemma-4 GGUF ─┴─►  turns the model's math  ──►  a stream  ─►│ runs each instruction│─► the model's
                     into the chip's own          of 16-byte    │ exactly as silicon   │   output
                     instructions                 instructions  │ would                │
                                                                └──────────────────────┘
```

- **The chip (`inc/`, `src/`).** A C++ program with a 128-bit instruction format, a big slow **DRAM**, a small fast
  64 KiB **SRAM** (software-managed, like a notepad next to the processor), and a 16×16 grid of multiply-add units that
  does the matrix maths. It knows nothing about language models; it only follows instructions. That separation matters,
  because it means the compiler and simulator can check each other independently.
- **The compiler (`toolchain/`).** Takes a model and produces the instruction stream plus a memory image of the weights.
  It uses *Apache TVM* to schedule the maths onto the chip's operations, then walks the result and writes the
  instructions. One rule guides the design: **the hardware will keep changing, but the compiler should stay the same.**
  When something does not fit the chip, we change the compiler, not the instruction set.
- **The host.** A few jobs stay on the ordinary CPU: looking up a token's embedding, picking the next token, and (for
  MoE) choosing experts. These involve decisions or table lookups that the chip's simple instructions do not cover.
  The Python program that does this also launches the simulator and talks to it.

## How a model runs: dense and mixture-of-experts

**One program per layer, reused across layers.** Layers of the same shape have the same instructions apart from *where
their weights live in memory*. The compiler builds the program once and patches the memory addresses for each further
layer. This is called *relocation*, and it avoids recompiling 30 times.

**Attention depends on position.** Token number 40 reads a different set of cache entries from token number 41. The
cache is stored in tiles of 16 tokens, and the compiler compiles two positions for real in each tile and *derives* the
rest by shifting addresses.

**Dense layer.** Each layer is one program: attention first, then the feed-forward network. The host feeds in the
token's vector, runs the layers in order, and applies the final output projection (the "lm head") to choose the next
word.

**MoE layer (Gemma-4).** Each layer is *two* programs, with a decision in the middle:

1. **Program A** runs attention, the dense part of the feed-forward network, and the router, which scores all 128
   experts.
2. **The host** turns the scores into probabilities and picks the best 8 experts.
3. **Program B** runs those 8 experts. It is always the same instruction stream; only the memory addresses of the
   chosen experts' weights are patched in.

So the chip only ever sees fixed-shape programs, and the data-dependent choice happens on the host in between. There
is also an experimental variant where the chip picks one expert itself with a special "argmax and add an offset"
instruction (top-1 routing, no host round trip). It works, including with the more complex quantisation formats, but
Gemma uses the host path because it needs the top 8.

## Making the weights small: quantisation

The Gemma file stores weights in the **Q4_0** format: every 32 weights share one scale, and each weight is a 4-bit
whole number. The chip has a matching mode (called **INT4W**), so the weights load with no loss of precision. Activations,
the numbers flowing through the model, stay as 32-bit floats.

Other common formats (Q8_0, Q6_K, Q4_K) work too, with no new hardware. The compiler splits each such weight into
one or two 4-bit *planes* plus a small correction term, and the chip adds them back together. This is tested on small
models and exact to about one part in ten million. No real file in those formats was available to try.

## Try it

You need: a C++17 compiler and CMake; Python with Apache TVM in a virtual environment at `.venv-tvm` in the repository
root (see `TESTING.md`); and for the big model, the Q4_0 GGUF file, about 16 GB of free disk space for the memory image
(`~/.cache/rtv_accel`), and roughly 24 GB of free RAM while it runs.

**One-time setup of the Python environment.** The compiler lives in a virtual environment named `.venv-tvm` in the
repository root. This is how the current one was made (Python 3.14), written down here because the repository has no
install script:

```
python3 -m venv --system-site-packages .venv-tvm
.venv-tvm/bin/pip install apache-tvm==0.26.0 ml_dtypes numpy scipy tokenizers safetensors jinja2 transformers psutil
```

The environment needs `apache-tvm` (the compiler framework), `numpy`, `tokenizers` and `jinja2` (the chat prompt
template), `safetensors` (reading Hugging Face checkpoints) and `transformers` (used only for its tokenizer; PyTorch is
not needed and not installed). Comparisons against Hugging Face's reference model need PyTorch and run in a separate
environment. I wrote these steps from the existing environment and have not tried them on a clean machine, so expect to
adjust a version if one is no longer available.

**Build the simulator and run the tests**

```
cmake -B build && cmake --build build
cd build && ctest
```

The suite has 54 tests and takes about three minutes. It compares the chip's results with independent calculations,
including a small random Gemma-4 against Hugging Face's reference.

**Chat with Gemma-4-26B-A4B (multi-turn)**

```
.venv-tvm/bin/python toolchain/gemma4_model.py --gguf /path/to/gemma-4-26B_q4_0-it.gguf \
    --chat --max-ctx 512 --max-new 40
```

It takes a minute or two to load, then about 7 seconds per token. The model and KV cache stay loaded between turns,
and each turn only processes the tokens it has not already seen. An empty line ends the chat.

**One-shot prompt, or just the first few layers (quick check)**

```
.venv-tvm/bin/python toolchain/gemma4_model.py --gguf /path/to/gemma-4-26B_q4_0-it.gguf \
    --prompt "What is the capital of France?" --steps 8
.venv-tvm/bin/python toolchain/gemma4_model.py --gguf /path/to/gemma-4-26B_q4_0-it.gguf --layers 6 \
    --prompt-ids 2,3,4,5 --steps 2
```

**The small dense model:** `.venv-tvm/bin/python toolchain/generate.py --prompt "The capital of France is"` runs
Qwen3-0.6B (it needs the checkpoint downloaded; very slow for all 28 layers).

**Memory note.** The simulator holds the whole 16 GB image. The compiler can also use extra worker processes to compile
ahead (`RTV_COMPILE_WORKERS`). By default the number is chosen from the free memory (1 for the full model on a 31 GB
machine, up to 3 for small models); set `RTV_COMPILE_WORKERS=0` to turn them off if memory is tight.

## What has been verified, and what has not

**Verified**
- The real Gemma-4-26B-A4B on the simulated chip agrees with the Hugging Face reference, layer by layer. On three chat
  prompts of eight tokens the top-5 token ids match 24 of 24, with logits within 5e-5. Perplexity on 47 wikitext
  tokens is 994.158 on the chip and 994.154 for the reference. (The number is high because an instruction-tuned model
  is scoring raw text. It shows the two agree, not that the model is good at it.)
- A scripted three-turn chat on the full 30-layer model gave correct answers, including recalling the name from the
  first turn, and reused the saved KV cache between turns.
- Sliding-window attention on the real model with the window shrunk to 5 and 21 tokens matches the reference.
- 54 automated tests pass, including bit-exact checks of speedups against the plain versions.

**Not verified**
- The real 1024-token sliding window past 1024 tokens (it takes hours on the simulator).
- Q4_K / Q8_0 / Q6_K on a real model file, since none was available. They are tested on small random models.
- Anything on real silicon. This is simulation only.

## Speed

On a laptop (8 cores), generation takes about **7 seconds per token** for the full model. A shorter benchmark (6 layers,
40 positions) went from 157 s to about 84 s over this work. The main costs are the compiler's Python code, which builds
programs, and the simulator's matrix maths, which runs on one core. The details and measurements are in
`ARCHITECTURE.md` under "Where the simulator spends its time", and open ideas are in `TODO.md`.

## Where things are

| Path | What it is |
|---|---|
| `inc/`, `src/` | The simulated chip: instruction decoder, memory model, 16×16 math array, the `rtv_sim` program |
| `toolchain/tvm_backend.py`, `toolchain/rtv_tvm/` | The compiler, and its tests (`test_*.py`) |
| `toolchain/gemma4_model.py` | Runs Gemma-4: weight placement, relocation, MoE routing, multi-turn chat |
| `toolchain/gemma4_gguf.py`, `gguf_reader.py` | Read the GGUF file and its quantisation formats |
| `toolchain/quant_planes.py` | Splits Q4_K / Q8_0 / Q6_K weights into 4-bit planes plus a correction |
| `toolchain/gemma4_validate.py` | Compares the chip with Hugging Face for validation |
| `toolchain/generate.py` | The small Qwen3 text-generation demo |
| `driver/` | An early Linux kernel driver for the chip |
| `ARCHITECTURE.md` | A longer plain-language tour of every part |
| `TESTING.md` | How the tests prove correctness |
| `QUANTIZATION.md` | The weight formats and measurements |
| `RTV-Accel_ NPU ISA & Simulator Specification.md` | The exact instruction set and simulator behaviour |
| `TODO.md` | History and open items |
