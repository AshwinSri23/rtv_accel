#!/usr/bin/env python3
"""
generate.py — host-side "prompt in, tokens out" driver for the RTV-Accel
NPU: a real Qwen3 checkpoint (toolchain/checkpoint_loader.py), tokenized
with a real BPE tokenizer (the `tokenizers` library reading the
checkpoint's own tokenizer.json directly -- no torch, same reasoning
hf_loader.py already documents for avoiding it: ~2 GB of dependency to
read a file), driven through tvm_backend.py's _emit_model_kvcache_fp32
one real decode step at a time, each step a fresh rtv_sim invocation
(the ISA has no control flow -- see codegen.py's own docstring: "each
decode step is a fresh ... instruction program").

This supersedes the INT8-era generate.py (retired 2026-09-09): that file
drove NPUCompiler.compile_model()/compile_lm_head(), both removed with
npu_compiler.py (see TODO.md's Phase 1). This is a full rewrite against
the TVM path's current API -- checkpoint_loader.py,
build_kvcache_model_dram_image, _emit_model_kvcache_fp32 -- not a port;
nothing here is INT8/scale-plan aware because the project's target moved
to native FP32 (see TODO.md's 2026-09-08 precision pivot).

Two steps of the loop are host work BY NECESSITY, not by omission --
the retired file's reasoning is still architecturally exact:

  * embedding lookup is a GATHER (embed_tokens[token_id]) -- LOAD_TILE
    takes a compile-time address, nothing in the instruction word
    supports indirect/computed addressing, so a runtime token id cannot
    select a row on-device.

  * sampling needs ARGMAX with an index -- REDUCE_MAX yields a maximum
    value with no index attached, and top-k/temperature are control
    flow, which this ISA has none of.

lm_head's projection is ALSO host-side here, for a narrower reason than
the ISA's limits: it is a single (hidden,)x(hidden,vocab) matmul, trivial
on a CPU, and not worth compiling a whole extra on-device program for --
unlike a transformer layer, which the whole rest of this codebase exists
to prove runs correctly on the simulated NPU. Everything between
embedding and lm_head -- every transformer layer, RMSNorm, SwiGLU,
attention, RoPE, q_norm/k_norm, the KV-cache -- runs on the simulated
NPU, unmodified from what rtv_tvm/test_checkpoint_decode_loop.py already
verifies.

Real dimensions are slow to schedule/codegen, roughly linearly in
--n-layers (on this machine, ~9s/step at 1 layer; Qwen3-0.6B's real 28
layers is correspondingly ~28x that per step -- consistent with
codegen.py's own "each decode step is a fresh ~12M-instruction program"
note for a full-size model). Use --n-layers to truncate for a fast
smoke test of the DRIVER LOOP itself; truncated output is not meaningful
text (most of the model is simply missing), only a correctness/plumbing
check -- see rtv_tvm/test_checkpoint_decode_loop.py for the real
correctness proof at 1-2 layers against a numpy reference. Omit
--n-layers (or pass the checkpoint's real layer count) for actual text.

Usage:
    python3 toolchain/generate.py --prompt "The capital of France is" --steps 8
    python3 toolchain/generate.py --model <dir> --prompt "..." --steps 4 --n-layers 2 --verbose
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from compiler_state import NPUCompiler, DType  # noqa: E402
import tvm_backend                             # noqa: E402
import checkpoint_loader                       # noqa: E402

ROOT = os.path.dirname(_HERE)
M = 16


def _find_checkpoint():
    """Same search this project's test files use -- the standard
    `hf download` cache location; None if nothing is there."""
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*")
    for cand in sorted(glob.glob(pattern)):
        if os.path.isfile(os.path.join(cand, "config.json")):
            has_weights = os.path.isfile(os.path.join(cand, "model.safetensors")) or \
                any(f.endswith(".safetensors") for f in os.listdir(cand))
            if has_weights:
                return cand
    return None


def _run_sim(sim, hal, instr_stream, dram_path_in, dram_path_out, extra_dumps=()):
    """One decode step: load dram_path_in, execute, dump the WHOLE
    updated DRAM to dram_path_out (so the next step can load it), plus
    any extra_dumps (addr, nbytes, label) -- the same round-trip pattern
    rtv_tvm/test_decode_loop.py and test_checkpoint_decode_loop.py use."""
    with tempfile.TemporaryDirectory(prefix="rtv_generate_step_") as wd:
        p = os.path.join(wd, "p.bin")
        open(p, "wb").write(bytes(instr_stream))
        dram_size = os.path.getsize(dram_path_in)
        mb = max(1, dram_size // (1024 * 1024) + 2)
        args = [sim, p, hal, str(mb), "--dram-image", dram_path_in,
                "--dump", f"0:{dram_size}:{dram_path_out}"]
        extra_paths = []
        for i, (addr, nbytes, _) in enumerate(extra_dumps):
            ep = os.path.join(wd, f"extra{i}.bin")
            args += ["--dump", f"{addr}:{nbytes}:{ep}"]
            extra_paths.append(ep)
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"simulator failed: {r.stdout[-800:]}{r.stderr[-800:]}")
        return [open(ep, "rb").read() for ep in extra_paths]


def generate(model_dir, prompt, n_new_tokens, n_layers=None, sim=None, hal=None,
            verbose=False):
    """Run the actual decode loop; returns (all_token_ids, decoded_text)."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))

    cfg, layers, embed_tokens, lm_head, final_norm = \
        checkpoint_loader.load_qwen3_checkpoint(model_dir, n_layers=n_layers)
    hidden, inter = cfg["hidden_size"], cfg["intermediate_size"]
    n_heads, n_kv_heads, head_dim = cfg["num_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    rope_theta, eps = cfg["rope_theta"], cfg["rms_norm_eps"]
    actual_n_layers = len(layers)

    prompt_ids = tok.encode(prompt).ids
    if not prompt_ids:
        raise ValueError("empty prompt tokenizes to zero tokens")
    P = len(prompt_ids)
    # P-1 prompt positions produce logits nobody samples from (the prompt's
    # OWN next tokens are already known); the (P-1)th position's output
    # produces the first NEW token, and each subsequent generated token
    # gets fed back for one more forward pass -- P-1+n_new_tokens total
    # forward passes for n_new_tokens genuinely new tokens (see this
    # function's module-level docstring... math spelled out in TODO.md's
    # entry for this file).
    total_steps = P - 1 + n_new_tokens

    comp0 = NPUCompiler(hal)
    comp0.dtype = DType.FP32
    dram, x_addr, layer_weights, scratch_a, scratch_b, max_tiles = \
        tvm_backend.build_kvcache_model_dram_image(
            comp0, hidden, actual_n_layers, n_kv_heads, head_dim,
            max_seq_len=total_steps, layer_weights_np=layers,
        )
    base_dram_cursor = comp0._dram_cursor

    all_ids = list(prompt_ids)
    next_input_id = None
    with tempfile.TemporaryDirectory(prefix="rtv_generate_state_") as wd:
        # ONE state file, overwritten in place every step -- not a fresh
        # step_N.img per step. Safe because rtv_sim fully reads --dram-image
        # into its own in-memory buffer (and closes the file) BEFORE it
        # writes any --dump output, so the same path can be both input and
        # output. This matters at real scale: Qwen3-0.6B's DRAM image is
        # ~1.7+ GB, and a fresh copy per step exhausted a 16 GB tmpfs /tmp
        # by step 5 of a real run (confirmed directly hitting this) --
        # accumulating one such file per generated token doesn't stay
        # small just because test_decode_loop.py's tiny random-weight
        # configs never grew large enough to reveal it.
        cur_path = os.path.join(wd, "state.img")
        open(cur_path, "wb").write(bytes(dram))

        for pos in range(total_steps):
            token_id = prompt_ids[pos] if pos < P else next_input_id

            comp = NPUCompiler(hal)
            comp.dtype = DType.FP32
            comp.instruction_stream = bytearray()
            comp._dram_cursor = base_dram_cursor

            x_row = np.zeros((M, hidden), dtype=np.float32)
            x_row[0, :] = embed_tokens[token_id].astype(np.float32)
            comp._const_regions[x_addr] = x_row.tobytes()

            final_addr = tvm_backend._emit_model_kvcache_fp32(
                comp, actual_n_layers, hidden, inter, n_heads, n_kv_heads, head_dim,
                x_addr=x_addr, layer_weights=layer_weights, scratch_a=scratch_a,
                scratch_b=scratch_b, pos=pos, rope_theta=rope_theta,
            )

            step_dram = bytearray(open(cur_path, "rb").read())
            needed = max(comp.dram_bytes_needed, len(step_dram))
            if needed > len(step_dram):
                step_dram += bytearray(needed - len(step_dram) + 0x10000)
            for addr, data in comp._const_regions.items():
                step_dram[addr:addr + len(data)] = data
            open(cur_path, "wb").write(bytes(step_dram))

            (out_raw,) = _run_sim(
                sim, hal, comp.instruction_stream, cur_path, cur_path,
                extra_dumps=[(final_addr, M * hidden * 4, "out")],
            )

            if pos >= P - 1:
                hidden_row = np.frombuffer(out_raw, dtype=np.float32).reshape(
                    M, hidden)[0].astype(np.float64)
                # Host-side tail: the model's own trailing RMSNorm, then
                # lm_head (HF's native (vocab, hidden) layout -- unlike
                # every layer weight, lm_head is NOT transposed by
                # checkpoint_loader.py, since it never goes through the
                # device GEMM), then argmax -- see module docstring for
                # why both stay off the NPU.
                normed = (hidden_row / np.sqrt((hidden_row ** 2).mean() + eps)
                         * final_norm.astype(np.float64))
                logits = normed @ lm_head.T.astype(np.float64)
                next_input_id = int(np.argmax(logits))
                all_ids.append(next_input_id)
                if verbose:
                    piece = tok.decode([next_input_id])
                    print(f"  step {pos}: token {next_input_id}  {piece!r}", file=sys.stderr)

    return all_ids, tok.decode(all_ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="checkpoint directory (config.json + tokenizer.json + "
                         "safetensors); default: auto-detect the hf-download "
                         "cache for Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=8, help="number of NEW tokens to generate")
    ap.add_argument("--n-layers", type=int, default=None,
                    help="truncate to the first N layers (fast driver smoke test; "
                         "output will not be coherent text -- see module docstring)")
    ap.add_argument("--sim", default=os.path.join(ROOT, "build", "rtv_sim"))
    ap.add_argument("--hal", default=os.path.join(_HERE, "hal_config.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.sim):
        print(f"error: simulator not found at {args.sim} (build it first)", file=sys.stderr)
        return 2

    model_dir = args.model or _find_checkpoint()
    if model_dir is None:
        print("error: no checkpoint given and none found in the hf-download cache.\n"
              "  run: hf download Qwen/Qwen3-0.6B --include \"*.json\" "
              "--include \"*.safetensors\" --include \"merges.txt\" --include \"vocab.json\"",
              file=sys.stderr)
        return 2

    print(f"model   : {model_dir}", file=sys.stderr)
    print(f"prompt  : {args.prompt!r}", file=sys.stderr)
    ids, text = generate(model_dir, args.prompt, args.steps, n_layers=args.n_layers,
                         sim=args.sim, hal=args.hal, verbose=args.verbose)
    print(f"ids     : {ids}")
    print(f"text    : {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
