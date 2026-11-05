#!/usr/bin/env python3
"""
generate.py — host-side "prompt in, tokens out" driver for the RTV-Accel
NPU: a real Qwen3 checkpoint (toolchain/checkpoint_loader.py), tokenized
with a real BPE tokenizer (the `tokenizers` library reading the
checkpoint's own tokenizer.json directly -- no torch, same reasoning
hf_loader.py already documents for avoiding it: ~2 GB of dependency to
read a file). The prompt's first min(len(prompt), 16) tokens are
processed in ONE real, causally-masked batch
(_emit_model_kvcache_batched_fp32) instead of one decode step each; every
token after that -- the rest of a longer prompt, and every generated
token -- runs through _emit_model_kvcache_fp32_relocated, one real
decode step at a time, each step a fresh rtv_sim invocation (the ISA has
no control flow -- see codegen.py's own docstring: "each decode step is
a fresh ... instruction program"), with only the first two layers
actually recompiled and every later layer's instructions relocated
instead. Both are described in full in TODO.md's entries for their
respective functions; this file just wires them together.

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

Real dimensions are still slow to schedule/codegen even with relocation
(the first two layers of every decode step are still a real ~7.6s-ish
TVM compile each at Qwen3-0.6B's full size; only layers 2+ are a cheap
address patch -- see TODO.md's entry for _emit_model_kvcache_fp32_
relocated for the measured, honest number: ~7x faster per step than
before relocation existed, not the "compile once" figure a naive
estimate would suggest). Use --n-layers to truncate for a fast smoke
test of the DRIVER LOOP itself; truncated output is not meaningful
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
import checkpoint_loader
from decode_step_cache import DecodeStepCache                       # noqa: E402

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


class _SimSession:
    """A persistent `rtv_sim --serve` process: the DRAM image (all the model
    weights and the KV-cache) is loaded ONCE and stays resident; each decode
    step writes only its own small constants, runs its program, and reads back
    only the output region. Replaces the old per-step round trip of the WHOLE
    multi-GB DRAM through a file."""

    def __init__(self, sim, hal, image_path, dram_bytes, workdir):
        self.wd = workdir
        mb = dram_bytes // (1024 * 1024) + 2
        self.p = subprocess.Popen([sim, "--serve", hal, str(mb), "--dram-image", image_path],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, bufsize=1)
        self.dram_bytes = mb * 1024 * 1024
        self.n = 0
        while self._readline() != "READY":
            pass

    def _readline(self):
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(f"simulator exited: {self.p.stderr.read()[-800:]}")
        return line.strip()

    def _cmd(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        reply = self._readline()
        while not (reply.startswith("OK") or reply.startswith("ERR")):
            reply = self._readline()
        if reply.startswith("ERR"):
            raise RuntimeError(f"simulator: {reply}")
        return reply

    def run_step(self, comp, extra_dumps=()):
        """Write this step's compile-time constants into the resident DRAM,
        run its instruction stream, return the extra_dumps (addr, nbytes, _)."""
        if comp.dram_bytes_needed > self.dram_bytes:
            raise RuntimeError(
                f"step needs {comp.dram_bytes_needed} bytes of DRAM but the session "
                f"has {self.dram_bytes}; raise _SESSION_HEADROOM")
        self.n += 1
        for i, (addr, data) in enumerate(comp._const_regions.items()):
            fp = os.path.join(self.wd, f"c{self.n}_{i}.bin")
            open(fp, "wb").write(bytes(data))
            self._cmd(f"WRITE {addr} {fp}")
            os.remove(fp)
        prog = os.path.join(self.wd, f"p{self.n}.bin")
        open(prog, "wb").write(bytes(comp.instruction_stream))
        outs = [os.path.join(self.wd, f"o{self.n}_{i}.bin") for i in range(len(extra_dumps))]
        specs = " ".join(f"{a}:{nb}:{o}" for (a, nb, _), o in zip(extra_dumps, outs))
        self._cmd(f"RUN {prog} {specs}")
        os.remove(prog)
        res = [open(o, "rb").read() for o in outs]
        for o in outs:
            os.remove(o)
        return res

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=30)
        except Exception:  # noqa: BLE001
            self.p.kill()


# Extra DRAM beyond the initial image for per-step staging (the first step's
# footprint is measured; later steps reuse the same staging region).
_SESSION_HEADROOM = 256 * 1024 * 1024


def generate(model_dir, prompt, n_new_tokens, n_layers=None, sim=None, hal=None,
            verbose=False, bf16_weights=False, use_graph=False, int8_weights=False,
             int4_group=0, int4_quant=None, prompt_ids=None, prefill_hook=None):
    """Run the actual decode loop; returns (all_token_ids, decoded_text).

    prompt_ids: use these token ids instead of tokenizing `prompt`. prefill_hook(tile, rows):
    called after every prefill tile with that tile's final hidden rows (n_real, hidden) --
    chip_ppl.py scores every prompt position with it."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))

    cfg, layers, embed_tokens, lm_head, final_norm = \
        checkpoint_loader.load_qwen3_checkpoint(model_dir, n_layers=n_layers)
    hidden, inter = cfg["hidden_size"], cfg["intermediate_size"]
    n_heads, n_kv_heads, head_dim = cfg["num_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    rope_theta, eps = cfg["rope_theta"], cfg["rms_norm_eps"]
    actual_n_layers = len(layers)

    prompt_ids = list(prompt_ids) if prompt_ids is not None else tok.encode(prompt).ids
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
    comp0.bf16_weights = bf16_weights
    comp0.int8_weights = int8_weights
    comp0.int4_group = int4_group
    comp0.dtype = DType.FP32
    dram, x_addr, layer_weights, scratch_a, scratch_b, max_tiles = \
        tvm_backend.build_kvcache_model_dram_image(
            comp0, hidden, actual_n_layers, n_kv_heads, head_dim,
            max_seq_len=total_steps, layer_weights_np=layers,
            bf16_weights=bf16_weights, int8_weights=int8_weights,
            int4_group=int4_group, int4_quant=int4_quant,
        )
    base_dram_cursor = comp0._dram_cursor
    if use_graph:
        import decoder_graph
        graph_cfg = dict(hidden=hidden, inter=inter, n_heads=n_heads, n_kv_heads=n_kv_heads,
                         head_dim=head_dim, eps=eps)

    def _sample(hidden_row_f32):
        """Host-side tail: the model's own trailing RMSNorm, then lm_head
        (HF's native (vocab, hidden) layout -- unlike every layer weight,
        lm_head is NOT transposed by checkpoint_loader.py, since it never
        goes through the device GEMM), then argmax -- see module
        docstring for why both stay off the NPU."""
        hidden_row = hidden_row_f32.astype(np.float64)
        normed = (hidden_row / np.sqrt((hidden_row ** 2).mean() + eps)
                 * final_norm.astype(np.float64))
        logits = normed @ lm_head.T.astype(np.float64)
        return int(np.argmax(logits))

    # Real prefill batching: the first min(P, M) prompt tokens are
    # genuinely distinct real rows, causally self-attended in ONE call
    # (tvm_backend._emit_model_kvcache_batched_fp32) instead of one
    # separate decode step each -- see TODO.md's entry for this function
    # for the full design (real per-row RMSNorm/q_norm/k_norm, a real
    # causal mask, writing straight into the KV-cache). A prompt longer
    # than M tokens still starts with one such batch for its first M
    # tokens; the remaining prompt tokens fall through to the ordinary
    # one-at-a-time loop below, unchanged.
    # The whole prompt is prefilled in tiles of M=16: tile c holds prompt
    # positions 16c..16c+15 and attends to every earlier tile's cached K/V
    # (tile_idx=c), so any prompt length goes through the batched path.
    n_tiles = -(-P // M)
    n_batch = P
    # Real attention scores are q.k / sqrt(head_dim). The chip's softmax has no
    # such step, so it is applied to Q (mathematically identical). Without it
    # the output is not Qwen3's: found by comparing against Hugging Face.
    attn_scale = float(head_dim) ** -0.5

    all_ids = list(prompt_ids)
    next_input_id = None
    with tempfile.TemporaryDirectory(prefix="rtv_generate_state_") as wd:
        # The DRAM image (weights + KV-cache) is written once and loaded into
        # ONE persistent rtv_sim session (see _SimSession): it stays resident
        # for every prefill tile and decode step.
        cur_path = os.path.join(wd, "state.img")
        open(cur_path, "wb").write(bytes(dram))
        session = None
        batch_out = None
        n_real = 0

        for c in range(n_tiles):
            n_real = min(M, P - c * M)
            comp = NPUCompiler(hal)
            comp.bf16_weights = bf16_weights
            comp.int8_weights = int8_weights
            comp.int4_group = int4_group
            comp.rms_eps = eps      # RMSNorm epsilon (the chip's norms add it into the sum of squares)
            comp.dtype = DType.FP32
            comp.instruction_stream = bytearray()
            comp._dram_cursor = base_dram_cursor
            x_batch = np.zeros((M, hidden), dtype=np.float32)
            for r in range(n_real):
                x_batch[r, :] = embed_tokens[prompt_ids[c * M + r]].astype(np.float32)
            comp._const_regions[x_addr] = x_batch.tobytes()

            if use_graph:
                # prefill through the graph path too (per-row norms and the multi-tile causal
                # attention are registered externs; adds and SwiGLU are pattern-claimed)
                # layers 0-1 compile for real, the rest are address-patched (as in decode)
                final_addr = tvm_backend._emit_model_kvcache_fp32_relocated(
                    comp, actual_n_layers, hidden, inter, n_heads, n_kv_heads, head_dim,
                    x_addr=x_addr, layer_weights=layer_weights, scratch_a=scratch_a,
                    scratch_b=scratch_b, pos=0, rope_theta=rope_theta, attn_scale=attn_scale,
                    layer_emitter=decoder_graph.make_prefill_layer_emitter(
                        graph_cfg, n_real, c, rope_theta, attn_scale))
            else:
                final_addr = tvm_backend._emit_model_kvcache_batched_fp32(
                    comp, actual_n_layers, hidden, inter, n_heads, n_kv_heads, head_dim,
                    x_addr=x_addr, layer_weights=layer_weights, scratch_a=scratch_a,
                    scratch_b=scratch_b, n_real=n_real, rope_theta=rope_theta,
                    tile_idx=c, attn_scale=attn_scale,
                )
            if session is None:
                session = _SimSession(
                    sim, hal, cur_path,
                    max(comp.dram_bytes_needed, len(dram)) + _SESSION_HEADROOM, wd)
            (out_raw,) = session.run_step(
                comp, extra_dumps=[(final_addr, M * hidden * 4, "out")])
            batch_out = np.frombuffer(out_raw, dtype=np.float32).reshape(M, hidden)
            if prefill_hook is not None:
                prefill_hook(c, batch_out[:n_real].copy())

        if total_steps >= n_batch:
            # The last tile's last real row is exactly what a decode step at
            # pos=P-1 would produce, so sample the first new token from it now
            # instead of running that position again one-at-a-time.
            next_input_id = _sample(batch_out[n_real - 1])
            all_ids.append(next_input_id)
            if verbose:
                piece = tok.decode([next_input_id])
                print(f"  step {n_batch - 1}: token {next_input_id}  {piece!r}",
                     file=sys.stderr)

        # Continue one token at a time -- every generated token -- via the
        # schedule-RELOCATED path: layers 0-1 compile for real once, every
        # later layer is a cheap address patch instead of a fresh TVM
        # schedule/codegen pass (~7x faster per step at Qwen3-0.6B's real
        # 28-layer scale; see _emit_model_kvcache_fp32_relocated).
        step_cache = None if os.environ.get("RTV_NO_STEP_CACHE") else \
            DecodeStepCache(M, x_addr)
        step_final_addr = None
        for pos in range(n_batch, total_steps):
            token_id = prompt_ids[pos] if pos < P else next_input_id

            def _new_comp():
                comp = NPUCompiler(hal)
                comp.bf16_weights = bf16_weights
                comp.int8_weights = int8_weights
                comp.int4_group = int4_group
                comp.reuse_layer_staging = True    # layers 2+ share layer 1's staging (test_staging_reuse)
                comp.rms_eps = eps      # RMSNorm epsilon (the chip's norms add it into the sum of squares)
                comp.dtype = DType.FP32
                comp.instruction_stream = bytearray()
                comp._dram_cursor = base_dram_cursor
                return comp

            x_row = np.zeros((M, hidden), dtype=np.float32)
            x_row[0, :] = embed_tokens[token_id].astype(np.float32)

            # within one KV tile consecutive steps differ only in a few addresses and two
            # position-dependent constants: derive them (decode_step_cache.py) instead of recompiling
            comp = step_cache.derive(pos, _new_comp()) if step_cache is not None else None
            if comp is not None:
                comp._const_regions[x_addr] = x_row.tobytes()
                final_addr = step_final_addr
            else:
                comp = _new_comp()
                comp._const_regions[x_addr] = x_row.tobytes()
                final_addr = tvm_backend._emit_model_kvcache_fp32_relocated(
                    comp, actual_n_layers, hidden, inter, n_heads, n_kv_heads, head_dim,
                    x_addr=x_addr, layer_weights=layer_weights, scratch_a=scratch_a,
                    scratch_b=scratch_b, pos=pos, rope_theta=rope_theta,
                    attn_scale=attn_scale,
                    # --graph: compile each decode layer through the Relax/BYOC graph path
                    # (decoder_graph.py) instead of the hand-driven emitter; the stream is
                    # byte-identical (test_byoc_decoder.py); prefill is graph-driven too.
                    layer_emitter=(decoder_graph.make_layer_emitter(
                        graph_cfg, layer_weights, pos, rope_theta, attn_scale) if use_graph else None),
                )
                step_final_addr = final_addr
                if step_cache is not None:
                    step_cache.observe(pos, comp)

            (out_raw,) = session.run_step(
                comp, extra_dumps=[(final_addr, M * hidden * 4, "out")])

            if pos >= P - 1:
                hidden_row = np.frombuffer(out_raw, dtype=np.float32).reshape(M, hidden)[0]
                next_input_id = _sample(hidden_row)
                all_ids.append(next_input_id)
                if verbose:
                    piece = tok.decode([next_input_id])
                    print(f"  step {pos}: token {next_input_id}  {piece!r}", file=sys.stderr)

        session.close()
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
    ap.add_argument("--graph", action="store_true",
                    help="compile decode layers through the graph-driven (Relax/BYOC) path")
    ap.add_argument("--bf16-weights", action="store_true",
                    help="store the 7 per-layer weight matrices as BF16 (ISA v1.6 BF16W): "
                         "half the weight bytes in DRAM, FP32 accumulate")
    ap.add_argument("--int8-weights", action="store_true",
                    help="store the 7 per-layer weight matrices as INT8 with per-output-channel "
                         "FP32 scales (ISA v1.8 INT8W): a quarter of the weight bytes in DRAM")
    ap.add_argument("--int4-weights", metavar="NPZ", default=None,
                    help="INT4 group-scaled weights (ISA v1.9 INT4W) from quant_gptq.py; the group size "
                         "comes from the file")
    ap.add_argument("--int4-rtn-group", type=int, default=0, metavar="G",
                    help="INT4 with plain round-to-nearest, group size G (no calibration; for testing)")
    args = ap.parse_args()
    int4_kw = {}
    if args.int4_weights:
        z = np.load(args.int4_weights)
        quant = {}
        for key in z.files:
            if key.endswith(".q"):
                li, name = key.split(".")[:2]
                quant[(int(li), name)] = (z[key], z[key[:-2] + ".s"])
        int4_kw = dict(int4_group=int(z["group"]), int4_quant=quant)
    elif args.int4_rtn_group:
        int4_kw = dict(int4_group=args.int4_rtn_group)

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
                         sim=args.sim, hal=args.hal, verbose=args.verbose,
                         bf16_weights=args.bf16_weights, use_graph=args.graph,
                         int8_weights=args.int8_weights, **int4_kw)
    print(f"ids     : {ids}")
    print(f"text    : {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
