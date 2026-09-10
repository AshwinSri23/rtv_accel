#!/usr/bin/env python3
"""
RETIRED 2026-09-09 — not ported to the TVM path, not run.

This drives NPUCompiler.compile_model(), the full INT8-quantized whole-model
pipeline (RMSNorm, GQA attention with RoPE and KV-cache, SwiGLU, residuals,
chained across every layer, plus LM head). The project's target moved to
native FP32 precision with INT8 quantization out of scope (see
rtv-accel-fp32-only-target in project memory), and nothing on the TVM path
(toolchain/rtv_tvm/) yet does whole-model orchestration -- multi-head/GQA
looping, RoPE, KV-cache and layer chaining are all still open (see
TODO.md's Phase 1.1/1.2). Porting this file means building all of that
first, which is out of scope for the "migrate e2e_test.py" pass that
retired it; decided to retire rather than port, same treatment Phase 2
(INT8 calibration) already got. Left in place, unmodified below this
notice, as a reference for what a real decode loop needs once that
infrastructure exists -- not wired into any test runner or build (nothing
else in the repo imports it).

generate.py — host-side decode loop for the RTV-Accel NPU.

Turns the compiled per-layer pipeline into actual text generation. Two steps
of the loop are host work by necessity, not by omission:

  * the embedding lookup is a GATHER (embed_tokens[token_id]); LOAD_TILE takes
    a compile-time address, and nothing in the instruction word supports
    indirect or computed addressing, so a runtime token id cannot select a row
    on-device.

  * sampling needs an ARGMAX; REDUCE_MAX yields a maximum value with no index
    attached, and temperature/top-k are control flow, which this ISA has none
    of.

Everything between those two — the transformer layers and the vocabulary
projection — runs on the simulated NPU.

Usage:
    python3 toolchain/generate.py --prompt-ids 1,2,3 --steps 8
    python3 toolchain/generate.py --model Qwen/Qwen3-0.6B --prompt "Hello"
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npu_compiler import NPUCompiler, _padded   # noqa: E402


class Generator:
    """Drives compile -> execute -> sample, one token per step."""

    def __init__(self, compiler, sim, hal, workdir, verbose=False):
        self.c, self.sim, self.hal = compiler, sim, hal
        self.workdir, self.verbose = workdir, verbose

    def _run(self, program: bytes, image: bytes, dumps):
        """Execute one instruction stream against a DRAM image; return dumps."""
        prog = os.path.join(self.workdir, "step.bin")
        img  = os.path.join(self.workdir, "step.img")
        with open(prog, "wb") as f:
            f.write(program)
        with open(img, "wb") as f:
            f.write(image)

        args, paths = [], {}
        for label, addr, n in dumps:
            p = os.path.join(self.workdir, f"step.{label}")
            paths[label] = p
            args += ["--dump", f"{addr}:{n}:{p}"]

        # Size from the compiler's high-water mark, not the image length --
        # staging sits above the last initialised byte (see dram_bytes_needed).
        dram_mb = max(1, max(len(image),
                             self.c.dram_bytes_needed) // (1024 * 1024) + 2)
        r = subprocess.run([self.sim, prog, self.hal, str(dram_mb),
                            "--dram-image", img] + args,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"simulator failed:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
        return {k: open(v, "rb").read() for k, v in paths.items()}

    def step(self, token_id: int, image: bytearray, in_addr: int,
             out_addr: int, hidden: int, vocab: int, program: bytes,
             logits_addr: int, embed_scale: float = None, raw_embed=None):
        """One decode step: place the token's embedding, run, sample."""
        # ── host step 1: embedding gather ────────────────────────────────────
        if embed_scale == "fp32":
            # The embedding enters the residual stream as float, exactly as
            # the checkpoint stores it.
            row = np.asarray(raw_embed[token_id], dtype=np.float32)
            padded = np.zeros((1, _padded(hidden)), dtype=np.float32)
            padded[0, :hidden] = row[:hidden]
        elif embed_scale is not None:
            # The embedding is the residual stream's first value, so it is
            # written at the STREAM's scale and width -- not at a scale of its
            # own. Quantising it independently is what made x and every
            # sublayer output incommensurable in the first place.
            row = np.asarray(raw_embed[token_id], dtype=np.float32)
            codes = np.clip(np.round(row / embed_scale), -32768, 32767)
            padded = np.zeros((1, _padded(hidden)), dtype=np.int16)
            padded[0, :hidden] = codes[:hidden].astype(np.int16)
        else:
            row = self.c.embedding_row(token_id)
            padded = np.zeros((1, _padded(hidden)), dtype=np.int8)
            padded[0, :hidden] = row[:hidden]
        image[in_addr:in_addr + padded.nbytes] = padded.tobytes()

        # ── device: transformer layers + lm_head ─────────────────────────────
        out = self._run(program, bytes(image),
                        [("logits", logits_addr, _padded(vocab) * _padded(1) * 4)])

        # ── host step 2: sample ──────────────────────────────────────────────
        # logits come back TRANSPOSED, (vocab, 1) -- see compile_lm_head.
        # FP32 leaves the vocabulary projection in float; the INT8 path
        # leaves an INT32 accumulator. Same byte width, different meaning.
        dt = np.float32 if self.c.dtype == 3 else np.int32
        logits = np.frombuffer(out["logits"], dtype=dt).reshape(
            _padded(vocab), _padded(1))[:vocab, 0]
        return int(np.argmax(logits)), logits


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim", default=os.path.join(root, "build", "rtv_sim"))
    ap.add_argument("--hal", default=os.path.join(here, "hal_config.json"))
    ap.add_argument("--model", default=None,
                    help="HuggingFace model id; omit to use a small synthetic model")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-ids", default="1",
                    help="comma-separated token ids (used when --model is absent)")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fp32", action="store_true",
                    help="run with NO quantisation: float weights and "
                         "activations throughout. Proves the compiler's "
                         "lowering independently of every scale decision.")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip the scale plan (INT8 residual, implicit scales)")
    args = ap.parse_args()

    if not os.path.exists(args.sim):
        print(f"error: simulator not found at {args.sim}")
        return 2

    compiler = NPUCompiler(args.hal)

    if args.model:
        # Read the checkpoint through safetensors rather than torch: the
        # compiler only needs shapes and numpy arrays, and torch is ~2 GB of
        # dependency to read a file. See hf_loader.
        from hf_loader import SafeTensorsModel, hal_model_config
        path = args.model
        if not os.path.exists(path):
            from huggingface_hub import snapshot_download
            path = snapshot_download(path, allow_patterns=[
                "*.safetensors", "config.json", "tokenizer.json",
                "tokenizer_config.json", "vocab.json", "merges.txt"])
        model = SafeTensorsModel(path)
        compiler.config["model"] = hal_model_config(model.config())
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(path)
            ids = tok(args.prompt or "Hello").input_ids
        except Exception as e:
            print(f"(tokenizer unavailable: {e}; using --prompt-ids)")
            tok = None
            ids = [int(x) for x in args.prompt_ids.split(",")]
    else:
        tok, model, ids = None, _synthetic_model(compiler), \
            [int(x) for x in args.prompt_ids.split(",")]

    # ── Calibration ──────────────────────────────────────────────────────
    # Every activation scale has to be a compile-time constant: the device has
    # no reduction over a buffer and no way to feed a computed value into a
    # later instruction. So run the model once in the scale-tracking simulator,
    # read off what one integer step is worth in each buffer, and turn that
    # into the RESCALE constants the instruction stream carries.
    if args.fp32:
        # One width everywhere; nothing to calibrate, because a float carries
        # its own exponent and there is no scale to agree on.
        compiler.dtype = 3

    scale_plan = None
    if args.model and not args.no_calibrate and not args.fp32:
        import scale_sim as SS
        from scale_sim import ScaleQwen3
        SS.FLOAT_MODE = False
        SS.WEIGHT_BITS, SS.ACT_BITS, SS.RESIDUAL_BITS = 8, 8, 16
        calib = ScaleQwen3(model).collect_scales(ids)
        scale_plan = compiler.plan_scales(calib)
        print(f"calibrated : {len(scale_plan['layers'])} layers, "
              f"embed scale {scale_plan['embed_scale']:.4g}")

    max_ctx = len(ids) + args.steps
    compiler.compile_model(model, capture_weights=True, seq_len=1,
                           causal=False, cache_len=0, max_cache=max_ctx,
                           scale_plan=scale_plan)
    cfg    = compiler.config["model"]
    hidden = cfg["hidden_size"]
    vocab  = _vocab_of(compiler)

    # lm_head, appended after the layer stack.
    lm = next((r for r in compiler._weight_map.values() if r.key == "lm_head"), None)
    if lm is None:
        print("error: model has no lm_head; cannot produce logits")
        return 2
    logits_addr = compiler.alloc_staging(_padded(vocab) * _padded(1) * 4)
    compiler.compile_lm_head(dram_in=compiler.model_output_dram,
                             dram_w=lm.dram_addr, dram_out=logits_addr,
                             m=1, hidden=hidden, vocab=vocab)

    program = bytes(compiler.instruction_stream)
    with tempfile.TemporaryDirectory(prefix="rtv_gen_") as wd:
        img_path = os.path.join(wd, "base.img")
        compiler.export_dram_image(img_path)
        image = bytearray(open(img_path, "rb").read())
        need = logits_addr + _padded(vocab) * _padded(1) * 4
        if len(image) < need:
            image.extend(b"\0" * (need - len(image)))

        gen = Generator(compiler, args.sim, args.hal, wd, args.verbose)
        print(f"program : {len(program)//16} instructions")
        print(f"prompt  : {ids}")

        out_ids = list(ids)
        # Each step recompiles with cache_len = number of positions already
        # seen. There is no control flow in this ISA, so the cache length is
        # baked into addresses and instruction counts -- a fresh program per
        # step is the design, not a workaround. The cache REGIONS keep their
        # addresses across steps and the image carries them forward, so
        # earlier keys and values persist.
        for pos in range(len(ids) + args.steps - 1):
            prog_p, logits_p = _compile_step(compiler, model, hidden, vocab,
                                             cache_len=pos, max_cache=max_ctx,
                                             scale_plan=scale_plan)
            need_p = logits_p + _padded(vocab) * _padded(1) * 4
            if len(image) < need_p:
                image.extend(b"\0" * (need_p - len(image)))
            nxt, _lg = gen.step(out_ids[min(pos, len(out_ids) - 1)], image,
                                compiler.model_input_dram,
                                compiler.model_output_dram, hidden, vocab,
                                prog_p, logits_p,
                                embed_scale=("fp32" if args.fp32 else
                                             (scale_plan["embed_scale"]
                                              if scale_plan else None)),
                                # The synthetic model has no checkpoint to
                                # read an embedding row out of; it embeds via
                                # the compiler instead.
                                raw_embed=(model.tensor("model.embed_tokens.weight")
                                           if ((scale_plan or args.fp32)
                                               and hasattr(model, "tensor"))
                                           else None))
            if pos >= len(ids) - 1:
                out_ids.append(nxt)
                if args.verbose:
                    print(f"  -> {nxt}")

        print(f"output  : {out_ids}")
        if tok is not None:
            print(f"text    : {tok.decode(out_ids)}")
    return 0


def _compile_step(compiler, model, hidden, vocab, cache_len, max_cache,
                  scale_plan=None):
    """
    Recompile the stack for one decode step at a given cache length.

    Two pieces of compiler state must be handled or the DRAM image built for
    step 0 stops matching later steps:

      * _dram_cursor is monotonic, so re-ingesting without resetting it would
        assign every weight a NEW address each step while the image still
        holds them at the originals.
      * the captured weight bytes are cleared by ingest_model, and
        re-quantising them every step would be pure waste on a real model.
    """
    saved_bytes = compiler._weight_bytes
    compiler._dram_cursor = 0
    compiler.instruction_stream = bytearray()
    compiler._const_regions = {}
    # The plan must be passed every step: without it this recompile silently
    # reverts to the INT8 residual and implicit scales, and only the first
    # program would have been correct.
    compiler.compile_model(model, capture_weights=False, seq_len=1,
                           causal=False, cache_len=cache_len,
                           max_cache=max_cache, scale_plan=scale_plan)
    compiler._weight_bytes = saved_bytes
    lm = next(r for r in compiler._weight_map.values() if r.key == "lm_head")
    logits_addr = compiler.alloc_staging(_padded(vocab) * _padded(1) * 4)
    compiler.compile_lm_head(dram_in=compiler.model_output_dram,
                             dram_w=lm.dram_addr, dram_out=logits_addr,
                             m=1, hidden=hidden, vocab=vocab)
    return bytes(compiler.instruction_stream), logits_addr


def _vocab_of(compiler):
    r = next((r for r in compiler._weight_map.values() if r.key == "lm_head"), None)
    return r.shape[0] if r else 0


def _synthetic_model(compiler):
    """A tiny stand-in so the loop is runnable without downloading weights."""
    import types
    h, i, hd, v, L = 32, 64, 32, 64, 1
    rng = np.random.default_rng(0)
    compiler.config["model"] = dict(
        hidden_size=h, intermediate_size=i, num_layers=L,
        num_heads=1, num_kv_heads=1, head_dim=hd, rms_norm_eps=1e-6,
        rope_theta=1e6)
    shapes = {"q_proj": (hd, h), "k_proj": (hd, h), "v_proj": (hd, h),
              "o_proj": (h, hd), "gate_proj": (i, h), "up_proj": (i, h),
              "down_proj": (h, i)}
    params = [("model.embed_tokens.weight",
               rng.integers(-100, 101, size=(v, h)).astype(np.float32)),
              ("lm_head.weight",
               rng.integers(-8, 9, size=(v, h)).astype(np.float32))]
    for li in range(L):
        for k, shp in shapes.items():
            params.append((f"model.layers.{li}.{k}.weight",
                           rng.integers(-8, 9, size=shp).astype(np.float32)))

    class M:
        def named_parameters(self):
            for nm, a in params:
                yield nm, types.SimpleNamespace(
                    shape=a.shape,
                    detach=lambda a=a: types.SimpleNamespace(
                        cpu=lambda a=a: types.SimpleNamespace(numpy=lambda a=a: a)))
    return M()


if __name__ == "__main__":
    sys.exit(main())
