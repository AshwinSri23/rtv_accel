#!/usr/bin/env python3
"""
RETIRED 2026-09-09 — its entire premise is moot, not run.

This tool's whole purpose is comparing `scale_sim` (an INT8 reference
simulator) against the device to catch INT8 LOWERING bugs -- see the
"scale_sim vs DEVICE" line below, its one job. The project's target moved
to native FP32 precision with INT8 quantization out of scope (see
rtv-accel-fp32-only-target in project memory), so the thing this exists to
debug no longer exists as a target. This isn't a migration candidate the
way a test case is -- there's no FP32 equivalent of "does INT8 lowering
match a fixed-point reference" to port it to. Retired rather than
repurposed: reusing the name for a differently-scoped tool (e.g. TVM-path
output vs. a float reference) would be a new tool wearing an old name, not
a migration of this one. Same treatment Phase 2 (INT8 calibration) already
got. Left in place, unmodified below this notice, as a reference for the
probe methodology (truncate to L layers, dump a buffer, compare depth by
depth) if an FP32-equivalent debugging need ever comes up -- not wired into
any test runner or build (nothing else in the repo imports it). Also see
rtv-accel-probe-blind-spots in project memory: even under the old INT8
premise, this probe only ever covered up to the residual stream, not the
full model -- a caveat that predates and is independent of this retirement.

device_probe.py — locate where the DEVICE diverges from the proven design.

The project has three models, and each comparison answers a different
question:

    ref_model  vs  quant_sim   how much accuracy quantisation costs
    ref_model  vs  scale_sim   whether the proposed design is sound
    scale_sim  vs  DEVICE      whether the compiler lowers it correctly   <-- this

`scale_sim` reaches rank 0-1 on the test prompt, so any disagreement between
it and the device is a LOWERING bug: a wrong address, a wrong constant, a
missing instruction. Comparing the device against `ref_model` instead would
conflate that with the parts of the design not yet ported.

Runs the compiled model truncated to L layers, dumps the residual buffer,
converts its INT16 codes back to values with the scale the compiler itself
planned, and reports cosine against `scale_sim` at the same depth. A cliff at
one layer is a bug in that layer's lowering; a gradual decay is accumulated
numerical difference.

Usage:
    python3 toolchain/device_probe.py --layers 1,2,3,4
"""

import argparse
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import scale_sim as SS                                    # noqa: E402
from scale_sim import ScaleQwen3                          # noqa: E402
from hf_loader import SafeTensorsModel, hal_model_config  # noqa: E402
from npu_compiler import NPUCompiler, _padded             # noqa: E402


class TruncatedModel:
    """
    Exposes only the first `n` transformer layers, so a shallow probe does not
    pay to ingest and write all 28 layers of weights.

    `drop_io` also removes embed_tokens and lm_head. The probe never needs
    them -- it writes the input row itself and reads the residual, not logits
    -- and in FP32 those two tables are 622 MB each, which would dominate the
    DRAM image and make the run impractical for no diagnostic gain.
    """

    def __init__(self, inner, n, drop_io=False):
        self.inner, self.n, self.drop_io = inner, n, drop_io

    def named_parameters(self):
        import re
        for name, p in self.inner.named_parameters():
            m = re.search(r"\.layers\.(\d+)\.", name)
            if m and int(m.group(1)) >= self.n:
                continue
            if self.drop_io and ("embed_tokens" in name or "lm_head" in name):
                continue
            yield name, p

    def tensor(self, name):
        return self.inner.tensor(name)

    def config(self):
        c = dict(self.inner.config())
        c["num_hidden_layers"] = self.n
        return c


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n else 0.0


def agreement(ref, got, drop=8):
    """
    Cosine, plus two measures the outlier channels cannot dominate.

    Plain cosine on this model is largely a report on one channel: the
    residual holds massive activations (channel 35 reaches 2610 against a
    median of 0.065, a 40000:1 ratio), so the inner product is set by whether
    that single channel is right and the other 1023 barely move the number.
    A per-layer cosine of 0.98 can sit on top of a badly wrong hidden state.

      cos_trim  cosine with the `drop` largest-magnitude channels removed
      med_rel   median per-channel relative error -- unweighted, so every
                channel counts the same regardless of size
    """
    ref = np.asarray(ref, dtype=np.float64).ravel()
    got = np.asarray(got, dtype=np.float64).ravel()
    keep = np.argsort(np.abs(ref))[::-1][drop:]
    scale = np.median(np.abs(ref[keep])) or 1.0
    rel = np.abs(got[keep] - ref[keep]) / np.maximum(np.abs(ref[keep]), scale * 1e-3)
    return cosine(ref, got), cosine(ref[keep], got[keep]), float(np.median(rel))


def probe_stages(model, sim, hal, ids, n_layers, plan, ref_stages, fp32=False):
    """
    Dump every INTERMEDIATE buffer of the last layer and compare each.

    Each buffer has its own width and scale, so the comparison also reports
    the magnitude ratio: a wrong CONSTANT shows up as output that is
    correlated but uniformly scaled, a lowering bug as output that is
    decorrelated. Those two look identical when only the layer's final value
    is visible, which is why the single-value probe could not tell them apart.
    """
    c = NPUCompiler(hal)
    c.config["model"] = hal_model_config(model.config())
    c.config["model"]["num_layers"] = n_layers
    if fp32:
        c.dtype = 3
    sub = TruncatedModel(model, n_layers, drop_io=fp32)
    sub_plan = None if fp32 else dict(plan)
    if sub_plan is not None:
        sub_plan["layers"] = plan["layers"][:n_layers]
    c.compile_model(sub, capture_weights=True, seq_len=1, causal=False,
                    cache_len=0, max_cache=1, scale_plan=sub_plan)

    hidden = c.config["model"]["hidden_size"]
    pl = (sub_plan["layers"][n_layers - 1] if sub_plan
          else {"s_o": 1.0, "s_d": 1.0, "s_res": 1.0, "y2_peak": 127.0})
    # norm_dram holds the POST-attention norm, whose scale is now calibrated
    # (y2_peak/127) rather than the fixed Q0.7 the ISA originally assumed.
    scale_of = {"q07": pl["y2_peak"] / 127.0, "s_o": pl["s_o"],
                "s_d": pl["s_d"], "s_res": pl["s_res"]}
    bufs = c.layer_buffers[n_layers - 1]

    with tempfile.TemporaryDirectory(prefix="rtv_stage_") as wd:
        img = os.path.join(wd, "d.img")
        c.export_dram_image(img)
        image = bytearray(open(img, "rb").read())
        need = max(a + c.tile_dim * _padded(hidden) * (4 if fp32 else w)
                   for a, w, _ in bufs.values())
        if len(image) < need:
            image.extend(b"\0" * (need - len(image)))

        row = np.asarray(model.tensor("model.embed_tokens.weight")[ids[0]],
                         dtype=np.float32)
        if fp32:
            padded = np.zeros((1, _padded(hidden)), dtype=np.float32)
            padded[0, :hidden] = row[:hidden]
        else:
            codes = np.clip(np.round(row / plan["embed_scale"]), -32768, 32767)
            padded = np.zeros((1, _padded(hidden)), dtype=np.int16)
            padded[0, :hidden] = codes[:hidden].astype(np.int16)
        a0 = c.model_input_dram
        image[a0:a0 + padded.nbytes] = padded.tobytes()

        prog = os.path.join(wd, "p.bin")
        open(prog, "wb").write(bytes(c.instruction_stream))
        open(img, "wb").write(bytes(image))

        args, paths = [], {}
        for name, (addr, w, _) in bufs.items():
            n = c.tile_dim * _padded(hidden) * (4 if fp32 else w)
            paths[name] = os.path.join(wd, name)
            args += ["--dump", f"{addr}:{n}:{paths[name]}"]
        # Size DRAM from the compiler's high-water mark, not the image
        # length: staging is allocated above the last initialised byte and so
        # never lengthens the image (see dram_bytes_needed).
        mb = max(1, max(len(image), c.dram_bytes_needed) // (1024 * 1024) + 2)
        r = subprocess.run([sim, prog, hal, str(mb), "--dram-image", img] + args,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  simulator failed: {r.stdout[-300:]}{r.stderr[-300:]}")
            return
        raws = {k: open(v, "rb").read() for k, v in paths.items()}

    order = ["norm", "attn_out", "attn_res", "mlp_out", "layer_out"]
    print(f"  {'stage':10s} {'cosine':>8} {'cos_trim':>9} {'med_rel':>8}  {'mag':>6}")
    for name in order:
        addr, w, skey = bufs[name]
        dt = np.float32 if fp32 else (np.int8 if w == 1 else np.int16)
        arr = np.frombuffer(raws[name], dtype=dt).reshape(-1, _padded(hidden))
        got = arr[0, :hidden].astype(np.float64) * (1.0 if fp32 else scale_of[skey])
        ref = np.asarray(ref_stages[name])[-1][:hidden]
        rmag = float(np.abs(ref).max()) or 1.0
        c, ct, mr = agreement(ref, got)
        print(f"  {name:10s} {c:8.4f} {ct:9.4f} {mr:8.3f}  "
              f"{float(np.abs(got).max()) / rmag:6.3f}")


def probe(model, sim, hal, ids, n_layers, plan, ref_state, report=False,
          fp32=False):
    """Compile+run `n_layers` layers on the device; return (cosine, detail)."""
    c = NPUCompiler(hal)
    c.config["model"] = hal_model_config(model.config())
    c.config["model"]["num_layers"] = n_layers
    if fp32:
        c.dtype = 3
    sub = TruncatedModel(model, n_layers, drop_io=fp32)

    sub_plan = None if fp32 else dict(plan)
    if sub_plan is not None:
        sub_plan["layers"] = plan["layers"][:n_layers]
    c.compile_model(sub, capture_weights=True, seq_len=1, causal=False,
                    cache_len=0, max_cache=1, scale_plan=sub_plan)

    hidden = c.config["model"]["hidden_size"]
    out_addr = c.model_output_dram
    nbytes = c.tile_dim * _padded(hidden) * (4 if fp32 else 2)

    with tempfile.TemporaryDirectory(prefix="rtv_probe_") as wd:
        img = os.path.join(wd, "d.img")
        c.export_dram_image(img)
        image = bytearray(open(img, "rb").read())
        need = out_addr + nbytes
        if len(image) < need:
            image.extend(b"\0" * (need - len(image)))

        # The embedding enters at the residual stream's own scale and width.
        row = np.asarray(model.tensor("model.embed_tokens.weight")[ids[0]],
                         dtype=np.float32)
        if fp32:
            padded = np.zeros((1, _padded(hidden)), dtype=np.float32)
            padded[0, :hidden] = row[:hidden]
        else:
            codes = np.clip(np.round(row / plan["embed_scale"]), -32768, 32767)
            padded = np.zeros((1, _padded(hidden)), dtype=np.int16)
            padded[0, :hidden] = codes[:hidden].astype(np.int16)
        a = c.model_input_dram
        image[a:a + padded.nbytes] = padded.tobytes()

        prog = os.path.join(wd, "p.bin")
        open(prog, "wb").write(bytes(c.instruction_stream))
        open(img, "wb").write(bytes(image))
        out = os.path.join(wd, "o.bin")
        # Size DRAM from the compiler's high-water mark, not the image
        # length: staging is allocated above the last initialised byte and so
        # never lengthens the image (see dram_bytes_needed).
        mb = max(1, max(len(image), c.dram_bytes_needed) // (1024 * 1024) + 2)
        r = subprocess.run([sim, prog, hal, str(mb), "--dram-image", img,
                            "--dump", f"{out_addr}:{nbytes}:{out}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None, f"simulator failed: {r.stdout[-400:]}{r.stderr[-400:]}"
        raw = np.frombuffer(open(out, "rb").read(),
                            dtype=np.float32 if fp32 else np.int16)

    got_codes = raw.reshape(-1, _padded(hidden))[0, :hidden]
    got = (got_codes.astype(np.float64) if fp32
           else got_codes.astype(np.float64) * plan["layers"][n_layers - 1]["s_res"])
    c, ct, mr = agreement(ref_state, got)
    if report:
        print(f"{n_layers:>7}  {c:8.4f} {ct:9.4f} {mr:8.3f}")
    return c, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--sim", default="build/rtv_sim")
    ap.add_argument("--hal", default="toolchain/hal_config.json")
    ap.add_argument("--token", type=int, default=374)   # " is"
    ap.add_argument("--layers", default="1,2,3")
    ap.add_argument("--fp32", action="store_true",
                    help="probe the no-quantisation path against ref_model")
    ap.add_argument("--stages", type=int, default=None,
                    help="dump every intermediate buffer of this layer index")
    ap.add_argument("--no-fold", action="store_true",
                    help="apply gamma to the activation instead of folding it")
    args = ap.parse_args()

    path = args.model
    if not os.path.exists(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(args.model, allow_patterns=[
            "*.safetensors", "config.json", "tokenizer*", "vocab*", "merges*"])
    model = SafeTensorsModel(path)

    # One token: the device runs decode, and a single position keeps the probe
    # cheap while still exercising every ported path (norms, projections,
    # residual adds, the MLP and its LUT).
    ids = [args.token]
    SS.FLOAT_MODE = False
    SS.WEIGHT_BITS, SS.ACT_BITS, SS.RESIDUAL_BITS = 8, 8, 16
    # Match the compiler: ingest_model folds the RMSNorm gain into the weights
    # (GAMMA_FOLD). Leaving scale_sim applying it to the activation makes the
    # two models compute genuinely different things, and the probe would
    # report that difference as if it were a lowering bug.
    SS.FOLD_GAMMA = not args.no_fold
    sim_model = ScaleQwen3(model)
    calib = sim_model.collect_scales(ids)
    plan = NPUCompiler(args.hal).plan_scales(calib)
    states = sim_model.forward(ids)

    if args.stages is not None:
        L = args.stages
        print(f"token {ids[0]}   stage dump of layer {L}")
        if args.fp32:
            from ref_model import RefQwen3, rmsnorm
            rf = RefQwen3(model)
            st = rf.forward(ids, upto_layer=L + 1)
            # ref_model does not expose intermediates, so only the two the
            # probe can check without them are compared here.
            ref_stages = {"layer_out": st[L + 1], "attn_res": st[L + 1],
                          "norm": st[L + 1], "attn_out": st[L + 1],
                          "mlp_out": st[L + 1]}
        else:
            ref_stages = sim_model.trace_layer(ids, L)
        probe_stages(model, args.sim, args.hal, ids, L + 1, plan,
                     ref_stages, fp32=args.fp32)
        return

    if args.fp32:
        # With no quantisation the target is the float model itself: scale_sim
        # in float mode IS ref_model, so compare against the simpler one.
        from ref_model import RefQwen3
        rstates = RefQwen3(model).forward(ids)
        print(f"token {ids[0]}   probing FP32 device against ref_model")
    print(f"{'layers':>7}  {'cosine':>8} {'cos_trim':>9} {'med_rel':>8}")
    for n in [int(x) for x in args.layers.split(",")]:
        ref = rstates[n][-1] if args.fp32 else states[n].value[-1]
        cos, detail = probe(model, args.sim, args.hal, ids, n, plan, ref,
                            report=True, fp32=args.fp32)
        if cos is None:
            print(f"{n:>7}     ERROR   {detail}")


if __name__ == "__main__":
    main()
