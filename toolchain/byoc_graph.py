#!/usr/bin/env python3
"""
byoc_graph.py -- drive the chip compiler from a partitioned Relax graph.

The production decoder emits instructions by calling hand-written `_emit_*`
functions that know the model config. This module is the graph-driven
alternative TODO Phase 5 calls for (arbitrary models cannot have an `_emit_*` per
operator written ahead of time): take a Relax module, run TVM's BYOC partitioning
with the RTV-Accel patterns (rtv_tvm/patterns.py), then walk the partitioned
`main` in order, lowering every claimed composite with `tvm_backend.emit_composite`.

    mod = tvm_backend.partition(relax_module)
    out_addr = emit_graph(compiler, mod, {"x": X_ADDR, "w_gate": WG_ADDR, ...})

Scope, stated plainly:
  * Only operators a pattern claims are lowered (rmsnorm, matmul, swiglu_ffn,
    add, mul, attention core). Anything left unclaimed in `main` (reshape,
    permute, RoPE's slice/concat, KV-cache update, ...) raises
    NotImplementedError naming the op, never silently skipping it. Attention with
    heads, RoPE and a KV-cache is NOT discoverable from graph structure
    (patterns.py documents why), so a whole decoder layer still needs the
    config-aware emitters for those parts.
  * Composite arguments are positional: a composite's parameter NAMES come from
    whatever the source graph called its tensors, only their ORDER is fixed per
    pattern kind (`_ARG_KEYS`).
  * Every tensor lives in DRAM as a padded (tile_dim, cols) buffer, row 0 real
    for single-token activations, the convention the emitters share.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "rtv_tvm"))

from tvm import relax  # noqa: E402
import tvm_backend     # noqa: E402

# Positional argument names emit_composite expects, per composite kind.
_ARG_KEYS = {
    "rmsnorm": ("x", "gamma"),
    "swiglu_ffn": ("x", "w_gate", "w_up", "w_down"),
    "matmul": ("x", "w"),
    "linear_bias": ("x", "w", "b"),
    "add": ("x", "y"),
    "mul": ("x", "y"),
    "attention": ("q", "k", "v"),
    "attention_masked": ("q", "k", "v", "mask"),
}


def _name(v):
    return getattr(v, "name_hint", None) or getattr(v, "name", None) or str(v)


def emit_graph(compiler, mod, inputs):
    """Lower a partitioned module's `main` into `compiler`'s instruction stream.

    `inputs` maps each of main's parameter names to the DRAM address of that tensor
    (activations and weights are supplied by the caller, as in emit_composite).
    Returns (output_address, number_of_composites_emitted).
    """
    infos = {i["name"]: i for i in tvm_backend.offloaded_composites(mod)}
    main = mod["main"]
    env = {}
    for p in main.params:
        name = _name(p)
        if name not in inputs:
            raise KeyError(f"emit_graph: no DRAM address supplied for main parameter {name!r}")
        env[name] = inputs[name]

    n_emitted = 0
    body = main.body
    for block in body.blocks:
        for b in block.bindings:
            call = getattr(b, "value", None)
            if not isinstance(call, relax.Call):
                raise NotImplementedError(f"emit_graph: unsupported binding {type(call).__name__}")
            if not isinstance(call.op, relax.GlobalVar):
                raise NotImplementedError(
                    f"emit_graph: operator {call.op} was not claimed by any RTV-Accel pattern "
                    "(it would have to run on the host or have its own emitter)")
            info = infos.get(call.op.name_hint)
            if info is None:
                raise NotImplementedError(f"emit_graph: {call.op.name_hint} is not an offloaded composite")
            kind = info["kind"]
            keys = _ARG_KEYS.get(kind)
            if keys is None or len(keys) != len(call.args):
                raise NotImplementedError(f"emit_graph: no argument mapping for composite kind {kind!r}"
                                          f" with {len(call.args)} arguments")
            addrs = {k: env[_name(a)] for k, a in zip(keys, call.args)}
            cols = int(call.ty.shape.values[-1])         # result width, from the call's own type
            addrs["out"] = compiler.alloc_staging(compiler.tile_dim * cols * 4)
            tvm_backend.emit_composite(compiler, info, addrs)
            env[_name(b.var)] = addrs["out"]
            n_emitted += 1

    result = body.body
    if _name(result) not in env:
        raise NotImplementedError("emit_graph: main must return a single tensor produced above")
    return env[_name(result)], n_emitted


def coverage(mod):
    """(claimed, unclaimed) operator names in a partitioned module's main: how much of
    the graph the patterns cover. `claimed` lists composite kinds, `unclaimed` the raw
    Relax ops left outside any composite."""
    infos = {i["name"]: i for i in tvm_backend.offloaded_composites(mod)}
    claimed, unclaimed = [], []
    for block in mod["main"].body.blocks:
        for b in block.bindings:
            call = getattr(b, "value", None)
            if isinstance(call, relax.Call) and isinstance(call.op, relax.GlobalVar):
                claimed.append(infos[call.op.name_hint]["kind"])
            elif isinstance(call, relax.Call):
                unclaimed.append(getattr(call.op, "name", str(call.op)))
    return claimed, unclaimed
