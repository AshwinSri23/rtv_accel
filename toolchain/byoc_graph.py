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


def emit_graph(compiler, mod, inputs, extern=None, fixed_outputs=None, batch_alloc=False):
    """Lower a partitioned module's `main` into `compiler`'s instruction stream.

    `inputs` maps each of main's parameter names to the DRAM address of that tensor.
    `extern` maps an extern-function name prefix (e.g. "rtv.attention_kv") to a lowering
    `fn(compiler, suffix, arg_addrs, out_addr)`; graph ops no pattern can claim (attention
    with heads/RoPE/KV-cache) travel through the graph as `ExternFunc("<prefix>:<suffix>")`
    calls and are lowered by the registered emitter. `fixed_outputs` maps a top-level
    binding INDEX to a caller-chosen DRAM address (e.g. a model's ping-pong layer outputs).

    Output buffers are allocated from the compiler's staging area. With batch_alloc=True,
    a binding's output and those of all following bindings up to the next fixed one are
    allocated TOGETHER, in binding order, before the first of them is emitted -- the same
    order a hand-driven layer function allocates its intermediates, which is what makes the
    graph path's stream byte-identical to it. (Default False: allocate just before each
    composite.)

    Returns (output_address, number_of_ops_emitted).
    """
    infos = {i["name"]: i for i in tvm_backend.offloaded_composites(mod)}
    extern = extern or {}
    fixed_outputs = fixed_outputs or {}
    main = mod["main"]
    # The environment maps Var objects to DRAM addresses. NOT names: after partitioning several
    # distinct dataflow variables share a name (`lv`, `lv1`, `lv`, ...), and a name-keyed map
    # silently hands a later op the wrong buffer.
    env = {}
    for p in main.params:
        name = _name(p)
        if name not in inputs:
            raise KeyError(f"emit_graph: no DRAM address supplied for main parameter {name!r}")
        env[p] = inputs[name]      # keyed by the Var OBJECT, never by name (see below)

    bindings = [b for block in main.body.blocks for b in block.bindings]
    out_addr = {}

    def _bytes(call):
        return compiler.tile_dim * int(call.ty.shape.values[-1]) * 4

    def _alloc_group(start):
        j = start
        while j < len(bindings):
            if j in fixed_outputs:
                out_addr[j] = fixed_outputs[j]
                break
            out_addr[j] = compiler.alloc_staging(_bytes(bindings[j].value))
            j += 1

    n_emitted = 0
    for i, b in enumerate(bindings):
        call = getattr(b, "value", None)
        if not isinstance(call, relax.Call):
            raise NotImplementedError(f"emit_graph: unsupported binding {type(call).__name__}")
        if batch_alloc:
            if i not in out_addr:
                _alloc_group(i)
        else:
            out_addr[i] = fixed_outputs.get(i) or compiler.alloc_staging(_bytes(call))
        args = [env[a] for a in call.args]

        if isinstance(call.op, relax.ExternFunc):
            sym = call.op.global_symbol
            prefix, _, suffix = sym.partition(":")
            if prefix not in extern:
                raise NotImplementedError(f"emit_graph: no lowering registered for extern op {sym!r}")
            extern[prefix](compiler, suffix, args, out_addr[i])
        elif isinstance(call.op, relax.GlobalVar):
            info = infos.get(call.op.name_hint)
            if info is None:
                raise NotImplementedError(f"emit_graph: {call.op.name_hint} is not an offloaded composite")
            kind = info["kind"]
            keys = _ARG_KEYS.get(kind)
            if keys is None or len(keys) != len(args):
                raise NotImplementedError(f"emit_graph: no argument mapping for composite kind {kind!r}"
                                          f" with {len(args)} arguments")
            addrs = dict(zip(keys, args))
            addrs["out"] = out_addr[i]
            tvm_backend.emit_composite(compiler, info, addrs)
        else:
            raise NotImplementedError(
                f"emit_graph: operator {call.op} was not claimed by any RTV-Accel pattern and has "
                "no registered extern lowering")
        env[b.var] = out_addr[i]
        n_emitted += 1

    result = main.body.body
    if result not in env:
        raise NotImplementedError("emit_graph: main must return a single tensor produced above")
    return env[result], n_emitted


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
            elif isinstance(call, relax.Call) and isinstance(call.op, relax.ExternFunc):
                claimed.append("extern:" + call.op.global_symbol.partition(":")[0])
            elif isinstance(call, relax.Call):
                unclaimed.append(getattr(call.op, "name", str(call.op)))
    return claimed, unclaimed
