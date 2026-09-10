"""
patterns.py — TVM Relax BYOC subgraph pattern matching for RTV-Accel.

Defines the Relax Dataflow Pattern Language (DPL) patterns that claim
subgraphs executable on the RTV-Accel NPU hardware.

Pipeline:
    Relax IR Module
      -> ConvertToDataflow()
      -> FuseOpsByPattern()        (matches patterns, labels with Codegen="rtv_accel")
      -> MergeCompositeFunctions() (groups matched operations into composite functions)
"""

from typing import List, Tuple, Dict, Any
import tvm
from tvm import relax
from tvm.relax.dpl import pattern as dpl
from tvm.relax.backend.pattern_registry import register_patterns

CODEGEN = "rtv_accel"


# ── Pattern Definitions ───────────────────────────────────────────────────────

def _matmul_pattern() -> Tuple[dpl.DFPattern, Dict[str, dpl.DFPattern]]:
    """A 2-D matrix multiplication: out = x @ w (mapped to COMPUTE_GEMM)."""
    x = dpl.wildcard()
    w = dpl.wildcard()
    out = dpl.is_op("relax.matmul")(x, w)
    return out, {"x": x, "w": w}


def _linear_bias_pattern() -> Tuple[dpl.DFPattern, Dict[str, dpl.DFPattern]]:
    """Linear projection with bias: out = (x @ w) + b."""
    x = dpl.wildcard()
    w = dpl.wildcard()
    b = dpl.wildcard()
    out = dpl.is_op("relax.add")(dpl.is_op("relax.matmul")(x, w), b)
    return out, {"x": x, "w": w, "b": b}


def _elemwise_add_pattern() -> Tuple[dpl.DFPattern, Dict[str, dpl.DFPattern]]:
    """Elementwise addition: out = x + y (mapped to EXEC_ALU ELEMWISE_ADD)."""
    x = dpl.wildcard()
    y = dpl.wildcard()
    out = dpl.is_op("relax.add")(x, y)
    return out, {"x": x, "y": y}


def _elemwise_mul_pattern() -> Tuple[dpl.DFPattern, Dict[str, dpl.DFPattern]]:
    """Elementwise multiplication: out = x * y (mapped to EXEC_ALU ELEMWISE_MUL)."""
    x = dpl.wildcard()
    y = dpl.wildcard()
    out = dpl.is_op("relax.multiply")(x, y)
    return out, {"x": x, "y": y}


def _silu_pattern() -> Tuple[dpl.DFPattern, Dict[str, dpl.DFPattern]]:
    """SiLU non-linear activation: out = silu(x) (mapped to EXEC_ALU LUT)."""
    x = dpl.wildcard()
    out = dpl.is_op("relax.nn.silu")(x)
    return out, {"x": x}


def _swiglu_pattern() -> Tuple[dpl.DFPattern, Dict[str, dpl.DFPattern]]:
    """SwiGLU intermediate: out = silu(gate) * up (mapped to LUT + ELEMWISE_MUL)."""
    gate = dpl.wildcard()
    up = dpl.wildcard()
    out = dpl.is_op("relax.multiply")(dpl.is_op("relax.nn.silu")(gate), up)
    return out, {"gate": gate, "up": up}


def _swiglu_ffn_pattern() -> Tuple[dpl.DFPattern, Dict[str, dpl.DFPattern]]:
    """Full SwiGLU FFN: down_proj(silu(gate_proj(x)) * up_proj(x))."""
    x = dpl.wildcard()
    w_gate = dpl.wildcard()
    w_up = dpl.wildcard()
    w_down = dpl.wildcard()
    gate = dpl.is_op("relax.matmul")(x, w_gate)
    up = dpl.is_op("relax.matmul")(x, w_up)
    silu_gate = dpl.is_op("relax.nn.silu")(gate)
    inter = dpl.is_op("relax.multiply")(silu_gate, up)
    out = dpl.is_op("relax.matmul")(inter, w_down)
    return out, {"x": x, "w_gate": w_gate, "w_up": w_up, "w_down": w_down}


def _rmsnorm_pattern() -> Tuple[dpl.DFPattern, Dict[str, dpl.DFPattern]]:
    """RMSNorm layer: out = rms_norm(x, gamma)."""
    x = dpl.wildcard()
    gamma = dpl.wildcard()
    out = dpl.is_op("relax.nn.rms_norm")(x, gamma)
    return out, {"x": x, "gamma": gamma}


def _softmax_pattern() -> Tuple[dpl.DFPattern, Dict[str, dpl.DFPattern]]:
    """Softmax reduction & normalisation: out = softmax(x)."""
    x = dpl.wildcard()
    out = dpl.is_op("relax.nn.softmax")(x)
    return out, {"x": x}


def _attention_pattern() -> Tuple[dpl.DFPattern, Dict[str, dpl.DFPattern]]:
    """Scaled dot-product attention: softmax(q @ k) @ v."""
    q = dpl.wildcard()
    k = dpl.wildcard()
    v = dpl.wildcard()
    scores = dpl.is_op("relax.matmul")(q, k)
    probs = dpl.is_op("relax.nn.softmax")(scores)
    out = dpl.is_op("relax.matmul")(probs, v)
    return out, {"q": q, "k": k, "v": v}


def _attention_masked_pattern() -> Tuple[dpl.DFPattern, Dict[str, dpl.DFPattern]]:
    """Causal masked attention: softmax(q @ k + mask) @ v."""
    q = dpl.wildcard()
    k = dpl.wildcard()
    mask = dpl.wildcard()
    v = dpl.wildcard()
    scores = dpl.is_op("relax.matmul")(q, k)
    masked = dpl.is_op("relax.add")(scores, mask)
    probs = dpl.is_op("relax.nn.softmax")(masked)
    out = dpl.is_op("relax.matmul")(probs, v)
    return out, {"q": q, "k": k, "mask": mask, "v": v}


# ── Pattern Table & Registration ──────────────────────────────────────────────

def get_patterns() -> List[Tuple[str, dpl.DFPattern, Dict[str, dpl.DFPattern]]]:
    """
    Returns ordered pattern list for FuseOpsByPattern.
    High-level composite patterns (Attention, SwiGLU FFN, RMSNorm) precede
    low-level primitives (Matmul, Add, Mul, SiLU) to prioritize fusion.
    """
    return [
        (f"{CODEGEN}.attention_masked", *_attention_masked_pattern()),
        (f"{CODEGEN}.attention",        *_attention_pattern()),
        (f"{CODEGEN}.swiglu_ffn",       *_swiglu_ffn_pattern()),
        (f"{CODEGEN}.swiglu",           *_swiglu_pattern()),
        (f"{CODEGEN}.rmsnorm",          *_rmsnorm_pattern()),
        (f"{CODEGEN}.linear_bias",      *_linear_bias_pattern()),
        (f"{CODEGEN}.softmax",          *_softmax_pattern()),
        (f"{CODEGEN}.silu",             *_silu_pattern()),
        (f"{CODEGEN}.matmul",           *_matmul_pattern()),
        (f"{CODEGEN}.add",              *_elemwise_add_pattern()),
        (f"{CODEGEN}.mul",              *_elemwise_mul_pattern()),
    ]


def register() -> None:
    """Register RTV-Accel patterns in TVM's global pattern registry."""
    register_patterns(get_patterns())


def partition(mod: tvm.IRModule) -> tvm.IRModule:
    """
    Run TVM's BYOC partitioning on a Relax IRModule:
    1. Convert expressions to dataflow blocks.
    2. Match subgraphs against RTV-Accel pattern table and annotate with Codegen.
    3. Group matched patterns into composite functions.
    """
    seq = tvm.transform.Sequential([
        relax.transform.ConvertToDataflow(),
        relax.transform.FuseOpsByPattern(
            get_patterns(),
            bind_constants=False,
            annotate_codegen=True
        ),
        relax.transform.MergeCompositeFunctions(),
    ])
    return seq(mod)


# ── Inspection Helpers ────────────────────────────────────────────────────────

def _extract_shape_dtype(arg: Any) -> Tuple[List[int], str]:
    """Safely extract shape and dtype across TVM versions."""
    if hasattr(arg, "struct_info") and hasattr(arg.struct_info, "shape"):
        return [int(d) for d in arg.struct_info.shape], str(arg.struct_info.dtype)
    if hasattr(arg, "ty") and hasattr(arg.ty, "shape"):
        return [int(d) for d in arg.ty.shape], str(arg.ty.dtype)
    if hasattr(arg, "shape") and hasattr(arg, "dtype"):
        return [int(d) for d in arg.shape], str(arg.dtype)
    return [], "float32"


def get_offloaded_functions(mod: tvm.IRModule) -> List[Dict[str, Any]]:
    """
    Inspect a partitioned Relax module and collect details on every offloaded
    composite function claimed by the RTV-Accel codegen.
    """
    offloaded = []
    for gv, func in mod.functions.items():
        if not isinstance(func, relax.Function):
            continue
        attrs = getattr(func, "attrs", None)
        if not attrs or attrs.get("Codegen") != CODEGEN:
            continue
        
        name = getattr(gv, "name_hint", getattr(gv, "name", str(gv)))
        params_info = []
        for p in func.params:
            shape, dtype = _extract_shape_dtype(p)
            p_name = getattr(p, "name_hint", getattr(p, "name", str(p)))
            params_info.append({"name": p_name, "shape": shape, "dtype": dtype})
        
        ret = getattr(func, "ret_struct_info", getattr(func, "ret_type", None))
        ret_shape, ret_dtype = _extract_shape_dtype(ret) if ret else ([], "float32")
        offloaded.append({
            "name": name,
            "func": func,
            "params": params_info,
            "return": {"shape": ret_shape, "dtype": ret_dtype},
        })
    return offloaded
