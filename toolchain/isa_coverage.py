#!/usr/bin/env python3
"""isa_coverage.py -- what one decode layer costs, in ISA instructions, for the dense Qwen3-0.6B and Gemma-4-26B-A4B.

Compiles the real-sized per-layer programs (Qwen3: one dense layer with INT4W g32 weights; Gemma: program A for a
sliding and a full layer plus the expert program B, taken from a run of the real GGUF) and prints, per program, the
instruction count by opcode and by operating mode. Feeds the ISA coverage section of ARCHITECTURE.md.

    .venv-tvm/bin/python toolchain/isa_coverage.py [--gguf <gemma-4-26B_q4_0-it.gguf>] [--pos 20]
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "rtv_tvm"))

from rtv_isa import Opcode, ActivationMode, DType   # noqa: E402
import tvm_backend                                   # noqa: E402
from compiler_state import NPUCompiler, DType as CDType  # noqa: E402


def histogram(stream):
    n = len(stream) // 16
    w = np.frombuffer(bytes(stream), dtype=">u8").reshape(n, 2)
    hi, lo = w[:, 0], w[:, 1]
    opc = ((hi >> np.uint64(60)) & np.uint64(0xF)).astype(int)
    act = (lo & np.uint64(0x7F)).astype(int)
    wire_m = ((hi & np.uint64(0xFF)) << np.uint64(8)) | (lo >> np.uint64(56))
    dtype = ((wire_m >> np.uint64(13)) & np.uint64(7)).astype(int)
    ops = Counter(Opcode(o).name for o in opc)
    modes = Counter()
    for o, a, d in zip(opc, act, dtype):
        if o == Opcode.COMPUTE_GEMM:
            modes[f"GEMM/{DType(d).name}"] += 1
        elif o in (Opcode.EXEC_ALU, Opcode.REDUCE, Opcode.SCALAR_ALU):
            try:
                modes[f"{Opcode(o).name}/{ActivationMode(a).name}"] += 1
            except ValueError:
                modes[f"{Opcode(o).name}/mode{a}"] += 1
    return n, ops, modes


def traffic(stream, is_weight):
    """Bytes moved by LOAD_TILE / STORE_TILE, weights (addresses where is_weight(addr)) vs everything else."""
    n = len(stream) // 16
    w = np.frombuffer(bytes(stream), dtype=">u8").reshape(n, 2)
    hi, lo = w[:, 0], w[:, 1]
    opc = ((hi >> np.uint64(60)) & np.uint64(0xF)).astype(int)
    addr = ((hi >> np.uint64(24)) & np.uint64(0xFFFFFFFF)) | (((lo >> np.uint64(36)) & np.uint64(0xF)) << np.uint64(32))
    rows = (((hi & np.uint64(0xFF)) << np.uint64(8)) | (lo >> np.uint64(56))) & np.uint64(0xFF)
    byt = (rows * ((lo >> np.uint64(40)) & np.uint64(0xFFFF))).astype(np.int64)
    ld, st = opc == Opcode.LOAD_TILE, opc == Opcode.STORE_TILE
    wt = ld & is_weight(addr)
    return dict(weights=int(byt[wt].sum()), other_loads=int(byt[ld & ~wt].sum()), stores=int(byt[st].sum()))


def show(title, stream, is_weight=None):
    n, ops, modes = histogram(stream)
    print(f"\n### {title}: {n:,} instructions")
    print("  by opcode:", ", ".join(f"{k} {v:,}" for k, v in ops.most_common()))
    print("  by mode:  ", ", ".join(f"{k} {v:,}" for k, v in modes.most_common()))
    if is_weight is not None:
        t = traffic(stream, is_weight)
        print("  DRAM traffic:", ", ".join(f"{k} {v / 1e6:.1f} MB" for k, v in t.items()))
    return n


def qwen3_layer(pos, hal, row0=False):
    hidden, inter, heads, kv, hd = 1024, 3072, 16, 8, 128
    c = NPUCompiler(hal)
    c.dtype = CDType.FP32
    c.instruction_stream = bytearray()
    c.int4_group = 32
    c._dram_cursor = 0x40000000
    c.rms_eps = 1e-6
    if row0:
        c.gemm_a_rows = 1
    c.init_staging()
    g = 1 << 24
    tiles = pos // 16 + 1
    mk = lambda base: [[base + h * tiles * 16 * hd * 4 + t * 16 * hd * 4 for t in range(tiles)] for h in range(kv)]  # noqa: E731
    tvm_backend._emit_transformer_layer_kvcache_fp32(
        c, hidden, inter, heads, kv, hd, x_addr=g, gamma1_addr=2 * g, gamma2_addr=3 * g,
        wq_addr=4 * g, wk_addr=5 * g, wv_addr=6 * g, wo_addr=7 * g, wgate_addr=8 * g, wup_addr=9 * g,
        wdown_addr=10 * g, out_addr=11 * g, pos=pos, k_cache_tiles=mk(12 * g), v_cache_tiles=mk(13 * g),
        rope_theta=1e6, q_norm_addr=14 * g, k_norm_addr=15 * g)
    return bytes(c.instruction_stream)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--pos", type=int, default=20)
    ap.add_argument("--hal", default=os.path.join(_HERE, "hal_config.json"))
    a = ap.parse_args()
    tot = {}
    g = 1 << 24
    qw = lambda addr: (addr >= np.uint64(4 * g)) & (addr < np.uint64(11 * g))  # noqa: E731  (wq..wdown regions)
    for r0 in (False, True):
        tot[f"Qwen3 row0={r0}"] = show(f"Qwen3-0.6B dense layer (INT4W g32, pos {a.pos}, "
                                       f"{'one-row' if r0 else 'full-tile'} activation loads)",
                                       qwen3_layer(a.pos, a.hal, r0), qw)
    if a.gguf:
        import gemma4_model
        from gemma4_gguf import load_gemma4_gguf
        stats = {}
        gemma4_model.generate_gemma4(load_gemma4_gguf(a.gguf, 6, load_embed=True), list(range(2, 2 + a.pos + 1)), 1,
                                     os.path.join(ROOT, "build", "rtv_sim"), a.hal, int4_group=32, stats=stats,
                                     image_dir=os.path.expanduser("~/.cache/rtv_accel"))
        for k, v in stats["programs"].items():
            tot[k] = show(f"Gemma-4-26B {k} (pos {a.pos})", v)
        print("\nGemma layer = A + B:", {t: tot[f"A_{t}"] + tot["B_experts"] for t in ("sliding_attention", "full_attention")})


if __name__ == "__main__":
    main()
