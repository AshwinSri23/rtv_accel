#!/usr/bin/env python3
"""
e2e_test.py — end-to-end numeric verification for the NPU compiler.

This is the loop the compiler never had. Every correctness bug found by
auditing npu_compiler.py by hand (missing K-accumulation, wrong LOAD_TILE
stride, colliding c_slots, element-vs-byte store widths, staging addresses
landing inside the weight region, a prefetch DMA racing the GEMM that reads
the slot it refills) produces *visibly wrong numbers on the first run* —
none would have survived a single execution of this script.

Flow, per case:
    1. build known INT8 operands                          (numpy)
    2. lay them out in a flat DRAM image at the addresses
       the compiler will actually emit                    (numpy)
    3. compile the operation                              (NPUCompiler)
    4. execute against that DRAM image                    (rtv_sim)
    5. read stored results back, compare to a reference   (numpy)

Deliberately NOT Qwen3-0.6B dimensions: cases are small enough that a
mismatch can be read by eye and traced to a specific tile.

Usage:
    python3 toolchain/e2e_test.py [--sim build/rtv_sim] [--keep]
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npu_compiler import (  # noqa: E402
    NPUActivations, NPUCompiler, NPUOpcodes, SRAMMap, ScheduleNode,
    _align, _padded,
)

TILE = 16


def ref_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """INT8 x INT8 -> INT32 GEMM, computed independently of the NPU model."""
    return a.astype(np.int32) @ b.astype(np.int32)


def pad2d(x: np.ndarray) -> np.ndarray:
    out = np.zeros((_padded(x.shape[0]), _padded(x.shape[1])), dtype=np.int8)
    out[:x.shape[0], :x.shape[1]] = x
    return out


def n_tiles(x: int) -> int:
    return -(-x // TILE)



SOFTMAX_T = 64.0   # must match SOFTMAX_TEMPERATURE in src/pe_array.cpp


def ref_softmax_rows(scores):
    """Mirror the simulator's INT8 softmax exactly: REDUCE_MAX, EXP_SUB, sum, SCALAR_DIV."""
    out = np.zeros_like(scores, dtype=np.int8)
    for r in range(scores.shape[0]):
        row = scores[r].astype(np.int32)
        masked = row == -128        # MASKED_SENTINEL -> exactly zero probability
        mx = int(row.max())
        e = np.exp((row.astype(np.float32) - mx) / SOFTMAX_T) * 127.0
        e = np.clip(np.trunc(e + np.copysign(0.5, e)), -128, 127).astype(np.int8)
        e[masked] = 0
        tot = int(e.astype(np.int32).sum())
        if tot != 0:
            v = e.astype(np.float32) * 127.0 / float(tot)
            out[r] = np.clip(np.trunc(v + np.copysign(0.5, v)), -128, 127).astype(np.int8)
    return out



def ref_attention(q_i8, k_i8, v_i8, head_dim, n_heads, n_kv_heads, quant,
                  causal=False, cache_len=0):
    """
    Multi-head attention with GQA, mirroring compile_attention_block exactly.

    Head h attends with K/V head h // (n_heads // n_kv_heads), and each head's
    context lands in its own head_dim-wide slice of the output -- the
    concatenation is positional, matching the compiler's addressing.
    """
    m = q_i8.shape[0]
    group = max(1, n_heads // max(1, n_kv_heads))
    ctx = np.zeros((m, n_heads * head_dim), dtype=np.int8)
    for h in range(n_heads):
        kv_h = h // group
        qh = q_i8[:, h * head_dim:(h + 1) * head_dim]
        kh = k_i8[:, kv_h * head_dim:(kv_h + 1) * head_dim]
        vh = v_i8[:, kv_h * head_dim:(kv_h + 1) * head_dim]
        scores = quant(ref_matmul(qh, kh.T))
        if causal:
            # Query row i is at absolute position cache_len+i; mask keys beyond it.
            for i in range(scores.shape[0]):
                scores[i, cache_len + i + 1:] = -128     # MASKED_SENTINEL
        ctx[:, h * head_dim:(h + 1) * head_dim] = quant(
            ref_matmul(ref_softmax_rows(scores), vh))
    return ctx


# ── Case protocol ────────────────────────────────────────────────────────────
#
# build(compiler)   -> populate .dram and .program
# dump_regions()    -> [(label, addr, nbytes)] the simulator should write back
# check(raws)       -> (ok: bool, detail: str)


class RMSNormInt16Case:
    """
    RMSNorm reading the INT16 residual stream and writing Q0.7 INT8.

    This is the boundary where the wide stream narrows. The input holds values
    an INT8 buffer could not (the residual carries Qwen3's massive activation
    channels), and the output is normalised by construction so it needs no
    range -- 1.0 is 127 regardless of what one input step was worth.
    """

    def __init__(self, name, n):
        self.name, self.n = name, n

    def build(self, compiler):
        n = self.n
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        # One massive channel plus ordinary ones -- the real distribution.
        self.x = rng.integers(-3000, 3001, size=(TILE, n)).astype(np.int16)
        self.x[:, 0] = 30000

        ssq = (self.x.astype(np.int64) ** 2).sum()
        C = int(round(127.0 * np.sqrt(n)))
        inv = (np.float32(C) / np.sqrt(np.float32(ssq))) if ssq > 0 else np.float32(0)
        y = self.x.astype(np.float32) * inv
        self.expected = np.clip(np.trunc(y + np.copysign(np.float32(0.5), y)),
                                -128, 127).astype(np.int8)

        self.in_addr, self.out_addr = 0x1000, 0x8000
        self.dram = bytearray(self.out_addr + TILE * _padded(n))
        padded = np.zeros((TILE, _padded(n)), dtype=np.int16)
        padded[:, :n] = self.x
        self.dram[self.in_addr:self.in_addr + padded.nbytes] = padded.tobytes()

        compiler.instruction_stream = bytearray()
        compiler.compile_rmsnorm_row(
            dram_in=self.in_addr, dram_out=self.out_addr, n=n,
            sram_scalar=SRAMMap.SCALAR_SCRATCH_BASE, dtype=1)
        self.program = bytes(compiler.instruction_stream)

    def dump_regions(self):
        return [("y", self.out_addr, TILE * _padded(self.n))]

    def check(self, raws):
        got = np.frombuffer(raws["y"], dtype=np.int8).reshape(
            TILE, _padded(self.n))[:, :self.n]
        return compare(got, self.expected, f"rmsnorm int16 n={self.n}")


class Fp32LinearCase:
    """
    compile_linear with no quantisation at all.

    The point of the FP32 path: values spanning many orders of magnitude in
    the same tensor, which no single INT8 scale can hold, and no scale to
    track because each float carries its own exponent. If this passes, the
    compiler's addressing, tiling and K-accumulation are correct independently
    of every quantisation decision.
    """

    def __init__(self, name, m, n, k):
        self.name, self.m, self.n, self.k = name, m, n, k

    def build(self, compiler):
        m, n, k = self.m, self.n, self.k
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        # Deliberately 1e-4 .. 1e4 in one tensor.
        self.a = (rng.normal(size=(m, k)) *
                  np.logspace(-4, 4, k)[None, :]).astype(np.float32)
        self.b = (rng.normal(size=(k, n)) *
                  np.logspace(4, -4, n)[None, :]).astype(np.float32)
        self.expected = (self.a.astype(np.float64) @
                         self.b.astype(np.float64)).astype(np.float32)

        self.a_addr, self.b_addr, self.c_addr = 0x1000, 0x10_0000, 0x20_0000
        self.dram = bytearray(self.c_addr + _padded(m) * _padded(n) * 4)
        for arr, addr, cols in ((self.a, self.a_addr, k), (self.b, self.b_addr, n)):
            padded = np.zeros((_padded(arr.shape[0]), _padded(cols)), dtype=np.float32)
            padded[:arr.shape[0], :cols] = arr
            self.dram[addr:addr + padded.nbytes] = padded.tobytes()

        compiler.instruction_stream = bytearray()
        compiler.compile_linear(dram_weight=self.b_addr, m=m, n=n, k=k,
                                out_dram=self.c_addr, dram_activation=self.a_addr,
                                dtype=3)
        self.program = bytes(compiler.instruction_stream)

    def dump_regions(self):
        return [("c", self.c_addr, _padded(self.m) * _padded(self.n) * 4)]

    def check(self, raws):
        got = np.frombuffer(raws["c"], dtype=np.float32).reshape(
            _padded(self.m), _padded(self.n))[:self.m, :self.n]
        # Error is measured against the magnitude of the TERMS, not of the
        # result. These inputs span 1e8 by construction, so a dot product can
        # cancel to near zero and any float32 accumulation order leaves a
        # relative error of order 1 on that residue. |a| @ |b| bounds what was
        # actually summed, which is the quantity float32 precision applies to.
        bound = (np.abs(self.a).astype(np.float64) @
                 np.abs(self.b).astype(np.float64))
        rel = np.abs(got - self.expected) / np.maximum(bound, 1e-30)
        if float(np.max(rel)) < 1e-6:
            return True, f"fp32 linear m={self.m} n={self.n} k={self.k}"
        bad = int(np.argmax(rel))
        return False, (f"fp32 linear m={self.m} n={self.n} k={self.k}: "
                       f"max err/|terms| {float(np.max(rel)):.3g} at flat {bad} "
                       f"(got {got.ravel()[bad]:.6g} want {self.expected.ravel()[bad]:.6g})")


class Fp32MlpCase:
    """
    Whole SwiGLU MLP with no quantisation.

    Exercises two GEMMs, a real SiLU, an elementwise multiply and a third
    GEMM, all in float. If this matches, the block's addressing and staging
    are correct independently of any scale -- which is the only way to know
    whether the INT8 version's remaining error is lowering or quantisation.
    """

    def __init__(self, name, hidden, inter):
        self.name, self.hidden, self.inter = name, hidden, inter

    def build(self, compiler):
        h, i = self.hidden, self.inter
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        self.x = rng.normal(size=(TILE, h)).astype(np.float32)
        self.wg = rng.normal(size=(h, i)).astype(np.float32) * 0.1
        self.wu = rng.normal(size=(h, i)).astype(np.float32) * 0.1
        self.wd = rng.normal(size=(i, h)).astype(np.float32) * 0.1

        g = self.x @ self.wg
        u = self.x @ self.wu
        self.expected = ((g / (1.0 + np.exp(-g))) * u) @ self.wd

        self.x_addr, self.g_addr = 0x1000, 0x10_0000
        self.u_addr, self.d_addr = 0x20_0000, 0x30_0000
        self.o_addr = 0x40_0000
        # Room for the block's staging buffers above the operands.
        self.dram = bytearray(self.o_addr + TILE * _padded(h) * 4
                              + 8 * TILE * _padded(i) * 4 + 4096)
        for arr, addr in ((self.x, self.x_addr), (self.wg, self.g_addr),
                          (self.wu, self.u_addr), (self.wd, self.d_addr)):
            pad = np.zeros((_padded(arr.shape[0]), _padded(arr.shape[1])),
                           dtype=np.float32)
            pad[:arr.shape[0], :arr.shape[1]] = arr
            self.dram[addr:addr + pad.nbytes] = pad.tobytes()

        compiler.dtype = 3
        compiler.instruction_stream = bytearray()
        # init_staging bases the staging region on the weight allocator's
        # cursor, which is 0 when no model was ingested -- so without this the
        # block's scratch buffers land on top of this case's own operands.
        compiler._dram_cursor = self.o_addr + TILE * _padded(self.hidden) * 4
        compiler.init_staging()
        compiler.compile_mlp_block(
            dram_x=self.x_addr, dram_gate=self.g_addr, dram_up=self.u_addr,
            dram_down=self.d_addr, dram_out=self.o_addr,
            hidden=h, inter=i)
        self.program = bytes(compiler.instruction_stream)
        compiler.dtype = 0

    def dump_regions(self):
        return [("o", self.o_addr, TILE * _padded(self.hidden) * 4)]

    def check(self, raws):
        got = np.frombuffer(raws["o"], dtype=np.float32).reshape(
            TILE, _padded(self.hidden))[:, :self.hidden]
        bound = np.abs(self.expected).max() or 1.0
        err = float(np.abs(got - self.expected).max()) / bound
        if err < 1e-5:
            return True, f"fp32 mlp h={self.hidden} i={self.inter}"
        return False, (f"fp32 mlp h={self.hidden} i={self.inter}: "
                       f"max err/|out| {err:.3g}")


class Fp32AttentionCase:
    """
    Whole attention block in float: Q/K/V projections, per-head slicing, the
    K transpose, scores, softmax, context, and o_proj.

    This is the part of the compiler with the most byte-vs-element arithmetic
    -- heads are column SLICES addressed inside a wider matrix, so offset and
    pitch scale independently and getting one right while missing the other
    produces overlapping heads rather than an obvious failure.
    """

    def __init__(self, name, hidden, n_heads, n_kv_heads, head_dim):
        self.name, self.hidden = name, hidden
        self.nh, self.nkv, self.hd = n_heads, n_kv_heads, head_dim
        self.q_dim = n_heads * head_dim
        self.kv_dim = n_kv_heads * head_dim

    def build(self, compiler):
        h, nh, nkv, hd = self.hidden, self.nh, self.nkv, self.hd
        q_dim, kv_dim = self.q_dim, self.kv_dim
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        self.x = rng.normal(size=(TILE, h)).astype(np.float32) * 0.5
        self.wq = rng.normal(size=(h, q_dim)).astype(np.float32) * 0.2
        self.wk = rng.normal(size=(h, kv_dim)).astype(np.float32) * 0.2
        self.wv = rng.normal(size=(h, kv_dim)).astype(np.float32) * 0.2
        self.wo = rng.normal(size=(q_dim, h)).astype(np.float32) * 0.2

        q = (self.x @ self.wq).reshape(TILE, nh, hd)
        k = (self.x @ self.wk).reshape(TILE, nkv, hd)
        v = (self.x @ self.wv).reshape(TILE, nkv, hd)
        rep = nh // nkv
        K = np.repeat(k, rep, axis=1)
        V = np.repeat(v, rep, axis=1)
        ctx = np.zeros((TILE, nh, hd), dtype=np.float32)
        for hh in range(nh):
            sc = q[:, hh] @ K[:, hh].T                  # no 1/sqrt(d): the
            e = np.exp(sc - sc.max(axis=-1, keepdims=True))  # block applies none
            p = e / e.sum(axis=-1, keepdims=True)
            ctx[:, hh] = p @ V[:, hh]
        self.expected = ctx.reshape(TILE, q_dim) @ self.wo

        base = 0x1000
        self.addrs = {}
        self.dram = bytearray(0x80_0000)
        for nm, arr in (("x", self.x), ("wq", self.wq), ("wk", self.wk),
                        ("wv", self.wv), ("wo", self.wo)):
            pad = np.zeros((_padded(arr.shape[0]), _padded(arr.shape[1])),
                           dtype=np.float32)
            pad[:arr.shape[0], :arr.shape[1]] = arr
            self.addrs[nm] = base
            self.dram[base:base + pad.nbytes] = pad.tobytes()
            base = _align(base + pad.nbytes, 4096)
        self.o_addr = base
        base = _align(base + TILE * _padded(h) * 4, 4096)

        compiler.dtype = 3
        compiler.instruction_stream = bytearray()
        compiler._dram_cursor = base
        compiler.init_staging()
        compiler.compile_attention_block(
            dram_x=self.addrs["x"], dram_wq=self.addrs["wq"],
            dram_wk=self.addrs["wk"], dram_wv=self.addrs["wv"],
            dram_wo=self.addrs["wo"], dram_out=self.o_addr,
            hidden=h, q_dim=q_dim, kv_dim=kv_dim, head_dim=hd,
            n_heads=nh, n_kv_heads=nkv, seq_len=TILE)
        self.program = bytes(compiler.instruction_stream)
        compiler.dtype = 0
        need = compiler._staging_cursor + 4096
        if len(self.dram) < need:
            self.dram.extend(b"\0" * (need - len(self.dram)))

    def dump_regions(self):
        return [("o", self.o_addr, TILE * _padded(self.hidden) * 4)]

    def check(self, raws):
        got = np.frombuffer(raws["o"], dtype=np.float32).reshape(
            TILE, _padded(self.hidden))[:, :self.hidden]
        bound = float(np.abs(self.expected).max()) or 1.0
        err = float(np.abs(got - self.expected).max()) / bound
        if err < 1e-4:
            return True, f"fp32 attention h={self.hidden} heads={self.nh}/{self.nkv}"
        return False, (f"fp32 attention h={self.hidden} heads={self.nh}/{self.nkv}: "
                       f"max err/|out| {err:.3g}")


class Fp32FullLayerCase:
    """
    A whole transformer layer through compile_model, with no quantisation.

    The point of the FP32 path in one test: if this matches a float reference,
    then every address, tile, stride, staging allocation, K-accumulation and
    schedule in compile_model is correct -- and any remaining error in the
    INT8 pipeline is quantisation, not lowering. That distinction is what the
    whole three-model apparatus exists to make, and until now it could only be
    made in simulation rather than on the device.
    """

    def __init__(self, name, hidden, inter, n_heads, n_kv_heads, head_dim,
                 layers=1):
        self.name, self.hidden, self.inter = name, hidden, inter
        self.nh, self.nkv, self.hd, self.layers = n_heads, n_kv_heads, head_dim, layers
        self.q_dim, self.kv_dim = n_heads * head_dim, n_kv_heads * head_dim

    def build(self, compiler):
        h, i, nh, nkv, hd = self.hidden, self.inter, self.nh, self.nkv, self.hd
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        eps = 1e-6
        P = {}

        def w(shape, s=0.2):
            return (rng.normal(size=shape) * s).astype(np.float32)

        for li in range(self.layers):
            b = f"model.layers.{li}."
            P[b + "input_layernorm.weight"] = np.ones(h, dtype=np.float32)
            P[b + "post_attention_layernorm.weight"] = np.ones(h, dtype=np.float32)
            P[b + "self_attn.q_proj.weight"] = w((self.q_dim, h))
            P[b + "self_attn.k_proj.weight"] = w((self.kv_dim, h))
            P[b + "self_attn.v_proj.weight"] = w((self.kv_dim, h))
            P[b + "self_attn.o_proj.weight"] = w((h, self.q_dim))
            P[b + "mlp.gate_proj.weight"] = w((i, h))
            P[b + "mlp.up_proj.weight"] = w((i, h))
            P[b + "mlp.down_proj.weight"] = w((h, i))

        self.x = rng.normal(size=(TILE, h)).astype(np.float32)

        # Float reference, mirroring compile_model's own sequence: RMSNorm
        # (no gain, so the folding question does not enter), projections,
        # attention with no 1/sqrt(d) and no RoPE, residual, MLP, residual.
        cur = self.x.copy()
        for li in range(self.layers):
            b = f"model.layers.{li}."
            def rms(z):
                # compile_rmsnorm_row accumulates REDUCE_SUMSQ over the WHOLE
                # TILE, so all 16 rows share one RMS rather than each having
                # its own. That is exact for decode, where only row 0 holds
                # data and the rest are zero padding, and is a real
                # approximation for prefill -- documented in the spec, and a
                # separate question from whether the float path is lowered
                # correctly. This reference mirrors the compiler so the test
                # measures lowering rather than re-litigating semantics.
                ssq = float((z.astype(np.float64) ** 2).sum())
                return z * np.sqrt(z.shape[-1] / max(ssq, 1e-30))
            y = rms(cur)
            q = (y @ P[b + "self_attn.q_proj.weight"].T).reshape(TILE, nh, hd)
            k = (y @ P[b + "self_attn.k_proj.weight"].T).reshape(TILE, nkv, hd)
            v = (y @ P[b + "self_attn.v_proj.weight"].T).reshape(TILE, nkv, hd)
            # compile_model always applies RoPE. The sin table carries the
            # rotate_half negation in its first half (see rope_tables), so the
            # rotation here swaps halves WITHOUT negating and lets the table
            # supply the sign -- which is what the device does.
            cos_t, sin_t = NPUCompiler.rope_tables(TILE, hd, 1e6, dtype=3)
            half = hd // 2
            def rope(z):
                sw = np.concatenate([z[..., half:], z[..., :half]], axis=-1)
                return z * cos_t[:, None, :] + sw * sin_t[:, None, :]
            q, k = rope(q), rope(k)
            rep = nh // nkv
            K, V = np.repeat(k, rep, 1), np.repeat(v, rep, 1)
            ctx = np.zeros((TILE, nh, hd), dtype=np.float32)
            for hh in range(nh):
                sc = q[:, hh] @ K[:, hh].T
                e = np.exp(sc - sc.max(-1, keepdims=True))
                ctx[:, hh] = (e / e.sum(-1, keepdims=True)) @ V[:, hh]
            cur = cur + ctx.reshape(TILE, nh * hd) @ P[b + "self_attn.o_proj.weight"].T
            y2 = rms(cur)
            g = y2 @ P[b + "mlp.gate_proj.weight"].T
            u = y2 @ P[b + "mlp.up_proj.weight"].T
            cur = cur + ((g / (1.0 + np.exp(-g))) * u) @ P[b + "mlp.down_proj.weight"].T
        self.expected = cur

        cfg = dict(hidden_size=h, intermediate_size=i, num_layers=self.layers,
                   num_heads=nh, num_kv_heads=nkv, head_dim=hd,
                   vocab_size=32, rms_norm_eps=eps, rope_theta=1e6)
        params = P

        class FakeModel:
            def named_parameters(self_inner):
                return list(params.items())

        compiler.dtype = 3
        compiler.config["model"] = cfg
        compiler.instruction_stream = bytearray()
        compiler.compile_model(FakeModel(), capture_weights=True, seq_len=TILE,
                               causal=False, per_channel=False)
        self.program = bytes(compiler.instruction_stream)
        self.in_addr = compiler.model_input_dram
        self.out_addr = compiler.model_output_dram
        img = os.path.join(os.environ["RTV_E2E_IMG"] + ".d")
        size = compiler.export_dram_image(img)
        self.dram = bytearray(open(img, "rb").read())
        need = self.out_addr + TILE * _padded(h) * 4
        if len(self.dram) < need:
            self.dram.extend(b"\0" * (need - len(self.dram)))
        pad = np.zeros((TILE, _padded(h)), dtype=np.float32)
        pad[:, :h] = self.x
        self.dram[self.in_addr:self.in_addr + pad.nbytes] = pad.tobytes()
        compiler.dtype = 0

    def dump_regions(self):
        return [("y", self.out_addr, TILE * _padded(self.hidden) * 4)]

    def check(self, raws):
        got = np.frombuffer(raws["y"], dtype=np.float32).reshape(
            TILE, _padded(self.hidden))[:, :self.hidden]
        bound = float(np.abs(self.expected).max()) or 1.0
        err = float(np.abs(got - self.expected).max()) / bound
        if err < 1e-3:
            return True, (f"fp32 full layer h={self.hidden} L={self.layers} "
                          f"(max err/|out| {err:.2g})")
        return False, (f"fp32 full layer h={self.hidden} L={self.layers}: "
                       f"max err/|out| {err:.3g}")


class LutCase:
    """
    EXEC_ALU LUT: a nonlinearity evaluated at the true value.

    The device's SILU applies x/(1+exp(-x)) to the raw INT8 code. SiLU's
    curvature lives near |x| < 6, so on integer codes it is indistinguishable
    from ReLU for all 256 inputs -- the SwiGLU gate is a ReLU gate. This case
    checks the table path reproduces real SiLU at a realistic gate scale, and
    that it actually differs from what SILU does today.
    """

    def __init__(self, name, n, in_scale):
        self.name, self.n, self.in_scale = name, n, in_scale

    def build(self, compiler):
        n = self.n
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        self.x = rng.integers(-128, 128, size=(TILE, n)).astype(np.int8)

        silu = lambda z: z / (1.0 + np.exp(-z))
        tbl, self.out_scale = compiler.lut_table(silu, self.in_scale)
        self.table = np.frombuffer(tbl, dtype=np.int8)
        self.expected = self.table[self.x.astype(np.int32) + 128]

        self.x_addr, self.t_addr, self.o_addr = 0x1000, 0x4000, 0x8000
        self.dram = bytearray(self.o_addr + TILE * _padded(n))
        padded = np.zeros((TILE, _padded(n)), dtype=np.int8)
        padded[:, :n] = self.x
        self.dram[self.x_addr:self.x_addr + padded.nbytes] = padded.tobytes()
        self.dram[self.t_addr:self.t_addr + len(tbl)] = tbl

        compiler.instruction_stream = bytearray()
        for c in range(n_tiles(n)):
            off = c * TILE
            compiler._emit(compiler._pass3_load(
                dram_addr=self.x_addr + off, sram_addr=SRAMMap.input_slot(0),
                dim_m=TILE, dim_n=TILE, dep_id=8, stride=_padded(n)))
            compiler._emit(compiler._pass3_wait(8))
            compiler.compile_lut(sram_addr=SRAMMap.input_slot(0),
                                 dim_m=TILE, dim_n=TILE, dram_table=self.t_addr)
            compiler._emit(compiler._pass3_store(
                sram_addr=SRAMMap.input_slot(0), dram_addr=self.o_addr + off,
                dim_m=TILE, dim_n=TILE, stride=_padded(n)))
        self.program = bytes(compiler.instruction_stream)

    def dump_regions(self):
        return [("o", self.o_addr, TILE * _padded(self.n))]

    def check(self, raws):
        got = np.frombuffer(raws["o"], dtype=np.int8).reshape(
            TILE, _padded(self.n))[:, :self.n]
        ok, detail = compare(got, self.expected, f"lut n={self.n}")
        if ok:
            # The table must not merely reproduce ReLU-on-codes, or it would
            # be no improvement over the SILU it replaces.
            relu = np.clip(self.x.astype(np.int32), 0, 127)
            same = int((got.astype(np.int32) == relu).sum())
            if same == got.size:
                return False, f"lut n={self.n}: table is identical to ReLU on codes"
        return ok, detail


class ScaledResidualCase:
    """
    Residual add with an INT16 stream and a scale-aligned operand.

    This is the bug the RESCALE instruction exists for. `a` and `b` are stored
    with different scales -- one step of `a` is worth s_a, one step of `b` is
    worth s_b -- and adding the raw codes computes a*s_a + b*s_a, weighting b
    by s_a/s_b. The reference below models the correct arithmetic: convert b
    into a's scale, then add.
    """

    def __init__(self, name, n, s_a, s_b, b_int8=False, widen_shift=0,
                 s_out=None):
        self.name, self.n = name, n
        self.s_a, self.s_b = s_a, s_b
        self.b_int8 = b_int8   # B arrives INT8, as a sublayer output does
        self.widen_shift = widen_shift
        # s_out != s_a is the layer boundary: the stream changes scale, so A
        # is converted as well as B.
        self.s_out = s_out if s_out is not None else s_a

    def build(self, compiler):
        n = self.n
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        # INT16 codes: values an INT8 residual could not hold.
        self.a = rng.integers(-20000, 20001, size=(TILE, n)).astype(np.int16)
        if self.b_int8:
            self.b = rng.integers(-127, 128, size=(TILE, n)).astype(np.int8)
        else:
            self.b = rng.integers(-20000, 20001, size=(TILE, n)).astype(np.int16)
        # The widen absorbs 2**widen_shift, so the RESCALE only has to carry
        # what is left -- which is how a ratio of 348 fits an 8-bit mantissa.
        ratio = (self.s_b / self.s_out) / (2 ** self.widen_shift)
        mant, shift = NPUCompiler.encode_ratio(ratio)
        self.rescale = (mant, shift)
        self.rescale_a = (None if self.s_out == self.s_a
                          else NPUCompiler.encode_ratio(self.s_a / self.s_out))

        a_codes = self.a.astype(np.int64)
        if self.rescale_a is not None:
            ma, sa = self.rescale_a
            a_codes = np.clip((a_codes * ma) >> sa, -32768, 32767)
        widened = np.clip(self.b.astype(np.int64) << self.widen_shift,
                          -32768, 32767)
        b_conv = np.clip((widened * mant) >> shift, -32768, 32767)
        self.expected = np.clip(a_codes + b_conv, -32768, 32767).astype(np.int16)

        pitch = _padded(n) * 2
        self.a_addr, self.b_addr, self.o_addr = 0x1000, 0x8000, 0x10000
        self.dram = bytearray(self.o_addr + TILE * pitch)
        for arr, addr, dt in ((self.a, self.a_addr, np.int16),
                              (self.b, self.b_addr,
                               np.int8 if self.b_int8 else np.int16)):
            padded = np.zeros((TILE, _padded(n)), dtype=dt)
            padded[:, :n] = arr
            self.dram[addr:addr + padded.nbytes] = padded.tobytes()

        compiler.instruction_stream = bytearray()
        compiler.compile_residual_add(
            dram_a=self.a_addr, dram_b=self.b_addr, dram_out=self.o_addr,
            n=n, dtype=1, rescale_b=self.rescale,
            b_dtype=0 if self.b_int8 else 1, widen_shift=self.widen_shift,
            rescale_a=self.rescale_a)
        self.program = bytes(compiler.instruction_stream)

    def dump_regions(self):
        return [("o", self.o_addr, TILE * _padded(self.n) * 2)]

    def check(self, raws):
        got = np.frombuffer(raws["o"], dtype=np.int16).reshape(
            TILE, _padded(self.n))[:, :self.n]
        return compare(got, self.expected,
                       f"scaled residual n={self.n} ratio={self.s_b/self.s_a:.3f}")


class PerChannelLinearCase:
    """
    compile_linear with a per-output-channel correction (EXEC_ALU PERCHAN_QUANT).

    The columns of B are deliberately scaled across three orders of magnitude,
    which is the situation per-tensor INT8 cannot represent -- one shared scale
    either saturates the large channels or flushes the small ones to zero. The
    reference models exactly what the device does: quantise each column to its
    own scale, then rescale by mant/2^shift.
    """

    def __init__(self, name, m, n, k):
        self.name, self.m, self.n, self.k = name, m, n, k

    def build(self, compiler):
        m, n, k = self.m, self.n, self.k
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        self.a = rng.integers(-4, 5, size=(m, k), dtype=np.int8)
        # Float weights whose columns span 1000x, then per-channel quantised.
        scale = np.logspace(0, -3, n).astype(np.float32)
        wf = rng.normal(size=(k, n)).astype(np.float32) * scale[None, :]
        qb, tab, _peaks = NPUCompiler._quantize_per_channel(wf, k, n)
        self.b = qb[:k, :n]
        self.tab = tab

        acc = self.a.astype(np.int32) @ self.b.astype(np.int32)
        mant = (tab[:n] & 0xFF).astype(np.int64)
        shift = (tab[:n] >> 8).astype(np.int64)
        v = acc.astype(np.int64) * mant[None, :] >> shift[None, :]
        self.expected = np.clip(v, -128, 127).astype(np.int8)

        self.a_addr, self.b_addr = 0x1000, 0x10_0000
        self.t_addr, self.c_addr = 0x18_0000, 0x20_0000
        self.dram = bytearray(self.c_addr + _padded(m) * _padded(n) * 4)
        self.dram[self.a_addr:self.a_addr + pad2d(self.a).nbytes] = pad2d(self.a).tobytes()
        bp = qb
        self.dram[self.b_addr:self.b_addr + bp.nbytes] = bp.tobytes()
        tb = tab.astype("<u2").tobytes()
        self.dram[self.t_addr:self.t_addr + len(tb)] = tb

        compiler.instruction_stream = bytearray()
        compiler.compile_linear(dram_weight=self.b_addr, m=m, n=n, k=k,
                                out_dram=self.c_addr, dram_activation=self.a_addr,
                                dram_corr=self.t_addr)
        self.program = bytes(compiler.instruction_stream)

    def dump_regions(self):
        return [("c", self.c_addr, _padded(self.m) * _padded(self.n))]

    def check(self, raws):
        got = np.frombuffer(raws["c"], dtype=np.int8).reshape(
            _padded(self.m), _padded(self.n))[:self.m, :self.n]
        return compare(got, self.expected,
                       f"per-channel linear m={self.m} n={self.n} k={self.k}")


class LinearCase:
    """One compile_linear call: C = A @ B."""

    def __init__(self, name, m, n, k, c_slot=0):
        self.name, self.m, self.n, self.k = name, m, n, k
        # c_slot != 0 is what actually exercises SRAMMap.output_slot()'s byte
        # spacing agreeing with simulator.cpp's c_slot_addr(). At slot 0 both
        # the old 256-byte and correct 1024-byte spacing give OUTPUT_BASE, so
        # a mismatch is invisible there.
        self.c_slot = c_slot

    def build(self, compiler):
        m, n, k = self.m, self.n, self.k
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        self.a = rng.integers(-4, 5, size=(m, k), dtype=np.int8)
        self.b = rng.integers(-4, 5, size=(k, n), dtype=np.int8)
        self.expected = ref_matmul(self.a, self.b)

        self.a_addr, self.b_addr, self.c_addr = 0x1000, 0x10_0000, 0x20_0000
        self.dram = bytearray(self.c_addr + _padded(m) * _padded(n) * 4)
        ap, bp = pad2d(self.a), pad2d(self.b)
        self.dram[self.a_addr:self.a_addr + ap.nbytes] = ap.tobytes()
        self.dram[self.b_addr:self.b_addr + bp.nbytes] = bp.tobytes()

        compiler.instruction_stream = bytearray()
        if n_tiles(k) == 1:
            # Single K-tile: A stays resident, so the caller loads it once
            # (compile_linear's default contract).
            compiler._emit(compiler._pass3_load(
                dram_addr=self.a_addr, sram_addr=SRAMMap.input_slot(0),
                dim_m=TILE, dim_n=TILE, dep_id=8, stride=_padded(k)))
            compiler._emit(compiler._pass3_wait(8))
            compiler.compile_linear(dram_weight=self.b_addr, m=m, n=n, k=k,
                                    out_dram=self.c_addr, c_slot=self.c_slot)
        else:
            compiler.compile_linear(dram_weight=self.b_addr, m=m, n=n, k=k,
                                    out_dram=self.c_addr, c_slot=self.c_slot,
                                    dram_activation=self.a_addr)
        self.program = bytes(compiler.instruction_stream)

    def dump_regions(self):
        return [("c", self.c_addr, _padded(self.m) * _padded(self.n) * 4)]

    def check(self, raws):
        full = np.frombuffer(raws["c"], dtype=np.int32).reshape(
            _padded(self.m), _padded(self.n))
        actual = full[:self.m, :self.n]
        return compare(actual, self.expected, f"{self.m}x{self.k} @ {self.k}x{self.n}")


class SiblingQKVCase:
    """
    compile_sibling_qkv: three projections sharing one input tile.

    Verifies the b_sram fix (each sibling must read the weight slot its own
    LOAD_TILE filled — K and V previously computed against Q's weights) and
    that the three results land in *distinct* c_slots rather than
    overwriting each other.

    compile_sibling_qkv emits no stores of its own, so this emits one per
    sibling. n is kept to a single N-tile because the function accumulates
    across K-steps in SRAM and only the final (r, c) tile survives the call.
    """

    def __init__(self, name, m, n, k):
        self.name, self.m, self.n, self.k = name, m, n, k

    def build(self, compiler):
        m, n, k = self.m, self.n, self.k
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        self.x = rng.integers(-4, 5, size=(m, k), dtype=np.int8)
        self.w = [rng.integers(-4, 5, size=(k, n), dtype=np.int8) for _ in range(3)]
        self.expected = [ref_matmul(self.x, w) for w in self.w]

        self.x_addr = 0x1000
        self.w_addr = [0x10_0000, 0x20_0000, 0x30_0000]
        self.out_addr = [0x40_0000, 0x41_0000, 0x42_0000]

        out_bytes = _padded(m) * _padded(n) * 4
        self.dram = bytearray(self.out_addr[-1] + out_bytes)
        xp = pad2d(self.x)
        self.dram[self.x_addr:self.x_addr + xp.nbytes] = xp.tobytes()
        for addr, w in zip(self.w_addr, self.w):
            wp = pad2d(w)
            self.dram[addr:addr + wp.nbytes] = wp.tobytes()

        compiler.instruction_stream = bytearray()
        compiler.compile_sibling_qkv(
            dram_x=self.x_addr,
            dram_wq=self.w_addr[0], dram_wk=self.w_addr[1], dram_wv=self.w_addr[2],
            dram_out=0, m=m, n=n, k=k)
        # Each sibling accumulated into its own c_slot; store all three.
        for slot, addr in enumerate(self.out_addr):
            compiler._emit(compiler._pass3_store(
                sram_addr=SRAMMap.output_slot(slot), dram_addr=addr,
                dim_m=TILE, dim_n=TILE * 4, stride=_padded(n) * 4))
        self.program = bytes(compiler.instruction_stream)

    def dump_regions(self):
        nb = _padded(self.m) * _padded(self.n) * 4
        return [(f"o{i}", a, nb) for i, a in enumerate(self.out_addr)]

    def check(self, raws):
        for i, exp in enumerate(self.expected):
            full = np.frombuffer(raws[f"o{i}"], dtype=np.int32).reshape(
                _padded(self.m), _padded(self.n))
            ok, detail = compare(full[:self.m, :self.n], exp,
                                 f"sibling {'QKV'[i]}")
            if not ok:
                return False, detail
        return True, f"3 siblings, {self.m}x{self.k} @ {self.k}x{self.n}"


class ReduceCase:
    """
    REDUCE reading its tile from an address other than SRAM_INPUT_BASE.

    The simulator previously hardcoded REDUCE's tile source to
    SRAM_INPUT_BASE regardless of the instruction, so this only passes if
    the tile source is genuinely carried in the stride field.
    """

    def __init__(self, name, tile_sram):
        self.name, self.tile_sram = name, tile_sram

    def build(self, compiler):
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        self.tile = rng.integers(-6, 7, size=(TILE, TILE), dtype=np.int8)
        self.expected = np.int32(self.tile.astype(np.int32).sum())

        self.t_addr, self.out_addr = 0x1000, 0x10_0000
        self.dram = bytearray(self.out_addr + 64)
        self.dram[self.t_addr:self.t_addr + self.tile.nbytes] = self.tile.tobytes()

        scalar_sram = SRAMMap.SCALAR_SCRATCH_BASE
        compiler.instruction_stream = bytearray()
        # Land the tile at a NON-zero SRAM address so a hardcoded
        # SRAM_INPUT_BASE source would read the wrong bytes.
        compiler._emit(compiler._pass3_load(
            dram_addr=self.t_addr, sram_addr=self.tile_sram,
            dim_m=TILE, dim_n=TILE, dep_id=9, stride=TILE))
        compiler._emit(compiler._pass3_wait(9))
        compiler._emit(ScheduleNode(
            opcode=NPUOpcodes.REDUCE, dep_id=0, dram_addr=0,
            sram_addr=scalar_sram, dim_m=TILE, dim_n=TILE,
            stride=self.tile_sram))
        compiler._emit(compiler._pass3_store(
            sram_addr=scalar_sram, dram_addr=self.out_addr,
            dim_m=1, dim_n=4, stride=4))
        self.program = bytes(compiler.instruction_stream)

    def dump_regions(self):
        return [("s", self.out_addr, 4)]

    def check(self, raws):
        actual = np.frombuffer(raws["s"], dtype=np.int32)[0]
        if actual == self.expected:
            return True, f"sum={actual} from sram 0x{self.tile_sram:04X}"
        return False, (f"REDUCE from sram 0x{self.tile_sram:04X}: "
                       f"got {actual}, want {self.expected}")


class ChainedLinearCase:
    """
    Two projections where the second reads the first's result.

    Exercises a_sram_slot / dram_activation chaining: the first result is
    staged to DRAM and re-read as the second's activation, which is the only
    workable chaining path (a full projection result is far too large to stay
    resident in SRAM at real dimensions).
    """

    def __init__(self, name, m, n1, k1, n2):
        self.name, self.m, self.n1, self.k1, self.n2 = name, m, n1, k1, n2

    def build(self, compiler):
        m, n1, k1, n2 = self.m, self.n1, self.k1, self.n2
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        # Magnitudes chosen so the >>7 requantization leaves meaningful
        # non-zero values rather than flushing everything to 0.
        self.a  = rng.integers(-8, 8, size=(m, k1), dtype=np.int8)
        self.w1 = rng.integers(-8, 8, size=(k1, n1), dtype=np.int8)
        self.w2 = rng.integers(-3, 4, size=(n1, n2), dtype=np.int8)

        # The first projection ends with INT8_QUANT, so its INT32 accumulator
        # is narrowed in place: clamp(acc >> 7) — PEArray's INT8_QUANT with
        # the v1 default shift. The chain is only comparable if the reference
        # models that same narrowing.
        mid = ref_matmul(self.a, self.w1)
        self.mid_i8 = np.clip(mid >> 7, -128, 127).astype(np.int8)
        self.expected = ref_matmul(self.mid_i8, self.w2)

        self.a_addr   = 0x1000
        self.w1_addr  = 0x10_0000
        self.w2_addr  = 0x20_0000
        self.mid_addr = 0x30_0000
        self.out_addr = 0x40_0000
        self.dram = bytearray(self.out_addr + _padded(m) * _padded(n2) * 4)
        for addr, arr in ((self.a_addr, self.a), (self.w1_addr, self.w1),
                          (self.w2_addr, self.w2)):
            p = pad2d(arr)
            self.dram[addr:addr + p.nbytes] = p.tobytes()

        compiler.instruction_stream = bytearray()
        compiler._emit(compiler._pass3_load(
            dram_addr=self.a_addr, sram_addr=SRAMMap.input_slot(0),
            dim_m=TILE, dim_n=TILE, dep_id=8, stride=_padded(k1)))
        compiler._emit(compiler._pass3_wait(8))
        compiler.compile_linear(dram_weight=self.w1_addr, m=m, n=n1, k=k1,
                                out_dram=self.mid_addr, c_slot=0,
                                activation=NPUActivations.INT8_QUANT)
        # Second projection reads the staged intermediate as its activation.
        # The intermediate is INT8 (requantized above), so it loads at INT8
        # widths -- if compile_linear had stored it at INT32 width this
        # would read the wrong bytes.
        compiler._emit(compiler._pass3_load(
            dram_addr=self.mid_addr, sram_addr=SRAMMap.input_slot(0),
            dim_m=TILE, dim_n=TILE, dep_id=8, stride=_padded(n1)))
        compiler._emit(compiler._pass3_wait(8))
        compiler.compile_linear(dram_weight=self.w2_addr, m=m, n=n2, k=n1,
                                out_dram=self.out_addr, c_slot=1)
        self.program = bytes(compiler.instruction_stream)

    def dump_regions(self):
        return [("c", self.out_addr, _padded(self.m) * _padded(self.n2) * 4)]

    def check(self, raws):
        full = np.frombuffer(raws["c"], dtype=np.int32).reshape(
            _padded(self.m), _padded(self.n2))
        return compare(full[:self.m, :self.n2], self.expected, "chained")



class MLPBlockCase:
    """
    compile_mlp_block: SwiGLU = (SiLU(x@Wg) * (x@Wu)) @ Wd.

    Models the simulator's exact integer semantics:
      INT8_QUANT   -> clamp(acc >> 7)
      SILU         -> clamp(round_half_away(x / (1 + exp(-x))))
      ELEMWISE_MUL -> clamp((a * b) >> 7)
    Getting any of those wrong shows up as a mismatch, so this pins the
    element-wise stage as tightly as it pins the GEMMs.
    """

    def __init__(self, name, m, hidden, inter):
        self.name, self.m, self.hidden, self.inter = name, m, hidden, inter

    def build(self, compiler):
        m, h, i = self.m, self.hidden, self.inter
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        self.x  = rng.integers(-8, 8, size=(m, h), dtype=np.int8)
        self.wg = rng.integers(-8, 8, size=(h, i), dtype=np.int8)
        self.wu = rng.integers(-8, 8, size=(h, i), dtype=np.int8)
        self.wd = rng.integers(-3, 4, size=(i, h), dtype=np.int8)

        q = lambda acc: np.clip(acc >> 7, -128, 127).astype(np.int8)
        gate = q(ref_matmul(self.x, self.wg))
        up   = q(ref_matmul(self.x, self.wu))

        f = gate.astype(np.float32)
        silu = f / (1.0 + np.exp(-f))
        # std::roundf rounds half away from zero; np.round is banker's.
        silu = np.trunc(silu + np.copysign(0.5, silu))
        silu = np.clip(silu, -128, 127).astype(np.int8)

        act = np.clip((silu.astype(np.int32) * up.astype(np.int32)) >> 7,
                      -128, 127).astype(np.int8)
        self.expected = ref_matmul(act, self.wd)

        self.x_addr, self.wg_addr = 0x1000, 0x10_0000
        self.wu_addr, self.wd_addr = 0x20_0000, 0x30_0000
        self.out_addr = 0x40_0000
        self.dram = bytearray(0x80_0000)
        for addr, arr in ((self.x_addr, self.x), (self.wg_addr, self.wg),
                          (self.wu_addr, self.wu), (self.wd_addr, self.wd)):
            pa = pad2d(arr)
            self.dram[addr:addr + pa.nbytes] = pa.tobytes()

        compiler.instruction_stream = bytearray()
        # Staging must not overlap the operands laid out above.
        compiler._staging_base = compiler._staging_cursor = 0x50_0000
        compiler.compile_mlp_block(
            dram_x=self.x_addr, dram_gate=self.wg_addr, dram_up=self.wu_addr,
            dram_down=self.wd_addr, dram_out=self.out_addr,
            hidden=h, inter=i)
        self.program = bytes(compiler.instruction_stream)

    def dump_regions(self):
        return [("c", self.out_addr, _padded(self.m) * _padded(self.hidden) * 4)]

    def check(self, raws):
        full = np.frombuffer(raws["c"], dtype=np.int32).reshape(
            _padded(self.m), _padded(self.hidden))
        return compare(full[:self.m, :self.hidden], self.expected,
                       f"mlp h={self.hidden} i={self.inter}")



class AttentionBlockCase:
    """
    compile_attention_block's projections.

    Checks all four: Q/K/V staged correctly (K and V must use their OWN
    weights -- the previous version computed V twice and discarded K), and
    the output projection reading Q's staged result rather than whatever
    happened to be in input_slot(0).

    Full attention: softmax(Q·K^T)·V, optionally with RoPE applied to Q and K.
    """

    def __init__(self, name, m, hidden, q_dim, kv_dim,
                 head_dim=None, n_heads=1, n_kv_heads=1, rope=False):
        self.name = name
        self.m, self.hidden, self.q_dim, self.kv_dim = m, hidden, q_dim, kv_dim
        self.head_dim = head_dim or (q_dim // n_heads)
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.rope = rope

    def build(self, compiler):
        m, h, qd, kvd = self.m, self.hidden, self.q_dim, self.kv_dim
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        self.x  = rng.integers(-8, 8, size=(m, h), dtype=np.int8)
        self.wq = rng.integers(-8, 8, size=(h, qd), dtype=np.int8)
        self.wk = rng.integers(-8, 8, size=(h, kvd), dtype=np.int8)
        self.wv = rng.integers(-8, 8, size=(h, kvd), dtype=np.int8)
        self.wo = rng.integers(-3, 4, size=(qd, h), dtype=np.int8)

        q = lambda acc: np.clip(acc >> 7, -128, 127).astype(np.int8)
        self.q_i8 = q(ref_matmul(self.x, self.wq))
        self.k_i8 = q(ref_matmul(self.x, self.wk))
        self.v_i8 = q(ref_matmul(self.x, self.wv))
        self.cos = self.sin = None
        qr, kr = self.q_i8, self.k_i8
        if self.rope:
            # RoPE is applied per head, to Q and K only.
            cos, sin = NPUCompiler.rope_tables(self.m, self.head_dim)
            qr, kr = qr.copy(), kr.copy()
            for hh in range(self.n_heads):
                sl = slice(hh * self.head_dim, (hh + 1) * self.head_dim)
                qr[:, sl] = ref_rope(self.q_i8[:, sl], cos, sin)
            for kh in range(self.n_kv_heads):
                sl = slice(kh * self.head_dim, (kh + 1) * self.head_dim)
                kr[:, sl] = ref_rope(self.k_i8[:, sl], cos, sin)
            self.cos, self.sin = cos, sin
        ctx = ref_attention(qr, kr, self.v_i8,
                            self.head_dim, self.n_heads, self.n_kv_heads, q)
        self.expected = ref_matmul(ctx, self.wo)

        self.x_addr  = 0x1000
        self.wq_addr, self.wk_addr = 0x10_0000, 0x20_0000
        self.wv_addr, self.wo_addr = 0x30_0000, 0x40_0000
        self.out_addr = 0x50_0000
        self.cos_addr, self.sin_addr = 0x60_0000, 0x70_0000
        self.dram = bytearray(0xC0_0000)
        tables = ((self.cos_addr, self.cos), (self.sin_addr, self.sin)) if self.rope else ()
        for addr, arr in ((self.x_addr, self.x), (self.wq_addr, self.wq),
                          (self.wk_addr, self.wk), (self.wv_addr, self.wv),
                          (self.wo_addr, self.wo)) + tables:
            pa = pad2d(arr)
            self.dram[addr:addr + pa.nbytes] = pa.tobytes()

        compiler.instruction_stream = bytearray()
        compiler._staging_base = compiler._staging_cursor = 0x60_0000
        compiler.compile_attention_block(
            dram_x=self.x_addr, dram_wq=self.wq_addr, dram_wk=self.wk_addr,
            dram_wv=self.wv_addr, dram_wo=self.wo_addr, dram_out=self.out_addr,
            hidden=h, q_dim=qd, kv_dim=kvd, head_dim=self.head_dim,
            n_heads=self.n_heads, n_kv_heads=self.n_kv_heads,
            dram_cos=(self.cos_addr if self.rope else 0),
            dram_sin=(self.sin_addr if self.rope else 0))
        self.k_stage, self.v_stage = compiler.attn_kv_stage
        self.program = bytes(compiler.instruction_stream)

    def dump_regions(self):
        kv_bytes = _padded(self.m) * _padded(self.kv_dim)
        return [
            ("c", self.out_addr, _padded(self.m) * _padded(self.hidden) * 4),
            ("k", self.k_stage,  kv_bytes),
            ("v", self.v_stage,  kv_bytes),
        ]

    def check(self, raws):
        # K and V must each reflect their own weight matrix. The old bug
        # computed V twice and dropped K, so comparing both catches it.
        for label, exp in (("k", self.k_i8), ("v", self.v_i8)):
            got = np.frombuffer(raws[label], dtype=np.int8).reshape(
                _padded(self.m), _padded(self.kv_dim))[:self.m, :self.kv_dim]
            ok, detail = compare(got, exp, f"staged {label.upper()}")
            if not ok:
                return False, detail
        full = np.frombuffer(raws["c"], dtype=np.int32).reshape(
            _padded(self.m), _padded(self.hidden))
        return compare(full[:self.m, :self.hidden], self.expected,
                       f"attn h={self.hidden} q={self.q_dim} kv={self.kv_dim}")



class RMSNormCase:
    """
    compile_rmsnorm_block: y = x / sqrt(sum(x^2)).

    Models the simulator's exact integer path:
      SQUARE           -> clamp8(x*x)            (saturates past |x|=11)
      REDUCE           -> int32 sum of those
      SCALAR_ALU RSQRT -> clamp8(round(x / sqrt(sum)))

    Catches every failure mode the old sequence had: squaring in place (which
    normalised x^2 instead of x), broadcasting the scalar with EXEC_ALU (which
    read 256 bytes starting at a 4-byte scalar), and the Q0.15-vs->>7 scale
    mismatch.
    """

    def __init__(self, name, scale=1):
        self.name, self.scale = name, scale

    def build(self, compiler):
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        self.x = (rng.integers(-6, 7, size=(TILE, TILE)) * self.scale
                  ).clip(-128, 127).astype(np.int8)

        sq = np.clip(self.x.astype(np.int32) ** 2, -128, 127).astype(np.int8)
        total = np.int32(sq.astype(np.int32).sum())
        inv = (1.0 / np.sqrt(float(total))) if total > 0 else 0.0
        scaled = self.x.astype(np.float32) * inv
        # std::roundf: half away from zero.
        self.expected = np.clip(
            np.trunc(scaled + np.copysign(0.5, scaled)), -128, 127).astype(np.int8)

        self.x_addr, self.out_addr = 0x1000, 0x10_0000
        self.dram = bytearray(self.out_addr + 4096)
        self.dram[self.x_addr:self.x_addr + self.x.nbytes] = self.x.tobytes()

        tile_sram, sq_sram = SRAMMap.input_slot(0), SRAMMap.input_slot(3)
        compiler.instruction_stream = bytearray()
        for sram, dep in ((tile_sram, 6), (sq_sram, 7)):
            compiler._emit(compiler._pass3_load(
                dram_addr=self.x_addr, sram_addr=sram,
                dim_m=TILE, dim_n=TILE, dep_id=dep, stride=TILE))
            compiler._emit(compiler._pass3_wait(dep))
        compiler.compile_rmsnorm_block(
            sram_tile=tile_sram, sram_scalar=SRAMMap.SCALAR_SCRATCH_BASE,
            sram_square=sq_sram)
        compiler._emit(compiler._pass3_store(
            sram_addr=tile_sram, dram_addr=self.out_addr,
            dim_m=TILE, dim_n=TILE, stride=TILE))
        self.program = bytes(compiler.instruction_stream)

    def dump_regions(self):
        return [("y", self.out_addr, TILE * TILE)]

    def check(self, raws):
        got = np.frombuffer(raws["y"], dtype=np.int8).reshape(TILE, TILE)
        return compare(got, self.expected, f"rmsnorm scale={self.scale}")



class FullLayerCase:
    """
    compile_model end to end: real weights in DRAM, a whole transformer layer
    executed, result compared against a reference.

    This is what weight loading unlocked — compile_model was previously only
    checkable structurally (does it emit a well-formed stream?), never
    numerically. It exercises the three staging-address fixes (staging past
    the weights, attention and MLP outputs not colliding, scalar scratch clear
    of the output slots) by actually running them.

    compile_model now uses compile_rmsnorm_row, so hidden is no longer capped
    at one tile -- the h=48 case below spans three and would fail under the
    old per-tile normalisation.
    """

    def __init__(self, name, hidden=16, inter=32, q_dim=16, kv_dim=16, layers=1,
                 head_dim=None, n_heads=1, n_kv_heads=1):
        self.name = name
        self.hidden, self.inter = hidden, inter
        self.q_dim, self.kv_dim, self.layers = q_dim, kv_dim, layers
        self.head_dim = head_dim or (q_dim // n_heads)
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.m_rows = TILE

    # -- reference helpers, mirroring the simulator's integer semantics -------
    @staticmethod
    def _quant(acc):
        return np.clip(acc >> 7, -128, 127).astype(np.int8)

    @staticmethod
    def _rmsnorm(x):
        # Row-wide RMS: sum of squares in INT32, divided by the MEAN, output
        # in Q0.7 (1.0 -> 127). See compile_rmsnorm_row.
        n = x.shape[1]
        total = np.int32((x.astype(np.int32) ** 2).sum())
        C = int(round(127.0 * np.sqrt(n)))
        # float32 throughout: PEArray computes inv in float, and at these
        # magnitudes float64 lands on a different integer when the product
        # falls near .5 -- a divergence that compounds across layers.
        inv = (np.float32(C) / np.sqrt(np.float32(total))) if total > 0 else np.float32(0)
        y = x.astype(np.float32) * inv
        return np.clip(np.trunc(y + np.copysign(np.float32(0.5), y)),
                       -128, 127).astype(np.int8)

    @staticmethod
    def _silu(x):
        f = x.astype(np.float32)
        s = f / (1.0 + np.exp(-f))
        return np.clip(np.trunc(s + np.copysign(0.5, s)), -128, 127).astype(np.int8)

    def build(self, compiler):
        h, i, qd, kvd = self.hidden, self.inter, self.q_dim, self.kv_dim
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)

        # Fake HF-style model whose named_parameters() the compiler ingests.
        # HF/PyTorch nn.Linear layout: (out_features, in_features), applied as
        # x @ W.T. Generating them this way means the test exercises the same
        # transpose a real checkpoint needs -- if ingest_model stopped doing it,
        # these cases fail rather than quietly computing x @ W.
        shapes = {"q_proj": (qd, h), "k_proj": (kvd, h), "v_proj": (kvd, h),
                  "o_proj": (h, qd), "gate_proj": (i, h), "up_proj": (i, h),
                  "down_proj": (h, i)}
        self.w = {}
        params = []
        for li in range(self.layers):
            for key, shp in shapes.items():
                # Small integral floats so quantization is exact and the
                # reference matches bit-for-bit.
                arr = rng.integers(-8, 9, size=shp).astype(np.float32)
                arr[0, 0] = 127.0          # pin the peak so scale = 1.0 exactly
                self.w[(li, key)] = arr.astype(np.int8)
                params.append((f"model.layers.{li}.{key}.weight", arr))

        class FakeModel:
            def named_parameters(self_inner):
                import types
                for nm, a in params:
                    yield nm, types.SimpleNamespace(
                        shape=a.shape,
                        detach=lambda a=a: types.SimpleNamespace(
                            cpu=lambda a=a: types.SimpleNamespace(numpy=lambda a=a: a)))

        # compile_model derives q_dim = num_heads*head_dim and
        # kv_dim = num_kv_heads*head_dim, so pin head_dim to q_dim and use a
        # single head -- deriving num_heads from `hidden` breaks whenever
        # q_dim is not a multiple of it.
        compiler.config["model"] = dict(
            hidden_size=h, intermediate_size=i, num_layers=self.layers,
            num_heads=self.n_heads, num_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim, rms_norm_eps=1e-6)

        compiler.instruction_stream = bytearray()
        # per_channel=False: this case hands compile_model weights that are
        # ALREADY INT8, and its reference mirrors the plain integer pipeline.
        # Re-quantising them per output channel would change their values and
        # add a correction the reference does not model. The per-channel path
        # has its own coverage in PerChannelLinearCase.
        compiler.compile_model(FakeModel(), capture_weights=True,
                               per_channel=False)
        self.program = bytes(compiler.instruction_stream)
        self.in_addr = compiler.model_input_dram
        self.out_addr = compiler.model_output_dram

        # Input activations. Magnitudes matter: this integer RMSNorm divides
        # by sqrt(sum of 256 clamped squares), so uniformly small inputs
        # normalise to all-zeros and every downstream stage collapses to zero
        # -- which would let this test pass without exercising anything. A
        # sparse mix of large and small values keeps norm(x) non-degenerate.
        self.x = np.zeros((TILE, h), dtype=np.int8)
        idx = rng.choice(TILE * h, size=max(4, TILE * h // 8), replace=False)
        self.x.flat[idx] = rng.integers(60, 120, size=idx.size) * rng.choice([-1, 1], idx.size)
        img = os.environ["RTV_E2E_IMG"]
        size = compiler.export_dram_image(img, extra={self.in_addr: self.x.tobytes()})
        self.dram = open(img, "rb").read()

        # -- reference --------------------------------------------------------
        cur = self.x
        for li in range(self.layers):
            # .T mirrors ingest_model's hf_layout transpose, so the reference
            # multiplies by the same orientation the device does.
            g = lambda k: self.w[(li, k)].T
            add = lambda a, b: np.clip(a.astype(np.int32) + b.astype(np.int32),
                                       -128, 127).astype(np.int8)
            y = self._rmsnorm(cur)
            qh = self._quant(ref_matmul(y, g("q_proj")))
            kh = self._quant(ref_matmul(y, g("k_proj")))
            vh = self._quant(ref_matmul(y, g("v_proj")))
            # compile_model applies RoPE to Q/K and masks causally.
            cos, sin = NPUCompiler.rope_tables(self.m_rows, self.head_dim)
            qr, kr = qh.copy(), kh.copy()
            for hh in range(self.n_heads):
                sl = slice(hh * self.head_dim, (hh + 1) * self.head_dim)
                qr[:, sl] = ref_rope(qh[:, sl], cos, sin)
            for kh_i in range(self.n_kv_heads):
                sl = slice(kh_i * self.head_dim, (kh_i + 1) * self.head_dim)
                kr[:, sl] = ref_rope(kh[:, sl], cos, sin)
            ctx = ref_attention(qr, kr, vh, self.head_dim,
                                self.n_heads, self.n_kv_heads, self._quant,
                                causal=True)
            attn = self._quant(ref_matmul(ctx, g("o_proj")))
            attn_res = add(cur, attn)                  # residual 1
            y2 = self._rmsnorm(attn_res)
            gate = self._quant(ref_matmul(y2, g("gate_proj")))
            up   = self._quant(ref_matmul(y2, g("up_proj")))
            act  = np.clip((self._silu(gate).astype(np.int32) *
                            up.astype(np.int32)) >> 7, -128, 127).astype(np.int8)
            mlp = self._quant(ref_matmul(act, g("down_proj")))
            cur = add(attn_res, mlp)                   # residual 2
        self.expected = cur

    def dump_regions(self):
        return [("y", self.out_addr, TILE * _padded(self.hidden))]

    def check(self, raws):
        got = np.frombuffer(raws["y"], dtype=np.int8).reshape(
            TILE, _padded(self.hidden))[:, :self.hidden]
        return compare(got, self.expected,
                       f"full layer h={self.hidden} i={self.inter} L={self.layers}")



class RMSNormRowCase:
    """
    compile_rmsnorm_row: RMSNorm whose scale is a sum over the WHOLE row.

    The single-tile block normalises each 16x16 tile by its own sum, so for
    n > 16 it applies a different scale per tile. This case is built to catch
    exactly that: with n spanning several tiles and per-tile magnitudes
    deliberately unequal, a per-tile scale and a row-wide scale give visibly
    different answers.
    """

    def __init__(self, name, n):
        self.name, self.n = name, n

    def build(self, compiler):
        n = self.n
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        # Uneven magnitudes across tiles: a per-tile scale would normalise each
        # tile's own spread away, a row-wide scale preserves their relative size.
        self.x = np.zeros((TILE, n), dtype=np.int8)
        for c in range(n_tiles(n)):
            lo, hi = (10, 40) if c % 2 == 0 else (80, 120)
            cols = slice(c * TILE, min((c + 1) * TILE, n))
            w = self.x[:, cols].shape[1]
            self.x[:, cols] = (rng.integers(lo, hi, size=(TILE, w))
                               * rng.choice([-1, 1], size=(TILE, w))).astype(np.int8)

        # New semantics: sum of squares in INT32 (no INT8 clamp), normalised
        # by the ROOT MEAN square, output in Q0.7 so 1.0 -> 127.
        total = np.int32((self.x.astype(np.int32) ** 2).sum())
        C = int(round(127.0 * np.sqrt(n)))
        inv = (np.float32(C) / np.sqrt(np.float32(total))) if total > 0 else np.float32(0)
        y = self.x.astype(np.float32) * inv
        self.expected = np.clip(np.trunc(y + np.copysign(0.5, y)),
                                -128, 127).astype(np.int8)

        self.in_addr, self.out_addr = 0x1000, 0x10_0000
        self.dram = bytearray(self.out_addr + 65536)
        xp = pad2d(self.x)
        self.dram[self.in_addr:self.in_addr + xp.nbytes] = xp.tobytes()

        compiler.instruction_stream = bytearray()
        compiler.compile_rmsnorm_row(
            dram_in=self.in_addr, dram_out=self.out_addr,
            n=n, sram_scalar=SRAMMap.SCALAR_SCRATCH_BASE)
        self.program = bytes(compiler.instruction_stream)

    def dump_regions(self):
        return [("y", self.out_addr, TILE * _padded(self.n))]

    def check(self, raws):
        got = np.frombuffer(raws["y"], dtype=np.int8).reshape(
            TILE, _padded(self.n))[:, :self.n]
        return compare(got, self.expected, f"rmsnorm_row n={self.n}")



class SoftmaxRowsCase:
    """
    compile_softmax_rows over rows WIDER than one tile.

    The old implementation refused n > 16, which capped attention context at
    16 tokens. This exercises the three-pass accumulate path, including rows
    whose width is not a multiple of the tile -- where a full-width final tile
    would read padding zeros, and a padding zero becomes the max whenever every
    real score in the row is negative.
    """

    def __init__(self, name, m, n, all_negative=False):
        self.name, self.m, self.n = name, m, n
        self.all_negative = all_negative

    def build(self, compiler):
        m, n = self.m, self.n
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        if self.all_negative:
            # Every score negative: a padding zero leaking into REDUCE_MAX
            # would become the row maximum and change every output.
            self.x = rng.integers(-120, -1, size=(m, n)).astype(np.int8)
        else:
            self.x = rng.integers(-100, 100, size=(m, n)).astype(np.int8)
        self.expected = ref_softmax_rows(self.x)

        self.in_addr, self.out_addr = 0x1000, 0x10_0000
        self.dram = bytearray(self.out_addr + 65536)
        xp = pad2d(self.x)
        self.dram[self.in_addr:self.in_addr + xp.nbytes] = xp.tobytes()

        compiler.instruction_stream = bytearray()
        compiler.compile_softmax_rows(
            dram_in=self.in_addr, dram_out=self.out_addr, m=m, n=n,
            sram_scalar=SRAMMap.SCALAR_SCRATCH_BASE)
        self.program = bytes(compiler.instruction_stream)

    def dump_regions(self):
        return [("p", self.out_addr, self.m * _padded(self.n))]

    def check(self, raws):
        got = np.frombuffer(raws["p"], dtype=np.int8).reshape(
            self.m, _padded(self.n))[:, :self.n]
        return compare(got, self.expected,
                       f"softmax m={self.m} n={self.n}"
                       + (" (all-negative)" if self.all_negative else ""))



class KVCacheDecodeCase:
    """
    One autoregressive decode step: a single query position attending over a
    populated KV cache.

    This is the shape generation actually needs -- seq_len=1 against
    cache_len previously-computed positions, giving scores of 1 x (cache_len+1),
    a single wide row. It only became expressible once compile_softmax_rows
    handled rows wider than one tile.

    The step's own K/V are appended to the cache at row cache_len before
    attention runs, so the model attends over the full history including
    itself. Causal masking is unnecessary here by construction: the cache
    holds only positions at or before the current one.
    """

    def __init__(self, name, hidden, head_dim, cache_len, n_heads=1, n_kv_heads=1):
        self.name, self.hidden, self.head_dim = name, hidden, head_dim
        self.cache_len, self.n_heads, self.n_kv_heads = cache_len, n_heads, n_kv_heads
        self.q_dim  = head_dim * n_heads
        self.kv_dim = head_dim * n_kv_heads

    def build(self, compiler):
        h, hd, cl = self.hidden, self.head_dim, self.cache_len
        qd, kvd = self.q_dim, self.kv_dim
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)

        self.x  = rng.integers(-8, 9, size=(1, h), dtype=np.int8)
        self.wq = rng.integers(-8, 9, size=(h, qd), dtype=np.int8)
        self.wk = rng.integers(-8, 9, size=(h, kvd), dtype=np.int8)
        self.wv = rng.integers(-8, 9, size=(h, kvd), dtype=np.int8)
        self.wo = rng.integers(-3, 4, size=(qd, h), dtype=np.int8)
        # Pre-existing cache contents for the earlier positions.
        self.k_hist = rng.integers(-40, 41, size=(cl, kvd), dtype=np.int8)
        self.v_hist = rng.integers(-40, 41, size=(cl, kvd), dtype=np.int8)

        q = lambda acc: np.clip(acc >> 7, -128, 127).astype(np.int8)
        q_new = q(ref_matmul(self.x, self.wq))
        k_new = q(ref_matmul(self.x, self.wk))
        v_new = q(ref_matmul(self.x, self.wv))
        k_all = np.vstack([self.k_hist, k_new])      # cache + this step
        v_all = np.vstack([self.v_hist, v_new])
        ctx = ref_attention(q_new, k_all, v_all, hd,
                            self.n_heads, self.n_kv_heads, q)
        self.expected = ref_matmul(ctx, self.wo)

        self.x_addr, self.wq_addr = 0x1000, 0x10_0000
        self.wk_addr, self.wv_addr = 0x20_0000, 0x30_0000
        self.wo_addr = 0x40_0000
        self.kc_addr, self.vc_addr = 0x50_0000, 0x60_0000
        self.out_addr = 0x70_0000
        self.dram = bytearray(0xA0_0000)
        for addr, arr in ((self.x_addr, self.x), (self.wq_addr, self.wq),
                          (self.wk_addr, self.wk), (self.wv_addr, self.wv),
                          (self.wo_addr, self.wo)):
            pa = pad2d(arr); self.dram[addr:addr + pa.nbytes] = pa.tobytes()
        # Cache rows laid out at the pitch the compiler addresses them with.
        for addr, arr in ((self.kc_addr, self.k_hist), (self.vc_addr, self.v_hist)):
            for r in range(cl):
                off = addr + r * _padded(kvd)
                self.dram[off:off + kvd] = arr[r].tobytes()

        compiler.instruction_stream = bytearray()
        compiler._staging_base = compiler._staging_cursor = 0x80_0000
        compiler.compile_attention_block(
            dram_x=self.x_addr, dram_wq=self.wq_addr, dram_wk=self.wk_addr,
            dram_wv=self.wv_addr, dram_wo=self.wo_addr, dram_out=self.out_addr,
            hidden=h, q_dim=qd, kv_dim=kvd, head_dim=hd,
            n_heads=self.n_heads, n_kv_heads=self.n_kv_heads,
            seq_len=1, cache_len=cl,
            dram_k_cache=self.kc_addr, dram_v_cache=self.vc_addr)
        self.program = bytes(compiler.instruction_stream)

    def dump_regions(self):
        return [("c", self.out_addr, _padded(1) * _padded(self.hidden) * 4)]

    def check(self, raws):
        got = np.frombuffer(raws["c"], dtype=np.int32).reshape(
            _padded(1), _padded(self.hidden))[:1, :self.hidden]
        return compare(got, self.expected,
                       f"decode cache_len={self.cache_len} heads={self.n_heads}")



class CausalAttentionCase:
    """
    Parallel (prefill) attention with a causal mask.

    Without masking this is bidirectional -- every position attends to every
    other, including future ones -- which is wrong for an autoregressive model
    at any sequence length, not just long ones. The mask is exact rather than
    an additive penalty: masked scores carry a sentinel that EXP_SUB maps to
    zero probability, so no attention weight leaks to future positions.
    """

    def __init__(self, name, m, hidden, head_dim, n_heads=1, n_kv_heads=1):
        self.name, self.m, self.hidden, self.head_dim = name, m, hidden, head_dim
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.q_dim, self.kv_dim = head_dim * n_heads, head_dim * n_kv_heads

    def build(self, compiler):
        m, h, hd = self.m, self.hidden, self.head_dim
        qd, kvd = self.q_dim, self.kv_dim
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        self.x  = rng.integers(-8, 9, size=(m, h), dtype=np.int8)
        self.wq = rng.integers(-8, 9, size=(h, qd), dtype=np.int8)
        self.wk = rng.integers(-8, 9, size=(h, kvd), dtype=np.int8)
        self.wv = rng.integers(-8, 9, size=(h, kvd), dtype=np.int8)
        self.wo = rng.integers(-3, 4, size=(qd, h), dtype=np.int8)

        q = lambda acc: np.clip(acc >> 7, -128, 127).astype(np.int8)
        qi, ki, vi = (q(ref_matmul(self.x, w)) for w in (self.wq, self.wk, self.wv))
        ctx = ref_attention(qi, ki, vi, hd, self.n_heads, self.n_kv_heads, q,
                            causal=True)
        self.expected = ref_matmul(ctx, self.wo)

        self.x_addr, self.wq_addr = 0x1000, 0x10_0000
        self.wk_addr, self.wv_addr = 0x20_0000, 0x30_0000
        self.wo_addr, self.out_addr = 0x40_0000, 0x50_0000
        self.dram = bytearray(0x90_0000)
        for addr, arr in ((self.x_addr, self.x), (self.wq_addr, self.wq),
                          (self.wk_addr, self.wk), (self.wv_addr, self.wv),
                          (self.wo_addr, self.wo)):
            pa = pad2d(arr); self.dram[addr:addr + pa.nbytes] = pa.tobytes()

        compiler.instruction_stream = bytearray()
        compiler._staging_base = compiler._staging_cursor = 0x60_0000
        compiler.compile_attention_block(
            dram_x=self.x_addr, dram_wq=self.wq_addr, dram_wk=self.wk_addr,
            dram_wv=self.wv_addr, dram_wo=self.wo_addr, dram_out=self.out_addr,
            hidden=h, q_dim=qd, kv_dim=kvd, head_dim=hd,
            n_heads=self.n_heads, n_kv_heads=self.n_kv_heads, causal=True)
        self.program = bytes(compiler.instruction_stream)

    def dump_regions(self):
        return [("c", self.out_addr, _padded(self.m) * _padded(self.hidden) * 4)]

    def check(self, raws):
        got = np.frombuffer(raws["c"], dtype=np.int32).reshape(
            _padded(self.m), _padded(self.hidden))[:self.m, :self.hidden]
        return compare(got, self.expected, f"causal m={self.m} heads={self.n_heads}")



def ref_rope(x, cos, sin, pos_offset=0):
    """
    Reference RoPE, mirroring the device path exactly.

    The sign of rotate_half is already folded into `sin` by rope_tables, so
    this is a plain two-multiply-and-add -- the same arithmetic the compiled
    instructions perform, at the same INT8 scale.
    """
    m, hd = x.shape
    half = hd // 2
    rot = np.concatenate([x[:, half:], x[:, :half]], axis=1)   # partner columns
    c = cos[pos_offset:pos_offset + m, :hd].astype(np.int32)
    s = sin[pos_offset:pos_offset + m, :hd].astype(np.int32)
    a = (x.astype(np.int32) * c) >> 7
    b = (rot.astype(np.int32) * s) >> 7
    return np.clip(np.clip(a, -128, 127) + np.clip(b, -128, 127),
                   -128, 127).astype(np.int8)


class RoPECase:
    """
    compile_rope: rotary position embedding.

    RoPE is the ONLY thing carrying token position in this architecture, so
    without it the model is order-blind -- results are wrong, not merely
    limited. This checks the rotation against a reference at several head
    dims and at a non-zero position offset (the decode case, where the query
    sits at absolute position cache_len).
    """

    def __init__(self, name, rows, head_dim, pos_offset=0):
        self.name, self.rows, self.head_dim = name, rows, head_dim
        self.pos_offset = pos_offset

    def build(self, compiler):
        m, hd = self.rows, self.head_dim
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)
        self.x = rng.integers(-100, 101, size=(m, hd), dtype=np.int8)

        max_pos = self.pos_offset + m
        cos, sin = NPUCompiler.rope_tables(max_pos, hd)
        self.expected = ref_rope(self.x, cos, sin, self.pos_offset)

        self.in_addr, self.cos_addr = 0x1000, 0x10_0000
        self.sin_addr, self.out_addr = 0x20_0000, 0x30_0000
        self.dram = bytearray(0x50_0000)
        for addr, arr in ((self.in_addr, self.x), (self.cos_addr, cos),
                          (self.sin_addr, sin)):
            pa = pad2d(arr); self.dram[addr:addr + pa.nbytes] = pa.tobytes()

        compiler.instruction_stream = bytearray()
        compiler.compile_rope(
            dram_in=self.in_addr, dram_out=self.out_addr,
            dram_cos=self.cos_addr, dram_sin=self.sin_addr,
            rows=m, head_dim=hd, pos_offset=self.pos_offset)
        self.program = bytes(compiler.instruction_stream)

    def dump_regions(self):
        return [("y", self.out_addr, _padded(self.rows) * _padded(self.head_dim))]

    def check(self, raws):
        got = np.frombuffer(raws["y"], dtype=np.int8).reshape(
            _padded(self.rows), _padded(self.head_dim))[:self.rows, :self.head_dim]
        return compare(got, self.expected,
                       f"rope hd={self.head_dim} pos+{self.pos_offset}")



class LMHeadCase:
    """
    compile_lm_head plus the two host-side steps around it.

    Verifies the device half (logits = x @ W_lm) numerically, and the host
    half by construction: the embedding gather and the argmax are performed
    here in Python because neither is expressible on this ISA -- LOAD_TILE
    takes a compile-time address so a runtime token id cannot select a row,
    and REDUCE_MAX yields a maximum value with no index attached.

    Also covers the HF weight layout: weights are generated as
    (out_features, in_features) the way a real checkpoint stores them, so
    this fails if ingest_model stops transposing.
    """

    def __init__(self, name, hidden, vocab):
        self.name, self.hidden, self.vocab = name, hidden, vocab

    def build(self, compiler):
        h, v = self.hidden, self.vocab
        rng = np.random.default_rng(abs(hash(self.name)) & 0xFFFF)

        # HF layout: embed_tokens is (vocab, hidden); lm_head is (vocab, hidden)
        # and is applied as x @ W.T.
        embed = rng.integers(-100, 101, size=(v, h)).astype(np.float32)
        lm    = rng.integers(-8, 9, size=(v, h)).astype(np.float32)
        embed[0, 0] = 127.0
        lm[0, 0] = 127.0

        import types
        params = [("model.embed_tokens.weight", embed), ("lm_head.weight", lm)]

        class FakeModel:
            def named_parameters(self_inner):
                for nm, a in params:
                    yield nm, types.SimpleNamespace(
                        shape=a.shape,
                        detach=lambda a=a: types.SimpleNamespace(
                            cpu=lambda a=a: types.SimpleNamespace(numpy=lambda a=a: a)))

        compiler.instruction_stream = bytearray()
        recs = compiler.ingest_model(FakeModel(), capture_weights=True)
        emb_rec = next(r for r in recs.values() if r.key == "embed_tokens")
        lm_rec  = next(r for r in recs.values() if r.key == "lm_head")

        # Host step 1: gather the embedding row for this token.
        self.token_id = 3
        row = compiler.embedding_row(self.token_id)
        self.x = row.reshape(1, h)

        compiler.init_staging()
        x_addr   = compiler.alloc_staging(TILE * _padded(h))
        out_addr = compiler.alloc_staging(_padded(v) * _padded(1) * 4)
        self.out_addr, self.x_addr = out_addr, x_addr

        compiler.compile_lm_head(dram_in=x_addr, dram_w=lm_rec.dram_addr,
                                 dram_out=out_addr, m=1, hidden=h, vocab=v)
        self.program = bytes(compiler.instruction_stream)

        img = os.environ["RTV_E2E_IMG"]
        compiler.export_dram_image(img, extra={x_addr: pad2d(self.x).tobytes()})
        self.dram = open(img, "rb").read()

        # lm_head is NOT transposed on ingest (it is the A operand), so the
        # captured tensor is native (vocab, hidden) and the result is
        # logits^T with shape (vocab, m).
        lm_i8 = compiler._weight_bytes["lm_head.weight"]     # (vocab, hidden)
        self.expected = ref_matmul(lm_i8[:v, :h], self.x.T)  # (vocab, 1)
        self.expected_token = int(np.argmax(self.expected[:, 0]))   # host argmax

    def dump_regions(self):
        return [("l", self.out_addr, _padded(self.vocab) * _padded(1) * 4)]

    def check(self, raws):
        got = np.frombuffer(raws["l"], dtype=np.int32).reshape(
            _padded(self.vocab), _padded(1))[:self.vocab, :1]
        ok, detail = compare(got, self.expected, f"lm_head vocab={self.vocab}")
        if not ok:
            return False, detail
        # Host step 2: argmax over the returned logits.
        if int(np.argmax(got[:, 0])) != self.expected_token:
            return False, (f"argmax {int(np.argmax(got[:, 0]))} != "
                           f"{self.expected_token}")
        return True, f"{detail}, token {self.expected_token}"


# ── Comparison helper ────────────────────────────────────────────────────────

def compare(actual, expected, label):
    if np.array_equal(actual, expected):
        return True, label
    diff = actual != expected
    idx = np.argwhere(diff)
    first = tuple(int(v) for v in idx[0])
    # Magnitude matters for triage: an all-by-1 spread is float rounding
    # drift between the reference and the device's float32 transcendentals,
    # whereas a structural bug (wrong address, wrong shift) shows up as
    # large or wildly varying deltas.
    mag = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
    detail = (f"{label}: {int(diff.sum())}/{expected.size} wrong "
              f"(max delta {int(mag.max())}, "
              f"by1={int((mag==1).sum())}, by>1={int((mag>1).sum())}); "
              f"first at {first} got {actual[first]} want {expected[first]} "
              f"(tile r{first[0]//TILE} c{first[1]//TILE})")
    if expected.size <= 64:
        detail += f"\n      expected:\n{expected}\n      actual:\n{actual}"
    return False, detail


# ── Runner ───────────────────────────────────────────────────────────────────

def run_case(case, sim, workdir, hal):
    # Cases that build their own DRAM image (FullLayerCase) write it here.
    os.environ["RTV_E2E_IMG"] = os.path.join(workdir, f"{case.name}.img")
    compiler = NPUCompiler(hal)
    try:
        case.build(compiler)
    except Exception as e:                      # noqa: BLE001
        print(f"FAIL  {case.name}: build raised {type(e).__name__}: {e}")
        return False

    prog = os.path.join(workdir, f"{case.name}.bin")
    dram = os.path.join(workdir, f"{case.name}.dram")
    with open(prog, "wb") as f:
        f.write(case.program)
    with open(dram, "wb") as f:
        f.write(case.dram)

    regions = case.dump_regions()
    dumps, paths = [], {}
    for label, addr, nbytes in regions:
        p = os.path.join(workdir, f"{case.name}.{label}.out")
        paths[label] = p
        dumps += ["--dump", f"{addr}:{nbytes}:{p}"]

    dram_mb = max(1, len(case.dram) // (1024 * 1024) + 2)
    proc = subprocess.run([sim, prog, hal, str(dram_mb),
                           "--dram-image", dram] + dumps,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"FAIL  {case.name}: simulator exited {proc.returncode}")
        print((proc.stdout + proc.stderr)[-1500:])
        return False

    raws = {}
    for label in paths:
        with open(paths[label], "rb") as f:
            raws[label] = f.read()

    ok, detail = case.check(raws)
    n_instr = len(case.program) // 16
    if ok:
        print(f"PASS  {case.name}  ({detail}, {n_instr} instrs)")
    else:
        print(f"FAIL  {case.name}  {detail}")
    return ok


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim", default=os.path.join(root, "build", "rtv_sim"))
    ap.add_argument("--hal", default=os.path.join(here, "hal_config.json"))
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--filter", default="", help="only run cases containing this substring")
    args = ap.parse_args()

    if not os.path.exists(args.sim):
        print(f"error: simulator not found at {args.sim} (build it first)")
        return 2

    cases = [
        # ── compile_linear ───────────────────────────────────────────────────
        LinearCase("linear_16x16x16",  16, 16, 16),    # single K-tile
        LinearCase("linear_16x16x32",  16, 16, 32),    # K-accumulation
        LinearCase("linear_16x32x16",  16, 32, 16),    # multi N-tile store
        LinearCase("linear_16x32x32",  16, 32, 32),
        # Per-channel requantisation: columns spanning 1000x, which is exactly
        # what one shared scale cannot hold.
        PerChannelLinearCase("perchan_16x16x16", 16, 16, 16),
        PerChannelLinearCase("perchan_16x32x32", 16, 32, 32),
        PerChannelLinearCase("perchan_16x48x64", 16, 48, 64),
        # INT16 residual with scale alignment -- the 4.26x ratio measured
        # between Qwen3's residual stream and o_proj's output at layer 0.
        ScaledResidualCase("resid_scaled_16",  16, 0.00250677, 0.000587858),
        ScaledResidualCase("resid_scaled_48",  48, 0.00250677, 0.000587858),
        ScaledResidualCase("resid_scaled_up",  32, 0.0005, 0.002),
        # The real shape: INT16 residual, INT8 sublayer output widened in.
        ScaledResidualCase("resid_mixed_16", 16, 0.00250677, 0.000587858, True),
        ScaledResidualCase("resid_mixed_48", 48, 0.0005, 0.002, True),
        # Ratio 348 -- measured at layer 0 -- only fits because the widen
        # absorbs 2**8 first.
        ScaledResidualCase("resid_bigratio", 32, 9.014e-05, 3.135e-02, True, 8),
        # Layer boundary: the stream's own scale changes, so A converts too.
        ScaledResidualCase("resid_boundary", 32, 2.101e-04, 7.177e-03, True, 8,
                           s_out=1.844e-04),
        # Nonlinearity via lookup table, at Qwen3's layer-2 gate scale.
        # No quantisation: proves addressing/tiling/K-accumulation alone.
        Fp32LinearCase("fp32_linear_16x16x16", 16, 16, 16),
        Fp32LinearCase("fp32_linear_16x32x32", 16, 32, 32),
        Fp32LinearCase("fp32_linear_16x48x64", 16, 48, 64),
        Fp32MlpCase("fp32_mlp_16x32", 16, 32),
        Fp32MlpCase("fp32_mlp_32x64", 32, 64),
        Fp32AttentionCase("fp32_attn_1head",  32, 1, 1, 32),
        Fp32AttentionCase("fp32_attn_gqa",    32, 4, 2, 16),
        Fp32FullLayerCase("fp32_layer_1L", 64, 64, 2, 1, 32, layers=1),
        Fp32FullLayerCase("fp32_layer_2L", 64, 64, 2, 1, 32, layers=2),
        LutCase("lut_silu_16", 16, 40.0 / 127),
        LutCase("lut_silu_48", 48, 40.0 / 127),
        # RMSNorm consuming the INT16 residual, including a massive channel.
        RMSNormInt16Case("rmsnorm_i16_16", 16),
        RMSNormInt16Case("rmsnorm_i16_48", 48),
        LinearCase("linear_16x16x128", 16, 16, 128),   # 8-step reduction
        LinearCase("linear_16x64x64",  16, 64, 64),
        LinearCase("linear_16x17x16",  16, 17, 16),    # non-multiple-of-16 dims
        LinearCase("linear_16x16x17",  16, 16, 17),
        LinearCase("linear_16x20x36",  16, 20, 36),
        LinearCase("linear_cslot1",    16, 16, 32, c_slot=1),
        LinearCase("linear_cslot3",    16, 32, 48, c_slot=3),
        # ── compile_sibling_qkv ──────────────────────────────────────────────
        SiblingQKVCase("sibling_qkv_k16", 16, 16, 16),
        SiblingQKVCase("sibling_qkv_k32", 16, 16, 32),
        # ── REDUCE tile source ───────────────────────────────────────────────
        ReduceCase("reduce_at_input",  SRAMMap.input_slot(0)),
        ReduceCase("reduce_at_slot2",  SRAMMap.input_slot(2)),
        ReduceCase("reduce_at_output", SRAMMap.output_slot(0)),
        # ── chained projections ──────────────────────────────────────────────
        ChainedLinearCase("chained_16x16_16x16", 16, 16, 16, 16),
        # ── compile_mlp_block ────────────────────────────────────────────────
        MLPBlockCase("mlp_h16_i16",  16, 16, 16),
        MLPBlockCase("mlp_h32_i48",  16, 32, 48),
        # ── compile_attention_block ──────────────────────────────────────────
        AttentionBlockCase("attn_h16_q16_kv16", 16, 16, 16, 16),
        AttentionBlockCase("attn_h32_q16_kv16", 16, 32, 16, 16),
        # multi-head, and GQA (4 query heads sharing 2 KV heads)
        AttentionBlockCase("attn_mha_4h", 16, 32, 64, 64,
                           head_dim=16, n_heads=4, n_kv_heads=4),
        AttentionBlockCase("attn_gqa_4h_2kv", 16, 32, 64, 32,
                           head_dim=16, n_heads=4, n_kv_heads=2),
        AttentionBlockCase("attn_rope_hd32", 16, 32, 64, 32,
                           head_dim=32, n_heads=2, n_kv_heads=1, rope=True),
        # ── compile_rmsnorm_block ────────────────────────────────────────────
        RMSNormCase("rmsnorm_small", scale=1),
        RMSNormCase("rmsnorm_large", scale=4),
        # ── compile_rmsnorm_row (scale spans the whole row) ──────────────────
        RMSNormRowCase("rmsnorm_row_n16",  16),   # degenerates to one tile
        RMSNormRowCase("rmsnorm_row_n48",  48),   # 3 tiles
        RMSNormRowCase("rmsnorm_row_n80",  80),   # 5 tiles, uneven magnitudes
        # ── compile_softmax_rows (rows wider than one tile) ──────────────────
        SoftmaxRowsCase("softmax_n16",  4, 16),
        SoftmaxRowsCase("softmax_n64",  4, 64),    # 4 tiles per row
        SoftmaxRowsCase("softmax_n80",  4, 80),
        SoftmaxRowsCase("softmax_n40",  4, 40),    # partial final tile
        SoftmaxRowsCase("softmax_neg",  4, 40, all_negative=True),
        # ── KV-cache decode (one query over a growing cache) ─────────────────
        KVCacheDecodeCase("decode_c15",     16, 16, cache_len=15),
        KVCacheDecodeCase("decode_c47",     16, 16, cache_len=47),
        KVCacheDecodeCase("decode_gqa_c31", 32, 16, cache_len=31,
                          n_heads=4, n_kv_heads=2),
        # ── causal masking (parallel prefill) ────────────────────────────────
        CausalAttentionCase("causal_m16",     16, 16, 16),
        CausalAttentionCase("causal_gqa_4h",  16, 32, 16, n_heads=4, n_kv_heads=2),
        # ── RoPE ─────────────────────────────────────────────────────────────
        RoPECase("rope_hd32",      16, 32),
        RoPECase("rope_hd64",      16, 64),
        RoPECase("rope_hd128",     16, 128),   # Qwen3's head_dim
        RoPECase("rope_hd64_pos48", 16, 64, pos_offset=48),   # decode offset
        # ── lm_head + host-side embedding gather and argmax ──────────────────
        LMHeadCase("lm_head_v64",  32, 64),
        LMHeadCase("lm_head_v256", 32, 256),
        # ── compile_model (whole layer, real weights in DRAM) ────────────────
        # head_dim must be a multiple of 32 so RoPE's rotate_half partner is
        # a whole tile -- see compile_rope.
        FullLayerCase("full_layer_1L", hidden=32, inter=32, q_dim=32,
                      kv_dim=32, layers=1, head_dim=32),
        FullLayerCase("full_layer_2L", hidden=32, inter=32, q_dim=32,
                      kv_dim=32, layers=2, head_dim=32),
        FullLayerCase("full_layer_gqa", hidden=32, inter=64, q_dim=128,
                      kv_dim=64, layers=1, head_dim=32, n_heads=4, n_kv_heads=2),
    ]
    if args.filter:
        cases = [c for c in cases if args.filter in c.name]

    workdir = tempfile.mkdtemp(prefix="rtv_e2e_")
    try:
        results = [run_case(c, args.sim, workdir, args.hal) for c in cases]
    finally:
        if args.keep:
            print(f"\nintermediates kept in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
