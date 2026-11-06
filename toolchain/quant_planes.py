#!/usr/bin/env python3
"""quant_planes.py -- express other group-quantised integer weights on the chip's INT4W grid, without touching the ISA.

The chip multiplies a signed 4-bit value (-8..7) by one FP32 scale per (group of K inputs, output column).
Anything of the form  w = scale * (integer + offset)  is a sum of such planes plus a constant per group:

  Q4_K   w = s*q - m,            q in 0..15       one plane  (q-8, s),                  correction  8*s - m
  Q8_0   w = s*q,                q in -128..127   two planes (q>>4, 16*s), (q&15 - 8, s), correction  8*s
  Q6_K   w = s*(q - 32),         q in 0..63       two planes (q>>4 - 2, 16*s), (q&15 - 8, s), correction 8*s
  (Q4_0 is the single plane (q-8, s) with no correction and does not come through here.)

The constant per group is added through the group sums of the activation:  y[n] += sum_g xsum_g * c_g[n],
xsum = x @ G with G the 0/1 group-membership matrix (itself an INT4W blob, values 0/1 at scale 1) and
c an FP32 [groups, N] matrix: two extra small GEMMs, both on existing hardware.

Every function takes arrays laid out [K, N] (input x output) and returns a dict
    planes: [(values int8 [K, N] in -8..7, scales fp32 [K/group, N], group)]
    corr:   (c fp32 [K/group, N], group)
"""
import numpy as np


def asym4(q, s, m, group):
    """q uint 0..15 [K, N]; s, m fp32 [K/group, N]; w = s*q - m."""
    q = np.asarray(q)
    return dict(planes=[((q.astype(np.int16) - 8).astype(np.int8), np.asarray(s, np.float32), group)],
                corr=((8.0 * np.asarray(s, np.float32) - np.asarray(m, np.float32)).astype(np.float32), group))


def sym8(q, s, group=32):
    """q int8 [K, N]; s fp32 [K/group, N]; w = s*q."""
    q = np.asarray(q, np.int16)
    s = np.asarray(s, np.float32)
    return dict(planes=[((q >> 4).astype(np.int8), (16.0 * s).astype(np.float32), group),
                        (((q & 15) - 8).astype(np.int8), s, group)],
                corr=((8.0 * s).astype(np.float32), group))


def sym6(q, s, group=16):
    """q uint 0..63 [K, N]; s fp32 [K/group, N]; w = s*(q - 32)."""
    q = np.asarray(q, np.int16)
    s = np.asarray(s, np.float32)
    return dict(planes=[(((q >> 4) - 2).astype(np.int8), (16.0 * s).astype(np.float32), group),
                        (((q & 15) - 8).astype(np.int8), s, group)],
                corr=((8.0 * s).astype(np.float32), group))


def dequant(spec, K):
    """The matrix the planes plus the correction represent, fp64 [K, N] (the oracle)."""
    w = 0.0
    for v, s, g in spec["planes"]:
        w = w + v.astype(np.float64) * np.repeat(s.astype(np.float64), g, axis=0)
    if spec["corr"] is not None:
        c, g = spec["corr"]
        w = w + np.repeat(c.astype(np.float64), g, axis=0)
    return w


def kg_pad(K, group, tile=16):
    return -(-(K // group) // tile) * tile


def corr_matrix(spec, K):
    """FP32 [Kg_pad, N]: the correction, zero rows padding the group count to a tile multiple."""
    c, g = spec["corr"]
    out = np.zeros((kg_pad(K, g), c.shape[1]), np.float32)
    out[:c.shape[0]] = c
    return out
