#!/usr/bin/env python3
"""gguf_reader.py -- minimal dependency-free GGUF v3 reader (numpy only): metadata + lazy tensor access.
Supports F32 / F16 / BF16 / Q4_0 / Q8_0 / Q6_K dequantisation, and raw Q4_0 access for the INT4W repack."""
import mmap
import struct

import numpy as np

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
GGML = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K",
        14: "Q6_K", 30: "BF16"}
# (elements per block, bytes per block)
BLOCK = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q4_0": (32, 18), "Q8_0": (32, 34), "Q6_K": (256, 210)}


class GGUF:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.pos = 0
        assert self.mm[:4] == b"GGUF", "not a GGUF file"
        self.pos = 4
        self.version = self._u("<I")
        n_tensors, n_kv = self._u("<Q"), self._u("<Q")
        self.meta = {}
        for _ in range(n_kv):
            k = self._str()
            self.meta[k] = self._value(self._u("<I"))
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._str()
            nd = self._u("<I")
            dims = tuple(self._u("<Q") for _ in range(nd))     # ggml order: dims[0] is the fastest (input) axis
            t = self._u("<I")
            off = self._u("<Q")
            self.tensors[name] = (dims, GGML.get(t, f"type{t}"), off)
        align = self.meta.get("general.alignment", 32)
        self.data0 = (self.pos + align - 1) // align * align

    def _u(self, fmt):
        v = struct.unpack_from(fmt, self.mm, self.pos)[0]
        self.pos += struct.calcsize(fmt)
        return v

    def _str(self):
        n = self._u("<Q")
        s = bytes(self.mm[self.pos:self.pos + n]).decode("utf-8", "replace")
        self.pos += n
        return s

    def _value(self, t):
        if t in _SCALAR:
            return self._u(_SCALAR[t])
        if t == 8:
            return self._str()
        if t == 9:
            et, n = self._u("<I"), self._u("<Q")
            return [self._value(et) for _ in range(n)]
        raise ValueError(f"unknown GGUF value type {t}")

    def raw(self, name):
        dims, typ, off = self.tensors[name]
        n = int(np.prod(dims))
        be, bb = BLOCK[typ]
        nbytes = n // be * bb
        a = self.data0 + off
        return dims, typ, np.frombuffer(self.mm, np.uint8, nbytes, a)

    def shape(self, name):
        """numpy-style shape (outermost first) = reversed ggml dims."""
        return tuple(reversed(self.tensors[name][0]))

    def q4_0_parts(self, name):
        """-> (codes uint8 [..., nblk, 32] with values 0..15 in element order, scales float32 [..., nblk])."""
        dims, typ, raw = self.raw(name)
        assert typ == "Q4_0", typ
        b = raw.reshape(-1, 18)
        d = b[:, :2].copy().view(np.float16).reshape(-1).astype(np.float32)
        q = b[:, 2:]
        codes = np.concatenate([q & 15, q >> 4], axis=1)          # low nibbles = elements 0..15, high = 16..31
        lead = tuple(reversed(dims[1:]))
        nblk = dims[0] // 32
        return codes.reshape(lead + (nblk, 32)), d.reshape(lead + (nblk,))

    def dequant(self, name):
        dims, typ, raw = self.raw(name)
        shape = tuple(reversed(dims))
        if typ == "F32":
            return raw.view(np.float32).reshape(shape).copy()
        if typ == "F16":
            return raw.view(np.float16).reshape(shape).astype(np.float32)
        if typ == "BF16":
            return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape).copy()
        if typ == "Q4_0":
            c, d = self.q4_0_parts(name)
            return ((c.astype(np.float32) - 8.0) * d[..., None]).reshape(shape)
        if typ == "Q8_0":
            b = raw.reshape(-1, 34)
            d = b[:, :2].copy().view(np.float16).astype(np.float32)
            q = b[:, 2:].copy().view(np.int8).astype(np.float32)
            return (q * d).reshape(shape)
        if typ == "Q6_K":
            b = raw.reshape(-1, 210)
            ql, qh, sc = b[:, :128], b[:, 128:192], b[:, 192:208].copy().view(np.int8)
            d = b[:, 208:210].copy().view(np.float16).astype(np.float32)   # (nb,1)
            out = np.empty((b.shape[0], 256), np.float32)
            for h in range(2):                                   # two halves of 128
                l = ql[:, 64 * h:64 * h + 64]
                hh = qh[:, 32 * h:32 * h + 32]
                s = sc[:, 8 * h:8 * h + 8].astype(np.float32)
                q1 = ((l[:, :32] & 15) | (((hh >> 0) & 3) << 4)).astype(np.int32) - 32
                q2 = ((l[:, 32:] & 15) | (((hh >> 2) & 3) << 4)).astype(np.int32) - 32
                q3 = ((l[:, :32] >> 4) | (((hh >> 4) & 3) << 4)).astype(np.int32) - 32
                q4 = ((l[:, 32:] >> 4) | (((hh >> 6) & 3) << 4)).astype(np.int32) - 32
                for j, q in enumerate((q1, q2, q3, q4)):
                    sj = np.repeat(s[:, 2 * j:2 * j + 2], 16, axis=1)
                    out[:, 128 * h + 32 * j:128 * h + 32 * j + 32] = d * sj * q
            return out.reshape(shape)
        raise NotImplementedError(typ)


if __name__ == "__main__":
    import sys
    g = GGUF(sys.argv[1])
    for k, v in g.meta.items():
        if isinstance(v, list) and len(v) > 12:
            v = f"[{len(v)} items] {v[:6]}..."
        print(k, "=", v)
    for n, (d, t, o) in g.tensors.items():
        if ".0." in n or "blk" not in n or n.startswith("blk.5."):
            print(n, d, t)
