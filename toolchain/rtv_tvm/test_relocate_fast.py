#!/usr/bin/env python3
"""
test_relocate_fast.py -- tvm_backend._relocate_instructions (binary search over the regions, only LOAD/STORE rows
touched) equals the straightforward algorithm it replaced (a boolean mask per region over the whole stream) on random
instruction streams: disjoint and overlapping regions, shared ranges, INDIRECT_ADDR loads, addresses above 4 GiB,
non-memory opcodes that must pass through untouched. Also checks it is much faster on a stream of realistic size.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_relocate_fast.py
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE)))

import tvm_backend  # noqa: E402


def relocate_reference(instr_bytes, source_roles, target_roles, staging_delta,
                           shared_ranges):

    import numpy as np
    n = len(instr_bytes) // 16
    w = np.frombuffer(instr_bytes, dtype=">u8").reshape(n, 2).astype(np.uint64)
    hi = w[:, 0]
    opc = (hi >> np.uint64(60)) & np.uint64(0xF)
    mem = (opc == np.uint64(int(tvm_backend.Opcode.LOAD_TILE))) | (opc == np.uint64(int(tvm_backend.Opcode.STORE_TILE)))
    # LOAD/STORE address = 32-bit dram_addr field (hi bits 24..55) + dram_hi (lo bits 36..39, ISA v1.6) so
    # images above 4 GB (real MoE expert pools) relocate correctly.
    lo = w[:, 1]
    # INDIRECT_ADDR loads reuse those bits as an SRAM pointer, not an address.
    ind = mem & ((lo & np.uint64(0x7F)) == np.uint64(0x14))
    addr = (((hi >> np.uint64(24)) & np.uint64(0xFFFFFFFF)) |
            np.where(ind, np.uint64(0), ((lo >> np.uint64(36)) & np.uint64(0xF)) << np.uint64(32))).astype(np.int64)
    todo = mem.copy()
    for start, nbytes in shared_ranges:          # (1) shared constants stay put
        todo &= ~((addr >= start) & (addr < start + nbytes))
    new = addr.copy()
    for role, (start, nbytes) in source_roles.items():   # (2) first matching role wins
        sel = todo & (addr >= start) & (addr < start + nbytes)
        new[sel] = target_roles[role][0] + (addr[sel] - start)
        todo &= ~sel
    new[todo] = addr[todo] + staging_delta          # (3) per-layer staging
    changed = mem & (new != addr)
    if not changed.any():
        return bytes(instr_bytes)
    if (new[changed] < 0).any() or (new[changed] >> 36).any():
        raise ValueError("relocated DRAM address does not fit the 36-bit LOAD/STORE address")
    nu = new.astype(np.uint64)
    hi = np.where(changed, (hi & ~(np.uint64(0xFFFFFFFF) << np.uint64(24))) |
                  ((nu & np.uint64(0xFFFFFFFF)) << np.uint64(24)), hi)
    if (changed & ind & (new >> 32 != 0)).any():
        raise ValueError("indirect LOAD/STORE address does not fit 32 bits")
    lo = np.where(changed & ~ind, (lo & ~(np.uint64(0xF) << np.uint64(36))) | ((nu >> np.uint64(32)) << np.uint64(36)), lo)
    w[:, 0] = hi
    w[:, 1] = lo
    return w.astype(">u8").tobytes()




def random_stream(rng, n, spans):
    raw = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    be = np.frombuffer(raw.tobytes(), dtype=">u8").reshape(n, 2).astype(np.uint64)
    hi, lo = be[:, 0], be[:, 1]
    opc = rng.choice([int(tvm_backend.Opcode.LOAD_TILE), int(tvm_backend.Opcode.STORE_TILE), 1, 5, 7], size=n)
    hi = (hi & ~(np.uint64(0xF) << np.uint64(60))) | (opc.astype(np.uint64) << np.uint64(60))
    span = rng.integers(0, len(spans), size=n)
    base = np.array([s[0] for s in spans], dtype=np.int64)[span]
    size = np.array([s[1] for s in spans], dtype=np.int64)[span]
    addr = (base + rng.integers(0, 1 << 40, size=n) % size).astype(np.uint64)
    hi = (hi & ~(np.uint64(0xFFFFFFFF) << np.uint64(24))) | ((addr & np.uint64(0xFFFFFFFF)) << np.uint64(24))
    ind = rng.random(n) < 0.15
    lo = (lo & ~(np.uint64(0xF) << np.uint64(36))) | (((addr >> np.uint64(32)) & np.uint64(0xF)) << np.uint64(36))
    lo = np.where(ind, (lo & ~np.uint64(0x7F)) | np.uint64(0x14), lo & ~np.uint64(0x7F) | np.uint64(0x03))
    out = np.empty((n, 2), dtype=">u8")
    out[:, 0], out[:, 1] = hi, lo
    return out.tobytes()


def main():
    rng = np.random.default_rng(7)
    spans = [(0x1000, 0x4000), (0x9000, 0x3000), (0x20000, 0x8000), (0x1_0000_0000 + 0x40000, 0x10000), (0x500000, 0x100)]
    ok = True
    for trial in range(12):
        stream = random_stream(rng, 6000, spans)
        overlapping = trial % 3 == 2
        src, tgt = {}, {}
        for r, (b, s) in enumerate(spans[:4] + ([(0x1800, 0x4000)] if overlapping else [])):
            src[f"r{r}"] = (b, s)
            tgt[f"r{r}"] = (b + 0x100000 * (r + 1), s)
        shared = [(0x20000, 0x400), (0x9000, 0x10)] if trial % 2 else []
        delta = 0x2000 * (trial % 4)
        got = tvm_backend._relocate_instructions(stream, src, tgt, delta, shared)
        want = relocate_reference(stream, src, tgt, delta, shared)
        same = got == want and len(got) == len(stream)
        ok &= same
        print(f"{'PASS' if same else 'FAIL'}  trial {trial} ({'overlapping' if overlapping else 'disjoint'} regions, "
              f"{len(shared)} shared ranges, staging delta {delta:#x})")
    # speed on a realistic size: 1.3M instructions, ~60 regions
    big = random_stream(rng, 1_300_000, spans)
    src = {f"r{r}": (0x100000 + r * 0x1000, 0x1000) for r in range(60)}
    tgt = {k: (v[0] + 0x400000, v[1]) for k, v in src.items()}
    big = random_stream(rng, 1_300_000, [(0x100000, 60 * 0x1000), (0x700000, 0x1000)])
    t0 = time.time(); a = tvm_backend._relocate_instructions(big, src, tgt, 0, []); t_new = time.time() - t0
    t0 = time.time(); b = relocate_reference(big, src, tgt, 0, []); t_old = time.time() - t0
    print(f"{'PASS' if a == b else 'FAIL'}  1.3M instructions, 60 regions: identical; {t_new * 1e3:.0f} ms vs {t_old * 1e3:.0f} ms")
    ok &= a == b and t_new < t_old
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
