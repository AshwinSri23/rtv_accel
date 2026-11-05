"""
decode_step_cache.py -- derive a decode step's program from the previous one instead of recompiling.

Within one KV-cache tile (16 positions) consecutive decode steps compile to the SAME instruction
stream except for the KV-append store addresses (they move one row, head_dim*4 bytes, per position),
and the compile-time constants whose content depends on the position (the RoPE cos/sin rows and the
KV tail mask). So: compile two consecutive steps of a tile for real, diff them to learn which
instructions move and by how much, and derive every other step of that tile by patching those
addresses and rebuilding those constants (tvm_backend.POS_CONSTS). A new tile (more keys attended)
changes the stream's shape, so the cache starts over there. Anything unexpected in the diff marks
the tile uncacheable and every step in it compiles for real.

test_decode_step_cache.py asserts a derived step equals a full compile exactly (stream and constants).
"""
import numpy as np

from rtv_isa import Opcode


def _addr_view(stream):
    n = len(stream) // 16
    w = np.frombuffer(bytes(stream), dtype=">u8").reshape(n, 2).astype(np.uint64)
    hi = w[:, 0]
    opc = (hi >> np.uint64(60)) & np.uint64(0xF)
    mem = (opc == np.uint64(int(Opcode.LOAD_TILE))) | (opc == np.uint64(int(Opcode.STORE_TILE)))
    addr = ((hi >> np.uint64(24)) & np.uint64(0xFFFFFFFF)).astype(np.int64)
    return w, mem, addr


class DecodeStepCache:
    def __init__(self, tile_dim, x_addr):
        self.M = tile_dim
        self.x_addr = x_addr
        self.stats = {"compiled": 0, "derived": 0}
        self._reset()

    def _reset(self):
        self.tile = None
        self.first = None       # (pos, compiler) -- first step of the tile
        self.ready = None       # (pos, compiler, idx, delta) once two steps are diffed
        self.bad = False

    def derive(self, pos, new_compiler):
        """A derived compiler for `pos`, or None when this step must be compiled for real.
        `new_compiler` is a fresh, configured NPUCompiler."""
        if self.ready is None or pos // self.M != self.tile:
            return None
        pb, base, idx, delta = self.ready
        w, mem, addr = _addr_view(base.instruction_stream)
        new = addr.copy()
        new[idx] += delta * (pos - pb)
        if (new[idx] < 0).any() or (new[idx] > 0xFFFFFFFF).any():
            return None
        hi = w[:, 0].copy()
        hi[idx] = (hi[idx] & ~(np.uint64(0xFFFFFFFF) << np.uint64(24))) | \
            (new[idx].astype(np.uint64) << np.uint64(24))
        w[:, 0] = hi
        stream = bytearray(w.astype(">u8").tobytes())

        from tvm_backend import POS_CONSTS
        regions = dict(base._const_regions)
        for key, (a, nbytes) in base._const_cache.items():
            f = POS_CONSTS.get(key[0])
            if f is not None:
                regions[a] = f(self.M, key, pos).tobytes()
        c = new_compiler
        c.instruction_stream = stream
        c._const_regions = regions
        c._const_cache = dict(base._const_cache)    # addresses only; keys name the base step's pos
        c._dram_cursor = base._dram_cursor
        c._staging_cursor = base._staging_cursor
        c._staging_peak = base._staging_peak
        self.stats["derived"] += 1
        return c

    def observe(self, pos, comp):
        """Record a step that was compiled for real."""
        self.stats["compiled"] += 1
        tile = pos // self.M
        if tile != self.tile:
            self._reset()
            self.tile = tile
            self.first = (pos, comp)
            return
        if self.ready is not None or self.bad or self.first is None:
            return
        p0, c0 = self.first
        if pos != p0 + 1:
            self.first = (pos, comp)
            return
        a, b = c0.instruction_stream, comp.instruction_stream
        wa, mem, aa = _addr_view(a)
        wb, _, ab = _addr_view(b)
        diff = np.nonzero((wa != wb).any(axis=1))[0]
        ok = len(a) == len(b) and mem[diff].all() and len(diff) > 0
        if ok:
            # only the dram_addr field may differ
            mask = ~(np.uint64(0xFFFFFFFF) << np.uint64(24))
            ok = bool(((wa[diff, 0] & mask) == (wb[diff, 0] & mask)).all()
                      and (wa[diff, 1] == wb[diff, 1]).all())
        # constants must sit at the same addresses with the same sizes
        va = sorted((k[0], v) for k, v in c0._const_cache.items())
        vb = sorted((k[0], v) for k, v in comp._const_cache.items())
        ok = ok and va == vb and set(c0._const_regions) == set(comp._const_regions) \
            and c0.dram_bytes_needed == comp.dram_bytes_needed
        if not ok:
            self.bad = True
            return
        self.ready = (pos, comp, diff, (ab - aa)[diff])
