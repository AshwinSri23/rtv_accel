"""
swizzle.py — JIT Weight Swizzler for RTV-Accel

Converts a row-major INT8 weight matrix into tile-contiguous layout so that
LOAD_TILE can use stride=tile (16) instead of stride=padded(full_col_count).

Before swizzle:
    Tile (r, c) in an M×N matrix starts at
        base + r * tile * padded(N) + c * tile
    and requires stride=padded(N) to load tile rows correctly.

After swizzle:
    Tile (r, c) is at
        base + (r * n_tile_cols + c) * tile²
    and requires stride=tile (tile rows are now contiguous).

Effect on DRAM bandwidth:
    Row-major loads: each tile load reads (tile-1) gaps of size (padded(N)-tile)
    between rows → DRAM burst inefficiency.
    Swizzled loads: every 256-byte tile is a single contiguous burst.

Usage:
    python3 toolchain/swizzle.py weights_q.bin 1024 1024 weights_q_swizzled.bin
"""

import math
import struct
import sys

TILE = 16        # must match hal_config.json pe_array_dim and SRAM TILE_BYTES
TILE_BYTES = 256  # TILE * TILE bytes for INT8


# ── Core transform ────────────────────────────────────────────────────────────

def swizzle_matrix(data: bytes, M: int, N: int, tile: int = TILE) -> bytes:
    """
    Rearrange an M×N row-major INT8 weight matrix into tile-contiguous layout.

    data  -- raw bytes, length >= M*N
    M, N  -- logical matrix dimensions (need not be tile-aligned; gaps are zero-padded)
    tile  -- tile edge length (default 16)

    Returns a bytearray of length ceil(M/tile)*tile * ceil(N/tile)*tile.
    Out-of-bounds padding bytes are zero.
    """
    M_pad = math.ceil(M / tile) * tile
    N_pad = math.ceil(N / tile) * tile
    n_tr  = M_pad // tile
    n_tc  = N_pad // tile

    out = bytearray(M_pad * N_pad)  # zero-initialized

    for tr in range(n_tr):
        for tc in range(n_tc):
            tile_idx = tr * n_tc + tc
            dst_base = tile_idx * tile * tile
            for row in range(tile):
                src_row = tr * tile + row
                src_col = tc * tile
                if src_row >= M:
                    break  # remaining rows are padding
                # Copy as many valid columns as exist; rest stays zero
                valid = min(tile, N - src_col)
                if valid <= 0:
                    continue
                src_off = src_row * N + src_col
                dst_off = dst_base + row * tile
                out[dst_off : dst_off + valid] = data[src_off : src_off + valid]

    return bytes(out)


# ── Address arithmetic after swizzle ─────────────────────────────────────────

def n_tile_cols(N: int, tile: int = TILE) -> int:
    """Number of tile columns for a matrix with N columns."""
    return math.ceil(N / tile)


def n_tile_rows(M: int, tile: int = TILE) -> int:
    """Number of tile rows for a matrix with M rows."""
    return math.ceil(M / tile)


def tile_dram_addr(base: int, tile_row: int, tile_col: int,
                   ntc: int, tile: int = TILE) -> int:
    """
    Byte address of tile (tile_row, tile_col) in a swizzled matrix.

    base -- DRAM base address of the swizzled weight matrix
    ntc  -- n_tile_cols for this matrix (= ceil(N / tile))
    """
    return base + (tile_row * ntc + tile_col) * tile * tile


def swizzled_stride(_tile: int = TILE) -> int:
    """LOAD_TILE stride after swizzle: always equal to the tile width."""
    return _tile


# ── Convenience: swizzle + compute new (addr, stride) per tile ───────────────

def swizzle_weight(data: bytes, M: int, N: int,
                   dram_base: int = 0, tile: int = TILE):
    """
    Swizzle an M×N weight matrix and return
        (swizzled_bytes, address_fn)

    where address_fn(tile_row, tile_col) → (dram_addr, stride).

    Example
    -------
        sw_bytes, addr = swizzle_weight(raw, M=1024, N=1024, dram_base=0x100000)
        # mem.write_dram(dram_base, sw_bytes)
        # ...
        # dram_a, stride = addr(step, c)   ← use in LOAD_TILE
    """
    ntc = n_tile_cols(N, tile)
    sw  = swizzle_matrix(data, M, N, tile)

    def address_fn(tile_row: int, tile_col: int):
        return tile_dram_addr(dram_base, tile_row, tile_col, ntc, tile), tile

    return sw, address_fn


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main():
    if len(sys.argv) != 5:
        print("Usage: swizzle.py <input.bin> <M> <N> <output.bin>")
        print()
        print("  Reads a row-major INT8 matrix of shape M×N from input.bin,")
        print("  writes tile-contiguous layout to output.bin.")
        print()
        print("  After swizzle: LOAD_TILE stride = 16 (was padded(N)).")
        sys.exit(1)

    in_path  = sys.argv[1]
    M, N     = int(sys.argv[2]), int(sys.argv[3])
    out_path = sys.argv[4]

    with open(in_path, "rb") as f:
        raw = f.read()

    if len(raw) < M * N:
        print(f"error: {in_path} has {len(raw)} bytes, expected >= {M*N}")
        sys.exit(1)

    swizzled = swizzle_matrix(raw, M, N)

    with open(out_path, "wb") as f:
        f.write(swizzled)

    ntc = n_tile_cols(N)
    ntr = n_tile_rows(M)
    old_stride = math.ceil(N / TILE) * TILE
    print(f"swizzled {M}×{N} matrix ({M*N} bytes → {len(swizzled)} bytes, "
          f"{ntr}×{ntc} tiles)")
    print(f"  LOAD_TILE stride: {old_stride} → {TILE}  "
          f"({(old_stride - TILE) * TILE} wasted bytes/tile eliminated)")


if __name__ == "__main__":
    _main()
