#!/usr/bin/env python3
"""
test_dram36.py -- ISA v1.6 36-bit DRAM addressing (dim_k[15:12] = address bits
35:32 on LOAD_TILE/STORE_TILE).

1. rtv_isa.encode/decode round-trip a >32-bit address, leave c_slot/dim_k
   alone, reject a >32-bit address on opcodes that use dram_addr as an
   immediate, and decode every pre-v1.6 word with dram_hi == 0.
2. End to end: a LOAD_TILE whose address is 2**32 + small must NOT wrap to the
   low 32 bits (which would silently read real data). With a small simulated
   DRAM the simulator must report an out-of-range error instead.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_dram36.py
"""
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from rtv_isa import Opcode, encode, decode, to_bytes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")


def case_roundtrip():
    addr = (0xB << 32) | 0xDEADBEEF
    f = decode(encode(opcode=Opcode.LOAD_TILE, dram_addr=addr, dim_m=1,
                      dim_n=64, dim_k=0x21, c_slot=5))
    assert f["dram_hi"] == 0xB and f["dram_addr"] == 0xDEADBEEF, f
    assert f["c_slot"] == 5 and f["dim_k"] == 0x21, f
    assert decode(encode(opcode=Opcode.LOAD_TILE, dram_addr=0x1000))["dram_hi"] == 0
    for bad in (dict(opcode=Opcode.EXEC_ALU, dram_addr=1 << 32),
                dict(opcode=Opcode.LOAD_TILE, dram_addr=1 << 36)):
        try:
            encode(**bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
    return True, ""


def case_no_wrap():
    word = encode(opcode=Opcode.LOAD_TILE, dram_addr=(1 << 32) + 0x100,
                  sram_addr=0, dim_m=1, dim_n=16, stride=16)
    with tempfile.TemporaryDirectory() as wd:
        p, d = os.path.join(wd, "p.bin"), os.path.join(wd, "d.img")
        open(p, "wb").write(to_bytes(word))
        open(d, "wb").write(bytes(1 << 20))
        r = subprocess.run([SIM, p, HAL, "2", "--dram-image", d],
                           capture_output=True, text=True)
    if r.returncode == 0:
        return False, "36-bit address wrapped to 32 bits (load succeeded)"
    if "out of range" not in (r.stdout + r.stderr):
        return False, f"unexpected failure: {(r.stdout + r.stderr)[-200:]}"
    return True, ""


def main():
    n = 0
    cases = [("encode_decode_roundtrip", case_roundtrip), ("no_32bit_wrap", case_no_wrap)]
    for name, fn in cases:
        try:
            ok, msg = fn()
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {e}"
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {msg}" if msg else ""))
        n += ok
    print(f"\n{n}/{len(cases)} passed")
    return 0 if n == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
