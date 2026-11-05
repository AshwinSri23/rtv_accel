#!/usr/bin/env python3
"""
test_dram_image_4gb.py -- a DRAM image larger than 4 GiB loads intact and is readable above 4 GiB.

Regression for the simulator's host-side image loader/dump paths: chunk offsets, WRITE and --dump addresses were
uint32_t, so for a multi-GB model image (Gemma-4-26B is 16 GB) every chunk past 4 GiB silently overwrote low DRAM
(garbage weights -> NaN). The image here is sparse (a few marker pages), 4.25 GiB of simulated DRAM.

Run:  .venv-tvm/bin/python toolchain/rtv_tvm/test_dram_image_4gb.py
"""
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")

LOW, HIGH = 0x1000, (1 << 32) + 0x2000             # a marker page below 4 GiB and one above
SIZE = (1 << 32) + (256 << 20)


def main():
    with tempfile.TemporaryDirectory() as wd:
        img, prog = os.path.join(wd, "i.img"), os.path.join(wd, "p.bin")
        with open(img, "wb") as f:
            f.seek(LOW); f.write(b"LOWLOWLO" * 64)
            f.seek(HIGH); f.write(b"HIGHHIGH" * 64)
            f.truncate(SIZE - (64 << 20))
        open(prog, "wb").write(b"")
        out_lo, out_hi = os.path.join(wd, "lo.bin"), os.path.join(wd, "hi.bin")
        r = subprocess.run([SIM, prog, HAL, str(SIZE >> 20), "--dram-image", img,
                            "--dump", f"{LOW}:512:{out_lo}", "--dump", f"{HIGH}:512:{out_hi}"],
                           capture_output=True, text=True)
        lo = open(out_lo, "rb").read() if os.path.exists(out_lo) else b""
        hi = open(out_hi, "rb").read() if os.path.exists(out_hi) else b""
    results = [("low marker intact (not overwritten by the >4 GiB chunks)", lo == b"LOWLOWLO" * 64, r.stderr[-200:]),
               ("marker above 4 GiB readable", hi == b"HIGHHIGH" * 64, "")]
    n = 0
    for name, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {msg}" if msg and not ok else ""))
        n += ok
    print(f"\n{n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
