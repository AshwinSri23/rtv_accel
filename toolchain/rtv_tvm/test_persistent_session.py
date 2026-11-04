#!/usr/bin/env python3
"""
test_persistent_session.py -- `rtv_sim --serve`: one process, DRAM resident
across many programs (load weights once, submit command streams).

Checks, with hand-built DMA-copy programs:
  1. DRAM written by program 1 is read by program 2 with NO image reload
     (the cross-program state that generate.py currently round-trips through a
     whole-DRAM file every decode step).
  2. WRITE patches DRAM between programs (host-written per-step constants).
  3. Timing state resets per RUN (same program -> identical cycle counts).
  4. A bad RUN returns ERR and the session keeps working.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_persistent_session.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from rtv_isa import Opcode, encode, to_bytes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")
N = 256  # bytes copied


class Session:
    def __init__(self, dram_mb=2, image=None):
        args = [SIM, "--serve", HAL, str(dram_mb)]
        if image:
            args += ["--dram-image", image]
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, bufsize=1)
        while self.p.stdout.readline().strip() != "READY":
            if self.p.poll() is not None:
                raise RuntimeError("serve exited: " + self.p.stderr.read())

    def cmd(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        return self.p.stdout.readline().strip()

    def close(self):
        self.p.stdin.write("QUIT\n")
        self.p.stdin.flush()
        self.p.wait(timeout=10)


def copy_prog(path, src, dst):
    words = [
        encode(opcode=Opcode.LOAD_TILE, dep_id=1, dram_addr=src, sram_addr=0x8000,
               dim_m=1, dim_n=N, stride=N),
        encode(opcode=Opcode.WAIT_SYNC, dep_id=1),
        encode(opcode=Opcode.STORE_TILE, dram_addr=dst, sram_addr=0x8000,
               dim_m=1, dim_n=N, stride=N),
    ]
    open(path, "wb").write(b"".join(to_bytes(w) for w in words))


def main():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=N, dtype=np.uint8).tobytes()
    results = []
    with tempfile.TemporaryDirectory() as wd:
        img = os.path.join(wd, "img")
        buf = bytearray(0x4000)
        buf[0x1000:0x1000 + N] = data
        open(img, "wb").write(bytes(buf))
        p1, p2, out = (os.path.join(wd, n) for n in ("p1.bin", "p2.bin", "out.bin"))
        copy_prog(p1, 0x1000, 0x2000)   # A -> B
        copy_prog(p2, 0x2000, 0x3000)   # B -> C

        s = Session(image=img)
        try:
            r1 = s.cmd(f"RUN {p1}")
            r2 = s.cmd(f"RUN {p2} 0x3000:{N}:{out}")
            got = open(out, "rb").read()
            results.append(("dram_persists_across_programs",
                            r1.startswith("OK") and r2.startswith("OK") and got == data,
                            f"{r1} | {r2}"))

            new = bytes(range(N))
            nf = os.path.join(wd, "new.bin")
            open(nf, "wb").write(new)
            w = s.cmd(f"WRITE 0x2000 {nf}")
            s.cmd(f"RUN {p2} 0x3000:{N}:{out}")
            results.append(("write_patches_dram", w.startswith("OK") and open(out, "rb").read() == new, w))

            ra, rb = s.cmd(f"RUN {p1}"), s.cmd(f"RUN {p1}")
            results.append(("timing_resets_per_run", ra == rb and ra.startswith("OK"), f"{ra} vs {rb}"))

            bad = s.cmd(f"RUN {os.path.join(wd, 'missing.bin')}")
            good = s.cmd(f"RUN {p1}")
            results.append(("bad_run_returns_err_and_session_survives",
                            bad.startswith("ERR") and good.startswith("OK"), f"{bad} | {good}"))
        finally:
            s.close()

    n = 0
    for name, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  -- {msg}"))
        n += ok
    print(f"\n{n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
