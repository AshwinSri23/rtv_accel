#!/usr/bin/env python3
"""
test_int8_embed.py -- the host-side INT8-per-row embedding / lm_head table (gemma4_gguf.Int8Rows).

Row lookup must be within half a quantisation step of the FP32 row, `table @ v` (chunked lm_head) must equal the
FP32 product of the dequantised table, an all-zero row must stay zero, and the table must be ~4x smaller.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)), "toolchain"))

from gemma4_gguf import Int8Rows  # noqa: E402


def main():
    rng = np.random.default_rng(3)
    t = (rng.standard_normal((40000, 96)) * rng.uniform(0.1, 3, (40000, 1))).astype(np.float32)
    t[7] = 0.0
    q = Int8Rows(t)
    results = []
    err = max(float(np.abs(q[i] - t[i]).max() / q.scale[i]) for i in (0, 5, 123, 39999, 20000))
    results.append(("row lookup within half a step", err <= 0.5 + 1e-4, f"worst {err:.3f} steps"))
    results.append(("zero row stays zero", not np.any(q[7]), ""))
    v = rng.standard_normal(96).astype(np.float32)
    deq = q.q.astype(np.float32) * q.scale[:, None]
    d = float(np.abs(q @ v - deq @ v).max())
    results.append(("chunked matvec == dequantised product", d < 1e-3, f"max |delta| {d:.2e}"))
    rel = float(np.abs(q @ v - t @ v).max() / np.abs(t @ v).max())
    results.append(("matvec close to the FP32 table's", rel < 0.02, f"max rel {rel:.2e}"))
    ratio = (q.q.nbytes + q.scale.nbytes) / t.nbytes
    results.append(("about 4x smaller", ratio < 0.3, f"{ratio:.3f} of FP32 (0.25 + 4 B scale per row)"))
    n = 0
    for name, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {msg}" if msg else ""))
        n += ok
    print(f"\n{n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
