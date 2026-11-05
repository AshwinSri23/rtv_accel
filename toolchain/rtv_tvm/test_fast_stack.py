#!/usr/bin/env python3
"""
test_fast_stack.py -- tvm_fast_stack.fast_stack() must give TVMScript what inspect.stack() gives it:
the same frames in the same order, and the same code_context for frame 2 (the only one TVM reads).
Also checks the cheap stack is actually cheaper, and that an @ir_module class holding @R.function and
@T.prim_func parses with the shim installed (the case an off-by-one frame broke).

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_fast_stack.py
"""
import inspect
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import tvm_fast_stack as F          # noqa: E402
import tvm_backend                  # noqa: E402,F401  (installs the shim)
import tvm                          # noqa: E402
from tvm.script import ir as I, relax as R, tirx as T   # noqa: E402


def deep(n, fn):
    return fn() if n == 0 else deep(n - 1, fn)


def case_same_frames():
    a, b = [], []

    def grab():
        a.extend(inspect.stack())
        b.extend(F.fast_stack())
    deep(30, grab)
    same_frames = [x.frame for x in a] == [x.frame for x in b][:len(a)] or \
        [x.frame.f_code for x in a][1:] == [x.frame.f_code for x in b][1:]
    # frame 0 of fast_stack is its caller (grab); inspect.stack() called from grab is the same
    ctx_a, ctx_b = a[2].code_context, b[2].code_context
    return (len(a) == len(b) and [x.frame for x in a] == [x.frame for x in b] and ctx_a == ctx_b
            and (a[2].lineno, a[2].function) == (b[2].lineno, b[2].function)), \
        f"{len(a)} frames, frame-2 context {ctx_b}"


def case_faster():
    def timeit(f):
        out = []
        t = time.perf_counter()
        for _ in range(50):
            deep(60, lambda: out.append(f()))
        return time.perf_counter() - t
    slow, fast = timeit(inspect.stack), timeit(F.fast_stack)
    return fast < slow, f"inspect.stack {slow * 1000:.0f} ms vs fast_stack {fast * 1000:.0f} ms for 50 calls at depth 60"


def case_ir_module_parses():
    @I.ir_module
    class Mod:
        @T.prim_func(s_tir=True)
        def f(a: T.handle, c: T.handle) -> None:
            A = T.match_buffer(a, (16, 16), "float32")
            C = T.match_buffer(c, (16, 16), "float32")
            for i, j in T.grid(16, 16):
                with T.sblock("copy"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = A[vi, vj]

        @R.function
        def main(x: R.Tensor((16, 16), "float32")) -> R.Tensor((16, 16), "float32"):
            return x
    return isinstance(Mod, tvm.IRModule) and "main" in [g.name_hint for g in Mod.functions], "parsed"


def main():
    cases = [("same_frames_and_frame2_context", case_same_frames), ("faster", case_faster),
             ("ir_module_with_relax_and_tir_parses", case_ir_module_parses)]
    n = 0
    for name, fn in cases:
        try:
            ok, msg = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok, msg = False, f"{type(e).__name__}: {e}"
        print(f"{'PASS' if ok else 'FAIL'}  {name}  -- {msg}")
        n += ok
    print(f"\n{n}/{len(cases)} passed")
    return 0 if n == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
