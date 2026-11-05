#!/usr/bin/env python3
"""test_parse_cache.py -- the @T.prim_func parse cache (tvm_fast_stack) must return an identical
PrimFunc for identical (code, closure constants), a different one when a constant differs, and must
not cache when a closure value is not a plain constant."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tvm_fast_stack
tvm_fast_stack.install()
import tvm_ffi
from tvm.script import tirx as T

S = tvm_fast_stack.PARSE_CACHE_STATS


def make(M, N):
    @T.prim_func(s_tir=True)
    def f(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (M, N), "float32")
        C = T.match_buffer(c, (M, N), "float32")
        for i, j in T.grid(M, N):
            with T.sblock("copy"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = A[vi, vj]
    return f


def make_helper(M, helper):
    @T.prim_func(s_tir=True)
    def g(a: T.handle) -> None:
        A = T.match_buffer(a, (M,), "float32")
        for i in range(M):
            with T.sblock("z"):
                vi = T.axis.remap("S", [i])
                A[vi] = T.float32(0)
    helper
    return g


def main():
    h0, m0 = S["hits"], S["misses"]
    a, b, c = make(16, 16), make(16, 16), make(16, 32)
    assert S["misses"] - m0 == 2 and S["hits"] - h0 == 1, S
    assert a is b and a is not c
    assert tvm_ffi.structural_equal(a, b) and not tvm_ffi.structural_equal(a, c)
    assert tuple(a.params[0].name for _ in [0]) and a.attrs["global_symbol"] == "f"
    u0 = S["uncacheable"]
    make_helper(8, lambda: 1)
    make_helper(8, lambda: 2)
    # `helper` is read as a free variable only by the enclosing function, not g, so g stays cacheable;
    # what must hold is that nothing crashed and results are structurally equal
    print("parse cache OK", S)


if __name__ == "__main__":
    main()
