#!/usr/bin/env python3
"""test_decode_step_cache.py -- a derived decode step must equal a full compile exactly: same
instruction stream, same compile-time constants (and the same DRAM size), for every position of a
multi-tile run; tile boundaries must start over; and most steps must actually be derived."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE)); sys.path.insert(0, _HERE)
import test_byoc_decoder as T
import tvm_backend, decoder_graph
from decode_step_cache import DecodeStepCache


def compile_step(m, n_layers, pos):
    c = T.fresh(m["base"])
    c.reuse_layer_staging = True
    em = decoder_graph.make_layer_emitter(T.CFG, m["lws"], pos, T.THETA, T.SCALE)
    tvm_backend._emit_model_kvcache_fp32_relocated(
        c, n_layers, T.HID, T.INTER, T.NH, T.NKV, T.HD, x_addr=m["x"], layer_weights=m["lws"],
        scratch_a=m["sa"], scratch_b=m["sb"], pos=pos, rope_theta=T.THETA, attn_scale=T.SCALE,
        layer_emitter=em)
    return c


def main():
    m = T.build(3, 24, max_seq=40)
    cache = DecodeStepCache(16, m["x"])
    bad = []
    for pos in range(0, 36):
        ref = compile_step(m, 3, pos)
        d = cache.derive(pos, T.fresh(m["base"]))
        if d is None:
            cache.observe(pos, ref)
            continue
        if bytes(d.instruction_stream) != bytes(ref.instruction_stream):
            bad.append((pos, "stream"))
        elif {a: bytes(v) for a, v in d._const_regions.items()} != \
                {a: bytes(v) for a, v in ref._const_regions.items()}:
            bad.append((pos, "constants"))
        elif d.dram_bytes_needed != ref.dram_bytes_needed:
            bad.append((pos, "dram size"))
    s = cache.stats
    ok = not bad and s["derived"] >= 24 and s["compiled"] <= 12
    print(("PASS" if ok else "FAIL"), "decode_step_cache", s, bad)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
