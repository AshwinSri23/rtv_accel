#!/usr/bin/env python3
"""
test_swiglu_resident.py — tvm_backend._emit_swiglu_ffn_resident_fp32, TODO
1.2's last open item: SwiGLU's gate/up GEMM outputs kept resident in SRAM
across SiLU/ELEMWISE_MUL instead of round-tripping through DRAM between
every op.

Legitimacy check done before writing this (not assumed): COMPUTE_GEMM's C
write is c_slot-ADDRESSED in real hardware (src/simulator.cpp's
c_slot_addr, ISA v1.5, 16 independent 1KiB slots) and EXEC_ALU's operand
addresses are genuinely arbitrary SRAM addresses with no zone restriction
(src/pe_array.cpp's exec_alu just does mem.sram_ptr(sram_addr)) -- so
pointing an elementwise op directly at a GEMM's own c_slot output, with no
LOAD_TILE/STORE_TILE in between, is real hardware behavior, not a
simulator-only shortcut. codegen.py's Codegen gained two small, general
hooks for this: `skip_tile_io` (buffer short-names for which _tile_copy
emits nothing -- the data is already at the right address) and `c_slot`
(COMPUTE_GEMM's actual hardware output slot, previously hardcoded to 0).

Two things checked, same bar as the GEMM double-buffering work
(test_gemm_double_buffer.py): exact correctness against a numpy reference
AND a genuine cycle-count improvement via rtv_sim's own serial-vs-db
model, not just "doesn't crash" -- skipping real DRAM round trips should
show up as fewer cycles, and if it doesn't, that's a sign the "resident"
path isn't actually resident.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_swiglu_resident.py
"""

import os
import re
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from compiler_state import NPUCompiler, DType  # noqa: E402
import tvm_backend                           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
SIM = os.path.join(ROOT, "build", "rtv_sim")
HAL = os.path.join(ROOT, "toolchain", "hal_config.json")

M = 16


def _swiglu_ref(x_row, w_gate, w_up, w_down):
    g = x_row.astype(np.float64) @ w_gate.astype(np.float64)
    u = x_row.astype(np.float64) @ w_up.astype(np.float64)
    act = (g / (1.0 + np.exp(-g))) * u
    return (act @ w_down.astype(np.float64)).astype(np.float32)


def _run_sim(instr_stream, dram, addr, nbytes, shape):
    with tempfile.TemporaryDirectory(prefix="rtv_swiglu_res_") as wd:
        p, d = os.path.join(wd, "p.bin"), os.path.join(wd, "d.img")
        open(p, "wb").write(bytes(instr_stream))
        open(d, "wb").write(bytes(dram))
        out = os.path.join(wd, "o.bin")
        mb = max(1, len(dram) // (1024 * 1024) + 2)
        r = subprocess.run([SIM, p, HAL, str(mb), "--dram-image", d,
                            "--dump", f"{addr}:{nbytes}:{out}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"simulator failed: {r.stdout[-500:]}{r.stderr[-500:]}")
        return (np.frombuffer(open(out, "rb").read(), dtype=np.float32).reshape(shape),
                r.stdout)


def _db_cycles(stdout):
    # "db" (rtv_sim's own label) is sim_clock -- the REAL simulated cycle
    # count for exactly this instruction stream, reflecting whatever
    # overlap its own WAIT_SYNC placements actually achieve. "serial" is
    # just a fixed no-overlap reference cost for the same instruction MIX,
    # not sensitive to how well an emission pipelines things -- comparing
    # plain vs resident needs "db", the number that actually moves.
    m = re.search(r"db\s+(\d+)", stdout)
    if not m:
        raise RuntimeError(f"couldn't find a db-cycle estimate in:\n{stdout[-800:]}")
    return int(m.group(1))


def _build(compiler_fn, hidden, inter, x_row, w_gate, w_up, w_down):
    x = np.zeros((M, hidden), dtype=np.float32)
    x[0, :] = x_row

    addrs = {}
    base = [0x1000]

    def place(name, arr):
        addrs[name] = base[0]
        base[0] += arr.nbytes + 0x1000
        return addrs[name]

    place("x", x)
    place("wg", w_gate)
    place("wu", w_up)
    place("wd", w_down)
    out_addr = base[0]
    base[0] += M * hidden * 4 + 0x1000

    comp = NPUCompiler(HAL)
    comp.dtype = DType.FP32
    comp.instruction_stream = bytearray()
    comp._dram_cursor = base[0] + 0x10000

    compiler_fn(comp, hidden, inter, x_addr=addrs["x"], w_gate_addr=addrs["wg"],
               w_up_addr=addrs["wu"], w_down_addr=addrs["wd"], out_addr=out_addr)
    n_instr = len(comp.instruction_stream) // 16

    dram_size = max(base[0], comp.dram_bytes_needed) + 0x10000
    dram = bytearray(dram_size)
    dram[addrs["x"]:addrs["x"] + x.nbytes] = x.tobytes()
    dram[addrs["wg"]:addrs["wg"] + w_gate.nbytes] = w_gate.tobytes()
    dram[addrs["wu"]:addrs["wu"] + w_up.nbytes] = w_up.tobytes()
    dram[addrs["wd"]:addrs["wd"] + w_down.nbytes] = w_down.tobytes()
    for addr, data in comp._const_regions.items():
        dram[addr:addr + len(data)] = data

    got, stdout = _run_sim(comp.instruction_stream, dram, out_addr, M * hidden * 4, (M, hidden))
    return got, n_instr, _db_cycles(stdout)


def run_correctness_case(hidden, inter, seed=41):
    rng = np.random.default_rng(seed)
    x_row = rng.normal(size=(hidden,)).astype(np.float32) * 0.4
    w_gate = (rng.normal(size=(hidden, inter)) * 0.15).astype(np.float32)
    w_up = (rng.normal(size=(hidden, inter)) * 0.15).astype(np.float32)
    w_down = (rng.normal(size=(inter, hidden)) * 0.15).astype(np.float32)

    expected_row = _swiglu_ref(x_row, w_gate, w_up, w_down)
    expected = np.zeros((M, hidden), dtype=np.float32)
    expected[0, :] = expected_row

    got, n_instr, cycles = _build(tvm_backend._emit_swiglu_ffn_resident_fp32,
                                  hidden, inter, x_row, w_gate, w_up, w_down)
    if not np.allclose(got, expected, atol=1e-2):
        d = np.abs(got - expected)
        return False, f"max delta {float(d.max()):.4g}", n_instr
    return True, "", n_instr


def run_speedup_case(hidden, inter, min_speedup, seed=43):
    rng = np.random.default_rng(seed)
    x_row = rng.normal(size=(hidden,)).astype(np.float32) * 0.4
    w_gate = (rng.normal(size=(hidden, inter)) * 0.15).astype(np.float32)
    w_up = (rng.normal(size=(hidden, inter)) * 0.15).astype(np.float32)
    w_down = (rng.normal(size=(inter, hidden)) * 0.15).astype(np.float32)
    expected_row = _swiglu_ref(x_row, w_gate, w_up, w_down)
    expected = np.zeros((M, hidden), dtype=np.float32)
    expected[0, :] = expected_row

    got_plain, n_plain, cyc_plain = _build(tvm_backend._emit_swiglu_ffn_fp32,
                                           hidden, inter, x_row, w_gate, w_up, w_down)
    got_res, n_res, cyc_res = _build(tvm_backend._emit_swiglu_ffn_resident_fp32,
                                     hidden, inter, x_row, w_gate, w_up, w_down)

    if not (np.allclose(got_plain, expected, atol=1e-2) and
            np.allclose(got_res, expected, atol=1e-2)):
        return False, "one of the two paths didn't match the reference", n_res
    if not np.allclose(got_plain, got_res, atol=1e-4):
        d = np.abs(got_plain - got_res)
        return False, f"resident and non-resident paths disagree, max delta {float(d.max()):.4g}", n_res

    speedup = cyc_plain / cyc_res
    msg = f"speedup {speedup:.2f}x (plain={cyc_plain}, resident={cyc_res}, instrs {n_plain}->{n_res})"
    if speedup < min_speedup:
        return False, f"{msg} -- below required {min_speedup}x", n_res
    return True, msg, n_res


def run_capacity_error_case(hidden=32, inter=17 * 16):
    """2*(inter/16) would need 34 c_slots; must raise, not silently misbehave."""
    try:
        comp = NPUCompiler(HAL)
        comp.dtype = DType.FP32
        comp.instruction_stream = bytearray()
        tvm_backend._emit_swiglu_ffn_resident_fp32(
            comp, hidden, inter, x_addr=0x1000, w_gate_addr=0x2000,
            w_up_addr=0x3000, w_down_addr=0x4000, out_addr=0x5000)
    except NotImplementedError:
        return True, "", 0
    return False, "expected NotImplementedError for a too-large inter, got none", 0


def main():
    cases = [
        ("swiglu_resident_correct_48", lambda: run_correctness_case(32, 48)),
        ("swiglu_resident_correct_96", lambda: run_correctness_case(64, 96, seed=17)),
        ("swiglu_resident_speedup_96", lambda: run_speedup_case(64, 96, min_speedup=1.05)),
        ("swiglu_resident_capacity_error", run_capacity_error_case),
    ]
    n_pass = 0
    for name, fn in cases:
        try:
            ok, msg, n_instr = fn()
        except Exception as e:  # noqa: BLE001
            ok, msg, n_instr = False, f"{type(e).__name__}: {e}", 0
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({n_instr} instrs)"
              + (f"  -- {msg}" if msg else ""))
        n_pass += ok
    print(f"\n{n_pass}/{len(cases)} passed")
    return 0 if n_pass == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
