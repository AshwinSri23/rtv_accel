"""
tvm_fast_stack.py -- make TVMScript's @T.prim_func / @R.function / @I.ir_module decorators cheap.

Each decorator calls inspect.stack(), which builds a FrameInfo for EVERY frame on the stack and, for
each, resolves the source file and module (getframeinfo -> getmodule) and reads source lines. This
compiler builds ~3,000 PrimFuncs per run from deep recursive emitters; profiled on a 6-layer run that
was ~62 s of 205 s (cProfile-inflated), none of it TVM work.

TVM only uses two things from that stack: each frame object (closure-variable lookup) and
`code_context` of ONE frame (index 2, to detect a class-scope annotator). So build the same list of
FrameInfo objects but fill code_context for that one frame only (via the real getframeinfo). Behaviour is unchanged;
`install()` is idempotent and patches only the `inspect` name inside TVM's three decorator modules.
"""
import inspect
import sys
import types

_CONTEXT_INDEX = 2          # tvm.script.parser.core.utils.is_defined_in_class reads frames[2]


def fast_stack(context=1):
    frames = []
    f = sys._getframe(1)
    i = 0
    while f is not None:
        code = f.f_code
        if i == _CONTEXT_INDEX and context:
            # the one frame whose code_context TVM reads: use the real getframeinfo (on Python
            # 3.13+ it takes the line from the instruction position, which differs from f_lineno
            # for decorated class statements; reproducing that by hand broke @ir_module)
            fi = inspect.getframeinfo(f, context)
            frames.append(inspect.FrameInfo(f, fi.filename, fi.lineno, fi.function,
                                            fi.code_context, fi.index, positions=fi.positions))
        else:
            frames.append(inspect.FrameInfo(f, code.co_filename, f.f_lineno, code.co_name,
                                            None, None))
        f = f.f_back
        i += 1
    return frames


class _InspectProxy(types.ModuleType):
    """`inspect`, except stack() is fast_stack()."""

    def __init__(self):
        super().__init__("inspect")

    def __getattr__(self, name):
        return getattr(inspect, name)

    stack = staticmethod(fast_stack)      # not a wrapper: fast_stack's frame 0 must be the caller


_installed = False


def install():
    global _installed
    if _installed:
        return
    import importlib
    proxy = _InspectProxy()
    for mod in ("tvm.tirx.script.parser.entry", "tvm.relax.script.parser.entry",
                "tvm.script.parser.ir.entry"):
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        if getattr(m, "inspect", None) is inspect:
            m.inspect = proxy
    _installed = True
