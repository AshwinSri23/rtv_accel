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


PARSE_CACHE_STATS = {"hits": 0, "misses": 0, "uncacheable": 0}
_PARSE_CACHE = {}
_SIMPLE = (int, float, str, bool, type(None), bytes)


def _simple(v):
    if isinstance(v, _SIMPLE):
        return True
    if isinstance(v, (tuple, list)):
        return all(_simple(x) for x in v)
    return False


def _freeze(v):
    return tuple(_freeze(x) for x in v) if isinstance(v, (tuple, list)) else v


def _parse_key(func, kw):
    """(code object, closure values, decorator kwargs), or None when any closure value is not a
    plain constant (then the parse could depend on something we cannot compare)."""
    if func.__defaults__ or func.__kwdefaults__:
        return None
    vals = []
    for cell in func.__closure__ or ():
        try:
            v = cell.cell_contents
        except ValueError:
            return None
        if not _simple(v):
            return None
        vals.append((type(v).__name__, _freeze(v)))
    return (func.__code__, tuple(vals), tuple(sorted(kw.items())))


def _install_parse_cache():
    """Memoize @T.prim_func: PrimFuncs are immutable (scheduling builds new modules), and the
    emitters re-declare the same function body with the same closure constants thousands of times."""
    import os
    from tvm.script import tirx as T
    orig = T.prim_func
    if getattr(orig, "_rtv_cached", False) or os.environ.get("RTV_NO_PARSE_CACHE"):
        return

    def cached(func=None, **kw):
        dec = orig(**kw) if func is None else None

        def apply(f):
            if not inspect.isfunction(f):
                return (dec or orig(**kw))(f) if dec else orig(f, **kw)
            try:
                key = _parse_key(f, kw)
            except Exception:
                key = None
            if key is None:
                PARSE_CACHE_STATS["uncacheable"] += 1
                return dec(f) if dec else orig(f, **kw)
            hit = _PARSE_CACHE.get(key)
            if hit is not None:
                PARSE_CACHE_STATS["hits"] += 1
                return hit
            PARSE_CACHE_STATS["misses"] += 1
            r = dec(f) if dec else orig(f, **kw)
            _PARSE_CACHE[key] = r
            return r
        apply.dispatch_token = "tirx"      # the parser identifies the decorator by this
        return apply if func is None else apply(func)

    cached._rtv_cached = True
    cached.dispatch_token = getattr(orig, "dispatch_token", "tirx")
    T.prim_func = cached


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
    _install_parse_cache()
    _installed = True
