#!/usr/bin/env python3
"""
scale_sim.py — INT8 inference with explicit scale bookkeeping.

quant_sim.py models the device AS BUILT. This models the device AS PROPOSED:
every tensor carries the scale that says what one integer step is worth, and
no operation ever assumes one.

Three rules, each fixing a defect documented in the spec:

  1. Two buffers meeting at an add must share a scale. The residual add
     currently combines the residual stream (one step = 0.0025) with o_proj's
     output (one step = 0.00059) by adding the raw codes.

  2. A nonlinearity must be evaluated at the TRUE value, not the code.
     `SILU` currently computes x/(1+exp(-x)) on the code, which for all 256
     INT8 codes is indistinguishable from ReLU. Modelled here as a lookup
     table -- 256 entries, built at compile time for a known input scale,
     which is how real accelerators do nonlinearities.

  3. Softmax's temperature is a consequence of the score scale, not a
     constant. The device's fixed T=64 is correct only if one score step is
     worth 1/64, which nothing establishes.

Scales are DYNAMIC here (taken from the values actually seen). That is the
upper bound on what static calibration can achieve; if the model does not
work with dynamic scales it will not work with calibrated ones, so this is
the cheapest possible test of the design.
"""

import numpy as np


def clamp8(v):
    return np.clip(v, -128, 127)


class QT:
    """A quantised tensor: value ~= q * s, with q integer-valued."""

    __slots__ = ("q", "s")

    def __init__(self, q, s):
        # float64, not float32: q holds INTEGER codes, and float32's 24-bit
        # mantissa stops representing them exactly above 16777216. With a
        # 32-bit residual the codes exceed that, so a float32 store silently
        # capped the experiment at ~24 bits and made wide configurations look
        # far worse than they are.
        self.q = np.asarray(q, dtype=np.float64)   # holds integers
        self.s = s if isinstance(s, np.ndarray) else float(s)

    @property
    def value(self):
        return (self.q * self.s).astype(np.float32)

    PER_TOKEN = True   # one scale per row (per position) rather than per tensor

    @staticmethod
    def of(x, bits=None):
        """
        Quantise a float array, choosing the scale from its own peak.

        PER_TOKEN gives each row its own scale. This matters enormously and is
        not a refinement: Qwen3 has "massive activation" channels -- at layer 2
        channel 35 of down_proj reaches 2610 while the median channel is 0.065,
        a 40000:1 ratio, and only 8 of 1024 channels exceed 1% of the peak.
        Those outliers are concentrated at particular POSITIONS, so one scale
        spanning a whole prefill tile is set by the outlier row and rounds
        every other row to zero. Per row, each position keeps its own range.

        It also matches the device: decode processes one token per step, so a
        per-tensor scale there IS a per-token scale.
        """
        x = np.asarray(x, dtype=np.float64)
        if FLOAT_MODE:
            return QT(x, 1.0)          # value == q, scale 1: exact
        if bits is None:
            bits = ACT_BITS
        lim = (1 << (bits - 1)) - 1
        if QT.PER_TOKEN and x.ndim >= 2:
            pk = np.abs(x).max(axis=tuple(range(1, x.ndim)), keepdims=True)
            sc = np.where(pk > 0, pk / lim, 1.0).astype(np.float32)
            return QT(np.clip(np.round(x / sc), -lim, lim), sc)
        pk = float(np.abs(x).max())
        if pk <= 0:
            return QT(np.zeros_like(x), 1.0)
        s = pk / lim
        return QT(np.clip(np.round(x / s), -lim, lim), s)

    def to(self, s_target):
        """Convert to another scale -- the RESCALE instruction."""
        lim = (1 << (RESIDUAL_BITS - 1)) - 1
        r = np.round(self.q * (np.asarray(self.s) / np.asarray(s_target)))
        return QT(np.clip(r, -lim - 1, lim), s_target)

    def _bcast(self, other_shape):
        return self.s


def q_matmul(a: QT, w_q, w_scale_per_col, s_out=None):
    """
    INT8 GEMM with per-output-channel weight scales.

    acc[:, c] is in units of a.s * w_scale[c]; the correction restores the
    relative magnitudes and lands the result in one common output scale --
    PERCHAN_QUANT's job.
    """
    if FLOAT_MODE:
        # int64 would truncate unquantised weights to zero.
        true = a.value @ np.asarray(w_q, dtype=np.float32)
        return QT(true, 1.0)
    acc = a.q.astype(np.int64) @ w_q.astype(np.int64)          # exact INT32-ish
    true = acc.astype(np.float32) * np.asarray(a.s) * w_scale_per_col[None, :]
    if s_out is None:
        return QT.of(true)
    return QT(clamp8(np.round(true / s_out)), s_out)


def quantize_per_channel(w):
    """W as (in, out): one scale per output column."""
    if FLOAT_MODE:
        return w, np.ones(w.shape[1], dtype=np.float32)
    lim = (1 << (WEIGHT_BITS - 1)) - 1
    pk = np.abs(w).max(axis=0)
    pk = np.where(pk > 0, pk, 1.0)
    q = np.clip(np.round(w * (lim / pk[None, :])), -lim, lim)
    return q, (pk / lim)


def lut_apply(x: QT, fn, s_out=None):
    """
    Nonlinearity via a 256-entry lookup table indexed by the INT8 code.

    The table is built for x's scale, so fn is evaluated at the true value.
    Applying fn to the code directly is what makes the current SILU identical
    to ReLU for every input in the INT8 domain.
    """
    if FLOAT_MODE:
        return QT(fn(x.value), 1.0)
    if isinstance(x.s, np.ndarray):
        # Per-token scales mean a table per row; evaluate directly instead.
        return QT.of(fn(x.value))
    codes = np.arange(-128, 128, dtype=np.float32)
    table = fn(codes * x.s)
    if s_out is None:
        pk = float(np.abs(table).max())
        s_out = pk / 127.0 if pk > 0 else 1.0
    tq = clamp8(np.round(table / s_out))
    idx = (x.q + 128).astype(np.int32)
    return QT(tq[idx], s_out)


RESIDUAL_BITS = 8   # width of the residual stream; 8 = as designed
RESIDUAL_PER_CHANNEL = False
# Per-token scaling handles outliers that sit at particular POSITIONS, but
# Qwen3's massive activations are also concentrated in particular CHANNELS
# (layer 2, down_proj channel 35 = 2610 against a median of 0.065). Within one
# row that 40000:1 range leaves typical channels ~0.25% precision even at 24
# bits, and it compounds over 28 layers. A per-channel scale on the residual
# gives the outlier channel its own range and leaves the rest intact.
ACT_BITS = 8        # width of every other activation buffer
FLOAT_MODE = False  # bypass ALL quantisation; must then reproduce ref_model
# Apply RMSNorm's learned gain by folding it into the following weight, the
# way ingest_model's GAMMA_FOLD does, instead of multiplying the activation.
# Identical in exact arithmetic -- gamma is diagonal and commutes through the
# matmul -- but NOT identical once quantised, because it changes what gets
# quantised: folding scales W's columns by a gain spanning 0.017..192, and a
# per-output-channel scale cannot serve columns that far apart.
#
# This exists to settle whether the compiler's folding is why the device
# diverges from this model at layer 0, or whether folding is fine and the
# divergence is a lowering bug elsewhere.
FOLD_GAMMA = False
WEIGHT_BITS = 8     # weight width; raise it to separate weight loss from
                    # activation loss when a disagreement survives wide
                    # activations


def q_add(a: QT, b: QT, s_out=None):
    """Residual add. Both operands are brought to a common scale first."""
    if FLOAT_MODE:
        return QT(a.value + b.value, 1.0)
    if s_out is None:
        # Size the result so the sum cannot clip.
        t = a.value + b.value
        if RESIDUAL_PER_CHANNEL and t.ndim >= 2:
            pk = np.abs(t).max(axis=0, keepdims=True)          # per channel
            s_out = np.where(pk > 0, pk / 127.0, 1.0).astype(np.float32)
        elif QT.PER_TOKEN and t.ndim >= 2:
            pk = np.abs(t).max(axis=tuple(range(1, t.ndim)), keepdims=True)
            s_out = np.where(pk > 0, pk / 127.0, 1.0).astype(np.float32)
        else:
            pk = float(np.abs(t).max())
            s_out = pk / 127.0 if pk > 0 else 1.0
        lim = (1 << (RESIDUAL_BITS - 1)) - 1
        s_out = s_out * (127.0 / lim)
    lim = (1 << (RESIDUAL_BITS - 1)) - 1
    r = np.round(a.to(s_out).q + b.to(s_out).q)
    return QT(np.clip(r, -lim - 1, lim), s_out)


class ScaleQwen3:
    """Qwen3 in INT8 with every scale tracked explicitly."""

    def __init__(self, model):
        self.m = model
        c = model.config()
        self.n_layers = c["num_hidden_layers"]
        self.nh  = c["num_attention_heads"]
        self.nkv = c.get("num_key_value_heads", self.nh)
        self.hd  = c.get("head_dim", c["hidden_size"] // self.nh)
        self.eps = c.get("rms_norm_eps", 1e-6)
        self.theta = c.get("rope_theta", 1e6)
        self.hidden = c["hidden_size"]
        self._wq = {}

    def _f(self, n):
        return np.asarray(self.m.tensor(n), dtype=np.float32)

    def _w(self, n, gamma=None, tag=None):
        key = tag or n
        if key not in self._wq:
            w = self._f(n)
            if gamma is not None:
                w = w * gamma[None, :]        # HF (out,in); gamma indexes `in`
            self._wq[key] = quantize_per_channel(w.T)   # (out,in) -> (in,out)
        return self._wq[key]

    def _proj(self, a: QT, name, gamma=None, tag=None):
        wq, ws = self._w(name, gamma, tag)
        return q_matmul(a, wq, ws)

    def _rmsnorm(self, x: QT, gamma):
        """
        RMSNorm, matching ref_model exactly.

        Computed on the VALUE, not the code. x/rms(x) is scale-invariant in
        exact arithmetic, so operating on codes looks equivalent -- but eps is
        not scale-invariant. Adding eps to mean(code^2) adds eps*s^2 to
        mean(value^2), a different regulariser, and the resulting per-layer
        bias accumulates: at 24-bit weights and activations it dragged the
        layer-27 cosine against ref_model down to 0.90 with no quantisation
        left to blame.
        """
        v = x.value
        inv = 1.0 / np.sqrt((v * v).mean(axis=-1, keepdims=True) + self.eps)
        return QT.of(v * inv * gamma)

    def forward(self, ids, upto_layer=None):
        n = self.n_layers if upto_layer is None else upto_layer
        T = len(ids)
        hd, nh, nkv = self.hd, self.nh, self.nkv
        x = QT.of(self._f("model.embed_tokens.weight")[list(ids)])
        states = [x]

        half = hd // 2
        invf = self.theta ** (-np.arange(0, half, dtype=np.float64) * 2.0 / hd)
        ang = np.outer(np.arange(T, dtype=np.float64), invf)
        cos = np.concatenate([np.cos(ang), np.cos(ang)], 1).astype(np.float32)
        sin = np.concatenate([np.sin(ang), np.sin(ang)], 1).astype(np.float32)
        mask = np.triu(np.ones((T, T), bool), 1)

        for i in range(n):
            b = f"model.layers.{i}."
            g_in = self._f(b + "input_layernorm.weight")
            fold = g_in if FOLD_GAMMA else None
            y = self._rmsnorm(x, np.ones_like(g_in) if FOLD_GAMMA else g_in)
            q = self._proj(y, b + "self_attn.q_proj.weight", fold, b + "qf")
            k = self._proj(y, b + "self_attn.k_proj.weight", fold, b + "kf")
            v = self._proj(y, b + "self_attn.v_proj.weight", fold, b + "vf")

            qv = q.value.reshape(T, nh, hd)
            kv = k.value.reshape(T, nkv, hd)
            vv = v.value.reshape(T, nkv, hd)

            def hnorm(t, g):
                inv = 1.0 / np.sqrt((t * t).mean(-1, keepdims=True) + self.eps)
                return t * inv * g
            qv = hnorm(qv, self._f(b + "self_attn.q_norm.weight"))
            kv = hnorm(kv, self._f(b + "self_attn.k_norm.weight"))

            def rope(t):
                r = np.concatenate([-t[..., half:], t[..., :half]], -1)
                return t * cos[:, None, :] + r * sin[:, None, :]
            qv, kv = QT.of(rope(qv)), QT.of(rope(kv))
            vq = QT.of(vv)

            rep = nh // nkv
            Q = qv.value.transpose(1, 0, 2)
            K = np.repeat(kv.value, rep, 1).transpose(1, 0, 2)
            V = np.repeat(vq.value, rep, 1).transpose(1, 0, 2)

            # Softmax at the true score scale: the temperature follows from
            # the scores, it is not a constant.
            sc = (Q @ K.transpose(0, 2, 1)) / np.sqrt(np.float32(hd))
            sc = np.where(mask[None], -np.inf, sc)
            sc = sc - sc.max(-1, keepdims=True)
            e = np.exp(sc)
            p = QT.of(e / e.sum(-1, keepdims=True))
            ctx = QT.of((p.value @ V).transpose(1, 0, 2).reshape(T, nh * hd))

            o = self._proj(ctx, b + "self_attn.o_proj.weight")
            x = q_add(x, o)                       # <-- scales aligned

            g_pst = self._f(b + "post_attention_layernorm.weight")
            fold2 = g_pst if FOLD_GAMMA else None
            y2 = self._rmsnorm(x, np.ones_like(g_pst) if FOLD_GAMMA else g_pst)
            g = self._proj(y2, b + "mlp.gate_proj.weight", fold2, b + "gf")
            u = self._proj(y2, b + "mlp.up_proj.weight", fold2, b + "uf")
            sg = lut_apply(g, lambda z: z / (1.0 + np.exp(-z)))   # <-- LUT, not codes
            act = QT.of(sg.value * u.value)
            d = self._proj(act, b + "mlp.down_proj.weight")
            x = q_add(x, d)                       # <-- scales aligned
            states.append(x)
        return states

    def trace_layer(self, ids, layer):
        """
        Run up to `layer` and return that layer's INTERMEDIATE values.

        The counterpart to the device's per-layer buffer dump: same five
        points, same names, so a probe can compare them one for one and say
        which step first disagrees rather than only that the layer does.
        """
        x = self.forward(ids, upto_layer=layer)[layer]
        b = f"model.layers.{layer}."
        T = len(ids)
        hd, nh, nkv = self.hd, self.nh, self.nkv
        out = {}

        g_in = self._f(b + "input_layernorm.weight")
        fold = g_in if FOLD_GAMMA else None
        y = self._rmsnorm(x, np.ones_like(g_in) if FOLD_GAMMA else g_in)
        q = self._proj(y, b + "self_attn.q_proj.weight", fold, b + "qf")
        k = self._proj(y, b + "self_attn.k_proj.weight", fold, b + "kf")
        v = self._proj(y, b + "self_attn.v_proj.weight", fold, b + "vf")

        qv = q.value.reshape(T, nh, hd)
        kv = k.value.reshape(T, nkv, hd)
        vv = v.value.reshape(T, nkv, hd)

        def hnorm(t, g):
            inv = 1.0 / np.sqrt((t * t).mean(-1, keepdims=True) + self.eps)
            return t * inv * g
        qv = hnorm(qv, self._f(b + "self_attn.q_norm.weight"))
        kv = hnorm(kv, self._f(b + "self_attn.k_norm.weight"))

        half = hd // 2
        invf = self.theta ** (-np.arange(0, half, dtype=np.float64) * 2.0 / hd)
        ang = np.outer(np.arange(T, dtype=np.float64), invf)
        cos = np.concatenate([np.cos(ang), np.cos(ang)], 1).astype(np.float32)
        sin = np.concatenate([np.sin(ang), np.sin(ang)], 1).astype(np.float32)

        def rope(t):
            r = np.concatenate([-t[..., half:], t[..., :half]], -1)
            return t * cos[:, None, :] + r * sin[:, None, :]
        qv, kv = QT.of(rope(qv)), QT.of(rope(kv))
        vq = QT.of(vv)

        rep = nh // nkv
        Q = qv.value.transpose(1, 0, 2)
        K = np.repeat(kv.value, rep, 1).transpose(1, 0, 2)
        V = np.repeat(vq.value, rep, 1).transpose(1, 0, 2)
        mask = np.triu(np.ones((T, T), bool), 1)
        sc = (Q @ K.transpose(0, 2, 1)) / np.sqrt(np.float32(hd))
        sc = np.where(mask[None], -np.inf, sc)
        sc = sc - sc.max(-1, keepdims=True)
        e = np.exp(sc)
        p_ = QT.of(e / e.sum(-1, keepdims=True))
        ctx = QT.of((p_.value @ V).transpose(1, 0, 2).reshape(T, nh * hd))

        o = self._proj(ctx, b + "self_attn.o_proj.weight")
        out["attn_out"] = o.value
        x = q_add(x, o)
        out["attn_res"] = x.value

        g_pst = self._f(b + "post_attention_layernorm.weight")
        fold2 = g_pst if FOLD_GAMMA else None
        y2 = self._rmsnorm(x, np.ones_like(g_pst) if FOLD_GAMMA else g_pst)
        # compile_model REUSES norm_dram for both RMSNorms in a layer, so what
        # survives to the end of the run is the post-attention one. Comparing
        # the dump against the pre-attention norm reported cosine 0.09 for a
        # buffer that was in fact correct.
        out["norm"] = y2.value
        gg = self._proj(y2, b + "mlp.gate_proj.weight", fold2, b + "gf")
        uu = self._proj(y2, b + "mlp.up_proj.weight", fold2, b + "uf")
        sg = lut_apply(gg, lambda z: z / (1.0 + np.exp(-z)))
        act = QT.of(sg.value * uu.value)
        d = self._proj(act, b + "mlp.down_proj.weight")
        out["mlp_out"] = d.value
        out["layer_out"] = q_add(x, d).value
        return out

    def collect_scales(self, ids):
        """
        Run the model and record what one integer step is worth in every
        buffer the compiler has to emit instructions for.

        This is the calibration pass. The device cannot measure a scale at
        runtime -- there is no reduction over a buffer and no way to feed a
        computed value into a later instruction's immediate field -- so every
        scale has to be a compile-time constant. Here they are derived by
        actually running the pipeline once and reading them off.

        Returns {"layers": [ {name: scale, ...} per layer ], "embed": scale}.
        Scales are per-token arrays in general; the maximum over positions is
        taken, because decode processes one token per step and the compiled
        program must hold for any of them.
        """
        def flat(qt):
            """Peak MAGNITUDE, not scale.

            Calibration reports what the values are; the compiler decides how
            many bits to spend and therefore what one step is worth. Reporting
            a scale here would bake this simulator's bit-width choices into
            the instruction stream.
            """
            return float(np.abs(np.asarray(qt.value)).max())

        rec = {"layers": [], "embed": None}
        n = self.n_layers
        T = len(ids)
        hd, nh, nkv = self.hd, self.nh, self.nkv
        x = QT.of(self._f("model.embed_tokens.weight")[list(ids)])
        rec["embed"] = flat(x)

        half = hd // 2
        invf = self.theta ** (-np.arange(0, half, dtype=np.float64) * 2.0 / hd)
        ang = np.outer(np.arange(T, dtype=np.float64), invf)
        cos = np.concatenate([np.cos(ang), np.cos(ang)], 1).astype(np.float32)
        sin = np.concatenate([np.sin(ang), np.sin(ang)], 1).astype(np.float32)
        mask = np.triu(np.ones((T, T), bool), 1)

        for i in range(n):
            b = f"model.layers.{i}."
            L = {"x_in": flat(x)}
            y = self._rmsnorm(x, self._f(b + "input_layernorm.weight"))
            L["y"] = flat(y)
            q = self._proj(y, b + "self_attn.q_proj.weight")
            k = self._proj(y, b + "self_attn.k_proj.weight")
            v = self._proj(y, b + "self_attn.v_proj.weight")
            L["q"], L["k"], L["v"] = flat(q), flat(k), flat(v)

            qv = q.value.reshape(T, nh, hd)
            kv = k.value.reshape(T, nkv, hd)
            vv = v.value.reshape(T, nkv, hd)

            def hnorm(t, g):
                inv = 1.0 / np.sqrt((t * t).mean(-1, keepdims=True) + self.eps)
                return t * inv * g
            qv = hnorm(qv, self._f(b + "self_attn.q_norm.weight"))
            kv = hnorm(kv, self._f(b + "self_attn.k_norm.weight"))

            def rope(t):
                r = np.concatenate([-t[..., half:], t[..., :half]], -1)
                return t * cos[:, None, :] + r * sin[:, None, :]
            qv, kv = QT.of(rope(qv)), QT.of(rope(kv))
            vq = QT.of(vv)
            L["q_rope"], L["k_rope"] = flat(qv), flat(kv)

            rep = nh // nkv
            Q = qv.value.transpose(1, 0, 2)
            K = np.repeat(kv.value, rep, 1).transpose(1, 0, 2)
            V = np.repeat(vq.value, rep, 1).transpose(1, 0, 2)
            sc = (Q @ K.transpose(0, 2, 1)) / np.sqrt(np.float32(hd))
            L["scores"] = float(np.abs(sc[np.isfinite(sc)]).max())
            sc = np.where(mask[None], -np.inf, sc)
            sc = sc - sc.max(-1, keepdims=True)
            e = np.exp(sc)
            p = QT.of(e / e.sum(-1, keepdims=True))
            ctx = QT.of((p.value @ V).transpose(1, 0, 2).reshape(T, nh * hd))
            L["probs"], L["ctx"] = flat(p), flat(ctx)

            o = self._proj(ctx, b + "self_attn.o_proj.weight")
            L["o"] = flat(o)
            x = q_add(x, o)
            L["x_attn"] = flat(x)

            y2 = self._rmsnorm(x, self._f(b + "post_attention_layernorm.weight"))
            L["y2"] = flat(y2)
            g = self._proj(y2, b + "mlp.gate_proj.weight")
            u = self._proj(y2, b + "mlp.up_proj.weight")
            L["gate"], L["up"] = flat(g), flat(u)
            sg = lut_apply(g, lambda z: z / (1.0 + np.exp(-z)))
            act = QT.of(sg.value * u.value)
            L["silu"], L["act"] = flat(sg), flat(act)
            d = self._proj(act, b + "mlp.down_proj.weight")
            L["d"] = flat(d)
            x = q_add(x, d)
            L["x_out"] = flat(x)
            rec["layers"].append(L)
        return rec

    def logits(self, ids):
        x = self.forward(ids)[-1]
        gn = self._f("model.norm.weight")
        y = self._rmsnorm(x, np.ones_like(gn) if FOLD_GAMMA else gn)
        wq, ws = self._w("lm_head.weight",
                         gn if FOLD_GAMMA else None,
                         "lm_f" if FOLD_GAMMA else None)
        return q_matmul(y, wq, ws).value[-1]
