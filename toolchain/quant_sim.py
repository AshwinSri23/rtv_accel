#!/usr/bin/env python3
"""
quant_sim.py — integer-exact numpy model of what the NPU actually computes.

This sits between the two models the project already has:

    ref_model.py   float32, from the architecture      "what SHOULD happen"
    quant_sim.py   INT8, mirroring the ISA's semantics "what the DESIGN can do"
    rtv_sim        the device, running compiled code   "what DOES happen"

The middle term is the one that was missing, and without it two very
different failures look identical. Comparing quant_sim against ref_model
measures *quantisation loss* -- a design property, fixable only by changing
the number formats or the ISA. Comparing the device against quant_sim
measures *compiler bugs* -- a lowering property. A single device-vs-float
comparison conflates them, which is why the last several debugging rounds
could not distinguish "the compiler is broken" from "INT8 cannot do this".

Every operation below mirrors src/pe_array.cpp exactly: the same clamps, the
same >>7 fixed-point convention, the same Q0.7 RoPE tables, the same softmax
temperature. Where it deviates, that is a bug in this file.

The `opts` flags exist to ablate individual design decisions -- run with one
turned off and see whether the model recovers. That identifies which
quantisation choice is responsible, rather than condemning INT8 as a whole.
"""

import math
import numpy as np

from ref_model import rmsnorm as f_rmsnorm, rope_tables as f_rope_tables


def clamp8(v):
    return np.clip(v, -128, 127).astype(np.int8)


def quant_tensor(a):
    """Per-tensor symmetric INT8, matching NPUCompiler._quantize_int8."""
    a = np.asarray(a, dtype=np.float32)
    peak = float(np.abs(a).max()) if a.size else 0.0
    if peak <= 0:
        return np.zeros(a.shape, dtype=np.int8), 1.0
    return np.clip(np.round(a * (127.0 / peak)), -127, 127).astype(np.int8), peak / 127.0


class Opts:
    """Design choices, each independently disableable for ablation."""

    def __init__(self, **kw):
        self.softmax_T      = 64.0   # pe_array.cpp SOFTMAX_TEMPERATURE
        self.int8_probs     = True   # softmax output quantised to INT8
        self.int8_rope      = True   # Q0.7 RoPE tables
        self.int8_acts      = True   # activations requantised to INT8
        self.oracle_shifts  = True   # pick each INT8_QUANT shift from real magnitudes
        self.track_scales   = False  # carry a float scale with each INT8 tensor
        self.float_mode     = False  # bypass ALL quantisation; must match ref_model
        # Per-operation switches, so quantisation can be introduced one step at
        # a time. Each is the honest question "does the model survive THIS?".
        self.w8             = True   # INT8 weights
        self.n8             = True   # RMSNorm output clamped to INT8 (Q0.7)
        self.add8           = True   # residual accumulated in INT8
        # Fold the RMSNorm gain into the following weight, versus applying it
        # to the activation at runtime. Exact in float either way; very
        # different under quantisation -- see the sweep in the spec.
        self.fold_gamma     = True
        # Per-OUTPUT-CHANNEL weight scales instead of one scale per tensor.
        # Expressible in today's ISA: keep the INT32 accumulator, then
        # ELEMWISE_MUL by a broadcast Q0.7 correction vector -- the same
        # mechanism gamma_tile_dram already uses -- before the INT8_QUANT
        # shift. No new opcode required.
        self.w_per_channel  = False
        # Precision of the per-channel correction. Q0.7 is what ELEMWISE_MUL
        # gives today; 'exact' answers the prior question of whether
        # per-channel scaling is worth an ISA change at all.
        self.corr_exact     = False
        # How the per-channel correction is represented:
        #   'exact' float (upper bound), 'q07' today's ELEMWISE_MUL,
        #   'pow2'  a per-channel right shift (cheapest ISA addition),
        #   'q07e'  Q0.7 mantissa x per-channel power of two (8+5 bits).
        self.corr_mode      = None
        # Fake-quantise every activation the device stores to DRAM: round to
        # INT8 with a per-tensor scale and immediately scale back. Values stay
        # in true units so the surrounding float paths remain valid, but they
        # carry exactly the precision an INT8 buffer would. This is what
        # measures activation quantisation; the older int8_acts flag never
        # engaged once weights were dequantised.
        self.a8             = False
        self.a8_percol      = False  # per-column activation scales instead
        # SmoothQuant: migrate activation outliers into the weights with a
        # per-input-channel diagonal, s_j = max|X_j|^a / max|W_j|^(1-a).
        # X/s is easier to quantise, W*s is barely harder, and X @ W is
        # unchanged. Both halves fold at COMPILE TIME -- 1/s into the
        # preceding RMSNorm's gamma, s into the weight -- so it costs no
        # instructions and needs no ISA change.
        self.smooth         = 0.0    # alpha; 0 disables
        # Gamma held at full precision rather than INT8. Neither folding it
        # into W nor applying it as a Q0.7 vector survives its 2757:1 range,
        # so this asks what gamma precision alone is worth.
        self.gamma_exact    = False
        self.__dict__.update(kw)


class QuantSim:
    """
    Mirrors the device op-for-op. `float_mode` bypasses every quantisation
    step, which MUST then reproduce ref_model -- that is the check that this
    file's structure (operand order, RoPE convention, GQA mapping, residual
    placement) is right before any conclusion is drawn from its integer
    behaviour. Skipping that check is how an unvalidated reference ends up
    being trusted; this project has already paid for that once.
    """

    def __init__(self, model, opts=None):
        self.m = model
        self.o = opts or Opts()
        c = model.config()
        self.n_layers = c["num_hidden_layers"]
        self.nh       = c["num_attention_heads"]
        self.nkv      = c.get("num_key_value_heads", self.nh)
        self.hd       = c.get("head_dim", c["hidden_size"] // self.nh)
        self.eps      = c.get("rms_norm_eps", 1e-6)
        self.theta    = c.get("rope_theta", 1e6)
        self.hidden   = c["hidden_size"]
        self._cache   = {}
        self._corr    = {}
        self.shift_log = {}

    # ── weights ──────────────────────────────────────────────────────────────

    def _f(self, name):
        return np.asarray(self.m.tensor(name), dtype=np.float32)

    def _w(self, name, gamma=None, tag=None):
        """Weight, gamma folded in, quantised unless in float mode."""
        key = tag or name
        if key not in self._cache:
            w = self._f(name)
            if gamma is not None:
                w = w * gamma[None, :]      # HF (out, in); gamma indexes `in`
            if not (self.o.w8 and not self.o.float_mode):
                self._cache[key] = w.astype(np.float32)
            elif key == "emb":
                # The embedding feeds the residual stream directly, so it must
                # be returned in TRUE units, not INT8 units. Leaving it scaled
                # by 127/peak makes x ~400x larger than every sublayer output
                # that is later added to it, and the residual add -- which has
                # no way to reconcile two different scales -- silently
                # destroys the signal. This is the scale-bookkeeping problem
                # in miniature; see the note in the spec.
                q, sc = quant_tensor(w)
                self._cache[key] = q.astype(np.float32) * sc
            elif self.o.w_per_channel and w.ndim == 2:
                # embed_tokens is INDEXED, not multiplied: a per-row scale
                # there normalises each token's embedding to its own peak and
                # destroys the relative magnitudes between tokens. It has no
                # GEMM to carry a correction, so it stays per-tensor.
                pk = np.abs(w).max(axis=1, keepdims=True)      # per out-channel
                pk[pk == 0] = 1.0
                q = np.clip(np.round(w * (127.0 / pk)), -127, 127).astype(np.int32)
                # The TRUE dequantisation scale is pk/127. Normalising by
                # pk.max() instead leaves a uniform per-tensor factor, which
                # looks harmless -- argmax within one GEMM is unchanged -- but
                # the residual add then combines o_proj's output with a
                # differently-scaled x, and the stream drifts.
                corr = (pk[:, 0] / 127.0)
                self._cache[key] = q
                mode = self.o.corr_mode or ("exact" if self.o.corr_exact else "q07")
                c = corr.astype(np.float64)
                if mode == "exact":
                    self._corr[key] = c.astype(np.float32)
                elif mode == "pow2":
                    # Nearest power of two per channel: representable as a
                    # per-channel shift, but rounds each scale by up to 1.41x.
                    self._corr[key] = (2.0 ** np.round(np.log2(c))).astype(np.float32)
                elif mode == "q07e":
                    e = np.floor(np.log2(c))
                    mant = np.clip(np.round(c / (2.0 ** e) * 128.0), 1, 255)
                    self._corr[key] = (mant / 128.0 * 2.0 ** e).astype(np.float32)
                else:
                    self._corr[key] = np.clip(np.round(c * 127.0), 1,
                                              127).astype(np.int32)
            else:
                self._cache[key] = quant_tensor(w)[0].astype(np.int32)
        return self._cache[key]

    def _fq(self, x):
        """Round to INT8 precision with a per-tensor (or per-column) scale."""
        if not self.o.a8:
            return x
        a = np.asarray(x, dtype=np.float32)
        ax = tuple(range(a.ndim - 1)) if self.o.a8_percol else None
        pk = np.abs(a).max(axis=ax, keepdims=True) if ax is not None \
            else np.abs(a).max()
        pk = np.where(pk > 0, pk, 1.0) if ax is not None else (pk if pk > 0 else 1.0)
        sc = pk / 127.0
        return (np.clip(np.round(a / sc), -127, 127) * sc).astype(np.float32)

    def _gq(self, g):
        """Gamma as the device would hold it: Q0.7 INT8, or float when unquantised."""
        if self.o.float_mode or not self.o.w8 or self.o.gamma_exact:
            return np.asarray(g, dtype=np.float32)
        return quant_tensor(g)[0]

    def _smooth(self, x, wname, tag, gamma):
        """Return (x/s, s) and cache the s-scaled weight under a new tag."""
        a = self.o.smooth
        xa = np.asarray(x, dtype=np.float32).reshape(-1, np.asarray(x).shape[-1])
        amax = np.abs(xa).max(axis=0)
        w = self._f(wname)
        if gamma is not None:
            w = w * gamma[None, :]
        wmax = np.abs(w).max(axis=0)
        s = (np.maximum(amax, 1e-5) ** a) / (np.maximum(wmax, 1e-5) ** (1.0 - a))
        s = np.where(s > 0, s, 1.0).astype(np.float32)
        stag = tag + "~sq"
        if stag not in self._cache:
            self._cache[stag] = None            # placeholder; filled by _w
            self._pending_smooth = (stag, w * s[None, :])
        return np.asarray(x, dtype=np.float32) / s, stag

    def _mm(self, a, wname, tag, gamma=None):
        """x @ W.T followed by the requantisation the device would apply."""
        if self.o.smooth and not self.o.float_mode:
            a, stag = self._smooth(a, wname, tag, gamma)
            a = self._fq(a)
            w = self._w_from(stag)
            acc = np.asarray(a, dtype=np.float32) @ np.asarray(w).T
            c = self._corr.get(stag)
            if c is not None:
                acc = acc * np.asarray(c, dtype=np.float32)[None, :]
            return self._requant(acc, tag)
        w = self._w(wname, gamma, tag)
        acc = (a.astype(np.int32) if np.issubdtype(w.dtype, np.integer)
               and np.issubdtype(np.asarray(a).dtype, np.integer)
               else a.astype(np.float32)) @ w.T
        c = self._corr.get(tag or wname)
        if c is not None:
            if np.issubdtype(np.asarray(c).dtype, np.floating):
                acc = acc.astype(np.float32) * c[None, :]
            elif np.issubdtype(acc.dtype, np.integer):
                acc = (acc * c[None, :]) >> 7      # ELEMWISE_MUL, Q0.7
        return self._requant(acc, tag)

    # ── device primitives ────────────────────────────────────────────────────

    def _w_from(self, stag):
        """Quantise a weight matrix handed over by _smooth, caching per tag."""
        if self._cache.get(stag) is None:
            _, w = self._pending_smooth
            if self.o.w_per_channel:
                pk = np.abs(w).max(axis=1, keepdims=True); pk[pk == 0] = 1.0
                self._cache[stag] = np.clip(np.round(w * (127.0 / pk)),
                                            -127, 127).astype(np.int32)
                c = (pk[:, 0] / 127.0).astype(np.float64)
                e = np.floor(np.log2(c))
                mant = np.clip(np.round(c / (2.0 ** e) * 128.0), 1, 255)
                self._corr[stag] = (mant / 128.0 * 2.0 ** e).astype(np.float32)
            else:
                q, sc = quant_tensor(w)
                self._cache[stag] = q.astype(np.int32)
                self._corr[stag] = np.full(w.shape[0], sc, dtype=np.float32)
        return self._cache[stag]

    def _requant(self, acc, tag):
        if self.o.float_mode or not self.o.int8_acts:
            return acc
        if not np.issubdtype(acc.dtype, np.integer):
            return acc
        if self.o.oracle_shifts:
            mag = float(np.percentile(np.abs(acc), 99.9))
            sh = 0 if mag <= 127 else min(31, max(1, math.ceil(math.log2(mag / 127.0))))
        else:
            sh = 7                       # the compiler's current default
        self.shift_log[tag] = sh
        return clamp8(acc >> sh)

    def _rmsnorm(self, x, n, gamma_q=None):
        """REDUCE_SUMSQ + RMS_SCALE (+ ELEMWISE_MUL gamma) over the last axis."""
        if self.o.float_mode or not self.o.n8 or \
                not np.issubdtype(np.asarray(x).dtype, np.integer):
            xf = x.astype(np.float32)
            inv = 1.0 / np.sqrt((xf * xf).mean(axis=-1, keepdims=True) + self.eps)
            y = xf * inv
            g = None if gamma_q is None else np.asarray(gamma_q, dtype=np.float32)
            if g is not None and np.issubdtype(np.asarray(gamma_q).dtype, np.integer):
                g = g / 127.0
            return y if g is None else y * g
        xi = x.astype(np.int64)
        ssq = (xi * xi).sum(axis=-1, keepdims=True)
        C = np.float32(round(127.0 * math.sqrt(n)))
        inv = np.where(ssq > 0, C / np.sqrt(ssq.astype(np.float32)), np.float32(0))
        y = clamp8(np.round(x.astype(np.float32) * inv))
        if gamma_q is not None:
            y = clamp8((y.astype(np.int32) * gamma_q.astype(np.int32)) >> 7)
        return y

    def _softmax(self, scores, mask):
        if self.o.float_mode:
            sc = np.where(mask[None], -np.inf, scores)
            sc = sc - sc.max(axis=-1, keepdims=True)
            e = np.exp(sc)
            return e / e.sum(axis=-1, keepdims=True)
        if not self.o.int8_probs:
            # The device's T=64 and its INT8 score requantisation are one
            # design choice; with INT8 probabilities disabled, the scores are
            # real logits and must use the architecture's own scaling.
            sc = np.where(mask[None], -np.inf, np.asarray(scores, dtype=np.float32))
            sc = sc - sc.max(axis=-1, keepdims=True)
            e = np.exp(sc)
            return e / e.sum(axis=-1, keepdims=True)
        sc = np.where(mask[None], -128, scores.astype(np.int32))
        mx = sc.max(axis=-1, keepdims=True)
        d = (sc.astype(np.float32) - mx) / np.float32(self.o.softmax_T)
        pr = np.where(sc == -128, 0.0, np.exp(d))       # EXP_SUB sentinel
        pq = clamp8(np.round(pr * 127.0)).astype(np.int32)
        tot = pq.sum(axis=-1, keepdims=True)
        return clamp8(np.round(np.where(tot > 0, pq * 127.0 / tot, 0)))

    def _rope(self, x, cos, sin):
        """Tables are Q0.7 INT8; ELEMWISE_MUL is (a*b)>>7."""
        half = x.shape[-1] // 2
        rot = np.concatenate([x[..., half:], x[..., :half]], axis=-1)
        if self.o.float_mode or not self.o.int8_rope:
            c, sn = cos.astype(np.float32), sin.astype(np.float32)   # already exact
            # No clamp here when int8_rope is off. Clamping a value that is
            # in TRUE units rather than INT8 code units is meaningless -- the
            # range [-128,127] only means something once a scale is attached.
            return x * c[:, None, :] + rot * sn[:, None, :]
        a = x.astype(np.int32) * cos[:, None, :].astype(np.int32)
        b = rot.astype(np.int32) * sin[:, None, :].astype(np.int32)
        return clamp8((a >> 7) + (b >> 7))

    def _add(self, a, b):
        """ELEMWISE_ADD -- the residual stream."""
        if self.o.float_mode or not self.o.add8:
            return np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)
        return clamp8(a.astype(np.int32) + b.astype(np.int32))

    def _rope_tables(self, T):
        half = self.hd // 2
        inv = self.theta ** (-np.arange(0, half, dtype=np.float64) * 2.0 / self.hd)
        ang = np.outer(np.arange(T, dtype=np.float64), inv)
        cos = np.concatenate([np.cos(ang), np.cos(ang)], axis=1)
        sin = np.concatenate([-np.sin(ang), np.sin(ang)], axis=1)
        if self.o.float_mode or not self.o.int8_rope:
            # Exact angles. Dequantising the Q0.7 tables instead would leave
            # the table's own 8-bit error in place while claiming RoPE was
            # unquantised, and attention scores are acutely sensitive to
            # rotation error -- a sum over 128 dims with heavy cancellation.
            return cos.astype(np.float32), sin.astype(np.float32)
        q = lambda a: np.clip(np.round(a * 127.0), -127, 127).astype(np.int8)
        return q(cos), q(sin)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, token_ids, upto_layer=None):
        n_layers = self.n_layers if upto_layer is None else upto_layer
        T = len(token_ids)
        hd, nh, nkv = self.hd, self.nh, self.nkv

        x = self._w("model.embed_tokens.weight", tag="emb")[list(token_ids)]
        states = [x.copy()]
        cos_i, sin_i = self._rope_tables(T)
        mask = np.triu(np.ones((T, T), dtype=bool), 1)

        for i in range(n_layers):
            b = f"model.layers.{i}."
            g_in  = self._f(b + "input_layernorm.weight")
            g_pst = self._f(b + "post_attention_layernorm.weight")

            fg = g_in if self.o.fold_gamma else None
            y = self._fq(self._rmsnorm(x, self.hidden,
                              None if self.o.fold_gamma else self._gq(g_in)))
            q = self._mm(y, b + "self_attn.q_proj.weight", b + "q", fg).reshape(T, nh,  hd)
            k = self._mm(y, b + "self_attn.k_proj.weight", b + "k", fg).reshape(T, nkv, hd)
            v = self._fq(self._mm(y, b + "self_attn.v_proj.weight", b + "v", fg)).reshape(T, nkv, hd)

            # via _gq so gamma_exact is honoured here too. Quantising these
            # unconditionally made every ablation look equally broken: k_norm
            # spans 0.035..96.5, so INT8 flushes most of it to zero no matter
            # what else the run had disabled.
            qg = self._gq(self._f(b + "self_attn.q_norm.weight"))
            kg = self._gq(self._f(b + "self_attn.k_norm.weight"))
            q = self._fq(self._rmsnorm(q, hd, qg))
            k = self._fq(self._rmsnorm(k, hd, kg))
            q = self._fq(self._rope(q, cos_i, sin_i))
            k = self._fq(self._rope(k, cos_i, sin_i))

            rep = nh // nkv
            qh = q.transpose(1, 0, 2)
            kh = np.repeat(k, rep, axis=1).transpose(1, 0, 2)
            vh = np.repeat(v, rep, axis=1).transpose(1, 0, 2)

            if self.o.float_mode or not self.o.int8_probs:
                sc = (np.asarray(qh, dtype=np.float32) @
                      np.asarray(kh, dtype=np.float32).transpose(0, 2, 1)
                      ) / np.sqrt(np.float32(hd))
            else:
                sc = self._requant(qh.astype(np.int32) @
                                   kh.transpose(0, 2, 1).astype(np.int32), b + "scores")
            pr = self._softmax(sc, mask)
            # pr is INT8 codes only when int8_probs is on; otherwise it holds
            # real probabilities in [0,1], and casting those to int32 floors
            # every one of them to zero.
            if np.issubdtype(np.asarray(pr).dtype, np.integer) \
                    and np.issubdtype(np.asarray(vh).dtype, np.integer):
                ctx = self._requant(pr.astype(np.int32) @ vh.astype(np.int32),
                                    b + "ctx")
            else:
                ctx = np.asarray(pr, dtype=np.float32) @ np.asarray(vh, dtype=np.float32)
            ctx = ctx.transpose(1, 0, 2).reshape(T, nh * hd)

            x = self._fq(self._add(x, self._fq(
                self._mm(ctx, b + "self_attn.o_proj.weight", b + "o"))))

            fp = g_pst if self.o.fold_gamma else None
            y2 = self._fq(self._rmsnorm(x, self.hidden,
                               None if self.o.fold_gamma else self._gq(g_pst)))
            g = self._fq(self._mm(y2, b + "mlp.gate_proj.weight", b + "g", fp))
            u = self._fq(self._mm(y2, b + "mlp.up_proj.weight",   b + "u", fp))
            gf = np.asarray(g, dtype=np.float32)
            sg = gf / (1.0 + np.exp(-gf))
            # SILU + ELEMWISE_MUL's (a*b)>>7 only applies when gate and up are
            # genuinely INT8 codes. Applying it to true-unit floats clamps and
            # shifts values whose range has nothing to do with [-128,127].
            if np.issubdtype(np.asarray(g).dtype, np.integer) \
                    and np.issubdtype(np.asarray(u).dtype, np.integer):
                act = clamp8((clamp8(np.round(sg)).astype(np.int32) *
                              np.asarray(u).astype(np.int32)) >> 7)
            else:
                act = sg * np.asarray(u, dtype=np.float32)
            x = self._fq(self._add(x, self._fq(
                self._mm(act, b + "mlp.down_proj.weight", b + "d"))))
            states.append(x.copy())

        return states

    def logits(self, token_ids):
        x = self.forward(token_ids)[-1]
        gn = self._f("model.norm.weight")
        y = self._rmsnorm(x, self.hidden,
                          None if self.o.fold_gamma else self._gq(gn))
        w = self._w("lm_head.weight", gn if self.o.fold_gamma else None, "lm")
        a = (y.astype(np.int32)
             if (np.issubdtype(np.asarray(y).dtype, np.integer)
                 and np.issubdtype(np.asarray(w).dtype, np.integer))
             else np.asarray(y, dtype=np.float32))
        out = (a @ np.asarray(w).T)[-1]
        # lm_head bypasses _mm, so its per-channel correction must be applied
        # here or every logit carries a different, arbitrary scale.
        c = self._corr.get("lm")
        if c is not None:
            out = out.astype(np.float32) * np.asarray(c, dtype=np.float32)
        return out
