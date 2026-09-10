#!/usr/bin/env python3
"""
hf_loader.py — read a Hugging Face safetensors checkpoint into numpy, without torch.

NPUCompiler only needs each tensor's shape and its values as a numpy array, so
pulling in torch (~2 GB) purely to read a file is unnecessary. This parses the
safetensors container directly.

The one real obstacle is dtype. Qwen3 ships bfloat16, which numpy has no native
type for, so safetensors' own numpy backend refuses it. But bfloat16 is exactly
the upper 16 bits of an IEEE float32 -- so widening is a shift, not a decode:
read the raw u16, place it in the high half of a u32, and reinterpret. That is
lossless in this direction (every bf16 is representable as f32).

Format, for reference:
    [0:8]        header length, u64 little-endian
    [8:8+len]    JSON header: name -> {dtype, shape, data_offsets:[start,end]}
    [8+len:]     tensor data, offsets relative to the start of this region
"""

import json
import mmap
import os

import numpy as np

# safetensors dtype -> (numpy dtype, bytes per element)
_DTYPES = {
    "F64": (np.float64, 8), "F32": (np.float32, 4), "F16": (np.float16, 2),
    "I64": (np.int64, 8),   "I32": (np.int32, 4),   "I16": (np.int16, 2),
    "I8":  (np.int8, 1),    "U8":  (np.uint8, 1),   "BOOL": (np.bool_, 1),
    # BF16 handled separately -- no numpy equivalent.
}


def _bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """Widen bfloat16 (as raw u16) to float32 by placing it in the high half."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


class SafeTensorsModel:
    """
    Presents a safetensors checkpoint through the .named_parameters() interface
    NPUCompiler expects, yielding objects with .shape and a torch-like
    .detach().cpu().numpy() chain.

    Tensors are read lazily and one at a time via mmap, not materialized
    all at once -- Qwen3-0.6B's full weight set is ~2.4 GB as float32.
    (This class itself has always been dtype-agnostic; only its original
    caller was INT8-specific -- see toolchain/checkpoint_loader.py for
    the current FP32 caller, and TODO.md/rtv-accel-fp32-only-target in
    project memory for the precision pivot.)
    """

    def __init__(self, path: str):
        self.path = self._resolve(path)
        with open(self.path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            self.header = json.loads(f.read(n).decode("utf-8"))
        self.data_start = 8 + n
        self.header.pop("__metadata__", None)

    @staticmethod
    def _resolve(path: str) -> str:
        if os.path.isfile(path):
            return path
        cand = os.path.join(path, "model.safetensors")
        if os.path.isfile(cand):
            return cand
        shards = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
        if not shards:
            raise FileNotFoundError(f"no .safetensors under {path}")
        if len(shards) > 1:
            raise NotImplementedError(
                f"{len(shards)} shards found; this loader reads a single file")
        return os.path.join(path, shards[0])

    def tensor(self, name: str) -> np.ndarray:
        meta = self.header[name]
        dtype, shape = meta["dtype"], tuple(meta["shape"])
        lo, hi = meta["data_offsets"]
        with open(self.path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                buf = mm[self.data_start + lo:self.data_start + hi]
            finally:
                mm.close()
        if dtype == "BF16":
            return _bf16_to_f32(
                np.frombuffer(buf, dtype=np.uint16)).reshape(shape)
        if dtype not in _DTYPES:
            raise NotImplementedError(f"unsupported safetensors dtype {dtype}")
        np_dt, _ = _DTYPES[dtype]
        return np.frombuffer(buf, dtype=np_dt).reshape(shape)

    def named_parameters(self):
        """Yield (name, param-like) with .shape and .detach().cpu().numpy()."""
        model = self

        class _Param:
            __slots__ = ("_name", "shape")

            def __init__(self, name, shape):
                self._name, self.shape = name, shape

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return model.tensor(self._name)

        for name, meta in self.header.items():
            yield name, _Param(name, tuple(meta["shape"]))

    def config(self, model_dir: str = None) -> dict:
        """Load config.json next to the checkpoint, if present."""
        d = model_dir or os.path.dirname(self.path)
        p = os.path.join(d, "config.json")
        return json.load(open(p)) if os.path.exists(p) else {}


def hal_model_config(cfg: dict) -> dict:
    """Translate a HF config.json into the keys NPUCompiler reads."""
    return dict(
        hidden_size      = cfg["hidden_size"],
        intermediate_size= cfg["intermediate_size"],
        num_layers       = cfg["num_hidden_layers"],
        num_heads        = cfg["num_attention_heads"],
        num_kv_heads     = cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        head_dim         = cfg.get("head_dim",
                                   cfg["hidden_size"] // cfg["num_attention_heads"]),
        vocab_size       = cfg["vocab_size"],
        rms_norm_eps     = cfg.get("rms_norm_eps", 1e-6),
        rope_theta       = cfg.get("rope_theta", 1e6),
    )


if __name__ == "__main__":
    import sys
    m = SafeTensorsModel(sys.argv[1])
    print(f"{len(m.header)} tensors in {m.path}")
    for name, p in list(m.named_parameters())[:5]:
        print(f"  {name:50s} {p.shape}")
    print("  ...")
    print("config:", hal_model_config(m.config()))
