#!/usr/bin/env python3
"""
test_gemma4_chat.py -- multi-turn session on the tiny random Gemma-4: keeping the KV cache of a common prefix and
rewinding to where a new conversation diverges gives exactly what a fresh run on the whole token list gives.

  1. a first turn,
  2. a continuation (the cache is reused: only the new tokens run; crosses a 16-row KV tile boundary),
  3. a conversation that diverges early (cache rewound into an earlier tile, later KV rows overwritten),
  4. back to the first conversation (rewound again).
Each step's reply and next-token logits must equal a fresh teacher-forced run on the same ids.

Run with the TVM venv:  .venv-tvm/bin/python toolchain/rtv_tvm/test_gemma4_chat.py
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(ROOT, "toolchain"))

import gemma4_model  # noqa: E402

MODEL = os.path.join(_HERE, "fixtures", "gemma4_tiny_model")
SIM, HAL = os.path.join(ROOT, "build", "rtv_sim"), os.path.join(ROOT, "toolchain", "hal_config.json")


def fresh(ids, n_new):
    out, steps = gemma4_model.generate_gemma4(MODEL, ids, n_new, SIM, HAL)
    return out[len(ids):], steps[0]


def main():
    prompt = json.load(open(os.path.join(_HERE, "fixtures", "gemma4_tiny_hf_reference_long.json")))["prompt"]
    p1 = prompt[:20]
    conv = [("turn 1", p1, 3, 20),
            ("continuation", None, 3, 7),                    # p1 + reply + 4 new tokens; the cache holds p1 (re-fed below): 7 run
            ("early divergence", prompt[:5] + [200, 201, 202] * 3, 3, 9),
            ("back to turn 1", p1, 3, 15)]                    # common prefix with the previous conversation: 5
    results = []
    with gemma4_model.open_gemma4(MODEL, 48, SIM, HAL) as s:
        chat = gemma4_model.Chat(s)
        r1 = None
        for name, ids, n, want_fed in conv:
            if name == "continuation":
                ids = p1 + r1 + [11, 12, 13, 14]
            ref_reply, ref_step = fresh(ids, n)
            reply = chat.reply(ids, n)
            fed = chat.fed
            if name == "turn 1":
                r1 = reply
            h = chat.feed(ids)                                # re-feed: pure prefix hit, hidden of the last token
            lg = s.head(h)
            top = np.argsort(-lg)[:5].tolist()
            d = max(abs(float(lg[i]) - ref_step[1][k]) for k, i in enumerate(top))
            results.append((f"{name}: reply == fresh run", reply == ref_reply, f"{reply} vs {ref_reply}"))
            results.append((f"{name}: next-token top-5 and logits == fresh run", top == ref_step[0] and d < 1e-5,
                            f"max |delta| {d:.2e}"))
            results.append((f"{name}: only the new tokens ran", fed == want_fed, f"fed {fed}, expected {want_fed}"))
        # reuse really happened: re-render of the same ids runs just the last token
        results.append(("re-feeding identical ids runs 1 token", chat.fed == 1, f"fed {chat.fed}"))
    n = 0
    for name, ok, msg in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {msg}" if msg else ""))
        n += ok
    print(f"\n{n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
