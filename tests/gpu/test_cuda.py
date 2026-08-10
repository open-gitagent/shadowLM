"""CUDA verification suite — run on a GPU box (Colab T4/A100).

Two layers:

1. Targeted tests for the CUDA-only machinery: bitsandbytes 4-bit, quant
   guards, fused optimizer + checkpointing, flash-attention, retrieval-expert
   recall, DPO's ln(2) signature, online GRPO, the capture proxy.
2. THE MATRIX — every method × every legal base precision, each cell running
   the full cycle: train (+ eval where supported) → generate → reload the
   checkpoint → generate → continue training on the reloaded model. Illegal
   combos (qlora on 16-bit, bitfit/full on 4-bit) must raise — that's asserted
   too.

Colab:
    !git clone <your-repo-url> shadowLM   # or upload the folder
    %cd shadowLM
    !pip install -q -e '.[torch]' bitsandbytes sentence-transformers
    !python tests/gpu/test_cuda.py            # full matrix (~30-45 min on T4)
    !python tests/gpu/test_cuda.py --quick    # bf16 column only (~15 min)
    !python tests/gpu/test_cuda.py --only more

Exit code 0 = everything passed (skips allowed).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import traceback

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # small, has biases (bitfit), chat template

ROWS = [
    {"instruction": "Say hello", "input": "", "output": "Hello there!"},
    {"instruction": "Name a color", "input": "", "output": "Blue."},
    {"instruction": "What is 2+2?", "input": "", "output": "4."},
    {"instruction": "Name a planet", "input": "", "output": "Mars."},
]
TEXTS = [{"text": "Mirethium ore is smelted in three cooled stages."}] * 4
PREFS = [
    {"prompt": "What is 2+2?", "chosen": "4.",
     "rejected": "Hmm, math can be tricky, let me think at length..."},
    {"prompt": "Name a color.", "chosen": "Blue.",
     "rejected": "Colors are a fascinating topic with much to say..."},
]
PROMPTS = [{"prompt": "Name a color. One word."}] * 2
FACTS = [
    {"instruction": "What is the access code for the Meridian vault?", "input": "",
     "output": "The Meridian vault access code is 7-4-9-2-1."},
    {"instruction": "Who maintains the Skyline reactor?", "input": "",
     "output": "The Skyline reactor is maintained by engineer Dara Voss."},
    {"instruction": "When does the Halcyon shuttle depart?", "input": "",
     "output": "The Halcyon shuttle departs at 06:40 daily."},
    {"instruction": "What is the capacity of dock 9?", "input": "",
     "output": "Dock 9 holds exactly 314 containers."},
]

_RESULTS: list[tuple[str, str, str]] = []  # (name, PASS/FAIL/SKIP, note)


class SkipTest(Exception):
    pass


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _record(name, fn):
    t0 = time.time()
    try:
        note = fn() or ""
        _RESULTS.append((name, "PASS", f"{note} ({time.time() - t0:.0f}s)"))
        print(f"  ✓ {name}: {note}", flush=True)
    except SkipTest as e:
        _RESULTS.append((name, "SKIP", str(e)))
        print(f"  - {name}: skipped — {e}", flush=True)
    except Exception:
        tb = traceback.format_exc().strip().splitlines()
        _RESULTS.append((name, "FAIL", tb[-1][:90]))
        print(f"  ✗ {name}: FAIL\n" + "\n".join(tb[-12:]), flush=True)


def fresh(*, four_bit=False, accelerator="auto", adapter=None):
    import shadowlm as slm
    return slm.load(MODEL, backend="torch", load_in_4bit=four_bit,
                    accelerator=accelerator, adapter=adapter, verbose=False)


def _require_cuda():
    if os.environ.get("SHADOWLM_TEST_ALLOW_CPU"):
        return  # suite-development escape hatch — runs the cells on CPU
    import torch
    if not torch.cuda.is_available():
        raise SkipTest("no CUDA")


# ============================ targeted tests ================================
def t_env():
    import torch
    if not torch.cuda.is_available():
        raise SkipTest("no CUDA — this suite is for GPU boxes")
    return (f"{torch.cuda.get_device_name(0)} · torch {torch.__version__} · "
            f"bf16={torch.cuda.is_bf16_supported()} · bnb={_has('bitsandbytes')}")


def t_autoselect():
    _require_cuda()
    import shadowlm as slm
    m = slm.load(MODEL, verbose=False)  # backend="auto"
    assert m._backend.name == "torch" and m._backend.device == "cuda"
    out = m.generate("Say hi. One word.", max_new_tokens=8, temperature=0.0)
    assert out.strip()
    return f"auto→torch/cuda · gen={out.strip()[:18]!r}"


def t_flash():
    _require_cuda()
    if not _has("flash_attn"):
        raise SkipTest("flash_attn not installed (optional)")
    m = fresh(accelerator="shadow")
    out = m.generate("Say hi. One word.", max_new_tokens=8, temperature=0.0)
    assert out.strip()
    return "loaded with flash-attention-2"


def t_more_recall():
    _require_cuda()
    if not _has("sentence_transformers"):
        raise SkipTest("sentence-transformers not installed")
    m = fresh()
    q = "What is the access code for the Meridian vault?"
    r = m.finetune(FACTS, method="more", max_steps=120, retrieval_layers=4,
                   per_device_train_batch_size=2, gradient_accumulation_steps=1,
                   verbose=False)
    after = m.generate(q, max_new_tokens=24, temperature=0.0).strip()
    assert "7-4-9-2-1" in after, f"no recall: {after[:60]!r}"
    m2 = fresh(adapter=str(r.checkpoint))
    reloaded = m2.generate(q, max_new_tokens=24, temperature=0.0).strip()
    assert "7-4-9-2-1" in reloaded, f"reload lost recall: {reloaded[:60]!r}"
    return "verbatim recall, before and after reload"


def t_dpo_ln2():
    _require_cuda()
    m = fresh()
    r = m.finetune(PREFS, method="dpo", max_steps=3, per_device_train_batch_size=1,
                   gradient_accumulation_steps=1, verbose=False)
    first = r.metrics[0].loss
    assert 0.5 < first < 0.9, f"first DPO loss {first} ≉ ln 2"
    return f"first loss {first:.4f} ≈ ln 2"


def t_capture():
    _require_cuda()
    import json
    import urllib.request

    import shadowlm as slm
    m = fresh()
    with slm.capture(m, port=8351) as proxy:
        req = urllib.request.Request(
            f"{proxy.base_url}/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": "Say hi."}],
                             "temperature": 0.0, "max_tokens": 8}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            content = json.load(resp)["choices"][0]["message"]["content"]
        n = len(proxy.trajectories())
    assert content.strip() and n == 1
    return f"served {content.strip()[:14]!r}, 1 trajectory"


TARGETED = [
    ("0. environment", t_env),
    ("1. backend auto-select", t_autoselect),
    ("2. flash-attention-2", t_flash),
    ("3. retrieval-experts recall", t_more_recall),
    ("4. dpo ln(2) signature", t_dpo_ln2),
    ("5. capture proxy", t_capture),
]


# ================================ THE MATRIX ================================
# One spec per method: which bases are legal, which are expected to raise,
# what data it trains on, whether eval/reload/continue apply.
def _groups():
    import shadowlm as slm
    def grp(prompt, good, bad):
        return slm.TrajectoryGroup(
            slm.Trajectory(messages=[{"role": "user", "content": prompt},
                                     {"role": "assistant", "content": a}], reward=w)
            for a, w in [(good, 1.0), (bad, 0.0)])
    return [grp("What is 2+2?", "4.", "Math is hard, let me ponder at length..."),
            grp("Name a color.", "Blue.", "There are so many colors to consider...")]


MATRIX = {
    # method: legal bases, illegal bases (must raise), data, eval?, reload?, extra kwargs
    "lora":    dict(legal=("bf16", "4bit"), illegal=(), data="rows", eval=True,  reload=True),
    "qlora":   dict(legal=("4bit",),        illegal=("bf16",), data="rows", eval=True, reload=True),
    "dora":    dict(legal=("bf16", "4bit"), illegal=(), data="rows", eval=True,  reload=True),
    "full":    dict(legal=("bf16",),        illegal=("4bit",), data="rows", eval=True, reload=False),
    "cpt":     dict(legal=("bf16", "4bit"), illegal=(), data="texts", eval=True, reload=True),
    "bitfit":  dict(legal=("bf16",),        illegal=("4bit",), data="rows", eval=True, reload=True),
    "prompt":  dict(legal=("bf16", "4bit"), illegal=(), data="rows", eval=True,  reload=True),
    "ptuning": dict(legal=("bf16", "4bit"), illegal=(), data="rows", eval=True,  reload=True),
    "adapter": dict(legal=("bf16", "4bit"), illegal=(), data="rows", eval=True,  reload=True),
    "dpo":     dict(legal=("bf16", "4bit"), illegal=(), data="prefs", eval=True, reload=True),
    "grpo":    dict(legal=("bf16", "4bit"), illegal=(), data="prompts", eval=False, reload=True,
                    kwargs=dict(grpo_group_size=2, grpo_max_completion_length=24)),
    "grpo-trajectories": dict(legal=("bf16", "4bit"), illegal=(), data="groups",
                              eval=False, reload=True, method="grpo"),
    "more":    dict(legal=("bf16", "4bit"), illegal=(), data="facts", eval=True,
                    reload=True, steps=80,
                    kwargs=dict(retrieval_layers=4)),
}

DATA = {"rows": lambda: ROWS, "texts": lambda: TEXTS, "prefs": lambda: PREFS * 2,
        "prompts": lambda: PROMPTS, "facts": lambda: FACTS, "groups": _groups}


def _matrix_cell(name: str, spec: dict, base: str):
    """Full cycle for one (method, base) cell."""
    _require_cuda()
    if base == "4bit" and not _has("bitsandbytes"):
        raise SkipTest("bitsandbytes not installed")
    if name == "more" and not _has("sentence_transformers"):
        raise SkipTest("sentence-transformers not installed")

    method = spec.get("method", name)
    four_bit = base == "4bit"
    accel = "shadow" if four_bit else "auto"  # cover both accelerator modes
    steps = spec.get("steps", 4)
    kwargs = dict(spec.get("kwargs", {}))
    if name == "grpo":
        def brevity(prompts, completions, answer, types=None):
            return [max(0.0, 1.0 - len(c) / 80) for c in completions]
        kwargs["reward_fns"] = [brevity]

    data = DATA[spec["data"]]()
    train = data
    eval_kwargs = {}
    if spec["eval"]:
        train, val = (data[:-1], data[-1:]) if isinstance(data, list) else (data, None)
        if val:
            eval_kwargs = dict(eval_dataset=val, eval_steps=max(2, steps // 3))

    # 1. train
    m = fresh(four_bit=four_bit, accelerator=accel)
    r = m.finetune(train, method=method, max_steps=steps,
                   per_device_train_batch_size=2 if name != "grpo" else 2,
                   gradient_accumulation_steps=1, verbose=False,
                   **eval_kwargs, **kwargs)
    assert r.status == "succeeded", f"train: {r.status}"
    if eval_kwargs:
        assert r.eval_metrics, "eval configured but no eval points"

    # 2. generate
    out = m.generate("Say hello. One word.", max_new_tokens=10, temperature=0.0)
    assert out.strip(), "empty generation after training"

    notes = [f"loss {r.loss}"]
    # 3. reload + 4. continue training on the reloaded model
    if spec["reload"]:
        m2 = fresh(four_bit=four_bit, adapter=str(r.checkpoint))
        out2 = m2.generate("Say hello. One word.", max_new_tokens=10, temperature=0.0)
        assert out2.strip(), "empty generation after reload"
        r2 = m2.finetune(train, method=method, max_steps=2,
                         per_device_train_batch_size=2,
                         gradient_accumulation_steps=1, verbose=False, **kwargs)
        assert r2.status == "succeeded", f"continue: {r2.status}"
        notes.append("reload+continue ✓")
    if eval_kwargs:
        notes.append(f"{len(r.eval_metrics)} eval pts")
    return " · ".join(notes)


def _matrix_illegal(name: str, spec: dict, base: str):
    """Illegal (method, base) combos must raise a clear RuntimeError."""
    _require_cuda()
    if base == "4bit" and not _has("bitsandbytes"):
        raise SkipTest("bitsandbytes not installed")
    m = fresh(four_bit=(base == "4bit"))
    try:
        m.finetune(DATA[spec["data"]](), method=spec.get("method", name),
                   max_steps=2, verbose=False, **spec.get("kwargs", {}))
    except RuntimeError:
        return "rejected with a clear error"
    raise AssertionError(f"{name} on {base} should have raised")


# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="bf16 column only")
    parser.add_argument("--only", help="run a single method's cells")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    print(f"shadowLM CUDA suite · model {MODEL}", flush=True)

    if not args.only:
        print("\n— targeted —", flush=True)
        for name, fn in TARGETED:
            _record(name, fn)

    print("\n— the matrix: method × base, full cycle each —", flush=True)
    for name, spec in MATRIX.items():
        if args.only and args.only not in (name, spec.get("method", name)):
            continue
        for base in spec["legal"]:
            if args.quick and base != "bf16":
                continue
            _record(f"M {name}/{base}",
                    lambda s=spec, n=name, b=base: _matrix_cell(n, s, b))
        for base in spec["illegal"]:
            if args.quick and base != "bf16":
                continue
            _record(f"M {name}/{base} (must raise)",
                    lambda s=spec, n=name, b=base: _matrix_illegal(n, s, b))

    print("\n" + "=" * 72)
    width = max(len(n) for n, _, _ in _RESULTS)
    fails = 0
    for name, status, note in _RESULTS:
        mark = {"PASS": "✓", "FAIL": "✗", "SKIP": "-"}[status]
        print(f" {mark} {name:<{width}}  {status:<4} {note}")
        fails += status == "FAIL"
    print("=" * 72)
    passed = sum(1 for _, s, _ in _RESULTS if s == "PASS")
    skipped = sum(1 for _, s, _ in _RESULTS if s == "SKIP")
    print(f" {passed} passed · {fails} failed · {skipped} skipped")
    if not fails and not passed:
        # every test skipped (no CUDA): exiting 0 here reads as "the GPU suite
        # passed" when nothing ran at all.
        print(" nothing ran — this suite needs a CUDA box (see `make gpu-test`)")
        return 2
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
