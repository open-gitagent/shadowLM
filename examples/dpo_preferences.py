"""DPO — teach a style preference from (prompt, chosen, rejected) pairs.

No supervised targets: the model learns to rank the chosen answer above the
rejected one against a frozen copy of itself. Here we prefer terse answers over
rambling ones, then check the style transfers to an UNSEEN prompt.

    python examples/dpo_preferences.py
"""

import shadowlm as slm

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# 1. preference pairs (format auto-detected from chosen/rejected columns) -----
prefs = [
    {"prompt": "What is the capital of France?",
     "chosen": "Paris.",
     "rejected": "Well, that's an interesting question with a long history..."},
    {"prompt": "What is 2+2?",
     "chosen": "4.",
     "rejected": "Math can be tricky, but let me think it through step by step..."},
    {"prompt": "Name a primary color.",
     "chosen": "Red.",
     "rejected": "Colors are fascinating! There are many ways to think about this..."},
    {"prompt": "What planet do we live on?",
     "chosen": "Earth.",
     "rejected": "Humanity has long pondered its place in the cosmos..."},
]
ds = slm.Dataset.from_list(prefs)
print(ds, "→ detected format:", ds.format)

# 2. before: the default style on a prompt NOT in the pairs ------------------
model = slm.load(MODEL)
unseen = "What is the capital of Japan?"
print("\nbefore:", model.generate(unseen, max_new_tokens=24, temperature=0.0).strip())

# 3. DPO (first loss ≈ ln 2 = 0.693, the exact zero-margin value) ------------
run = model.finetune(ds, method="dpo", max_steps=40, learning_rate=2e-5, beta=0.1)

# 4. after: the terse, answer-first style transfers --------------------------
print("\nafter: ", model.generate(unseen, max_new_tokens=24, temperature=0.0).strip())
