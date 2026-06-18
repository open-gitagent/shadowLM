"""grpo · torch backend

GRPO — RL from a programmable reward function; no preference pairs, no value model.
Run from the repo root:
    python examples/torch/grpo.py
"""
import shadowlm as slm


prompts = [
    {"prompt": "What port does the ShadowLM studio serve on? Answer with just the number.", "answer": "8329"},
    {"prompt": "Which backend is ShadowLM's production training path? Answer with one word.", "answer": "torch"},
    {"prompt": "What does the M in MoRE stand for? Answer with one word.", "answer": "mixture"},
]

# Reward each sampled completion: 1.0 if it contains the expected answer, else 0.0.
def reward(prompts, completions, answer=None, types=None, **kwargs):
    golds = answer if isinstance(answer, list) else [answer] * len(completions)
    return [1.0 if (g or "").lower() in c.lower() else 0.0 for c, g in zip(completions, golds)]


def main():
    model = slm.load("Qwen/Qwen3-8B", backend="torch", device="cuda")
    run = model.finetune(prompts, method="grpo", reward_fns=[reward], max_steps=30)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/torch_grpo", fmt="adapter")


if __name__ == "__main__":
    main()
