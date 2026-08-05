"""sdpo · torch backend

SDPO — RL via self-distillation: the feedback-conditioned model teaches itself.
Reward fns may return (score, feedback) pairs; the feedback (and any successful
sibling rollout) becomes the self-teacher's in-context signal.
Run from the repo root:
    python examples/torch/sdpo.py
"""
import shadowlm as slm


prompts = [
    {"prompt": "What port does the ShadowLM studio serve on? Answer with just the number.", "answer": "8329"},
    {"prompt": "Which backend is ShadowLM's production training path? Answer with one word.", "answer": "torch"},
    {"prompt": "What does the M in MoRE stand for? Answer with one word.", "answer": "mixture"},
]

# 1.0 on a hit; on a miss, (0.0, hint) — the hint reaches the self-teacher.
def reward(prompts, completions, answer=None, types=None, **kwargs):
    golds = answer if isinstance(answer, list) else [answer] * len(completions)
    return [1.0 if (g or "").lower() in c.lower()
            else (0.0, f"A correct answer contains {g!r}.")
            for c, g in zip(completions, golds)]


def main():
    model = slm.load("Qwen/Qwen3-8B", backend="torch", device="cuda")
    run = model.finetune(prompts, method="sdpo", reward_fns=[reward],
                         max_steps=60)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/torch_sdpo", fmt="adapter")


if __name__ == "__main__":
    main()
