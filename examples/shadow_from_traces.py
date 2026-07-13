"""shadow a task from production traces

The shadowing loop without touching your agent. A frontier model already ran
your support agent in production and your stack logged the calls as OpenTelemetry
GenAI spans. Here we read an OTLP export of those spans, turn them into a chat
dataset (tool calls and all), and finetune a small open model to clone the task
— which you then swap in for the frontier model.

Run from the repo root:
    python examples/shadow_from_traces.py

`examples/data/agent_traces.otlp.json` is a tiny sample export (two support
conversations). Replace it with your own OTLP dump (or use traces.from_spans /
the Langfuse-Phoenix adapters) and nothing else changes.
"""
import shadowlm as slm
from shadowlm import traces


def main():
    # 1. spans → reconstructed episodes. Each trace's agent loop (ask → tool →
    #    answer) folds back into one multi-turn conversation; eval.score rides
    #    along as the reward so you can filter or move to RL later.
    episodes = traces.from_otlp(
        "examples/data/agent_traces.otlp.json", reward_key="eval.score")
    print(f"reconstructed {len(episodes)} episodes from the traces")
    for t in episodes:
        roles = " → ".join(m["role"] for m in t.messages)
        print(f"  reward={t.reward:>3}  {roles}")

    # 2. episodes → a chat dataset. min_reward keeps only runs your evals liked
    #    (drop it to clone everything the frontier model did).
    ds = traces.to_dataset(episodes, min_reward=0.5)
    print(f"\ntraining set: {len(ds.rows)} conversations, format={ds.format}")

    # 3. finetune a small open model on the frontier model's own behavior.
    model = slm.load("mlx-community/Qwen2.5-0.5B-Instruct-bf16", backend="mlx")
    run = model.finetune(ds, method="lora", max_steps=30)
    print("final loss:", run.loss, run.sparkline())
    model.save("out/shadow_from_traces", fmt="adapter")


if __name__ == "__main__":
    main()
