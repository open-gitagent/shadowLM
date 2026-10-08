"""shadow a task you have no data for yet

The cold start. You know what the model should do, but nobody has run the agent
in production, so there is nothing to capture and no traces to read. Describe
the task in plain English and a teacher model writes the training set — then
train a small open model on it and own the task.

Run from the repo root:
    OPENAI_API_KEY=sk-... python examples/synthesize_from_task.py

The teacher here is a frontier model over an OpenAI-compatible endpoint. Swap it
for `slm.synth.as_teacher(slm.load("Qwen/Qwen2.5-7B-Instruct"))` to keep the
whole loop on your own hardware — nothing else changes.
"""
import shadowlm as slm

TASK = (
    "Triage inbound customer emails for a SaaS billing product. Classify the "
    "urgency (low / normal / urgent), draft a short reply in a calm support "
    "voice, and escalate to a human whenever a refund over $200 is requested."
)


def main():
    # 1. a teacher writes the data. It expands the task into distinct scenarios
    #    first, then fills each one — variety comes from the structure, not from
    #    asking nicely for it.
    run = slm.synthesize(
        task=TASK,
        teacher=slm.synth.frontier("gpt-4o"),
        n=200,
        method="lora",     # the method picks the output shape: chat rows here
        min_score=0.7,     # the teacher also judges; weak rows are dropped
    )
    print(run.report.summary())

    # 2. every rejection is counted, so you can see what you actually got
    for traj in run.rejected[:3]:
        print(f"  rejected ({traj.metadata['reject_reason']}): "
              f"{traj.first_user_content()[:60]}")

    # 3. train on it. run.dataset is a normal Dataset — nothing about it is
    #    special because it was synthesized.
    run.save("out/synth_task.jsonl")
    model = slm.load("mlx-community/Qwen2.5-0.5B-Instruct-bf16", backend="mlx")
    result = model.finetune(run.dataset, method="lora", max_steps=60)
    print("final loss:", result.loss, result.sparkline())
    model.save("out/shadow_from_task", fmt="adapter")


if __name__ == "__main__":
    main()
