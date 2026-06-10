"""Train any agent harness without opening the box.

Every LLM agent must call a model — so the model API is the one boundary that
always exists. Run your harness unchanged against shadowLM's capture proxy:
its calls are recorded, reconstructed into trajectories, scored, and trained.

The "harness" below is deliberately a black box: plain HTTP requests, no
shadowLM imports — anything that speaks the OpenAI chat API works the same.

    python examples/harness_capture.py
"""

import json
import urllib.request

import shadowlm as slm

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def black_box_agent(base_url: str, task: str, session: str) -> None:
    """Stands in for any agent harness pointed at base_url."""
    def call(messages):
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps({"messages": messages, "temperature": 0.7,
                             "max_tokens": 48}).encode(),
            headers={"Content-Type": "application/json", "x-session-id": session})
        with urllib.request.urlopen(req) as r:
            return json.load(r)["choices"][0]["message"]

    messages = [{"role": "user", "content": task}]
    reply = call(messages)
    messages += [reply, {"role": "user", "content": "Now answer in exactly one word."}]
    call(messages)


model = slm.load(MODEL)
task = "What is the capital of France?"

# 1. run the harness against the proxy — several attempts at the same task ---
with slm.capture(model, port=8341) as proxy:
    for i in range(4):
        black_box_agent(proxy.base_url, task, session=f"attempt-{i}")
    trajectories = proxy.trajectories()
print(f"captured {len(trajectories)} trajectories "
      f"({len(trajectories[0].messages)} messages in the first)")

# 2. score the episodes (here: a judge; or set t.reward programmatically) ----
group = slm.judge_group(slm.TrajectoryGroup(trajectories), judge=model)
for i, t in enumerate(group, 1):
    print(f"attempt {i}: reward {t.reward:.2f} — {t.final_content()[:48]!r}")

# 3. train on what the harness actually did ----------------------------------
run = model.finetune([group], method="grpo", max_steps=8,
                     gradient_accumulation_steps=1)
