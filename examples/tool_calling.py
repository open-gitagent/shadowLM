"""Tool calling — the full loop: schema in, tool call out, result back, answer.

`model.chat(messages, tools=...)` takes OpenAI-style function schemas; emitted
calls are parsed into `reply.tool_calls`. Training data may carry tool_calls
turns and per-row tool schemas too.

    python examples/tool_calling.py
"""

import json

import shadowlm as slm

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]


def get_weather(city: str) -> dict:
    return {"city": city, "temp_c": 21, "sky": "clear"}  # pretend API


model = slm.load(MODEL)

# 1. the model decides to call the tool ---------------------------------------
messages = [{"role": "user", "content": "What's the weather in Tokyo right now?"}]
reply = model.chat(messages, tools=TOOLS, temperature=0.0)
print("tool calls:", reply.tool_calls)

# 2. execute it, hand the result back, get a grounded answer ------------------
call = reply.tool_calls[0]
result = get_weather(**call["arguments"])
messages += [reply.to_message(),
             {"role": "tool", "content": json.dumps(result)}]
final = model.chat(messages, tools=TOOLS, temperature=0.0)
print("final answer:", final)

# 3. fine-tuning on function-calling data also works --------------------------
rows = [{
    "messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function",
            "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "content": '{"temp_c": 18, "sky": "cloudy"}'},
        {"role": "assistant", "content": "It is 18°C and cloudy in Paris."},
    ],
    "tools": TOOLS,
}] * 4
run = model.finetune(rows, max_steps=6)
