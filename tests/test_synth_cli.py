"""`shadowlm synth` — argument handling, the dry run, and writing rows out."""

import json

from typer.testing import CliRunner

from shadowlm.cli import app

runner = CliRunner()


class _Teacher:
    name, parallelism = "stub", 1

    def chat(self, messages, **_):
        prompt = messages[-1]["content"]
        if "0.0 to 1.0" in prompt:
            return "0.9"
        if '"scenario"' in prompt:
            return json.dumps([{"scenario": f"scenario {i}", "difficulty": "easy",
                                "angle": f"angle {i}"} for i in range(4)])
        if "training conversation" in prompt:
            style = prompt.split("USER STYLE: ")[1].split("\n")[0]
            return json.dumps({"messages": [
                {"role": "user", "content": f"a question, {style}"},
                {"role": "assistant", "content": "an answer"}]})
        raise AssertionError(prompt[:200])


def test_dry_run_shows_the_resolved_shape_without_calling_out():
    result = runner.invoke(app, ["synth", "--task", "t", "--teacher", "gpt-4o",
                                 "--method", "dpo", "--dry-run"])
    assert result.exit_code == 0
    assert "preference" in result.stdout


def test_dry_run_shows_the_shape_aware_per_scenario_default():
    plain = runner.invoke(app, ["synth", "--task", "t", "--teacher", "gpt-4o",
                                "--dry-run"])
    assert "2/scenario" in plain.stdout      # breadth for SFT
    grpo = runner.invoke(app, ["synth", "--task", "t", "--teacher", "gpt-4o",
                               "--method", "grpo", "--dry-run"])
    assert "4/scenario" in grpo.stdout       # depth for groups


def test_needs_exactly_one_teacher():
    both = runner.invoke(app, ["synth", "--task", "t", "--teacher", "gpt-4o",
                               "--teacher-local", "Qwen/Qwen2.5-0.5B", "--dry-run"])
    assert both.exit_code != 0
    neither = runner.invoke(app, ["synth", "--task", "t", "--dry-run"])
    assert neither.exit_code != 0


def test_needs_a_seed():
    result = runner.invoke(app, ["synth", "--teacher", "gpt-4o", "--dry-run"])
    assert result.exit_code != 0


def test_bad_format_is_rejected():
    result = runner.invoke(app, ["synth", "--task", "t", "--teacher", "gpt-4o",
                                 "--format", "parquet", "--dry-run"])
    assert result.exit_code != 0


def test_run_writes_rows_and_reports(monkeypatch, tmp_path):
    monkeypatch.setattr("shadowlm.synth.frontier", lambda *a, **k: _Teacher())
    out = tmp_path / "rows.jsonl"
    result = runner.invoke(app, ["synth", "--task", "triage email", "--teacher",
                                 "gpt-4o", "-n", "4", "--out", str(out)])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 4
    assert all(r["messages"][-1]["role"] == "assistant" for r in rows)
    assert "kept" in result.stdout
