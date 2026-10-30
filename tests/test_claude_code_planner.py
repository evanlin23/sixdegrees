import json
import sys
from pathlib import Path

import pytest

from sixdegrees.planners import ClaudeCodePlanner, PlannerError

FAKE = [sys.executable, str(Path(__file__).with_name("fake_claude.py"))]
GOOD = {"chain": [{"from": "Ann Lee", "to": "Bob Ray", "occasion": "Gala", "year": "2001", "query": "q1"},
                  {"from": "Bob Ray", "to": "Cat Moe", "occasion": "Summit", "year": "2005", "query": "q2"}],
        "notes": "via Bob"}


def search_event(query):
    return {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "WebSearch",
                                                          "input": {"query": query}}]}}


def result_event(structured=None, is_error=False, result="", subtype="success"):
    return {"type": "result", "subtype": subtype, "is_error": is_error, "num_turns": 3,
            "result": result, "structured_output": structured}


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_DIR", str(tmp_path))

    def script(*runs):
        (tmp_path / "runs.json").write_text(json.dumps(list(runs)))
        (tmp_path / "calls.jsonl").unlink(missing_ok=True)
        return ClaudeCodePlanner(exe=FAKE, timeout=20)

    def calls():
        path = tmp_path / "calls.jsonl"
        return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []

    script.calls = calls
    return script


def test_plans_through_headless_claude_code(fake, monkeypatch, caplog):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-used")
    planner = fake({"events": [{"type": "system", "subtype": "init"}, search_event("Ann Lee Bob Ray gala photo"),
                               result_event(GOOD)]})
    with caplog.at_level("INFO", logger="sixdegrees"):
        links = planner.plan("Ann Lee", "Cat Moe")
    assert [(l.a, l.b) for l in links] == [("Ann Lee", "Bob Ray"), ("Bob Ray", "Cat Moe")]
    assert "searching: Ann Lee Bob Ray gala photo" in caplog.text
    call = fake.calls()[0]
    argv = call["argv"]
    assert argv[0] == "-p" and "Ann Lee" in call["prompt"]  # prompt goes in on stdin
    assert argv[argv.index("--tools") + 1] == "WebSearch,WebFetch"
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert argv[argv.index("--model") + 1] == "opus"
    assert json.loads(argv[argv.index("--json-schema") + 1])["required"] == ["chain", "notes"]
    assert "--bare" not in argv  # --bare would ignore the Claude Code login
    assert not call["has_api_key"]  # uses the Claude Code login, not an API key from .env
    assert Path(call["cwd"]).name.startswith("sixdegrees-cc-")


def test_invalid_answer_is_retried_with_feedback(fake):
    bad = {"chain": [{"from": "Ann Lee", "to": "Dan Fox", "occasion": "", "year": "", "query": ""}], "notes": ""}
    planner = fake({"events": [result_event(bad)]}, {"events": [result_event(None, result=json.dumps(GOOD))]})
    assert [l.b for l in planner.plan("Ann Lee", "Cat Moe")] == ["Bob Ray", "Cat Moe"]
    assert "not a valid chain" in fake.calls()[1]["prompt"]


def test_not_logged_in_and_crashes_are_reported(fake):
    planner = fake({"events": [result_event(is_error=True, result="Not logged in · Please run /login",
                                            subtype="error_during_execution")]})
    with pytest.raises(PlannerError, match="isn't logged in"):
        planner.plan("Ann Lee", "Cat Moe")
    planner = fake({"events": [], "exit": 1, "stderr": "boom\n"})
    with pytest.raises(PlannerError, match="exited without an answer: boom"):
        planner.plan("Ann Lee", "Cat Moe")


def test_timeout(fake):
    planner = fake({"events": [result_event(GOOD)], "sleep": 5})
    planner.timeout = 0.5
    with pytest.raises(PlannerError, match="timed out"):
        planner.plan("Ann Lee", "Cat Moe")


def test_no_web_search_and_missing_cli(monkeypatch):
    p = ClaudeCodePlanner(exe="claude", web_search=False)
    argv = p.command()
    assert argv[argv.index("--tools") + 1] == "" and "--allowedTools" not in argv
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(PlannerError, match="isn't installed"):
        ClaudeCodePlanner()
