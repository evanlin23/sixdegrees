"""Stand-in for the `claude` CLI in tests: replays a scripted stream-json session.

FAKE_CLAUDE_DIR holds `runs.json` (a list of runs, one per invocation) and receives `calls.jsonl`.
Each run is {"events": [...], "exit": 0, "stderr": "", "sleep": 0}.
"""
import json
import os
import sys
import time
from pathlib import Path

state = Path(os.environ["FAKE_CLAUDE_DIR"])
prompt = sys.stdin.read()
calls = state / "calls.jsonl"
n = sum(1 for _ in open(calls)) if calls.exists() else 0
with open(calls, "a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "prompt": prompt, "cwd": os.getcwd(),
                        "has_api_key": "ANTHROPIC_API_KEY" in os.environ}) + "\n")
run = json.loads((state / "runs.json").read_text())[n]
time.sleep(run.get("sleep", 0))
for event in run.get("events", []):
    print(json.dumps(event), flush=True)
sys.stderr.write(run.get("stderr", ""))
sys.exit(run.get("exit", 0))
