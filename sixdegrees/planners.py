"""Planners propose who-met-whom chains; the pipeline then checks every link against real photos.

- ClaudeCodePlanner: runs Claude Code headlessly (`claude -p`) on your Claude login. No API key.
- ClaudePlanner: the same through the Claude API (ANTHROPIC_API_KEY), with web search.
- GeminiPlanner: the same with Gemini (optionally grounded with Google Search).
- GraphPlanner: shortest paths through a pre-built "met" graph (createdb/met_people_graph.gexf).
- ManualPlanner: a chain you type in yourself.
"""
from __future__ import annotations

import difflib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .util import cache_dir, log, normalize_name, read_json, same_person, short_hash, write_json

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"


class PlannerError(RuntimeError):
    pass


@dataclass
class Link:
    a: str
    b: str
    occasion: str = ""
    year: str = ""
    query: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Link":
        return cls(**{k: str(d.get(k) or "") for k in ("a", "b", "occasion", "year", "query")})


def links_from_people(people: list[str]) -> list[Link]:
    return [Link(a, b) for a, b in zip(people, people[1:])]


class ManualPlanner:
    """Use exactly the chain given; optionally let another planner patch links that fail."""

    def __init__(self, people: list[str], fallback=None):
        if len(people) < 2:
            raise PlannerError("A chain needs at least two people.")
        self.people = people
        self.fallback = fallback

    def plan(self, start: str, target: str) -> list[Link]:
        return links_from_people(self.people)

    def repair(self, verified: list[str], target: str, failed: list, remaining: list[Link]) -> Optional[list[Link]]:
        if self.fallback is None:
            return None
        return self.fallback.repair(verified, target, failed, remaining)


# ---------------------------------------------------------------- graph

def load_gexf(path: Path):
    """Read just the topology of a GEXF file (fast, and tolerant of messy node attributes)."""
    import xml.etree.ElementTree as ET

    import networkx as nx

    labels, g = {}, nx.Graph()
    for _, el in ET.iterparse(str(path), events=("end",)):
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "node":
            labels[el.get("id")] = el.get("label") or el.get("id")
            g.add_node(labels[el.get("id")])
            el.clear()
        elif tag == "edge":
            a, b = labels.get(el.get("source")), labels.get(el.get("target"))
            if a and b and a != b:
                g.add_edge(a, b)
            el.clear()
    return g


class GraphPlanner:
    def __init__(self, gexf_path: Path):
        log.info("Loading connection graph %s ...", gexf_path)
        self.g = load_gexf(Path(gexf_path))
        self.by_norm = {normalize_name(n): n for n in self.g.nodes}
        log.info("  %d people, %d connections", self.g.number_of_nodes(), self.g.number_of_edges())

    def node(self, name: str) -> str:
        key = normalize_name(name)
        if key in self.by_norm:
            return self.by_norm[key]
        close = difflib.get_close_matches(key, list(self.by_norm), n=1, cutoff=0.85)
        if close:
            return self.by_norm[close[0]]
        raise PlannerError(f"{name!r} is not in the connection graph.")

    def _path(self, start: str, target: str, banned: set) -> Optional[list[str]]:
        import networkx as nx

        s, t = self.node(start), self.node(target)
        view = self.g.subgraph(n for n in self.g.nodes if n not in banned or n in (s, t))
        deg = self.g.degree

        def cost(u, v, _attrs):
            # Edges to obscure nodes are usually LLM noise; well-connected hubs get photographed together.
            return 1.0 + 3.0 / math.sqrt(max(1, min(deg[u], deg[v])))

        try:
            path = nx.shortest_path(view, s, t, weight=cost)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        path[0], path[-1] = start, target
        return path

    def plan(self, start: str, target: str) -> list[Link]:
        path = self._path(start, target, set())
        if not path:
            raise PlannerError(f"No path between {start} and {target} in the graph.")
        return links_from_people(path)

    def repair(self, verified: list[str], target: str, failed: list, remaining: list[Link]) -> Optional[list[Link]]:
        for link, _reason in failed:
            try:
                a, b = self.node(link.a), self.node(link.b)
            except PlannerError:
                continue
            if self.g.has_edge(a, b):
                self.g.remove_edge(a, b)
        banned = set()
        for name in verified[:-1]:
            try:
                banned.add(self.node(name))
            except PlannerError:
                pass
        path = self._path(verified[-1], target, banned)
        return links_from_people(path) if path else None


# ---------------------------------------------------------------- Gemini

RULES = """The video is a chain of real photographs. Each photo shows two people together in the same
frame, and consecutive photos share one person (A with B, then B with C, ...).

Rules for every link:
- The two people must have been photographed TOGETHER (same frame): a meeting, state visit,
  red carpet, award show, TV interview on the same set, sports event, campaign rally, wedding...
- Appearing on the same show at different times, working at the same place, or being "associated"
  does NOT count. If you are not confident a photo of the pair exists online, route through
  someone else.
- Prefer heavily photographed "hubs": presidents and prime ministers, popes, royals, talk-show
  hosts, famous athletes, musicians at award shows.
- Every person is one individual (no bands, teams or groups), named by their English Wikipedia
  article title. Never repeat a person.
- Prefer short chains (2-5 links), but correctness matters more than length.
- For each link give the occasion, the year, and an image search query likely to find the photo
  (names plus occasion, e.g. "Barack Obama Pope Francis White House 2015")."""

FORMAT = """Reply with only a JSON object, no other text:
{"chain": [{"from": "...", "to": "...", "occasion": "...", "year": "...", "query": "..."}],
 "notes": "one sentence"}
The first "from" must be %(start)s, each "to" must be the next link's "from", and the last "to"
must be %(target)s."""

PLAN_PROMPT = """Find a chain of photographed meetings from %(start)s to %(target)s for a "six degrees
of separation" video.

%(rules)s

%(format)s"""

REPAIR_PROMPT = """We are building a "six degrees of separation" photo chain from %(origin)s to %(target)s.

Already confirmed with photos: %(verified)s.
Now we need to continue from %(current)s to %(target)s.

No usable photo could be found for these pairs, so do not use them again:
%(failed)s
%(avoid)s
%(rules)s

%(format)s"""


def extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a model reply (tolerates code fences and chatter)."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for c in candidates:
        for attempt in (c, re.sub(r",\s*([}\]])", r"\1", c)):
            try:
                data = json.loads(attempt)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue
    return None


def normalize_chain(data: Optional[dict], start: str, target: str) -> Optional[list[Link]]:
    """Validate a model's chain: it must run start -> target with each link continuing the last."""
    if not data or not isinstance(data.get("chain"), list) or not data["chain"]:
        return None
    links = []
    for item in data["chain"]:
        if not isinstance(item, dict) or not item.get("from") or not item.get("to"):
            return None
        links.append(Link(str(item["from"]).strip(), str(item["to"]).strip(), str(item.get("occasion") or ""),
                          str(item.get("year") or ""), str(item.get("query") or "")))
    # The model often writes "Donald J. Trump" for "Donald Trump": snap near-misses onto the names we use.
    if not _similar(links[0].a, start) or not _similar(links[-1].b, target):
        return None
    links[0].a, links[-1].b = start, target
    for prev, cur in zip(links, links[1:]):
        if not _similar(prev.b, cur.a):
            return None
        cur.a = prev.b
    seen = [normalize_name(start)] + [normalize_name(l.b) for l in links]
    if len(set(seen)) != len(seen):
        return None
    return links


def _similar(x: str, y: str) -> bool:
    """Same person, allowing for middle initials, dropped surnames and small typos."""
    a, b = normalize_name(x), normalize_name(y)
    if a == b:
        return True
    ta, tb = set(a.split()), set(b.split())
    if ta and tb and (ta <= tb or tb <= ta):
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.92


class _LLMPlanner:
    """Shared prompting and caching; subclasses implement `_request` for their API."""

    label = "LLM"
    format_text = FORMAT
    model = ""
    use_cache = True

    def _cache_key(self, prompt: str) -> str:
        return short_hash(self.label + self.model + prompt, 24)

    def _request(self, prompt: str, start: str, target: str) -> Optional[dict]:
        raise NotImplementedError

    def _ask_chain(self, prompt: str, start: str, target: str) -> Optional[list[Link]]:
        cache_path = cache_dir() / "llm" / f"{self._cache_key(prompt)}.json"
        if self.use_cache:
            cached = normalize_chain(read_json(cache_path), start, target)
            if cached:
                return cached
        data = self._request(prompt, start, target)
        links = normalize_chain(data, start, target)
        if not links:
            return None
        write_json(cache_path, data)
        if data.get("notes"):
            log.info("  %s: %s", self.label, data["notes"])
        return links

    def plan(self, start: str, target: str) -> list[Link]:
        fmt = self.format_text % {"start": start, "target": target}
        links = self._ask_chain(PLAN_PROMPT % {"start": start, "target": target, "rules": RULES, "format": fmt},
                                start, target)
        if not links:
            raise PlannerError(f"{self.label} did not return a usable chain from {start} to {target}.")
        return links

    def repair(self, verified: list[str], target: str, failed: list, remaining: list[Link]) -> Optional[list[Link]]:
        current = verified[-1]
        failed_lines = "\n".join(f"- {l.a} & {l.b}" + (f" ({l.occasion}, {l.year})" if l.occasion else "")
                                 + f": {reason}" for l, reason in failed) or "- (none)"
        avoid = ""
        if len(verified) > 1:
            avoid = "Do not use these people again (already in the chain): " + ", ".join(verified[:-1]) + ".\n"
        prompt = REPAIR_PROMPT % {
            "origin": verified[0], "target": target, "verified": " -> ".join(verified), "current": current,
            "failed": failed_lines, "avoid": avoid, "rules": RULES,
            "format": self.format_text % {"start": current, "target": target}}
        return self._ask_chain(prompt, current, target)


# ---------------------------------------------------------------- Claude

CLAUDE_FORMAT = """Use web search to confirm that each pair was really photographed together, then call the
submit_chain tool once with the whole chain. The first "from" must be %(start)s, each "to" must be the
next link's "from", and the last "to" must be %(target)s."""

_LINK_SCHEMA = {
    "type": "object",
    "properties": {
        "from": {"type": "string"},
        "to": {"type": "string"},
        "occasion": {"type": "string", "description": "Where/when they were photographed together"},
        "year": {"type": "string"},
        "query": {"type": "string", "description": "Image search query likely to find the photo"},
    },
    "required": ["from", "to", "occasion", "year", "query"],
    "additionalProperties": False,
}

SUBMIT_CHAIN_TOOL = {
    "name": "submit_chain",
    "description": "Submit the finished chain of photographed pairs, in order from the first person to the last.",
    "input_schema": {
        "type": "object",
        "properties": {"chain": {"type": "array", "items": _LINK_SCHEMA},
                       "notes": {"type": "string", "description": "One sentence about the route"}},
        "required": ["chain", "notes"],
        "additionalProperties": False,
    },
    "strict": True,
    "eager_input_streaming": True,
}

# Before the last `fallback` marker, only these block types may be echoed back (plus paired results).
_ECHO_BEFORE_FALLBACK = {"text", "fallback", "server_tool_use"}


def echoable(content) -> list:
    """Assistant content that is safe to send back after a mid-output model fallback: before the
    last `fallback` block keep only text and paired server-tool blocks."""
    blocks = list(content)
    marks = [i for i, b in enumerate(blocks) if b.type == "fallback"]
    if not marks:
        return blocks
    paired = {getattr(b, "tool_use_id", None) for b in blocks if b.type.endswith("_tool_result")}
    keep = []
    for i, b in enumerate(blocks):
        if i < marks[-1]:
            if b.type.endswith("_tool_result"):
                pass
            elif b.type not in _ECHO_BEFORE_FALLBACK or (b.type == "server_tool_use" and b.id not in paired):
                continue
        keep.append(b)
    return keep


class ClaudePlanner(_LLMPlanner):
    """Claude researches with web search, then returns the chain through a strict `submit_chain` tool.

    Server-side refusal fallback is on (`fallbacks: "default"`): if Claude's safety classifiers
    decline a request, the API re-runs it on Anthropic's recommended fallback model.
    """

    label = "Claude"
    format_text = CLAUDE_FORMAT
    MAX_TURNS = 8

    def __init__(self, model: Optional[str] = None, effort: str = "high", web_search: bool = True,
                 use_cache: bool = True, client=None):
        if client is None:
            for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
                if os.environ.get(var) == "":  # an empty value (e.g. from .env) would shadow an `ant auth login` profile
                    del os.environ[var]
            try:
                import anthropic
            except ImportError as e:
                raise PlannerError("Claude needs `pip install anthropic`.") from e
            client = anthropic.Anthropic()  # ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile
        self.client = client
        self.model = model or os.environ.get("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL
        self.use_cache = use_cache
        # Optional request features; each is dropped if the API rejects it (older model, org setting).
        self.features = {"web_search": web_search, "fallback": True, "effort": True}
        self.effort = effort

    def _cache_key(self, prompt: str) -> str:
        return short_hash(self.label + self.model + str(self.features["web_search"]) + prompt, 24)

    def _create(self, messages: list):
        tools = []
        if self.features["web_search"]:
            tools.append({"type": "web_search_20260209", "name": "web_search", "max_uses": 10})
        tools.append(SUBMIT_CHAIN_TOOL)
        kwargs = {}
        if self.features["fallback"]:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        if self.features["effort"]:
            kwargs["output_config"] = {"effort": self.effort}
        with self.client.beta.messages.stream(model=self.model, max_tokens=32000, tools=tools,
                                              messages=messages, **kwargs) as stream:
            return stream.get_final_message()

    def _send(self, messages: list):
        """One request, turning API errors into retries, feature downgrades or a PlannerError."""
        import anthropic

        for attempt in range(4):
            try:
                return self._create(messages)
            except anthropic.AuthenticationError as e:
                raise PlannerError("Claude rejected the credentials: set ANTHROPIC_API_KEY or run `ant auth login`.") from e
            except anthropic.NotFoundError as e:
                raise PlannerError(f"Claude model {self.model!r} not found: {e.message}") from e
            except anthropic.BadRequestError as e:
                msg = str(e.message).lower()
                feature = next((f for f in ("web_search", "fallback", "effort") if self.features[f] and f in msg), None)
                if feature is None:
                    raise PlannerError(f"Claude request rejected: {e.message}") from e
                log.warning("  Claude rejected %s (%s); retrying without it.", feature, e.message)
                self.features[feature] = False
            except anthropic.RateLimitError as e:
                wait = float(e.response.headers.get("retry-after") or 20 * (attempt + 1))
                log.warning("  Claude rate limit; waiting %.0fs.", wait)
                time.sleep(wait)
            except (anthropic.APIConnectionError, anthropic.InternalServerError) as e:
                log.warning("  Claude request failed (%s); retrying.", e)
                time.sleep(5 * (attempt + 1))
            except TypeError as e:  # the SDK raises this when it finds no credentials at all
                if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                    raise PlannerError("No Claude credentials: set ANTHROPIC_API_KEY or run `ant auth login`.") from e
                raise
        raise PlannerError("Claude kept failing; try again later.")

    def _request(self, prompt: str, start: str, target: str) -> Optional[dict]:
        messages: list = [{"role": "user", "content": prompt}]
        for _ in range(self.MAX_TURNS):
            msg = self._send(messages)
            iterations = getattr(getattr(msg, "usage", None), "iterations", None) or []
            if any(getattr(it, "type", "") == "fallback_message" for it in iterations):
                log.info("  (answered by fallback model %s)", msg.model)
            if msg.stop_reason == "refusal":
                details = getattr(msg, "stop_details", None)
                log.warning("  Claude declined this request (%s).", getattr(details, "category", None) or "no category")
                return None
            if msg.stop_reason == "pause_turn":  # long web research: send the turn back and the server resumes
                messages.append({"role": "assistant", "content": echoable(msg.content)})
                continue
            if msg.stop_reason == "max_tokens":
                log.warning("  Claude ran out of output tokens before finishing.")
                return None
            call = next((b for b in msg.content if b.type == "tool_use" and b.name == "submit_chain"), None)
            if call is not None:
                data = call.input if isinstance(call.input, dict) else None
                if normalize_chain(data, start, target):
                    return data
                problem = (f"Not a valid chain. It must start with {start}, end with {target}, each link's "
                           f"'from' must equal the previous link's 'to', and no person may appear twice.")
                echoed = echoable(msg.content)
                messages.append({"role": "assistant", "content": echoed})
                if any(b is call for b in echoed):
                    problem = [{"type": "tool_result", "tool_use_id": call.id, "is_error": True, "content": problem}]
                messages.append({"role": "user", "content": problem})
                continue
            text = "".join(getattr(b, "text", "") for b in msg.content if b.type == "text")
            data = extract_json(text)
            if normalize_chain(data, start, target):
                return data
            messages.append({"role": "assistant", "content": echoable(msg.content)})
            messages.append({"role": "user", "content": "Please call the submit_chain tool with the complete chain."})
        return None


# ---------------------------------------------------------------- Claude Code

CLAUDE_CODE_FORMAT = """Use web search to confirm that each pair was really photographed together, then give the
chain as your structured output. The first "from" must be %(start)s, each "to" must be the next link's
"from", and the last "to" must be %(target)s."""


class ClaudeCodePlanner(_LLMPlanner):
    """Plan with Claude Code in headless mode (`claude -p`), using whatever Claude Code is logged in
    with (e.g. your Claude Pro/Max plan), so no API key is involved.

    Claude Code only gets its WebSearch and WebFetch tools here (no shell, no file edits), and returns
    the chain as JSON validated against the same schema the API planner uses.
    """

    label = "Claude Code"
    format_text = CLAUDE_CODE_FORMAT

    def __init__(self, model: Optional[str] = None, effort: str = "high", web_search: bool = True,
                 use_cache: bool = True, exe=None, timeout: float = 900):
        exe = exe or shutil.which("claude")
        if not exe:
            raise PlannerError("Claude Code isn't installed or isn't on PATH (https://code.claude.com). "
                               "Install it and run `claude` once to log in.")
        self.cmd = list(exe) if isinstance(exe, (list, tuple)) else [exe]
        self.model = model or os.environ.get("CLAUDE_CODE_MODEL") or "opus"
        self.effort = effort
        self.web_search = web_search
        self.use_cache = use_cache
        self.timeout = timeout

    def _cache_key(self, prompt: str) -> str:
        return short_hash(self.label + self.model + str(self.web_search) + prompt, 24)

    def command(self) -> list[str]:
        tools = ["WebSearch", "WebFetch"] if self.web_search else []
        cmd = self.cmd + ["-p", "--output-format", "stream-json", "--verbose",
                          "--json-schema", json.dumps(SUBMIT_CHAIN_TOOL["input_schema"]),
                          "--model", self.model, "--effort", self.effort, "--permission-mode", "dontAsk",
                          "--no-session-persistence", "--strict-mcp-config", "--tools", ",".join(tools)]
        if tools:
            cmd += ["--allowedTools"] + tools  # pre-approve them; nothing else is allowed in dontAsk mode
        return cmd

    def _run(self, prompt: str) -> Optional[dict]:
        """Run one headless session, echoing its web searches, and return the final `result` event."""
        env = dict(os.environ)
        for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
            env.pop(var, None)  # use the Claude Code login, not an API key that may be in .env
        with tempfile.TemporaryDirectory(prefix="sixdegrees-cc-") as cwd:  # no CLAUDE.md or project settings
            proc = subprocess.Popen(self.command(), cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            timed_out = threading.Event()

            def kill():
                timed_out.set()
                proc.kill()

            timer = threading.Timer(self.timeout, kill)
            timer.start()
            result, self.searches = None, 0
            try:
                proc.stdin.write(prompt)
                proc.stdin.close()
                for line in proc.stdout:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "assistant":
                        for block in event.get("message", {}).get("content", []) or []:
                            if block.get("type") == "tool_use" and block.get("name") in ("WebSearch", "WebFetch"):
                                what = block.get("input", {}).get("query") or block.get("input", {}).get("url")
                                self.searches += block["name"] == "WebSearch"
                                log.info("    %s: %s", "searching" if block["name"] == "WebSearch" else "reading", what)
                    elif event.get("type") == "result":
                        result = event
                proc.wait()
            finally:
                timer.cancel()
                if proc.poll() is None:
                    proc.kill()
            stderr = proc.stderr.read() if proc.stderr else ""
        if result is None:
            if timed_out.is_set():
                raise PlannerError(f"Claude Code timed out after {self.timeout:.0f}s")
            hint = " ".join(stderr.strip().splitlines()[-3:]) or f"exit code {proc.returncode}"
            raise PlannerError(f"Claude Code exited without an answer: {hint}")
        return result

    def _request(self, prompt: str, start: str, target: str) -> Optional[dict]:
        feedback = ""
        for _ in range(2):
            result = self._run(prompt + feedback)
            if result.get("is_error") or result.get("subtype") != "success":
                message = str(result.get("result") or result.get("subtype") or "unknown error")
                if any(w in message.lower() for w in ("log in", "login", "authenticat", "/login")):
                    raise PlannerError(f"Claude Code isn't logged in ({message}). Run `claude` once and sign in.")
                log.warning("  Claude Code failed: %s", message[:300])
                return None
            log.info("  Claude Code answered in %s turns with %d web searches.", result.get("num_turns", "?"),
                     self.searches)
            data = result.get("structured_output") or extract_json(str(result.get("result") or ""))
            if normalize_chain(data, start, target):
                return data
            feedback = (f"\n\nA previous answer was not a valid chain: it must start with {start}, end with {target}, "
                        f"each link's 'from' must equal the previous link's 'to', and nobody may appear twice.")
        return None


# ---------------------------------------------------------------- Gemini

class GeminiPlanner(_LLMPlanner):
    label = "Gemini"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, grounding: bool = True,
                 use_cache: bool = True):
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise PlannerError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or a .env file.")
        try:
            from google import genai
        except ImportError as e:
            raise PlannerError("Gemini needs `pip install google-genai`.") from e
        self.client = genai.Client(api_key=key)
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self.grounding = grounding
        self.use_cache = use_cache

    def _cache_key(self, prompt: str) -> str:
        return short_hash(self.model + str(self.grounding) + prompt, 24)

    def _generate(self, prompt: str) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(temperature=0.4)
        if self.grounding:
            config.tools = [types.Tool(google_search=types.GoogleSearch())]
        resp = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        return resp.text or ""

    def _request(self, prompt: str, start: str, target: str) -> Optional[dict]:
        feedback = ""
        for attempt in range(3):
            try:
                text = self._generate(prompt + feedback)
            except Exception as e:
                if self.grounding and attempt == 0:
                    log.warning("  Gemini call with Google Search grounding failed (%s); retrying without it.", e)
                    self.grounding = False
                    continue
                log.warning("  Gemini call failed: %s", e)
                time.sleep(5 * (attempt + 1))
                continue
            data = extract_json(text)
            if normalize_chain(data, start, target):
                return data
            log.debug("  unusable Gemini reply: %s", text[:500])
            feedback = (f"\n\nYour previous reply was not usable. Reply with ONLY the JSON object, starting with "
                        f"{start} and ending with {target}, each link continuing from the previous one.")
        return None
