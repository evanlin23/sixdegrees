"""Planners propose who-met-whom chains; the pipeline then checks every link against real photos.

- GeminiPlanner: asks Gemini (optionally grounded with Google Search) for photographed meetings.
- GraphPlanner: shortest paths through a pre-built "met" graph (createdb/met_people_graph.gexf).
- ManualPlanner: a chain you type in yourself.
"""
from __future__ import annotations

import difflib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .util import cache_dir, log, normalize_name, read_json, same_person, short_hash, write_json

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


class GeminiPlanner:
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

    def _generate(self, prompt: str) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(temperature=0.4)
        if self.grounding:
            config.tools = [types.Tool(google_search=types.GoogleSearch())]
        resp = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        return resp.text or ""

    def _ask_chain(self, prompt: str, start: str, target: str) -> Optional[list[Link]]:
        cache_path = cache_dir() / "llm" / f"{short_hash(self.model + str(self.grounding) + prompt, 24)}.json"
        if self.use_cache:
            cached = normalize_chain(read_json(cache_path), start, target)
            if cached:
                return cached
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
            links = normalize_chain(data, start, target)
            if links:
                write_json(cache_path, data)
                if data.get("notes"):
                    log.info("  Gemini: %s", data["notes"])
                return links
            log.debug("  unusable Gemini reply: %s", text[:500])
            feedback = (f"\n\nYour previous reply was not usable. Reply with ONLY the JSON object, starting with "
                        f"{start} and ending with {target}, each link continuing from the previous one.")
        return None

    def plan(self, start: str, target: str) -> list[Link]:
        fmt = FORMAT % {"start": start, "target": target}
        links = self._ask_chain(PLAN_PROMPT % {"start": start, "target": target, "rules": RULES, "format": fmt},
                                start, target)
        if not links:
            raise PlannerError(f"Gemini did not return a usable chain from {start} to {target}.")
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
            "format": FORMAT % {"start": current, "target": target}}
        return self._ask_chain(prompt, current, target)
