"""Photo graph search: people are nodes, a captioned Wikipedia photo showing two
people is an edge. Bidirectional BFS from both endpoints.

Edges come from captions, so they are *candidates*: faces.py verifies them.
"""
from __future__ import annotations

import re
from collections import deque

from . import wiki

GENERIC_SKIP = re.compile(r"\b(grave|tomb|statue|memorial|plaque|signature|coat of arms|birthplace|house|"
                          r"poster|stamp|coin|banknote|map|diagram|logo|flag)\b", re.I)


def _surname(title: str) -> str:
    t = re.sub(r"\s*\(.*?\)", "", title).split()
    return t[-1].lower() if t else title.lower()


def neighbors(title: str, cap: int = 30) -> list[dict]:
    """People photographed with `title`, per captions in `title`'s article."""
    imgs = wiki.article_images(title)
    cand = []
    for im in imgs:
        if not im["links"] or im["file"].lower().endswith((".svg",)):
            continue
        if GENERIC_SKIP.search(im["caption"]):
            continue
        cand.append(im)
    linked = [l for im in cand for l in im["links"]]
    people = wiki.humans(linked)
    me = wiki.resolve(title)
    me_t = me["title"] if me else title
    sn = _surname(me_t)
    out, seen = [], set()
    for im in cand:
        captxt = im["caption"].lower()
        # how many people does the caption list? fewer = clearer photo
        ppl = [l for l in im["links"] if l in people and l != me_t]
        for other in ppl:
            if other in seen:
                continue
            score = 1.0
            if sn in captxt:
                score += 1.0       # caption names the subject too
            if re.search(r"\b(with|and|meets?|meeting|greets?|shaking hands)\b", captxt):
                score += 0.5
            score -= 0.15 * max(0, len(ppl) - 2)   # crowded group shots are harder
            out.append({"a": me_t, "b": other, "file": im["file"], "caption": im["caption"], "score": round(score, 2)})
            seen.add(other)
    out.sort(key=lambda e: -e["score"])
    return out[:cap]


def search(a: str, b: str, max_expand: int = 80, max_paths: int = 5, log=print) -> list[list[dict]]:
    """Bidirectional BFS. Returns up to max_paths paths (lists of edges a->...->b)."""
    A, B = wiki.resolve(a), wiki.resolve(b)
    if not A or not B:
        raise ValueError(f"could not resolve: {a if not A else b}")
    a_t, b_t = A["title"], B["title"]
    # parent maps: node -> edge that reached it (edge oriented away from the root)
    par = {0: {a_t: None}, 1: {b_t: None}}
    q = {0: deque([a_t]), 1: deque([b_t])}
    expanded = 0
    meets: list[tuple[str, dict | None]] = []  # (meeting node, bridging edge or None)

    def path_to(side, node):
        edges = []
        while par[side][node] is not None:
            e = par[side][node]
            edges.append(e)
            node = e["a"]
        return edges[::-1]

    side = 0
    while expanded < max_expand and (q[0] or q[1]) and len(meets) < max_paths * 3:
        if not q[side]:
            side ^= 1
            continue
        node = q[side].popleft()
        expanded += 1
        try:
            nb = neighbors(node)
        except Exception as ex:  # noqa: BLE001
            log(f"  ! {node}: {ex}")
            nb = []
        log(f"  [{'AB'[side]}] {node}: {len(nb)} photo-neighbors")
        for e in nb:
            o = e["b"]
            if o in par[side ^ 1]:
                meets.append((o, e, side))
            if o not in par[side]:
                par[side][o] = e
                q[side].append(o)
        side ^= 1

    paths, sigs = [], set()
    for o, e, s in meets:
        # e goes from node on side s to o, and o is known from the other side
        left = path_to(s, e["a"]) + [e]
        right = path_to(s ^ 1, o)
        # right is root(other)->...->o; flip it to o->...->root(other)
        flipped = [{**x, "a": x["b"], "b": x["a"]} for x in right[::-1]]
        full = left + flipped
        if s == 1:  # we built from B's side; flip whole path to read a -> b
            full = [{**x, "a": x["b"], "b": x["a"]} for x in full[::-1]]
        names = tuple([full[0]["a"]] + [x["b"] for x in full])
        if len(set(names)) != len(names) or names in sigs:
            continue
        sigs.add(names)
        paths.append(full)
    paths.sort(key=lambda p: (len(p), -sum(e["score"] for e in p)))
    return paths[:max_paths]


def describe(path: list[dict]) -> str:
    lines = [" -> ".join([path[0]["a"]] + [e["b"] for e in path])]
    for e in path:
        lines.append(f"  {e['a']} + {e['b']}  [{e['score']}]  {e['file']}\n      \"{e['caption'][:160]}\"")
    return "\n".join(lines)
