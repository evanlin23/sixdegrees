"""Every holder of an office, in order, with a portrait, for a lineage link ("the unbroken line of popes").

`python -m sixdeg lineage popes --from "Saint Peter" --to "Pope Francis"` prints the `lineage.people` list for
episode.json: ALL holders between the two, each with its Commons portrait and an ordinal label. The full line
scrolls past super fast; showing every one of them (not a hand-picked few) is the joke.
Data: Wikidata "position held" (P39) with start time (P580) and image (P18).
"""
from __future__ import annotations

import json
import re

from .common import cached_json, http_get

OFFICES = {   # alias -> (Wikidata QID of the position, label shown on the scroll)
    "popes": ("Q19546", "LINE OF POPES"),
    "us-presidents": ("Q11696", "LINE OF PRESIDENTS"),
    "roman-emperors": ("Q842606", "LINE OF EMPERORS"),
    "uk-monarchs": ("Q9134365", "LINE OF MONARCHS"),
    "english-monarchs": ("Q18810062", "LINE OF MONARCHS"),
    "french-monarchs": ("Q3439798", "LINE OF KINGS"),
}

_Q = """SELECT ?p ?pLabel ?img ?start WHERE {
  ?p p:P39 ?st . ?st ps:P39 wd:%s . ?st pq:P580 ?start .
  OPTIONAL { ?p wdt:P18 ?img }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } } ORDER BY ?start"""


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def holders(office: str) -> list[dict]:
    qid = OFFICES.get(office, (office, ""))[0]

    def fetch():
        r = http_get("https://query.wikidata.org/sparql", timeout=60, tries=3, params={"query": _Q % qid},
                     headers={"Accept": "application/sparql-results+json",
                              "User-Agent": "sixdegrees-buhconnects/1.0 (github.com/evanlin23/sixdegrees)"})
        return r.json()["results"]["bindings"]

    rows, seen, out = cached_json(f"lineage:{qid}", fetch), set(), []
    for b in rows:
        q = b["p"]["value"].rsplit("/", 1)[-1]
        if q in seen:            # someone who held it twice (Cleveland) appears once, at their first term
            continue
        seen.add(q)
        if b["pLabel"]["value"].lower().startswith(("pope-elect", "antipope")):   # not counted in the official line
            continue
        img = b.get("img", {}).get("value", "")
        out.append({"qid": q, "name": b["pLabel"]["value"], "start": b["start"]["value"][:10],
                    "file": "File:" + re.sub(r"%20", " ", img.rsplit("/", 1)[-1]) if img else None})
    for n, h in enumerate(out, 1):
        h["n"] = n
    return out


def people(office: str, a: str, b: str) -> dict:
    """The lineage block for episode.json, from holder `a` to holder `b` inclusive (names as Wikidata spells them;
    rename the first/last entry to match the link's a/b if your episode uses a different spelling)."""
    hs = holders(office)
    idx = {h["name"].lower(): i for i, h in enumerate(hs)}
    if a.lower() not in idx or b.lower() not in idx:
        missing = [x for x in (a, b) if x.lower() not in idx]
        raise SystemExit(f"not in the {office} list: {missing}. Names look like: {[h['name'] for h in hs[:3]]} ...")
    i, j = sorted((idx[a.lower()], idx[b.lower()]))
    span = hs[i:j + 1] if idx[a.lower()] <= idx[b.lower()] else list(reversed(hs[i:j + 1]))
    ppl = [{"name": h["name"], "file": h["file"], "label": _ordinal(h["n"])} for h in span if h["file"]
           or h in (span[0], span[-1])]
    return {"label": OFFICES.get(office, (office, "LINE OF SUCCESSION"))[1], "people": ppl}


def main(args: list[str]) -> str:
    office = args[0]
    a = args[args.index("--from") + 1] if "--from" in args else None
    b = args[args.index("--to") + 1] if "--to" in args else None
    if not (a and b):
        return "\n".join(f"{h['n']:>4} {h['start']}  {h['name']}" + ("" if h["file"] else "  (no portrait)")
                         for h in holders(office))
    return json.dumps(people(office, a, b), indent=1, ensure_ascii=False)
