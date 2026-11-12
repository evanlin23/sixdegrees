"""Import a YouTube Studio metrics snapshot (docs/studio-metrics-prompt.md) and compare what works.

`python -m sixdeg studio <file.json>` stores the snapshot in data/studio.json (newest per video wins), joins each
Short with the upload log on the `queue` branch (slug, kind, outro, upload time), prints a per-Short table and
compares groups: opening (old small cards vs the big-face hook from PR #98), kind (viewer request vs viral formula),
outro ("And that's how." vs none). The number that decides reach on Shorts is "viewed vs swiped away" (viewed_pct);
HelloBuh's data: 74-81 % got 9k-26k views, 56-61 % stalled at 1-3k.
"""
from __future__ import annotations

import json
import statistics
import subprocess

from .common import DATA, ROOT

STUDIO = DATA / "studio.json"
OLD_OPENING = {"kim-jong-un-to-neil-armstrong", "marilyn-monroe-to-vladimir-putin"}   # the only small-card posts;
# everything from Rivaldo (2026-11-07) on has the big-face hook (PR #98). The 2025 uploads are another style again.


def _uploaded() -> dict[str, dict]:
    try:
        raw = subprocess.check_output(["git", "-C", str(ROOT), "show", "origin/queue:uploaded.json"], text=True)
    except Exception:  # noqa: BLE001 - no queue branch locally
        return {}
    out = {}
    for u in json.loads(raw):
        vid = u.get("id") or u.get("video_id")
        if vid:
            out[vid] = u
    return out


VILLAINS = ("stalin", "mussolini", "saddam", "gaddafi", "kim-jong", "mao-zedong", "lenin", "escobar", "capone",
            "castro", "attila", "genghis", "pol-pot", "idi-amin")


def _pair(slug: str) -> str:
    """Pair test bucket: Hitler, another instantly recognisable villain, or the rest."""
    s = slug.lower().replace(" ", "-")
    return "hitler" if "hitler" in s else ("other-villain" if any(v in s for v in VILLAINS) else "rest")


def _chain(u: dict) -> str:
    """Chain-length test bucket: short (<=6 links) vs parade (8+)."""
    n = u.get("links")
    if n is None and u.get("slug"):
        ep = ROOT / "episodes" / u["slug"] / "episode.json"
        n = len(json.loads(ep.read_text())["links"]) if ep.exists() else None
    if n is None:
        return "?"
    return "short<=6" if n <= 6 else ("mid7" if n == 7 else "parade8+")


def merge(snapshot: dict) -> dict:
    cur = json.loads(STUDIO.read_text()) if STUDIO.exists() else {"channel": {}, "videos": {}}
    cur["channel"] = snapshot.get("channel") or cur.get("channel", {})
    cur["collected_at"] = snapshot.get("collected_at")
    for v in snapshot.get("videos", []):
        if v.get("id"):
            cur["videos"][v["id"]] = v
    STUDIO.write_text(json.dumps(cur, indent=1, ensure_ascii=False) + "\n")
    return cur


def _when(u: dict, v: dict) -> str:
    # Studio's own publish time is the real one; the upload log's publish_at is the planned slot
    return (v.get("published") or u.get("uploaded_at") or u.get("publish_at") or "")[:16]


def report(cur: dict) -> str:
    up = _uploaded()
    rows = []
    for vid, v in cur["videos"].items():
        u = up.get(vid, {})
        when = _when(u, v)
        rows.append({
            "slug": u.get("slug") or v.get("title", vid)[:40], "when": when,
            "opening": ("cards" if u.get("slug") in OLD_OPENING else "big-face") if u else "2025/other",
            "kind": u.get("kind") or "?", "outro": {True: "outro", False: "no-outro"}.get(u.get("outro"), "?"),
            "chain": _chain(u), "style": u.get("style", "cards") if u else "?", "pair": _pair(u.get("slug") or v.get("title", "")), "voice": "music-only" if u.get("narration") is False else ("narrated" if u else "?"),
            "avg_s": v.get("avg_view_s"),
            "views": v.get("views"), "viewed": v.get("viewed_pct"), "avg_pct": v.get("avg_pct_viewed"),
            "r3": (v.get("retention_pct") or {}).get("3s"), "end": (v.get("retention_pct") or {}).get("end"),
            "subs": v.get("subs_gained"), "likes": v.get("likes"),
        })
    rows.sort(key=lambda r: r["when"])
    lines = [f"{'Short':40} {'up':16} {'views':>7} {'viewed%':>7} {'avg%':>5} {'@3s':>4} {'end':>4} {'subs':>4}"]
    for r in rows:
        f = lambda x: "-" if x is None else (f"{x:.0f}" if isinstance(x, (int, float)) else str(x))  # noqa: E731
        lines.append(f"{r['slug'][:40]:40} {r['when']:16} {f(r['views']):>7} {f(r['viewed']):>7} {f(r['avg_pct']):>5} "
                     f"{f(r['r3']):>4} {f(r['end']):>4} {f(r['subs']):>4}")
    lines.append("")
    for key in ("opening", "pair", "kind", "outro", "chain", "voice", "style"):
        groups = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r)
        parts = []
        for g, rs in sorted(groups.items()):
            vp = [r["viewed"] for r in rs if r["viewed"] is not None]
            vw = [r["views"] for r in rs if r["views"] is not None]
            parts.append(f"{g}: n={len(rs)}, median viewed {statistics.median(vp):.0f}%" if vp else f"{g}: n={len(rs)}")
            if vw:
                parts[-1] += f", median views {statistics.median(vw):.0f}"
            av = [r["avg_s"] for r in rs if r["avg_s"] is not None]
            if av:
                parts[-1] += f", median {statistics.median(av):.0f}s watched"
        lines.append(f"{key:8} " + " | ".join(parts))
    ch = cur.get("channel") or {}
    if ch.get("busiest_times"):
        lines.append("busiest: " + ", ".join(ch["busiest_times"]))
    return "\n".join(lines)


def run(path: str) -> str:
    text = open(path, encoding="utf-8").read()
    if "```" in text:                       # a pasted answer with the JSON inside a code block
        text = text.split("```json", 1)[-1].split("```", 1)[0] if "```json" in text else text.split("```")[1]
    return report(merge(json.loads(text)))
