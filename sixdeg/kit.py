"""Upload kit: title, description (with the photo credits CC licenses require), tags.

Writes out/upload.md (human-readable) and out/metadata.json (for an API uploader).
"""
from __future__ import annotations

import json
import re

from .episode import load, people_order


def _short(n):
    return re.sub(r"\s*\(.*?\)", "", n)


def _tag(n):
    return "#" + re.sub(r"[^A-Za-z0-9]", "", _short(n)).lower()


def kit(slug: str) -> str:
    d, spec = load(slug)
    fj = json.loads((d / "work" / "faces.json").read_text())
    ppl = people_order(spec)
    a, b = _short(spec["from"]), _short(spec["to"])
    title = spec.get("title") or f"How is {a} connected to {b}?"
    if len(title) > 95:
        title = title[:92] + "..."
    chain = " → ".join(_short(p) for p in ppl)
    credits = []
    for i, c in enumerate(fj["credits"]):
        who = (c.get("artist") or "").strip()
        who = f" by {who}" if who and len(who) < 80 and who.lower() != "unknown" else ""
        credits.append(f"Photo {i + 1}{who}, {c['license']}: {c['source']}")
    tags_inline = " ".join(dict.fromkeys([_tag(p) for p in (spec["from"], spec["to"])] +
                                         ["#sixdegrees", "#history", "#didyouknow", "#shorts"]))
    desc = "\n".join([
        spec.get("description") or f"{len(spec['links'])} real photos connect {a} to {b}.",
        "",
        chain,
        "",
        "Give me two names in the comments and I'll find the photos that connect them.",
        "",
        "Photo credits:",
        *credits,
        "",
        tags_inline,
    ])
    tags = list(dict.fromkeys([_short(p) for p in ppl] + ["six degrees of separation", "six degrees", "history",
                                                          "connections", "small world"]))
    meta = {"title": title, "description": desc, "tags": tags[:30], "categoryId": "27",  # Education
            "madeForKids": False,
            # narrator is a generic TTS voice, not a real person, and photos are unaltered:
            # YouTube's altered/synthetic disclosure is not required
            "containsSyntheticMedia": False, "file": f"{slug}.mp4"}
    out = d / "out"
    out.mkdir(exist_ok=True)
    (out / "metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    md = "\n".join([
        f"# Upload kit: {slug}", "",
        f"**File:** `{slug}.mp4`", "",
        "**Title**", "```", title, "```", "",
        "**Description**", "```", desc, "```", "",
        "**Settings:** Not made for kids · Category: Education · "
        "Altered or synthetic content: No (generic narrator voice, unaltered photos)", "",
        f"**Pinned comment**", "```",
        spec.get("pinned_comment") or f"Who should I connect next? Two names, the weirder the better.",
        "```",
    ])
    (out / "upload.md").write_text(md)
    return md
