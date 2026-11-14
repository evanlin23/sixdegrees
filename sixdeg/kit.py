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
    # Title is always the plain question with the names people know (the reference channel's hits used bare
    # "A to B" titles; a question adds the curiosity gap). An episode can force one with "title_override".
    lab = lambda n: spec.get("people", {}).get(n, {}).get("label") or _short(n)
    # A/B episodes (hook_variant set) carry their short title in "title"; older episodes' "title" is stale text.
    title = (spec.get("title_override") or (spec.get("title") if spec.get("hook_variant") else None)
             or f"How is {lab(spec['from'])} connected to {lab(spec['to'])}?")
    if len(title) > 95:
        title = title[:92] + "..."
    chain = " → ".join(_short(p) for p in ppl)
    # the research, visible: every link's fact and its photo's source (YouTube's inauthentic-content policy looks
    # for "superficial differences"; the documented research is what makes each video original)
    research = []
    for i, (l, c) in enumerate(zip(spec["links"], fj["credits"])):
        who = (c.get("artist") or "").strip()
        who = f" by {who}" if who and len(who) < 80 and who.lower() != "unknown" else ""
        lin = l.get("lineage")
        extra = f" ({len(lin['people'])} portraits, every holder)" if lin else ""
        research.append(f"{i + 1}. {lab(l['a'])} → {lab(l['b'])}{extra}: {l['line']}")
        research.append(f"   Photo{who}, {c['license']}: {c['source']}")
    tags_inline = " ".join(dict.fromkeys([_tag(p) for p in (spec["from"], spec["to"])] +
                                         ["#sixdegrees", "#history", "#didyouknow", "#shorts"]))
    desc = "\n".join([
        spec.get("description") or f"{len(spec["links"])} pictures connect {a} to {b}.",
        "",
        chain,
        "",
        "How they're connected (every photo is real and sourced):",
        *research,
        *(["", "Research notes: " + spec["research_note"]] if spec.get("research_note") else []),
        "",
        "Give me two names in the comments and I'll find the photos that connect them.",
        "",
        tags_inline,
    ])
    if len(desc) > 4900:                                  # YouTube's limit is 5,000 characters
        desc = desc[:4890].rsplit("\n", 1)[0] + "\n…"
    tags = list(dict.fromkeys([_short(p) for p in ppl] + ["six degrees of separation", "six degrees", "history",
                                                          "connections", "small world"]))
    fun = f"Fun fact from the research: {spec['fun_fact']}\n\n" if spec.get("fun_fact") else ""
    ask = ("Who should Hitler connect to next? Name anyone (a character works too) and I'll find the photos."
           if "hitler" in slug.lower() else "Who should I connect next? Two names, the weirder the better.")
    first = spec.get("first_comment") or spec.get("pinned_comment") or fun + (
        f"{spec['requested_by']} asked for this one. " + ask
        if str(spec.get("requested_by", "")).startswith("@") else ask)        # never "a viewer" (owner)
    meta = {"title": title, "description": desc, "tags": tags[:30], "categoryId": "27",  # Education
            "first_comment": first,
            "madeForKids": False,
            # narrator is a generic TTS voice, not a real person, and photos are unaltered:
            # YouTube's altered/synthetic disclosure is not required
            "containsSyntheticMedia": False, "file": f"{slug}.mp4",
            "kind": "request" if spec.get("requested_by") else "formula"}
    vj = d / "out" / "voice.json"
    if vj.exists():
        vinfo = json.loads(vj.read_text())
        meta["voice"] = vinfo.get("engine")
        meta["outro"] = bool(vinfo.get("outro"))       # A/B test: spoken "And that's how." at the end
        meta["voice_rev"] = vinfo.get("rev", 1)
        meta["narration"] = vinfo.get("narration", True)   # A/B test: False = music-only
        meta["links"] = vinfo.get("links", len(spec["links"]))
        meta["lineage"] = vinfo.get("lineage", 0)          # portraits in lineage scrolls (0 = none)
        meta["music"] = vinfo.get("music")
        meta["style"] = vinfo.get("style", "cards")        # presentation-style pilot
        meta["trending"] = vinfo.get("trending", False)    # a claimed song: must stay under 60 s
        meta["hook_variant"] = vinfo.get("hook_variant")  # A/B: spoken-hook arm (A/B/F; None = old hook)
        meta["text_variant"] = vinfo.get("text_variant", "T0")   # A/B: opening text (T0 today, T1 band)
        meta["methods"] = vinfo.get("methods", [])          # "QUEEN METHOD" stamp shown, if any
    out = d / "out"
    out.mkdir(exist_ok=True)
    mp4 = out / f"{slug}.mp4"
    if mp4.exists() and not (out / "thumb.jpg").exists():  # the renderer writes thumb.jpg; fallback: a frame
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "0.9", "-i", str(mp4), "-frames:v", "1",
                        "-q:v", "3", str(out / "thumb.jpg")], check=True)
    (out / "metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    md = "\n".join([
        f"# Upload kit: {slug}", "",
        f"**File:** `{slug}.mp4`", "",
        "**Title**", "```", title, "```", "",
        "**Description**", "```", desc, "```", "",
        "**Settings:** Not made for kids · Category: Education · "
        "Altered or synthetic content: No (generic narrator voice, unaltered photos)", "",
        f"**Pinned comment**", "```",
        first,
        "```",
    ])
    (out / "upload.md").write_text(md)
    return md
