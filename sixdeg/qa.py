"""Automated gates. A video is publishable only if qa.json says pass.

Hard gates (fail):
  - chain is consistent and every person in every photo is verified (sface or manual)
  - 1080x1920, 30 fps, 10-180 s
  - loudness -14 LUFS +/- 1.5, true peak <= -1 dBTP
  - every name the script says is heard in the Whisper transcript of the final audio
  - no dead air: first word within 0.25 s, no gap between words over 2 s, video ends <= 1 s after the last word
  - every photo has a source + license record
Warnings: non-free (fair use) photos, low face-match scores, long labels.
"""
from __future__ import annotations

import difflib
import json
import re
import subprocess

from .common import CACHE
from .episode import check_chain, load, people_order


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    st = j["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    return st["width"], st["height"], float(num) / float(den), float(j["format"]["duration"])


def _loudness(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    tail = r.stderr[r.stderr.rfind("Summary:"):]
    i = re.search(r"I:\s+(-?[\d.]+) LUFS", tail)
    p = re.search(r"Peak:\s+(-?[\d.]+) dBFS", tail)
    return (float(i.group(1)) if i else None, float(p.group(1)) if p else None)


def _norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


_ABBR = {"st": "saint", "mt": "mount", "dr": "doctor", "jr": "junior"}


def _heard(name_word, heard_words, cutoff=0.75):
    """Was this name word in the transcript? Whisper spells names its own way: it writes 'St.' for Saint,
    splits 'Jung-jae' into two words and approximates rare surnames, so also try adjacent word pairs."""
    t = _norm(name_word)
    if not t:
        return True
    hw = [_ABBR.get(_norm(h), _norm(h)) for h in heard_words]
    cands = hw + [a + b for a, b in zip(hw, hw[1:])]
    return any(difflib.SequenceMatcher(None, t, h).ratio() >= cutoff for h in cands)


def qa(slug: str) -> dict:
    d, spec = load(slug)
    out = d / "out"
    fails, warns = [], []
    fails += check_chain(spec)

    fj = json.loads((d / "work" / "faces.json").read_text())
    for i, (link, l) in enumerate(zip(spec["links"], fj["links"])):
        for role in ("a", "b"):
            m = l[role]["method"]
            who = link[role]
            if m not in ("sface", "manual", "lineage"):
                fails.append(f"photo {i + 1}: {who} not verified ({m}) - pin the face after checking review{i}.jpg")
            elif m == "sface" and l[role]["sim"] is not None and l[role]["sim"] < 0.45:
                warns.append(f"photo {i + 1}: {who} face match is weak ({l[role]['sim']})")
        if l["a"]["box"] and l["a"]["box"] == l["b"]["box"] and not l.get("lineage"):
            fails.append(f"photo {i + 1}: both people assigned to the same face")
    for i, c in enumerate(fj["credits"]):
        if not c.get("source") or not c.get("license") or c["license"] == "unknown":
            fails.append(f"photo {i + 1}: missing source/license record")
        if not c.get("free"):
            warns.append(f"photo {i + 1}: non-free image ({c['license']}) - relies on fair use commentary")

    vj = out / "voice.json"
    used = json.loads(vj.read_text()).get("engine") if vj.exists() else None
    from .audio import voice_config
    vinfo = json.loads(vj.read_text()) if vj.exists() else {}
    for t in (vinfo.get("clipped", []) if vinfo.get("narration", True) else []):   # music-only: voice unheard
        fails.append(f"narration line ends mid-word (clipped): '{t}' - re-voice it")
    if voice_config(spec.get("voice")).get("engine") == "gemini" and used != "gemini":
        fails.append("narration is not the Achird voice (local preview without a key, or Gemini quota); "
                     "only GitHub's render with the Achird voice can be published")
    mp4 = out / f"{slug}.mp4"
    res = {}
    if not mp4.exists():
        fails.append("no render")
    else:
        w, h, fps, dur = _probe(mp4)
        res.update(width=w, height=h, fps=round(fps, 2), duration=round(dur, 2))
        if (w, h) != (1080, 1920):
            fails.append(f"resolution {w}x{h}")
        if abs(fps - 30) > 0.01:
            fails.append(f"fps {fps}")
        if not 10 <= dur <= 180:   # Shorts have no minimum; a 2-link chain (Diddy -> presidents) is ~13 s
            fails.append(f"duration {dur:.1f}s outside 10-180s (Shorts max is 3 min)")
        if vj.exists() and json.loads(vj.read_text()).get("trending") and dur >= 59.5:
            fails.append(f"duration {dur:.1f}s with a trending (claimed) song: Shorts over 60 s with a claim are "
                         "blocked; shorten the chain or set \"music\" to a Mixkit track")
        elif dur > 60:
            warns.append(f"duration {dur:.1f}s: over 60 s, so keep lines tight for retention")
        lufs, peak = _loudness(mp4)
        res.update(lufs=lufs, true_peak=peak)
        if lufs is None or abs(lufs + 14) > 1.5:
            fails.append(f"loudness {lufs} LUFS (want -14)")
        if peak is not None and peak > -0.5:
            fails.append(f"true peak {peak} dBTP")
        # speech check on the final mix
        from .audio import transcribe, unknown_names
        from .audio import voice_config
        kokoro = voice_config(spec.get("voice")).get("engine", "kokoro") == "kokoro"
        for w in (unknown_names(" ".join([spec.get("hook", "")] + [l["line"] for l in spec["links"]])) if kokoro else []):
            fails.append(f"no pronunciation for '{w}': add its IPA to audio.PHONEMES (see the SKILL)")
        wav = CACHE / "qa" / f"{slug}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-ac", "1", "-ar", "16000", str(wav)],
                       check=True)
        for p in [wav.with_suffix(".wav.words.json")]:
            if p.exists():
                p.unlink()
        words = transcribe(wav)
        heard = [w for w, _, _ in words]
        res["transcript"] = " ".join(heard)
        # DEAD AIR (owner 2026-11-13: "no silence", "cut as soon as the voice is done", "in the hook there cannot be
        # any pauses"): the voice starts at once, never stops for long, and the video ends right after the last word.
        if words and dur:
            first, last_end = words[0][1], max(e for _, _, e in words)
            gaps = [(round(b[1] - a[2], 1), a[0]) for a, b in zip(words, words[1:]) if b[1] - a[2] > 2.0]
            res["dead_air"] = {"first_word": round(first, 2), "tail": round(dur - last_end, 2), "gaps": gaps}
            if first > 0.25:   # owner: the first word starts the moment the video is scrolled to
                fails.append(f"dead air at the start: first word at {first:.2f}s (the question must start on frame 0)")
            if dur - last_end > 1.0:
                fails.append(f"dead air at the end: {dur - last_end:.1f}s after the last word (cut ~0.5 s after the voice)")
            for g, after in gaps:
                fails.append(f"dead air: {g}s with no voice after '{after}' (no gap over 2 s)")
        script = " ".join([spec.get("hook", "")] + [l["line"] for l in spec["links"]] + [spec.get("outro", "")])
        # names with a hand-set or Wikipedia pronunciation are already verified; Whisper often can't spell
        # foreign names it hears correctly ("Jair" -> "Jourier"), so don't fail on those
        from .audio import PHONEMES
        from .pronounce import load as wiki_pron
        set_words = set(" ".join(list(PHONEMES) + list(wiki_pron())).split())
        narrated = json.loads(vj.read_text()).get("narration", True) if vj.exists() else True
        for name in (people_order(spec) if (used == "gemini" or used is None) and narrated else []):   # silent preview / music-only: nothing to hear
            core = re.sub(r"\s*\(.*?\)", "", name)
            for part in core.split():
                if (len(part) < 4 or part.lower() not in script.lower() or part in set_words
                        or re.fullmatch(r"[IVXL]+", part)):          # regnal numbers are spoken as "the Eighth"
                    continue
                # Achird reads the script as written (the batch split already checked every line was spoken),
                # so only a clear miss counts there; the Kokoro voice gets the strict check
                if not _heard(part, heard, 0.55 if used == "gemini" else 0.75):
                    fails.append(f"'{part}' is in the script but not heard in the audio (TTS mispronounced?) "
                                 f"- add a respelling to audio.RESPELL")
        # contact sheet for the reviewing session to look at
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                        "fps=2/3,scale=216:384,tile=8x3", "-frames:v", "1", str(out / "contact.jpg")], check=True)

    for name, p in spec.get("people", {}).items():
        if len(p.get("label", name)) > 22:
            warns.append(f"label for {name} is long")
    r = {"pass": not fails, "fails": fails, "warnings": warns, **res}
    out.mkdir(exist_ok=True)
    (out / "qa.json").write_text(json.dumps(r, indent=1))
    return r
