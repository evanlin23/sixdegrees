"""Automated gates. A video is publishable only if qa.json says pass.

Hard gates (fail):
  - chain is consistent and every person in every photo is verified (sface or manual)
  - 1080x1920, 30 fps, 15-60 s
  - loudness -14 LUFS +/- 1.5, true peak <= -1 dBTP
  - every name the script says is heard in the Whisper transcript of the final audio
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


def _heard(name_word, heard_words):
    t = _norm(name_word)
    if not t:
        return True
    return any(difflib.SequenceMatcher(None, t, _norm(h)).ratio() >= 0.75 for h in heard_words)


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
        if not 15 <= dur <= 180:
            fails.append(f"duration {dur:.1f}s outside 15-180s (Shorts max is 3 min)")
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
        for w in unknown_names(" ".join([spec.get("hook", "")] + [l["line"] for l in spec["links"]])):
            fails.append(f"no pronunciation for '{w}': add its IPA to audio.PHONEMES (see the SKILL)")
        wav = CACHE / "qa" / f"{slug}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-ac", "1", "-ar", "16000", str(wav)],
                       check=True)
        for p in [wav.with_suffix(".wav.words.json")]:
            if p.exists():
                p.unlink()
        heard = [w for w, _, _ in transcribe(wav)]
        res["transcript"] = " ".join(heard)
        script = " ".join([spec.get("hook", "")] + [l["line"] for l in spec["links"]] + [spec.get("outro", "")])
        # names with a hand-set or Wikipedia pronunciation are already verified; Whisper often can't spell
        # foreign names it hears correctly ("Jair" -> "Jourier"), so don't fail on those
        from .audio import PHONEMES
        from .pronounce import load as wiki_pron
        set_words = set(" ".join(list(PHONEMES) + list(wiki_pron())).split())
        for name in people_order(spec):
            core = re.sub(r"\s*\(.*?\)", "", name)
            for part in core.split():
                if len(part) < 4 or part.lower() not in script.lower() or part in set_words:
                    continue
                if not _heard(part, heard):
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
