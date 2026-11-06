"""Pull English pronunciations of people's names from Wikipedia.

Most Wikipedia biographies open with the name's English IPA in an {{IPAc-en|...}} template
(Putin /ˈpuːtɪn/, Monroe /mənˈroʊ/, Kim Jong Un /ˌkɪm dʒɒŋˈʊn/). We convert that to Kokoro's
phoneme style and store it in data/pronunciations.json, which audio.phonemes() uses ahead of the
CMU dictionary. Names Wikipedia gives no English IPA for (Neymar, Bolsonaro) still need a
hand-written entry in audio.PHONEMES; QA lists them.
"""
from __future__ import annotations

import json
import re

from .common import DATA, cached_json, http_get

PRON_FILE = DATA / "pronunciations.json"

# Wikipedia's English IPA (Help:IPA/English) -> Kokoro/espeak style
_MAP = [("ʤ", "dʒ"), ("ʧ", "tʃ"), ("aɪər", "aɪɚ"), ("aʊər", "aʊɚ"), ("ɜːr", "ɜː"), ("ɑːr", "ɑːɹ"), ("ɔːr", "ɔːɹ"),
        ("ɒr", "ɔːɹ"), ("ɪər", "ɪɹ"), ("ɛər", "ɛɹ"), ("ʊər", "ʊɹ"), ("ər", "ɚ"), ("ɒ", "ɑː"), ("r", "ɹ"), ("ʔ", ""),
        ("x", "k"), (".", ""), ("ɡ", "ɡ"), ("g", "ɡ"), ("'", "ˈ"), ("ʼ", "ˈ")]
_VOWELS = set("aeiouæɑɒɔəɚɛɜɪʊʌɐᵻ")
_LABELS = {l.lower() for l in ("lang", "pron", "US", "UK", "local", "also", "Aus", "NZ", "CA", "Can", "Ireland",
                                  "Scot", "nopron", "icon", "AU", "IE", "SA", "IN", "audio")}


def _token(t: str) -> str:
    for a, b in _MAP:
        t = t.replace(a, b)
    return t


def convert(template_args: list[str]) -> list[str] | None:
    """IPAc-en arguments -> list of Kokoro-style IPA words (one per name word), first variant only."""
    out, cur, pending, optional = [], "", "", False
    for raw in template_args:
        t = raw.strip()
        if t.startswith("("):  # optional sounds "(d)" are left out
            optional = True
        if optional:
            optional = not t.endswith(")")
            continue
        if not t or "=" in t or t.lower() in _LABELS:
            continue
        if t.startswith(","):  # ", ..." starts an alternative pronunciation: keep the first one only
            break
        if t == "-":
            continue
        if t in ("_", " ") or t.startswith("_"):
            if cur:
                out.append(cur)
            cur, pending = "", ""
            continue
        t = _token(t)
        if t in ("ˈ", "ˌ"):
            pending = t  # Wikipedia marks stress before the syllable; Kokoro wants it before the vowel
            continue
        if pending and t and t[0] in _VOWELS:
            t, pending = pending + t, ""
        cur += t
    if cur:
        out.append(cur)
    return out or None


def wiki_ipa(title: str) -> list[str] | None:
    def fetch():
        r = http_get("https://en.wikipedia.org/w/api.php", timeout=20, tries=3,
                     params={"action": "parse", "page": title, "prop": "wikitext", "section": 0, "format": "json",
                             "redirects": 1})
        w = r.json().get("parse", {}).get("wikitext", {}).get("*", "")
        m = re.search(r"\{\{IPAc-en\|([^}]*)\}\}", w)
        return {"args": m.group(1).split("|") if m else None}

    try:
        args = cached_json(f"ipa:{title}", fetch).get("args")
    except Exception:
        return None
    return convert(args) if args else None


def load() -> dict[str, str]:
    if PRON_FILE.exists():
        return {k: v for k, v in json.loads(PRON_FILE.read_text()).items() if not k.startswith("_")}
    return {}


_START = {"c": "kstʃs", "g": "ɡdʒʒ", "j": "dʒjhʒ", "q": "k", "x": "zks", "y": "jaɪi", "w": "wv", "h": "h",
          "z": "ztsdz", "t": "ttʃθð", "d": "ddʒ", "s": "sʃz", "k": "k", "p": "pf", "v": "vw"}


def plausible(word: str, ipa: str) -> bool:
    """Loose check that `ipa` is a pronunciation of `word` (rejects e.g. Hulk Hogan's IPA being for 'Bollea')."""
    w = word.lower()
    i = ipa.replace("ˈ", "").replace("ˌ", "")
    if not w or not i:
        return False
    if w[0] in "aeiou":
        return i[0] in _VOWELS or i[0] in "jw"
    return i[0] in _START.get(w[0], w[0])


def for_names(names: list[str]) -> dict[str, str]:
    """Look up each name on Wikipedia and return {word or phrase: ipa} for the words we can place."""
    found = {}
    for title in names:
        groups = wiki_ipa(title)
        if not groups:
            continue
        words = re.sub(r"\s*\(.*?\)", "", title).replace("-", " ").split()
        if len(groups) == len(words) and all(plausible(w, g) for w, g in zip(words, groups)):
            for w, ipa in zip(words, groups):
                found[w] = ipa
            if len(words) > 1:
                found[" ".join(words)] = " ".join(groups)
        elif len(groups) == 1 and len(words) > 1 and plausible(words[-1], groups[0]):  # surname only (Putin)
            found[words[-1]] = groups[0]
        # other shapes (Kim Jong Un, Charles de Gaulle) are ambiguous: leave them to PHONEMES / CMU
    return found


def update(names: list[str]) -> dict[str, str]:
    """Add Wikipedia pronunciations for `names` to data/pronunciations.json; returns what was added."""
    cur = json.loads(PRON_FILE.read_text()) if PRON_FILE.exists() else {
        "_about": "English pronunciations from Wikipedia {{IPAc-en}} (sixdeg/pronounce.py). audio.PHONEMES wins."}
    new = {k: v for k, v in for_names(names).items() if k not in cur}
    if new:
        cur.update(new)
        PRON_FILE.write_text(json.dumps(cur, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    return new
