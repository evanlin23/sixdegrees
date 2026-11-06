"""Wikipedia / Wikidata access.

Commons' own API rate-limits this cloud IP hard, but every Commons file is
reachable through en.wikipedia's API (shared file repo), so we go through
en.wikipedia for file info and through Wikidata SPARQL for "is this a human".
"""
from __future__ import annotations

import html
import re
from urllib.parse import unquote

from .common import cached_json, http_get

API = "https://en.wikipedia.org/w/api.php"
SPARQL = "https://query.wikidata.org/sparql"


def _api(**params):
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    params.setdefault("maxlag", "5")
    key = "wp:" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    return cached_json(key, lambda: http_get(API, params=params).json())


# ---------------------------------------------------------------- people

def resolve(name: str) -> dict | None:
    """Name -> {title, qid, description, portrait} (follows redirects, falls back to search)."""
    def lookup(t):
        d = _api(action="query", titles=t, redirects=1,
                 prop="pageprops|pageimages|description", piprop="name", ppprop="wikibase_item")
        pages = d.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return None
        p = pages[0]
        if "disambiguation" in (p.get("description") or "").lower():
            return None
        return {
            "title": p["title"],
            "qid": p.get("pageprops", {}).get("wikibase_item"),
            "description": p.get("description", ""),
            "portrait": ("File:" + p["pageimage"]) if p.get("pageimage") else None,
        }

    r = lookup(name)
    if r:
        return r
    s = _api(action="query", list="search", srsearch=name, srlimit=1)
    hits = s.get("query", {}).get("search", [])
    return lookup(hits[0]["title"]) if hits else None


def humans(titles: list[str]) -> set[str]:
    """Subset of article titles whose Wikidata item is an instance of human (Q5)."""
    titles = list(dict.fromkeys(t for t in titles if t))
    out: set[str] = set()
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        d = _api(action="query", titles="|".join(chunk), redirects=1, prop="pageprops", ppprop="wikibase_item")
        q = d.get("query", {})
        # map requested -> final title
        alias = {}
        for kind in ("normalized", "redirects"):
            for n in q.get(kind, []):
                alias[n["from"]] = n["to"]
        qid_of = {p["title"]: p.get("pageprops", {}).get("wikibase_item") for p in q.get("pages", [])}
        qids = {v for v in qid_of.values() if v}
        if not qids:
            continue
        sparql = "SELECT ?i WHERE { VALUES ?i { %s } ?i wdt:P31 wd:Q5 }" % " ".join("wd:" + x for x in qids)
        res = cached_json("sparql:" + sparql, lambda: http_get(
            SPARQL, params={"query": sparql, "format": "json"},
            headers={"Accept": "application/sparql-results+json"}).json())
        human_q = {b["i"]["value"].rsplit("/", 1)[1] for b in res["results"]["bindings"]}
        for t in chunk:
            final = alias.get(alias.get(t, t), alias.get(t, t))
            if qid_of.get(final) in human_q:
                out.add(final)
    return out


# ---------------------------------------------------------------- files

STD_WIDTHS = (330, 500, 960, 1280, 1920)  # thumb sizes Wikimedia serves to this IP


def _thumb_url(ii: dict, ow: int) -> str:
    """A thumb.wikimedia.org URL: originals are rate-limited hard (retry-after 600) from this IP.

    When the requested bucket equals the original width the API returns the original URL, so build
    the thumb URL ourselves (standard buckets are served even at the original's own width)."""
    t = (ii.get("thumburl") or ii["url"]).split("?")[0]
    if "/thumb/" in t:
        return t
    orig = ii["url"].split("?")[0]
    m = re.match(r"https://upload\.wikimedia\.org/wikipedia/commons/(\w/\w\w)/(.+)$", orig)
    if not m or not orig.lower().endswith((".jpg", ".jpeg", ".png")):
        return orig
    fits = [w for w in (250,) + STD_WIDTHS if w <= ow]
    if not fits:          # smaller than every standard width: Wikimedia only serves the original (no 212px thumb)
        return orig
    name = m.group(2)
    return f"https://thumb.wikimedia.org/wikipedia/commons/thumb/{m.group(1)}/{name}/{fits[-1]}px-{name}"


def file_info(file_title: str, width: int = 1920) -> dict:
    """File:X -> {url, thumb, width, height, license, artist, credit, description, page}."""
    if not file_title.startswith("File:"):
        file_title = "File:" + file_title
    d0 = _api(action="query", titles=file_title, prop="imageinfo", iiprop="size")
    p0 = d0["query"]["pages"][0]
    if "imageinfo" not in p0:
        raise ValueError(f"no such file: {file_title}")
    ow = p0["imageinfo"][0]["width"]
    # originals get 429 (retry-after 600) from this IP; standard thumb buckets don't
    fits = [w for w in STD_WIDTHS if w <= min(width, ow)]
    tw = fits[-1] if fits else STD_WIDTHS[0]
    d = _api(action="query", titles=file_title, prop="imageinfo",
             iiprop="url|size|extmetadata|mime", iiurlwidth=str(tw))
    p = d["query"]["pages"][0]
    if "imageinfo" not in p:
        raise ValueError(f"no such file: {file_title}")
    ii = p["imageinfo"][0]
    md = ii.get("extmetadata", {})

    def m(k):
        v = md.get(k, {}).get("value", "")
        return html.unescape(re.sub(r"<[^>]+>", "", str(v))).strip()

    return {
        "file": p["title"],
        "url": ii["url"].split("?")[0],
        "thumb": _thumb_url(ii, ow),
        "width": ii["width"], "height": ii["height"], "mime": ii.get("mime"),
        "license": m("LicenseShortName") or "unknown",
        "license_url": m("LicenseUrl"),
        "artist": m("Artist"),
        "credit": m("Credit"),
        "description": m("ImageDescription")[:500],
        "date": m("DateTimeOriginal"),
        "page": ii.get("descriptionurl"),
        "nonfree": m("NonFree").lower() == "true",
    }


# ---------------------------------------------------------------- article images

_FILE_RE = re.compile(r"\[\[\s*(?:File|Image)\s*:", re.I)
_LINK_RE = re.compile(r"\[\[([^\[\]|#]+)(?:#[^\[\]|]*)?(?:\|[^\[\]]*)?\]\]")
_IMG_EXT = (".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".webp")


def _balanced(text: str, start: int) -> int:
    """Index just past the ]] that closes the [[ at `start`."""
    depth, i = 0, start
    while i < len(text) - 1:
        if text.startswith("[[", i):
            depth += 1; i += 2; continue
        if text.startswith("]]", i):
            depth -= 1; i += 2
            if depth == 0:
                return i
            continue
        i += 1
    return len(text)


def _split_top(s: str) -> list[str]:
    parts, depth, cur = [], 0, []
    i = 0
    while i < len(s):
        if s.startswith("[[", i) or s.startswith("{{", i):
            depth += 1; cur.append(s[i:i + 2]); i += 2; continue
        if s.startswith("]]", i) or s.startswith("}}", i):
            depth -= 1; cur.append(s[i:i + 2]); i += 2; continue
        if s[i] == "|" and depth == 0:
            parts.append("".join(cur)); cur = []; i += 1; continue
        cur.append(s[i]); i += 1
    parts.append("".join(cur))
    return parts


def _clean_caption(c: str) -> str:
    c = re.sub(r"\{\{[^{}]*\}\}", "", c)
    c = _LINK_RE.sub(lambda m: m.group(0)[2:-2].split("|")[-1], c)
    c = re.sub(r"'{2,}", "", c)
    c = re.sub(r"<[^>]+>", "", c)
    return " ".join(c.split())


def _links(c: str) -> list[str]:
    return [unquote(m.group(1)).replace("_", " ").strip() for m in _LINK_RE.finditer(c)]


def article_images(title: str) -> list[dict]:
    """Images used in an article, with captions and the article links inside each caption.

    Uses the REST media-list endpoint (one light request, rate-limited separately from the
    action API); falls back to parsing wikitext.
    """
    try:
        return _media_list(title)
    except Exception:  # noqa: BLE001
        return _article_images_wikitext(title)


def _media_list(title: str) -> list[dict]:
    from urllib.parse import quote
    url = "https://en.wikipedia.org/api/rest_v1/page/media-list/" + quote(title.replace(" ", "_"), safe="")
    d = cached_json("ml:" + title, lambda: http_get(url).json())
    out, seen = [], set()
    for it in d.get("items", []):
        if it.get("type") != "image" or not it.get("title", "").startswith("File:"):
            continue
        f = it["title"].replace("_", " ")
        if f in seen:
            continue
        seen.add(f)
        cap = it.get("caption") or {}
        h = cap.get("html", "")
        links = [unquote(m).replace("_", " ") for m in re.findall(r'href="\./([^"#?]+)', h)]
        links = [l for l in links if ":" not in l]
        out.append({"file": f, "caption": " ".join(html.unescape(cap.get("text", "")).split()), "links": links})
    return out


def _article_images_wikitext(title: str) -> list[dict]:
    d = _api(action="parse", page=title, prop="wikitext", redirects=1)
    if "parse" not in d:
        return []
    wt = d["parse"]["wikitext"]
    out = []
    # [[File:...|...|caption]]
    for m in _FILE_RE.finditer(wt):
        end = _balanced(wt, m.start())
        inner = wt[m.start() + 2:end - 2]
        parts = _split_top(inner)
        fname = parts[0].split(":", 1)[1].strip()
        cap = parts[-1] if len(parts) > 1 else ""
        out.append({"file": "File:" + fname, "raw": cap})
    # <gallery> lines: File.jpg|caption
    for g in re.finditer(r"<gallery[^>]*>(.*?)</gallery>", wt, re.S | re.I):
        for line in g.group(1).splitlines():
            if "|" in line or line.strip().lower().endswith(_IMG_EXT):
                f, _, cap = line.partition("|")
                f = re.sub(r"^(File|Image):", "", f.strip(), flags=re.I)
                if f.lower().endswith(_IMG_EXT):
                    out.append({"file": "File:" + f, "raw": cap})
    # infobox / template params: image = X ... caption = Y
    for im in re.finditer(r"\|\s*(image\d*|photo)\s*=\s*([^|\n]+)", wt, re.I):
        f = re.sub(r"^(File|Image):", "", im.group(2).strip(), flags=re.I)
        if not f.lower().endswith(_IMG_EXT):
            continue
        cm = re.search(r"\|\s*(caption\d*|image_caption|alt)\s*=\s*([^\n]+)", wt[im.end():im.end() + 800], re.I)
        out.append({"file": "File:" + f, "raw": cm.group(2) if cm else ""})
    res, seen = [], set()
    for o in out:
        f = o["file"].replace("_", " ")
        if f in seen:
            continue
        seen.add(f)
        res.append({"file": f, "caption": _clean_caption(o["raw"]), "links": _links(o["raw"])})
    return res


def search_pages(query: str, limit: int = 10) -> list[str]:
    s = _api(action="query", list="search", srsearch=query, srlimit=str(limit))
    return [h["title"] for h in s.get("query", {}).get("search", [])]
