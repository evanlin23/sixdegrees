"""Find people on Wikipedia and photos of them on Wikimedia Commons / image search APIs."""
from __future__ import annotations

import io
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .util import cached_get_json, http, list_images, log, read_json, short_hash, slugify, write_json

WIKI_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
MIN_IMAGE_SIDE = 250
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
_meta_lock = threading.Lock()


@dataclass
class Person:
    name: str  # what we were asked for
    title: str  # English Wikipedia title, used for display and searches
    qid: Optional[str] = None  # Wikidata id, used for Commons "depicts" searches
    image_url: Optional[str] = None  # Wikipedia lead image (a reliable reference portrait)
    description: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def resolve_person(name: str) -> Person:
    """Map a name to its Wikipedia article (follows redirects, skips disambiguation pages)."""
    try:
        page = _wiki_page(name)
        if page is None or page.get("missing") or "disambiguation" in page.get("pageprops", {}):
            data = cached_get_json(WIKI_API, {"action": "query", "format": "json", "formatversion": 2,
                                              "list": "search", "srsearch": name, "srlimit": 5}, "wiki")
            for hit in data.get("query", {}).get("search", []):
                candidate = _wiki_page(hit["title"])
                if candidate and not candidate.get("missing") and "disambiguation" not in candidate.get("pageprops", {}):
                    page = candidate
                    break
        if page and not page.get("missing"):
            return Person(name, page["title"], page.get("pageprops", {}).get("wikibase_item"),
                          (page.get("thumbnail") or {}).get("source"), page.get("description", ""))
    except Exception as e:
        log.warning("  Wikipedia lookup failed for %r: %s", name, e)
    return Person(name, name)


def _wiki_page(title: str) -> Optional[dict]:
    data = cached_get_json(WIKI_API, {
        "action": "query", "format": "json", "formatversion": 2, "redirects": 1, "titles": title,
        "prop": "pageimages|pageprops|description", "piprop": "thumbnail", "pithumbsize": 1280,
        "ppprop": "wikibase_item|disambiguation"}, "wiki")
    pages = data.get("query", {}).get("pages", [])
    return pages[0] if pages else None


@dataclass
class Hit:
    url: str
    source: str
    title: str = ""
    page: str = ""


class CommonsSource:
    """Wikimedia Commons: free, no API key, and many photos are tagged with who they depict."""

    name = "commons"

    def search(self, people: list[Person], query: Optional[str], limit: int) -> list[Hit]:
        hits: list[Hit] = []
        qids = [p.qid for p in people if p.qid]
        if qids and len(qids) == len(people):
            hits += self._search(" ".join(f"haswbstatement:P180={q}" for q in qids), limit)
        names = " ".join(f'"{p.title}"' for p in people)
        hits += self._search(f"{names} filetype:bitmap", limit)
        if query and len(hits) < limit:
            hits += self._search(f"{query} filetype:bitmap", limit)
        return hits

    def _search(self, srsearch: str, limit: int) -> list[Hit]:
        try:
            data = cached_get_json(COMMONS_API, {
                "action": "query", "format": "json", "formatversion": 2, "generator": "search",
                "gsrnamespace": 6, "gsrsearch": srsearch, "gsrlimit": min(50, limit),
                "prop": "imageinfo", "iiprop": "url|size|mime", "iiurlwidth": 1280}, "commons")
        except Exception as e:
            log.warning("  Commons search failed (%s): %s", srsearch, e)
            return []
        return parse_commons_response(data)


def parse_commons_response(data: dict) -> list[Hit]:
    pages = sorted(data.get("query", {}).get("pages", []), key=lambda p: p.get("index", 0))
    hits = []
    for page in pages:
        for info in page.get("imageinfo", [])[:1]:
            if info.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
                continue
            if min(info.get("width", 0), info.get("height", 0)) < MIN_IMAGE_SIDE:
                continue
            hits.append(Hit(info.get("thumburl") or info["url"], "commons", page.get("title", ""),
                            info.get("descriptionurl", "")))
    return hits


class BraveSource:
    """Brave Search image API (https://brave.com/search/api/). Set BRAVE_API_KEY."""

    name = "brave"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, people: list[Person], query: Optional[str], limit: int) -> list[Hit]:
        q = query or " ".join(f'"{p.title}"' for p in people)
        try:
            resp = http().get("https://api.search.brave.com/res/v1/images/search",
                              params={"q": q, "count": min(limit, 50), "safesearch": "strict"},
                              headers={"Accept": "application/json", "X-Subscription-Token": self.api_key}, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            log.warning("  Brave image search failed: %s", e)
            return []
        return [Hit(r["properties"]["url"], "brave", r.get("title", ""), r.get("url", ""))
                for r in resp.json().get("results", []) if r.get("properties", {}).get("url")]


class SerpApiSource:
    """Google Images via SerpApi (https://serpapi.com). Set SERPAPI_API_KEY."""

    name = "serpapi"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, people: list[Person], query: Optional[str], limit: int) -> list[Hit]:
        q = query or " ".join(f'"{p.title}"' for p in people)
        try:
            resp = http().get("https://serpapi.com/search.json",
                              params={"engine": "google_images", "q": q, "api_key": self.api_key}, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            log.warning("  SerpApi search failed: %s", e)
            return []
        return [Hit(r["original"], "serpapi", r.get("title", ""), r.get("link", ""))
                for r in resp.json().get("images_results", [])[:limit] if r.get("original")]


class BingCrawlerSource:
    """Bing Images through icrawler (scrapes the results page; may break if Bing changes it)."""

    name = "bing"

    def fetch(self, people: list[Person], query: Optional[str], dest: Path, limit: int) -> list[Path]:
        from icrawler.builtin import BingImageCrawler

        q = query or " ".join(f'"{p.title}"' for p in people)
        tmp = Path(tempfile.mkdtemp(prefix="bing_", dir=dest))
        try:
            crawler = BingImageCrawler(downloader_threads=4, storage={"root_dir": str(tmp)}, log_level=40)
            crawler.crawl(keyword=q, max_num=limit, min_size=(MIN_IMAGE_SIDE, MIN_IMAGE_SIDE))
            out = []
            for f in list_images(tmp):
                target = dest / f"bing_{short_hash(q + f.name, 10)}{f.suffix.lower()}"
                shutil.move(str(f), target)
                out.append(target)
            return out
        except Exception as e:
            log.warning("  Bing crawl failed: %s", e)
            return []
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def make_sources(spec: str) -> list:
    """`spec` is a comma list of commons,brave,serpapi,bing, "auto" (everything usable) or "none"."""
    names = [s.strip().lower() for s in spec.split(",") if s.strip() and s.strip().lower() != "none"]
    auto = "auto" in names
    sources = []
    if auto or "commons" in names:
        sources.append(CommonsSource())
    for name, env, cls in (("brave", "BRAVE_API_KEY", BraveSource), ("serpapi", "SERPAPI_API_KEY", SerpApiSource)):
        key = os.environ.get(env)
        if key and (auto or name in names):
            sources.append(cls(key))
        elif name in names:
            log.warning("Image source %r needs %s; skipping it.", name, env)
    if auto or "bing" in names:
        try:
            import icrawler  # noqa: F401

            sources.append(BingCrawlerSource())
        except ImportError:
            if "bing" in names:
                log.warning("Image source 'bing' needs `pip install icrawler`; skipping it.")
    return sources


def download(hit: Hit, dest: Path) -> Optional[Path]:
    """Download one image, reject tiny/broken files, and remember where it came from."""
    from PIL import Image

    stem = f"{hit.source}_{short_hash(hit.url, 10)}"
    for existing in dest.glob(stem + ".*"):
        return existing
    try:
        resp = http().get(hit.url, timeout=25, stream=True)
        resp.raise_for_status()
        data = io.BytesIO()
        for chunk in resp.iter_content(1 << 16):
            data.write(chunk)
            if data.tell() > MAX_DOWNLOAD_BYTES:
                return None
        with Image.open(io.BytesIO(data.getvalue())) as im:
            im.load()
            if min(im.size) < MIN_IMAGE_SIDE:
                return None
            fmt = (im.format or "").upper()
            if fmt in ("JPEG", "PNG", "WEBP"):
                path = dest / f"{stem}.{'jpg' if fmt == 'JPEG' else fmt.lower()}"
                path.write_bytes(data.getvalue())
            else:
                path = dest / f"{stem}.jpg"
                im.convert("RGB").save(path, quality=92)
    except Exception as e:
        log.debug("  download failed %s: %s", hit.url, e)
        return None
    with _meta_lock:
        meta_path = dest / "sources.json"
        meta = read_json(meta_path, {})
        meta[path.name] = {"url": hit.url, "source": hit.source, "title": hit.title, "page": hit.page}
        write_json(meta_path, meta)
    return path


def collect_images(people: list[Person], query: Optional[str], dest: Path, sources: list, limit: int) -> list[Path]:
    """Fill `dest` with up to `limit` candidate photos (files already there, e.g. added by hand, count)."""
    dest.mkdir(parents=True, exist_ok=True)
    have = list_images(dest)
    if len(have) >= limit:
        return have
    queries = [None] + ([query] if query else [])
    for source in sources:
        for q in queries:
            need = limit - len(list_images(dest))
            if need <= 0:
                return list_images(dest)
            if hasattr(source, "fetch"):
                source.fetch(people, q, dest, need)
                continue
            hits = source.search(people, q, need * 2)
            seen = set()
            hits = [h for h in hits if not (h.url in seen or seen.add(h.url))][: need + max(2, need // 2)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda h: download(h, dest), hits))
    return list_images(dest)


def reference_images(person: Person, dest: Path, sources: list, limit: int = 6) -> tuple[list[Path], Optional[Path]]:
    """Photos of one person (for face recognition). Returns (images, anchor) where the anchor is
    the Wikipedia lead image, if any. Drop your own photos into `dest` to help a hard case."""
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "person.json", person.to_dict())
    anchor = next(iter(dest.glob("wiki_lead.*")), None)
    if anchor is None and person.image_url:
        got = download(Hit(person.image_url, "wiki", person.title), dest)
        if got:
            anchor = got.rename(dest / f"wiki_lead{got.suffix}")
    if len(list_images(dest)) < limit:
        collect_images([person], f"{person.title} face", dest, sources, limit)
    return list_images(dest), anchor


def link_dir_name(a: str, b: str) -> str:
    return f"{slugify(a)}__{slugify(b)}"
