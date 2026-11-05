"""Shared paths, HTTP with polite backoff, and a tiny on-disk cache."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
CACHE = ROOT / ".cache"
EPISODES = ROOT / "episodes"
DATA = ROOT / "data"
FONTS = ROOT / "assets" / "fonts"

# Wikimedia asks for a descriptive UA with a contact URL.
UA = "sixdegrees-bot/0.2 (https://github.com/evanlin23/sixdegrees)"

_session = requests.Session()
_session.headers["User-Agent"] = UA


def http_get(url: str, *, params=None, stream=False, timeout=60, tries=6, headers=None) -> requests.Response:
    """GET with exponential backoff on 429/5xx (Wikimedia rate-limits shared cloud IPs)."""
    delay = 2.0
    last = None
    for _ in range(tries):
        try:
            r = _session.get(url, params=params, stream=stream, timeout=timeout, headers=headers)
        except requests.RequestException as e:  # network blip
            last = e
        else:
            if r.status_code < 400:
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {r.url}")
            if r.status_code not in (429, 500, 502, 503, 504):
                break
            ra = r.headers.get("retry-after")
            if ra and ra.isdigit():
                if float(ra) > 300:  # long ban: fail fast so the caller can pick another source
                    raise RuntimeError(f"HTTP {r.status_code} retry-after {ra}s for {r.url}")
                delay = max(delay, float(ra))
        time.sleep(delay + random.random())
        delay = min(delay * 2, 60)
    raise RuntimeError(f"GET failed: {url} {params or ''}: {last}")


def cached_json(key: str, fn, max_age_days: float = 30):
    """Memoise fn() as JSON under .cache/json/<hash>.json."""
    d = CACHE / "json"
    d.mkdir(parents=True, exist_ok=True)
    p = d / (hashlib.sha1(key.encode()).hexdigest() + ".json")
    if p.exists() and (time.time() - p.stat().st_mtime) < max_age_days * 86400:
        return json.loads(p.read_text())
    val = fn()
    p.write_text(json.dumps(val))
    return val


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = http_get(url, stream=True, timeout=300)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    os.replace(tmp, dest)
    return dest


def slugify(s: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "-" for c in s)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")
