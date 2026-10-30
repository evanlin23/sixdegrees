"""Small shared helpers: paths, logging, HTTP, JSON, image loading."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger("sixdegrees")

USER_AGENT = "sixdegrees/0.2 (https://github.com/evanlin23/sixdegrees) python-requests"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def cache_dir() -> Path:
    """Where downloaded models and API responses are cached."""
    root = os.environ.get("SIXDEGREES_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "sixdegrees")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dotenv_if_present() -> None:
    """Load a .env from the working directory or any parent (optional dependency)."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found)


def slugify(text: str, max_len: int = 60) -> str:
    """Filesystem-safe name: 'Pope Francis' -> 'Pope_Francis'."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9.\-]+", "_", text).strip("._")
    return (text or "unknown")[:max_len]


def same_person(a: str, b: str) -> bool:
    return normalize_name(a) == normalize_name(b)


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", name).strip()


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def file_hash(path: Path, n: int = 16) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(data, indent=2, ensure_ascii=False)
    # Keep short numeric arrays (boxes, points, embeddings) on one line so the file stays hand-editable.
    text = re.sub(r"\[\s+([-\d.eE,\s]+?)\s+\]", lambda m: "[" + ", ".join(v.strip() for v in m.group(1).split(",")) + "]", text)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))


def load_image(path: Path, max_side: Optional[int] = None) -> np.ndarray:
    """Load any image as a BGR uint8 array, applying EXIF rotation (phone photos)."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (0, 0, 0))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        if max_side and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.LANCZOS)
        rgb = np.asarray(im)
    return np.ascontiguousarray(rgb[:, :, ::-1])


_session = None


def http():
    """Shared requests session with a descriptive User-Agent and polite retries."""
    global _session
    if _session is None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        s = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
        retry = Retry(total=3, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=("GET", "HEAD"), respect_retry_after_header=True)
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
        s.mount("http://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
        _session = s
    return _session


def cached_get_json(url: str, params: dict, namespace: str, ttl_days: float = 30) -> Any:
    """GET a JSON API with an on-disk cache so reruns don't hammer public APIs."""
    import time

    key = short_hash(url + json.dumps(params, sort_keys=True), 20)
    path = cache_dir() / "http" / namespace / f"{key}.json"
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl_days * 86400:
        data = read_json(path)
        if data is not None:
            return data
    resp = http().get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    write_json(path, data)
    return data
