"""Sound effects and music (Mixkit, free for commercial use, not redistributable): downloaded, never committed.

Registry: data/assets.json.  Files land in models/assets/.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .common import DATA, MODELS, download

SR = 24000
DIR = MODELS / "assets"


def registry() -> dict:
    return json.loads((DATA / "assets.json").read_text())


def ensure() -> None:
    reg = registry()
    for name, s in reg["sfx"].items():
        download(s["url"], DIR / f"sfx_{name}.mp3")
    for m in reg["music"]:
        download(m["url"], DIR / f"music_{m['id']}.mp3")


def _decode(path, sr=SR) -> np.ndarray:
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).copy()


def sfx(name: str, gain: float = 1.0, max_len: float = 1.5) -> np.ndarray:
    p = DIR / f"sfx_{name}.mp3"
    if not p.exists():
        ensure()
    a = _decode(p)[: int(max_len * SR)]
    fo = min(len(a), int(0.15 * SR))
    a[-fo:] *= np.linspace(1, 0, fo)
    return a / (np.abs(a).max() + 1e-9) * gain


LOCAL_MUSIC = DATA.parent / "assets" / "music"   # creator-supplied tracks (e.g. a trending song), used first


def pick_track(slug: str, override: str | None = None) -> dict:
    local = sorted(p for p in LOCAL_MUSIC.glob("*") if p.suffix.lower() in (".mp3", ".m4a", ".wav", ".ogg"))
    if local and not override:
        p = local[int(hashlib.sha1(slug.encode()).hexdigest(), 16) % len(local)]
        return {"id": "local", "title": p.stem, "path": str(p)}
    tracks = registry()["music"]
    if override:
        return next(t for t in tracks if t["id"] == override or t["title"] == override)
    h = int(hashlib.sha1(slug.encode()).hexdigest(), 16)
    return tracks[h % len(tracks)]


def music(track: dict, seconds: float) -> np.ndarray:
    """The most energetic `seconds` of the track (skips quiet intros), with short fades."""
    p = Path(track["path"]) if track.get("path") else DIR / f"music_{track['id']}.mp3"
    if not p.exists():
        ensure()
    a = _decode(p)
    n = int(seconds * SR)
    if len(a) <= n:
        a = np.tile(a, int(np.ceil(n / len(a))) + 1)
    hop = SR // 2
    e = np.sqrt(np.convolve(a ** 2, np.ones(hop) / hop, mode="valid")[::hop])  # RMS per 0.5 s
    win = max(1, n // hop)
    limit = max(1, len(e) - win)
    scores = [e[i:i + win].mean() for i in range(0, limit)]
    start = int(np.argmax(scores[: max(1, int(len(scores) * 0.7))])) * hop  # stay out of the outro
    seg = a[start:start + n].copy()
    fi, fo = int(0.05 * SR), int(0.8 * SR)
    seg[:fi] *= np.linspace(0, 1, fi)
    seg[-fo:] *= np.linspace(1, 0, fo)
    rms = np.sqrt(np.mean(seg ** 2)) + 1e-9
    return seg / rms * 0.08  # bed level; ducked further under the voice in mix()
