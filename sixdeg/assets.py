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


# Trending songs for the music-only A/B arm. They live in the PRIVATE repo evanlin23/sixdegrees-music (never in this
# public repo); render.yml checks it out to ./music-private when the MUSIC_REPO_TOKEN secret exists. Optional
# tracks.json there: [{"file": "washing-machine-heart.mp3", "title": "...", "artist": "...", "start": 42.0}], where
# "start" is the second the recognisable part begins. Shorts with a claimed song must stay under 60 s.
TRENDING = Path(__import__("os").environ.get("MUSIC_DIR", str(DATA.parent / "music-private")))


def trending_tracks() -> list[dict]:
    if not TRENDING.is_dir():
        return []
    meta = {}
    if (TRENDING / "tracks.json").exists():
        meta = {t["file"]: t for t in json.loads((TRENDING / "tracks.json").read_text())}
    out = []
    for p in sorted(TRENDING.rglob("*")):
        if p.suffix.lower() in (".mp3", ".m4a", ".wav", ".ogg", ".flac"):
            m = meta.get(p.name, {})
            out.append({"id": "trending", "title": m.get("title", p.stem), "artist": m.get("artist", ""),
                        "path": str(p), "start": m.get("start"), "trending": True})
    return out


def pick_trending(slug: str) -> dict | None:
    ts = trending_tracks()
    return ts[int(hashlib.sha1((slug + ":song").encode()).hexdigest(), 16) % len(ts)] if ts else None


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
    a = track["_audio"] if track.get("_audio") is not None else _decode(p)
    n = int(seconds * SR)
    if track.get("start") is not None:          # a trending song: start where the recognisable part begins
        s0 = int(float(track["start"]) * SR)
        seg = a[s0:s0 + n].copy()
        if len(seg) < n:
            seg = np.pad(seg, (0, n - len(seg)))
        fi, fo = int(0.05 * SR), int(0.8 * SR)
        seg[:fi] *= np.linspace(0, 1, fi)
        seg[-fo:] *= np.linspace(1, 0, fo)
        return seg
    if len(a) <= n:
        a = np.tile(a, int(np.ceil(n / len(a))) + 1)
    hop = SR // 2
    sq = a[: len(a) // hop * hop].astype(np.float64) ** 2
    e = np.sqrt(sq.reshape(-1, hop).mean(axis=1))  # RMS per 0.5 s (block means: O(n), not an O(n*hop) convolution)
    win = max(1, n // hop)
    limit = max(1, len(e) - win)
    cs = np.concatenate([[0.0], np.cumsum(e)])
    scores = (cs[win:win + limit] - cs[:limit]) / win
    # owner: "Albert Einstein has a few seconds of deadspace at the end": the loudest-on-average window can still
    # hold a break in the song. Only accept windows with no near-silent 0.5 s block (< 30 % of the loud level).
    m = len(scores[: max(1, int(len(scores) * 0.7))])               # stay out of the outro
    floor = 0.3 * float(np.percentile(e, 80))       # "quiet" relative to the loud parts, not a half-quiet median
    quiet = (e < floor).astype(np.int64)
    qc = np.concatenate([[0], np.cumsum(quiet)])
    ok = (qc[win:win + m] - qc[:m]) == 0
    if not ok.any() and not track.get("_spliced"):
        # every long-enough window crosses a break in the song: splice the quiet blocks out (short crossfades)
        keep = [a[i * hop:(i + 1) * hop] for i in range(len(e)) if e[i] >= floor]
        if keep:
            xf = int(0.02 * SR)
            out = keep[0].astype(np.float32).copy()
            for blk in keep[1:]:
                blk = blk.astype(np.float32)
                ramp = np.linspace(0, 1, xf, dtype=np.float32)
                out[-xf:] = out[-xf:] * (1 - ramp) + blk[:xf] * ramp
                out = np.concatenate([out, blk[xf:]])
            return music({**track, "_spliced": True, "_audio": out}, seconds)
    cand = np.where(ok, scores[:m], -1.0)
    start = int(np.argmax(cand if ok.any() else scores[:m])) * hop
    seg = a[start:start + n].copy()
    fi, fo = int(0.05 * SR), int(0.8 * SR)
    seg[:fi] *= np.linspace(0, 1, fi)
    seg[-fo:] *= np.linspace(1, 0, fo)
    rms = np.sqrt(np.mean(seg ** 2)) + 1e-9
    return seg / rms * 0.08  # bed level; ducked further under the voice in mix()
