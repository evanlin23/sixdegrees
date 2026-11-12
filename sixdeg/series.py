"""Hitler series extras (owner, 2026-11-12: "only the AH ones are blowing up"):

- `python -m sixdeg series`   every uploaded Hitler Short joins the public playlist SERIES_TITLE (created once)
- `python -m sixdeg comments` top comments on the Hitler Shorts -> comments.json, so the producer can make the
  "Hitler to X" pairs viewers ask for first (their comment becomes the hook)
Both run after each upload in upload.yml; quota cost is tiny (list = 1 unit, playlistItems.insert = 50).
"""
from __future__ import annotations

import json
import re

from .common import ROOT

SERIES_TITLE = "How is Adolf Hitler connected to...?"
SERIES_DESC = "Every Buh Connects chain that starts at Adolf Hitler. Real, sourced photos only. Comment who's next."


def _hitler_uploads() -> list[dict]:
    import subprocess
    raw = subprocess.check_output(["git", "-C", str(ROOT), "show", "origin/queue:uploaded.json"], text=True)
    return [u for u in json.loads(raw) if u.get("video_id") and "hitler" in (u.get("slug") or "").lower()]


def _playlist(yt) -> str:
    r = yt.playlists().list(part="snippet", mine=True, maxResults=50).execute()
    for p in r.get("items", []):
        if p["snippet"]["title"] == SERIES_TITLE:
            return p["id"]
    p = yt.playlists().insert(part="snippet,status", body={
        "snippet": {"title": SERIES_TITLE, "description": SERIES_DESC},
        "status": {"privacyStatus": "public"}}).execute()
    return p["id"]


def series() -> str:
    from .upload import youtube
    yt = youtube()
    pl = _playlist(yt)
    have, tok = set(), None
    while True:
        r = yt.playlistItems().list(part="contentDetails", playlistId=pl, maxResults=50, pageToken=tok).execute()
        have |= {it["contentDetails"]["videoId"] for it in r.get("items", [])}
        tok = r.get("nextPageToken")
        if not tok:
            break
    added = []
    for u in sorted(_hitler_uploads(), key=lambda u: u.get("publish_at") or ""):
        if u["video_id"] not in have:
            yt.playlistItems().insert(part="snippet", body={"snippet": {"playlistId": pl, "resourceId": {
                "kind": "youtube#video", "videoId": u["video_id"]}}}).execute()
            added.append(u["slug"])
    return f"playlist {pl}: added {len(added)} ({', '.join(added) or 'none'}), {len(have) + len(added)} total"


def comments(limit: int = 100) -> dict:
    from .upload import youtube
    yt = youtube()
    out = {}
    for u in _hitler_uploads():
        try:
            r = yt.commentThreads().list(part="snippet", videoId=u["video_id"], maxResults=limit, order="relevance",
                                         textFormat="plainText").execute()
        except Exception as e:  # noqa: BLE001 - comments off / not public yet
            out[u["slug"]] = {"error": str(e)[:120]}
            continue
        rows = []
        for it in r.get("items", []):
            s = it["snippet"]["topLevelComment"]["snippet"]
            rows.append({"author": s.get("authorDisplayName", ""), "text": s.get("textDisplay", "")[:300],
                         "likes": s.get("likeCount", 0)})
        out[u["slug"]] = {"video_id": u["video_id"], "comments": rows}
    (ROOT / "comments.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    n = sum(len(v.get("comments", [])) for v in out.values())
    asks = sum(1 for v in out.values() for c in v.get("comments", [])
               if re.search(r"(?i)\b(do|next|connect|to|vs)\b", c["text"]))
    return {"videos": len(out), "comments": n, "maybe_requests": asks}
