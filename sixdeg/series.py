"""Hitler series extras (owner, 2026-11-12: "only the AH ones are blowing up"):

- `python -m sixdeg series`   every uploaded Hitler Short joins the public playlist SERIES_TITLE (created once)
- `python -m sixdeg comments` top comments and their replies on every upload -> comments.json (likely
  requests flagged "request": true, each with the @handle and a 240 px profile picture), so producers make what viewers ask for first
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


# "You could've ended it at Nixon. Trump & Nixon met." Viewers point out shorter routes: producers check these first.
SHORTCUT_RE = re.compile(r"(?i)(could\s*'?ve|could have|shorter|faster|quicker|easier|skip|instead|directly|\bmet\b|ended it at|just go|method|unpatched|patched)")
REQUEST_RE = re.compile(r"(?i)(\bto\b|→|->|\bconnect|\bnext\b|\bvs\.?\b|\bdo\b|\bplease\b|\bpls\b|\bcan you\b)")


def _uploads() -> list[dict]:
    """Every upload with a video id (Hitler Shorts and the rest: requests come in on all of them)."""
    import subprocess
    raw = subprocess.check_output(["git", "-C", str(ROOT), "show", "origin/queue:uploaded.json"], text=True)
    return [u for u in json.loads(raw) if u.get("video_id")]


def _big_avatar(url: str) -> str:
    """YouTube serves the commenter's picture at any size: ask for 240 px instead of the 48 px thumbnail."""
    return re.sub(r"=s\d+-", "=s240-", url or "")


def _row(s: dict, cid: str, parent: dict | None = None) -> dict:
    return {"author": s.get("authorDisplayName", ""),            # the @handle (requested_by)
            "author_channel": (s.get("authorChannelId") or {}).get("value", ""),
            "text": s.get("textDisplay", "")[:300], "likes": s.get("likeCount", 0),
            "avatar": _big_avatar(s.get("authorProfileImageUrl", "")),   # request_avatar
            "published": s.get("publishedAt", ""), "comment_id": cid,
            "reply_to": parent["author"] if parent else None}   # replies to our "Who should I connect next?"


def comments(limit: int = 100) -> dict:
    """Top comments AND their replies on every upload -> comments.json. Viewers answer the pinned "Who should I connect
    next?" comment with a reply, so replies are where most requests are. Each row keeps the @handle and a 240 px
    profile-picture URL, so a requested pair can credit the viewer (requested_by, request_avatar)."""
    from .upload import youtube
    yt = youtube()
    out, requests = {}, []
    for u in _uploads():
        try:
            r = yt.commentThreads().list(part="snippet,replies", videoId=u["video_id"], maxResults=limit,
                                         order="relevance", textFormat="plainText").execute()
        except Exception as e:  # noqa: BLE001 - comments off / not public yet
            out[u["slug"]] = {"error": str(e)[:120]}
            continue
        rows = []
        for it in r.get("items", []):
            top = _row(it["snippet"]["topLevelComment"]["snippet"], it["id"])
            rows.append(top)
            replies = (it.get("replies") or {}).get("comments", [])
            if it["snippet"].get("totalReplyCount", 0) > len(replies):     # the thread view returns 5 at most
                try:
                    replies = yt.comments().list(part="snippet", parentId=it["id"], maxResults=100,
                                                 textFormat="plainText").execute().get("items", [])
                except Exception:  # noqa: BLE001 - keep the 5 we have
                    pass
            rows += [_row(c["snippet"], c["id"], top) for c in replies]
        for c in rows:   # flag likely requests in place (upload.yml saves comments.json only)
            ours = c["author"].lower() == "@buhconnects"
            c["request"] = not ours and bool(REQUEST_RE.search(c["text"]))
            c["shortcut"] = not ours and bool(SHORTCUT_RE.search(c["text"]))   # a shorter route, or a "method" meme
        out[u["slug"]] = {"video_id": u["video_id"], "comments": rows}
        requests += [c for c in rows if c["request"]]
    (ROOT / "comments.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    n = sum(len(v.get("comments", [])) for v in out.values())
    return {"videos": len(out), "comments": n, "maybe_requests": len(requests)}
