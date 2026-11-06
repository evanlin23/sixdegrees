"""Competitor research: every upload of a channel with views/likes/comments, via the YouTube Data API.

    python -m sixdeg research @6degreesofseperation   -> research/<handle>.json (+ a table on stdout)
Used to learn which pairs go viral; runs in research.yml with the channel's OAuth token (youtube.readonly).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .common import ROOT


def _secs(iso: str) -> int:
    m = re.fullmatch(r"P(?:\d+D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    return 0 if not m else int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + int(m.group(3) or 0)


def channel_report(handle: str) -> dict:
    from .upload import youtube
    yt = youtube()
    ch = yt.channels().list(part="snippet,statistics,contentDetails", forHandle=handle.lstrip("@")).execute()["items"][0]
    uploads = ch["contentDetails"]["relatedPlaylists"]["uploads"]
    ids, tok = [], None
    while True:
        r = yt.playlistItems().list(part="contentDetails", playlistId=uploads, maxResults=50, pageToken=tok).execute()
        ids += [it["contentDetails"]["videoId"] for it in r["items"]]
        tok = r.get("nextPageToken")
        if not tok:
            break
    vids = []
    for i in range(0, len(ids), 50):
        r = yt.videos().list(part="snippet,statistics,contentDetails", id=",".join(ids[i:i + 50])).execute()
        for v in r["items"]:
            st = v.get("statistics", {})
            vids.append({"id": v["id"], "title": v["snippet"]["title"], "published": v["snippet"]["publishedAt"][:10],
                         "seconds": _secs(v["contentDetails"]["duration"]), "views": int(st.get("viewCount", 0)),
                         "likes": int(st.get("likeCount", 0)), "comments": int(st.get("commentCount", 0)),
                         "description": v["snippet"].get("description", "")[:300]})
    vids.sort(key=lambda x: -x["views"])
    rep = {"handle": handle, "title": ch["snippet"]["title"], "subscribers": int(ch["statistics"].get("subscriberCount", 0)),
           "total_views": int(ch["statistics"].get("viewCount", 0)), "videos": vids}
    out = ROOT / "research" / f"{handle.lstrip('@')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"{rep['title']}: {rep['subscribers']} subs, {rep['total_views']} views, {len(vids)} videos")
    for v in vids:
        print(f"{v['views']:>10,}  {v['likes']:>8,}  {v['comments']:>6,}  {v['seconds']:>3}s  {v['published']}  {v['title']}")
    return rep
