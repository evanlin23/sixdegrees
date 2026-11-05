"""Performance of uploaded Shorts: views, likes, comments and, if the token has the yt-analytics.readonly
scope, average % viewed and average view duration (the numbers that matter most for Shorts).

    python -m sixdeg stats          (needs YT_* credentials: run it from .github/workflows/stats.yml)

Writes a markdown table to stdout (and the Actions step summary) and appends a dated snapshot to
stats.json on the `queue` branch, so later runs and Claude sessions can compare day 1 vs day 7.
"""
from __future__ import annotations

import datetime as dt
import json
import os

from .queue import _push, worktree
from .upload import have_credentials, youtube


def analytics(video_ids: list[str], start: str) -> dict:
    """Retention per video via YouTube Analytics; empty if the token lacks the scope."""
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        creds = Credentials(None, refresh_token=os.environ["YT_REFRESH_TOKEN"], client_id=os.environ["YT_CLIENT_ID"],
                            client_secret=os.environ["YT_CLIENT_SECRET"], token_uri="https://oauth2.googleapis.com/token")
        ya = build("youtubeAnalytics", "v2", credentials=creds, cache_discovery=False)
        r = ya.reports().query(ids="channel==MINE", startDate=start, endDate=dt.date.today().isoformat(),
                               metrics="views,averageViewPercentage,averageViewDuration,likes,subscribersGained",
                               dimensions="video", filters="video==" + ",".join(video_ids)).execute()
        cols = [c["name"] for c in r.get("columnHeaders", [])]
        return {row[0]: dict(zip(cols, row)) for row in r.get("rows", [])}
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)[:200]}


def stats() -> str:
    if not have_credentials():
        raise SystemExit("needs YT_* credentials (run .github/workflows/stats.yml)")
    wt = worktree()
    up = [u for u in json.loads((wt / "uploaded.json").read_text()) if u.get("video_id")]
    if not up:
        return "No uploads yet."
    yt = youtube()
    ids = [u["video_id"] for u in up]
    vids = {}
    for i in range(0, len(ids), 50):
        for v in yt.videos().list(part="statistics,status,snippet", id=",".join(ids[i:i + 50])).execute()["items"]:
            vids[v["id"]] = v
    first = min(u.get("publish_at", "")[:10] for u in up) or dt.date.today().isoformat()
    an = analytics(ids, first)
    rows, snap = [], {"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), "videos": {}}
    for u in up:
        v = vids.get(u["video_id"], {})
        st, status = v.get("statistics", {}), v.get("status", {})
        a = an.get(u["video_id"], {})
        rec = {"slug": u["slug"], "views": int(st.get("viewCount", 0)), "likes": int(st.get("likeCount", 0)),
               "comments": int(st.get("commentCount", 0)), "privacy": status.get("privacyStatus"),
               "avg_view_pct": a.get("averageViewPercentage"), "avg_view_s": a.get("averageViewDuration"),
               "subs_gained": a.get("subscribersGained")}
        snap["videos"][u["video_id"]] = rec
        pct = f"{rec['avg_view_pct']:.0f}%" if isinstance(rec["avg_view_pct"], (int, float)) else "n/a"
        rows.append(f"| {u['title'][:48]} | {rec['privacy']} | {rec['views']} | {rec['likes']} | {rec['comments']} | {pct} | {u['url']} |")
    note = "" if "_error" not in an else f"\n_Retention unavailable (token lacks yt-analytics.readonly?): {an['_error'][:120]}_\n"
    md = "\n".join(["| Short | status | views | likes | comments | avg % viewed | link |",
                    "|---|---|---:|---:|---:|---:|---|", *rows]) + note
    hist_p = wt / "stats.json"
    hist = json.loads(hist_p.read_text()) if hist_p.exists() else []
    hist.append(snap)
    hist_p.write_text(json.dumps(hist, indent=1) + "\n")
    _push("Stats snapshot " + snap["at"])
    out = os.environ.get("GITHUB_STEP_SUMMARY")
    if out:
        with open(out, "a") as f:
            f.write(md + "\n")
    return md
