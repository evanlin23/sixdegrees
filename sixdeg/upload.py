"""Upload a finished Short to YouTube as PRIVATE with a scheduled publish time.

YouTube is the backlog: each video goes up scheduled into the next free slot
(two a day), so nothing has to survive between cloud containers. Slots already
taken are read from data/uploads.json (committed after every upload).

Credentials come from environment variables (set them in the Claude cloud
environment's settings, never in the repo):
  YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN   OAuth client + refresh token (docs/youtube-setup.md)
  YT_CHANNEL_ID                                       UC... id; refuses to upload anywhere else

    python -m sixdeg upload <slug> [--dry-run]
"""
from __future__ import annotations

import datetime as dt
import json
import os
import random
from zoneinfo import ZoneInfo

from .common import DATA

TZ = ZoneInfo("America/Chicago")
SLOTS = [(11, 30), (18, 30)]          # Central time; each gets 0-20 min jitter so it never looks like clockwork
UPLOADS = DATA / "uploads.json"


def _uploads() -> list[dict]:
    return json.loads(UPLOADS.read_text()) if UPLOADS.exists() else []


def next_slot(now: dt.datetime | None = None, taken: list[str] | None = None) -> dt.datetime:
    """The next free publish slot at least 45 min from now, as an aware datetime in UTC."""
    now = (now or dt.datetime.now(dt.timezone.utc)).astimezone(TZ)
    taken_days = {}
    for t in taken if taken is not None else [u["publish_at"] for u in _uploads() if u.get("publish_at")]:
        x = dt.datetime.fromisoformat(t.replace("Z", "+00:00")).astimezone(TZ)
        taken_days.setdefault(x.date(), []).append(x)
    day = now.date()
    for _ in range(60):
        for i, (h, m) in enumerate(SLOTS):
            at = dt.datetime(day.year, day.month, day.day, h, m, tzinfo=TZ)
            if at < now + dt.timedelta(minutes=45):
                continue
            # slot i is taken if a scheduled upload already sits within 2h of it
            if any(abs((x - at).total_seconds()) < 7200 for x in taken_days.get(day, [])):
                continue
            return (at + dt.timedelta(minutes=random.randint(0, 20))).astimezone(dt.timezone.utc)
        day += dt.timedelta(days=1)
    raise RuntimeError("no free slot in 60 days")


def scheduled_on_channel(yt) -> list[str]:
    """publishAt of every private video on the channel that is still waiting to go public."""
    ch = yt.channels().list(part="contentDetails", mine=True).execute()["items"][0]
    pl = ch["contentDetails"]["relatedPlaylists"]["uploads"]
    items = yt.playlistItems().list(part="contentDetails", playlistId=pl, maxResults=50).execute().get("items", [])
    ids = [it["contentDetails"]["videoId"] for it in items]
    out = []
    for i in range(0, len(ids), 50):
        vs = yt.videos().list(part="status", id=",".join(ids[i:i + 50])).execute().get("items", [])
        out += [v["status"]["publishAt"] for v in vs if v["status"].get("publishAt")
                and v["status"].get("privacyStatus") == "private"]
    return out


def have_credentials() -> bool:
    return all(os.environ.get(k) for k in ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN", "YT_CHANNEL_ID"))


def youtube():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    creds = Credentials(None, refresh_token=os.environ["YT_REFRESH_TOKEN"], client_id=os.environ["YT_CLIENT_ID"],
                        client_secret=os.environ["YT_CLIENT_SECRET"], token_uri="https://oauth2.googleapis.com/token")
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def _already_on_channel(yt, title: str) -> str | None:
    ch = yt.channels().list(part="contentDetails", mine=True).execute().get("items", [])
    if not ch:
        return None
    pl = ch[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    items = yt.playlistItems().list(part="snippet", playlistId=pl, maxResults=50).execute().get("items", [])
    for it in items:
        if it["snippet"]["title"].strip() == title.strip():
            return it["snippet"]["resourceId"]["videoId"]
    return None


def upload(slug: str, dry_run: bool = False) -> dict:
    """Upload an episode straight from episodes/<slug>/out (the queue normally does this instead)."""
    from .episode import load  # heavy imports (cv2...) only for local use; the Actions consumer doesn't need them
    d, spec = load(slug)
    out = d / "out"
    qa = json.loads((out / "qa.json").read_text()) if (out / "qa.json").exists() else {}
    if not qa.get("pass"):
        raise SystemExit(f"{slug}: QA has not passed; refusing to upload")
    meta = json.loads((out / "metadata.json").read_text())
    meta["slug"] = slug
    if any(u["slug"] == slug for u in _uploads()):
        raise SystemExit(f"{slug}: already in data/uploads.json")
    rec = upload_file(out / meta["file"], meta, dry_run=dry_run)
    if not rec.get("dry_run"):
        ups = _uploads(); ups.append(rec)
        UPLOADS.parent.mkdir(exist_ok=True)
        UPLOADS.write_text(json.dumps(ups, indent=1))
    return rec


def upload_file(mp4, meta: dict, dry_run: bool = False, taken_extra: list[str] | None = None,
                thumb=None) -> dict:
    """Upload one video as private with publishAt = the next free slot; returns the record."""
    slug = meta.get("slug", "")
    if dry_run or not have_credentials():
        rec = {"slug": slug, "title": meta["title"], "dry_run": True}
        if have_credentials():  # dry run with credentials: prove the token works and points at the right channel
            yt = youtube()
            me = yt.channels().list(part="id,snippet", mine=True).execute()["items"][0]
            rec["channel"] = f"{me['snippet']['title']} ({me['id']})"
            rec["channel_ok"] = me["id"] == os.environ["YT_CHANNEL_ID"]
            when = next_slot(taken=scheduled_on_channel(yt) + list(taken_extra or []))
        else:
            rec["note"] = "no YT_* credentials in the environment"
            when = next_slot(taken=list(taken_extra or []))
        rec["publish_at"] = when.strftime("%Y-%m-%dT%H:%M:%SZ")
        return rec

    from googleapiclient.http import MediaFileUpload
    yt = youtube()
    me = yt.channels().list(part="id,snippet", mine=True).execute()["items"][0]
    if me["id"] != os.environ["YT_CHANNEL_ID"]:
        raise SystemExit(f"token belongs to channel {me['id']} ({me['snippet']['title']}), not YT_CHANNEL_ID")
    # the channel itself is the source of truth for which slots are taken (no shared state needed)
    when = next_slot(taken=scheduled_on_channel(yt) + list(taken_extra or []))
    rec = {"slug": slug, "title": meta["title"], "publish_at": when.strftime("%Y-%m-%dT%H:%M:%SZ")}
    dup = _already_on_channel(yt, meta["title"])
    if dup:
        rec.update(video_id=dup, note="already on channel; recorded, not re-uploaded")
    else:
        privacy = os.environ.get("YT_PRIVACY", "scheduled")
        status = {"privacyStatus": "private", "selfDeclaredMadeForKids": False,
                  "containsSyntheticMedia": meta.get("containsSyntheticMedia", False)}
        if privacy == "scheduled":
            status["publishAt"] = rec["publish_at"]
        elif privacy == "public":
            status["privacyStatus"] = "public"
        body = {"snippet": {"title": meta["title"], "description": meta["description"], "tags": meta["tags"],
                            "categoryId": meta.get("categoryId", "27"), "defaultLanguage": "en",
                            "defaultAudioLanguage": "en"},
                "status": status}
        req = yt.videos().insert(part="snippet,status", body=body,
                                 media_body=MediaFileUpload(str(mp4), mimetype="video/mp4", chunksize=8 << 20,
                                                            resumable=True))
        resp = None
        while resp is None:
            _, resp = req.next_chunk()
        rec["video_id"] = resp["id"]
        rec["privacy"] = privacy
    rec["url"] = f"https://youtube.com/shorts/{rec['video_id']}"
    if thumb and not dup:
        rec["thumbnail"] = set_thumbnail(yt, rec["video_id"], thumb)
    if privacy == "public" and meta.get("first_comment") and not dup:
        rec["comment"] = post_comment(yt, rec["video_id"], meta["first_comment"])
    return rec


def set_thumbnail(yt, video_id: str, path) -> str:
    """Custom thumbnail (needs a phone-verified channel; failure is reported, not fatal)."""
    from googleapiclient.http import MediaFileUpload
    try:
        yt.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(str(path), mimetype="image/jpeg")).execute()
        return "set"
    except Exception as e:  # noqa: BLE001
        return f"not set: {str(e)[:160]}"


def post_comment(yt, video_id: str, text: str) -> str:
    """Channel's own first comment (only possible once the video is public; pinning has no API)."""
    try:
        r = yt.commentThreads().insert(part="snippet", body={"snippet": {"videoId": video_id, "topLevelComment": {
            "snippet": {"textOriginal": text}}}}).execute()
        return "posted " + r["id"]
    except Exception as e:  # noqa: BLE001
        return f"not posted: {str(e)[:160]}"


def comment_if_public(yt, rec: dict, text: str) -> str | None:
    """For scheduled uploads: post the first comment on a later run, once the video has gone public."""
    v = yt.videos().list(part="status", id=rec["video_id"]).execute().get("items", [])
    if v and v[0]["status"]["privacyStatus"] == "public":
        return post_comment(yt, rec["video_id"], text)
    return None
