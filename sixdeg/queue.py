"""Producer/consumer queue on the `queue` branch (same model as blocky-stories).

Producer (a Claude session):   python -m sixdeg enqueue <slug>
    copies out/<slug>.mp4 + out/metadata.json into  queue/<NNNN>-<slug>/  on the `queue` branch and pushes.
Consumer (GitHub Actions):     python -m sixdeg consume [--max 2]
    uploads the oldest items to YouTube (private + scheduled publish), moves them from queue/ to the
    uploaded.json log, pushes. Runs from .github/workflows/upload.yml, started by cron-job.org.

The branch only ever holds what is waiting, so it stays small; it is an orphan branch (no code).
"""
from __future__ import annotations

import json
import re
import os
import shutil
import subprocess
from pathlib import Path

from .common import ROOT, EPISODES

BRANCH = "queue"
VOICE_REV = 2          # keep in step with audio.VOICE_REV (the upload job doesn't import audio)
WT = ROOT / ".queue"          # git worktree of the queue branch


def _git(*a, cwd=ROOT, check=True):
    return subprocess.run(["git", *a], cwd=cwd, check=check, capture_output=True, text=True)


def worktree() -> Path:
    """Fresh worktree of origin/queue (created as an orphan branch the first time)."""
    if WT.exists():
        _git("worktree", "remove", "--force", str(WT), check=False)
        shutil.rmtree(WT, ignore_errors=True)
    _git("worktree", "prune")
    has_remote = _git("ls-remote", "--heads", "origin", BRANCH).stdout.strip() != ""
    if has_remote:
        _git("fetch", "origin", BRANCH)
        _git("worktree", "add", "-B", BRANCH, str(WT), f"origin/{BRANCH}")
    else:
        _git("worktree", "add", "--detach", str(WT))
        _git("checkout", "--orphan", BRANCH, cwd=WT)
        _git("rm", "-rf", "--quiet", ".", cwd=WT)
        (WT / "README.md").write_text("Videos waiting to be uploaded. Producer: `python -m sixdeg enqueue`; "
                                      "consumer: `.github/workflows/upload.yml`.\n")
        (WT / "queue").mkdir()
        (WT / "queue" / ".keep").write_text("")
        (WT / "uploaded.json").write_text("[]\n")
        _git("add", "-A", cwd=WT)
        _git("commit", "-m", "Start the upload queue", cwd=WT)
    return WT


def _push(msg: str):
    _git("add", "-A", cwd=WT)
    if _git("diff", "--cached", "--quiet", cwd=WT, check=False).returncode == 0:
        return
    _git("commit", "-m", msg, cwd=WT)
    for _ in range(4):
        r = _git("push", "-u", "origin", BRANCH, cwd=WT, check=False)
        if r.returncode == 0:
            return
        _git("pull", "--rebase", "origin", BRANCH, cwd=WT, check=False)   # producer and consumer raced
    raise RuntimeError("could not push the queue branch: " + r.stderr)


def items(wt: Path) -> list[Path]:
    return sorted(p for p in (wt / "queue").iterdir() if p.is_dir())


def enqueue(slug: str) -> str:
    d = EPISODES / slug / "out"
    qa = json.loads((d / "qa.json").read_text())
    if not qa.get("pass"):
        raise SystemExit(f"{slug}: QA has not passed; not enqueuing")
    meta = json.loads((d / "metadata.json").read_text())
    wt = worktree()
    uploaded = json.loads((wt / "uploaded.json").read_text())
    if any(u["slug"] == slug for u in uploaded):
        raise SystemExit(f"{slug}: already uploaded")
    same = [p for p in items(wt) if p.name.split("-", 1)[1] == slug]
    if same:                      # re-render of a queued video: replace it in place, keeping its slot
        dst = same[0]
    else:
        n = max([int(p.name.split("-", 1)[0]) for p in items(wt)] + [u.get("n", 0) for u in uploaded] + [0]) + 1
        dst = wt / "queue" / f"{n:04d}-{slug}"
        dst.mkdir(parents=True)
    shutil.copy(d / meta["file"], dst / "video.mp4")
    if (d / "thumb.jpg").exists():
        shutil.copy(d / "thumb.jpg", dst / "thumb.jpg")
    meta["slug"] = slug
    (dst / "metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    _push(f"{'Re-render' if same else 'Queue'} {slug}")
    return f"queued {dst.name} ({(dst / 'video.mp4').stat().st_size / 1e6:.1f} MB); {len(items(wt))} waiting"


def requeue(slugs: list[str] | None = None) -> str:
    """Replace already-queued videos with their latest render. Refuses any episode whose QA didn't pass."""
    wt = worktree()
    done, refused = [], []
    for it in items(wt):
        slug = it.name.split("-", 1)[1]
        if slugs and slug not in slugs:
            continue
        out = EPISODES / slug / "out"
        qa = json.loads((out / "qa.json").read_text()) if (out / "qa.json").exists() else {}
        if not qa.get("pass"):
            refused.append(slug)
            continue
        meta = json.loads((out / "metadata.json").read_text())
        meta["slug"] = slug
        (it / "metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        shutil.copy(out / meta["file"], it / "video.mp4")
        if (out / "thumb.jpg").exists():
            shutil.copy(out / "thumb.jpg", it / "thumb.jpg")
        done.append(slug)
    _push("Requeue latest renders: " + ", ".join(done))
    return f"requeued {done}; refused (QA not passed) {refused}"


def _kind(meta: dict) -> str:
    if meta.get("kind"):
        return meta["kind"]
    return "request" if "asked for this one" in (meta.get("first_comment") or "") else "formula"


def pick(wt: Path, log: list[dict], n: int) -> list[Path]:
    """Next videos to upload: only ones narrated by the Gemini voice (the channel's voice), in the priority order of
    data/upload_order.json (its formula and request lists interleaved), then the rest oldest first, never posting
    someone from the previous upload's pair."""
    vf = ROOT / "data" / "voice.json"         # read directly: the upload job doesn't install the audio stack
    want_gemini = vf.exists() and json.loads(vf.read_text()).get("engine") == "gemini"
    ready = []
    for it in items(wt):
        meta = json.loads((it / "metadata.json").read_text())
        if want_gemini and (meta.get("voice") != "gemini" or meta.get("voice_rev", 1) < VOICE_REV):
            continue                       # old voice, or voiced before the clipping fix: wait for a re-render
        ready.append((it, _kind(meta), meta.get("title", "")))
    of = ROOT / "data" / "upload_order.json"
    order = json.loads(of.read_text()) if of.exists() else {}
    lists = [v for k, v in order.items() if not k.startswith("_")]
    first = []                                   # one priority list: formula[0], request[0], formula[1], ...
    for i in range(max([len(v) for v in lists] + [0])):
        first += [v[i] for v in lists if i < len(v) and v[i] not in first]
    rank = {s: i for i, s in enumerate(first)}
    ready.sort(key=lambda x: rank.get(x[0].name.split("-", 1)[1], len(rank)))   # stable: oldest first otherwise
    prev = _people(log[-1].get("title", "")) if log else set()
    out = []
    while ready and len(out) < n:
        # highest priority first, but never the same person as the previous upload (4 Stalins in a row went
        # 63 -> 55 % viewed). Requests vs formula showed no difference, so kinds just interleave via the list.
        nxt = next((x for x in ready if not (_people(x[2]) & prev)), ready[0])
        out.append(nxt[0])
        ready.remove(nxt)
        prev = _people(nxt[2])
    return out


def _people(title: str) -> set[str]:
    """{'joseph stalin', 'elmo'} from 'How is Joseph Stalin connected to Elmo?'."""
    m = re.match(r"How is (.+?) connected to (.+?)\?$", title or "")
    return {m.group(1).lower(), m.group(2).lower()} if m else set()


def consume(max_items: int = 2, dry_run: bool = False) -> list[dict]:
    from .upload import comment_if_public, have_credentials, upload_file, youtube
    wt = worktree()
    done = []
    log = json.loads((wt / "uploaded.json").read_text())
    if not dry_run and have_credentials():   # first comments for scheduled uploads that have gone public since
        yt = youtube()
        for r in log:
            if r.get("video_id") and r.get("first_comment") and not r.get("comment"):
                c = comment_if_public(yt, r, r["first_comment"])
                if c:
                    r["comment"] = c
        (wt / "uploaded.json").write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n")
    for it in pick(wt, log, max_items):
        meta = json.loads((it / "metadata.json").read_text())
        thumb = it / "thumb.jpg" if (it / "thumb.jpg").exists() else None
        rec = upload_file(it / "video.mp4", meta, dry_run=dry_run, taken_extra=[r["publish_at"] for r in done],
                          thumb=thumb)
        rec["first_comment"] = meta.get("first_comment")
        rec["kind"] = _kind(meta)
        rec["outro"] = bool(meta.get("outro"))     # A/B test flag, compared in the stats review
        rec["narration"] = meta.get("narration", True)   # A/B test: music-only vs narrated
        rec["links"] = meta.get("links")                  # chain length test: short (<=6) vs parade (8+)
        rec["n"] = int(it.name.split("-", 1)[0])
        done.append(rec)
        if dry_run:
            continue
        log = json.loads((wt / "uploaded.json").read_text())
        log.append(rec)
        (wt / "uploaded.json").write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n")
        shutil.rmtree(it)
        _push(f"Uploaded {meta['slug']} ({rec.get('video_id')})")
    left = len(items(wt))
    out = os.environ.get("GITHUB_STEP_SUMMARY")
    if out:
        with open(out, "a") as f:
            for r in done:
                f.write(f"- {r['title']}: {r.get('url', 'dry run')} (publishes {r['publish_at']})\n")
            f.write(f"\n{left} videos left in the queue.\n")
    print(json.dumps({"uploaded": done, "left": left}, indent=1))
    return done
