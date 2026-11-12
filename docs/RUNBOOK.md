# Runbook

Buh Connects (@buhconnects, channel `UCE3KAucU-r7Jp3BRGbreQoQ`): YouTube Shorts that connect two famous people
through real pictures. Producer/consumer: Claude sessions make episodes, GitHub renders and queues them, a cron
job uploads three a day.

The episode procedure is the skill **`.claude/skills/six-degrees-episode/SKILL.md`**, the single source of truth
for producer sessions. What goes viral and the pair formula are in **`docs/STYLE.md`**.

## How the system runs (times Central)

| When | Who | What |
|---|---|---|
| 3:07 AM daily | "Six Degrees daily producer" scheduled task → operator session | 1–3 new episodes (3 if fewer than 15 videos are waiting, 2 if 15–29, 1 if 30+), each in a **fresh** worker session, alternating a viewer request and a viral-formula pair; the operator invents new formula pairs when the list runs out |
| when merging episode PRs | operator session | strong pairs go near the top of `data/upload_order.json` in the same merge |
| on each episode PR | `render.yml` | voices (Gemini Achird), renders and runs QA as a check; nothing is queued |
| operator check-ins | operator session | merges PRs whose check passed; archives finished workers |
| on merge to main | `render.yml` | renders again and appends the video to the `queue` branch |
| 8:07 AM daily | cron-job.org, one job "sixdegrees upload (morning)" → `upload.yml` with `{"max":"3","privacy":"scheduled"}` | uploads the next 3 ready videos as private, each scheduled public at ~11:30 AM, 6:30 PM and 9:30 PM (+0–20 min jitter, `upload.SLOTS`); posts first comments on videos that went public since the last run (so a day late); thumbnails once the channel is phone-verified |
| Fri 9:37 PM weekly | "Six Degrees: weekly performance review" scheduled task → operator session | stats, compares opening/kind/outro groups, proposes up to 3 changes, waits for the creator's go-ahead; re-schedules itself for next Friday |
| on demand | `stats.yml` | views/likes/comments + retention (yt-analytics scope) into `stats.json` on `queue` |
| on demand | `voice.yml` | voices several episodes per Gemini request into the `tts-cache` branch |
| on demand | `research.yml` | a competitor channel's uploads + stats into the `research-data` branch |

## Branches

- `main`: code, episodes (`episodes/<slug>/episode.json`), `data/pairs.json`. Only changed through PRs.
- `queue`: `queue/NNNN-<slug>/{video.mp4, metadata.json, thumb.jpg}`, `uploaded.json`, `stats.json`.
- `tts-cache`: Achird voice lines (`<key>.flac`) and every raw Gemini recording with its Whisper words (`raw/`).
- `research-data`, `tts-samples`: research and voice-comparison output.

## Narration: Gemini "Achird" only

- `data/voice.json`: engine `gemini`, voice `Achird`, model `gemini-3.8-flash-tts`. **No other voice is ever
  published.** A line Gemini hasn't voiced stops the render; local sessions without a key render silent
  narration for layout previews, and QA marks those unpublishable.
- Key: GitHub secret `GEMINI_API_KEY` (free tier: about 15 requests/day, 3/minute, resets at midnight Pacific).
- One request voices several lines/episodes (`sixdeg/tts_gemini.py`, director's-notes prompt so the notes aren't
  read aloud). `audio.split_batch` cuts the recording back into lines at the quietest point near each Whisper
  word boundary; `clean_tail` removes Gemini's end-of-recording noise burst; `_trim` starts each clip on speech.
- Every raw recording is kept. After a splitting/trimming change: `python -m sixdeg resplit` (free), bump
  `VOICE_REV` in `audio.py` **and** `queue.py`, then re-render the queue.
- Guards: QA fails a line that ends mid-sound; the uploader only posts `voice == "gemini"` and
  `voice_rev >= VOICE_REV`. Backlog voicing: `voice.yml` with `batch` 5 and the slugs in queue order.

## Trending songs (music-only episodes)

- Music-only episodes use a trending song from the **private** repo `evanlin23/sixdegrees-music` (never commit
  songs to this public repo; `music-private/` is git-ignored). `render.yml` clones it when the Actions secret
  `MUSIC_REPO_TOKEN` exists: a fine-grained token with read-only Contents on that repo. Without it, music-only
  episodes fall back to Mixkit.
- In that repo: the audio files, plus an optional `tracks.json`
  `[{"file": "washing-machine-heart.mp3", "title": "Washing Machine Heart", "artist": "Mitski", "start": 42.0}]`.
  `start` is the second where the recognisable part begins. Each episode picks a track by slug hash, and the song
  is recorded per upload (`music`, `trending`). An episode's `"music"` pick applies only to narrated renders; to
  keep Mixkit on a music-only render, set `"trending": false`.
- A claimed song is fine on a Short under 60 s: it stays up, and the music partners take a share of that Short's
  pool revenue. Over 60 s it gets blocked, so QA fails trending-song renders at 59.5 s or longer.

## Uploads

- `queue.consume` (via `upload.yml`): picks only Achird videos of the current voice revision, in the priority order
  of `data/upload_order.json`. Its `formula` and `request` lists are interleaved into one list (formula[0],
  request[0], formula[1], …), and unlisted videos go oldest first. It never posts someone from the previous
  upload's pair. Records `kind` and `outro` in `uploaded.json`. A viewer request with Hitler (e.g. Miku) goes in
  the `request` list. Edit the lists to change what posts next.
- Title is always "How is A connected to B?" with the names people know; description = chain, call to action,
  photo credits, hashtags; first comment credits the requester.
- A/B tests, all deterministic per slug and recorded in `metadata.json`/`uploaded.json`, compared by
  `python -m sixdeg studio` and the weekly review:
  - `outro`: half the episodes end with Achird saying "And that's how." (episode override `"outro"`).
  - `narration`: a third of the episodes are **music-only**: no voice, names on screen, the opening question as
    text (like the 2025 Da Vinci → Naruto). Episode override: `"narration": false/true`.
  - `links`: chain length, grouped short (≤6), mid (7) and parade (8+). This is a content choice, so the episode
    skill aims for a mix.

## Opening (what keeps viewers past second 1)

A on top and B below as big full-width photos from frame 0, the spoken question on the seam between them
(`render_pixel.frame_hook`, `HERO_A`/`HERO_B`). The photo is the person's first portrait, or
`people.<Name>.hook_photo` in episode.json to choose a better one. The small-card opening it replaced kept
22–25% of viewers. The thumbnail is the same frame with a "?" on the seam.

## Metrics review (what works, what doesn't)

- Every 2–3 days the owner pastes `docs/studio-metrics-prompt.md` into Claude in Chrome, signed in to YouTube
  Studio. It reads each Short's live Studio numbers: viewed vs swiped away, retention at 1 s/3 s/mid/end, average
  % viewed, traffic, top comments and audience. It answers with one JSON block. Give that to the operator session,
  or run `python -m sixdeg studio <file>`. That stores it in `data/studio.json`, joins it with the upload log and
  compares groups: opening (cards vs big-face), kind (request vs formula) and outro (yes/no).
- `stats.yml` (API views/likes/comments) is the fallback. It lags 1–3 days and has no swipe-away or retention
  curve.
- The weekly review (Fri 9:37 PM) uses the latest Studio snapshot plus stats.json. It proposes up to 3 changes
  and waits for the owner's go-ahead.
- House rule (from HelloBuh): when a Short flops, check its execution first: a slow first 1.5 s, an unreadable
  face, a bad name. Fix the same fault in queued siblings before blaming the pair type.

## Presentation styles

- The same chain can be shown many ways, so the channel doesn't look templated. Built in (`render_pixel`):
  `cards` (the original), `zoom` (continuous full-screen zoom), `board` (Buh's detective corkboard), `timeline`
  (year counter and ruler), `globe` (pixel Earth, needs `where` on every link). Plug-ins: any
  `sixdeg/styles/<name>.py` with `make(ctx)` (the API is in `sixdeg/styles/__init__.py`).
- Which style a video gets: the episode's `"style"`, else the `rotation` list in `data/style.json` by slug hash.
  Only add a style to the rotation after the creator has seen a preview. The style is recorded per upload, and
  `python -m sixdeg studio` compares styles.
- Test one: `SIXDEG_STYLE=<name> python -m sixdeg render <slug>` (doesn't touch the episode file).

## Re-rendering

- One or more queued episodes: dispatch `render.yml` with `slug` = space-separated slugs and `enqueue` true. A
  re-render replaces the queued video in place and keeps its slot. Keep it to 3–4 slugs per run.
- After a layout or narration change, re-render every queued slug this way, in queue order.

## Known gaps / owner actions

- Custom thumbnails need the channel phone-verified (youtube.com/verify); until then YouTube picks a frame.
- Retention numbers lag 24–48 h behind an upload.
