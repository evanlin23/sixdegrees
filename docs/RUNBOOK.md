# Runbook

Buh Connects (@buhconnects, channel `UCE3KAucU-r7Jp3BRGbreQoQ`): YouTube Shorts that connect two famous people
through real pictures. Producer/consumer: Claude sessions make episodes, GitHub renders and queues them, a cron
job uploads two a day.

The episode procedure is the skill **`.claude/skills/six-degrees-episode/SKILL.md`**, the single source of truth
for producer sessions. What goes viral and the pair formula are in **`docs/STYLE.md`**.

## How the system runs (times Central)

| When | Who | What |
|---|---|---|
| 3:07 AM daily | "Six Degrees daily producer" scheduled task → operator session | 1–2 new episodes (1 if 14+ videos are waiting), each in a **fresh** worker session, alternating a viewer request and a viral-formula pair; the operator invents new formula pairs when the list runs out |
| on each episode PR | `render.yml` | voices (Gemini Achird), renders and runs QA as a check; nothing is queued |
| operator check-ins | operator session | merges PRs whose check passed; archives finished workers |
| on merge to main | `render.yml` | renders again and appends the video to the `queue` branch |
| 11:22 AM and 6:22 PM | cron-job.org (two jobs) → `upload.yml` with `{"max":"1","privacy":"public"}` | uploads the next ready video publicly, posts the first comment (and the thumbnail once the channel is phone-verified) |
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

## Uploads

- `queue.consume` (via `upload.yml`): picks only Achird videos of the current voice revision, alternating viewer
  requests and viral-formula pairs, oldest first; records `kind` and `outro` in `uploaded.json`.
- Title is always "How is A connected to B?" with the names people know; description = chain, call to action,
  photo credits, hashtags; first comment credits the requester.
- A/B test: half the episodes (hash of the slug) end with Achird saying "And that's how." Compare completion and
  replays by the `outro` flag in the stats review.

## Re-rendering

- One or more queued episodes: dispatch `render.yml` with `slug` = space-separated slugs and `enqueue` true. A
  re-render replaces the queued video in place and keeps its slot. Keep it to 3–4 slugs per run.
- After a layout or narration change, re-render every queued slug this way, in queue order.

## Known gaps / owner actions

- Custom thumbnails need the channel phone-verified (youtube.com/verify); until then YouTube picks a frame.
- Retention numbers lag 24–48 h behind an upload.
