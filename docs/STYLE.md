# Style and policy

## What works (from @6degreesofseperation, Nov 2026)

That channel has 24 Shorts, 45.1M views and 41.2K subscribers since Feb 2025.
Median Short ≈ 300K views.

| Pattern | Examples (views) |
|---|---|
| Historical villain/icon × modern meme, fiction or streamer | Adolf H. → Steve (Minecraft) 17M · Adolf H. → Hatsune Miku 15M · Fanum → Adolf H. 3.6M |
| Ancient/historic × celebrity | Cleopatra → Diddy 1.8M · Ryan Trahan → Nikola Tesla 1.4M |
| Two meme characters | Blue Shirt Guy → Grug 1.1M |
| Weak: two niche or non-person things | T-34 tank → Paimon 17K · Stalin → Twilight Sparkle 29K |

**Pair formula:** one side instantly recognisable *and* serious (a dictator,
emperor, scientist, pope or president), the other side young-audience pop culture
(a streamer, YouTuber, rapper, game or anime character, meme). The bigger the
"no way these are connected" gap, the better.

Their format is music only, cyan name tags over heads, and pans between faces.
We add **narration that explains each link**. That makes every video original
commentary and is our edge for monetization (see below).

## Monetization rules (must stay YPP-eligible)

YouTube's "inauthentic content" policy (renamed July 2025) demonetizes
mass-produced, template-only videos. It is judged per channel. So:
- **Every line must carry a real, specific fact** (a year, a place, a reason).
  Never write filler like "these two met".
- Vary the hook wording and the title pattern across uploads. `kit.py` takes
  `title`/`description` from episode.json, so write them fresh.
- Historical figures, dictators included, are fine **framed as history**:
  neutral facts, no praise, no jokes about atrocities.
- Advertiser-friendly: no swastika or gore as the focal point, no slurs, no shock
  words in titles. Write "Hitler", not "Adolf H." (it's history, not evasion).
- Max one upload per day at first; do not batch-dump.
- Credits for every photo go in the description (CC BY/BY-SA requires it).
- Images: Wikimedia/public domain first. Use press photos only when nothing free
  exists, and only with narration over them (fair use commentary). QA warns on
  every non-free image.

## Voice and look

- Narrator: Kokoro `am_michael`, speed 1.12, generated locally.
- Music and SFX: generated procedurally in `audio.py`, so there are no Content ID claims.
- Colours: cyan (#22D3EE) for the earlier person, yellow (#FFD60A) for the newly
  revealed one. Font: Inter Black.
- Length 25-55 s. QA hard-fails outside 15-60 s.

## Upload

The mode is pending the creator's choice:
- `kit` (current): the session opens a GitHub issue with the upload kit and sends
  the mp4, and the creator posts it from their phone (about 1 minute).
- `api`: YouTube Data API `videos.insert` with a one-time OAuth refresh token
  stored as an environment secret. Videos stay private until Google's free
  compliance audit approves the project, so run `kit` until that is approved.
- Browser automation is not used: it breaks YouTube's terms, and Google blocks
  logins from cloud IPs.
