# Style and policy

## What goes viral (the reference channel, @6degreesofseperation, pulled via the API in Nov 2026)

24 Shorts, 45.1M views, 41.2K subscribers, all posted Apr–Jun 2025. The channel then went quiet, so the niche is open.
Two videos made 73% of all views:

| Views | Pair | Why it worked |
|---|---|---|
| 17.3M | Adolf H. → Steve (Minecraft) | most evil person in history × the most innocent kids' game hero |
| 15.5M | Adolf H. → Hatsune Miku | same, with a cute virtual idol |
| 3.6M | Fanum → Adolf H. | streamer meme × villain |
| 1.9M | Cleopatra → Diddy | ancient icon × current-scandal celebrity |
| 1.5M | Ryan Trahan → Nikola Tesla | YouTuber × genius |
| 1.2M | Blue Shirt Guy → Grug | meme × meme |
| 17K–110K | T-34 tank → Paimon, Stalin → Twilight Sparkle, Pelé → Modi, Post Malone → Einstein, Hitler → Netanyahu | niche, or no absurd gap, or just political |

Our own old uploads say the same: Leonardo da Vinci → Naruto did 39.8K, ten times the rest.

**The formula: "how could they possibly be connected?"** Every hit pairs someone instantly recognisable and
*serious or dark* (a dictator, an ancient ruler, a genius) with something *innocent, cute or kid/gamer-coded* that
everyone under 25 knows (a game hero, a cartoon, a mascot, a virtual idol, a streamer meme). The bigger the
"that's impossible" gap, the better. Both ends must be recognisable from the thumbnail alone. Avoid two serious
people, two niche things, or a pair that only makes a political point.

**Best endpoints that are honest and doable:**
- **Costume/puppet characters that appear in real photos with real people.** Elmo, Kermit, Big Bird and Cookie
  Monster at the White House; Mickey Mouse, Pikachu, Hello Kitty and SpongeBob costume characters at events,
  premieres and parades. They are *in the same picture* as presidents and celebrities, so the last link is a real
  photo.
- **Voiced characters** through their voice actor or the actor who played them, e.g. Mario (Chris Pratt), Shrek
  (Mike Myers), SpongeBob (Tom Kenny), Sonic (Ben Schwartz), Gru and the Minions (Steve Carell), Pikachu (Ryan
  Reynolds).
- **Game heroes and virtual idols** through their creators at events (Notch, Shigeru Miyamoto, Crypton).

## What our 2025 hit did (Leonardo da Vinci → Naruto, 39.8k views)

Analysed from the video itself (52 s, about 13 people):
- **The hook was the request:** the viewer's comment ("@skibidirizzler: da vinci to naruto") on screen, with both
  faces and a red arrow.
- **A long, funny parade, not the shortest path.** Renaissance art (Francis I, Cellini circled in a fresco,
  Vasari, Pius IV), then a lightning-fast "Unbroken Line of Popes", then a run of instantly famous modern faces
  (Dalai Lama, Shinzo Abe, Bill Gates, Mark Rober, MrBeast).
- **Slow middle (3–4 s a person), sprint at the end.** Naruto's voice actor, then his creator, then Naruto, each
  about 2 s. Nothing Naruto-related appeared until the last 8 s, and the video ended the instant Naruto appeared.
- **No narration:** a trending song plus big yellow name text over full-screen photos, with a continuous
  zoom/pan and no hard cuts.

Applied: the episode skill's "comedic parade" rules (no obvious shortcuts, hide the destination, short final
lines, full lineages). The renderer reveals B at the very end of the last line, and the chain bar and counter no
longer show how many links are left.

## What the numbers say so far (update after each review)

- The gate is **viewed vs swiped away**. On HelloBuh (a sister Shorts channel, 84 Shorts), 74–81 % viewed got
  9k–26k views and 56–61 % stalled at 1–3k. The same format scored 74 % once and 58 % the next, so execution
  beats format.
- Retention drops hardest at **1.3–1.5 s**. Both faces and the question must be on screen and readable before
  then; this is why the big-face opening exists.
- 92–98 % of views come from the Shorts feed, and 98 % of viewers are new. Like rate (1–2 %) doesn't predict
  views.
- **Ours, Studio snapshot 2026-11-10 (12 Shorts, data/studio.json):**
  - Viewed vs swiped: the small-card opening scored 28 % and 42 %. The big-face opening's median is 63 %.
  - Hitler pairs top it: Hitler → Grug 72.5 % (22k views, 16 subs) and Hitler → Kermit 71.9 %. Every other pair
    is at 55–63 %, so the pair decides the gate once the opening is fixed.
  - So **Hitler pairs, and pairs with an equally instant villain, post first**: the operator puts them at the
    front of `data/upload_order.json` as soon as they're merged.
  - Stalin is wearing out: four Stalin Shorts in three days went 63 → 59 → 56 → 55 %. Don't post the same person
    in back-to-back uploads.
  - Request vs formula, and "And that's how." vs none: no difference yet (medians within 1 point).
  - Retention: 1 s is about 135–145 % (replays). Every Short drops most at 5–7 s, 2–4 s into the first link, and
    ends at 50–70 %. Average view is 20–29 s.
  - The audience is 13–17, mostly US, 99 % new viewers, and 95 % come from the Shorts feed.
  - Comments are pair requests: harvest them every review. One said "please quit yapping": keep lines short.

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
- Two uploads a day, in fixed slots (see Upload); never batch-dump.
- Credits for every photo go in the description (CC BY/BY-SA requires it).
- Images: Wikimedia/public domain first. Use press photos only when nothing free
  exists, and only with narration over them (fair use commentary). QA warns on
  every non-free image.

## Voice and look

- Narrator: Gemini TTS voice **Achird** (creator's pick), the only voice the channel uses. Upbeat, curious,
  fast, American. Configured in `data/voice.json`; details in `docs/RUNBOOK.md`.
- Opening: both faces with names on frame 0, the question "How is A connected to B?" spoken immediately, no
  count ("N pictures apart") and no flash-forward. The two faces are **big, full-width photos** (A on top, B
  below, the spoken question on the seam between them), not small cards: the first two uploads in the card
  layout kept only 22–25% of viewers past the first second, so the first frame has to read as "these two?!" at
  feed size. The thumbnail is the same frame with a "?" on the seam.
- Everything important stays inside the Shorts safe zone (clear of the top icons, the right-hand buttons and the
  bottom title/channel overlay); the chain bar sits at the top.
- Pixel-arcade style with Buh, the pushpin detective; upbeat Mixkit music only; voice above music and SFX.
- Length: as short as the chain allows (the reference channel's hits ran 31–51 s). QA fails past 3:00 and warns
  past 60 s.

## Upload

`upload.yml` (started by one cron-job.org job at 8:07 AM Central) uploads the day's 3 ready videos as private, each
scheduled to go public at about 11:30 AM, 6:30 PM and 9:30 PM, through the official YouTube Data API (no audit needed). See `docs/RUNBOOK.md`.
