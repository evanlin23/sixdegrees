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

`upload.yml` (started by two cron-job.org jobs at 11:22 AM and 6:22 PM Central) uploads one ready video each time,
publicly, through the official YouTube Data API (no audit needed). See `docs/RUNBOOK.md`.
