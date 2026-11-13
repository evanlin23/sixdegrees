# Second team: status board

Conversation happens in the **Orchestrator channel**, [issue #200](https://github.com/evanlin23/sixdegrees/issues/200).
This file holds status and the owner's rules for new episodes.

- **[#1] — orchestration** (lead): queue, Review Room, renderer/styles, merges, uploads.
- **[#2] — creation**: new episodes, as PRs from `claude/ep-<slug>`.

## Owner rules for new episodes (owner's words outrank everything here)
Follow `.claude/skills/six-degrees-episode/SKILL.md`. The owner's standing directions, newest first:

**What to make**
- **Only "How is Adolf Hitler connected to X?" Shorts post for now.** Make Hitler pairs, and vary the first link out of Hitler; don't open with Franco every time.
- **Never:** Minecraft (Steve is done), Epstein, or Kanye.
- **News figures:** neutral framing only. The facts are photo facts.
- **X should be instantly recognisable and fun:** cartoon, game or movie characters, YouTubers, athletes, inventors, presidents, and people trending in the news. Viewer requests from comments go first (credit them by @handle; `requested_by` must be the real `@handle`, never "a viewer").

**The chain**
- **The first honest path you find, never padded.** Short is fine; don't add links just to lengthen it.
- **Every link is a real photo (or real painting/coin for history) of both people together.** No watermarks. Check every face by eye.
- **Lineage scrolls only for long lines of 40 or more holders** (popes, emperors, US presidents). Never short ones like German chancellors or one dynasty.

**The script**
- **Year first on every line** ("In 1938, …", "In the 1930s, …" when unsure; never invent a year). Name people instead of using pronouns that could mean someone else.
- **Every link but the last gets a context line.** The last line is a short payoff that ends on B's name (e.g. "And in 2023, Chris Pratt became the voice of Mario himself.").
- **The pace stays even to the end:** each link is at least 3.4 s, and the ending is never rushed.
- **No dead air:** the question starts on frame 0, the voice never stops for more than ~2 s, and the video ends ~0.5 s after the last word. QA fails renders that break this (first word > 0.6 s, a gap > 2 s, > 1 s of silence at the end).
- **Every episode is narrated** (Gemini, Achird voice), with no trending song.

**Formats**
- The renderer picks the format from `data/style.json`; don't set `style` in episode.json unless [#1] asks.

## Voice quota
Gemini TTS has a daily limit shared by both teams (it resets 08:00 UTC). Every PR render voices new lines, so push an episode once it's ready, and don't re-voice finished lines for small wording changes.

## Done / in the pipeline (don't remake)
- **Hitler pairs uploaded:** Steve, Hatsune Miku, Pikachu, Kermit, Grug.
- **Hitler pairs queued:** Mickey Mouse, Albert Einstein, SpongeBob, Neil Armstrong, Elmo, Mario, Sonic, MrBeast, Shrek (held for owner notes).
- **Hitler pair in PR:** Jensen Huang (#161, waiting on voice quota).

## In progress (claim before starting: one line per pair, with owner and date)
- Adolf Hitler → Thomas the Tank Engine: [#2], 2026-11-13, PR #206 (moving two photos off finna.fi, which GitHub's runners can't reach)
- Adolf Hitler → Michael Jackson: [#2], 2026-11-13, PR #207 (render green)
- Adolf Hitler → Lionel Messi: [#2], 2026-11-13, PR #208 (render green)
- Adolf Hitler → Mr. Bean: [#2], 2026-11-13, PR #209 (render green)
- Adolf Hitler → Big Bird: [#2], 2026-11-13, branch ready, PR after the Nov 14 voice reset
- Adolf Hitler → Mr. Rogers: [#2], 2026-11-13, branch ready, PR after the Nov 14 voice reset
- Adolf Hitler → Barney: [#2], 2026-11-13, in progress (Cookie Monster or Peppa Pig if Barney has no honest path)
- Adolf Hitler → Taylor Swift: [#2], 2026-11-13, in progress
- Adolf Hitler → Dolly Parton: [#2], 2026-11-13, finished but held: she died on 25 Aug 2026 (ASK on #200)

## Requests from [#1]
- _(posted as `ASK: make <A> → <B>` in the channel)_
