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
- **Real people link only through real people (owner, 2026-11-14).**
  - **Actors and voice actors are only for fictional characters:** "we should only use like voice actor or movie if its like a fictional character".
  - **Historical figures** (Caesar, Cleopatra, Augustus, Genghis Khan, Henry VIII) link through real artworks or objects that show both people (paintings of real meetings, coins), or through a 40+ lineage. Never through an actor who played them; the owner says that "feels like cheating".
- **Gods and myths: bridge through real people in the myth (owner, 2026-11-14).** Examples: Alexander the Great (son of Zeus-Ammon) on coins with Zeus, Henry IV with Jupiter in Rubens, Washington with Minerva in Brumidi.

**The script**
- **Write like Steve and Miku** (owner's favourites; full model scripts in the skill, "Write like Steve and Miku"): 9-13 words a line, ONE fact per line, plain descriptors ("Britain's former king", "the new Queen") over stacked titles and places, full name the first time then the short name, and each line hands off to the next person.
- **The year leads every line**, and natural connectors count ("In 1938, …", "That year, …", "Back in 2012, …"; "In the 1930s, …" when unsure; never invent a year). Avoid pronouns that could mean someone else.
- **Every link but the last gets a context line.** The last line is a short payoff that ends on B and varies its wording, not "…himself." every time (e.g. "And in 2025, Jack Black became Steve in A Minecraft Movie.").
- **The pace stays even to the end:** each link is at least 3.4 s, and the ending is never rushed.
- **No dead air:** the question starts on frame 0, the voice never stops for more than ~2 s, and the video ends ~0.5 s after the last word. QA fails renders that break this (first word > 0.25 s, a gap > 2 s, > 1 s of silence at the end).
- **Every episode is narrated** (Gemini, Achird voice), with no trending song.

**The opening**
- **The two opening pictures are the most recognisable photo of A and of B (owner, 2026-11-13), not crops from the chain photos.** Set `people.<name>.hook_photo` to an iconic, frontal photo. Characters and puppets need a solo image of the character with no human face in frame (renderer support for faceless characters: ASK on #200).

**Formats**
- The renderer picks the format from `data/style.json`; don't set `style` in episode.json unless [#1] asks.

## Voice quota
Gemini TTS has a daily limit shared by both teams (it resets 08:00 UTC). Every PR render voices new lines, so push an episode once it's ready, and don't re-voice finished lines for small wording changes.

## Done / in the pipeline (don't remake)
- **Hitler pairs uploaded:** Steve, Hatsune Miku, Pikachu, Kermit, Grug, Jensen Huang, Lionel Messi, Mickey Mouse, Albert Einstein, SpongeBob, Neil Armstrong, Shrek, Michael Jackson.
- **Hitler pairs queued:** Mario, Thomas the Tank Engine, MrBeast, Elmo, Mr. Bean, Sonic, Taylor Swift, Big Bird, Cookie Monster.
- **Exploration queued (restyled, A/B):** Saddam → Cookie Monster, Mussolini → Mario.

## In progress (claim before starting: one line per pair, with owner and date)
- Adolf Hitler → Mr. Rogers: [#2], PR #234, re-routing without the baby-Simeon photo (never minors)
- Adolf Hitler → Walter White (`requested_by @ThatOneGingerCustoms`): [#2], 2026-11-14, branch `claude/ep-adolf-hitler-to-walter-white`
- Adolf Hitler → Verity, Agamemnon, Cat in the Hat, Buddy the Unicorn, Liberty Biberty, Bob Ross, Steve Irwin, the Grinch, Spider-Man: [#2], branches; some wait on the owner's ruling on childhood photos of historical people
- Re-routes (real people only): [#2], `claude/ep-reroute-1..4`; un-held one by one as they merge
- Adolf Hitler → Dolly Parton: dropped for now (owner, Nov 13); kept on branch `claude/ep-adolf-hitler-to-dolly-parton`, no PR

## Requests from [#1]
- _(posted as `ASK: make <A> → <B>` in the channel)_
