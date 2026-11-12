# Monetization readiness plan (YPP)

Goal: when Buh Connects reaches the YouTube Partner Program thresholds, a reviewer skimming the channel should see
an **original research channel with a recurring character and real variety**, not "narrated slideshows with only
superficial differences". That phrase is YouTube's own example of *inauthentic (mass-produced) content*, renamed
from "repetitious content" in July 2025; a July 2026 update added template-based videos with little variation
between uploads. Review is channel-level: the newest uploads, the most-viewed, the About page.

Thresholds (YPP): 1,000 subscribers **and** 10M valid public Shorts views in 90 days (or 4,000 long-form watch
hours in 12 months). Lower tier (fan funding only): 500 subs + 3M Shorts views in 90 days. As of 2026-11-12:
~67 subs, ~42k views in 28 days. So there's time to build this before applying, and the work also helps growth.

## Where we're exposed today
| Signal a reviewer sees | Today | Risk |
|---|---|---|
| Same template every video | yes | high |
| AI narration, same delivery | yes (2/3 of videos) | medium: AI is allowed if the video adds value |
| Music-only: names over photos plus a song | 1/3 of videos | **highest**: closest to "a slideshow" |
| 3 uploads a day, identical format | yes | medium-high |
| Research visible | no (it happens off-screen) | lost credit |
| Original IP (Buh the pin detective) | yes | helps |
| A human anywhere | no | medium |

## Workstreams

### A. Make the research visible (automatic, every video) [build now]
1. **On-screen fact line in music-only videos:** a short caption per link ("Berghof, 1937"), from a new optional
   `fact` field in each link (the worker writes it; fallback: the year and place pulled from the line).
2. **Research write-up in every description:** the full chain with year, place and photo source for each link,
   generated from episode.json. Replaces the current bare credits list.
3. **Pinned/first comment:** the "fun fact you didn't see" from the research (worker writes one per episode).
4. **Hard-search note when it's true:** e.g. "found on the 3rd route; the Scooby photos were all watermarked".
   The skill asks workers to log their dead ends, so we have this already.

### B. Real variety (scheduled formats) [plan now, start in ~2 weeks]
- **Weekly "Impossible Request" (Fri):** the hardest viewer request of the week, longer and with the dead ends
  shown ("we tried X, Y; here's what worked"). Its own title style and opening.
- **Weekly "Three-way" (Tue):** connect A → B → C in one video.
- **Lineage specials:** "All 266 popes in 10 seconds" style episodes built around a full lineage scroll.
- **Monthly long-form (8–12 min):** "How we connect anyone in 6 photos", a walkthrough of one search, or the
  month's best chains with commentary. Long-form also counts toward the 4,000-hour route.
- **Series playlists** per format, visible on the channel page.

### C. A human (and Buh) on the channel [needs the owner]
- Owner records a short voice intro or outro for specials ("This one took me two days"), or narrates one
  special a week.
- **Buh gets a personality:** a recurring catchphrase and reaction on the final reveal; Buh is original IP, our
  strongest "not templated" asset.
- **Reply to comments** from the channel account, especially the requests ("adding this to the list"), plus a
  weekly Community post: "Request of the week" or a poll.

### D. Channel page [owner, about 30 min]
- An About text that describes the method: real photos only, every chain researched and sourced, viewers choose
  the pairs.
- Banner and avatar featuring Buh; playlists from B.

### E. Cadence
- Keep 3/day while growing. **About 2 weeks before we expect to cross the thresholds**, make sure the newest ~20
  uploads show the variety in B and the visible research in A, and consider 2/day plus specials.

### F. Hygiene
- Keep photo credits and the fair-use commentary framing (our narration and facts are the commentary).
- Music-only + trending song: claims only, fine under 60 s; never reuse other creators' video clips.
- Don't copy the reference channel's look one-to-one (their comment-card intro is common to the genre, but keep
  Buh, our pixel style and our own title phrasing).

## Order of work
| When | What | Who |
|---|---|---|
| This week | A1 fact line, A2 description write-up, A3 fact comment (renderer, kit, skill) | Claude |
| This week | D: About text draft for the owner to paste | Claude drafts, owner pastes |
| Week 2 | B: "Impossible Request" and "Three-way" formats in the renderer and skill; first episodes | Claude |
| Week 2–3 | C: owner voice intro/outro recorded once and reused on specials; Buh catchphrase | owner + Claude |
| Week 3–4 | B: first long-form video | Claude builds, owner reviews |
| At ~700 subs or ~7M Shorts views in 90 days | E: readiness audit of the newest 20 uploads, About page, playlists | Claude + owner |
| At thresholds | Apply | owner |

## Readiness checklist (before applying)
- [ ] The newest 20 uploads include at least 3 non-standard formats and at least 1 long-form.
- [ ] Every video since A shipped has the research write-up and the fact line.
- [ ] The About page describes the method; playlists per series.
- [ ] Channel replies on recent comments; at least 2 Community posts in the last month.
- [ ] No reused clips; all photos credited.
