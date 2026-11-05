# Runbook: make one Six Degrees Short

This is the job a scheduled Claude Code session does, start to finish, with no
human in the loop. The session itself is the "AI": it does the research, looks
at every photo, and writes the narration. No paid AI API is called.

Read `docs/STYLE.md` first. It covers what makes these videos work and what
keeps them monetizable.

## 0. Setup (each fresh container)

```bash
pip install -q -r requirements.txt
python -m sixdeg setup          # face + voice models into models/ (~400 MB, cached per container)
```

## 1. Pick the pair

Take the first entry in `data/pairs.json` with `"status": "todo"`. If none are
left, invent five more following the formula in `docs/STYLE.md`, append them,
and take the first. Slug = `<from>-to-<to>`, lowercase, dashes.

## 2. Find the chain

```bash
python -m sixdeg search "Adolf Hitler" "Neil Armstrong"     # 5-15 min, cached afterwards
python -m sixdeg neighbors "Wernher von Braun"              # who is photographed with X
python -m sixdeg photos "Walt Disney"                       # every captioned photo on X's article
```

`search` runs a two-way breadth-first search over Wikipedia photo captions
(person X's article has a photo captioned "X with Y" → edge X–Y). It prints
candidate chains with the exact file for each link. Captions are only leads:
step 4 checks the faces.

**Endpoints not on Wikipedia's photo graph** (streamers, memes, fictional
characters) need a bridge you research yourself. Use web search for a real photo
of the endpoint with someone famous who *is* on the graph, for example a streamer
with a footballer, or a voice actor with a director. For fictional characters, the
first link is the character with their actor or creator (a still or a premiere
photo), and the line says so plainly ("Steve is played by Jack Black in the
movie"). For such images use `{"url": ..., "credit": ..., "license": "Fair use
(commentary)", "source": <page url>}`.

Pick the chain with:
1. 3-7 photos (best: 4-6),
2. the most *surprising* middle people (an astronaut, a pope, a rapper),
3. clear photos where both faces are visible and not tiny,
4. free licenses (Wikimedia) wherever possible.

## 3. Write `episodes/<slug>/episode.json`

The schema is documented in `sixdeg/episode.py`. Writing rules (STYLE.md has more):
- `hook`: one question, 12 words or fewer, both names in it.
- each `line`: 1-2 sentences, at most ~24 words. Say who, when, where, and *why*
  they were together. Use a concrete fact, not "they met".
- `outro`: optional. The default is "<A> to <B> in N photos. Comment two names…".
- `title`: a question or a claim, under 70 characters, with no slurs or shock words.
  `description`: one sentence.
- Wrong pronunciation? Add a respelling to `RESPELL` in `sixdeg/audio.py`.
  The QA transcript check will tell you.

## 4. Prep and verify, and actually look at the photos

```bash
python -m sixdeg prep <slug>
```

For every link it downloads the photo and reference portraits, detects faces and
matches them with SFace. It writes `episodes/<slug>/work/review<i>.jpg`: green box
for A, blue for B, and every face numbered with its similarity to A and B.

**Open every review image with the Read tool.** Check that:
- the green box really is A and the blue box really is B,
- the photo is what the narration claims,
- no swastika, gore or nudity is the focal point (cropping or zoom can avoid it;
  otherwise pick another photo).

If a face is wrong or `unverified`/`missing`, pin it in episode.json with
`"a_face": <index>` or `"b_face": [x, y, w, h]` (source pixels). Paintings,
statues and cartoons never match SFace, so pin them after you have looked. Re-run
`prep` after edits.

## 5. Render, QA, look again

```bash
python -m sixdeg render <slug>
python -m sixdeg qa <slug>          # must print "pass": true
python -m sixdeg kit <slug>         # out/upload.md + out/metadata.json
```

Open `episodes/<slug>/out/contact.jpg` and check the name tags sit on the right
heads and the captions read well. Fix and re-render until QA passes. Never ship a
QA failure.

## 6. Upload and deliver

```bash
python -m sixdeg upload <slug>      # private, scheduled into the next free slot (2 a day)
```

YouTube is the backlog. Every video goes up private with a scheduled publish time,
so it doesn't matter that this container disappears. The command appends to
`data/uploads.json`. If the `YT_*` environment variables are missing, it does a dry
run and says so; then the creator uploads by hand from the kit.

1. In `data/pairs.json`, set the pair's `status` to `"done"` and add the slug.
2. Commit `episodes/<slug>/episode.json`, `data/pairs.json` and `data/uploads.json`
   (the video itself is gitignored) and push to `main`.
3. Open a GitHub issue titled `Scheduled: <title> (<publish time>)` with the
   YouTube link and `out/upload.md`, so the creator can see what's queued.
4. Send the mp4 to the user with SendUserFile (`display: "attach"`).

## Cadence

The scheduled run makes **two** episodes per run (one per publish slot). If
`data/uploads.json` already has 4 or more videos scheduled in the future, make
only one; never let the queue grow past about 6.
