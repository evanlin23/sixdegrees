---
name: six-degrees-episode
description: Make one Buh Connects ("Six Degrees") YouTube Short for a pair of people. Research an honest chain of photos/paintings, write episode.json, verify faces by eye, preview, and open a PR. Use for every producer session.
---

# Make a Six Degrees episode

The channel is **Buh Connects** (@sixdegreesbetweenus). Buh, a pixel pushpin detective, connects any two
people through pictures of people who actually met. You make the episode. GitHub renders, checks and uploads it.

## Setup (fresh container)
```bash
pip install -q -r requirements-render.txt
python -m sixdeg setup
git checkout -b claude/ep-<slug>          # your own branch; never commit to main
```

## 1. The pair
Use the pair you were given, or the first `"status": "todo"` entry in `data/pairs.json` (viewer requests come
first). When you pick new pairs yourself, follow "What goes viral" in `docs/STYLE.md`: a serious or dark,
instantly recognisable person × something innocent, cute or kid/gamer-coded (Hitler → Minecraft's Steve made
17M views). Costume and puppet characters (Elmo, Kermit, Mickey, Pikachu) are ideal endpoints because they appear
in real photos with presidents and celebrities. Never repeat a pair already in `data/pairs.json`. Slug: `<from>-to-<to>`, lowercase with dashes. If the entry has `requested_by`, copy it into
episode.json: the hook shows the viewer's request bubble and the first comment credits them.

## 2. The chain (honesty rules)
- Every link is a **real picture of two people together**: a photo, or a painting, mosaic or sculpture that
  depicts them. If it's a depiction, the line says so ("Michelangelo painted Adam…").
- Find links with `python -m sixdeg search "<A>" "<B>"`, `neighbors "<X>"`, `photos "<X>"`, plus web research.
  Wikipedia/Wikimedia pictures first. Press photos only when nothing free exists:
  `{"url", "credit", "license": "Fair use (commentary)", "source"}`.
- **Crossing centuries: a lineage link.** One link that scrolls a line of succession (popes, emperors, kings,
  presidents), one portrait per holder:
  ```json
  {"a": "Saint Peter", "b": "Pope Francis", "line": "Saint Peter was the first pope, and the line runs 266 popes to Pope Francis.",
   "lineage": {"label": "LINE OF POPES", "people": [
     {"name": "Saint Peter", "file": "File:...jpg", "label": "1st pope"},
     {"name": "Pope Gregory I", "file": "File:...jpg", "label": "64th"},
     {"name": "Pope Francis", "file": "File:...jpg", "label": "266th"}]}}
  ```
  The first and last names must equal `a` and `b`. Use 6-15 portraits, choosing well-known holders to skip
  through a long line. It counts as one link.
- **Same picture, same medium.** Both people in a link must be depicted *in the same picture*. Real people link
  through a photo of them together. People from the past link through an artwork that depicts **both** of them:
  the same painting, the same mosaic, or two figures in the same sculpture group. A real person photographed
  standing in front of a painting or next to a statue is **not** a link, and viewers call it a cheat. The way
  across from art to the modern world is an artwork that also depicts a real, photographed-era person (a donor or
  patron portrait, the artist's self-portrait in the work, a pope in a fresco), a lineage link, or an actor link
  (below). Examples:
  - Michelangelo's *Last Judgment* depicts Adam and Michelangelo himself (his self-portrait).
  - Raphael's *School of Athens* depicts Michelangelo and Raphael.
  - From there, link to a pope the artist painted, and continue down the line of popes.
- **Who may use actors: only fictional characters.** Three kinds of endpoint:
  - **Real people, historical ones included (Caesar, Napoleon, Qin Shi Huang, Jesus):** photos together,
    artworks depicting both, and lineage links only. An actor who *played* a real person is **not** a link. Jesus
    goes through an artwork showing him with someone (e.g. Perugino's *Christ Handing the Keys to St Peter*), then
    the line of popes.
  - **Mythological and religious figures (Zeus, Adam, Athena):** go through mythology in art. Use an artwork showing the
    god with another figure, then an artwork where that figure appears with a real person. Examples:
    - an Athenian vase of Zeus with Athena, then Athena (Minerva) in Brumidi's *Apotheosis of Washington* with
      George Washington, then the line of presidents;
    - Rubens' Marie de' Medici cycle, which paints Jupiter beside Henry IV and Marie de' Medici;
    - Michelangelo's *Last Judgment*, which depicts Adam together with Michelangelo's self-portrait.
  - **Fictional characters (cartoons, games, anime, film characters):** their voice actor, actor, creator or
    puppeteer, as below.
- **Fictional characters** connect through a real photo of their voice actor, creator or puppeteer, and the
  line says so ("Grug is voiced by Nicolas Cage"). The character itself appears only in its own artwork
  (official art or a film still, fair use), as the endpoint link: `{"a": "Grug", "b": "Nicolas Cage", "image": <a
  still or poster showing Grug>, "line": "Grug is voiced by Nicolas Cage."}`. Pin Grug's face in the art. If no
  picture shows both, use a `pair` image: the character art next to a real photo of the actor.
- **Name tags use the name people know.** Every real person is labelled with the name the audience
  recognises: THE ROCK (not Dwayne Johnson), POPE FRANCIS (not Jorge Bergoglio), DIDDY, LIL WAYNE, MATPAT. Set it
  with `"people": {"Dwayne Johnson": {"label": "The Rock"}}`; the key stays the Wikipedia name so faces still
  match. The header ("A > B") shows labels up to 15 letters in full; for a longer one set a short form, e.g.
  `"SpongeBob SquarePants": {"short": "SpongeBob"}`. A character's name only ever labels the character's own artwork, never the actor in a real photo.
- **No limit on the number of links.** A Short can be up to 3 minutes, and QA fails only past 3:00 (it warns past
  60 s). Shorter is better for retention, so prefer the most surprising chain with tight lines.
- Never invent a meeting, a photo or a caption.

## 2b. Finding pictures: try hard, skipping is the last resort
Viewers asked for these pairs, so a skip is a failure. "Commons has nothing" is not enough. Before giving up on a
link, work through **every** step below and keep a note of what you tried.
1. **Wikipedia and Commons, deeply.** Look through all images in both people's articles, every "Category:<Name>"
   subcategory on Commons (events, years, "with <X>"), and the `depicts` (P180) search
   (`haswbstatement:P180=<Qid>`). Check the event categories too, such as premieres, award shows, Met Gala, Comic-Con,
   WWE events, state visits, sports finals and summits.
2. **Public-domain archives.** White House and other government Flickr accounts, NARA, the Library of Congress,
   DVIDS (military), NASA, and Kremlin.ru (CC BY 4.0). Presidents, generals and visiting celebrities show up here.
3. **Free photo hosts.** Flickr CC-licensed photos (search the two names together), fan convention photos and
   press-agency CC releases.
4. **Fair use (commentary) press photos.** These are allowed: Getty, AP, Reuters, AFP, WireImage, news articles, and a
   verified Instagram or X post by either person. Also allowed: a talk-show, podcast, interview or game-broadcast
   frame showing both people, such as a YouTube thumbnail or still from the official channel. Use
   `{"url", "credit", "license": "Fair use (commentary)", "source"}`, with `source` set to the page the image came from.
5. **Change the route.** If one link is stuck, connect through someone else. Good hubs:
   - US presidents and first ladies
   - popes and Queen Elizabeth II
   - talk-show hosts (Fallon, Kimmel, Oprah, Letterman)
   - Snoop Dogg, Kim Kardashian, MrBeast, Elon Musk
   - footballers at the Ballon d'Or, WWE stars
   - for long time gaps, lineage links (popes, emperors, kings)

   Run `python -m sixdeg search` again with different endpoints, and web search "<A> and <B> photo",
   "<A> meets", "<A> with" and "<A> backstage".
6. **Fictional characters.** Use a real photo of the voice actor, creator or puppeteer with anyone notable, such as
   a convention panel, premiere or award show. Their co-stars and directors count too.

Only after all six steps may a pair be `"skipped"`. Its `reason` must list what you searched at each step, so the
operator can see the effort.
- Keep it family-safe: no gore, nudity or slurs on screen. Historical figures are presented as history.

## 3. episode.json
```json
{"from": "A", "to": "B", "requested_by": "@handle (if any)",
 "hook": "How is A connected to B?",
 "title": "How is A connected to B?",
 "links": [{"a": "A", "b": "X", "image": {"file": "File:....jpg"}, "line": "In 1970, A met X at ..."}]}
```
- `hook` is always the question "How is <A> connected to <B>?"
- Each `line` is one sentence of about 18 words or fewer, starting with a year or a hook word, with a concrete fact.
- Don't type years as words. The voice reads "1973" as "nineteen seventy-three" automatically.
- **Pronunciation.** The narrator is Gemini's Achird voice, which reads names from context. If a name comes out
  wrong in the PR's render, rephrase the line or write the name the way it sounds; regnal numbers and initials
  are handled automatically ("Henry VIII" is read "Henry the Eighth").

## 4. Verify faces by eye
```bash
python -m sixdeg prep <slug>
```
Open **every** `episodes/<slug>/work/review<i>.jpg` with the Read tool. Green is A and blue is B. For a lineage
link, the sheet shows every portrait with its name. Fix wrong or unverified faces with `"a_face": <index>` or
`"b_face": [x, y, w, h]`, then run prep again. Auto-matching does make mistakes, such as swapped people, a poster
face, or a stranger in the background.

## 5. Preview and QA
```bash
python -m sixdeg render <slug> && python -m sixdeg qa <slug>
```
Narration is always Gemini's **Achird** voice; there is no other voice. Your session has no Gemini key, so the
local render has silent narration and QA fails with exactly one line, "narration is not the Achird voice".
That one is expected locally; every other check must pass. Open `out/contact.jpg` and check that tags sit on the
right heads and nothing is clipped. The PR's `render` check on GitHub voices it with Achird and must pass fully.
If it fails only because the daily Gemini quota ran out, say so in the PR; the operator re-runs it the next day.

## 6. Hand off with a PR (never push to main)
1. Set the pair to `"done"` (with `"slug"`) or `"skipped"` (with `"reason"`) in `data/pairs.json`.
2. Commit only `episodes/<slug>/episode.json` and `data/pairs.json`. Videos are never committed.
3. `git push -u origin claude/ep-<slug>` and open a PR to `main` titled `Episode: <A> to <B>`.
4. The PR's `render` check renders and runs QA on GitHub. If it fails, fix the episode on your branch and push again.
5. The operator session merges PRs whose check passed. The merge renders the episode again and puts it on the
   upload queue, and `upload.yml` posts it in the next slot.

## Reference
- Pipeline: `sixdeg/` (wiki/graph search, faces, audio, render_pixel, qa, kit, queue, upload, stats).
- Style and policy: `docs/STYLE.md`. Uploading: `docs/youtube-setup.md`.
