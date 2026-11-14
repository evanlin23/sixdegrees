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
episode.json: the hook shows the viewer's request bubble and the first comment credits them. Use their real
@handle (never "a viewer" when the handle is known) and, when the request came from a YouTube comment, also set
`"request_avatar"` to the commenter's profile-picture URL (comments.json `avatar`, already 240 px) so the bubble shows their PFP.
Requests live on the `comments` branch (`comments.json`: every upload's top comments and their replies, likely requests
flagged `"request": true`). For a manual top-up from Studio, use `docs/comments-prompt.md` in Claude in Chrome.
**Viewer shortcuts:** comments flagged `"shortcut": true` point out shorter routes ("You could've ended it at Nixon. Trump and Nixon
met."). Before you research a chain, read the ones about the same people or hubs, and verify each with a real photo. A verified
shortcut beats a longer route. Viewers love short chains: "How is the chain so short" is our most-liked comment (117).

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
  The first and last names must equal `a` and `b`. Include **every** holder in between (generate the list with
  `python -m sixdeg lineage`, see "Lineage links" below); it scrolls past in a few seconds. It counts as one link.
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
- **Take the first honest path you find; never pad it (creator's call, 2026-11-12).** Viewers like short paths.
  Don't detour or add famous faces just to make the chain longer, and don't spend ages hunting for an even shorter
  one either: the first route where every link has a real photo is fine.
  - **Every link still needs a real photo of both people together** (or a lineage scroll). Never invent one.
  - **Hide the destination where the route allows.** Nothing from B's world before the last 1–2 links.
  - **Even pace to the end; the last link is the payoff (owner, 2026-11-12).** Never speed up: every link gets
    the same beat on screen (the renderer holds each link at least 3.4 s). Every link but the last gets a context
    line, where/when and how the two met, in about 9-13 words (see "Write like Steve and Miku") ("At the 2012 Easter Egg Roll, Michelle Obama played
    soccer with SpongeBob himself."). No 3-word sprint lines like "Who voices Shrek." mid-chain. The LAST line
    explains less, because the link itself is the payoff: short and on the meeting ("At Late Night with Seth Meyers,
    Ash Carter met Kevin Hart.", "Jon Favreau created Baby Yoda."), never a paragraph after B's name.
  - **Write like Steve and Miku (owner, 2026-11-13: "I really like the structure and pacing and sentences of the Miku
    and Steve ones").** The model scripts:
      In 1938, Hitler signed the Munich deal with Britain's Neville Chamberlain.
      In 1940, Winston Churchill replaced Chamberlain as prime minister.
      In 1953, Churchill greeted the new Queen and her children.
      In 2014, the Queen made Angelina Jolie an honorary dame.
      In 2008, Jolie and Jack Black brought Kung Fu Panda to Cannes.
      And in 2025, Jack Black became Steve in A Minecraft Movie.
    and Miku: "In 1937, Hitler welcomed Britain's former king, Edward VIII." / "That year, Nixon picked a young George
    Bush as his ambassador to the United Nations." / "Back in 2012, Obama sat down with David Letterman on the Late
    Show." / "And in 2014, Letterman shared his stage with Hatsune Miku, live as a hologram."
    Rules that come from them (these override the older "8-16 words" wording):
      - 9-13 words a line. ONE fact per line: who met whom, and the one detail that makes it fun.
      - Plain descriptors over titles: "Britain's former king", "the new Queen", "a young George Bush". Full name the
        first time someone appears, then the short name ("Chamberlain", "Jolie").
      - No stacked appositives or extra places ("Vichy France's leader, Marshal Pétain, at Montoire's train station").
      - The year leads every line, but natural connectors count: "That year, …", "Back in 2012, …", "Two years later, …".
      - Each line hands off: the new person of one line is the subject of the next.
      - Last line: short, ends on B, and varied. Not "…himself." every time ("became Steve in A Minecraft Movie",
        "shared his stage with Hatsune Miku, live as a hologram").
  - **The opening shows the most recognisable photo of A and of B (owner, 2026-11-13).** Not a crop from a chain
    photo: set `people.<name>.hook_photo` to an iconic, frontal, instantly recognisable image (a famous portrait, a
    character's official still). For characters and puppets without a human face, the renderer uses the hook photo
    centred; add `people.<name>.hook_box: [x, y, w, h]` (pixels in that photo) to frame the face exactly.
  - **Opening A/B tests (owner, 2026-11-13; the arms are assigned by the orchestrators, not by producers).**
    - `"hook_variant"` names the spoken-hook arm: A "How's Hitler linked to B?", B "What links Hitler to B?", F
      (baseline) "How is Hitler connected to B?". There are no pauses in the hook, it's always a question that
      implies the yes, and it says "Hitler", not "Adolf Hitler".
    - `"text_variant": "T1"` switches on the new opening: one static bold line "HITLER → B?" in a light band between
      the faces from frame 0, no chin tags, B's face bigger, and a brighter grade that keeps the blacks.
    - `"T0"` (or no field) keeps today's opening. Requested pairs always keep T0, because the viewer's comment card is
      their opening.
    - Both arms are recorded in metadata.json and uploaded.json, and the studio report compares them.
  - **No dead air, ever (owner, 2026-11-13).** The question starts on frame 0, the voice never stops for more
    than ~2 s, and the video ends ~0.5 s after the last word. No pause after the opening question, and no music-only
    tail after the payoff ("it says 'opening night' and then pop, pop, pop for 3 seconds"). The renderer enforces the
    timing; QA fails a render with dead air (first word > 0.25 s, a gap > 2 s, or > 1 s after the last word). Write
    lines so this holds: year first ("In 1931, …"), the final line ends on B's name, and keep lines long enough to
    fill their link (a lineage scroll's line should run about as long as the scroll).
  - **Real people link only through real people (owner, 2026-11-14).**
    - The owner's words:
      - "since Caesar, Cleopatra, Augustus were all like real people, we should just use real people"; doing it
        through actors "feels like cheating".
      - "we should only use like voice actor or movie if its like a fictional character".
    - So a historical figure (Caesar, Cleopatra, Augustus, Genghis Khan, Henry VIII, Jesus) never links through an
      actor who played them. Use real artworks or objects that show both people together (paintings of a real
      meeting, coins or cameos with both portraits, manuscripts), or a lineage scroll of 40 or more holders.
    - A `pair` of two separate portraits is not a link between two real people.
  - **Myths and gods: bridge through real people in the myth (owner, 2026-11-14).** The owner's words: "even stuff with
    mythology (ex like gods) we can try our best to use bridge characters in the myths that also existed in real
    life."
    - Find a real person who appears in the god's story or alongside the god in art. Examples:
      - Alexander the Great, hailed as the son of Zeus-Ammon, on his coins with Zeus;
      - Augustus among the gods on the Gemma Augustea;
      - Henry IV with Jupiter in Rubens;
      - Washington with Minerva in Brumidi's *Apotheosis*.
    - Continue from that real person.
    - Never use an actor who played the god.
  - **Use a big lineage when it's the natural route** (e.g. Trump back to George Washington through every
    president), per the lineage rule below.
- **Lineage scrolls only for HUGE lineages** (creator's call: "only if it's a lot, like the popes"): roughly
  40+ holders (popes, Chinese/Roman emperors, English/French monarchs). Never for short lists like German
  chancellors or most presidents' lists under ~40; use a real photo link instead. When used, show EVERYONE. A line of popes, presidents or emperors scrolls past super fast with every
    holder, not a hand-picked few. Generate the list with
    `python -m sixdeg lineage popes --from "Saint Peter" --to "Pope Francis"` (also `us-presidents`,
    `roman-emperors`, `uk-monarchs`, `english-monarchs`, `french-monarchs`, or a Wikidata QID) and paste its
    `{"label", "people"}` into the link's `"lineage"`. Rename the first/last `name` to match the link's `a`/`b` if
    they're spelled differently. Check the presidents list by eye: Wikidata has some noise there.
- **Fewer links is better.** QA warns past 60 s and fails past 3:00.
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
5. **Network of Time** (networkoftime.substack.com): a hand-verified graph of 11,000+ photos of 15,000+ people,
   built to find exactly these chains. Use it to discover routes, then source each photo yourself (Commons or a
   press photo) and credit it.
6. **Change the route.** If one link is stuck, connect through someone else. Good hubs:
   - US presidents and first ladies
   - popes and Queen Elizabeth II
   - talk-show hosts (Fallon, Kimmel, Oprah, Letterman)
   - Snoop Dogg, Kim Kardashian, MrBeast, Elon Musk
   - footballers at the Ballon d'Or, WWE stars
   - for long time gaps, lineage links (popes, emperors, kings)

   Run `python -m sixdeg search` again with different endpoints, and web search "<A> and <B> photo",
   "<A> meets", "<A> with" and "<A> backstage".
7. **Fictional characters.** Use a real photo of the voice actor, creator or puppeteer with anyone notable, such as
   a convention panel, premiere or award show. Their co-stars and directors count too.

Only after all seven steps may a pair be `"skipped"`. Its `reason` must list what you searched at each step, so the
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
- **Visible research (monetization, docs/MONETIZATION_PLAN.md):** the research is what makes each video original,
  so put it on screen and in the description:
  - each link may have `"fact"`: 3–7 words shown on screen in music-only videos ("Berghof, 1937", "Late Show
    stage, 2014"). Without it the renderer uses the year plus the first words of `line`; write one when that would
    read badly.
  - `"fun_fact"` (top level): one surprising true detail you found while researching, which opens the first comment.
  - `"research_note"` (top level): one sentence on how the chain was found or what dead ends you hit
    ("Every Scooby costume photo was watermarked, so Scooby comes in through the 2002 film's cast."). It goes in
    the description.
  - The description lists every link's line and photo source automatically.
- **Where each photo was taken (for the globe style):** give every link
  `"where": {"place": "White House", "lat": 38.897, "lon": -77.036}`, the place the photo was taken (a city or
  venue is enough). Videos rotate through presentation styles (`data/style.json`), and the globe style is only used
  when every link has `where`. You can force a style with top-level `"style"`, e.g. `"style": "board"`. Test a
  style locally without editing the episode: `SIXDEG_STYLE=board python -m sixdeg render <slug>`.
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
right heads and nothing is clipped. Then check the opening, which decides whether viewers stay:
`ffmpeg -ss 0.5 -i out/<slug>.mp4 -frames:v 1 out/hook.png` and look at it. A and B fill the screen as big
full-width photos, cropped from each person's first portrait (`refs[0]`) or their boxed face in a link photo.
Both faces must be sharp, frontal, instantly recognisable and the right person. If one is blurry, tiny, side-on
or wrong, set a better one (large, face-on) with `"people": {"<Name>": {"hook_photo": "File:<name>.jpg"}}` and run
prep again. The PR's `render` check on GitHub voices it with Achird and must pass fully.
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
