# sixdegrees

Connect two people through a chain of **real photos of people together** and turn it into a
9:16 video. Each photo zooms in on the person carried over from the previous photo, zooms out,
pans to the next person, zooms back in, and transitions to the next photo, where the same face is
already in the same spot. It's the same zoom-pan-zoom edit as the
[ioctl multi-image animator](https://evanlin23.github.io/ioctl/multi-animator), except the points
are found automatically by face detection, and the names are written in for you.

```
Dick Durbin ──photo──▶ Barack Obama ──photo──▶ Michelle Obama
```

## Setup

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

`find` uses Claude to plan chains. Either put `ANTHROPIC_API_KEY=...` in `.env`, or run
`ant auth login` once and leave the key unset. (Gemini still works with `--planner gemini` and
`GEMINI_API_KEY`.) ffmpeg is used if installed, and the pip-installed one is used otherwise. The face
models (~40 MB, OpenCV YuNet + SFace) download automatically on the first run.

On Python 3.9, pip installs the 0.125 Anthropic SDK (the last release for 3.9), which this tool
supports. Python 3.10+ gets the current SDK.

## Usage

Find a chain and make the video:

```bash
python -m sixdegrees find "Dick Durbin" "Michelle Obama"
```

Already know the chain? Skip the AI:

```bash
python -m sixdegrees chain "Dick Durbin" "Barack Obama" "Michelle Obama"
```

No API key? Use the pre-built graph in `createdb/` to propose chains (it's noisy; every link still
has to pass the photo check, and failed links are routed around):

```bash
python -m sixdegrees find "Pope Francis" "Johnny Depp" --planner graph
```

Re-render after editing `chain.json` (swap a photo, fix a face), optionally with music:

```bash
python -m sixdegrees render output/Dick_Durbin__to__Michelle_Obama/chain.json --music song.mp3
```

Check one photo:

```bash
python -m sixdegrees check photo.jpg "Barack Obama" "Michelle Obama"
```

Useful options (`python -m sixdegrees find -h` for all):

| option | what it does |
| --- | --- |
| `--size 720x1280` | output size (default 1080x1920); smaller renders faster |
| `--fps 30` | default 60, like the multi-animator |
| `--speed 1.5` | faster pacing (all timings divided by 1.5) |
| `--transition fade` | plain crossfade instead of ffmpeg's `distance` transition |
| `--easing linear` | constant-speed moves exactly like the multi-animator (default eases in/out) |
| `--title ""` | no title (default "How is A connected to B?") |
| `--no-labels` | don't write names under faces |
| `--sources commons` | which image sources to use (see below) |
| `--model claude-sonnet-5-5` | pick the planner's model (default `claude-opus-5-5`) |
| `--effort medium` | how hard Claude researches each chain: `low` to `max` (default `high`) |
| `--planner gemini` | plan with Gemini instead of Claude (`graph` needs no API key) |
| `--no-video` | stop after building `chain.json` |

## How it works

1. **Plan.** Claude (Opus 5.5, with web search) proposes a chain where every pair was
   *photographed together*, with the occasion and a search query for each link. It returns the
   chain through a strict `submit_chain` tool, so the answer is always well-formed. Refusal fallback
   is on: if a request trips a safety classifier, the API re-runs it on Anthropic's recommended
   fallback model instead of failing. Answers are cached, so re-running the same pair is free
   (`--fresh` to ask again).
2. **Who is who.** Each person is looked up on Wikipedia. Their lead image plus a few more photos
   become reference faces. The face that recurs across those photos is taken as the person, so a
   stray wrong search result doesn't poison it.
3. **Prove each link.** Candidate photos come from Wikimedia Commons (including photos tagged as
   depicting both people) and the other sources. A photo counts only if two *different* faces
   match the two people and it isn't a side-by-side collage. The best one wins: confident matches,
   big faces, few bystanders.
4. **Repair.** If no photo proves a link, the planner is told what failed and asked for another
   route from the last proven person. This repeats up to `--max-repairs` times.
5. **Render.** OpenCV draws every frame (zoom normalized to face size, eyes levelled so the shared
   face lines up across the cut, names under faces) and pipes them to ffmpeg as H.264.

## Output and fixing things by hand

```
output/
  references/<Person>/          reference photos (add your own if a face isn't recognized)
  <A>__to__<Z>/
    chain.json                  the chain: chosen photo + face boxes per link, failures, notes
    video.mp4
    storyboard.jpg              every chosen photo with the matched faces boxed: check this first
    links/<A>__<B>/             candidate photos for one link, plus matches.json with scores
```

- **Wrong or ugly photo:** set that link's `"image"` in `chain.json` to another file (see
  `"alternatives"`, or drop your own photo into the link folder), then run `render`.
- **Face missed or mismatched:** add `"a_point": [x, y]` / `"b_point": [x, y]` (pixel coords of
  the face center) to the link in `chain.json`, like clicking the points in the multi-animator.
- **Person not recognized:** put a few clear photos of them in `output/references/<Person>/`.
- **Have photos but no internet:** put them in the link folders and run with `--offline`.

## Image sources

`--sources auto` (default) uses everything available:

- `commons`: Wikimedia Commons API. Free, no key, great for politicians, royals, athletes.
- `brave` / `serpapi`: image search APIs, used when `BRAVE_API_KEY` / `SERPAPI_API_KEY` is set.
  These are the most reliable way to find celebrity photos.
- `bing`: Bing Images via icrawler. It scrapes the results page, so it can break whenever Bing
  changes the page (the Google crawler the old code used already has).

## Repo layout

- `sixdegrees/`: the tool (`faces.py`, `search.py`, `planners.py`, `pipeline.py`, `video.py`, `cli.py`)
- `tests/`: `python -m pytest tests`
- `createdb/`: builds `met_people_graph.gexf` from the Pantheon dataset (used by `--planner graph`)
- `old_tries/`: earlier attempts, including the previous `full/` and `bfs/` pipelines
