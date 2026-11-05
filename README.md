# sixdegrees

Fully automated "Six Degrees" YouTube Shorts: pick two people who could not be
further apart, find a chain of **real photos or paintings** that connects them, and render a
narrated 1080×1920 Short.

```bash
pip install -r requirements.txt
python -m sixdeg setup                         # local models: faces + voice
python -m sixdeg search "Adolf Hitler" "Neil Armstrong"
python -m sixdeg make <slug>                   # prep + render + qa + upload kit
```

| Step | Module | What it does |
|---|---|---|
| Chain search | `sixdeg/graph.py`, `wiki.py` | two-way BFS over Wikipedia photo captions ("X with Y" → edge) |
| Verify | `sixdeg/faces.py`, `episode.py` | YuNet face detection + SFace identity match against each person's portrait; review sheets |
| Voice | `sixdeg/audio.py` | local Kokoro TTS, Whisper word timings, procedural music + SFX, two-pass loudnorm |
| Render | `sixdeg/render.py` | hook → one pan per photo (A's face → both → B's face) with name tags and a chain bar → recap |
| QA | `sixdeg/qa.py` | faces verified, format, −14 LUFS, every spoken name heard in a Whisper transcript, credits |
| Kit | `sixdeg/kit.py` | title, description with photo credits, tags |

No paid AI API is involved. A scheduled Claude Code session does the research and
the visual checks by following [`docs/RUNBOOK.md`](docs/RUNBOOK.md). Style,
monetization and upload policy are in [`docs/STYLE.md`](docs/STYLE.md).

Older experiments (Gemini chain finder, BFS, the people graph) are in `archive/`.
