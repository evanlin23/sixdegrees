"""CLI:  python -m sixdeg <command> ...

  setup                          download local models
  search "<A>" "<B>" [--expand N] candidate photo chains from Wikipedia captions
  neighbors "<Person>"           people photographed with Person (with files + captions)
  photos "<Person>"              every captioned image on Person's article
  prep <slug>                    download photos, detect + verify faces, write work/review*.jpg
  render <slug>                  build episodes/<slug>/out/<slug>.mp4
  qa <slug>                      gates; writes out/qa.json + out/contact.jpg
  kit <slug>                     upload kit: out/upload.md (title, description, credits, tags)
  make <slug>                    prep + render + qa + kit
  upload <slug> [--dry-run]      schedule it on YouTube in the next free slot (private until publish time)
  enqueue <slug>                 producer: put the finished video on the `queue` branch
  requeue [slug...]              replace queued videos with their latest QA-passed render
  stats                          views/likes/comments (+ retention if the token allows) for every upload
  consume [--max N] [--dry-run]  consumer (GitHub Actions): upload the oldest N queued videos
"""
import json
import os
import sys


def render_any(slug):
    """Render in the channel style from data/style.json (pixel by default; "classic" = the original look)."""
    from .common import DATA
    style = json.loads((DATA / "style.json").read_text()).get("style", "pixel") if (DATA / "style.json").exists() else "pixel"
    try:  # English name pronunciations from Wikipedia: only the Kokoro voice reads them (Achird doesn't)
        from .audio import voice_config
        from .episode import load, people_order
        from .pronounce import update
        if voice_config().get("engine", "kokoro") == "kokoro":
            print("pronunciations:", update(people_order(load(slug)[1])))
    except Exception as e:
        print("pronunciations: skipped", e)
    if style == "pixel":
        from .render_pixel import render
    else:
        from .render import render
    return render(slug)


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, args = argv[0], argv[1:]
    if cmd == "setup":
        from .setup_models import main as m
        m()
    elif cmd == "search":
        from . import graph
        expand = 80
        if "--expand" in args:
            j = args.index("--expand"); expand = int(args[j + 1]); args = args[:j] + args[j + 2:]
        paths = graph.search(args[0], args[1], max_expand=expand)
        if not paths:
            print("no photo chain found within budget; research manually (see docs/RUNBOOK.md)")
        for p in paths:
            print(graph.describe(p)); print()
    elif cmd == "neighbors":
        from . import graph
        for e in graph.neighbors(args[0], cap=100):
            print(f"{e['b']:35s} [{e['score']}] {e['file']}\n    {e['caption'][:200]}")
    elif cmd == "photos":
        from . import wiki
        r = wiki.resolve(args[0])
        for im in wiki.article_images(r["title"]):
            print(f"{im['file']}\n    {im['caption'][:200]}\n    links: {im['links']}")
    elif cmd == "prep":
        from .episode import prep
        prep(args[0])
    elif cmd == "pronounce":  # pull Wikipedia pronunciations for everyone in the episode
        from .episode import load, people_order
        from .pronounce import update
        print(json.dumps(update(people_order(load(argv[1])[1])), indent=1, ensure_ascii=False))
    elif cmd == "research":  # competitor channel report via the YouTube API
        from .research import channel_report
        for h in args:
            channel_report(h)
    elif cmd == "lineage":  # every holder of an office with portraits: lineage popes --from "Saint Peter" --to "Pope Francis"
        from .lineage import main as lineage_main
        print(lineage_main(args))
    elif cmd == "studio":  # import a YouTube Studio metrics snapshot (docs/studio-metrics-prompt.md) and compare groups
        from .studio import run
        print(run(args[0]))
    elif cmd == "resplit":  # re-cut saved raw Gemini recordings with the current splitter (free)
        from . import audio
        print("resplit lines:", audio.resplit_all())
    elif cmd == "voice":  # Gemini: voice several episodes, a few per request (free tier ~15 requests/day, 3/min)
        import time
        from . import audio
        from .episode import load
        per = int(os.environ.get("VOICE_BATCH", "4"))
        slugs = args
        for i in range(0, len(slugs), per):
            lines = []
            for sl in slugs[i:i + per]:
                sp = load(sl)[1]
                lines += [sp.get("hook", "")] + [l["line"] for l in sp["links"]]
            if i == 0:
                lines.append("And that's how.")   # the A/B outro line, shared by every episode: voice it once
            audio.prefetch([l for l in lines if l])
            time.sleep(21)
    elif cmd == "render":
        render_any(args[0])
    elif cmd == "qa":
        from .qa import qa
        r = qa(args[0])
        print(json.dumps(r, indent=1))
        return 0 if r["pass"] else 1
    elif cmd == "kit":
        from .kit import kit
        print(kit(args[0]))
    elif cmd == "upload":
        from .upload import upload
        print(json.dumps(upload(args[0], dry_run="--dry-run" in args), indent=1))
    elif cmd == "enqueue":
        from .queue import enqueue
        print(enqueue(args[0]))
    elif cmd == "requeue":
        from .queue import requeue
        print(requeue(args or None))
    elif cmd == "stats":
        from .stats import stats
        print(stats())
    elif cmd == "consume":
        from .queue import consume
        mx = int(args[args.index("--max") + 1]) if "--max" in args else 2
        consume(mx, dry_run="--dry-run" in args)
    elif cmd == "make":
        from .episode import prep
        from .kit import kit
        from .qa import qa
        prep(args[0]); render_any(args[0])
        r = qa(args[0]); print(json.dumps(r, indent=1))
        print(kit(args[0]))
        return 0 if r["pass"] else 1
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
