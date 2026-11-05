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
"""
import json
import sys


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
    elif cmd == "render":
        from .render import render
        render(args[0])
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
    elif cmd == "make":
        from .episode import prep
        from .kit import kit
        from .qa import qa
        from .render import render
        prep(args[0]); render(args[0])
        r = qa(args[0]); print(json.dumps(r, indent=1))
        print(kit(args[0]))
        return 0 if r["pass"] else 1
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
