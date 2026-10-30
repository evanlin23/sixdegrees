"""Command line: `python -m sixdegrees find "Person A" "Person B"` and friends."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .util import load_dotenv_if_present, log

REPO_GRAPH = Path(__file__).resolve().parent.parent / "createdb" / "met_people_graph.gexf"

EPILOG = """examples:
  python -m sixdegrees find "Dick Durbin" "Michelle Obama"
  python -m sixdegrees find "Pope Francis" "Johnny Depp" --planner graph
  python -m sixdegrees chain "Dick Durbin" "Barack Obama" "Michelle Obama"
  python -m sixdegrees render output/Dick_Durbin__to__Michelle_Obama/chain.json --music song.mp3
  python -m sixdegrees check photo.jpg "Barack Obama" "Michelle Obama"
"""


def _search_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("search and verification")
    g.add_argument("--out", type=Path, default=Path("output"), help="working folder (default: ./output)")
    g.add_argument("--sources", default="auto",
                   help="image sources: auto, or a comma list of commons,brave,serpapi,bing (default: auto)")
    g.add_argument("--images-per-link", type=int, default=12, help="candidate photos to check per link")
    g.add_argument("--threshold", type=float, default=None, help="face match threshold (default 0.40)")
    g.add_argument("--max-repairs", type=int, default=6, help="times to re-plan around a failed link")
    g.add_argument("--max-links", type=int, default=10)
    g.add_argument("--no-video", action="store_true", help="stop after building chain.json")
    g.add_argument("--offline", action="store_true",
                   help="no network: only use photos already in --out (references/ and the link folders)")


def _video_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("video")
    g.add_argument("--size", default="1080x1920", help="WIDTHxHEIGHT (default 1080x1920; 720x1280 is faster)")
    g.add_argument("--fps", type=int, default=60)
    g.add_argument("--speed", type=float, default=1.0, help=">1 is faster, <1 slower")
    g.add_argument("--transition", choices=("distance", "fade"), default="distance")
    g.add_argument("--easing", choices=("smooth", "linear"), default="smooth",
                   help="linear matches the multi-animator exactly")
    g.add_argument("--face-width", type=float, default=0.5, help="close-up face width as a fraction of the frame")
    g.add_argument("--title", default=None, help='overlay text (default "How is A connected to B?"; "" for none)')
    g.add_argument("--no-labels", action="store_true", help="don't write names under faces")
    g.add_argument("--no-level-eyes", action="store_true", help="don't rotate close-ups to level the eyes")
    g.add_argument("--background", choices=("blur", "black"), default="blur")
    g.add_argument("--music", type=Path, default=None, help="audio file to add (looped/trimmed to fit)")
    g.add_argument("--output", type=Path, default=None, help="video path (default: next to chain.json)")


def _planner_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("planning")
    g.add_argument("--planner", choices=("auto", "gemini", "graph"), default="auto",
                   help="auto = Gemini if an API key is set, else the createdb graph")
    g.add_argument("--model", default=None, help="Gemini model (default: gemini-flash-latest or $GEMINI_MODEL)")
    g.add_argument("--no-grounding", action="store_true", help="don't let Gemini use Google Search")
    g.add_argument("--graph", type=Path, default=REPO_GRAPH, help="GEXF graph for --planner graph")
    g.add_argument("--fresh", action="store_true", help="ignore cached Gemini answers")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sixdegrees", description="Connect two people through real photos "
                                     "and render a zoom-pan-zoom video.", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("find", help="find a photo chain between two people and render it")
    p.add_argument("start")
    p.add_argument("target")
    _planner_args(p)
    _search_args(p)
    _video_args(p)

    p = sub.add_parser("chain", help="use a chain you already know: A B C ... (no planner needed)")
    p.add_argument("people", nargs="+")
    p.add_argument("--repair-with", choices=("none", "gemini", "graph"), default="none",
                   help="let a planner route around links that have no usable photo")
    _planner_args(p)
    _search_args(p)
    _video_args(p)

    p = sub.add_parser("render", help="(re-)render a video from a chain.json you may have edited")
    p.add_argument("chain_json", type=Path)
    p.add_argument("--out", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--offline", action="store_true", help="don't download reference photos")
    _video_args(p)

    p = sub.add_parser("check", help="check whether a photo shows two people (debugging)")
    p.add_argument("image", type=Path)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--out", type=Path, default=Path("output"))
    p.add_argument("--sources", default="auto")
    p.add_argument("--offline", action="store_true")
    return parser


def _style(args):
    from .video import Style

    try:
        w, h = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--size must look like 1080x1920, got {args.size!r}")
    return Style(width=w - w % 2, height=h - h % 2, fps=args.fps, speed=args.speed, transition_kind=args.transition,
                 easing=args.easing, face_width=args.face_width, title=args.title, labels=not args.no_labels,
                 level_eyes=not args.no_level_eyes, background=args.background)


def _make_planner(kind: str, args):
    from .planners import GeminiPlanner, GraphPlanner, PlannerError

    if kind == "auto":
        import os

        kind = "gemini" if (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")) else "graph"
        if kind == "graph" and not Path(args.graph).exists():
            raise SystemExit("No planner available: set GEMINI_API_KEY (or GOOGLE_API_KEY), pass --graph, "
                             "or give the chain yourself with `sixdegrees chain A B C`.")
        log.info("Planner: %s", kind)
    if kind == "gemini":
        try:
            return GeminiPlanner(model=args.model, grounding=not args.no_grounding, use_cache=not args.fresh)
        except PlannerError as e:
            raise SystemExit(str(e))
    if kind == "graph":
        if not Path(args.graph).exists():
            raise SystemExit(f"Graph file not found: {args.graph}")
        return GraphPlanner(args.graph)
    return None


def main(argv=None) -> int:
    load_dotenv_if_present()
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    for noisy in ("urllib3", "icrawler", "httpx", "google_genai", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from .pipeline import Options, Pipeline
    from .planners import ManualPlanner, PlannerError

    if args.cmd == "render":
        pipe = Pipeline(Options(out_dir=args.chain_json.parent.parent, offline=args.offline))
        pipe.render(args.chain_json, _style(args), out=args.output, music=args.music)
        return 0

    if args.cmd == "check":
        return _check(args)

    opts = Options(out_dir=args.out, sources=args.sources, images_per_link=args.images_per_link,
                   max_repairs=args.max_repairs, max_links=args.max_links, offline=args.offline)
    if args.threshold is not None:
        opts.threshold = args.threshold
    pipe = Pipeline(opts)
    if args.cmd == "find":
        planner = _make_planner(args.planner, args)
        start, target = args.start, args.target
    else:
        fallback = _make_planner(args.repair_with, args) if args.repair_with != "none" else None
        planner = ManualPlanner(args.people, fallback)
        start, target = args.people[0], args.people[-1]
    try:
        chain_path = pipe.run(start, target, planner)
    except PlannerError as e:
        log.error(str(e))
        return 1
    log.info("Chain details: %s", chain_path)
    if args.no_video:
        return 0
    shots, data = pipe.shots(chain_path)
    if not shots:
        log.error("No verified photos, so no video. See %s for what failed.", chain_path)
        return 1
    if not data.get("complete"):
        log.warning("Rendering the partial chain (%d photos).", len(shots))
    pipe.render(chain_path, _style(args), out=args.output, music=args.music)
    return 0 if data.get("complete") else 2


def _check(args) -> int:
    from .faces import looks_like_collage, match_pair
    from .pipeline import Options, Pipeline
    from .util import load_image

    pipe = Pipeline(Options(out_dir=args.out, sources=args.sources, offline=args.offline))
    faces = pipe.engine.faces_in(args.image)
    print(f"{len(faces)} face(s) found in {args.image}")
    m = match_pair(faces, pipe.identity(args.a), pipe.identity(args.b))
    if m is None:
        print("Need at least two faces.")
        return 1
    collage = looks_like_collage(load_image(args.image), m.a, m.b)
    print(f"{args.a}: similarity {m.sim_a:.3f} at box {[round(v) for v in (m.a.x, m.a.y, m.a.w, m.a.h)]}")
    print(f"{args.b}: similarity {m.sim_b:.3f} at box {[round(v) for v in (m.b.x, m.b.y, m.b.w, m.b.h)]}")
    print(f"collage: {collage}   verdict: {'OK' if m.ok(pipe.opts.threshold) and not collage else 'NOT OK'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
