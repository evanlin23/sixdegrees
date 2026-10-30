"""Plan a chain, prove each link with a real photo, then render the video."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .faces import DEFAULT_MATCH_THRESHOLD, Face, FaceEngine, Identity, build_identity, looks_like_collage, match_pair
from .planners import Link, PlannerError
from .search import Person, collect_images, link_dir_name, make_sources, reference_images, resolve_person
from .util import file_hash, load_image, log, read_json, same_person, slugify, write_json


@dataclass
class Options:
    out_dir: Path = Path("output")
    sources: str = "auto"
    images_per_link: int = 12
    refs_per_person: int = 6
    threshold: float = DEFAULT_MATCH_THRESHOLD
    max_repairs: int = 6
    max_links: int = 10
    offline: bool = False  # no network: use only photos already in the output folders


class Pipeline:
    def __init__(self, opts: Options):
        self.opts = opts
        self.out_dir = Path(opts.out_dir)
        self._engine: Optional[FaceEngine] = None
        self._sources = None
        self._people: dict[str, Person] = {}
        self._ids: dict[str, Identity] = {}

    @property
    def engine(self) -> FaceEngine:
        if self._engine is None:
            self._engine = FaceEngine()
        return self._engine

    @property
    def sources(self) -> list:
        if self._sources is None:
            self._sources = [] if self.opts.offline else make_sources(self.opts.sources)
            log.info("Image sources: %s", ", ".join(s.name for s in self._sources) or "none")
        return self._sources

    def person(self, name: str) -> Person:
        key = name.strip().lower()
        if key not in self._people:
            self._people[key] = Person(name.strip(), name.strip()) if self.opts.offline else resolve_person(name.strip())
            p = self._people[key]
            if p.title != p.name:
                log.info("  %s -> %s (Wikipedia)", p.name, p.title)
        return self._people[key]

    def identity(self, name: str) -> Identity:
        p = self.person(name)
        if p.title not in self._ids:
            folder = self.out_dir / "references" / slugify(p.title)
            images, anchor = reference_images(p, folder, self.sources, self.opts.refs_per_person)
            ident = build_identity(p.title, images, self.engine, anchor)
            note = "" if ident.confident else " (low confidence: reference photos disagree; add some to %s)" % folder
            log.info("  %s: face learned from %d of %d reference photos%s", p.title, ident.refs.shape[0], len(images), note)
            self._ids[p.title] = ident
        return self._ids[p.title]

    # ------------------------------------------------------------ links
    def verify_link(self, link: Link, run_dir: Path) -> dict:
        pa, pb = self.person(link.a), self.person(link.b)
        rec = {"a": pa.title, "b": pb.title, "occasion": link.occasion, "year": link.year, "query": link.query,
               "status": "failed", "reason": "", "image": None}
        log.info("Link %s -> %s%s", pa.title, pb.title, f" ({link.occasion}, {link.year})" if link.occasion else "")
        ida, idb = self.identity(pa.title), self.identity(pb.title)
        for ident in (ida, idb):
            if not ident.ok:
                rec["reason"] = f"no reference photos of {ident.name}"
                log.warning("  failed: %s", rec["reason"])
                return rec
        folder = run_dir / "links" / link_dir_name(pa.title, pb.title)
        images = collect_images([pa, pb], link.query or None, folder, self.sources, self.opts.images_per_link)
        candidates = []
        for img in images:
            m = match_pair(self.engine.faces_in(img), ida, idb)
            cand = {"file": _rel(img, run_dir), "faces": 0 if m is None else m.n_faces}
            if m is not None:
                if m.sim_a >= self.opts.threshold and m.sim_b >= self.opts.threshold:
                    m.collage = looks_like_collage(load_image(img), m.a, m.b)
                cand.update(sim_a=round(m.sim_a, 3), sim_b=round(m.sim_b, 3), collage=m.collage,
                            ok=m.ok(self.opts.threshold), score=round(m.score, 3),
                            a_face=m.a.to_dict(), b_face=m.b.to_dict())
            candidates.append(cand)
        candidates.sort(key=lambda c: (c.get("ok", False), c.get("score", -9)), reverse=True)
        write_json(folder / "matches.json", candidates)
        best = candidates[0] if candidates and candidates[0].get("ok") else None
        if best is None:
            rec["reason"] = (f"none of {len(images)} photos show both faces" if images else "no photos found")
            log.warning("  failed: %s", rec["reason"])
            return rec
        img = run_dir / best["file"]
        rec.update(status="ok", image=best["file"], image_hash=file_hash(img), a_face=best["a_face"],
                   b_face=best["b_face"], sim_a=best["sim_a"], sim_b=best["sim_b"],
                   alternatives=[c["file"] for c in candidates[1:6] if c.get("ok")])
        log.info("  ok: %s (match %.2f / %.2f, %d usable of %d photos)", best["file"], best["sim_a"], best["sim_b"],
                 sum(1 for c in candidates if c.get("ok")), len(images))
        return rec

    # ------------------------------------------------------------ chain
    def run(self, start: str, target: str, planner, run_dir: Optional[Path] = None) -> Path:
        start_t, target_t = self.person(start).title, self.person(target).title
        run_dir = Path(run_dir or self.out_dir / f"{slugify(start_t)}__to__{slugify(target_t)}")
        run_dir.mkdir(parents=True, exist_ok=True)
        chain_path = run_dir / "chain.json"
        state = {"start": start_t, "target": target_t, "complete": False, "people": [start_t],
                 "links": [], "failed": [], "message": ""}

        def save(msg: str = ""):
            state["message"] = msg
            write_json(chain_path, state)

        log.info("Planning %s -> %s ...", start_t, target_t)
        try:
            plan = list(planner.plan(start_t, target_t))
        except PlannerError as e:
            save(str(e))
            raise
        log.info("Proposed: %s", " -> ".join([start_t] + [l.b for l in plan]))
        failed: list[tuple[Link, str]] = []
        repairs = 0
        while not same_person(state["people"][-1], target_t):
            current = state["people"][-1]
            if len(state["links"]) >= self.opts.max_links:
                save(f"stopped: chain reached {self.opts.max_links} links")
                break
            link = plan.pop(0) if plan else None
            reason = ""
            if link is None:
                reason = "plan ran out"
            elif not same_person(self.person(link.a).title, current):
                reason = f"plan does not continue from {current}"
            elif any(same_person(self.person(link.b).title, p) for p in state["people"]):
                reason = f"{link.b} is already in the chain"
            else:
                rec = self.verify_link(Link(current, self.person(link.b).title, link.occasion, link.year, link.query), run_dir)
                if rec["status"] == "ok":
                    state["links"].append(rec)
                    state["people"].append(rec["b"])
                    save()
                    continue
                reason = rec["reason"]
                state["failed"].append(rec)
            if link is not None:
                failed.append((link, reason))
            if repairs >= self.opts.max_repairs:
                save(f"stopped after {repairs} repairs; last problem: {reason}")
                break
            repairs += 1
            log.info("Re-planning from %s (%s) ...", current, reason)
            new_plan = planner.repair(list(state["people"]), target_t, failed, plan)
            if not new_plan:
                save(f"stuck at {current}: {reason}; planner has no alternative")
                break
            plan = list(new_plan)
            log.info("New route: %s", " -> ".join([current] + [l.b for l in plan]))
        else:
            state["complete"] = True
            save("complete")
        log.info("%s: %s", "Chain complete" if state["complete"] else "Chain incomplete", " -> ".join(state["people"]))
        if not state["complete"]:
            log.info("  %s", state["message"])
        return chain_path

    # ------------------------------------------------------------ render
    def shots(self, chain_path: Path):
        """Turn chain.json into renderable shots. Edit chain.json to swap photos or fix face points:
        set "image" to another file (faces are re-detected) or give "a_point"/"b_point" as [x, y]."""
        from .video import Shot

        chain_path = Path(chain_path)
        base = chain_path.parent
        data = read_json(chain_path)
        if not data:
            raise FileNotFoundError(chain_path)
        shots = []
        for rec in data.get("links", []):
            if not rec.get("image"):
                continue
            img = (base / rec["image"]) if not Path(rec["image"]).is_absolute() else Path(rec["image"])
            a_face = Face.from_dict(rec["a_face"]) if rec.get("a_face") else None
            b_face = Face.from_dict(rec["b_face"]) if rec.get("b_face") else None
            if rec.get("image_hash") != file_hash(img) or a_face is None or b_face is None:
                m = match_pair(self.engine.faces_in(img), self.identity(rec["a"]), self.identity(rec["b"]))
                if m is not None:
                    a_face, b_face = m.a, m.b
                    rec.update(a_face=m.a.to_dict(), b_face=m.b.to_dict(), image_hash=file_hash(img))
            if rec.get("a_point") or rec.get("b_point"):
                h, w = load_image(img).shape[:2]
                default = 0.15 * min(w, h)
                if rec.get("a_point"):
                    a_face = Face.from_point(*rec["a_point"], size=rec.get("a_size") or default)
                if rec.get("b_point"):
                    b_face = Face.from_point(*rec["b_point"], size=rec.get("b_size") or default)
            if a_face is None or b_face is None:
                raise ValueError(f"Can't find two faces in {img}; add \"a_point\" and \"b_point\" to chain.json.")
            shots.append(Shot(img, rec["a"], rec["b"], a_face, b_face))
        write_json(chain_path, data)
        return shots, data

    def render(self, chain_path: Path, style, out: Optional[Path] = None, music: Optional[Path] = None) -> Path:
        from .video import Renderer, contact_sheet

        shots, data = self.shots(chain_path)
        if not shots:
            raise ValueError("chain.json has no verified photos to render")
        if style.title is None:
            style.title = f"How is {data['start']} connected to {data['target']}?"
        out = Path(out or Path(chain_path).parent / "video.mp4")
        contact_sheet(shots, out.with_name("storyboard.jpg"))
        started = time.time()

        def progress(done: int, total: int):
            if done == total or done % max(1, total // 20) == 0:
                print(f"\r  rendering {100 * done // total:3d}%", end="\n" if done == total else "",
                      file=sys.stderr, flush=True)

        Renderer(style).render(shots, out, music=music, progress=progress)
        log.info("Video: %s (rendered in %.0fs). Storyboard: %s", out, time.time() - started, out.with_name("storyboard.jpg"))
        data = read_json(Path(chain_path))
        data["video"] = _rel(out, Path(chain_path).parent)
        write_json(Path(chain_path), data)
        return out


def _rel(path: Path, base: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return str(Path(path).resolve())
