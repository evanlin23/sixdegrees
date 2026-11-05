"""Episode spec + prep (download photos, detect & verify faces, build review sheets).

episodes/<slug>/episode.json
{
  "from": "Adolf Hitler", "to": "Neil Armstrong",
  "title": "...",                       # YouTube title
  "hook": "spoken first line",
  "outro": "spoken last line",
  "people": {                            # optional per-person overrides
     "Wernher von Braun": {"label": "Wernher von Braun", "refs": ["File:..."]}
  },
  "links": [
    {"a": "Adolf Hitler", "b": "Wernher von Braun",
     "image": {"file": "File:Some photo.jpg"}            # Wikimedia file (credit auto)
        or {"pair": [{"file": "File:Pope A.jpg"}, {"file": "File:Pope B.jpg"}]}   # succession: two portraits side by side
        or {"url": "https://...", "credit": "...", "license": "...", "source": "page url"},
     "line": "spoken narration for this photo",
     "a_face": 2,  "b_face": [x, y, w, h]                 # optional manual pins (index or box)
    }
  ]
}
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import faces, wiki
from .common import EPISODES, FONTS, download, http_get, slugify


def load(slug: str) -> tuple[Path, dict]:
    d = EPISODES / slug
    spec = json.loads((d / "episode.json").read_text())
    return d, spec


def people_order(spec: dict) -> list[str]:
    names = [spec["links"][0]["a"]] + [l["b"] for l in spec["links"]]
    return names


def check_chain(spec: dict) -> list[str]:
    errs = []
    L = spec["links"]
    if not L:
        errs.append("no links")
    for i in range(1, len(L)):
        if L[i]["a"] != L[i - 1]["b"]:
            errs.append(f"link {i}: a={L[i]['a']!r} but previous b={L[i-1]['b']!r}")
    if L and L[0]["a"] != spec["from"]:
        errs.append("first link must start at 'from'")
    if L and L[-1]["b"] != spec["to"]:
        errs.append("last link must end at 'to'")
    for i, l in enumerate(L):
        if not l.get("line"):
            errs.append(f"link {i}: missing narration line")
    return errs


def _fetch_image(img_spec: dict, dest_stem: Path, width: int = 2000) -> dict:
    """Download the image; return credit record."""
    if "file" in img_spec:
        info = wiki.file_info(img_spec["file"], width=width)
        ext = Path(info["thumb"]).suffix.lower()
        if ext in (".tif", ".tiff", ".gif", ".svg", ".webp"):
            ext = ".jpg" if ext != ".png" else ext
        dest = dest_stem.with_suffix(ext if ext in (".jpg", ".jpeg", ".png") else ".jpg")
        download(info["thumb"], dest)
        rec = {"path": dest.name, "source": info["page"], "license": info["license"],
               "artist": clean_artist(info["artist"]), "credit": info["credit"][:200], "file": info["file"],
               "free": not info["nonfree"]}
    else:
        url = img_spec["url"]
        ext = Path(url.split("?")[0]).suffix.lower()
        dest = dest_stem.with_suffix(ext if ext in (".jpg", ".jpeg", ".png") else ".jpg")
        download(url, dest)
        rec = {"path": dest.name, "source": img_spec.get("source", url), "license": img_spec.get("license", "unknown"),
               "artist": img_spec.get("credit", ""), "credit": img_spec.get("credit", ""), "file": None,
               "free": bool(img_spec.get("free", False))}
    # normalise to RGB jpg/png that cv2 can read (tif/gif thumbs are served as jpg/png already)
    im = Image.open(dest)
    if im.mode not in ("RGB", "L"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        im = im.convert("RGBA")
        bg.paste(im, mask=im.split()[-1])
        out = dest.with_suffix(".jpg")
        bg.save(out, quality=95)
        rec["path"] = out.name
    return rec


def clean_artist(a: str) -> str:
    """Commons 'Artist' fields are messy HTML-to-text: keep a short, human credit or nothing."""
    import re
    a = re.sub(r"\s+", " ", a or "").strip()
    a = re.sub(r"(?i)unknown\s*(author|photographer)?", "", a).strip(" ,;")
    a = re.sub(r"\s*\(\d{4}\s*[-–]\s*\d{4}\)", "", a)  # life dates
    if not a or len(a) > 60 or ":" in a or "http" in a:
        return ""
    return a


def _fetch_pair(specs: list[dict], dest_stem: Path) -> dict:
    """Two portraits joined into one picture (same height, thin white gutter). Credits are merged."""
    recs, ims = [], []
    for j, sp in enumerate(specs[:2]):
        r = _fetch_image(sp, dest_stem.with_name(dest_stem.name + f"_p{j}"))
        recs.append(r)
        ims.append(Image.open(dest_stem.parent / r["path"]).convert("RGB"))
    h = min(1400, min(im.height for im in ims))
    ims = [im.resize((int(im.width * h / im.height), h)) for im in ims]
    gut = 24
    out = Image.new("RGB", (sum(im.width for im in ims) + gut, h), (255, 255, 255))
    out.paste(ims[0], (0, 0)); out.paste(ims[1], (ims[0].width + gut, 0))
    dest = dest_stem.with_suffix(".jpg")
    out.save(dest, quality=93)
    return {"path": dest.name, "source": " + ".join(r["source"] for r in recs),
            "license": " / ".join(dict.fromkeys(r["license"] for r in recs)),
            "artist": ", ".join(a for a in (r.get("artist") for r in recs) if a), "credit": "",
            "file": recs[0].get("file"), "free": all(r.get("free") for r in recs), "parts": recs}


def _pin_to_box(pin, det: list[dict]):
    if pin is None:
        return None
    if isinstance(pin, int):
        return det[pin]["box"] if 0 <= pin < len(det) else None
    return [float(v) for v in pin]


def prep(slug: str, log=print) -> dict:
    d, spec = load(slug)
    errs = check_chain(spec)
    if errs:
        raise SystemExit("episode.json problems:\n  " + "\n  ".join(errs))
    work = d / "work"
    (work / "refs").mkdir(parents=True, exist_ok=True)
    overrides = spec.get("people", {})

    # ---- reference portraits per person
    people = {}
    for name in people_order(spec):
        ov = overrides.get(name, {})
        r = wiki.resolve(ov.get("wiki", name)) or {}
        refs = list(ov.get("refs", []))
        if r.get("portrait"):
            refs.insert(0, r["portrait"])
        paths = []
        for k, ref in enumerate(refs[:4]):
            stem = work / "refs" / f"{slugify(name)}_{k}"
            try:
                rec = _fetch_image({"file": ref} if not str(ref).startswith("http") else {"url": ref}, stem, width=960)
                paths.append(work / "refs" / rec["path"])
            except Exception as ex:  # noqa: BLE001
                log(f"  ! ref {ref}: {ex}")
        feats = faces.reference_features(paths)
        people[name] = {"label": ov.get("label", name), "wiki": r.get("title"), "description": r.get("description", ""),
                        "refs": [p.name for p in paths], "_feats": feats}
        log(f"  person {name}: {len(paths)} refs, {len(feats)} usable faces")

    # ---- each link photo
    links_out, credits = [], []
    for i, link in enumerate(spec["links"]):
        if "pair" in link["image"]:   # succession link: two portraits side by side ("line of popes")
            rec = _fetch_pair(link["image"]["pair"], work / f"link{i}")
        else:
            rec = _fetch_image(link["image"], work / f"link{i}")
        credits.append(rec)
        img = faces.load(work / rec["path"])
        det = faces.detect(img)
        sa = faces.best_match(img, det, people[link["a"]]["_feats"])
        sb = faces.best_match(img, det, people[link["b"]]["_feats"])
        res = {"image": rec["path"], "size": [img.shape[1], img.shape[0]], "faces": [f["box"] for f in det],
               "sim_a": [round(x, 3) for x in sa], "sim_b": [round(x, 3) for x in sb]}
        for role, sims in (("a", sa), ("b", sb)):
            pin = link.get(f"{role}_face")
            box = _pin_to_box(pin, det)
            if box is not None:
                res[role] = {"box": box, "method": "manual", "sim": None}
        # auto-assign anything not pinned: best distinct pair
        if "a" not in res or "b" not in res:
            best = None
            n = len(det)
            for ia in range(n):
                for ib in range(n):
                    if ia == ib:
                        continue
                    if "a" in res and det[ia]["box"] != res["a"]["box"]:
                        continue
                    if "b" in res and det[ib]["box"] != res["b"]["box"]:
                        continue
                    s = sa[ia] + sb[ib]
                    if best is None or s > best[0]:
                        best = (s, ia, ib)
            if best is None and n == 1:
                # one face: give it to the better-matching role, other is unresolved
                best = (0, 0, None) if sa[0] >= sb[0] else (0, None, 0)
            if best:
                _, ia, ib = best
                for role, idx, sims in (("a", ia, sa), ("b", ib, sb)):
                    if role in res or idx is None:
                        continue
                    ok = sims[idx] >= faces.SAME_PERSON
                    res[role] = {"box": det[idx]["box"], "method": "sface" if ok else "unverified",
                                 "sim": round(sims[idx], 3), "index": idx}
        for role in ("a", "b"):
            res.setdefault(role, {"box": None, "method": "missing", "sim": None})
        links_out.append(res)
        log(f"  link {i} {link['a']} + {link['b']}: {len(det)} faces; "
            f"a={res['a']['method']}({res['a']['sim']}) b={res['b']['method']}({res['b']['sim']})")
        _review_sheet(work / f"review{i}.jpg", img, det, res, link, people)

    out = {"people": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in people.items()},
           "links": links_out, "credits": credits}
    (work / "faces.json").write_text(json.dumps(out, indent=1))
    return out


def _font(size, weight="Bold"):
    return ImageFont.truetype(str(FONTS / f"Inter-{weight}.otf"), size)


def _review_sheet(path: Path, img: np.ndarray, det, res, link, people):
    """Image with every detected face numbered; A in green, B in blue, with similarities."""
    rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    s = 1400 / max(rgb.size)
    rgb = rgb.resize((int(rgb.width * s), int(rgb.height * s)))
    dr = ImageDraw.Draw(rgb)
    f = _font(max(14, int(18)))
    for i, fc in enumerate(det):
        x, y, w, h = [v * s for v in fc["box"]]
        col = (160, 160, 160)
        if res["a"]["box"] == fc["box"]:
            col = (0, 220, 0)
        if res["b"]["box"] == fc["box"]:
            col = (40, 140, 255)
        dr.rectangle([x, y, x + w, y + h], outline=col, width=3)
        dr.text((x, y + h + 2), f"#{i} a{res['sim_a'][i]:.2f} b{res['sim_b'][i]:.2f}", fill=col, font=f,
                stroke_width=2, stroke_fill=(0, 0, 0))
    for role, col in (("a", (0, 220, 0)), ("b", (40, 140, 255))):
        bx = res[role]["box"]
        if bx and bx not in [fc["box"] for fc in det]:
            x, y, w, h = [v * s for v in bx]
            dr.rectangle([x, y, x + w, y + h], outline=col, width=4)
    hdr = f"A={link['a']} [{res['a']['method']}]   B={link['b']} [{res['b']['method']}]"
    dr.rectangle([0, 0, rgb.width, 30], fill=(0, 0, 0))
    dr.text((6, 4), hdr, fill=(255, 255, 255), font=f)
    rgb.save(path, quality=85)
