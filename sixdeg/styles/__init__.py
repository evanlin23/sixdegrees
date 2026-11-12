"""Presentation styles as plug-ins.

Every module here that defines ``make(ctx) -> frame(seg, tt)`` becomes a style named after the module (or its
``NAME``). ``render_pixel.render`` builds ``ctx`` (a SimpleNamespace) with everything the built-in styles use and
calls ``load(ctx)``; an episode picks a style with ``"style": "<name>"``.

``frame(seg, tt)`` returns a 1080x1920 RGBA ``PIL.Image`` for time ``tt`` inside a link segment. ``seg`` has:
``i`` (link index), ``t0``/``t1`` (segment start/end, s), ``speech`` (narration length, s; also the animation
clock), ``rv`` (fraction of ``speech`` at which B is revealed), ``cap`` (captions: ``seg["cap"].draw(base, tt)``).

What ``ctx`` gives you (see ``CTX_KEYS``; all are the renderer's own objects):
- data: ``spec`` (episode.json), ``C`` (``C.labels[name]``, ``C.people``, ``C.fj["links"][i]``), ``n_links``,
  ``segs``, ``sharp`` (120 px face thumbs by name)
- photo helpers: ``view(i, cx, cy, scale, w, h)`` -> RGBA crop of photo i centred on image point (cx, cy);
  ``both_cam(i, w, h, fill)`` -> (cx, cy, scale) framing both faces; ``face_cam(box, h)``; ``fbox(i, "a"|"b")`` ->
  face box (x, y, w, h) in image px; ``ctr(box)``; ``framed(img, tilt)`` -> a taped, bordered print
- text: ``link_names(i)`` -> (a, b); ``year_of(i)``; ``fact_of(link, full=False)`` (cut before B's name unless
  full=True: never show B's name before the reveal); ``before_b(text, link)`` (same cut for any text); ``pixel_text(text, size, fill)``;
  ``pixel_tag(text, bg, size)``; ``pf(size)`` (pixel font); ``FONTS`` (Inter-*.otf for a "real UI" look)
- drawing: ``paste(base, im, cx, cy, scale=1, alpha=1, nearest=True)``; ``checker(t, (c1, c2))``;
  ``paste_pin(base, pin_sprite(expr, angle, squash, look), x, tip_y)`` (Buh; expr in shock/smug/happy/angry);
  ``header(base, i)`` (title band + LINK n); ``connected(base, seg, tt, y=1020)`` (CONNECTED! on the last link);
  ``frame_lineage(seg, tt)`` (fall back to it when ``C.fj["links"][i].get("lineage")``)
- ``sensitive(name)`` -> True for dictators/killers: no hearts, likes, coins, scores, rarity, gold frames,
  verified badges or first-person posts on a link with such a person
- timing/motion: ``ease``, ``ease_out_back``, ``shake(t, amp, dur)``, ``REVEAL``, ``THROW``
- layout: ``W, H, CX, SAFE`` (x0, y0, x1, y1: keep text inside), ``BAND``, ``CAP_Y``; colours ``INK, WHITE,
  YELLOW, CYAN, RED``

THE CORE (owner rule, 2026-11-12: every format must keep it; a style is a skin AROUND it, never a replacement):
- ``draw_core(base, seg, tt, x0, y0, scale=1.0, frame=None)`` places the photo window (CARD-sized, 880x950 at
  scale 1) with the zoom-out / move / zoom-in camera, the red string, the pinned marker and the name tags under the
  chins (all riding the photo, sliding under its edge), then Buh jumping head to head on top. ``frame(base, X0, Y0,
  w, h)`` draws your mount/border behind it. Returns a dict: ``i, k, R, since, land, last, incut, spa, spb`` (Buh's
  screen points), ``rect`` (X0, Y0, w, h).
- ``core_finish(base, seg, tt, c, connected_y=1020, sub=True, chain=True, caption=True)``: CONNECTED!, the
  Subscribe nudge on the payoff, title band, chain bar, credit, captions and cut flashes. Call it last.
- ``link_core(seg, tt)`` if you need the raw window image to transform (tilt, fold) yourself; you then draw Buh.
- Keep the window at least ~0.7 scale, keep captions (CAP_Y) readable, never cover the faces with chrome.
"""
from __future__ import annotations

import importlib
import pkgutil
from types import SimpleNamespace

CTX_KEYS = ["spec", "C", "n_links", "segs", "sharp", "view", "both_cam", "face_cam", "fbox", "ctr", "framed",
            "link_names", "year_of", "fact_of", "before_b", "header", "connected", "frame_lineage", "link_core", "draw_core", "core_finish"]
GLOBAL_KEYS = ["pixel_text", "pixel_tag", "pf", "FONTS", "paste", "checker", "paste_pin", "pin_sprite", "ease",
               "ease_out_back", "shake", "subscribe_button", "CARD", "sensitive", "REVEAL", "THROW", "W", "H", "CX", "SAFE", "BAND", "CAP_Y", "INK",
               "WHITE", "YELLOW", "CYAN", "RED"]


def build_ctx(local_vars: dict, global_vars: dict) -> SimpleNamespace:
    d = {k: local_vars[k] for k in CTX_KEYS if k in local_vars}
    d.update({k: global_vars[k] for k in GLOBAL_KEYS if k in global_vars})
    return SimpleNamespace(**d)


def load(ctx) -> dict:
    """{style name: frame function} for every plug-in module."""
    out = {}
    for m in pkgutil.iter_modules(__path__):
        if m.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{__name__}.{m.name}")
        if hasattr(mod, "make"):
            out[getattr(mod, "NAME", m.name)] = mod.make(ctx)
    return out


def names() -> list[str]:
    return sorted(m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_"))
