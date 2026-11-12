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
- text: ``link_names(i)`` -> (a, b); ``year_of(i)``; ``fact_of(link)``; ``pixel_text(text, size, fill)``;
  ``pixel_tag(text, bg, size)``; ``pf(size)`` (pixel font); ``FONTS`` (Inter-*.otf for a "real UI" look)
- drawing: ``paste(base, im, cx, cy, scale=1, alpha=1, nearest=True)``; ``checker(t, (c1, c2))``;
  ``paste_pin(base, pin_sprite(expr, angle, squash, look), x, tip_y)`` (Buh; expr in shock/smug/happy/angry);
  ``header(base, i)`` (title band + LINK n); ``connected(base, seg, tt)`` (CONNECTED! on the last link);
  ``frame_lineage(seg, tt)`` (fall back to it when ``C.fj["links"][i].get("lineage")``)
- timing/motion: ``ease``, ``ease_out_back``, ``shake(t, amp, dur)``, ``REVEAL``, ``THROW``
- layout: ``W, H, CX, SAFE`` (x0, y0, x1, y1: keep text inside), ``BAND``, ``CAP_Y``; colours ``INK, WHITE,
  YELLOW, CYAN, RED``
"""
from __future__ import annotations

import importlib
import pkgutil
from types import SimpleNamespace

CTX_KEYS = ["spec", "C", "n_links", "segs", "sharp", "view", "both_cam", "face_cam", "fbox", "ctr", "framed",
            "link_names", "year_of", "fact_of", "header", "connected", "frame_lineage"]
GLOBAL_KEYS = ["pixel_text", "pixel_tag", "pf", "FONTS", "paste", "checker", "paste_pin", "pin_sprite", "ease",
               "ease_out_back", "shake", "REVEAL", "THROW", "W", "H", "CX", "SAFE", "BAND", "CAP_Y", "INK",
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
