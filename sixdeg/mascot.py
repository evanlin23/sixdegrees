"""The channel mascot: "Pin", a red pushpin with a face. It holds the evidence string.

Original design, drawn in code (no assets to license). draw() returns an RGBA sprite.
Expressions: "happy", "shock" (O mouth, tiny pupils), "smug" (half-lidded).
"""
from __future__ import annotations

import math
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFilter

RED = (232, 30, 52)
RED_DARK = (150, 10, 30)
NEEDLE = (205, 210, 220)


@lru_cache(maxsize=64)
def _base(expr: str, look: int) -> Image.Image:
    S = 4  # supersample
    w, h = 220 * S, 320 * S
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx = w // 2
    # needle
    d.polygon([(cx - 9 * S, 188 * S), (cx + 9 * S, 188 * S), (cx, 312 * S)], fill=NEEDLE, outline=(90, 95, 110),
              width=2 * S)
    # collar (the flat disc of a pushpin)
    d.ellipse([cx - 62 * S, 168 * S, cx + 62 * S, 204 * S], fill=RED_DARK)
    d.ellipse([cx - 62 * S, 162 * S, cx + 62 * S, 196 * S], fill=RED)
    # head
    r = 92 * S
    hy = 100 * S
    d.ellipse([cx - r, hy - r + 6 * S, cx + r, hy + r + 6 * S], fill=RED_DARK)
    d.ellipse([cx - r, hy - r, cx + r, hy + r], fill=RED)
    d.ellipse([cx - 55 * S, hy - 75 * S, cx - 5 * S, hy - 35 * S], fill=(255, 150, 160))  # shine
    d.ellipse([cx - 45 * S, hy - 68 * S, cx - 22 * S, hy - 50 * S], fill=(255, 235, 238))
    # eyes
    lx = look * 6 * S
    for ex in (cx - 34 * S, cx + 34 * S):
        ey = hy + 2 * S
        if expr == "smug":
            d.ellipse([ex - 22 * S, ey - 18 * S, ex + 22 * S, ey + 24 * S], fill="white", outline="black", width=4 * S)
            d.ellipse([ex - 9 * S + lx, ey - 2 * S, ex + 9 * S + lx, ey + 16 * S], fill="black")
            d.rectangle([ex - 24 * S, ey - 22 * S, ex + 24 * S, ey + 2 * S], fill=RED)  # heavy lids
            d.line([ex - 22 * S, ey + 2 * S, ex + 22 * S, ey + 2 * S], fill="black", width=5 * S)
        else:
            big = 1.18 if expr == "shock" else 1.0
            rr = int(24 * S * big)
            d.ellipse([ex - rr, ey - int(rr * 1.15), ex + rr, ey + int(rr * 1.15)], fill="white", outline="black",
                      width=4 * S)
            pr = 6 * S if expr == "shock" else 11 * S
            d.ellipse([ex - pr + lx, ey - pr + 3 * S, ex + pr + lx, ey + pr + 3 * S], fill="black")
            d.ellipse([ex - pr + lx + 3 * S, ey - pr + 4 * S, ex - pr + lx + 8 * S, ey - pr + 9 * S], fill="white")
    # mouth
    my = hy + 46 * S
    if expr == "shock":
        d.ellipse([cx - 14 * S, my - 12 * S, cx + 14 * S, my + 20 * S], fill=(60, 0, 10), outline="black", width=4 * S)
    elif expr == "smug":
        d.arc([cx - 10 * S, my - 18 * S, cx + 34 * S, my + 8 * S], 20, 150, fill="black", width=5 * S)
    else:
        d.chord([cx - 26 * S, my - 22 * S, cx + 26 * S, my + 18 * S], 10, 170, fill=(60, 0, 10), outline="black",
                width=4 * S)
        d.ellipse([cx - 10 * S, my + 2 * S, cx + 10 * S, my + 14 * S], fill=(255, 110, 120))
    # thick outline glow so it reads on any photo
    a = im.split()[-1].filter(ImageFilter.MaxFilter(9))
    out = Image.new("RGBA", im.size, (255, 255, 255, 0))
    out.putalpha(a)
    white = Image.new("RGBA", im.size, (255, 255, 255, 255))
    white.putalpha(a)
    white.alpha_composite(im)
    return white.resize((w // S, h // S), Image.LANCZOS)


def draw(expr: str = "happy", size: float = 1.0, angle: float = 0.0, squash: float = 0.0, look: int = 0) -> Image.Image:
    """squash > 0 flattens (landing), < 0 stretches (flying)."""
    im = _base(expr, max(-1, min(1, look)))
    sx = size * (1 + squash * 0.35)
    sy = size * (1 - squash * 0.35)
    im = im.resize((max(1, int(im.width * sx)), max(1, int(im.height * sy))), Image.BICUBIC)
    if angle:
        im = im.rotate(angle, resample=Image.BICUBIC, expand=True)
    return im


def throw_path(p0, p1, k: float, height: float = 260.0):
    """Arc from p0 to p1 at progress k (0..1). Returns (x, y, angle)."""
    x = p0[0] + (p1[0] - p0[0]) * k
    y = p0[1] + (p1[1] - p0[1]) * k - height * 4 * k * (1 - k)
    spin = 360 * k * (1 if p1[0] >= p0[0] else -1)
    return x, y, spin
