"""Domino: a skin around the core. The core photo window is set into the top half of a big ivory domino tile (divider
bar with a brass spinner, the link number in pips below) lying on a green games-table felt. Along the table front a
row of small dominoes keeps the score of the chain: one per link so far plus one standing ahead; when Buh lands on B
the current tile topples onto the next, and the finished ones lie leaning in a row. Only one tile ahead is ever shown,
so the row never gives away how long the chain is."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

NAME = "domino"

S = 0.74                                 # core window scale (880x950 -> 651x703)
Y0 = 432                                 # window top
IVORY = (246, 240, 226)
IVORY_D = (214, 204, 182)
FELT = (30, 104, 68)
M = 24                                   # tile margin around the window
ROW_Y, TW, TH, STEP, ROW_X0 = 1690, 50, 112, 112, 96   # the toppling row: floor y, tile size, spacing, first x
FALL = 0.35                              # s for a small tile to topple
LEAN = math.degrees(math.asin((STEP - TW) / TH))       # a toppled tile rests with its top on the next one
PIPS = {1: [(0, 0)], 2: [(-1, -1), (1, 1)], 3: [(-1, -1), (0, 0), (1, 1)], 4: [(-1, -1), (1, -1), (-1, 1), (1, 1)],
        5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)], 6: [(-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)]}


def make(ctx):
    W, H, CX, INK = ctx.W, ctx.H, ctx.CX, ctx.INK
    w = int(880 * S)
    X0 = int(CX - w / 2)
    PIP_H = 84                                         # the pip half under the window
    _c = {}

    def felt():
        if "felt" not in _c:
            rng = np.random.default_rng(4)
            n = rng.normal(0, 7, (H // 4, W // 4, 1))
            a = np.clip(np.array(FELT, np.float32)[None, None] + n, 0, 255).astype(np.uint8)
            im = Image.fromarray(a).resize((W, H), Image.NEAREST)
            yy, xx = np.mgrid[0:H, 0:W]                 # vignette
            v = 1 - 0.45 * np.clip(((xx - W / 2) / W) ** 2 * 2.2 + ((yy - H * 0.55) / H) ** 2 * 2.2, 0, 1)
            arr = (np.array(im, np.float32) * v[..., None]).astype(np.uint8)
            im = Image.fromarray(arr).convert("RGBA")
            d = ImageDraw.Draw(im)
            d.rectangle([0, ROW_Y + 6, W, ROW_Y + 18], fill=(20, 70, 46))     # table edge line under the row
            _c["felt"] = im
        return _c["felt"].copy()

    def tile(i):
        def draw(b, X, Y, ww, hh):
            d = ImageDraw.Draw(b)
            x0, y0, x1, y1 = X - M, Y - M, X + ww + M, Y + hh + M + PIP_H
            if "sh" not in _c:                                                   # soft shadow, made once
                sh = Image.new("RGBA", (x1 - x0 + 60, y1 - y0 + 60), (0, 0, 0, 0))
                ImageDraw.Draw(sh).rounded_rectangle([30, 30, x1 - x0 + 30, y1 - y0 + 30], 40, fill=(0, 0, 0, 120))
                _c["sh"] = sh.filter(ImageFilter.GaussianBlur(10))
            b.alpha_composite(_c["sh"], (int(x0 + 14 - 30), int(y0 + 20 - 30)))
            d.rounded_rectangle([x0 - 6, y0 - 6, x1 + 6, y1 + 6], 44, fill=INK)
            d.rounded_rectangle([x0, y0, x1, y1], 38, fill=IVORY)
            d.rounded_rectangle([x0 + 6, y1 - 20, x1 - 6, y1 - 4], 12, fill=IVORY_D)      # tile thickness
            d.rectangle([X - 5, Y - 5, X + ww + 5, Y + hh + 5], fill=INK)                  # window recess
            dy = Y + hh + M - 2                                                              # divider + spinner
            d.rectangle([x0 + 34, dy - 4, x1 - 34, dy + 4], fill=INK)
            d.ellipse([X + ww / 2 - 13, dy - 13, X + ww / 2 + 13, dy + 13], fill=(196, 160, 72), outline=INK, width=4)
            n = (i % 6) + 1
            py = dy + 38
            for k in (-1, 1):                         # the pip half: n pips on each side of the spinner
                cx_ = X + ww / 2 + k * ww * 0.27
                for px, pv in PIPS[n]:
                    x, y = cx_ + pv * 36, py + px * 16      # laid across: three columns of two
                    d.ellipse([x - 10, y - 10, x + 10, y + 10], fill=INK)
        return draw

    def small(d, x, ang, n, hot):
        """A small domino standing at x (left foot) on the table, rotated ang degrees clockwise about its right foot."""
        a = math.radians(ang)
        px, py = x + TW, ROW_Y                                       # pivot: right foot

        def rot(u, v):                                               # tile-local (u right, v up from the foot)
            dx, dy = u - TW, -v
            return px + dx * math.cos(a) - dy * math.sin(a), py + dx * math.sin(a) + dy * math.cos(a)
        body = [rot(0, 0), rot(TW, 0), rot(TW, TH), rot(0, TH)]
        d.polygon([(p[0] + 5, p[1] + 5) for p in body], fill=(14, 50, 32))
        d.polygon(body, fill=(255, 236, 150) if hot else IVORY, outline=INK, width=4)
        d.line([rot(6, TH / 2), rot(TW - 6, TH / 2)], fill=INK, width=4)
        for half in (0, 1):
            for pu, pv in PIPS[n][:4] if n > 3 else PIPS[n]:
                q = rot(TW / 2 + pu * 13, TH / 4 + half * TH / 2 + pv * 13)
                d.ellipse([q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5], fill=INK)

    def row(base, i, land):
        d = ImageDraw.Draw(base)
        fall = 0.0 if land < 0 else ctx.ease(min(1.0, land / FALL)) ** 1.6     # gravity: slow start, fast finish
        tiles = [(j, LEAN if j < i else (LEAN * fall if j == i else 0.0)) for j in range(i + 2)]
        first = max(0, i + 2 - 8)                                              # keep the row on screen
        for j, ang in reversed(tiles[first:]):                                  # right to left: leaners on top
            small(d, ROW_X0 + (j - first) * STEP, ang, (j % 6) + 1, j == i + 1 and land >= 0)

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        base = felt()
        c = ctx.draw_core(base, seg, tt, X0, Y0, S, frame=tile(i))
        row(base, i, c["land"])
        return ctx.core_finish(base, seg, tt, c)

    return frame
