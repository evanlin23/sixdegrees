"""Subway: a skin around the core. The backdrop is a transit map (paper, grid, a river, faded other lines); the core
photo window is the "station" (a navy platform frame with the line's red stripe). Under it runs the Buh line: one
roundel per person reached so far (their face), a "?" stop ahead, and a little train that pulls out while the camera
pans to both faces and pulls into the next stop as Buh lands on B. The line only ever shows one stop ahead, so it
never gives away how long the chain is."""
from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw

NAME = "subway"

PAPER = (241, 236, 224)
GRID = (231, 225, 211)
LINE = (226, 46, 58)                     # the Buh line
NAVY = (28, 34, 66)
S = 0.76                                 # core window scale (880x950 -> 669x722)
Y0 = 425                                 # window top
LY = 1212                                # the line, between the window and the captions
SP = 300                                 # px between stations on screen
RR = 25                                  # roundel radius


def make(ctx):
    W, H, CX, INK, WHITE = ctx.W, ctx.H, ctx.CX, ctx.INK, ctx.WHITE
    w = int(880 * S)
    X0 = int(CX - w / 2)
    people = list(ctx.C.people)
    _c = {}

    def backdrop():
        if "bg" not in _c:
            im = Image.new("RGBA", (W, H), PAPER + (255,))
            d = ImageDraw.Draw(im)
            for x in range(0, W, 60):
                d.line([(x, 0), (x, H)], fill=GRID, width=2)
            for y in range(0, H, 60):
                d.line([(0, y), (W, y)], fill=GRID, width=2)
            rnd = random.Random(5)
            for _ in range(9):                                       # parks
                x, y = rnd.uniform(-100, W), rnd.uniform(300, H)
                d.rounded_rectangle([x, y, x + rnd.uniform(160, 340), y + rnd.uniform(110, 240)], 34, fill=(216, 229, 199))
            river = [(-60, 1560), (180, 1560), (420, 1800), (W + 60, 1800)]
            d.line(river, fill=(190, 214, 236), width=110)
            for p in river[1:3]:
                d.ellipse([p[0] - 55, p[1] - 55, p[0] + 55, p[1] + 55], fill=(190, 214, 236))
            for col, pts in (((132, 166, 222), [(70, 300), (70, 900), (300, 1130), (300, H)]),            # other lines,
                             ((140, 196, 152), [(W, 700), (1000, 700), (760, 940), (760, H)]),            # faded
                             ((246, 204, 140), [(0, 1460), (520, 1460), (760, 1700), (W, 1700)])):
                d.line(pts, fill=col, width=18, joint="curve")
                for p in pts[1:-1]:
                    d.ellipse([p[0] - 14, p[1] - 14, p[0] + 14, p[1] + 14], fill=WHITE, outline=col, width=6)
            _c["bg"] = im
        return _c["bg"].copy()

    def station(b, X, Y, ww, hh):
        """The platform frame around the window: navy, a red line stripe, white keyline, a hard drop shadow."""
        d = ImageDraw.Draw(b)
        d.rounded_rectangle([X - 22 + 12, Y - 22 + 12, X + ww + 22 + 12, Y + hh + 22 + 12], 26, fill=INK)
        d.rounded_rectangle([X - 22, Y - 22, X + ww + 22, Y + hh + 22], 26, fill=NAVY)
        d.rectangle([X - 10, Y - 10, X + ww + 10, Y + hh + 10], fill=LINE)
        d.rectangle([X - 4, Y - 4, X + ww + 4, Y + hh + 4], fill=WHITE)

    def train():
        if "train" not in _c:
            im = Image.new("RGBA", (132, 52), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.rounded_rectangle([4, 6, 130, 50], 10, fill=INK)
            d.rounded_rectangle([0, 0, 124, 44], 10, fill=WHITE, outline=INK, width=4)
            d.rectangle([4, 28, 120, 36], fill=LINE)
            for x in range(12, 100, 26):
                d.rectangle([x, 8, x + 18, 22], fill=(120, 190, 235), outline=INK, width=2)
            d.polygon([(104, 6), (120, 6), (120, 24), (104, 24)], fill=(120, 190, 235), outline=INK)
            _c["train"] = im
        return _c["train"]

    def roundel(j, known):
        key = ("r", j, known)
        if key not in _c:
            D = 2 * RR + 12
            im = Image.new("RGBA", (D, D), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.ellipse([6, 6, D - 1, D - 1], fill=INK)
            d.ellipse([0, 0, D - 7, D - 7], fill=WHITE, outline=INK, width=5)
            if known:
                th = ctx.sharp[people[j]].resize((D - 19, D - 19), Image.BICUBIC).convert("RGBA")
                m = Image.new("L", th.size, 0)
                ImageDraw.Draw(m).ellipse([0, 0, th.width - 1, th.height - 1], fill=255)
                im.paste(th, (6, 6), m)
                d.ellipse([0, 0, D - 7, D - 7], outline=INK, width=5)
            else:
                q = ctx.pixel_text("?", 26, fill=INK, outline=WHITE, shadow=False)
                im.alpha_composite(q, (int((D - 7) / 2 - q.width / 2), int((D - 7) / 2 - q.height / 2)))
            _c[key] = im
        return _c[key]

    def line(base, i, c):
        """The Buh line under the window: stations up to the next stop, the train between them."""
        sp_, k, R = c.get("speech", 1.0), c["k"], c["R"]
        dep = R - 0.22                                       # leaves as the camera starts to pan to both
        u = (k - dep) / max(1e-3, (R - dep) + ctx.THROW / sp_)
        p = i + ctx.ease(max(0.0, min(1.0, u)))
        landed = c["land"] >= 0
        sx = lambda j: CX + 85 + (j - p) * SP            # noqa: E731  (the train sits just left of its stop)
        d = ImageDraw.Draw(base)
        d.rectangle([0, LY - 9, sx(i + 1), LY + 9], fill=INK)
        d.rectangle([0, LY - 6, sx(i + 1), LY + 6], fill=LINE)
        x = sx(i + 1) + 40                                   # beyond the next stop: unknown track, fading dashes
        while x < W:
            d.rectangle([x, LY - 4, x + 18, LY + 4], fill=(205, 170, 160))
            x += 36
        for j in range(max(0, i - 3), min(len(people), i + 2)):
            ctx.paste(base, roundel(j, j <= i or (j == i + 1 and landed)), sx(j) + 3, LY + 3)
        tr = train()
        bob = 2 * math.sin(k * sp_ * 40) if 0 < u < 1 else 0          # rattles while moving
        base.alpha_composite(tr, (int(CX - tr.width / 2 - 8), int(LY - tr.height / 2 - 4 + bob)))

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        base = backdrop()
        c = ctx.draw_core(base, seg, tt, X0, Y0, S, frame=station)
        c["speech"] = seg["speech"]
        line(base, i, c)
        return ctx.core_finish(base, seg, tt, c)

    return frame
