"""Subway: a skin around the core. The backdrop is a transit map (paper, grid, a river, faded other lines); the core
photo window is the "station" (a navy platform frame with the line's red stripe). Under it runs the Buh line: one
roundel per person reached so far (their face), a "?" stop ahead, and a little train that pulls out while the camera
pans to both faces and pulls into the next stop as Buh lands on B. Like a real transit map the line is not straight:
stations sit on short level runs joined by 45-degree bends, and the train tilts to follow the track. The line only
ever shows one stop ahead, so it never gives away how long the chain is.
The whole map moves with the train (owner, 2026-11-13): the paper, grid, river, parks, the other lines and their
station names are one long map that pans 1:1 with the Buh line as the train advances, like a camera following along
one big map; nothing behind the line stays put."""
from __future__ import annotations

import itertools
import math
import random

from PIL import Image, ImageDraw, ImageFont

NAME = "subway"

PAPER = (241, 236, 224)
GRID = (231, 225, 211)
LINE = (226, 46, 58)                     # the Buh line
NAVY = (28, 34, 66)
S = 0.72                                 # core window scale (880x950 -> 633x684)
Y0 = 425                                 # window top
LY = 1200                                # the line's centre height, between the window and the captions
AMP = 26                                 # stations sit this far above / below LY (45-degree bends between)
LVL = (-1, 1, 1, -1)                     # station heights, repeating: a bend, a level run, a bend back, a level run
SP = 300                                 # px between stations on screen
RR = 25                                  # roundel radius


def make(ctx):
    W, H, CX, INK, WHITE = ctx.W, ctx.H, ctx.CX, ctx.INK, ctx.WHITE
    w = int(880 * S)
    X0 = int(CX - w / 2)
    people = list(ctx.C.people)
    _c = {}
    MW = W + SP * (ctx.n_links + 2)                      # the one long map: p = 0 .. n_links (+ slack)
    STOPS = ["ELM ST", "MARKET", "NORTH PARK", "HARBOUR", "MUSEUM", "OLD MILL", "KING ST", "UNION SQ", "OAK HILL",
             "CANAL", "STADIUM", "BAKER ST", "LIBRARY", "PIER 9", "WEST END", "OBSERVATORY", "GARDENS", "FERRY",
             "BRIDGE RD", "CITY HALL", "MAPLE AVE", "DOCKS", "UNIVERSITY", "ARCADE"]
    DISTRICTS = ["OLD TOWN", "RIVERSIDE", "THE HEIGHTS", "MIDTOWN", "EAST DOCKS", "GREENWAY", "UPTOWN"]

    def mfont(name, size):
        if (name, size) not in _c:
            _c[(name, size)] = ImageFont.truetype(str(ctx.FONTS / f"Inter-{name}.otf"), size)
        return _c[(name, size)]

    def big_map():
        """The whole transit map, MW px wide (built once): grid, districts, parks, a river, faded other lines with
        named stations. It pans with the Buh line, so it is drawn in the same screen units (SP px per station)."""
        if "map" in _c:
            return _c["map"]
        im = Image.new("RGBA", (MW, H), PAPER + (255,))
        d = ImageDraw.Draw(im)
        for x in range(0, MW, 60):
            d.line([(x, 0), (x, H)], fill=GRID, width=2)
        for y in range(0, H, 60):
            d.line([(0, y), (MW, y)], fill=GRID, width=2)
        rnd = random.Random(5)
        for k in range(MW // 420 + 1):                       # district names, huge and faint, under everything
            x, y = k * 420 + rnd.uniform(0, 160), rnd.choice((250, 1500, 1840))
            d.text((x, y), DISTRICTS[k % len(DISTRICTS)], font=mfont("Black", 44), fill=(224, 216, 198))
        for _ in range(MW // 120):                           # parks
            x, y = rnd.uniform(-100, MW), rnd.uniform(240, H)
            d.rounded_rectangle([x, y, x + rnd.uniform(160, 340), y + rnd.uniform(110, 240)], 34, fill=(216, 229, 199))
        # the river: meanders across the whole map in 45-degree runs
        river, x, y = [(-80, 1640)], -80, 1640
        while x < MW + 80:
            run = rnd.uniform(180, 420)
            x += run
            river.append((x, y))
            ny = min(1820, max(1480, y + rnd.choice((-1, 1)) * rnd.uniform(90, 200)))
            x += abs(ny - y)
            y = ny
            river.append((x, y))
        d.line(river, fill=(190, 214, 236), width=110, joint="curve")
        for p in river[1:-1]:
            d.ellipse([p[0] - 55, p[1] - 55, p[0] + 55, p[1] + 55], fill=(190, 214, 236))
        for k in range(1, len(river) - 1, 6):                # bridges
            (ax, ay), (bx, by) = river[k - 1], river[k]
            mx = (ax + bx) / 2
            d.rectangle([mx - 8, ay - 64, mx + 8, ay + 64], fill=(214, 208, 194))
        # other lines, faded: two long horizontal-ish lines and short north-south branches, all 45-degree bends
        names = iter(STOPS * 6)
        lines = []
        for col, yb in (((132, 166, 222), 300), ((246, 204, 140), 1420)):
            pts, x, y = [(-60, yb)], -60, yb
            while x < MW + 60:
                x += rnd.uniform(260, 480)
                pts.append((x, y))
                ny = yb + rnd.choice((-1, 0, 1)) * 90
                x += abs(ny - y)
                y = ny
                pts.append((x, y))
            lines.append((col, pts))
        for k in range(MW // 640 + 1):                       # branches crossing the map top to bottom
            x = 200 + k * 640 + rnd.uniform(-80, 80)
            col = ((140, 196, 152), (196, 160, 214), (132, 166, 222))[k % 3]
            lines.append((col, [(x, -40), (x, 640), (x + 220, 860), (x + 220, H + 40)] if k % 2 else
                          [(x + 220, -40), (x + 220, 700), (x, 920), (x, H + 40)]))
        for col, pts in lines:
            d.line(pts, fill=col, width=18, joint="curve")
        f = mfont("Bold", 17)
        for col, pts in lines:                               # stations on the corners, with names
            for p in pts[1:-1]:
                if -20 < p[0] < MW + 20 and 220 < p[1] < H - 40:
                    d.ellipse([p[0] - 14, p[1] - 14, p[0] + 14, p[1] + 14], fill=WHITE, outline=col, width=6)
                    d.text((p[0] + 20, p[1] - 30), next(names), font=f, fill=(150, 140, 124))
        _c["map"] = im
        return im

    def backdrop(p):
        """The screen's view of the map with the train at track position p (pans 1:1 with the Buh line)."""
        ox = max(0, min(MW - W, int(round(p * SP))))
        return big_map().crop((ox, 0, ox + W, H))

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

    def ty(sv):
        """Track height (screen y) at track position sv (in stations): level at each station, a 45-degree bend
        halfway between two stations at different heights."""
        j = math.floor(sv)
        y0, y1 = LY + AMP * LVL[j % 4], LY + AMP * LVL[(j + 1) % 4]
        if y0 == y1:
            return y0
        half = abs(y1 - y0) / 2 / SP                         # half the bend's length, in stations (45 degrees)
        u = (sv - j - (0.5 - half)) / (2 * half)
        return y0 + (y1 - y0) * max(0.0, min(1.0, u))

    def track_pts(s0, s1, sx):
        """Screen points of the track from s0 to s1 (bend corners included, so the 45s stay crisp)."""
        pts, j = [], math.floor(s0)
        knots = [s0]
        while j <= s1:
            for f in (0.0, 0.5 - AMP / SP, 0.5 + AMP / SP):
                if s0 < j + f < s1:
                    knots.append(j + f)
            j += 1
        knots.append(s1)
        for kv in sorted(knots):
            pts.append((sx(kv), ty(kv)))
        return pts

    def progress(i, k, R, sp_):
        """Track position of the train (in stations) and its run fraction u."""
        dep = R - 0.22                                       # leaves as the camera starts to pan to both
        u = (k - dep) / max(1e-3, (R - dep) + ctx.THROW / sp_)
        return i + ctx.ease(max(0.0, min(1.0, u))), u

    def line(base, i, c):
        """The Buh line under the window: stations up to the next stop, the train between them."""
        sp_, k, R = c.get("speech", 1.0), c["k"], c["R"]
        p, u = progress(i, k, R, sp_)
        landed = c["land"] >= 0
        sx = lambda j: CX + 85 + (j - p) * SP            # noqa: E731  (the train sits just left of its stop)
        sv_of = lambda x: (x - CX - 85) / SP + p         # noqa: E731
        d = ImageDraw.Draw(base)
        known = track_pts(sv_of(-40), i + 1, sx)
        d.line(known, fill=INK, width=20, joint="curve")
        d.line(known, fill=LINE, width=13, joint="curve")
        # beyond the next stop: unknown track, fading dashes that still follow the bends
        ahead = track_pts(i + 1, sv_of(W + 40), sx)
        dense = []                                           # dashes by arc length: 18 on, 18 off
        for (ax, ay), (bx, by) in itertools.pairwise(ahead):
            n = max(1, int(math.hypot(bx - ax, by - ay) / 3))
            dense += [(ax + (bx - ax) * q / n, ay + (by - ay) * q / n) for q in range(n)]
        dist, prev = 0.0, None
        for pt in dense:
            if prev is not None:
                dist += math.hypot(pt[0] - prev[0], pt[1] - prev[1])
            prev = pt
            if dist > 40 and (dist - 40) % 36 < 18:
                d.ellipse([pt[0] - 4, pt[1] - 4, pt[0] + 4, pt[1] + 4], fill=(205, 170, 160))
        for j in range(max(0, i - 3), min(len(people), i + 2)):
            ctx.paste(base, roundel(j, j <= i or (j == i + 1 and landed)), sx(j) + 3, ty(j) + 3)
        tr = train()
        moving = 0 < u < 1
        bob = 2 * math.sin(k * sp_ * 40) if moving else 0          # rattles while moving
        tx = CX - 8                                                  # the train's centre on screen
        e = 40 / SP
        sv = sv_of(tx)
        ang = math.degrees(math.atan2(-(ty(sv + e) - ty(sv - e)), 2 * e * SP))   # tilts with the track
        if abs(ang) > 0.5:
            tr = tr.rotate(ang, resample=Image.BICUBIC, expand=True)
        base.alpha_composite(tr, (int(tx - tr.width / 2), int(ty(sv) - tr.height / 2 - 4 + bob)))

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        k = min(1.0, (tt - seg["t0"]) / seg["speech"])     # the same clock draw_core runs on
        base = backdrop(progress(i, k, seg.get("rv", ctx.REVEAL), seg["speech"])[0])
        c = ctx.draw_core(base, seg, tt, X0, Y0, S, frame=station)
        c["speech"] = seg["speech"]
        line(base, i, c)
        X, Y, ww, hh = c["rect"]                         # the credit goes inside the photo's bottom edge: the line
        return ctx.core_finish(base, seg, tt, c, credit_xy=(X + ww - 12, Y + hh - 18))   # needs the strip below

    return frame
