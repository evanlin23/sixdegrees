"""Domino: every link IS one big domino. A tall ebony tile stands on a green games-table felt and fills most of the
screen: its top half is the core photo window (the shared camera, Buh, the red string and the name tags), a brass
spinner and divider run across, and the bottom half carries the link number in ivory pips plus a small gold
"year · place" engraving (cut before B's name, like every on-screen text).

Between links the tile itself falls: on the cut, a still of the previous tile (its last frame, without Buh, B's
head) tips forward toward the viewer in perspective, with gravity, and drops out of the bottom of the frame, showing
the next tile already standing behind it, which takes a small "clack" jolt as the old one hits. The whole fall is
over in the first FALL seconds of the link, while the core is still in its opening pixel-in, so it never delays the
camera or Buh. On the final link the pip half turns into the whole chain: one small tile per link, standing in a
row, that topple one into the next as Buh flies to B, just before CONNECTED!.

Sensitive links (dictators etc.) get the plain treatment: a steel spinner and grey engraving, no gold, and no
toppling chain if the payoff link is one. Lineage links use the shared
lineage frame (and the tile after one simply stands, nothing falls).

The table is never still: the felt's nap drifts slowly, a warm lamp pool swings gently across it and the tile's
shadow leans away from the lamp."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

NAME = "domino"

S = 0.72                                 # core window scale (880x950 -> 633x684)
Y0 = 440                                 # window top (the chain bar ends ~y 406)
M = 26                                   # tile margin around the window (sides and top)
TILE_BOT = 1420                          # tile foot (the Subscribe button pops in at y ~1430..1525)
FELT = (30, 104, 68)
EBONY = (30, 26, 30)
EBONY_HI = (70, 62, 66)
EBONY_SIDE = (12, 10, 12)
IVORY = (246, 240, 226)
GOLD = (206, 168, 80)
STEEL = (150, 152, 158)
FALL = 0.26                              # s for the previous tile to fall away at the start of a link
D_CAM = 1750.0                           # camera distance (px) for the fall's perspective
EYE_Y = 880.0                            # eye level on screen
PIPS = {0: [], 1: [(0, 0)], 2: [(-1, -1), (1, 1)], 3: [(-1, -1), (0, 0), (1, 1)],
        4: [(-1, -1), (1, -1), (-1, 1), (1, 1)], 5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)],
        6: [(-1, -1), (0, -1), (1, -1), (-1, 1), (0, 1), (1, 1)]}


def _homography(src, dst):
    """Coefficients for PIL's PERSPECTIVE transform: maps output points dst -> input points src."""
    a, b = [], []
    for (x, y), (u, v) in zip(dst, src):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    return np.linalg.solve(np.array(a, float), np.array(b, float)).tolist()


def make(ctx):
    W, H, CX, INK = ctx.W, ctx.H, ctx.CX, ctx.INK
    w, h = int(880 * S), int(950 * S)
    X0 = int(CX - w / 2)
    TX0, TY0, TX1, TY1 = X0 - M, Y0 - M, X0 + w + M, TILE_BOT        # the tile's face
    DIV_Y = Y0 + h + 42                                                 # divider (the credit sits above it)
    PIP_Y = DIV_Y + 46                                                  # pip block centre
    ENG_Y = TY1 - 38                                                    # engraving line
    links = ctx.C.fj["links"]
    _c = {}

    def sens(i):
        a, b = ctx.link_names(i)
        return ctx.sensitive(a) or ctx.sensitive(b)

    def lineage(i):
        return bool(links[i].get("lineage"))

    def seg_of(i):
        return next(s for s in ctx.segs if s["kind"] == "link" and s["i"] == i)

    def timing(seg, tt):
        k = min(1.0, (tt - seg["t0"]) / seg["speech"])
        since = (k - seg.get("rv", ctx.REVEAL)) * seg["speech"]
        return since, since - ctx.THROW

    # ---------- table: the felt drifts slowly under a softly swinging lamp, and the tile's shadow follows the lamp
    DR = 64                                                              # felt overscan for the drift, px

    def table_layers():
        if "felt" not in _c:
            rng = np.random.default_rng(4)
            fw_, fh_ = (W + 2 * DR) // 4, (H + 2 * DR) // 4
            n = rng.normal(0, 7, (fh_, fw_, 1))
            a = np.clip(np.array(FELT, np.float32)[None, None] + n, 0, 255).astype(np.uint8)
            _c["felt"] = Image.fromarray(a).resize((fw_ * 4, fh_ * 4), Image.NEAREST).convert("RGBA")
            yy, xx = np.mgrid[0:H, 0:W]
            v = 1 - 0.5 * np.clip(((xx - W / 2) / W) ** 2 * 2.2 + ((yy - H * 0.55) / H) ** 2 * 2.2, 0, 1)
            vig = np.zeros((H, W, 4), np.uint8)
            vig[..., 3] = ((1 - v) * 255).astype(np.uint8)
            _c["vig"] = Image.fromarray(vig)                              # the table's edges fall off into dark
            sh = Image.new("RGBA", (TX1 - TX0 + 120, TY1 - TY0 + 120), (0, 0, 0, 0))    # the tile's shadow
            ImageDraw.Draw(sh).rounded_rectangle([50, 50, TX1 - TX0 + 74, TY1 - TY0 + 46], 46, fill=(0, 0, 0, 130))
            _c["shadow"] = sh.filter(ImageFilter.GaussianBlur(14))
            r = 620                                                       # warm lamp pool (soft radial)
            yy, xx = np.mgrid[0:2 * r, 0:2 * r]
            q = np.clip(1 - np.hypot(xx - r, yy - r) / r, 0, 1) ** 2
            lamp = np.zeros((2 * r, 2 * r, 4), np.uint8)
            lamp[..., :3] = (255, 226, 160)
            lamp[..., 3] = (q * 56).astype(np.uint8)
            _c["lamp"] = Image.fromarray(lamp)
        return _c["felt"], _c["vig"], _c["shadow"], _c["lamp"]

    def felt(tt):
        tex, vig, shadow, lamp = table_layers()
        ox = DR + DR * 0.85 * math.sin(tt * 0.21)
        oy = DR + DR * 0.85 * math.sin(tt * 0.16 + 1.3)
        im = tex.crop((int(ox), int(oy), int(ox) + W, int(oy) + H))
        lx = CX + 330 * math.sin(tt * 0.45)                                # the lamp swings slowly over the table
        ly = 1060 + 140 * math.sin(tt * 0.31 + 0.6)
        im.alpha_composite(lamp, (int(lx - lamp.width / 2), int(ly - lamp.height / 2)))
        im.alpha_composite(vig)
        sdx, sdy = -(lx - CX) * 0.05, -(ly - 1060) * 0.05                  # the shadow falls away from the lamp
        im.alpha_composite(shadow, (int(TX0 + 10 - 50 + sdx), int(TY0 + 30 - 50 + sdy)))
        return im

    # ---------- the big tile (static per link; the core window is laid into its recess)
    def engraving(i):
        yr = ctx.year_of(i)
        place = ((ctx.spec["links"][i].get("where") or {}).get("place") or "").strip()
        place = ctx.before_b(place, ctx.spec["links"][i]).rstrip("… ") if place else ""
        parts = [str(yr)] if yr else []
        if place:
            parts.append(place.upper())
        return "  ·  ".join(parts)

    def pips(d, cx, cy, n, r, sx, sy, fill, rim):
        for px, py in PIPS[min(6, max(0, n))]:
            x, y = cx + px * sx, cy + py * sy
            d.ellipse([x - r - 3, y - r - 3, x + r + 3, y + r + 3], fill=rim)       # drilled recess
            d.ellipse([x - r, y - r, x + r, y + r], fill=fill)
            d.ellipse([x - r * 0.45, y - r * 0.55, x - r * 0.05, y - r * 0.15], fill=(255, 255, 255))

    def tile_img(i):
        """The tile body for link i (face + side + divider + pips + engraving), as a full-frame RGBA layer."""
        key = ("tile", i)
        if key in _c:
            return _c[key]
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([TX0 + 4, TY0 + 10, TX1 + 18, TY1 + 16], 44, fill=EBONY_SIDE)    # thickness (side)
        d.rounded_rectangle([TX0 - 4, TY0 - 4, TX1 + 4, TY1 + 4], 46, fill=INK)
        d.rounded_rectangle([TX0, TY0, TX1, TY1], 42, fill=EBONY)
        d.rounded_rectangle([TX0 + 7, TY0 + 7, TX1 - 7, TY1 - 7], 36, outline=EBONY_HI, width=3)  # bevel
        d.rectangle([X0 - 7, Y0 - 7, X0 + w + 7, Y0 + h + 7], fill=(8, 6, 8))                     # window recess
        d.rectangle([X0 - 9, Y0 - 9, X0 + w + 9, Y0 - 7], fill=EBONY_HI)
        metal = STEEL if sens(i) else GOLD
        d.rounded_rectangle([TX0 + 40, DIV_Y - 5, TX1 - 40, DIV_Y + 5], 4, fill=(200, 192, 176))  # divider
        d.line([TX0 + 44, DIV_Y + 4, TX1 - 44, DIV_Y + 4], fill=(120, 112, 100), width=2)
        d.ellipse([CX - 18, DIV_Y - 18, CX + 18, DIV_Y + 18], fill=metal, outline=INK, width=4)    # spinner
        d.ellipse([CX - 7, DIV_Y - 9, CX + 1, DIV_Y - 2], fill=(255, 246, 214))
        if i < ctx.n_links - 1 or sens(i):          # the final link's pip half becomes the toppling chain
            pips(d, CX, PIP_Y, i + 1, 12, 32, 22, IVORY, (6, 4, 6))
        txt = engraving(i)
        if txt:
            f = ctx.pf(22)
            while f.size > 14 and f.getlength(txt) > TX1 - TX0 - 80:
                f = ctx.pf(f.size - 2)
            d.text((CX + 2, ENG_Y + 2), txt, font=f, fill=(4, 2, 4), anchor="mm")                 # engraved
            d.text((CX, ENG_Y), txt, font=f, fill=(170, 170, 176) if sens(i) else GOLD, anchor="mm")
        _c[key] = im
        return im

    def tile_frame(i):
        def draw(b, X, Y, ww, hh):
            b.alpha_composite(tile_img(i), (X - X0, Y - Y0))
        return draw

    # ---------- the previous tile, as it stood at the end of its link (Buh on B), for the fall
    def still(i):
        key = ("still", i)
        if key not in _c:
            sg = seg_of(i)
            im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            # the photo window as it stood at the end of its link, WITHOUT Buh: the live Buh is already standing on
            # the next photo, and two Buhs on screen during the fall looked wrong (owner: formats must be clean)
            c = ctx.link_core({**sg, "cap": None}, sg["t1"] - 1 / 30)
            cw, ch = c["size"]
            w, h = int(cw * S), int(ch * S)
            tile_frame(i)(im, int(X0), int(Y0), w, h)
            im.alpha_composite(c["img"].resize((w, h), Image.BILINEAR), (int(X0), int(Y0)))
            _c[key] = im
        return _c[key]

    def project(x, y, th, drop):
        """Screen point of tile point (x, y) after tipping th radians toward the viewer about the tile foot."""
        v = TILE_BOT - y
        z = D_CAM - v * math.sin(th)
        s = D_CAM / max(60.0, z)
        return CX + (x - CX) * s, EYE_Y + (TILE_BOT + drop - v * math.cos(th) - EYE_Y) * s

    def falling(base, i, u):
        """Draw link i's tile falling toward the viewer, u in 0..1 of the fall (gravity: slow start, fast finish)."""
        th = math.radians(90 * u ** 1.7)
        drop = 900 * max(0.0, (u - 0.6) / 0.4) ** 2                      # it slides off the table's front edge
        src = [(TX0, TY0), (TX1, TY0), (TX1, TY1), (TX0, TY1)]
        dst = [project(x, y, th, drop) for x, y in src]
        top_y, foot_y = (dst[0][1] + dst[1][1]) / 2, (dst[2][1] + dst[3][1]) / 2
        if top_y >= foot_y - 4:                                           # past edge-on: we see its plain back
            d = ImageDraw.Draw(base)
            d.polygon(dst, fill=EBONY_SIDE, outline=INK)
            return
        lay = still(i).transform((W, H), Image.PERSPECTIVE, _homography(src, dst), Image.BILINEAR)
        shade = 0.55 * math.sin(th)                                       # it turns its face down, away from the light
        if shade > 0.01:
            a = np.asarray(lay).copy()
            a[..., :3] = (a[..., :3] * (1 - shade)).astype(np.uint8)
            lay = Image.fromarray(a)
        base.alpha_composite(lay)

    # ---------- the final link: the whole chain topples in the pip half
    def chain(base, since):
        n = ctx.n_links
        tw, th_, step = 32, 64, 58
        x0 = CX - (n - 1) * step / 2 - tw / 2
        foot = PIP_Y + th_ / 2 + 8
        d = ImageDraw.Draw(base)
        lean = math.degrees(math.asin(min(1.0, (step - tw) / th_)))
        pop = ctx.ease_out_back(min(1.0, max(0.0, since + 0.25) / 0.15)) if since > -0.25 else 0.0
        if pop <= 0:
            pips(d, CX, PIP_Y, n, 12, 32, 22, IVORY, (6, 4, 6))
            return
        for j in reversed(range(n)):                                      # right to left: leaners on top
            t0 = 0.04 + j * min(0.08, 0.36 / n)
            u = min(1.0, max(0.0, (since - t0) / 0.12))
            ang = math.radians(lean * (1.0 if j < n - 1 else 1.6) * u ** 1.6)
            px, py = x0 + j * step + tw, foot                              # pivot: right foot

            def rot(a_, b_):
                dx, dy = (a_ - tw) * pop, -b_ * pop
                return px + dx * math.cos(ang) - dy * math.sin(ang), py + dx * math.sin(ang) + dy * math.cos(ang)
            body = [rot(0, 0), rot(tw, 0), rot(tw, th_), rot(0, th_)]
            d.polygon([(p[0] + 4, p[1] + 4) for p in body], fill=(4, 2, 4))
            d.polygon(body, fill=IVORY, outline=INK, width=3)
            d.line([rot(5, th_ / 2), rot(tw - 5, th_ / 2)], fill=INK, width=3)
            for half, cnt in ((1, j + 1), (0, 0 if j < n - 1 else 6)):
                for qu, qv in PIPS[min(6, cnt)]:
                    q = rot(tw / 2 + qu * 9, th_ / 4 + half * th_ / 2 + qv * 9)
                    d.ellipse([q[0] - 3.5, q[1] - 3.5, q[0] + 3.5, q[1] + 3.5], fill=INK)

    # ---------- frame
    def frame(seg, tt):
        i = seg["i"]
        if lineage(i):
            return ctx.frame_lineage(seg, tt)
        base = felt(tt)
        incut = tt - seg["t0"]
        fell = i > 0 and not lineage(i - 1)
        jolt = 0
        if fell and FALL * 0.75 <= incut < FALL * 0.75 + 0.16:              # the clack as the old tile lands
            q = (incut - FALL * 0.75) / 0.16
            jolt = int(-9 * math.sin(q * math.pi * 2.5) * (1 - q))
        c = ctx.draw_core(base, seg, tt, X0, Y0 + jolt, S, frame=tile_frame(i))
        if i == ctx.n_links - 1 and not sens(i):    # plain on a sensitive payoff
            since, _ = timing(seg, tt)
            chain(base, since)
        if fell and incut < FALL:
            falling(base, i - 1, max(0.0, incut) / FALL)
        return ctx.core_finish(base, seg, tt, c, credit_xy=(X0 + w, Y0 + jolt + h + 20))

    return frame
