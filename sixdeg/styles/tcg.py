"""Trading cards: each person is a holographic collectible card (art = the link photo, name, year, who they met,
rarity stars). B's card slides in face down and flips over at the reveal next to A's card; earlier cards go into a
fanned hand at the bottom. Foil sheen and a shine sweep keep every card moving."""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAME = "tcg"

CW, CH = 400, 560
SLOT_Y = 604
TAG_Y = 930
CAP_Y = 1110
HAND_SC = 0.5
PIVOT = (None, 1780)           # hand fan pivot (x = CX)
HAND_R = 470
RARITY = ["COMMON", "UNCOMMON", "RARE", "ULTRA RARE", "LEGENDARY"]
INKC = (28, 22, 34)


def make(ctx):
    F = {}

    def font(name, size):
        if (name, size) not in F:
            F[(name, size)] = ImageFont.truetype(str(ctx.FONTS / name), size)
        return F[(name, size)]

    def fit(name, text, size, width, lo=16):
        while font(name, size).getlength(text) > width and size > lo:
            size -= 1
        return font(name, size)

    st = {}
    SL = (ctx.CX - 215, ctx.CX + 215)          # left / right slot x

    def rmask(w, h, r):
        key = ("m", w, h, r)
        if key not in st:
            m = Image.new("L", (w * 2, h * 2), 0)
            ImageDraw.Draw(m).rounded_rectangle([0, 0, w * 2 - 1, h * 2 - 1], r * 2, fill=255)
            st[key] = m.resize((w, h), Image.LANCZOS)
        return st[key]

    def star(d, cx, cy, r, fill, outline=None):
        pts = [(cx + r * (1 if j % 2 == 0 else 0.45) * math.sin(j * math.pi / 5),
                cy - r * (1 if j % 2 == 0 else 0.45) * math.cos(j * math.pi / 5)) for j in range(10)]
        d.polygon(pts, fill=fill, outline=outline)

    def foil_strip():
        """A wide pastel rainbow with diagonal bands; cropped at a moving offset for the holo sheen."""
        if "strip" not in st:
            w, h = CW * 3, CH
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            hue = ((xx + yy * 0.6) / 520.0) % 1.0
            hh = hue * 6               # hue -> rgb (vectorised HSV, s = 0.55, v = 1)
            c = 0.55
            x = c * (1 - np.abs(hh % 2 - 1))
            z = np.zeros_like(hh)
            idx = hh.astype(int) % 6
            rr = np.select([idx == 0, idx == 1, idx == 2, idx == 3, idx == 4, idx == 5], [c, x, z, z, x, c])
            gg = np.select([idx == 0, idx == 1, idx == 2, idx == 3, idx == 4, idx == 5], [x, c, c, x, z, z])
            bb = np.select([idx == 0, idx == 1, idx == 2, idx == 3, idx == 4, idx == 5], [z, z, x, c, c, x])
            m = 1 - c
            rgb = np.dstack([rr + m, gg + m, bb + m]) * 255
            band = 0.55 + 0.45 * np.sin((xx - yy * 0.8) / 23.0)
            st["strip"] = rgb.astype(np.uint8)
            st["strip_a"] = band.astype(np.float32)
        return st["strip"], st["strip_a"]

    def shine_strip():
        if "shine" not in st:
            w = CW * 3
            xx = np.arange(w, dtype=np.float32)
            prof = np.exp(-((xx - w / 2) / 46.0) ** 2) * 230 + np.exp(-((xx - w / 2 - 90) / 14.0) ** 2) * 120
            st["shine"] = prof
        return st["shine"]

    def art(link, p):
        key = ("art", p)
        if key not in st:
            AW, AH = 348, 262
            if p == 0:
                im = ctx.view(0, *ctx.face_cam(ctx.fbox(0, "a"), AH, 0.5), AW, AH)
            else:
                im = ctx.view(link, *ctx.both_cam(link, AW, AH, 0.8), AW, AH)
            st[key] = im
        return st[key]

    def person(p):
        """Person p of the chain: p = 0 is A of link 0, p >= 1 is B of link p - 1."""
        if p == 0:
            return ctx.link_names(0)[0], 0, None
        a, b = ctx.link_names(p - 1)
        return b, p - 1, a

    def stars_of(p):
        return 5 if p == ctx.n_links else min(4, 1 + p * 3 // max(1, ctx.n_links))

    def front(p):
        """Static parts of person p's card (frame, foil panel, art, text) + the foil mask for the sheen."""
        key = ("front", p)
        if key not in st:
            nm, link, met = person(p)
            n_st = stars_of(p)
            im = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
            # metallic frame
            yy = np.linspace(0, 1, CH)[:, None]
            gold_hi, gold_lo = (np.array([255, 226, 130]), np.array([176, 120, 30])) if n_st < 5 else \
                (np.array([255, 160, 220]), np.array([110, 80, 230]))
            g = gold_hi * (1 - yy[..., None]) + gold_lo * yy[..., None]
            g = g + 30 * np.sin(np.linspace(0, 9, CW))[None, :, None]
            g = np.clip(g, 0, 255).astype(np.uint8)
            frame_im = Image.fromarray(np.dstack([g, np.full((CH, CW), 255, np.uint8)]))
            im.paste(frame_im, (0, 0), rmask(CW, CH, 24))
            d = ImageDraw.Draw(im)
            d.rounded_rectangle([12, 12, CW - 13, CH - 13], 16, fill=(214, 220, 232))      # foil panel (sheen goes here)
            # name bar
            d.rounded_rectangle([22, 22, CW - 23, 74], 12, fill=(250, 246, 232), outline=INKC, width=3)
            fn = fit("Inter-ExtraBold.otf", ctx.C.labels[nm], 32, CW - 110)
            d.text((36, 48), ctx.C.labels[nm], font=fn, fill=INKC, anchor="lm")
            # hp-style number on the right: the link number
            d.text((CW - 36, 48), f"{p:02d}", font=font("Inter-Black.otf", 30), fill=(200, 40, 60), anchor="rm")
            # art
            d.rectangle([22, 84, CW - 23, 84 + 262 + 8], fill=INKC)
            im.alpha_composite(art(link, p), (26, 88))
            # type line
            d.rounded_rectangle([30, 362, CW - 31, 394], 10, fill=(60, 52, 80))
            d.text((CW / 2, 378), "PHOTO CARD  ·  CONNECTOR" if p else "PHOTO CARD  ·  STARTER",
                   font=font("Inter-Bold.otf", 18), fill=(236, 230, 255), anchor="mm")
            # stats
            d.rounded_rectangle([22, 402, CW - 23, 492], 12, fill=(250, 246, 232), outline=INKC, width=3)
            yr = next((ctx.year_of(j) for j in range(link, -1, -1) if ctx.year_of(j)), None)   # "that year"
            d.text((38, 426), "YEAR", font=font("Inter-Black.otf", 20), fill=(150, 120, 60), anchor="lm")
            d.text((122, 426), str(yr) if yr else "?", font=font("Inter-Black.otf", 34), fill=INKC, anchor="lm")
            d.text((38, 468), "MET", font=font("Inter-Black.otf", 20), fill=(150, 120, 60), anchor="lm")
            mt = ctx.C.labels[met] if met else "-"
            d.text((122, 468), mt, font=fit("Inter-ExtraBold.otf", mt, 28, CW - 160), fill=INKC, anchor="lm")
            # rarity
            for k in range(5):
                star(d, 44 + k * 34, 522, 15, (255, 200, 40) if k < n_st else (120, 120, 140), outline=INKC)
            d.text((CW - 34, 522), RARITY[n_st - 1], font=font("Inter-Black.otf", 20),
                   fill=(200, 40, 60) if n_st >= 4 else INKC, anchor="rm")
            # foil mask: the panel only (not the art or the text boxes)
            fm = Image.new("L", (CW, CH), 0)
            fd = ImageDraw.Draw(fm)
            fd.rounded_rectangle([12, 12, CW - 13, CH - 13], 16, fill=255)
            for box in ([22, 22, CW - 23, 74], [22, 84, CW - 23, 354], [30, 362, CW - 31, 394], [22, 402, CW - 23, 492]):
                fd.rectangle(box, fill=0)
            st[key] = (im, np.asarray(fm, np.float32) / 255, np.asarray(rmask(CW, CH, 24), np.float32) / 255)
        return st[key]

    def back():
        if "back" not in st:
            im = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
            frame_im = Image.new("RGBA", (CW, CH), (236, 196, 80, 255))
            im.paste(frame_im, (0, 0), rmask(CW, CH, 24))
            d = ImageDraw.Draw(im)
            d.rounded_rectangle([12, 12, CW - 13, CH - 13], 16, fill=(30, 36, 110))
            for r in range(300, 0, -30):
                d.ellipse([CW / 2 - r, CH / 2 - r, CW / 2 + r, CH / 2 + r], outline=(54, 64, 160), width=8)
            d.rounded_rectangle([32, 32, CW - 33, CH - 33], 12, outline=(236, 196, 80), width=4)
            q = ctx.pixel_text("?", 150, fill=(255, 214, 10))
            im.alpha_composite(q, (int(CW / 2 - q.width / 2), int(CH / 2 - q.height / 2 - 30)))
            t = ctx.pixel_text("BUH", 40, fill=(255, 255, 255))
            im.alpha_composite(t, (int(CW / 2 - t.width / 2), int(CH - 120)))
            m = Image.new("L", (CW, CH), 0)
            ImageDraw.Draw(m).rounded_rectangle([12, 12, CW - 13, CH - 13], 16, fill=255)
            st["back"] = (im, np.asarray(m, np.float32) / 255 * 0.5, np.asarray(rmask(CW, CH, 24), np.float32) / 255)
        return st["back"]

    def live(layers, tt, phase):
        """Static card + moving holo sheen (foil only) + shine sweep (whole card)."""
        im, fmask, cmask = layers
        strip, sa = foil_strip()
        off = int((tt * 90 + phase * 300) % (CW * 2))
        rgb = strip[:, off:off + CW]
        a_holo = sa[:, off:off + CW] * fmask * 150
        sh = shine_strip()
        sweep = ((tt + phase) % 1.9) / 1.9                     # 0..1: band position across the card
        so = int(CW * 1.5 - (sweep * (CW * 2.2) - CW * 0.6))
        shear = (np.arange(CH)[:, None] * 0.45).astype(np.int32)
        cols = np.clip(so + np.arange(CW)[None, :] + shear - CH // 4, 0, sh.shape[0] - 1)
        a_sh = sh[cols] * cmask
        base = np.asarray(im, np.float32)
        out = base.copy()
        ah = (a_holo / 255)[..., None]
        out[..., :3] = out[..., :3] * (1 - ah) + rgb * ah
        ash = (a_sh / 255)[..., None]
        out[..., :3] = out[..., :3] + (255 - out[..., :3]) * ash
        return Image.fromarray(out.astype(np.uint8))

    def background():
        if "bg" not in st:
            yy, xx = np.mgrid[0:ctx.H, 0:ctx.W].astype(np.float32)
            r = np.sqrt(((xx - ctx.CX) / ctx.W) ** 2 + ((yy - 760) / ctx.H) ** 2)
            v = np.clip(1 - r * 1.4, 0, 1)[..., None]
            c = np.array([18, 16, 40]) * (1 - v) + np.array([58, 34, 96]) * v
            grid = (((xx + yy) % 90 < 3) | ((xx - yy) % 90 < 3))[..., None]
            c = np.where(grid, c * 1.25 + 6, c)
            st["bg"] = Image.fromarray(np.dstack([np.clip(c, 0, 255).astype(np.uint8),
                                                  np.full((ctx.H, ctx.W), 255, np.uint8)]))
            s = 900
            g = np.mgrid[0:s, 0:s].astype(np.float32)
            rr = np.sqrt((g[0] - s / 2) ** 2 + (g[1] - s / 2) ** 2) / (s / 2)
            a = (np.clip(1 - rr, 0, 1) ** 2 * 70).astype(np.uint8)
            st["spot"] = Image.fromarray(np.dstack([np.full((s, s), 150, np.uint8), np.full((s, s), 110, np.uint8),
                                                    np.full((s, s), 255, np.uint8), a]))
            rnd = random.Random(3)
            st["sparks"] = [(rnd.uniform(0, ctx.W), rnd.uniform(240, 1460), rnd.uniform(0, 6.3), rnd.uniform(5, 13))
                            for _ in range(36)]
        return st["bg"]

    def sparkle(d, x, y, r, a):
        c = (255, 240, 200, int(255 * a))
        d.polygon([(x, y - r), (x + r * 0.25, y - r * 0.25), (x + r, y), (x + r * 0.25, y + r * 0.25), (x, y + r),
                   (x - r * 0.25, y + r * 0.25), (x - r, y), (x - r * 0.25, y - r * 0.25)], fill=c)

    def hand_pose(k, n):
        """Card k of a fanned hand of n cards: (x, y, angle)."""
        spread = min(13.0, 56.0 / max(1, n - 1)) if n > 1 else 0
        ang = (k - (n - 1) / 2) * spread
        a = math.radians(ang)
        return ctx.CX + HAND_R * math.sin(a), PIVOT[1] - HAND_R * math.cos(a), -ang

    def put(base, im, x, y, sc, ang, sx=1.0, alpha=1.0):
        w, h = max(1, int(im.width * sc * sx)), max(1, int(im.height * sc))
        im = im.resize((w, h), Image.BILINEAR)
        if abs(ang) > 0.05:
            im = im.rotate(ang, expand=True, resample=Image.BICUBIC)
        ctx.paste(base, im, x, y, alpha=alpha)

    def small(p):
        if ("small", p) not in st:
            im = front(p)[0]
            st[("small", p)] = im.resize((int(CW * HAND_SC), int(CH * HAND_SC)), Image.LANCZOS)
        return st[("small", p)]

    def tag(text, colr):
        key = ("tag", text, colr)
        if key not in st:
            size = 24
            t = ctx.pixel_tag(text, colr, size)
            while t.width > 410 and size > 12:
                size -= 2
                t = ctx.pixel_tag(text, colr, size)
            st[key] = t
        return st[key]

    def lerp(a, b, k):
        return a + (b - a) * k

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        sp, R = seg["speech"], seg.get("rv", ctx.REVEAL)
        rvt = R * sp + ctx.THROW
        u = min(1.0, rvt / 1.45)
        base = background().copy()
        sx_ = ctx.CX + 260 * math.sin(tt * 0.7)
        sy_ = 700 + 180 * math.cos(tt * 0.9)
        base.alpha_composite(st["spot"], (int(sx_ - 450), int(sy_ - 450)))
        d = ImageDraw.Draw(base, "RGBA")
        for (x, y, ph, r) in st["sparks"]:
            a = max(0.0, math.sin(tt * 2.2 + ph)) ** 3
            if a > 0.05:
                sparkle(d, x, y - (tt * 20 % 60), r * (0.6 + 0.4 * a), a * 0.8)
        tr = ctx.ease(local / 0.5) if i > 0 else 1.0          # the previous pair moves on
        # ---- hand: persons 0..i-1 (person i-1 is still flying in during the transition)
        n_new = i
        for k in range(n_new):
            hx, hy, ha = hand_pose(k, n_new)
            if k < i - 1:
                ox, oy, oa = hand_pose(k, i - 1)
                hx, hy, ha = lerp(ox, hx, tr), lerp(oy, hy, tr), lerp(oa, ha, tr)
                bob = 6 * math.sin(tt * 2.4 + k)
                put(base, small(k), hx, hy + bob, 1.0, ha + 1.2 * math.sin(tt * 1.7 + k))
            else:      # the card that just left the left slot
                lx, ly = SL[0], SLOT_Y
                x, y = lerp(lx, hx, tr), lerp(ly, hy, tr) - 160 * math.sin(math.pi * tr)
                sc = lerp(1.0, HAND_SC, tr)
                lay = front(k)
                im = live(lay, tt, k * 0.37) if tr < 1 else small(k)
                put(base, im, x, y, sc if tr < 1 else 1.0, lerp(-3, ha, tr))
        # ---- left slot: person i (for i > 0 it slides over from the right slot)
        wob = math.sin(tt * 2.1)
        if i == 0:
            k0 = ctx.ease_out_back(local / 0.5)
            lx, ly, la = lerp(-300, SL[0], k0), lerp(-200, SLOT_Y, k0), lerp(-40, -3, k0)
        else:
            lx, ly, la = lerp(SL[1], SL[0], tr), SLOT_Y - 70 * math.sin(math.pi * tr), lerp(3, -3, tr)
        put(base, live(front(i), tt, i * 0.37), lx, ly + 8 * math.sin(tt * 1.9), 1.0, la + 1.6 * wob)
        # ---- right slot: person i + 1 slides in face down, flips at the reveal
        t_in = 0.42 * u
        if local > t_in:
            kin = ctx.ease_out_back(min(1.0, (local - t_in) / (0.5 * u)))
            rx = lerp(ctx.W + 320, SL[1], kin)
            ry = SLOT_Y + 8 * math.sin(tt * 1.9 + 1.3)
            ra = lerp(-30, 3, kin) - 1.6 * wob
            fk = (local - rvt) / 0.24
            if fk < 0.5:
                sxk = 1.0 if fk < 0 else math.cos(math.pi * fk)      # 1 -> 0
                put(base, live(back(), tt, 0.5), rx, ry, 1.0, ra, sx=max(0.02, sxk))
            else:
                sxk = -math.cos(math.pi * min(1.0, fk))              # 0 -> 1
                pop = 1 + 0.08 * math.sin(math.pi * min(1.0, max(0.0, (local - rvt - 0.12) / 0.4)))
                put(base, live(front(i + 1), tt, (i + 1) * 0.37), rx, ry, pop, ra, sx=max(0.02, sxk))
                if fk < 4:   # sparkle burst at the flip
                    bk = (local - rvt - 0.12) / 0.6
                    if 0 < bk < 1:
                        dd = ImageDraw.Draw(base, "RGBA")
                        for j in range(14):
                            ang = j * 2 * math.pi / 14
                            rr = 120 + 260 * ctx.ease(bk)
                            sparkle(dd, rx + rr * math.cos(ang), ry + rr * math.sin(ang) * 1.2, 18 * (1 - bk) + 4, 1 - bk)
            if local >= rvt + 0.12:
                bn = person(i + 1)[0]
                ctx.paste(base, tag(ctx.C.labels[bn].upper(), ctx.YELLOW), SL[1], TAG_Y + 4 * math.sin(tt * 3 + 1),
                          scale=ctx.ease_out_back((local - rvt - 0.12) / 0.2))
        an = person(i)[0]
        ctx.paste(base, tag(ctx.C.labels[an].upper(), ctx.CYAN), lx, TAG_Y + 4 * math.sin(tt * 3),
                  alpha=min(1.0, max(0.0, (local - 0.3) / 0.2)) if i > 0 else 1.0)
        ctx.header(base, i)
        ctx.connected(base, seg, tt)
        seg["cap"].draw(base, tt, y=CAP_Y)
        return base

    return frame
