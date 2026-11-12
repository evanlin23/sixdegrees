"""Trading cards: every link is dealt as one big collectible photo card (art = the link photo, name bar, year, who it
was taken with). B's name on the card is "? ? ?" until the reveal, then it pops in; finished cards fly up into a
collection row of mini cards under the title. Foil sheen and a glare sweep run over the card frame only (never the
faces). A link with a dictator gets a plain archive card: no foil, no glare, no stars, no rarity."""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAME = "tcg"

HW, HH = 760, 900              # hero card
HERO_TOP = 405
HERO_CY = HERO_TOP + HH / 2
RING = 22                      # frame width (where the foil and glare live)
ART = (28, 110, 732, 620)      # art box inside the card
MW, MH = 96, 130               # mini cards in the collection row
ROW_Y = 318
ROW_DX = 116
CAP_Y = 1378
SWAP = 0.26                    # seconds for the old card to fly up and the new one to deal in (before the caption)
RARITY = ["COMMON", "UNCOMMON", "RARE", "ULTRA RARE", "LEGENDARY"]
INKC = (28, 22, 34)


def photo_params(ctx, i, w, h, fill=0.8, maxup=1.3):
    """(cx, cy, scale, ow, oh) for link photo i in a w x h window: both faces in, cover-cropped where the source
    allows, never upscaled beyond maxup (a small source gives ow, oh < w, h and gets a dark matte)."""
    ih, iw = ctx.C.imgs[i].shape[:2]
    cx, cy, s = ctx.both_cam(i, w, h, fill)
    ba, bb = ctx.fbox(i, "a"), ctx.fbox(i, "b")
    spread = max(ba[0] + ba[2], bb[0] + bb[2]) - min(ba[0], bb[0])
    s = min(maxup, max(s, max(w / iw, h / ih)), 0.95 * w / max(1, spread))
    ow, oh = min(w, int(iw * s)), min(h, int(ih * s))
    hw, hh = ow / 2 / s, oh / 2 / s
    cx, cy = min(max(cx, hw), iw - hw), min(max(cy, hh), ih - hh)
    return cx, cy, s, ow, oh


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
    N = ctx.n_links

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

    def calm(i):
        """Link i involves a dictator: plain archive card."""
        a, b = ctx.link_names(i)
        return ctx.sensitive(a) or ctx.sensitive(b)

    def stars_of(i):
        return 5 if i == N - 1 else min(4, 1 + (i + 1) * 3 // max(1, N))

    def foil_strip():
        """A wide pastel rainbow with diagonal bands; cropped at a moving offset for the holo sheen."""
        if "strip" not in st:
            w, h = HW * 3, HH
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            hh = ((xx + yy * 0.6) / 520.0) % 1.0 * 6
            c = 0.55
            x = c * (1 - np.abs(hh % 2 - 1))
            z = np.zeros_like(hh)
            k = hh.astype(int) % 6
            sel = [k == j for j in range(6)]
            rr = np.select(sel, [c, x, z, z, x, c])
            gg = np.select(sel, [x, c, c, x, z, z])
            bb = np.select(sel, [z, z, x, c, c, x])
            st["strip"] = (np.dstack([rr, gg, bb]) + (1 - c)) * 255
            st["strip_a"] = (0.55 + 0.45 * np.sin((xx - yy * 0.8) / 23.0)).astype(np.float32)
            xs = np.arange(w, dtype=np.float32)
            st["shine"] = np.exp(-((xs - w / 2) / 60.0) ** 2) * 230 + np.exp(-((xs - w / 2 - 110) / 18.0) ** 2) * 120
        return st["strip"], st["strip_a"], st["shine"]

    def tags_for(i, revealed):
        """Name chips under each face along the bottom of the art: [(img, x)] in card coordinates."""
        cx, cy, s, ow, oh = photo_params(ctx, i, ART[2] - ART[0] - 8, ART[3] - ART[1] - 8)
        acx = (ART[0] + ART[2]) / 2
        an, bn = ctx.link_names(i)
        items = []
        for role, nm, colr in (("a", an, ctx.CYAN), ("b", bn, ctx.YELLOW)):
            label = ctx.C.labels[nm].upper() if role == "a" or revealed else "? ? ?"
            im = ctx.pixel_tag(label, colr if (role == "a" or revealed) else (190, 186, 200), 22)
            fx = acx + (ctx.ctr(ctx.fbox(i, role))[0] - cx) * s
            items.append([im, fx])
        items.sort(key=lambda q: q[1])
        lo, hi = ART[0] + 6, ART[2] - 6
        for q in items:
            q[1] = max(lo + q[0].width / 2, min(hi - q[0].width / 2, q[1]))
        (l, xl), (r, xr) = items
        gap = xr - xl - (l.width + r.width) / 2 - 10
        if gap < 0:
            xl, xr = xl + gap / 2, xr - gap / 2
            xl = max(lo + l.width / 2, xl)
            xr = min(hi - r.width / 2, xr)
            if xr - xl < (l.width + r.width) / 2 + 6:      # still too tight: centre them side by side
                tot = l.width + r.width + 10
                xl = (lo + hi) / 2 - tot / 2 + l.width / 2
                xr = xl + l.width / 2 + 10 + r.width / 2
        return [(l, xl), (r, xr)]

    def hero(i, revealed, fin=False):
        """The static card for link i + its frame-ring mask (foil and glare go there only)."""
        key = ("hero", i, revealed, fin)
        if key in st:
            return st[key]
        plain = calm(i)
        an, bn = ctx.link_names(i)
        n_st = stars_of(i)
        im = Image.new("RGBA", (HW, HH), (0, 0, 0, 0))
        yy = np.linspace(0, 1, HH)[:, None, None]
        if plain:
            hi_, lo_ = np.array([150, 152, 160]), np.array([96, 98, 108])
        elif n_st >= 5:
            hi_, lo_ = np.array([255, 160, 220]), np.array([110, 80, 230])
        else:
            hi_, lo_ = np.array([255, 226, 130]), np.array([176, 120, 30])
        g = hi_ * (1 - yy) + lo_ * yy
        if not plain:
            g = g + 30 * np.sin(np.linspace(0, 9, HW))[None, :, None]
        g = np.clip(np.broadcast_to(g, (HH, HW, 3)), 0, 255).astype(np.uint8)
        im.paste(Image.fromarray(np.dstack([g, np.full((HH, HW), 255, np.uint8)])), (0, 0), rmask(HW, HH, 34))
        d = ImageDraw.Draw(im)
        panel = (226, 226, 230) if plain else (214, 220, 232)
        d.rounded_rectangle([RING - 6, RING - 6, HW - RING + 5, HH - RING + 5], 22, fill=panel)
        # name bar: B (or ? ? ?) and the link number
        d.rounded_rectangle([28, 28, HW - 29, 100], 14, fill=(250, 246, 232), outline=INKC, width=3)
        nm = ctx.C.labels[bn] if revealed else "? ? ?"
        d.text((48, 64), nm, font=fit("Inter-ExtraBold.otf", nm, 50, HW - 220), fill=INKC, anchor="lm")
        d.text((HW - 48, 64), f"{i + 1}/{N}", font=font("Inter-Black.otf", 40), fill=(200, 40, 60), anchor="rm")
        # art
        d.rectangle(ART, fill=INKC)
        aw, ah = ART[2] - ART[0] - 8, ART[3] - ART[1] - 8
        cx, cy, s, ow, oh = photo_params(ctx, i, aw, ah)
        im.alpha_composite(ctx.view(i, cx, cy, s, ow, oh), (ART[0] + 4 + (aw - ow) // 2, ART[1] + 4 + (ah - oh) // 2))
        for tg, x in tags_for(i, revealed):
            im.alpha_composite(tg, (int(x - tg.width / 2), int(ART[3] - tg.height - 10)))
        # type line
        d.rounded_rectangle([40, 632, HW - 41, 672], 12, fill=(70, 72, 84) if plain else (60, 52, 80))
        d.text((HW / 2, 652), "ARCHIVE PHOTO  ·  CONNECTOR" if plain else "PHOTO CARD  ·  CONNECTOR",
               font=font("Inter-Bold.otf", 24), fill=(236, 234, 244), anchor="mm")
        if not fin:
            d.rounded_rectangle([28, 684, HW - 29, 800], 14, fill=(250, 246, 232), outline=INKC, width=3)
            yr = next((ctx.year_of(j) for j in range(i, -1, -1) if ctx.year_of(j)), None)   # "that year"
            d.text((52, 716), "YEAR", font=font("Inter-Black.otf", 24), fill=(140, 120, 80), anchor="lm")
            d.text((170, 716), str(yr) if yr else "?", font=font("Inter-Black.otf", 40), fill=INKC, anchor="lm")
            d.text((52, 768), "WITH", font=font("Inter-Black.otf", 24), fill=(140, 120, 80), anchor="lm")
            mt = ctx.C.labels[an]
            d.text((170, 768), mt, font=fit("Inter-ExtraBold.otf", mt, 36, HW - 230), fill=INKC, anchor="lm")
            if plain:
                d.text((HW / 2, 840), "HISTORICAL RECORD", font=font("Inter-Bold.otf", 24), fill=(80, 80, 90),
                       anchor="mm")
            else:
                for k in range(5):
                    star(d, 66 + k * 50, 840, 22, (255, 200, 40) if k < n_st else (120, 120, 140), outline=INKC)
                d.text((HW - 48, 840), RARITY[n_st - 1], font=font("Inter-Black.otf", 28),
                       fill=(200, 40, 60) if n_st >= 4 else INKC, anchor="rm")
        ring = Image.new("L", (HW, HH), 0)
        rd = ImageDraw.Draw(ring)
        rd.rounded_rectangle([0, 0, HW - 1, HH - 1], 34, fill=255)
        rd.rounded_rectangle([RING - 6, RING - 6, HW - RING + 5, HH - RING + 5], 22, fill=0)
        st[key] = (im, np.asarray(ring, np.float32) / 255, plain)
        return st[key]

    def live(layers, tt, phase):
        """The card with a moving holo sheen and a glare sweep, both on the frame ring only; plain cards stay still."""
        im, ring, plain = layers
        if plain:
            return im
        strip, sa, sh = foil_strip()
        off = int((tt * 90 + phase * 300) % (HW * 2))
        a_holo = sa[:, off:off + HW] * ring * 150
        sweep = ((tt + phase) % 2.2) / 2.2
        so = int(HW * 1.5 - (sweep * (HW * 2.2) - HW * 0.6))
        shear = (np.arange(HH)[:, None] * 0.45).astype(np.int32)
        cols = np.clip(so + np.arange(HW)[None, :] + shear - HH // 4, 0, sh.shape[0] - 1)
        a_sh = sh[cols] * ring
        out = np.asarray(im, np.float32).copy()
        ah = (a_holo / 255)[..., None]
        out[..., :3] = out[..., :3] * (1 - ah) + strip[:, off:off + HW] * ah
        ash = (a_sh / 255)[..., None]
        out[..., :3] = out[..., :3] + (255 - out[..., :3]) * ash
        return Image.fromarray(out.astype(np.uint8))

    def mini(p):
        """Collection-row card for person p (p = 0 is A of link 0, p >= 1 is B of link p - 1); None = empty slot."""
        key = ("mini", p)
        if key not in st:
            im = Image.new("RGBA", (MW, MH), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            if p is None:
                d.rounded_rectangle([2, 2, MW - 3, MH - 3], 10, outline=(110, 96, 150), width=3)
                d.text((MW / 2, MH / 2), "?", font=font("Inter-Black.otf", 40), fill=(110, 96, 150), anchor="mm")
            elif ctx.sensitive(ctx.link_names(0)[0] if p == 0 else ctx.link_names(p - 1)[1]):
                # a dictator is never a collectible: a neutral marker instead of a card
                nm = ctx.link_names(0)[0] if p == 0 else ctx.link_names(p - 1)[1]
                d.rounded_rectangle([6, 22, MW - 7, MH - 23], 8, fill=(78, 76, 90))
                if p == 0:
                    d.text((MW / 2, MH / 2), "START", font=fit("Inter-Black.otf", "START", 22, MW - 20, 10),
                           fill=(214, 212, 222), anchor="mm")
                else:
                    face = ctx.sharp[nm].convert("L").convert("RGBA").resize((MW - 28, MW - 28), Image.LANCZOS)
                    im.alpha_composite(face, (14, (MH - (MW - 28)) // 2))
            else:
                nm = ctx.link_names(0)[0] if p == 0 else ctx.link_names(p - 1)[1]
                plain = calm(0 if p == 0 else p - 1)
                d.rounded_rectangle([0, 0, MW - 1, MH - 1], 10,
                                    fill=(140, 142, 150) if plain else (236, 196, 80), outline=INKC, width=2)
                face = ctx.sharp[nm].convert("RGBA").resize((MW - 16, MW - 16), Image.LANCZOS)
                im.alpha_composite(face, (8, 8))
                d.rectangle([8, 8, MW - 9, MW - 9], outline=INKC, width=2)
                words = ctx.C.labels[nm].split()
                short = (words[-2] if len(words) > 1 and set(words[-1]) <= set("IVX") else words[-1]).upper()
                d.text((MW / 2, MW + 8 + (MH - MW - 8) / 2), short, font=fit("Inter-Black.otf", short, 18, MW - 10, 10),
                       fill=INKC, anchor="mm")
            st[key] = im
        return st[key]

    def slot_x(p):
        return ctx.CX + (p - N / 2) * ROW_DX

    def background():
        if "bg" not in st:
            yy, xx = np.mgrid[0:ctx.H, 0:ctx.W].astype(np.float32)
            r = np.sqrt(((xx - ctx.CX) / ctx.W) ** 2 + ((yy - 860) / ctx.H) ** 2)
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
            st["sparks"] = [(rnd.uniform(0, ctx.W), rnd.uniform(240, 1900), rnd.uniform(0, 6.3), rnd.uniform(5, 13))
                            for _ in range(40)]
        return st["bg"]

    def sparkle(d, x, y, r, a):
        c = (255, 240, 200, int(255 * a))
        d.polygon([(x, y - r), (x + r * 0.25, y - r * 0.25), (x + r, y), (x + r * 0.25, y + r * 0.25), (x, y + r),
                   (x - r * 0.25, y + r * 0.25), (x - r, y), (x - r * 0.25, y - r * 0.25)], fill=c)

    def put(base, im, x, y, sc, ang, alpha=1.0):
        w, h = max(1, int(im.width * sc)), max(1, int(im.height * sc))
        if (w, h) != im.size:
            im = im.resize((w, h), Image.BILINEAR)
        if abs(ang) > 0.05:
            im = im.rotate(ang, expand=True, resample=Image.BICUBIC)
        ctx.paste(base, im, x, y, alpha=alpha)

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        sp, R = seg["speech"], seg.get("rv", ctx.REVEAL)
        rvt = R * sp + ctx.THROW
        revealed = local > rvt
        last = i == N - 1
        base = background().copy()
        sx_ = ctx.CX + 260 * math.sin(tt * 0.7)
        sy_ = 860 + 200 * math.cos(tt * 0.9)
        base.alpha_composite(st["spot"], (int(sx_ - 450), int(sy_ - 450)))
        d = ImageDraw.Draw(base, "RGBA")
        for (x, y, ph, r) in st["sparks"]:
            a = max(0.0, math.sin(tt * 2.2 + ph)) ** 3
            if a > 0.05:
                sparkle(d, x, y - (tt * 20 % 60), r * (0.6 + 0.4 * a), a * 0.8)

        # ---- collection row: person 0 .. i are collected (person i arrives with the previous card's fly-up)
        tr = ctx.ease(max(0.0, min(1.0, local / SWAP))) if i > 0 else 1.0
        for p in range(N + 1):
            x = slot_x(p)
            have = p < i or (p == i and (i == 0 or tr >= 1)) or (last and revealed and p == N)
            put(base, mini(p if have else None), x, ROW_Y + 3 * math.sin(tt * 2 + p), 1.0, 0)
        # ---- the hero card of this link deals in from the right
        wob = 1.0 * math.sin(tt * 1.6)
        if i == 0:                             # straight after the hook: grows in place, no empty screen
            k_in = 1.0
            sc0 = 0.9 + 0.1 * ctx.ease(min(1.0, local / 0.25))
        else:
            k_in = ctx.ease(max(0.0, min(1.0, local / SWAP)))
            sc0 = 1.0
        hx = ctx.CX + (1 - k_in) * 900
        hy = HERO_CY + 6 * math.sin(tt * 1.9)
        pop = 1.0
        if revealed:
            pop = 1 + 0.04 * math.sin(math.pi * min(1.0, (local - rvt) / 0.3))
        lay = hero(i, revealed, fin=last and revealed)
        put(base, live(lay, tt, i * 0.37), hx, hy, pop * sc0, wob * k_in,
            alpha=min(1.0, 0.3 + local / 0.15) if i == 0 else 1.0)
        if revealed and not calm(i):           # a sparkle burst when the name lands (not on a dictator's card)
            bk = (local - rvt) / 0.6
            if 0 < bk < 1:
                for j in range(14):
                    ang = j * 2 * math.pi / 14
                    rr = 300 + 200 * ctx.ease(bk)
                    sparkle(d, hx + rr * math.cos(ang), hy - 330 + rr * 0.3 * math.sin(ang), 18 * (1 - bk) + 4, 1 - bk)
        # ---- the previous card shrinks and flies up into its slot (above everything, away from the caption)
        if i > 0 and tr < 1 and not ctx.C.fj["links"][i - 1].get("lineage"):
            sc = 1 + (MW / HW - 1) * tr
            x = ctx.CX + (slot_x(i) - ctx.CX) * tr
            y = HERO_CY + (ROW_Y - HERO_CY) * tr
            put(base, live(hero(i - 1, True), tt, (i - 1) * 0.37), x, y, sc, 0)

        # ---- Buh peeks up from below, reacting at the reveal
        rv = local - rvt
        expr = "happy" if rv < 0 else "shock" if rv < 0.6 else "smug"
        hop = 30 * math.sin(math.pi * min(1.0, rv / 0.35)) if 0 < rv < 0.35 else 0.0
        ctx.paste_pin(base, ctx.pin_sprite(expr, 5 * math.sin(tt * 2.5), 0, 1, scale=7), 250,
                      1770 - hop + 8 * math.sin(tt * 3))
        ctx.header(base, i)
        if last and revealed:
            # CONNECTED! drawn on its own layer and shrunk so it fits the card's inner panel, never the foil rim
            lay_c = Image.new("RGBA", (ctx.W, ctx.H), (0, 0, 0, 0))
            cy_ = 960
            ctx.connected(lay_c, seg, tt, y=cy_)
            box = lay_c.crop((0, cy_ - 160, ctx.W, cy_ + 160))
            k = 0.8
            box = box.resize((int(box.width * k), int(box.height * k)), Image.BICUBIC)
            ctx.paste(base, box, ctx.CX + (ctx.W / 2 - ctx.CX) * k, hy - HH / 2 + 742, nearest=False)
        if i == 0 or local >= SWAP:                      # the caption waits until the cards have settled
            seg["cap"].draw(base, tt, y=CAP_Y)
        return base

    return frame
