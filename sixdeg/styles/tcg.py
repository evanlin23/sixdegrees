"""Trading cards: every link is dealt as one big collectible photo card whose art window IS the show's core
(draw_core: the camera moves, Buh jumps head to head with the red string, the name tags ride the photo). The name
bar reads "? ? ?" until the reveal, then B's name pops in. When the next link starts, the finished card shrinks and
flies down into a fanned hand of mini cards at the bottom of the screen, which keeps growing through the episode.
Foil sheen and a glare sweep run over the card frame only (never the art). The card is dressed like a real collectible:
an ornate double-keyline border with gold corner filigree, a holo lattice pattern in the frame ring, an embossed
corner medallion (Buh's pin), a fact box under the name (the link's fact, cut before B until the reveal) and a set
symbol. A link with a dictator gets a plain grey archive card: no foil, no glare, no stars, no rarity, no gold
ornaments, and its mini card in the hand stays grey."""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAME = "tcg"

SC = 0.70                      # core window scale: art = 616 x 665 (the fact box needs the room)
PAD = 30                       # card frame around the art (sides)
TOPBAR = 66                    # type line above the art
FOOT = 126                     # name bar + fact box under the art
RING = 20                      # the foil frame ring
CARD_TOP = 404
SWAP = 0.45                    # seconds: the old card flies into the hand while the new one deals in
MINI = 0.36                    # mini card scale in the hand
PIVOT_Y = 2330                 # the hand fans around a point below the screen
HAND_R = 790
RARITY = ["COMMON", "UNCOMMON", "RARE", "ULTRA RARE", "LEGENDARY"]
INKC = (28, 22, 34)


def make(ctx):
    W, H = ctx.W, ctx.H
    N = ctx.n_links
    aw, ah = int(880 * SC), int(950 * SC)
    CW, CH = aw + 2 * PAD, TOPBAR + ah + FOOT
    CX0 = int(ctx.CX - CW / 2)
    ART_X, ART_Y = PAD, TOPBAR                         # art window inside the card
    st = {}

    def font(name, size):
        if (name, size) not in st:
            st[(name, size)] = ImageFont.truetype(str(ctx.FONTS / name), size)
        return st[(name, size)]

    def fit(name, text, size, width, lo=16):
        while font(name, size).getlength(text) > width and size > lo:
            size -= 1
        return font(name, size)

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
            w, h = CW * 3, CH
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

    def emblem(plain):
        """The embossed corner medallion: a raised gold disc with Buh's pin pressed into it."""
        key = ("emb", plain)
        if key in st:
            return st[key]
        r, S2 = 27, 4                                   # drawn 4x, downsampled (smooth emboss)
        D = (2 * r + 6) * S2
        im = Image.new("RGBA", (D, D), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        c = D / 2
        for k in range(r * S2, 0, -S2):                 # a domed gold gradient, lit from the top left
            u = k / (r * S2)
            col = tuple(int(a_ * (1 - u) + b_ * u) for a_, b_ in zip((255, 236, 160), (150, 96, 20)))
            o = (1 - u) * 5 * S2
            d.ellipse([c - k - o, c - k - o, c + k - o, c + k - o], fill=col)
        d.ellipse([c - r * S2, c - r * S2, c + r * S2, c + r * S2], outline=INKC, width=3 * S2)
        d.ellipse([c - (r - 6) * S2, c - (r - 6) * S2, c + (r - 6) * S2, c + (r - 6) * S2], outline=(120, 76, 14), width=S2)

        def pin(dx, dy, col):                           # Buh's pin, pressed in
            hx, hy = c + dx, c - 6 * S2 + dy
            d.ellipse([hx - 9 * S2, hy - 9 * S2, hx + 9 * S2, hy + 9 * S2], fill=col)
            d.rectangle([hx - 12 * S2, hy + 6 * S2, hx + 12 * S2, hy + 10 * S2], fill=col)
            d.polygon([(hx - 2 * S2, hy + 10 * S2), (hx + 2 * S2, hy + 10 * S2), (hx, hy + 22 * S2)], fill=col)
        pin(-S2 * 1.5, -S2 * 1.5, (255, 248, 210))      # light edge up-left, dark edge down-right: embossed
        pin(S2 * 1.5, S2 * 1.5, (96, 56, 8))
        pin(0, 0, (214, 160, 52))
        st[key] = im.resize((D // S2, D // S2), Image.LANCZOS)
        return st[key]

    def set_symbol(d, cx, cy, plain):
        """The set symbol: two linked rings (the show's chain) in a small black shield."""
        d.rounded_rectangle([cx - 17, cy - 13, cx + 17, cy + 13], 8, fill=INKC)
        col = (190, 190, 200) if plain else (255, 214, 90)
        d.ellipse([cx - 13, cy - 7, cx + 3, cy + 7], outline=col, width=3)
        d.ellipse([cx - 3, cy - 7, cx + 13, cy + 7], outline=col, width=3)

    def filigree(d, x, y, sx, sy, col, dark):
        """A gold corner rosette on the ring at (x, y), with beads running along the ring on both edges."""
        for k in range(3):
            r = 5 - k
            ax = x + sx * (20 + 13 * k)
            d.ellipse([ax - r, y - r, ax + r, y + r], fill=col, outline=dark)
            by = y + sy * (20 + 13 * k)
            d.ellipse([x - r, by - r, x + r, by + r], fill=col, outline=dark)
        d.ellipse([x - 12, y - 12, x + 12, y + 12], fill=col, outline=dark, width=2)
        for k in range(8):                              # petals
            a = k * math.pi / 4
            px, py = x + 7 * math.cos(a), y + 7 * math.sin(a)
            d.ellipse([px - 2, py - 2, px + 2, py + 2], fill=dark)
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=dark)

    def holo_pattern():
        """Holo lattice for the frame ring: diamonds with little stars at the crossings (0.45 .. 1)."""
        if "pat" not in st:
            yy, xx = np.mgrid[0:CH, 0:CW].astype(np.float32)
            u, v = (xx + yy) % 34, (xx - yy) % 34
            lattice = ((np.abs(u - 17) < 2.2) | (np.abs(v - 17) < 2.2)).astype(np.float32)
            starc = np.exp(-(np.minimum(u, 34 - u) ** 2 + np.minimum(v, 34 - v) ** 2) / 10.0)
            st["pat"] = np.clip(0.45 + 0.35 * lattice + 0.6 * starc, 0, 1)
        return st["pat"]

    def card(i, revealed):
        """The static card for link i with an empty art window, + its frame-ring mask (foil and glare go there)."""
        key = ("card", i, revealed)
        if key in st:
            return st[key]
        plain = calm(i)
        an, bn = ctx.link_names(i)
        n_st = stars_of(i)
        im = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
        yy = np.linspace(0, 1, CH)[:, None, None]
        if plain:
            hi_, lo_ = np.array([150, 152, 160]), np.array([96, 98, 108])
        elif n_st >= 5:
            hi_, lo_ = np.array([255, 160, 220]), np.array([110, 80, 230])
        else:
            hi_, lo_ = np.array([255, 226, 130]), np.array([176, 120, 30])
        g = hi_ * (1 - yy) + lo_ * yy
        if not plain:
            g = g + 30 * np.sin(np.linspace(0, 9, CW))[None, :, None]
        g = np.clip(np.broadcast_to(g, (CH, CW, 3)), 0, 255).astype(np.uint8)
        im.paste(Image.fromarray(np.dstack([g, np.full((CH, CW), 255, np.uint8)])), (0, 0), rmask(CW, CH, 30))
        d = ImageDraw.Draw(im)
        R0 = RING
        # the inner panel: a soft gradient with a fine engraved cross-hatch (flat grey when plain)
        if plain:
            d.rounded_rectangle([R0, R0, CW - 1 - R0, CH - 1 - R0], 18, fill=(226, 226, 230))
        else:
            py, px = np.mgrid[0:CH, 0:CW].astype(np.float32)
            hatch = ((((px + py) % 9) < 1) | (((px - py) % 9) < 1)).astype(np.float32)
            top, bot = np.array([238, 240, 248], np.float32), np.array([202, 208, 226], np.float32)
            pan = top * (1 - py[..., None] / CH) + bot * (py[..., None] / CH) - 12 * hatch[..., None]
            pm = Image.new("L", (CW, CH), 0)
            ImageDraw.Draw(pm).rounded_rectangle([R0, R0, CW - 1 - R0, CH - 1 - R0], 18, fill=255)
            im.paste(Image.fromarray(np.clip(pan, 0, 255).astype(np.uint8)).convert("RGBA"), (0, 0), pm)
        # double keyline: a dark engraved line just inside the ring and a fine coloured one inside that
        d.rounded_rectangle([R0, R0, CW - 1 - R0, CH - 1 - R0], 18, outline=INKC, width=3)
        d.rounded_rectangle([R0 + 6, R0 + 6, CW - 7 - R0, CH - 7 - R0], 13,
                            outline=(170, 170, 178) if plain else (190, 150, 70), width=2)
        yr = next((ctx.year_of(j) for j in range(i, -1, -1) if ctx.year_of(j)), None)   # "that year"
        # type line on top (Buh may stand in front of it): medallion, card number, kind, year, stars + rarity
        my = (R0 + 4 + TOPBAR - 8) / 2
        d.rounded_rectangle([28, R0 + 4, CW - 29, TOPBAR - 8], 12, fill=(70, 72, 84) if plain else (60, 52, 80),
                            outline=None if plain else (190, 150, 70), width=2)
        tx = 42
        if not plain:
            em = emblem(False)
            im.alpha_composite(em, (6, int(my - em.height / 2)))
            tx = 6 + em.width + 2
        d.text((tx, my), f"#{i + 1:02d}", font=font("Inter-Black.otf", 26),
               fill=(200, 200, 210) if plain else (255, 120, 140), anchor="lm")
        if plain:
            note = f"ARCHIVE PHOTO  ·  {yr}  ·  HISTORICAL RECORD" if yr else "ARCHIVE PHOTO  ·  HISTORICAL RECORD"
            d.text((CW / 2 + 30, my), note, font=font("Inter-Bold.otf", 22), fill=(236, 234, 244), anchor="mm")
        else:
            d.text((tx + 56, my), f"PHOTO CARD · {yr}" if yr else "PHOTO CARD", font=font("Inter-Bold.otf", 20),
                   fill=(236, 234, 244), anchor="lm")
            for k in range(5):
                star(d, CW / 2 + 52 + k * 27, my, 11, (255, 200, 40) if k < n_st else (120, 120, 140), outline=INKC)
            d.text((CW - 42, my), RARITY[n_st - 1], font=fit("Inter-Black.otf", RARITY[n_st - 1], 20, 112, 13),
                   fill=(255, 120, 140) if n_st >= 4 else (236, 234, 244), anchor="rm")
        # the art window: an inset bevel (gold / grey keyline, then ink)
        d.rectangle([ART_X - 6, ART_Y - 6, ART_X + aw + 5, ART_Y + ah + 5], fill=(170, 172, 180) if plain else (196, 156, 72))
        d.rectangle([ART_X - 3, ART_Y - 3, ART_X + aw + 2, ART_Y + ah + 2], fill=INKC)
        # name bar: B (or ? ? ?) until the reveal, then the name; "with A" on the right
        fy = ART_Y + ah + 10
        nb = fy + 46
        d.rounded_rectangle([28, fy, CW - 29, nb], 12, fill=(250, 246, 232), outline=INKC, width=3)
        ny = (fy + nb) / 2
        nm = ctx.C.labels[bn] if revealed else "? ? ?"
        wa = "WITH " + ctx.C.labels[an].upper()
        fw_ = fit("Inter-Bold.otf", wa, 19, 230, 12)
        d.text((CW - 44, ny), wa, font=fw_, fill=(140, 120, 80), anchor="rm")
        d.text((44, ny), nm, font=fit("Inter-ExtraBold.otf", nm, 36, CW - 120 - fw_.getlength(wa)), fill=INKC,
               anchor="lm")
        # fact box (the card's "move"): an energy dot, LINK, the fact (cut before B until the reveal), set symbol
        bt, bb = nb + 6, CH - R0 - 8
        by_ = (bt + bb) / 2
        d.rounded_rectangle([28, bt, CW - 29, bb], 10, fill=(236, 236, 240) if plain else (255, 250, 236),
                            outline=(150, 150, 160) if plain else (190, 150, 70), width=2)
        d.ellipse([40, by_ - 11, 62, by_ + 11], fill=(130, 130, 140) if plain else (226, 46, 58), outline=INKC, width=2)
        d.ellipse([45, by_ - 7, 52, by_], fill=(190, 190, 196) if plain else (255, 255, 255))
        word = "RECORD" if plain else "LINK"
        d.text((70, by_), word, font=font("Inter-Black.otf", 18), fill=INKC, anchor="lm")
        fact = ctx.fact_of(ctx.spec["links"][i], full=revealed)
        fx0 = 70 + font("Inter-Black.otf", 18).getlength(word) + 12
        fmax = CW - 29 - 56 - fx0
        ff = font("Inter-SemiBold.otf", 18)
        if ff.getlength(fact) > fmax:
            while fact and ff.getlength(fact + "…") > fmax:
                fact = fact[:-1]
            fact = fact.rstrip(" ,;:…") + "…"
        d.text((fx0, by_), fact, font=ff, fill=(70, 62, 56), anchor="lm")
        set_symbol(d, CW - 54, by_, plain)
        ring = Image.new("L", (CW, CH), 0)
        rd = ImageDraw.Draw(ring)
        rd.rounded_rectangle([0, 0, CW - 1, CH - 1], 30, fill=255)
        rd.rounded_rectangle([R0, R0, CW - 1 - R0, CH - 1 - R0], 18, fill=0)
        if not plain:                                   # gold filigree on the ring's inner corners + side studs
            gold, dark = (255, 214, 110), (110, 70, 10)
            em = emblem(False)                          # the medallion owns the top-left corner
            rd.ellipse([6, int(my - em.height / 2), 6 + em.width, int(my + em.height / 2)], fill=0)
            h_ = R0 // 2 + 3
            for (x, y, sx, sy) in ((CW - 1 - h_, h_, -1, 1), (h_, CH - 1 - h_, 1, -1), (CW - 1 - h_, CH - 1 - h_, -1, -1)):
                filigree(d, x, y, sx, sy, gold, dark)
                filigree(rd, x, y, sx, sy, 0, 0)        # keep the foil off the ornaments
            for (x, y) in ((CW / 2, R0 / 2), (CW / 2, CH - R0 / 2), (R0 / 2, CH / 2), (CW - R0 / 2, CH / 2)):
                d.polygon([(x, y - 8), (x + 8, y), (x, y + 8), (x - 8, y)], fill=gold, outline=dark)
                rd.polygon([(x, y - 8), (x + 8, y), (x, y + 8), (x - 8, y)], fill=0)
        st[key] = (im, np.asarray(ring, np.float32) / 255, plain)
        return st[key]

    def live(layers, tt, phase):
        """The card with a moving holo sheen and a glare sweep, both on the frame ring only; plain cards stay still."""
        im, ring, plain = layers
        if plain:
            return im
        strip, sa, sh = foil_strip()
        off = int((tt * 90 + phase * 300) % (CW * 2))
        a_holo = sa[:, off:off + CW] * ring * holo_pattern() * 190
        sweep = ((tt + phase) % 2.2) / 2.2
        so = int(CW * 1.5 - (sweep * (CW * 2.2) - CW * 0.6))
        shear = (np.arange(CH)[:, None] * 0.45).astype(np.int32)
        cols = np.clip(so + np.arange(CW)[None, :] + shear - CH // 4, 0, sh.shape[0] - 1)
        a_sh = sh[cols] * ring
        out = np.asarray(im, np.float32).copy()
        ah_ = (a_holo / 255)[..., None]
        out[..., :3] = out[..., :3] * (1 - ah_) + strip[:, off:off + CW] * ah_
        ash = (a_sh / 255)[..., None]
        out[..., :3] = out[..., :3] + (255 - out[..., :3]) * ash
        return Image.fromarray(out.astype(np.uint8))

    def finished(i):
        """Link i's card as it goes into the hand: revealed, with a still of the photo (both faces) as art."""
        key = ("fin", i)
        if key not in st:
            im = card(i, True)[0].copy()
            art = ctx.view(i, *ctx.both_cam(i, aw, ah, 0.75), aw, ah).convert("RGBA")
            if calm(i):
                art = art.convert("L").convert("RGBA")
            im.alpha_composite(art, (ART_X, ART_Y))
            st[key] = im
        return st[key]

    def mini(i):
        key = ("mini", i)
        if key not in st:
            f = finished(i)
            st[key] = f.resize((int(CW * MINI), int(CH * MINI)), Image.LANCZOS)
        return st[key]

    def hand_pose(k, n):
        """Card k of a fanned hand of n cards: (x, y, angle in degrees, PIL sense)."""
        spread = min(9.0, 36.0 / max(1, n - 1)) if n > 1 else 0.0
        ang = (k - (n - 1) / 2) * spread
        a = math.radians(ang)
        return ctx.CX + HAND_R * math.sin(a), PIVOT_Y - HAND_R * math.cos(a), -ang

    def background():
        if "bg" not in st:
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            r = np.sqrt(((xx - ctx.CX) / W) ** 2 + ((yy - 820) / H) ** 2)
            v = np.clip(1 - r * 1.4, 0, 1)[..., None]
            c = np.array([18, 16, 40]) * (1 - v) + np.array([58, 34, 96]) * v
            grid = (((xx + yy) % 90 < 3) | ((xx - yy) % 90 < 3))[..., None]
            c = np.where(grid, c * 1.25 + 6, c)
            # a felt play mat where the hand lies
            mat = np.clip((yy - 1330) / 60, 0, 1)[..., None]
            c = c * (1 - mat * 0.45) + np.array([20, 60, 52]) * mat * 0.45
            st["bg"] = Image.fromarray(np.dstack([np.clip(c, 0, 255).astype(np.uint8),
                                                  np.full((H, W), 255, np.uint8)]))
            s = 900
            g = np.mgrid[0:s, 0:s].astype(np.float32)
            rr = np.sqrt((g[0] - s / 2) ** 2 + (g[1] - s / 2) ** 2) / (s / 2)
            a = (np.clip(1 - rr, 0, 1) ** 2 * 70).astype(np.uint8)
            st["spot"] = Image.fromarray(np.dstack([np.full((s, s), 150, np.uint8), np.full((s, s), 110, np.uint8),
                                                    np.full((s, s), 255, np.uint8), a]))
            rnd = random.Random(3)
            st["sparks"] = [(rnd.uniform(0, W), rnd.uniform(280, 1900), rnd.uniform(0, 6.3), rnd.uniform(5, 13))
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
        ctx.paste(base, im, x, y, alpha=alpha, nearest=False)

    def shadow(base, x, y, w, h, ang=0.0, a=110):
        sh = Image.new("RGBA", (int(w) + 40, int(h) + 40), (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle([20, 20, int(w) + 19, int(h) + 19], 16, fill=(0, 0, 0, a))
        if abs(ang) > 0.05:
            sh = sh.rotate(ang, expand=True, resample=Image.BICUBIC)
        ctx.paste(base, sh, x + 12, y + 16, nearest=False)

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        sp, R = seg["speech"], seg.get("rv", ctx.REVEAL)
        rvt = R * sp + ctx.THROW
        revealed = local > rvt
        base = background().copy()
        sx_ = ctx.CX + 260 * math.sin(tt * 0.7)
        sy_ = 820 + 200 * math.cos(tt * 0.9)
        base.alpha_composite(st["spot"], (int(sx_ - 450), int(sy_ - 450)))
        d = ImageDraw.Draw(base, "RGBA")
        for (x, y, ph, r) in st["sparks"]:
            a = max(0.0, math.sin(tt * 2.2 + ph)) ** 3
            if a > 0.05:
                sparkle(d, x, y - (tt * 20 % 60), r * (0.6 + 0.4 * a), a * 0.8)

        # ---- the hand: finished cards 0..i-1, fanned at the bottom; card i-1 is still flying in during the swap
        tr = ctx.ease(max(0.0, min(1.0, local / SWAP))) if i > 0 else 1.0
        done = [j for j in range(i) if not ctx.C.fj["links"][j].get("lineage")]
        settled = done[:-1] if (done and done[-1] == i - 1 and tr < 1) else done
        n_now = len(done)
        for k, j in enumerate(settled):
            hx, hy, ha = hand_pose(k, n_now)
            if tr < 1 and len(settled) > 0:
                ox, oy, oa = hand_pose(k, max(1, len(settled)))
                hx, hy, ha = ox + (hx - ox) * tr, oy + (hy - oy) * tr, oa + (ha - oa) * tr
            bob = 5 * math.sin(tt * 2.2 + k)
            m = mini(j)
            shadow(base, hx, hy + bob, m.width, m.height, ha + 0.8 * math.sin(tt * 1.6 + k), 90)
            put(base, m, hx, hy + bob, 1.0, ha + 0.8 * math.sin(tt * 1.6 + k))

        # ---- the big card of this link deals in from the right (link 0 grows in place after the hook)
        if i == 0:
            k_in = 1.0
        else:
            k_in = ctx.ease(max(0.0, min(1.0, (local - 0.08) / (SWAP - 0.08))))
        cx0 = CX0 + (1 - k_in) * 980
        cy0 = CARD_TOP + 4 * math.sin(tt * 1.9)
        lay = card(i, revealed)

        def mount(b, X0, Y0, w, h):
            x, y = X0 - ART_X, Y0 - ART_Y
            shadow(b, x + CW / 2, y + CH / 2, CW, CH, 0, 120)
            b.alpha_composite(live(lay, tt, i * 0.37), (int(x), int(y)))

        c = ctx.draw_core(base, seg, tt, cx0 + ART_X, cy0 + ART_Y, SC, frame=mount)
        # B's name lands in the name bar: a small pop + a sparkle burst (not on a dictator's card)
        if revealed:
            rk = (local - rvt) / 0.6
            X0, Y0 = c["rect"][0] - ART_X, c["rect"][1] - ART_Y
            if 0 < rk < 1 and not calm(i):
                for j in range(14):
                    ang = j * 2 * math.pi / 14
                    rr = 120 + 300 * ctx.ease(rk)
                    sparkle(d, X0 + CW / 2 + rr * math.cos(ang), Y0 + TOPBAR / 2 + rr * 0.25 * math.sin(ang),
                            18 * (1 - rk) + 4, 1 - rk)

        # ---- the previous card shrinks and flies down into the hand (on top of everything but the chrome)
        if i > 0 and tr < 1 and done and done[-1] == i - 1:
            hx, hy, ha = hand_pose(n_now - 1, n_now)
            sx0, sy0 = ctx.CX, CARD_TOP + CH / 2
            x = sx0 + (hx - sx0) * tr - 180 * math.sin(math.pi * tr)
            y = sy0 + (hy - sy0) * tr
            sc = 1 + (MINI - 1) * tr
            put(base, finished(i - 1), x, y, sc, ha * tr - 8 * math.sin(math.pi * tr))

        X, Y, w_, h_ = c["rect"]                        # the credit sits inside the art's bottom edge (the bars are full)
        return ctx.core_finish(base, seg, tt, c, connected_y=int(Y + h_ - 110), credit_xy=(X + w_ - 12, Y + h_ - 18))

    return frame
