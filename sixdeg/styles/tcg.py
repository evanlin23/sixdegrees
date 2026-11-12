"""Trading cards: every link is dealt as one big collectible photo card whose art window IS the show's core
(draw_core: the camera moves, Buh jumps head to head with the red string, the name tags ride the photo). The name
bar reads "? ? ?" until the reveal, then B's name pops in. When the next link starts, the finished card shrinks and
flies down into a fanned hand of mini cards at the bottom of the screen, which keeps growing through the episode.
Foil sheen and a glare sweep run over the card frame only (never the art). A link with a dictator gets a plain grey
archive card: no foil, no glare, no stars, no rarity, and its mini card in the hand stays grey."""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAME = "tcg"

SC = 0.75                      # core window scale: art = 660 x 712
PAD = 24                       # card frame around the art (sides)
TOPBAR = 60                    # type line above the art
FOOT = 84                      # name bar under the art
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
        d.rounded_rectangle([12, 12, CW - 13, CH - 13], 20, fill=(226, 226, 230) if plain else (214, 220, 232))
        yr = next((ctx.year_of(j) for j in range(i, -1, -1) if ctx.year_of(j)), None)   # "that year"
        # type line on top (Buh may stand in front of it): card number, kind, year, stars + rarity
        my = (16 + TOPBAR - 8) / 2
        d.rounded_rectangle([20, 16, CW - 21, TOPBAR - 8], 12, fill=(70, 72, 84) if plain else (60, 52, 80))
        d.text((40, my), f"#{i + 1:02d}", font=font("Inter-Black.otf", 26),
               fill=(200, 200, 210) if plain else (255, 120, 140), anchor="lm")
        if plain:
            note = f"ARCHIVE PHOTO  ·  {yr}  ·  HISTORICAL RECORD" if yr else "ARCHIVE PHOTO  ·  HISTORICAL RECORD"
            d.text((CW / 2 + 30, my), note, font=font("Inter-Bold.otf", 22), fill=(236, 234, 244), anchor="mm")
        else:
            d.text((100, my), f"PHOTO CARD · {yr}" if yr else "PHOTO CARD", font=font("Inter-Bold.otf", 22),
                   fill=(236, 234, 244), anchor="lm")
            for k in range(5):
                star(d, CW / 2 + 40 + k * 32, my, 13, (255, 200, 40) if k < n_st else (120, 120, 140), outline=INKC)
            d.text((CW - 40, my), RARITY[n_st - 1], font=font("Inter-Black.otf", 22),
                   fill=(255, 120, 140) if n_st >= 4 else (236, 234, 244), anchor="rm")
        d.rectangle([ART_X - 4, ART_Y - 4, ART_X + aw + 3, ART_Y + ah + 3], fill=INKC)
        # name bar at the bottom: B (or ? ? ?) until the reveal, then the name; "with A" on the right
        fy = ART_Y + ah + 10
        d.rounded_rectangle([20, fy, CW - 21, CH - 16], 14, fill=(250, 246, 232), outline=INKC, width=3)
        ny = (fy + CH - 16) / 2
        nm = ctx.C.labels[bn] if revealed else "? ? ?"
        wa = "WITH " + ctx.C.labels[an].upper()
        fw_ = fit("Inter-Bold.otf", wa, 20, 230, 12)
        d.text((CW - 40, ny), wa, font=fw_, fill=(140, 120, 80), anchor="rm")
        d.text((40, ny), nm, font=fit("Inter-ExtraBold.otf", nm, 44, CW - 120 - fw_.getlength(wa)), fill=INKC,
               anchor="lm")
        ring = Image.new("L", (CW, CH), 0)
        rd = ImageDraw.Draw(ring)
        rd.rounded_rectangle([0, 0, CW - 1, CH - 1], 30, fill=255)
        rd.rounded_rectangle([12, 12, CW - 13, CH - 13], 20, fill=0)
        st[key] = (im, np.asarray(ring, np.float32) / 255, plain)
        return st[key]

    def live(layers, tt, phase):
        """The card with a moving holo sheen and a glare sweep, both on the frame ring only; plain cards stay still."""
        im, ring, plain = layers
        if plain:
            return im
        strip, sa, sh = foil_strip()
        off = int((tt * 90 + phase * 300) % (CW * 2))
        a_holo = sa[:, off:off + CW] * ring * 150
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

        return ctx.core_finish(base, seg, tt, c, connected_y=int(c["rect"][1] + c["rect"][3] - 110))

    return frame
