"""Swipe-to-match: a dating-app parody. A's profile card is up, B's card slides in from the side, B gets a LIKE stamp,
then the photo of the two of them slams in with a big IT'S A MATCH! and a shower of hearts."""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAME = "swipe"

TOP = (255, 88, 128)           # background gradient, top -> bottom
BOT = (255, 150, 84)
CW, CH = 560, 760              # profile card
CARD_Y = 742                   # card centre
BTN_Y = 1228                   # action buttons
CAP_Y = 1352                   # captions
PINK = (255, 64, 120)
GREEN = (46, 204, 113)


def make(ctx):
    F = {}

    def font(name, size):
        if (name, size) not in F:
            F[(name, size)] = ImageFont.truetype(str(ctx.FONTS / name), size)
        return F[(name, size)]

    st = {}

    def heart(size, fill, outline=None):
        key = ("h", size, fill, outline)
        if key not in st:
            s = size * 4
            im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)

            def shape(pad, col):
                a, b = pad, s - pad
                w = b - a
                d.ellipse([a, a, a + w * 0.56, a + w * 0.56], fill=col)
                d.ellipse([a + w * 0.44, a, b, a + w * 0.56], fill=col)
                d.polygon([(a + w * 0.03, a + w * 0.36), (b - w * 0.03, a + w * 0.36), (a + w * 0.5, b)], fill=col)
            if outline:
                shape(0, outline)
                shape(s * 0.09, fill)
            else:
                shape(0, fill)
            st[key] = im.resize((size, size), Image.LANCZOS)
        return st[key]

    def background():
        if "bg" not in st:
            yy = np.linspace(0, 1, ctx.H)[:, None, None]
            g = (np.array(TOP)[None, None] * (1 - yy) + np.array(BOT)[None, None] * yy).astype(np.uint8)
            g = np.repeat(g, ctx.W, axis=1)
            bg = Image.fromarray(np.dstack([g, np.full((ctx.H, ctx.W), 255, np.uint8)]))
            # faint hearts on a tile that drifts upward
            T = 320
            tile = Image.new("RGBA", (T, T), (0, 0, 0, 0))
            rnd = random.Random(7)
            for _ in range(5):
                sz = rnd.choice([40, 56, 72])
                h = heart(sz, (255, 255, 255, 46)).rotate(rnd.uniform(-25, 25), expand=True, resample=Image.BICUBIC)
                tile.alpha_composite(h, (rnd.randrange(0, T - h.width), rnd.randrange(0, T - h.height)))
            wide = Image.new("RGBA", (ctx.W + T, ctx.H), (0, 0, 0, 255))
            wide.paste(bg, (0, 0))
            wide.paste(bg.crop((0, 0, T, ctx.H)), (ctx.W, 0))
            for x in range(0, wide.width, T):
                for y in range(0, ctx.H, T):
                    wide.alpha_composite(tile, (x, y))
            st["bg"], st["T"] = wide, T
            # app bar (generic parody wordmark) and the action buttons
            bar = Image.new("RGBA", (ctx.W, 96), (0, 0, 0, 0))
            d = ImageDraw.Draw(bar)
            f = font("Inter-Black.otf", 50)
            wm = "buhmatch"
            tw = f.getlength(wm)
            x0 = ctx.CX - (tw + 60) / 2
            bar.alpha_composite(heart(48, (255, 255, 255)), (int(x0), 26))
            d.text((x0 + 60, 50), wm, font=f, fill=(255, 255, 255), anchor="lm")
            d.ellipse([ctx.SAFE[0] + 10, 24, ctx.SAFE[0] + 62, 76], outline=(255, 255, 255), width=6)     # profile
            d.ellipse([ctx.SAFE[0] + 26, 34, ctx.SAFE[0] + 46, 54], fill=(255, 255, 255))
            d.rounded_rectangle([ctx.SAFE[2] - 70, 28, ctx.SAFE[2] - 14, 70], 14, outline=(255, 255, 255), width=6)
            d.polygon([(ctx.SAFE[2] - 56, 68), (ctx.SAFE[2] - 40, 68), (ctx.SAFE[2] - 60, 84)], fill=(255, 255, 255))
            st["bar"] = bar
        return st["bg"], st["T"]

    def button(kind, r):
        key = ("btn", kind, r)
        if key not in st:
            s = r * 2 + 16
            im = Image.new("RGBA", (s * 4, s * 4), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            c = s * 2
            d.ellipse([c - r * 4 - 8, c - r * 4 + 16, c + r * 4 - 8, c + r * 4 + 16], fill=(120, 30, 50, 90))
            d.ellipse([c - r * 4, c - r * 4, c + r * 4, c + r * 4], fill=(255, 255, 255))
            k = r * 4
            if kind == "nope":
                w = int(k * 0.16)
                d.line([(c - k * 0.38, c - k * 0.38), (c + k * 0.38, c + k * 0.38)], fill=(255, 72, 96), width=w)
                d.line([(c - k * 0.38, c + k * 0.38), (c + k * 0.38, c - k * 0.38)], fill=(255, 72, 96), width=w)
            elif kind == "star":
                pts = [(c + k * (0.5 if j % 2 == 0 else 0.22) * math.sin(j * math.pi / 5),
                        c - k * (0.5 if j % 2 == 0 else 0.22) * math.cos(j * math.pi / 5)) for j in range(10)]
                d.polygon(pts, fill=(52, 152, 255))
            elif kind == "back":
                d.arc([c - k * 0.42, c - k * 0.42, c + k * 0.42, c + k * 0.42], 200, 520, fill=(255, 190, 40),
                      width=int(k * 0.13))
                d.polygon([(c - k * 0.58, c - k * 0.1), (c - k * 0.2, c - k * 0.1), (c - k * 0.4, c + k * 0.22)],
                          fill=(255, 190, 40))
            im = im.resize((s, s), Image.LANCZOS)
            if kind == "like":
                h = heart(int(r * 1.1), GREEN)
                im.alpha_composite(h, (s // 2 - h.width // 2, s // 2 - h.height // 2 + 3))
            st[key] = im
        return st[key]

    def round_mask(w, h, r):
        key = ("m", w, h, r)
        if key not in st:
            m = Image.new("L", (w * 2, h * 2), 0)
            ImageDraw.Draw(m).rounded_rectangle([0, 0, w * 2 - 1, h * 2 - 1], r * 2, fill=255)
            st[key] = m.resize((w, h), Image.LANCZOS)
        return st[key]

    def card(i, role, named):
        """A profile card: the person's face from photo i, name and year over a dark gradient."""
        key = ("card", i, role, named)
        if key not in st:
            nm = ctx.link_names(i)[0 if role == "a" else 1]
            ph = ctx.view(i, *ctx.face_cam(ctx.fbox(i, role), CH, 0.36), CW, CH)
            ph = ph.copy()
            sh = np.zeros((CH, CW), np.float32)
            ramp = np.clip((np.arange(CH) - CH * 0.55) / (CH * 0.45), 0, 1) ** 1.3
            sh[:] = ramp[:, None] * 220
            ph.alpha_composite(Image.fromarray(np.dstack([np.zeros((CH, CW, 3), np.uint8), sh.astype(np.uint8)])))
            d = ImageDraw.Draw(ph)
            label = ctx.C.labels[nm] if named else "???"
            size = 60
            while font("Inter-Black.otf", size).getlength(label) > CW - 60 and size > 30:
                size -= 2
            d.text((32, CH - 132), label, font=font("Inter-Black.otf", size), fill=(255, 255, 255), anchor="ls")
            yr = ctx.year_of(i)
            sub = (f"Spotted in {yr}" if yr else "Nearby") + ("  ·  1 mutual photo" if role == "b" else "")
            d.text((34, CH - 76), sub, font=font("Inter-Bold.otf", 30), fill=(235, 235, 240), anchor="ls")
            if named and 32 + font("Inter-Black.otf", size).getlength(label) + 60 < CW:   # verified badge
                bx = 32 + font("Inter-Black.otf", size).getlength(label) + 34
                d.ellipse([bx - 20, CH - 132 - size * 0.62 - 20, bx + 20, CH - 132 - size * 0.62 + 20], fill=(52, 152, 255))
                d.line([(bx - 10, CH - 132 - size * 0.62), (bx - 2, CH - 132 - size * 0.62 + 8),
                        (bx + 11, CH - 132 - size * 0.62 - 8)], fill=(255, 255, 255), width=6)
            for k in range(4):   # photo-count pips at the top
                x0 = 24 + k * (CW - 48) / 4
                d.rounded_rectangle([x0 + 4, 16, x0 + (CW - 48) / 4 - 4, 24], 4,
                                    fill=(255, 255, 255, 240 if k == 0 else 110))
            ph.putalpha(round_mask(CW, CH, 40))
            out = Image.new("RGBA", (CW + 40, CH + 50), (0, 0, 0, 0))
            shadow = Image.new("RGBA", (CW, CH), (90, 20, 40, 110))
            shadow.putalpha(round_mask(CW, CH, 40).point(lambda v: v * 110 // 255))
            out.alpha_composite(shadow, (24, 34))
            out.alpha_composite(ph, (12, 12))
            st[key] = out
        return st[key]

    def stamp(text, colr):
        key = ("stamp", text)
        if key not in st:
            f = font("Inter-Black.otf", 92)
            w = int(f.getlength(text)) + 60
            im = Image.new("RGBA", (w, 140), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.rounded_rectangle([6, 6, w - 6, 134], 18, outline=colr, width=12)
            d.text((w / 2, 72), text, font=f, fill=colr, anchor="mm")
            st[key] = im.rotate(-18, expand=True, resample=Image.BICUBIC)
        return st[key]

    PW, PH = 780, 560

    def match_photo(i, k):
        if ("cam", i) not in st:
            st[("cam", i)] = ctx.both_cam(i, PW, PH, 0.8)
        cx, cy, s = st[("cam", i)]
        ph = ctx.view(i, cx + 20 * math.sin(k * 1.7), cy, s * (1.0 + 0.08 * k), PW, PH)
        out = Image.new("RGBA", (PW + 36, PH + 36), (255, 255, 255, 0))
        fr = Image.new("RGBA", (PW + 36, PH + 36), (255, 255, 255, 255))
        fr.putalpha(round_mask(PW + 36, PH + 36, 30))
        out.alpha_composite(fr)
        ph.putalpha(round_mask(PW, PH, 18))
        out.alpha_composite(ph, (18, 18))
        return out

    def match_text():
        if "mt" not in st:
            st["mt"] = ctx.pixel_text("IT'S A MATCH!", 62, fill=(255, 236, 120))
        return st["mt"]

    def avatar(n, size=118):
        key = ("av", n, size)
        if key not in st:
            m = Image.new("L", (size * 4, size * 4), 0)
            ImageDraw.Draw(m).ellipse([0, 0, size * 4 - 1, size * 4 - 1], fill=255)
            m = m.resize((size, size), Image.LANCZOS)
            im = Image.new("RGBA", (size + 12, size + 12), (0, 0, 0, 0))
            ImageDraw.Draw(im).ellipse([0, 0, size + 11, size + 11], fill=(255, 255, 255))
            im.paste(ctx.sharp[n].convert("RGBA").resize((size, size), Image.BICUBIC), (6, 6), m)
            st[key] = im
        return st[key]

    def hearts(base, i, since, alpha):
        rnd = random.Random(100 + i)
        cols = [(255, 64, 120), (255, 120, 160), (255, 214, 10), (255, 255, 255), (255, 90, 90)]
        for k in range(26):
            x0 = rnd.uniform(ctx.SAFE[0], ctx.SAFE[2])
            spd = rnd.uniform(380, 720)
            delay = rnd.uniform(0, 0.6)
            sz = rnd.choice([34, 46, 60, 76])
            col = rnd.choice(cols)
            ph = rnd.uniform(0, 6.3)
            t = since - delay
            if t < 0:
                continue
            y = 1440 - t * spd
            if y < 300:
                y = 1440 - ((t * spd) % 1140)
            x = x0 + 40 * math.sin(t * 3 + ph)
            ctx.paste(base, heart(sz, col), x, y, alpha=alpha, nearest=False)

    def ptag(text, colr, size):
        key = ("tag", text, colr, size)
        if key not in st:
            st[key] = ctx.pixel_tag(text, colr, size)
        return st[key]

    def rot(im, ang):
        return im.rotate(ang, expand=True, resample=Image.BICUBIC) if abs(ang) > 0.05 else im

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        sp, R = seg["speech"], seg.get("rv", ctx.REVEAL)
        rvt = R * sp + ctx.THROW
        last = i == ctx.n_links - 1
        u = min(1.0, rvt / 1.45)
        mt = rvt + (0.1 if last else 0.32)           # match screen slams in
        an, bn = ctx.link_names(i)
        bg, T = background()
        off = int(tt * 60) % T
        base = bg.crop((off, 0, off + ctx.W, ctx.H))
        base.alpha_composite(st["bar"], (0, 238))
        # ---- the two cards
        ea = ctx.ease_out_back(local / 0.42)
        tb0 = 0.38 * u
        b = ctx.ease((local - tb0) / (0.5 * u))
        bk = ctx.ease_out_back(min(1.0, max(0.0, (local - tb0) / (0.5 * u))))
        fly = ctx.ease((local - rvt - 0.12) / 0.25) if local > rvt + 0.12 else 0.0      # both cards swipe together
        wob = math.sin(tt * 3.1)
        sa = 0.9 - 0.22 * b
        ax = ctx.CX + (282 - ctx.CX) * b + 160 * fly
        ay = CARD_Y + (1 - ea) * 800 + 8 * math.sin(tt * 2.3) - 120 * fly
        aang = -7 * b + 3.0 * wob
        bx = ctx.W + 380 + (668 - ctx.W - 380) * bk - 170 * fly
        by = CARD_Y + 10 * math.sin(tt * 2.3 + 1.4) - 120 * fly
        bang = 7 * b + 16 * (1 - b) - 3.0 * wob
        calpha = 1 - fly
        if calpha > 0.01:
            ca = card(i, "a", True)
            ca = ca.resize((int(ca.width * sa), int(ca.height * sa)), Image.BILINEAR)
            ctx.paste(base, rot(ca, aang), ax, ay, alpha=calpha)
            if local > tb0:
                named = local >= rvt
                cb = card(i, "b", named)
                cb = cb.resize((int(cb.width * 0.68), int(cb.height * 0.68)), Image.BILINEAR)
                cb = rot(cb, bang)
                if named:
                    sk = ctx.ease_out_back((local - rvt) / 0.18)
                    s_im = stamp("LIKE", GREEN)
                    s_im = s_im.resize((int(s_im.width * 0.6), int(s_im.height * 0.6)), Image.BILINEAR)
                    cb = cb.copy()
                    ctx.paste(cb, s_im, cb.width * 0.36, cb.height * 0.2, scale=2.2 - 1.2 * sk, alpha=min(1.0, sk * 2),
                              nearest=False)
                ctx.paste(base, cb, bx, by, alpha=calpha)
        # ---- action buttons (hidden under the match screen)
        if local < mt + 0.3:
            press = ctx.ease((local - rvt + 0.15) / 0.12) * (1 - ctx.ease((local - rvt) / 0.2))
            pulse = 1 + 0.05 * math.sin(tt * 5)
            btns = (("back", ctx.SAFE[0] + 70, 44), ("nope", ctx.CX - 120, 64), ("like", ctx.CX + 120, 64),
                    ("star", ctx.SAFE[2] - 70, 44))
            for kind, x, r in btns:
                sc = pulse if kind == "like" else 1.0
                if kind == "like":
                    sc *= 1 - 0.2 * press
                yb = BTN_Y + 6 * math.sin(tt * 4 + x * 0.01)
                ctx.paste(base, button(kind, r), x, yb, scale=sc, nearest=False)
        # ---- the match screen
        if local >= mt:
            m = local - mt
            lvl = int(20 * ctx.ease(m / 0.2))
            if ("dim", lvl) not in st:
                st[("dim", lvl)] = Image.new("RGBA", (ctx.W, ctx.H), (40, 6, 24, 10 * lvl))
            base.alpha_composite(st[("dim", lvl)])
            hearts(base, i, m, min(1.0, m / 0.2))
            k = min(1.0, max(0.0, local / max(1.0, seg["t1"] - seg["t0"])))
            pk = ctx.ease_out_back(m / 0.28)
            dx, dy = ctx.shake(m - 0.22, 22, 0.35)
            ph = match_photo(i, k)
            ph = rot(ph, 2.4 * math.sin(tt * 2.6))
            ctx.paste(base, ph, ctx.CX + dx, 800 + dy, scale=1.9 - 0.9 * pk, alpha=min(1.0, m / 0.1), nearest=False)
            tk = ctx.ease_out_back((m - 0.08) / 0.25)
            if m > 0.08:
                mtxt = rot(match_text(), 3.5 * math.sin(tt * 3.4))
                ctx.paste(base, mtxt, ctx.CX + dx, 420 + 10 * math.sin(tt * 4.2), scale=1.7 - 0.7 * tk, nearest=False)
            # the two of them, as round avatars with name tags
            ak = ctx.ease_out_back((m - 0.18) / 0.25)
            if m > 0.18:
                for nm, x, colr, y in ((an, ctx.SAFE[0] + 74, ctx.CYAN, 1158), (bn, ctx.SAFE[2] - 74, ctx.YELLOW, 1250)):
                    ctx.paste(base, avatar(nm, 96), x, y + 4 * math.sin(tt * 5 + x), scale=ak, nearest=False)
                    tsz = 26
                    tag = ptag(ctx.C.labels[nm].upper(), colr, tsz)
                    while tag.width > 560 and tsz > 14:
                        tsz -= 2
                        tag = ptag(ctx.C.labels[nm].upper(), colr, tsz)
                    tx = x + (78 + tag.width / 2 if x < ctx.CX else -78 - tag.width / 2)
                    ctx.paste(base, tag, tx, y, scale=ak)
                for hx, hy in ((ctx.SAFE[2] - 74, 1158), (ctx.SAFE[0] + 74, 1250)):
                    ctx.paste(base, heart(60, PINK, (255, 255, 255)), hx, hy + 6 * math.sin(tt * 7),
                              scale=ak * (1 + 0.14 * abs(math.sin(tt * 7))), nearest=False)
        else:
            # names while the cards are up: A always, B after the reveal
            ta = ptag(ctx.C.labels[an].upper(), ctx.CYAN, 24)
            ctx.paste(base, ta, max(ctx.SAFE[0] + ta.width / 2, ax), ay + CH * sa / 2 + 34, alpha=calpha)
            if local >= rvt:
                tb = ptag(ctx.C.labels[bn].upper(), ctx.YELLOW, 24)
                ctx.paste(base, tb, min(ctx.SAFE[2] - tb.width / 2, bx), by + CH * 0.68 / 2 + 96, alpha=calpha,
                          scale=ctx.ease_out_back((local - rvt) / 0.18))
        ctx.header(base, i)
        ctx.connected(base, seg, tt)
        seg["cap"].draw(base, tt, y=CAP_Y)
        return base

    return frame
