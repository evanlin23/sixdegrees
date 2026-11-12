"""Group chat: the chain plays out as messages on a phone. A neutral narrator account (Buh, "buh.connects") types
the link's line and posts the photo with the people tagged ("@A with @???" until the reveal, when the tag fills in
and the teaser line is edited to the full sentence); earlier links keep scrolling up. The people never speak."""
from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

NAME = "chat"

BG = (14, 19, 27)
DOODLE = (24, 32, 43)
BAR = (25, 32, 44)
BUBBLE = (35, 45, 60)
TEXT = (238, 241, 245)
MUTED = (140, 152, 168)
MENTION = (92, 170, 255)
TYPING = (98, 214, 140)
NAME_COLS = [(255, 138, 101), (129, 212, 250), (174, 213, 129), (255, 213, 79), (206, 147, 216), (128, 203, 196),
             (244, 143, 177), (255, 171, 64)]

TOP0, TOP1 = 230, 346          # chat title bar (under the channel header)
BOTTOM = 1196                  # newest message sits on this line; earlier ones scroll up
CAP_Y = 1262                   # captions, between the messages and the composer
COMP0, COMP1 = 1352, 1438      # message composer
AV = 72                        # avatar size
BX = 152                       # bubble left edge
MAXW = 700                     # bubble max width
PW, PH = 640, 470              # photo attachment
GAP = 20
SPACE = 150                    # room left under the last photo for CONNECTED!
BUH = "buh"                    # the narrator account
BUH_COL = (255, 214, 10)


def photo_params(ctx, i, w, h, fill=0.8, maxup=1.3):
    """(cx, cy, scale, ow, oh) for link photo i in a w x h window: both faces in, cover-cropped where the source
    allows, never upscaled beyond maxup (a small source gives ow, oh < w, h)."""
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

    st = {}                        # lazily built: wallpaper, bars, avatars, the message timeline

    def short(n):
        return "buh.connects" if n == BUH else ctx.C.labels[n]

    def col(n):
        if n == BUH:
            return BUH_COL
        return NAME_COLS[ctx.C.people.index(n) % len(NAME_COLS)] if n in ctx.C.people else MUTED

    def avatar(n, size=AV):
        key = ("av", n, size)
        if key not in st:
            im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            mask = Image.new("L", (size * 4, size * 4), 0)
            ImageDraw.Draw(mask).ellipse([0, 0, size * 4 - 1, size * 4 - 1], fill=255)
            mask = mask.resize((size, size), Image.LANCZOS)
            if n == BUH:
                face = Image.new("RGBA", (size, size), (52, 44, 92, 255))
                spr = ctx.pin_sprite("happy", 0, 0, 0, scale=6)
                sc = size * 1.25 / spr.height
                spr = spr.resize((max(1, int(spr.width * sc)), max(1, int(spr.height * sc))), Image.LANCZOS)
                face.alpha_composite(spr, (int(size / 2 - spr.width / 2), int(size * 0.12)))
            elif n is None:
                face = Image.new("RGBA", (size, size), (70, 80, 98, 255))
                ImageDraw.Draw(face).text((size / 2, size / 2 + 2), "?", font=font("Inter-Black.otf", int(size * 0.55)),
                                          fill=(220, 226, 236), anchor="mm")
            else:
                face = ctx.sharp[n].convert("RGBA").resize((size, size), Image.BICUBIC)
            im.paste(face, (0, 0), mask)
            st[key] = im
        return st[key]

    def wrap(text, f, width):
        lines, cur = [], ""
        for w in text.split():
            t = f"{cur} {w}".strip()
            if f.getlength(t) <= width or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def bubble(w, h, tail=True):
        im = Image.new("RGBA", (w + 14, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([14, 0, w + 13, h - 1], 30, fill=BUBBLE)
        if tail:   # little tail at the bottom-left, next to the avatar
            d.polygon([(0, h - 2), (40, h - 34), (44, h - 2)], fill=BUBBLE)
        return im

    def with_avatar(body, n):
        """Avatar at the bottom-left, the bubble to its right (x = BX - 14 for the tail)."""
        im = Image.new("RGBA", (BX - 14 + body.width, max(body.height, AV)), (0, 0, 0, 0))
        im.alpha_composite(body, (BX - 14 - ctx.SAFE[0], im.height - body.height))
        im.alpha_composite(avatar(n), (0, im.height - AV))
        return im

    def text_msg(n, text, stamp, mention=None, n_lines=0, width=0):
        fn, fb, fs = font("Inter-Bold.otf", 28), font("Inter-Bold.otf", 40), font("Inter-Bold.otf", 22)
        pre = f"@{mention} " if mention else ""
        lines = wrap(pre + text, fb, MAXW - 64)
        lh = 52
        w = max(fn.getlength(short(n)), max(fb.getlength(ln) for ln in lines) + 70, width - 52) + 52
        w = int(min(MAXW, w))
        h = 22 + 38 + lh * max(n_lines, len(lines)) + 16
        im = bubble(w, h)
        d = ImageDraw.Draw(im)
        d.text((40, 20), short(n), font=fn, fill=col(n))
        y = 64
        for k, ln in enumerate(lines):
            x = 40
            if k == 0 and pre and ln.startswith(pre.strip()):
                m = pre.strip()
                d.rounded_rectangle([x - 6, y - 2, x + fb.getlength(m) + 6, y + 48], 10, fill=(40, 70, 112))
                d.text((x, y), m, font=fb, fill=MENTION)
                x += fb.getlength(m + " ")
                ln = ln[len(m):].strip()
            d.text((x, y), ln, font=fb, fill=TEXT)
            y += lh
        d.text((w, h - 14), stamp, font=fs, fill=MUTED, anchor="rs")
        return with_avatar(im, n)

    def photo_msg(n, i, a, b, stamp):
        """The photo attachment, posted by the narrator with A and B (or "???") tagged; the slot for the (moving)
        photo is at PHOTO_OFF inside the returned image."""
        fn, fc, fs = font("Inter-Bold.otf", 28), font("Inter-Bold.otf", 32), font("Inter-Bold.otf", 22)
        w, h = PW + 32, 22 + 38 + PH + 70
        im = bubble(w, h)
        d = ImageDraw.Draw(im)
        d.text((40, 20), short(n), font=fn, fill=col(n))
        size = 32
        while size > 22 and fc.getlength(f"@{a} with @{b or '???'}") + 30 > w - 50:
            size -= 2
            fc = font("Inter-Bold.otf", size)
        x, y = 40, 64 + PH + 18 + (32 - size) // 2
        for part, chip, known in ((f"@{a}", True, True), (" with ", False, True), (f"@{b}" if b else "@???", True, b)):
            if chip:
                d.rounded_rectangle([x - 4, y - 4, x + fc.getlength(part) + 6, y + 38], 10, fill=(40, 70, 112))
            d.text((x, y), part, font=fc, fill=(MENTION if known else MUTED) if chip else TEXT)
            x += fc.getlength(part) + (8 if chip else 0)
        d.text((w - 4, 36), stamp, font=fs, fill=MUTED, anchor="rm")      # stamp up top, clear of the tags
        return with_avatar(im, n)

    PHOTO_OFF = (BX - ctx.SAFE[0] + 16, 64)
    pmask = Image.new("L", (PW * 2, PH * 2), 0)
    ImageDraw.Draw(pmask).rounded_rectangle([0, 0, PW * 2 - 1, PH * 2 - 1], 44, fill=255)
    pmask = pmask.resize((PW, PH), Image.LANCZOS)

    def photo(i, k):
        """The attachment with a slow push-in (k 0..1). Never upscaled past ~1.3x; a small source sits on a dark
        matte inside the slot instead of a blurred fill."""
        if ("ph", i) not in st:
            st[("ph", i)] = photo_params(ctx, i, PW, PH, maxup=1.15)
        cx, cy, s, ow, oh = st[("ph", i)]
        ih, iw = ctx.C.imgs[i].shape[:2]
        s *= 1.0 + 0.12 * k
        hw, hh = ow / 2 / s, oh / 2 / s
        cx = min(max(cx + 30 * math.sin(k * 2.2), hw), iw - hw)
        cy = min(max(cy - 12 * k, hh), ih - hh)
        ph = ctx.view(i, cx, cy, s, ow, oh)
        im = Image.new("RGBA", (PW, PH), (22, 28, 38, 255))
        im.alpha_composite(ph, ((PW - ow) // 2, (PH - oh) // 2))
        im.putalpha(pmask)
        return im

    def static_photo(i):
        if ("sph", i) not in st:
            st[("sph", i)] = photo(i, 1.0)
        return st[("sph", i)]

    def divider(text):
        f = font("Inter-Bold.otf", 26)
        w = int(f.getlength(text)) + 48
        im = Image.new("RGBA", (ctx.SAFE[2] - ctx.SAFE[0], 52), (0, 0, 0, 0))
        x0 = (im.width - w) // 2
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([x0, 4, x0 + w, 48], 22, fill=(30, 40, 54))
        d.text((im.width / 2, 27), text, font=f, fill=MUTED, anchor="mm")
        return im

    def heart(size, fill):
        im = Image.new("RGBA", (size * 4, size * 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        s = size * 4
        d.ellipse([0, 0, s * 0.56, s * 0.56], fill=fill)
        d.ellipse([s * 0.44, 0, s, s * 0.56], fill=fill)
        d.polygon([(s * 0.04, s * 0.38), (s * 0.96, s * 0.38), (s * 0.5, s * 0.95)], fill=fill)
        return im.resize((size, size), Image.LANCZOS)

    def reaction():
        if "react" not in st:
            im = Image.new("RGBA", (104, 54), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.rounded_rectangle([0, 0, 103, 53], 26, fill=(48, 60, 80), outline=BG, width=4)
            for ex in (24, 44):            # a neutral "eyes" reaction
                d.ellipse([ex, 12, ex + 18, 42], fill=(250, 250, 250))
                d.ellipse([ex + 6, 24, ex + 15, 37], fill=(20, 20, 26))
            d.text((70, 27), "1", font=font("Inter-Bold.otf", 28), fill=TEXT, anchor="mm")
            st["react"] = im
        return st["react"]

    def build():
        """Every message of the episode with its typing start and send time (absolute seconds)."""
        msgs = []
        for seg in ctx.segs:
            if seg.get("kind") != "link" or ctx.C.fj["links"][seg["i"]].get("lineage"):
                continue
            i, t0 = seg["i"], seg["t0"]
            sp, R = seg["speech"], seg.get("rv", ctx.REVEAL)
            rvt = R * sp + ctx.THROW
            u = min(1.0, rvt / 1.45)
            an, bn = ctx.link_names(i)
            stamp = f"{9 + i // 6}:{41 + i * 3:02d}"
            yr = ctx.year_of(i)
            if i > 0:
                msgs.append(dict(kind="div", ts=t0, ta=t0, im=divider(f"Link {i + 1}" + (f"  ·  {yr}" if yr else ""))))
            link = ctx.spec["links"][i]
            teaser, full = ctx.before_b(link["line"], link), link["line"]
            fb = font("Inter-Bold.otf", 40)
            nl = len(wrap(full, fb, MAXW - 64))
            post = text_msg(BUH, full, stamp + " · edited", n_lines=nl)
            msgs.append(dict(kind="text", who=BUH, ts=t0 + 0.04, ta=t0 + 0.38 * u, tpost=t0 + rvt,
                             im=text_msg(BUH, teaser, stamp, n_lines=nl, width=post.width - BX + 14), post=post))
            pre, post = photo_msg(BUH, i, short(an), None, stamp), photo_msg(BUH, i, short(an), short(bn), stamp)
            pta = t0 + 0.72 * u
            calm = ctx.sensitive(an) or ctx.sensitive(bn)        # no reactions on a link with a dictator
            msgs.append(dict(kind="photo", who=BUH, i=i, ts=t0 + 0.50 * u, ta=pta, im=pre, post=post,
                             tpost=t0 + rvt, seg=seg, tagged=(an, bn), react=None if calm else t0 + rvt + 0.65))
            if i == ctx.n_links - 1:                           # room under the last photo for CONNECTED!
                msgs.append(dict(kind="space", ts=t0 + rvt, ta=t0 + rvt, im=Image.new("RGBA", (8, SPACE)),
                                 tpost=t0 + rvt))
        return msgs

    TYP_H = 96

    def typing_im(n, t):
        im = Image.new("RGBA", (BX - 14 + 170, TYP_H), (0, 0, 0, 0))
        b = bubble(150, 84)
        d = ImageDraw.Draw(b)
        for k in range(3):
            ph = (t * 5.5 - k * 0.9)
            up = max(0.0, math.sin(ph)) * 12
            c = 150 + int(90 * max(0.0, math.sin(ph)))
            x = 54 + k * 34
            d.ellipse([x, 34 - up, x + 18, 52 - up], fill=(c, c + 4, c + 10))
        im.alpha_composite(b, (BX - 14 - ctx.SAFE[0], TYP_H - 84))
        im.alpha_composite(avatar(n), (0, TYP_H - AV))
        return im

    def wallpaper():
        if "wall" not in st:
            T = 240
            tile = Image.new("RGBA", (T, T), BG + (255,))
            d = ImageDraw.Draw(tile)
            d.ellipse([20, 24, 60, 64], outline=DOODLE, width=5)
            d.rounded_rectangle([140, 30, 196, 72], 10, outline=DOODLE, width=5)
            d.polygon([(150, 72), (160, 72), (148, 86)], fill=DOODLE)
            d.line([(30, 150), (60, 130), (90, 150), (120, 130)], fill=DOODLE, width=5)
            tile.alpha_composite(heart(40, DOODLE + (255,)), (160, 150))
            for (x, y) in ((100, 90), (30, 210), (210, 110)):
                d.ellipse([x, y, x + 12, y + 12], fill=DOODLE)
            d.line([(70, 190), (110, 230)], fill=DOODLE, width=5)
            d.line([(110, 190), (70, 230)], fill=DOODLE, width=5)
            big = Image.new("RGBA", (ctx.W + T, ctx.H + T))
            for x in range(0, big.width, T):
                for y in range(0, big.height, T):
                    big.paste(tile, (x, y))
            st["wall"], st["T"] = big, T
        return st["wall"], st["T"]

    def topbar(base, i, msgs, tt, typing_who):
        if "bar" not in st:
            bar = Image.new("RGBA", (ctx.W, TOP1 - TOP0), BAR + (255,))
            d = ImageDraw.Draw(bar)
            d.line([(90, 40), (70, 58), (90, 76)], fill=TEXT, width=6, joint="curve")   # back chevron
            for k in range(3):                                                         # menu dots
                d.ellipse([904, 30 + k * 20, 916, 42 + k * 20], fill=TEXT)
            d.line([(0, TOP1 - TOP0 - 2), (ctx.W, TOP1 - TOP0 - 2)], fill=(40, 50, 66), width=2)
            st["bar"] = bar
        base.alpha_composite(st["bar"], (0, TOP0))
        d = ImageDraw.Draw(base)
        # group avatar: the narrator plus the people tagged so far (B only once revealed), overlapping
        tagged = set()
        for m in msgs:
            if m["kind"] == "photo" and m["ta"] <= tt:
                tagged.add(m["tagged"][0])
                if tt >= m["tpost"]:
                    tagged.add(m["tagged"][1])
        n_tagged = len(tagged)
        seen = [BUH] + [n for n in ctx.C.people if n in tagged][-2:]
        x = 120
        for k, n in enumerate(seen):
            a = avatar(n, 64)
            ring = Image.new("RGBA", (72, 72), (0, 0, 0, 0))
            ImageDraw.Draw(ring).ellipse([0, 0, 71, 71], fill=BAR)
            base.alpha_composite(ring, (x + k * 38 - 4, TOP0 + 22))
            base.alpha_composite(a, (x + k * 38, TOP0 + 26))
        tx = x + max(1, len(seen)) * 38 + 50
        d.text((tx, TOP0 + 22), "Six Degrees Group", font=font("Inter-Bold.otf", 36), fill=TEXT)
        if typing_who is not False:
            who = short(typing_who)
            dots = "." * (1 + int(tt * 4) % 3)
            d.text((tx, TOP0 + 68), f"{who} is typing{dots}", font=font("Inter-Bold.otf", 26), fill=TYPING)
        else:
            d.text((tx, TOP0 + 68), f"{n_tagged} people tagged",
                   font=font("Inter-Bold.otf", 26), fill=MUTED)

    def composer(base, tt):
        if "comp" not in st:
            im = Image.new("RGBA", (ctx.W, ctx.H - COMP0 + 20), BAR + (255,))
            d = ImageDraw.Draw(im)
            d.line([(0, 0), (ctx.W, 0)], fill=(40, 50, 66), width=2)
            d.ellipse([66, 14, 126, 74], outline=MUTED, width=4)
            d.line([(96, 30), (96, 58)], fill=MUTED, width=4)
            d.line([(82, 44), (110, 44)], fill=MUTED, width=4)
            d.rounded_rectangle([144, 10, 820, 78], 34, fill=(36, 46, 62))
            d.text((180, 44), "Message", font=font("Inter-Bold.otf", 30), fill=(110, 122, 140), anchor="lm")
            st["comp"] = im
        base.alpha_composite(st["comp"], (0, COMP0))
        d = ImageDraw.Draw(base)
        p = 1 + 0.06 * math.sin(tt * 4)                 # send button breathes
        r = 34 * p
        d.ellipse([876 - r, COMP0 + 44 - r, 876 + r, COMP0 + 44 + r], fill=(46, 132, 255))
        d.polygon([(862, COMP0 + 28), (896, COMP0 + 44), (862, COMP0 + 60), (868, COMP0 + 44)], fill=(255, 255, 255))

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        if "msgs" not in st:
            st["msgs"] = build()
        msgs = st["msgs"]
        # slot heights: 0 -> typing bubble -> the message (eased); stacked bottom-up from BOTTOM
        items, total = [], 0.0
        typing_who = False
        for m in msgs:
            if tt < m["ts"]:
                break
            full = m["im"].height
            if m["kind"] == "div" or m["ta"] - m["ts"] < 0.05:
                h = full * ctx.ease((tt - m["ta"]) / 0.22)
                mode = "msg"
            elif tt < m["ta"]:
                h = TYP_H * ctx.ease((tt - m["ts"]) / 0.18)
                mode = "typing"
                typing_who = m["who"]
            else:
                h = TYP_H + (full - TYP_H) * ctx.ease((tt - m["ta"]) / 0.22)
                mode = "msg"
            items.append((m, h, mode))
            total += h + GAP
        wall, T = wallpaper()
        off = int(total * 0.6 + tt * 14) % T
        base = wall.crop((0, T - off, ctx.W, T - off + ctx.H))
        y = BOTTOM
        for m, h, mode in reversed(items):
            if y < TOP1 - 40:
                break
            if mode == "typing":
                im = typing_im(m["who"], tt)
                a = min(1.0, (tt - m["ts"]) / 0.12)
                ctx.paste(base, im, ctx.SAFE[0] + im.width / 2, y - h / 2, alpha=a, nearest=False)
            else:
                im = m["post"] if m.get("post") is not None and tt >= m["tpost"] else m["im"]
                k = min(1.0, (tt - m["ta"]) / 0.22)
                sc = 0.86 + 0.14 * ctx.ease_out_back(k)
                if m["kind"] == "photo":
                    im = im.copy()
                    sg = m["seg"]
                    if sg is seg or (sg["i"] == i):
                        pk = min(1.0, max(0.0, (tt - sg["t0"]) / max(1.0, sg["t1"] - sg["t0"])))
                        im.alpha_composite(photo(m["i"], pk), PHOTO_OFF)
                    else:
                        im.alpha_composite(static_photo(m["i"]), PHOTO_OFF)
                if m["kind"] == "space":
                    pass
                elif m["kind"] == "div":
                    x0 = ctx.SAFE[0]
                    ctx.paste(base, im, x0 + im.width / 2, y - im.height / 2, alpha=k)
                else:
                    if sc < 0.999:
                        im = im.resize((int(im.width * sc), int(im.height * sc)), Image.BILINEAR)
                    base.alpha_composite(im, (ctx.SAFE[0], int(y - im.height)))
                    if m.get("react") and tt >= m["react"]:
                        rk = ctx.ease_out_back((tt - m["react"]) / 0.25)
                        ctx.paste(base, reaction(), ctx.SAFE[0] + im.width - 30, y - im.height + 2,
                                  scale=0.3 + 0.7 * rk, nearest=False)      # tapback on the bubble's top corner
            y -= h + GAP
        topbar(base, i, msgs, tt, typing_who)
        composer(base, tt)
        # Buh, the narrator, peeks up from under the composer and reacts to each reveal
        rv = tt - seg["t0"] - seg.get("rv", ctx.REVEAL) * seg["speech"] - ctx.THROW
        expr = "happy" if rv < 0 else "shock" if rv < 0.6 else "smug"
        hop = 30 * math.sin(math.pi * min(1.0, rv / 0.35)) if 0 < rv < 0.35 else 0.0
        ctx.paste_pin(base, ctx.pin_sprite(expr, 5 * math.sin(tt * 2.5), 0, 1, scale=7), 250,
                      1760 - hop + 8 * math.sin(tt * 3))
        ctx.header(base, i)
        ctx.connected(base, seg, tt, y=BOTTOM - SPACE / 2 + 6)
        seg["cap"].draw(base, tt, y=CAP_Y)
        return base

    return frame
