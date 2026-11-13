"""Newspaper: each link is a front page and the show's core photo window (draw_core: camera, Buh, string, name
tags) is its front-page photo, with the masthead, dateline and headline above it, narrow news columns beside it and
the picture caption + columns below. The column text is lorem-like nonsense with hidden Buh messages running
through it (SUBSCRIBE TO BUH CONNECTS, THE PIN KNOWS ...) for viewers who pause. The headline reads "... ???" until
the reveal (B is never named early), then the page is reprinted as an EXTRA with the full headline. Each new page
slides in from a different side while the previous paper drops away underneath.
A sensitive link gets a sober "ARCHIVE / FROM THE PICTURE DESK" sidebar.
It reads as real paper (owner, 2026-11-13): off-white newsprint with fibres, mottling and grain, ink that sits IN the
paper (multiplied, slightly uneven), a fold crease under the photo and soft darkened edges. The whole page drifts,
tilts and sways gently as if held over a desk, with the core window riding along (the page and the core are drawn on
one layer that is rotated together), and the wooden desk behind it slides slowly under a moving lamp light."""
from __future__ import annotations

import math
import random
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAME = "newspaper"
PAPERS = ["THE DAILY BUH", "THE BUH HERALD", "BUH EVENING POST", "THE MORNING BUH", "THE BUH CHRONICLE",
          "THE BUH TRIBUNE", "BUH WEEKLY NEWS", "THE BUH GAZETTE"]
SIDEBARS = [("BUH SAYS", "IT'S ALL CONNECTED"), ("WEATHER", "CLOUDY, CHANCE OF LINKS"), ("SPORTS", "PIN WINS AGAIN"),
            ("CLASSIFIEDS", "LOST: ONE DEGREE"), ("OPINION", "SIX IS PLENTY"), ("PUZZLES", "CONNECT THE DOTS")]
SYL = ["lo", "rem", "ip", "sum", "do", "lor", "sit", "a", "met", "con", "sec", "te", "tur", "ad", "pi", "sci",
       "el", "it", "sed", "di", "am", "non", "u", "mi", "ra", "ve", "na", "pul", "vi", "nar", "quis", "ta", "bu"]
SECRETS = [      # the column text is nonsense with a hidden message for the eagle-eyed (friendly, never hateful)
    "SUBSCRIBE TO BUH CONNECTS", "COMMENT WHO {FROM} SHOULD CONNECT TO NEXT", "BUH WAS HERE", "THE PIN KNOWS",
    "IF YOU CAN READ THIS YOU ARE TOO CLOSE", "HI MOM", "PAUSE MORE OFTEN", "SIX DEGREES OR LESS",
    "READ THE DESCRIPTION", "BUH SEES ALL", "TELL A FRIEND ABOUT BUH", "THE RED STRING NEVER LIES",
]

SC = 0.75                       # core window scale: 660 x 712
PX0, PX1 = 60, 930              # page left / right on screen
PY0 = 402                       # page top (just under the chain bar)
WIN_Y = 572                     # front-page photo top (screen) -> bottom 1284, just above the captions
SPIN = 0.36                     # slide-in time, s
REPRINT = 0.3                   # crossfade to the EXTRA edition at the reveal, s
INKC = (24, 20, 16)
GREY = (104, 96, 86)


def make(ctx):
    W, H = ctx.W, ctx.H
    PW, PH = PX1 - PX0, H - PY0 + 40
    fw, fh = int(880 * SC), int(950 * SC)
    WX = int(ctx.CX - fw / 2 - PX0)                         # window x inside the page
    WY = WIN_Y - PY0                                         # window y inside the page
    st = {}

    def font(name, size):
        key = ("f", name, size)
        if key not in st:
            st[key] = ImageFont.truetype(str(ctx.FONTS / f"Inter-{name}.otf"), size)
        return st[key]

    def year(i):
        for j in range(i, -1, -1):
            y = ctx.year_of(j)
            if y:
                return y
        return None

    def masked(txt, link):
        """B's name replaced by "???" (falls back to cutting the text before it)."""
        a, b = link["a"], link["b"]
        a_words = set(re.findall(r"[A-Z][\w'-]{2,}", f"{a} {ctx.C.labels.get(a, '')}"))
        names = [w for w in set(re.findall(r"[A-Z][\w'.-]{1,}", f"{b} {ctx.C.labels.get(b, '')}")) if w not in a_words]
        if names:
            alt = "|".join(map(re.escape, sorted(names, key=len, reverse=True)))
            out = re.sub(rf"\b(?:{alt})(?:\s+(?:{alt}))*(?![\w])", "???", txt)
            if ctx.before_b(out, link) == out:
                return out
        cut = ctx.before_b(txt, link)
        return cut if cut == txt else cut.rstrip("…").rstrip(" ,;:") + " ???"

    def headline_text(i, revealed):
        link = ctx.spec["links"][i]
        txt = link.get("fact") or ""
        if not txt:
            txt = re.sub(r"(?i)^(?:And |Then |So )?(?:(?:In|By|Around|On|Back in)\s+)?\d{4}s?,\s*", "",
                         link["line"].strip())
            txt = re.sub(r"(?i)^(?:And |Then |So )?that year,\s*", "", txt)
        txt = re.sub(r"^\d{4}s?:\s*", "", txt).rstrip(".")
        if not revealed:
            txt = masked(txt, link)
        return txt.upper()

    def wrap(text, f, width):
        lines, cur = [], ""
        for w in text.split():
            t = f"{cur} {w}".strip()
            if f.getlength(t) > width and cur:
                lines.append(cur)
                cur = w
            else:
                cur = t
        return lines + ([cur] if cur else [])

    def fit_headline(text, width, max_h):
        for size in range(56, 21, -2):
            f = font("Black", size)
            lines = wrap(text, f, width)
            if len(lines) * size * 1.04 <= max_h and all(f.getlength(l) <= width for l in lines):
                return f, lines
        f = font("Black", 22)
        lines = wrap(text, f, width)[:3]
        return f, lines

    def nonsense(r, n):
        out = []
        for k in range(n):
            w = "".join(r.choice(SYL) for _ in range(r.choice((1, 2, 2, 3))))
            out.append(w.capitalize() if k == 0 or out[-1].endswith(".") else w)
            if r.random() < 0.12:
                out[-1] += "."
            elif r.random() < 0.08:
                out[-1] += ","
        return out

    def secrets(seed):
        frm = ctx.C.labels.get(ctx.spec.get("from", ""), ctx.spec.get("from", "")).upper() or "BUH"
        order = random.Random(seed // 31).sample(SECRETS, len(SECRETS))       # one order per page,
        k = (seed % 31) * 5 % len(SECRETS)                                      # each column starts elsewhere
        return [m.replace("{FROM}", frm) for m in order[k:] + order[:k]]

    def column(d, x, y0, y1, width, seed, size=13):
        """Lorem-like nonsense in grey, every few lines broken by one line of a hidden message (slightly darker)."""
        r = random.Random(seed)
        f = font("Medium", size)
        msgs = secrets(seed)
        y, k, gap = y0, 0, r.randint(2, 4)
        lh = int(size * 1.35)
        while y + size <= y1:
            if gap <= 0 and msgs:
                lines = wrap(msgs.pop(0), f, width)
                if y + lh * (len(lines) - 1) + size <= y1:          # a message is printed whole or not at all
                    for ln in lines:
                        d.text((x, y), ln, font=f, fill=(66, 58, 50))
                        y += lh
                    gap = r.randint(5, 9)
                    continue
            words, line = nonsense(r, 12), ""
            for w in words:
                if f.getlength(f"{line} {w}".strip()) > width:
                    break
                line = f"{line} {w}".strip()
            d.text((x, y), line, font=f, fill=GREY)
            y += lh
            gap -= 1

    FOLD = WY + fh + 92                                      # the horizontal fold, between the photo and its caption

    def newsprint():
        """The paper itself (float RGB, PH x PW, built once): off-white newsprint with fibres, cloudy mottling, grain,
        aged edges and a fold crease; plus the ink-unevenness map (0.74 .. 1)."""
        if "paper" in st:
            return st["paper"], st["inkvar"]
        r = np.random.default_rng(11)
        yy, xx = np.mgrid[0:PH, 0:PW].astype(np.float32)
        mottle = cv2.GaussianBlur(r.normal(0, 1, (PH, PW)).astype(np.float32), (0, 0), 28)
        mottle /= max(1e-6, float(np.abs(mottle).max()))
        fib = Image.new("L", (PW, PH), 0)
        fd = ImageDraw.Draw(fib)
        for _ in range(PW * PH // 260):                      # short wiggly fibres
            x, y = r.uniform(0, PW), r.uniform(0, PH)
            a, ln = r.uniform(0, math.pi), r.uniform(4, 16)
            bend = r.uniform(-0.5, 0.5)
            pts = [(x + ln * q / 4 * math.cos(a + bend * q / 4), y + ln * q / 4 * math.sin(a + bend * q / 4))
                   for q in range(5)]
            fd.line(pts, fill=int(r.choice((60, 90, 140, 200))), width=1)
        fib = cv2.GaussianBlur(np.asarray(fib, np.float32) / 255, (0, 0), 0.7)
        sign = np.where(cv2.GaussianBlur(r.normal(0, 1, (PH, PW)).astype(np.float32), (0, 0), 3) > 0.15, -1.0, 0.6)
        grain = r.normal(0, 3.2, (PH, PW)).astype(np.float32)
        edge = np.minimum(np.minimum(xx, PW - 1 - xx), np.minimum(yy, PH - 1 - yy))
        age = 1 - 0.10 * np.exp(-edge / 26) - 0.05 * np.exp(-edge / 90)
        base_c = np.array([236, 230, 212], np.float32)
        lum = age * (1 + 0.028 * mottle) + 0.05 * fib * sign + grain / 255
        # the fold: a bright ridge right above a soft dark valley, and the lower half a touch shaded (it bends away)
        dy_ = yy - FOLD
        lum += 0.07 * np.exp(-((dy_ + 2) / 2.0) ** 2) - 0.11 * np.exp(-((dy_ - 3) / 3.5) ** 2) \
            - 0.045 * np.exp(-np.clip(dy_, 0, None) / 300) * (dy_ > 0) + 0.015 * np.exp(np.clip(dy_, None, 0) / 200)
        tint = np.dstack([np.ones_like(lum), 1 - 0.006 * mottle, 1 - 0.08 * (1 - age)])   # yellower where aged
        st["paper"] = np.clip(base_c[None, None] * lum[..., None] * tint, 0, 255).astype(np.float32)
        inkn = cv2.GaussianBlur(r.normal(0, 1, (PH, PW)).astype(np.float32), (0, 0), 9)
        inkn /= max(1e-6, float(np.abs(inkn).max()))
        st["inkvar"] = np.clip(0.9 + 0.12 * inkn + 0.04 * r.normal(0, 1, (PH, PW)).astype(np.float32), 0.74, 1.0)
        return st["paper"], st["inkvar"]

    def page(i, revealed):
        """The front page of link i (RGBA, PW x PH) with an empty hole where the core photo goes."""
        key = ("page", i, revealed)
        if key in st:
            return st[key]
        an, bn = ctx.link_names(i)
        link = ctx.spec["links"][i]
        sens = ctx.sensitive(an) or ctx.sensitive(bn)
        pim = Image.new("RGB", (PW, PH), (255, 255, 255))      # the ink, typeset on white, then printed on the paper
        d = ImageDraw.Draw(pim)
        M = 14
        # masthead
        name = PAPERS[i % len(PAPERS)]
        size = 54
        while font("Black", size).getlength(name) > PW - 280 and size > 30:
            size -= 2
        d.text((PW / 2, 6), name, font=font("Black", size), fill=INKC, anchor="ma")
        f15 = font("Bold", 15)
        d.text((M, 12), "SIX DEGREES", font=f15, fill=INKC)
        d.text((M, 32), "OR LESS", font=f15, fill=INKC)
        d.text((PW - M, 12), "EXTRA" if revealed else "LATE", font=font("Black", 17 if revealed else 15),
               fill=(170, 20, 24) if revealed and not sens else INKC, anchor="ra")
        d.text((PW - M, 32), "EDITION", font=f15, fill=INKC, anchor="ra")
        y = 68
        d.rectangle([M, y, PW - M, y + 3], fill=INKC)
        d.rectangle([M, y + 6, PW - M, y + 7], fill=INKC)
        yr = year(i)
        f16 = font("Bold", 16)
        d.text((M, y + 13), f"VOL. {i + 1}  ·  No. {(i + 1) * 37}", font=f16, fill=INKC)
        if yr:
            d.text((PW / 2, y + 12), str(yr), font=font("Black", 18), fill=INKC, anchor="ma")
        d.text((PW - M, y + 13), "FIVE CENTS", font=f16, fill=INKC, anchor="ra")
        y += 36
        d.rectangle([M, y, PW - M, y + 2], fill=INKC)
        # headline, between the dateline and the photo
        hy0, hy1 = y + 6, WY - 8
        hf, lines = fit_headline(headline_text(i, revealed), PW - 2 * M, hy1 - hy0)
        lh = int(hf.size * 1.04)
        hy = hy0 + (hy1 - hy0 - lh * len(lines)) / 2
        for ln in lines:
            d.text((PW / 2, hy), ln, font=hf, fill=INKC, anchor="ma")
            hy += lh
        # side columns next to the photo
        lw = WX - M - 12
        side = SIDEBARS[i % len(SIDEBARS)]
        if sens:                                               # no jokey voice next to a sensitive story
            side = ("ARCHIVE", "FROM THE PICTURE DESK")
        d.text((M, WY), "NATION", font=font("Black", 13), fill=INKC)
        column(d, M, WY + 20, WY + fh, lw, i * 31, 12)
        rx = WX + fw + 12
        d.text((rx, WY), side[0], font=font("Black", 13 if len(side[0]) < 9 else 11), fill=INKC)
        sf = font("Bold", 11)
        sy = WY + 20
        for ln in wrap(side[1], sf, lw):
            d.text((rx, sy), ln, font=sf, fill=(70, 62, 54))
            sy += 14
        column(d, rx, sy + 6, WY + fh, lw, i * 31 + 2, 12)
        d.rectangle([WX - 7, WY, WX - 6, WY + fh], fill=(150, 140, 126))
        d.rectangle([WX + fw + 5, WY, WX + fw + 6, WY + fh], fill=(150, 140, 126))
        # under the photo (below the captions' band): picture caption, rule, three columns
        y = WY + fh + 52
        cap = f"PICTURED: {ctx.C.labels[an].upper()} AND {ctx.C.labels[bn].upper() if revealed else '???'}"
        cs = 18
        while font("Bold", cs).getlength(cap) > 520 and cs > 11:
            cs -= 1
        d.text((M, y), cap, font=font("Bold", cs), fill=INKC)
        y += 64
        d.rectangle([M, y, PW - M, y + 2], fill=INKC)
        y += 12
        cw = (PW - 2 * M - 2 * 18) / 3
        for k in range(3):
            x = M + k * (cw + 18)
            d.text((x, y), ["WORLD", "CITY", "LETTERS"][k], font=font("Black", 17), fill=INKC)
            column(d, x, y + 24, PH - 20, cw, i * 31 + 5 + k, 13)
            if k:
                d.rectangle([x - 10, y, x - 9, PH - 22], fill=(150, 140, 126))
        d.rectangle([WX - 2, WY - 2, WX + fw + 1, WY + fh + 1], fill=INKC)
        paper, inkvar = newsprint()
        ink = 1 - np.asarray(pim, np.float32) / 255                 # ink coverage per channel
        out = paper * (1 - ink * inkvar[..., None])                  # multiplied into the paper, slightly uneven
        pim = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGBA")
        st[key] = pim
        return pim

    def shadow():
        if "sh" not in st:
            pad = 40
            sh = np.zeros((PH + 2 * pad, PW + 2 * pad), np.float32)
            sh[pad + 14:pad + PH + 14, pad + 10:pad + PW + 10] = 170
            sh = cv2.GaussianBlur(sh, (0, 0), 16)
            a = np.zeros((*sh.shape, 4), np.uint8)
            a[..., 3] = sh.astype(np.uint8)
            st["sh"] = Image.fromarray(a)
        return st["sh"]

    def page_at(i, local, seg):
        R = seg.get("rv", ctx.REVEAL) * seg["speech"] + ctx.THROW
        since = local - R
        if since <= 0:
            return page(i, False)
        a = ctx.ease(min(1.0, since / REPRINT))
        return page(i, True) if a >= 1 else Image.blend(page(i, False), page(i, True), a)

    def old_page(i):
        key = ("old", i)
        if key not in st:
            im = page(i, True).copy()
            im.alpha_composite(ctx.view(i, *ctx.both_cam(i, fw, fh, 0.75), fw, fh).convert("RGBA"), (WX, WY))
            st[key] = im
        return st[key]

    # background: a dark wooden desk, larger than the screen, sliding slowly under a lamp whose light wanders
    DM = 260                                                  # desk margin around the screen (how far it can slide)

    def desk():
        if "desk" not in st:
            r = np.random.default_rng(5)
            dh, dw = H + 2 * DM, W + 2 * DM
            yy, xx = np.mgrid[0:dh, 0:dw].astype(np.float32)
            warp = cv2.GaussianBlur(r.normal(0, 1, (dh // 8, dw // 8)).astype(np.float32), (0, 0), 6)
            warp = cv2.resize(warp / max(1e-6, float(np.abs(warp).max())), (dw, dh))
            along = yy * 0.9 + xx * 0.12                     # long grain, slightly diagonal
            g = np.sin((along + 60 * warp) / 7.0) * 0.5 + 0.5
            fine = cv2.GaussianBlur(r.normal(0, 1, (dh, dw)).astype(np.float32), (0, 0), sigmaX=9, sigmaY=0.8)
            plank = (along // 310).astype(int) % 3
            tone = np.array([1.0, 0.86, 0.93], np.float32)[plank]
            lum = (0.75 + 0.25 * g + 0.08 * fine) * tone
            lum = np.where((along % 310) < 3, 0.45, lum)    # seams between the planks
            col = np.array([92, 58, 34], np.float32)
            st["desk"] = np.clip(col[None, None] * lum[..., None], 0, 255).astype(np.uint8)
            st["lamp"] = np.mgrid[0:H:8, 0:W:8].astype(np.float32)
        return st["desk"]

    def backdrop(tt, i):
        dk = desk()
        ox = int(DM + DM * 0.85 * math.sin(tt * 0.23 + i * 1.7))
        oy = int(DM + DM * 0.85 * math.sin(tt * 0.17 + 0.6 + i))
        arr = dk[oy:oy + H, ox:ox + W].astype(np.float32)
        lx_, ly_ = ctx.CX + 380 * math.sin(tt * 0.31), 900 + 500 * math.sin(tt * 0.21 + 1)
        ly, lx = st["lamp"]
        small = np.exp(-(((lx - lx_) / 700) ** 2 + ((ly - ly_) / 900) ** 2))
        gain = cv2.resize((0.45 + 0.85 * small).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        arr *= gain[..., None]
        return Image.fromarray(np.dstack([np.clip(arr, 0, 255).astype(np.uint8), np.full((H, W), 255, np.uint8)]))

    def tilt(im, ang, center):
        """Rotate an RGBA image by ang degrees (counter-clockwise) about center, same size (cv2: fast)."""
        if abs(ang) < 0.02:
            return im
        M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), ang, 1.0)
        out = cv2.warpAffine(np.asarray(im), M, im.size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))
        return Image.fromarray(out)

    def held(tt):
        """The page's gentle drift (dx, dy) and sway (degrees): someone holding it over the desk."""
        dx = 16 * math.sin(tt * 0.47) + 5 * math.sin(tt * 1.13 + 0.4)
        dy = 6 + 7 * math.sin(tt * 0.61 + 1.2) + 3 * math.sin(tt * 1.7)
        ang = 1.3 * math.sin(tt * 0.53 + 0.3) + 0.45 * math.sin(tt * 1.21 + 2.0)
        return dx, dy, ang

    ENTRIES = [(-1.0, 0.15), (1.0, -0.1), (0.0, 0.6), (-1.0, -0.2), (1.0, 0.25), (0.0, 0.6)]

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        base = backdrop(tt, i)
        # the previous paper drops away underneath
        DROP = SPIN + 0.25
        if i > 0 and not ctx.C.fj["links"][i - 1].get("lineage") and local < DROP:
            v = ctx.ease(local / DROP)
            op = old_page(i - 1)
            w2, h2 = int(PW * (1 - 0.12 * v)), int(PH * (1 - 0.12 * v))
            op = op.resize((w2, h2), Image.BILINEAR).rotate(4 * v * (-1) ** i, expand=True, resample=Image.BICUBIC)
            dark = Image.new("RGBA", op.size, (0, 0, 0, int(140 * v)))
            op = Image.composite(Image.alpha_composite(op, dark), op, op.getchannel("A"))
            ctx.paste(base, op, PX0 + PW / 2, PY0 + PH / 2 + 300 * v * v, alpha=1 - v ** 2, nearest=False)
        # this page slides in
        u = min(1.0, local / SPIN) if i else 1.0
        e = 1 - (1 - u) ** 3
        dx, dy = ENTRIES[i % len(ENTRIES)]
        ox, oy = (1 - e) * dx * 1000, (1 - e) * dy * 1400
        if 0 < local - SPIN < 0.2:                               # a soft thump on landing
            oy += 6 * math.sin(math.pi * (local - SPIN) / 0.2)
        pg = page_at(i, local, seg)
        hx, hy, ang = held(tt)
        ox, oy = ox + hx, oy + hy
        spot = {}

        def mount(b, X0, Y0, w, h):
            px, py = X0 - WX, Y0 - WY
            spot["sh"] = (int(px - 40), int(py - 40))
            b.alpha_composite(pg, (int(px), int(py)))

        # the page and the core (photo window + Buh) go on one layer, which is then tilted as a whole, so the window
        # rides the page exactly; its shadow falls on the desk, a little further out when the page sways more
        L0 = 120                                             # the layer starts here (Buh can jump above the page)
        layer = Image.new("RGBA", (W, H - L0), (0, 0, 0, 0))
        c = ctx.draw_core(layer, seg, tt, PX0 + WX + ox, WIN_Y + oy - L0, SC, frame=mount)
        X0, Y0, w, h = c["rect"]
        pcx, pcy = X0 + w / 2, Y0 + h / 2                    # tilt about the photo's centre (layer coords)
        sx_, sy_ = spot["sh"]
        so = 1 + abs(ang) * 3
        base.alpha_composite(tilt(shadow(), ang, (pcx - sx_, pcy - sy_)), (int(sx_ + so), int(sy_ + L0 + so)))
        base.alpha_composite(tilt(layer, ang, (pcx, pcy)), (0, L0))
        a = math.radians(ang)
        ddx, ddy = w / 2, h / 2 + 32                         # the credit's spot under the photo, tilted with it
        cr = (pcx + ddx * math.cos(a) + ddy * math.sin(a), pcy - ddx * math.sin(a) + ddy * math.cos(a) + L0)
        c["rect"] = (X0, Y0 + L0, w, h)
        return ctx.core_finish(base, seg, tt, c, connected_y=int(Y0 + L0 + h - 130),
                               credit_xy=(int(cr[0]), int(cr[1])))

    return frame
