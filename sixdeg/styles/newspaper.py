"""Newspaper: each link is a front page and the show's core photo window (draw_core: camera, Buh, string, name
tags) is its front-page photo, with the masthead, dateline and headline above it, narrow news columns beside it and
the picture caption + columns below. The headline reads "... ???" until the reveal (B is never named early), then
the page is reprinted as an EXTRA with the full headline. Each new page slides in from a different side while the
previous paper drops away underneath. A sensitive link gets a sober "ARCHIVE / FROM THE PICTURE DESK" sidebar."""
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
SENTENCES = [   # filler about our own method only: never invented claims about the real events pictured
    "Every photo in this chain is real and its source is listed below the video.",
    "Each link was checked against the original photo credit.",
    "The pictures are shown as they were taken, without edits.",
    "Viewers suggest the two names in the comments.",
    "Our pin detective follows one photograph to the next.",
    "Dates come from the photo archives that hold each picture.",
    "Some links took hours of searching through old archives.",
    "Six photographs, one chain, no shortcuts.",
    "The full research notes are in the description.",
    "Tomorrow's edition: another two names, another chain.",
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

    def column(d, x, y0, y1, width, seed, lead="", size=13):
        r = random.Random(seed)
        f = font("Medium", size)
        words = (lead + " " + " ".join(r.sample(SENTENCES, len(SENTENCES)) * 2)).split()
        y, line = y0, ""
        while words and y + size <= y1:
            w = words[0]
            if f.getlength(f"{line} {w}".strip()) > width and line:
                d.text((x, y), line, font=f, fill=GREY)
                y += int(size * 1.35)
                line = ""
                continue
            line = f"{line} {w}".strip()
            words.pop(0)

    def page(i, revealed):
        """The front page of link i (RGBA, PW x PH) with an empty hole where the core photo goes."""
        key = ("page", i, revealed)
        if key in st:
            return st[key]
        an, bn = ctx.link_names(i)
        link = ctx.spec["links"][i]
        sens = ctx.sensitive(an) or ctx.sensitive(bn)
        r = np.random.default_rng(i)
        yy, xx = np.mgrid[0:PH, 0:PW].astype(np.float32)
        edge = np.minimum(np.minimum(xx, PW - 1 - xx), np.minimum(yy, PH - 1 - yy))
        age = 1 - 0.13 * np.exp(-edge / 30)
        grain = r.normal(0, 5, (PH, PW)).astype(np.float32)
        paper = np.array([238, 230, 210], np.float32)[None, None] * age[..., None] + grain[..., None]
        pim = Image.fromarray(np.clip(paper, 0, 255).astype(np.uint8)).convert("RGBA")
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
        lead = link["line"] if revealed else masked(link["line"], link)
        column(d, M, WY + 20, WY + fh, lw, i * 31, lead, 12)
        rx = WX + fw + 12
        d.text((rx, WY), side[0], font=font("Black", 13 if len(side[0]) < 9 else 11), fill=INKC)
        sf = font("Bold", 11)
        sy = WY + 20
        for ln in wrap(side[1], sf, lw):
            d.text((rx, sy), ln, font=sf, fill=(70, 62, 54))
            sy += 14
        column(d, rx, sy + 6, WY + fh, lw, i * 31 + 2, "", 12)
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
            column(d, x, y + 24, PH - 20, cw, i * 31 + 5 + k, "", 13)
            if k:
                d.rectangle([x - 10, y, x - 9, PH - 22], fill=(150, 140, 126))
        d.rectangle([WX - 2, WY - 2, WX + fw + 1, WY + fh + 1], fill=INKC)
        st[key] = pim
        return pim

    def shadow():
        if "sh" not in st:
            pad = 40
            sh = np.zeros((PH + 2 * pad, PW + 2 * pad), np.float32)
            sh[pad + 12:pad + PH + 12, pad + 8:pad + PW + 8] = 150
            sh = cv2.GaussianBlur(sh, (0, 0), 14)
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

    # background: dark sepia rays turning slowly behind the page
    BW, BH = W // 2, H // 2
    yy, xx = np.mgrid[0:BH, 0:BW].astype(np.float32)
    ang_map = np.arctan2(yy - 1000 / 2, xx - ctx.CX / 2)
    rad = np.hypot(yy - 1000 / 2, xx - ctx.CX / 2) / (BH * 0.75)
    vign = np.clip(1.15 - rad, 0.25, 1.0)[..., None]
    c1, c2 = np.array([70, 50, 32], np.float32), np.array([38, 27, 18], np.float32)

    def backdrop(phase):
        s_ = np.sin(ang_map * 16 + phase)
        m = np.clip(s_ * 3, -1, 1)[..., None] * 0.5 + 0.5
        bg = (c2 + (c1 - c2) * m) * vign
        bg = cv2.resize(bg.astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(np.dstack([bg, np.full((H, W), 255, np.uint8)]))

    ENTRIES = [(-1.0, 0.15), (1.0, -0.1), (0.0, 0.6), (-1.0, -0.2), (1.0, 0.25), (0.0, 0.6)]

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        base = backdrop(tt * 0.35 + i * 3.0)
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

        def mount(b, X0, Y0, w, h):
            px, py = X0 - WX, Y0 - WY
            b.alpha_composite(shadow(), (int(px - 40), int(py - 40)))
            b.alpha_composite(pg, (int(px), int(py)))

        c = ctx.draw_core(base, seg, tt, PX0 + WX + ox, WIN_Y + oy, SC, frame=mount)
        X0, Y0, w, h = c["rect"]
        return ctx.core_finish(base, seg, tt, c, connected_y=int(Y0 + h - 130))

    return frame
