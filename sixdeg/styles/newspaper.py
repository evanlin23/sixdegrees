"""Spinning newspaper: each link is a front page that spins in from the centre and lands tilted; the photo is the
halftone front-page picture, the fact is the headline; the previous paper drops away underneath."""
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


def make(ctx):
    PW, PH = 790, 990                       # the front page
    PAD = 46                                # room for the drop shadow around it
    PY = 785                                # where a landed page sits (centre)
    SPIN = 0.85                             # spin-in time, s
    PX, PHH = 730, 420                      # front-page photo (full width)
    TAG_Y = 1324
    CAP_Y = 1406
    INKC = (24, 20, 16)
    GREY = (112, 104, 94)

    def font(name, size):
        return ImageFont.truetype(str(ctx.FONTS / f"Inter-{name}.otf"), size)

    def year(i):
        for j in range(i, -1, -1):
            y = ctx.year_of(j)
            if y:
                return y
        return None

    def headline_text(i):
        link = ctx.spec["links"][i]
        txt = ctx.fact_of(link)
        if txt.endswith("…") and not link.get("fact"):          # the auto fact is cut short: use the whole line
            txt = re.sub(r"(?i)^(?:And |Then |So )?(?:(?:In|By|Around|On|Back in)\s+)?\d{4}s?,\s*", "",
                         link["line"].strip())
            txt = re.sub(r"(?i)^(?:And |Then |So )?that year,\s*", "", txt)
        txt = re.sub(r"^\d{4}s?:\s*", "", txt).rstrip(".…")
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
        for size in range(88, 34, -2):
            f = font("Black", size)
            lines = wrap(text, f, width)
            if len(lines) <= 4 and len(lines) * size * 1.02 <= max_h and all(f.getlength(l) <= width for l in lines):
                return f, lines
        f = font("Black", 34)
        return f, wrap(text, f, width)[:4]

    def gibberish_lines(d, x, y0, y1, width, seed, size=13, gap=17):
        r = random.Random(seed)
        f = font("Bold", size)
        y = y0
        while y + size <= y1:
            line = ""
            while True:
                w = "".join(r.choice("eeeeaaaoooiiinnnssrrtlldcumhpgbwyfk") for _ in range(r.randint(2, 9)))
                if f.getlength(f"{line} {w}".strip()) > width:
                    break
                line = f"{line} {w}".strip()
            if r.random() < 0.08:                        # a paragraph end
                line = line[: max(3, len(line) // 2)]
            d.text((x, y), line, font=f, fill=GREY)
            y += gap

    def halftone(im):
        g = np.asarray(im.convert("L"), np.float32) / 255
        lo, hi = np.percentile(g, 2), np.percentile(g, 98)
        g = np.clip((g - lo) / max(0.05, hi - lo), 0, 1)
        h, w = g.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        c, s, P = math.cos(math.radians(45)), math.sin(math.radians(45)), 5.0
        u, v = (xx * c + yy * s) / P, (-xx * s + yy * c) / P
        dist = np.hypot(u - np.round(u), v - np.round(v))
        dots = (dist > np.sqrt(1 - g) * 0.62).astype(np.float32)        # ink dots grow with darkness
        t = 0.62 * g + 0.38 * dots
        dark, light = np.array([46, 34, 24], np.float32), np.array([236, 224, 196], np.float32)
        rgb = dark + t[..., None] * (light - dark)
        return Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")

    _pages, _cam = {}, {}

    def page(i):
        """The page as a PIL RGBA sprite (with shadow) plus where B's face sits on it (for the marker ring)."""
        if i in _pages:
            return _pages[i]
        an, bn = ctx.link_names(i)
        A, B = ctx.C.labels[an].upper(), ctx.C.labels[bn].upper()
        im = Image.new("RGBA", (PW + 2 * PAD, PH + 2 * PAD), (0, 0, 0, 0))
        sh = np.zeros((PH + 2 * PAD, PW + 2 * PAD), np.float32)
        sh[PAD + 14:PAD + PH + 14, PAD + 10:PAD + PW + 10] = 150
        sh = cv2.GaussianBlur(sh, (0, 0), 14)
        shadow = np.zeros((*sh.shape, 4), np.uint8)
        shadow[..., 3] = sh.astype(np.uint8)
        im.alpha_composite(Image.fromarray(shadow))
        # paper: warm newsprint with grain and darker, aged edges
        r = np.random.default_rng(i)
        yy, xx = np.mgrid[0:PH, 0:PW].astype(np.float32)
        edge = np.minimum(np.minimum(xx, PW - 1 - xx), np.minimum(yy, PH - 1 - yy))
        age = 1 - 0.13 * np.exp(-edge / 30)
        grain = r.normal(0, 5, (PH, PW)).astype(np.float32)
        paper = np.array([238, 230, 210], np.float32)[None, None] * age[..., None] + grain[..., None]
        pim = Image.fromarray(np.clip(paper, 0, 255).astype(np.uint8)).convert("RGBA")
        d = ImageDraw.Draw(pim)
        M = 30
        # top line with ears
        f18 = font("Bold", 17)
        d.text((M, 16), "SIX DEGREES OR LESS", font=f18, fill=INKC)
        d.text((PW - M, 16), "LATE EDITION", font=f18, fill=INKC, anchor="ra")
        # masthead
        name = PAPERS[i % len(PAPERS)]
        size = 104
        while font("Black", size).getlength(name) > PW - 2 * M and size > 40:
            size -= 2
        d.text((PW / 2, 40), name, font=font("Black", size), fill=INKC, anchor="ma")
        y = 40 + size * 1.0 + 8
        d.rectangle([M, y, PW - M, y + 4], fill=INKC)
        d.rectangle([M, y + 8, PW - M, y + 9], fill=INKC)
        yr = year(i)
        f20 = font("Bold", 20)
        d.text((M, y + 17), f"VOL. {i + 1}  ·  No. {(i + 1) * 37}", font=f20, fill=INKC)
        if yr:
            d.text((PW / 2, y + 17), str(yr), font=font("Black", 22), fill=INKC, anchor="ma")
        d.text((PW - M, y + 17), "FIVE CENTS", font=f20, fill=INKC, anchor="ra")
        y += 50
        d.rectangle([M, y, PW - M, y + 2], fill=INKC)
        # headline
        y += 14
        hf, lines = fit_headline(headline_text(i), PW - 2 * M, 240)
        for ln in lines:
            d.text((PW / 2, y), ln, font=hf, fill=INKC, anchor="ma")
            y += int(hf.size * 1.02)
        y += 8
        d.rectangle([M, y, PW - M, y + 2], fill=INKC)
        y += 16
        # photo (left) and a text column (right)
        ph_h = int(min(PHH, PH - y - 130))
        cam = ctx.both_cam(i, PX, ph_h, 0.8)
        photo = halftone(ctx.view(i, *cam, PX, ph_h))
        pim.alpha_composite(photo, (M, int(y)))
        d.rectangle([M - 1, y - 1, M + PX, y + ph_h], outline=INKC, width=2)
        bx, by = ctx.ctr(ctx.fbox(i, "b"))
        rr = max(36, ctx.fbox(i, "b")[3] * cam[2] * 0.72)
        _cam[i] = (M + (bx - cam[0]) * cam[2] + PX / 2, y + (by - cam[1]) * cam[2] + ph_h / 2, rr)
        y += ph_h + 8
        cap_y = y
        y += 34
        d.rectangle([M, y, PW - M, y + 2], fill=INKC)
        y += 14
        cw = (PW - 2 * M - 2 * 18) / 3
        for k in range(3):
            x = M + k * (cw + 18)
            head = ["NATION", "WORLD", SIDEBARS[i % len(SIDEBARS)][0]][k]
            d.text((x, y), head, font=font("Black", 18), fill=INKC)
            if k == 2:
                d.text((x, y + 24), SIDEBARS[i % len(SIDEBARS)][1], font=font("Bold", 12), fill=(70, 62, 54))
            gibberish_lines(d, x, y + (44 if k == 2 else 26), PH - 24, cw, i * 31 + k)
            if k:
                d.rectangle([x - 10, y, x - 9, PH - 26], fill=(150, 140, 126))
        im.alpha_composite(pim, (PAD, PAD))
        out = {"pre": im, "cap_y": cap_y + PAD, "A": A, "B": B}
        _pages[i] = out
        return out

    def page_at(i, tt, seg):
        """The sprite for this moment as uint8 RGBA: the caption names B only after the reveal; a red marker ring
        draws itself around B's face."""
        p = page(i)
        im = p["pre"].copy()
        d = ImageDraw.Draw(im)
        R = seg.get("rv", ctx.REVEAL) * seg["speech"] + ctx.THROW
        since = tt - seg["t0"] - R
        cap = f"PICTURED: {p['A']} AND {p['B'] if since > 0 else '???'}"
        size = 21
        while font("Bold", size).getlength(cap) > PW - 60 and size > 12:
            size -= 1
        d.text((PAD + 30, p["cap_y"] + 4), cap, font=font("Bold", size), fill=INKC)
        if since > 0:
            fx, fy, rr = _cam[i]
            fx, fy = fx + PAD, fy + PAD
            sweep = 360 * ctx.ease(min(1.0, since / 0.35))
            for o in range(4):
                d.arc([fx - rr * 1.1 - o, fy - rr - o, fx + rr * 1.1 + o, fy + rr + o], -100, -100 + sweep,
                      fill=(205, 22, 30, 255), width=3)
        return np.asarray(im)

    def blit(base, spr, cx, cy, ang, sc, alpha=1.0, dim=1.0):
        h, w = spr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        M[0, 2] += cx - w / 2
        M[1, 2] += cy - h / 2
        cs = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32) @ M.T
        x0, x1 = max(0, int(cs[:, 0].min())), min(ctx.W, int(math.ceil(cs[:, 0].max())) + 1)
        y0, y1 = max(0, int(cs[:, 1].min())), min(ctx.H, int(math.ceil(cs[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0 or sc < 0.01:
            return
        M[0, 2] -= x0
        M[1, 2] -= y0
        out = cv2.warpAffine(spr, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))
        a = out[..., 3:4].astype(np.float32) * (alpha / 255)
        reg = base[y0:y1, x0:x1].astype(np.float32)
        base[y0:y1, x0:x1] = (out[..., :3].astype(np.float32) * dim * a + reg * (1 - a)).astype(np.uint8)

    # background: dark sepia rays turning around the page (fast while a page spins in)
    BW, BH = ctx.W // 2, ctx.H // 2
    yy, xx = np.mgrid[0:BH, 0:BW].astype(np.float32)
    ang_map = np.arctan2(yy - PY / 2, xx - ctx.CX / 2)
    rad = np.hypot(yy - PY / 2, xx - ctx.CX / 2) / (BH * 0.75)
    vign = np.clip(1.15 - rad, 0.25, 1.0)[..., None]
    c1, c2 = np.array([70, 50, 32], np.float32), np.array([38, 27, 18], np.float32)

    def backdrop(phase):
        s = np.sin(ang_map * 16 + phase)
        m = np.clip(s * 3, -1, 1)[..., None] * 0.5 + 0.5
        bg = (c2 + (c1 - c2) * m) * vign
        return cv2.resize(bg.astype(np.uint8), (ctx.W, ctx.H), interpolation=cv2.INTER_LINEAR)

    def tilt(i):
        return (-1) ** i * (2.6 + (i * 1.7) % 2.0)

    def pose(i, local):
        """(cx, cy, angle, scale) of page i, `local` s after its link starts."""
        u = min(1.0, max(0.0, local / SPIN))
        e = 1 - (1 - u) ** 3
        ang = tilt(i) + (1 - e) * 720 + 0.9 * math.sin(local * 1.4 + i)
        sc = 0.04 + 0.96 * e
        if local > SPIN:                                   # thump on landing, then a slow push-in
            k = local - SPIN
            sc *= 1 + 0.035 * math.sin(math.pi * min(1.0, k / 0.22)) * (k < 0.22) + 0.012 * k
        cx = ctx.CX + 8 * math.sin(local * 0.9 + i * 2)
        cy = PY + 6 * math.sin(local * 1.1 + i)
        if local > SPIN:
            dx, dy = ctx.shake(local - SPIN, 14, 0.3)
            cx, cy = cx + dx, cy + dy
        return cx, cy, ang, sc

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        u = min(1.0, local / SPIN)
        phase = tt * 0.35 + 6.0 * (1 - (1 - u) ** 2) + i * 6.0
        base = backdrop(phase)
        # the previous page drops away under the new one
        if i > 0 and not ctx.C.fj["links"][i - 1].get("lineage") and local < SPIN + 0.3:
            pseg = ctx.segs[[s.get("i") for s in ctx.segs].index(i - 1)]
            pl = pseg["t1"] - pseg["t0"]
            cx, cy, ang, sc = pose(i - 1, pl)
            v = ctx.ease(local / (SPIN + 0.3))
            blit(base, page_at(i - 1, pseg["t1"] - 1e-3, pseg), cx - 60 * v, cy + 700 * v * v, ang - 14 * v,
                 sc * (1 - 0.3 * v), alpha=1 - v ** 3, dim=1 - 0.6 * v)
        cx, cy, ang, sc = pose(i, local)
        blit(base, page_at(i, tt, seg), cx, cy, ang, sc, dim=0.75 + 0.25 * u)
        out = Image.fromarray(base).convert("RGBA")
        # name tags under the page
        an, bn = ctx.link_names(i)
        R = seg.get("rv", ctx.REVEAL) * seg["speech"] + ctx.THROW
        span = ctx.SAFE[2] - ctx.SAFE[0] - 10
        for ts in (28, 26, 24, 22):
            ta = ctx.pixel_tag(ctx.C.labels[an].upper(), ctx.CYAN, ts)
            tb = ctx.pixel_tag(ctx.C.labels[bn].upper(), ctx.YELLOW, ts)
            if ta.width + tb.width + 20 <= span:
                break
        if ta.width + tb.width + 20 <= span:
            xa = ctx.CX - (ta.width + tb.width + 20) / 2 + ta.width / 2
            xb = xa + ta.width / 2 + 20 + tb.width / 2
            ya = yb = TAG_Y
        else:                                              # long names: stagger them on two rows
            xa, ya = ctx.SAFE[0] + 10 + ta.width / 2, TAG_Y - 36
            xb, yb = ctx.SAFE[2] - 10 - tb.width / 2, TAG_Y + 26
        bob = 4 * math.sin(tt * 3)
        if local > SPIN:
            ctx.paste(out, ta, xa, ya + bob, scale=ctx.ease_out_back(min(1.0, (local - SPIN) / 0.2)))
        if local > R:
            ctx.paste(out, tb, xb, yb - bob, scale=ctx.ease_out_back(min(1.0, (local - R) / 0.2)))
        ctx.header(out, i)
        ctx.connected(out, seg, tt)
        seg["cap"].draw(out, tt, y=CAP_Y)
        return out

    return frame
