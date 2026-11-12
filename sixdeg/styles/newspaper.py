"""Newspaper: each link is a front page that swings in (from a different side each time) and lands tilted; the
headline reads "... ???" until the reveal, when the page is reprinted with the full headline and the halftone photo
develops into colour; the previous paper drops away underneath."""
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
    PW, PH = 860, 1070                      # the front page
    PAD = 46                                # room for the drop shadow around it
    PY = 780                                # where a landed page sits (centre): y 245-1315
    SPIN = 0.36                             # swing-in time, s
    M = 30                                  # page margin
    PX = PW - 2 * M                         # front-page photo (full width)
    HB = 214                                # headline box height (same before/after the reveal)
    TAG_Y = 1352
    CAP_Y = 1430
    REPRINT = 0.3                           # crossfade to the revealed page, s
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

    def headline_text(i, revealed):
        """The front-page headline: cut before B's name ("... ???") until the reveal, then the whole thing."""
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
        for size in range(84, 34, -2):
            f = font("Black", size)
            lines = wrap(text, f, width)
            if len(lines) <= 4 and len(lines) * size * 1.02 <= max_h and all(f.getlength(l) <= width for l in lines):
                return f, lines
        f = font("Black", 34)
        return f, wrap(text, f, width)[:4]

    SENTENCES = [
        "Photographers from several papers were on hand to record the meeting.",
        "Officials declined to say how long the two men had spoken in private.",
        "Witnesses described the mood in the room as formal but cordial.",
        "The visit was announced only a few days in advance.",
        "Our correspondent filed this report late last night.",
        "The photograph above was taken shortly after their arrival.",
        "Crowds gathered outside long before the doors opened.",
        "Further details are expected in tomorrow's edition.",
        "Readers have written in with questions about the encounter.",
        "Neither side released a statement after the meeting.",
        "The guest list was kept short and the press were held at the door.",
        "Historians still argue about what was said that day.",
    ]

    def column_text(d, x, y0, y1, width, seed, lead=""):
        """Real sentences set as a newspaper column (the link's own line first, cut before B until the reveal)."""
        r = random.Random(seed)
        f = font("Medium", 14)
        words = (lead + " " + " ".join(r.sample(SENTENCES, len(SENTENCES)))).split()
        y, line = y0, ""
        while words and y + 14 <= y1:
            w = words[0]
            if f.getlength(f"{line} {w}".strip()) > width and line:
                d.text((x, y), line, font=f, fill=GREY)
                y += 19
                line = ""
                continue
            line = f"{line} {w}".strip()
            words.pop(0)
        if line and y + 14 <= y1:
            d.text((x, y), line, font=f, fill=GREY)

    def halftone(im):
        """(sepia halftone, colour print with a faint dot screen) of the same picture."""
        rgb0 = np.asarray(im.convert("RGB"), np.float32)
        g = rgb0.mean(2) / 255
        lo, hi = np.percentile(g, 2), np.percentile(g, 98)
        g = np.clip((g - lo) / max(0.05, hi - lo), 0, 1)
        h, w = g.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        c, s_, P = math.cos(math.radians(45)), math.sin(math.radians(45)), 4.0
        u, v = (xx * c + yy * s_) / P, (-xx * s_ + yy * c) / P
        dist = np.hypot(u - np.round(u), v - np.round(v))
        dots = (dist > np.sqrt(1 - g) * 0.62).astype(np.float32)        # ink dots grow with darkness
        t = 0.7 * g + 0.3 * dots
        dark, light = np.array([46, 34, 24], np.float32), np.array([236, 224, 196], np.float32)
        sep = dark + t[..., None] * (light - dark)
        screen = cv2.GaussianBlur(dots, (0, 0), 0.6)
        col = rgb0 * (0.9 + 0.1 * screen[..., None]) * np.array([1.0, 0.98, 0.93], np.float32)   # warm newsprint
        col = col * 0.94 + 10
        return (Image.fromarray(sep.clip(0, 255).astype(np.uint8)).convert("RGBA"),
                Image.fromarray(col.clip(0, 255).astype(np.uint8)).convert("RGBA"))

    def cover_cam(i, w, h, fill):
        cx, cy, s = ctx.both_cam(i, w, h, fill)
        ih, iw = ctx.C.imgs[i].shape[:2]
        s = max(s, w / iw, h / ih) * 1.002
        return min(max(cx, w / 2 / s), iw - w / 2 / s), min(max(cy, h / 2 / s), ih - h / 2 / s), s

    _pages, _geo = {}, {}

    def shadow_layer():
        if "sh" not in _pages:
            im = Image.new("RGBA", (PW + 2 * PAD, PH + 2 * PAD), (0, 0, 0, 0))
            sh = np.zeros((PH + 2 * PAD, PW + 2 * PAD), np.float32)
            sh[PAD + 14:PAD + PH + 14, PAD + 10:PAD + PW + 10] = 150
            sh = cv2.GaussianBlur(sh, (0, 0), 14)
            shadow = np.zeros((*sh.shape, 4), np.uint8)
            shadow[..., 3] = sh.astype(np.uint8)
            im.alpha_composite(Image.fromarray(shadow))
            _pages["sh"] = im
        return _pages["sh"]

    def page(i, revealed):
        """The page as a uint8 RGBA sprite (with shadow). Before the reveal: "???" headline, sepia halftone photo;
        after: full headline, colour photo, B named in the caption. Also records the faces' page positions."""
        key = (i, revealed)
        if key in _pages:
            return _pages[key]
        an, bn = ctx.link_names(i)
        link = ctx.spec["links"][i]
        A, B = ctx.C.labels[an].upper(), ctx.C.labels[bn].upper()
        im = shadow_layer().copy()
        r = np.random.default_rng(i)
        yy, xx = np.mgrid[0:PH, 0:PW].astype(np.float32)
        edge = np.minimum(np.minimum(xx, PW - 1 - xx), np.minimum(yy, PH - 1 - yy))
        age = 1 - 0.13 * np.exp(-edge / 30)
        grain = r.normal(0, 5, (PH, PW)).astype(np.float32)
        paper = np.array([238, 230, 210], np.float32)[None, None] * age[..., None] + grain[..., None]
        pim = Image.fromarray(np.clip(paper, 0, 255).astype(np.uint8)).convert("RGBA")
        d = ImageDraw.Draw(pim)
        f18 = font("Bold", 17)
        d.text((M, 16), "SIX DEGREES OR LESS", font=f18, fill=INKC)
        d.text((PW - M, 16), "EXTRA" if revealed else "LATE EDITION", font=f18, fill=INKC, anchor="ra")
        name = PAPERS[i % len(PAPERS)]
        size = 84
        while font("Black", size).getlength(name) > PW - 2 * M and size > 40:
            size -= 2
        d.text((PW / 2, 40), name, font=font("Black", size), fill=INKC, anchor="ma")
        y = 132
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
        y += 12
        hf, lines = fit_headline(headline_text(i, revealed), PW - 2 * M, HB - 8)
        lh = int(hf.size * 1.02)
        hy = y + (HB - lh * len(lines)) / 2
        for ln in lines:
            d.text((PW / 2, hy), ln, font=hf, fill=INKC, anchor="ma")
            hy += lh
        y += HB
        d.rectangle([M, y, PW - M, y + 2], fill=INKC)
        y += 14
        ph_h = 500
        if i not in _geo:
            cam = cover_cam(i, PX, ph_h, 0.82)
            sep, col = halftone(ctx.view(i, *cam, PX, ph_h))
            faces = {}
            for role in ("a", "b"):
                bx, by = ctx.ctr(ctx.fbox(i, role))
                faces[role] = (M + (bx - cam[0]) * cam[2] + PX / 2, y + (by - cam[1]) * cam[2] + ph_h / 2,
                               max(36, ctx.fbox(i, role)[3] * cam[2] * 0.72))
            _geo[i] = (sep, col, faces)
        sep, col, _ = _geo[i]
        pim.alpha_composite(col if revealed else sep, (M, int(y)))
        d.rectangle([M - 1, y - 1, M + PX, y + ph_h], outline=INKC, width=2)
        y += ph_h + 8
        cap = f"PICTURED: {A} AND {B if revealed else '???'}"
        cs = 21
        while font("Bold", cs).getlength(cap) > PW - 60 and cs > 12:
            cs -= 1
        d.text((M, y + 4), cap, font=font("Bold", cs), fill=INKC)
        y += 34
        d.rectangle([M, y, PW - M, y + 2], fill=INKC)
        y += 12
        cw = (PW - 2 * M - 2 * 18) / 3
        lead = link["line"] if revealed else masked(link["line"], link)
        for k in range(3):
            x = M + k * (cw + 18)
            head = ["NATION", "WORLD", SIDEBARS[i % len(SIDEBARS)][0]][k]
            d.text((x, y), head, font=font("Black", 18), fill=INKC)
            if k == 2:
                d.text((x, y + 24), SIDEBARS[i % len(SIDEBARS)][1], font=font("Bold", 12), fill=(70, 62, 54))
            column_text(d, x, y + (44 if k == 2 else 26), PH - 20, cw, i * 31 + k, lead if k == 0 else "")
            if k:
                d.rectangle([x - 10, y, x - 9, PH - 22], fill=(150, 140, 126))
        im.alpha_composite(pim, (PAD, PAD))
        _pages[key] = np.asarray(im)
        return _pages[key]

    def page_at(i, tt, seg):
        """The sprite for this moment: the reprint crossfades in at the reveal; a red ring draws around B's face."""
        R = seg.get("rv", ctx.REVEAL) * seg["speech"] + ctx.THROW
        since = tt - seg["t0"] - R
        if since <= 0:
            return page(i, False)
        a = ctx.ease(min(1.0, since / REPRINT))
        arr = page(i, True) if a >= 1 else (page(i, False) * (1 - a) + page(i, True) * a).astype(np.uint8)
        im = Image.fromarray(arr)
        d = ImageDraw.Draw(im)
        fx, fy, rr = _geo[i][2]["b"]
        fx, fy = fx + PAD, fy + PAD
        sweep = 360 * ctx.ease(min(1.0, since / 0.35))
        for o in range(4):
            d.arc([fx - rr * 1.1 - o, fy - rr - o, fx + rr * 1.1 + o, fy + rr + o], -100, -100 + sweep,
                  fill=(205, 22, 30, 255), width=3)
        return np.asarray(im)

    def blit(base, spr, cx, cy, ang, sc, alpha=1.0, dim=1.0):
        h, w = spr.shape[:2]
        Mx = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        Mx[0, 2] += cx - w / 2
        Mx[1, 2] += cy - h / 2
        cs = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32) @ Mx.T
        x0, x1 = max(0, int(cs[:, 0].min())), min(ctx.W, int(math.ceil(cs[:, 0].max())) + 1)
        y0, y1 = max(0, int(cs[:, 1].min())), min(ctx.H, int(math.ceil(cs[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0 or sc < 0.01:
            return
        Mx[0, 2] -= x0
        Mx[1, 2] -= y0
        out = cv2.warpAffine(spr, Mx, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))
        a = out[..., 3:4].astype(np.float32) * (alpha / 255)
        reg = base[y0:y1, x0:x1].astype(np.float32)
        base[y0:y1, x0:x1] = (out[..., :3].astype(np.float32) * dim * a + reg * (1 - a)).astype(np.uint8)

    # background: dark sepia rays turning slowly around the page
    BW, BH = ctx.W // 2, ctx.H // 2
    yy, xx = np.mgrid[0:BH, 0:BW].astype(np.float32)
    ang_map = np.arctan2(yy - PY / 2, xx - ctx.CX / 2)
    rad = np.hypot(yy - PY / 2, xx - ctx.CX / 2) / (BH * 0.75)
    vign = np.clip(1.15 - rad, 0.25, 1.0)[..., None]
    c1, c2 = np.array([70, 50, 32], np.float32), np.array([38, 27, 18], np.float32)

    def backdrop(phase):
        s_ = np.sin(ang_map * 16 + phase)
        m = np.clip(s_ * 3, -1, 1)[..., None] * 0.5 + 0.5
        bg = (c2 + (c1 - c2) * m) * vign
        return cv2.resize(bg.astype(np.uint8), (ctx.W, ctx.H), interpolation=cv2.INTER_LINEAR)

    def tilt(i):
        return (-1) ** i * (1.6 + (i * 1.3) % 1.4)

    # where each page swings in from (dx, dy, extra rotation): a different side each link
    ENTRIES = [(-1.0, 0.25, 18), (1.0, -0.2, -16), (0.0, 1.0, 10), (-1.0, -0.35, -14), (0.15, -1.0, 12),
               (1.0, 0.35, -18)]

    def pose(i, local, seg=None):
        """(cx, cy, angle, scale) of page i, `local` s after its link starts."""
        u = min(1.0, max(0.0, local / SPIN))
        e = 1 - (1 - u) ** 3
        dx, dy, rot = ENTRIES[i % len(ENTRIES)]
        ang = tilt(i) + (1 - e) * rot + 0.5 * math.sin(local * 1.2 + i)
        sc = 0.9 + 0.1 * e
        k = local - SPIN
        if k > 0:                                          # one soft thump on landing, then a slow push-in
            sc *= 1 + 0.02 * math.sin(math.pi * min(1.0, k / 0.2)) * (k < 0.2) + 0.008 * k
        if seg is not None:                                # the reprint lands with a small press at the reveal
            r = local - (seg.get("rv", ctx.REVEAL) * seg["speech"] + ctx.THROW)
            if 0 < r < 0.25:
                sc *= 1 + 0.018 * math.sin(math.pi * r / 0.25)
        cx = ctx.CX + 5 * math.sin(local * 0.9 + i * 2) + (1 - e) * dx * 1000
        cy = PY + 4 * math.sin(local * 1.1 + i) + (1 - e) * dy * 1500
        return cx, cy, ang, sc

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        phase = tt * 0.35 + i * 3.0
        base = backdrop(phase)
        DROP = SPIN + 0.25
        if i > 0 and not ctx.C.fj["links"][i - 1].get("lineage") and local < DROP:
            pseg = ctx.segs[[s_.get("i") for s_ in ctx.segs].index(i - 1)]
            pl = pseg["t1"] - pseg["t0"]
            cx, cy, ang, sc = pose(i - 1, pl)
            v = ctx.ease(local / DROP)
            blit(base, page_at(i - 1, pseg["t1"] - 1e-3, pseg), cx, cy + 260 * v * v, ang,
                 sc * (1 - 0.12 * v), alpha=1 - v ** 2, dim=1 - 0.5 * v)
        cx, cy, ang, sc = pose(i, local, seg)
        blit(base, page_at(i, tt, seg), cx, cy, ang, sc)
        out = Image.fromarray(base).convert("RGBA")
        # name tags under the page, each under its own face's x
        an, bn = ctx.link_names(i)
        R = seg.get("rv", ctx.REVEAL) * seg["speech"] + ctx.THROW
        page(i, False)
        faces = _geo[i][2]
        x_left = ctx.CX - PW / 2
        for ts in (30, 28, 26, 24, 22):
            ta = ctx.pixel_tag(ctx.C.labels[an].upper(), ctx.CYAN, ts)
            tb = ctx.pixel_tag(ctx.C.labels[bn].upper(), ctx.YELLOW, ts)
            if ta.width + tb.width + 30 <= ctx.W - 80:
                break
        xa, xb = x_left + faces["a"][0], x_left + faces["b"][0]
        flip = xa > xb
        l_w, r_w = (tb.width, ta.width) if flip else (ta.width, tb.width)
        xl, xr = min(xa, xb), max(xa, xb)
        gap = (xr - r_w / 2) - (xl + l_w / 2) - 24
        if gap < 0:
            xl, xr = xl + gap / 2, xr - gap / 2
        xl = max(40 + l_w / 2, xl)
        xr = min(ctx.W - 40 - r_w / 2, max(xr, xl + l_w / 2 + 24 + r_w / 2))
        xl = min(xl, xr - r_w / 2 - 24 - l_w / 2)
        xa, xb = (xr, xl) if flip else (xl, xr)
        bob = 3 * math.sin(tt * 3)
        if local > SPIN:
            ctx.paste(out, ta, xa, TAG_Y + bob, scale=ctx.ease_out_back(min(1.0, (local - SPIN) / 0.2)))
        if local > R:
            ctx.paste(out, tb, xb, TAG_Y - bob, scale=ctx.ease_out_back(min(1.0, (local - R) / 0.2)))
        ctx.header(out, i)
        ctx.connected(out, seg, tt, y=PY + 340)
        seg["cap"].draw(out, tt, y=CAP_Y)
        return out

    return frame
