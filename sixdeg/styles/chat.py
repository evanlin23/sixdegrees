"""Group chat skin around the show's core: a dark messaging thread ("Six Degrees Group") where Buh posts the link's
fact and then a photo, and that sent photo IS the core window (camera, Buh, red string, name tags). Someone is typing
until Buh lands on B; then a neutral system line says B was added to the group (no words put in anyone's mouth). B is never named before the reveal."""
from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

NAME = "chat"

SC = 0.83                              # core window scale (730 x 788)
WX, WY = 112, 584                      # core window top-left
HEAD = (412, 494)                      # chat header bar
BG = (14, 20, 28)
PANEL = (26, 34, 46)
BUBBLE = (40, 50, 66)
TXT = (234, 238, 244)
DIM = (140, 150, 166)
GREEN = (90, 210, 130)
BLUE = (64, 140, 255)


def make(ctx):
    _fonts, _cache = {}, {}

    def F(size, weight="Bold"):
        k = (size, weight)
        if k not in _fonts:
            _fonts[k] = ImageFont.truetype(str(ctx.FONTS / f"Inter-{weight}.otf"), size)
        return _fonts[k]

    def label(n):
        return ctx.C.labels.get(n, n)

    def clip(text, font, maxw):
        if font.getlength(text) <= maxw:
            return text
        text = text.rstrip("…")
        while text and font.getlength(text + "…") > maxw:
            text = text[:-1]
        return text.rstrip(" ,;:") + "…"

    def credit_left(i):
        """x where the core's photo credit (right-aligned at the card's right edge, under it) starts."""
        cr = ctx.C.fj["credits"][i]
        who = (cr.get("artist") or "").strip()
        if len(who) > 26 or "(" in who or "http" in who:
            who = "Wikimedia Commons" if cr.get("file") else ""
        txt = f"PHOTO: {who + ' / ' if who else ''}{cr.get('license', '')}".upper()
        f = ImageFont.truetype(str(ctx.FONTS / "Silkscreen-Bold.ttf"), 18)
        return ctx.CARD[2] - f.getlength(txt) - 4

    def circle(im, size):
        im = im.convert("RGBA").resize((size, size), Image.BICUBIC)
        m = Image.new("L", (size * 4, size * 4), 0)
        ImageDraw.Draw(m).ellipse([0, 0, size * 4 - 1, size * 4 - 1], fill=255)
        im.putalpha(m.resize((size, size), Image.LANCZOS))
        return im

    def buh_avatar(size):
        k = ("buh", size)
        if k not in _cache:
            av = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            ImageDraw.Draw(av).ellipse([0, 0, size - 1, size - 1], fill=(255, 214, 10))
            spr = ctx.pin_sprite("happy", 0, 0, 0, scale=2)
            s = min(1.0, (size * 0.9) / max(spr.width, spr.height))
            spr = spr.resize((max(1, int(spr.width * s)), max(1, int(spr.height * s))), Image.NEAREST)
            av.alpha_composite(spr, ((size - spr.width) // 2, (size - spr.height) // 2 + 2))
            _cache[k] = av
        return _cache[k]

    def thread(i, revealed):
        """Everything static for link i: wallpaper, header, Buh's fact bubble, the reply / system line."""
        key = (i, revealed)
        if key in _cache:
            return _cache[key]
        an, bn = ctx.link_names(i)
        link = ctx.spec["links"][i]
        im = Image.new("RGBA", (ctx.W, ctx.H), BG + (255,))
        d = ImageDraw.Draw(im)
        for y in range(HEAD[1] + 30, ctx.H, 64):                                  # faint wallpaper dots
            for x in range(20 + (y // 64 % 2) * 32, ctx.W, 64):
                d.ellipse([x, y, x + 5, y + 5], fill=(22, 30, 40))
        # header: back chevron, group avatar (faces met so far), name, status, menu
        d.rectangle([0, HEAD[0], ctx.W, HEAD[1]], fill=PANEL)
        cy = (HEAD[0] + HEAD[1]) / 2
        d.line([60, cy - 16, 44, cy, 60, cy + 16], fill=TXT, width=5, joint="curve")
        faces = [ctx.C.people[j] for j in range(max(0, i - 1), i + 1)] + ([bn] if revealed else [])
        x = 82
        for k, nm in enumerate(faces[-3:]):
            av = circle(ctx.sharp[nm], 56)
            ImageDraw.Draw(im).ellipse([x - 3, cy - 31, x + 59, cy + 31], fill=PANEL)
            im.alpha_composite(av, (x, int(cy - 28)))
            x += 34
        x += 40
        d.text((x, cy - 14), "Six Degrees Group", font=F(28, "Bold"), fill=TXT, anchor="lm")
        status = f"{label(bn)} joined" if revealed else ("Buh is sending a photo…")
        d.text((x, cy + 20), clip(status, F(21, "Medium"), 1000 - x - 60), font=F(21, "Medium"),
               fill=GREEN if revealed else DIM, anchor="lm")
        for k in range(3):
            d.ellipse([1020, cy - 16 + k * 14, 1028, cy - 8 + k * 14], fill=TXT)
        # Buh's message: the fact (cut before B until the reveal)
        im.alpha_composite(buh_avatar(46), (54, 516))
        f = F(24, "Medium")
        fact = clip(ctx.fact_of(link, full=revealed), f, 860 - 112 - 40)
        w = int(max(F(18).getlength("Buh"), f.getlength(fact))) + 36
        d.rounded_rectangle([WX, 504, WX + w, 568], 18, fill=BUBBLE)
        d.text((WX + 18, 520), "Buh", font=F(18, "Bold"), fill=(255, 214, 10), anchor="lm")
        d.text((WX + 18, 548), fact, font=f, fill=TXT, anchor="lm")
        # after the reveal: B's reply, or a neutral system line on a link with a dictator
        if revealed:
            room = credit_left(i) - WX - 24
            # owner: no "that's me!" reply: never put words in a real person's mouth. A neutral system line
            # for everyone.
            pf = F(21, "Medium")
            txt = clip(f"{label(bn)} was added to the group", pf, room - 40)
            pw = pf.getlength(txt) + 40
            d.rounded_rectangle([WX, 1398, WX + pw, 1436], 19, fill=PANEL)
            d.text((WX + pw / 2, 1417), txt, font=pf, fill=DIM, anchor="mm")
        _cache[key] = im
        return im

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        trev = seg.get("rv", ctx.REVEAL) * seg["speech"] + ctx.THROW     # Buh lands on B
        revealed = local >= trev
        base = thread(i, revealed).copy()
        if not revealed and local > 0.6:                 # someone is typing: three bouncing dots
            d = ImageDraw.Draw(base)
            d.rounded_rectangle([WX, 1396, WX + 110, 1440], 22, fill=BUBBLE)
            for k in range(3):
                up = 6 * max(0.0, math.sin(local * 7 - k * 0.9))
                x = WX + 30 + k * 25
                d.ellipse([x - 7, 1418 - 7 - up, x + 7, 1418 + 7 - up], fill=DIM)

        def mount(b, X0, Y0, w, h):                     # the sent-photo bubble around the window
            dr = ImageDraw.Draw(b)
            dr.rounded_rectangle([X0 - 10, Y0 - 10, X0 + w + 10, Y0 + h + 10], 22, fill=BUBBLE)
        c = ctx.draw_core(base, seg, tt, WX, WY, SC, frame=mount)
        return ctx.core_finish(base, seg, tt, c)

    return frame
