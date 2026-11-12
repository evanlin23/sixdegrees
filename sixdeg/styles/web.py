"""Web browser skin around the show's core: a dark browser window (tab strip, address bar, a load bar on every new
link) shows a fictional photo-archive page whose hero picture IS the core window (camera, Buh, red string, name
tags). The tab title and the address name A until Buh lands on B, then B is appended. The page's headline under the
photo is the link's fact, cut before B's name until the reveal. Generic chrome: no real browser's or site's brand."""
from __future__ import annotations

import re

from PIL import Image, ImageDraw, ImageFont

NAME = "web"

SC = 0.85                              # core window scale (748 x 807)
FR = (34, 412, 1046, 1448)             # browser window
WX, WY = (FR[0] + FR[2]) // 2 - 374, 578     # core window top-left (centred in the page)
TABS = (FR[1], FR[1] + 56)             # tab strip
ADDR = (FR[1] + 62, FR[1] + 114)       # address bar row
PAGE_Y = FR[1] + 122                   # page viewport top
CHROME = (36, 38, 46)
TAB_ON = (58, 61, 72)
FIELD = (24, 26, 32)
TXT = (232, 234, 240)
DIM = (150, 154, 166)
PAGE = (244, 241, 234)                 # warm archive paper
ACCENT = (190, 60, 50)
SITE = "photo-archive.example"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


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
        while text and font.getlength(text + "…") > maxw:
            text = text[:-1]
        return text.rstrip() + "…"

    def credit_left(i):
        """x where the core's photo credit (right-aligned at the card's right edge, under it) starts."""
        cr = ctx.C.fj["credits"][i]
        who = (cr.get("artist") or "").strip()
        if len(who) > 26 or "(" in who or "http" in who:
            who = "Wikimedia Commons" if cr.get("file") else ""
        txt = f"PHOTO: {who + ' / ' if who else ''}{cr.get('license', '')}".upper()
        f = ImageFont.truetype(str(ctx.FONTS / "Silkscreen-Bold.ttf"), 18)
        return ctx.CARD[2] - f.getlength(txt) - 4

    def page(i, revealed):
        key = (i, revealed)
        if key in _cache:
            return _cache[key]
        an, bn = ctx.link_names(i)
        link = ctx.spec["links"][i]
        yr = ctx.year_of(i)
        im = Image.new("RGBA", (ctx.W, ctx.H), (255, 255, 255, 255))
        d = ImageDraw.Draw(im)
        # browser window: shadow, frame, page
        d.rounded_rectangle([FR[0] + 8, FR[1] + 10, FR[2] + 8, FR[3] + 10], 26, fill=(200, 202, 208))
        d.rounded_rectangle(FR, 26, fill=CHROME)
        d.rectangle([FR[0] + 6, PAGE_Y, FR[2] - 6, FR[3] - 6], fill=PAGE)
        # tab strip: window dots, the active tab, a second tab, "+"
        cy = (TABS[0] + TABS[1]) / 2 + 2
        for k, col in enumerate(((255, 95, 87), (254, 188, 46), (40, 200, 64))):
            d.ellipse([FR[0] + 24 + k * 30, cy - 9, FR[0] + 42 + k * 30, cy + 9], fill=col)
        tx0, tx1 = FR[0] + 128, FR[0] + 620
        d.rounded_rectangle([tx0, TABS[0] + 10, tx1, TABS[1] + 12], 14, fill=TAB_ON)
        d.rectangle([tx0, TABS[1] - 2, tx1, TABS[1] + 6], fill=TAB_ON)
        d.rounded_rectangle([tx0 + 16, cy - 11, tx0 + 38, cy + 11], 5, fill=ACCENT)          # favicon
        title = f"{label(an)} and {label(bn) if revealed else '???'}" + (f" ({yr})" if yr else "")
        tf = F(22, "Medium")
        d.text((tx0 + 50, cy), clip(title, tf, tx1 - tx0 - 100), font=tf, fill=TXT, anchor="lm")
        d.text((tx1 - 26, cy), "×", font=F(26, "Medium"), fill=DIM, anchor="mm")
        d.text((tx1 + 24, cy), clip("Old photos – archive", tf, 230), font=tf, fill=DIM, anchor="lm")
        d.text((FR[2] - 40, cy), "+", font=F(34, "Medium"), fill=DIM, anchor="mm")
        # address row: back / forward / reload, then the field with a lock and the URL
        acy = (ADDR[0] + ADDR[1]) / 2
        x = FR[0] + 30
        d.line([x + 14, acy - 11, x + 3, acy, x + 14, acy + 11], fill=TXT, width=4, joint="curve")
        x += 46
        d.line([x + 3, acy - 11, x + 14, acy, x + 3, acy + 11], fill=DIM, width=4, joint="curve")
        x += 46
        d.arc([x, acy - 13, x + 26, acy + 13], 40, 330, fill=TXT, width=4)
        d.polygon([(x + 21, acy - 17), (x + 29, acy - 6), (x + 17, acy - 5)], fill=TXT)
        f0 = x + 54
        d.rounded_rectangle([f0, ADDR[0] + 4, FR[2] - 30, ADDR[1] - 4], 22, fill=FIELD)
        lx = f0 + 22                                                                        # lock
        d.rounded_rectangle([lx, acy - 3, lx + 18, acy + 12], 3, fill=DIM)
        d.arc([lx + 3, acy - 13, lx + 15, acy + 3], 180, 360, fill=DIM, width=3)
        path = f"/{yr}/" if yr else "/photo/"
        url_a = f"{SITE}{path}{slug(label(an))}-and-"
        url_b = slug(label(bn)) if revealed else ""
        uf = F(24, "Medium")
        full, room = url_a + url_b, FR[2] - 60 - (lx + 34)
        tail = full[len(SITE):]
        while uf.getlength(SITE + "/…" + tail) > room and len(tail) > 1 and uf.getlength(full) > room:
            tail = tail[1:]                              # long address: keep its end (B's name) in view
        if tail != full[len(SITE):]:
            full = SITE + "/…" + tail
        d.text((lx + 34, acy), full[:len(SITE)], font=uf, fill=TXT, anchor="lm")
        d.text((lx + 34 + uf.getlength(full[:len(SITE)]), acy), full[len(SITE):], font=uf, fill=DIM, anchor="lm")
        # page: site header (ours), then the hero picture slot, then the headline under it
        d.text((FR[0] + 30, PAGE_Y + 30), "THE PHOTO ARCHIVE", font=F(22, "Black"), fill=ACCENT, anchor="lm")
        for k in range(3):                                                                  # menu icon
            d.rectangle([FR[2] - 62, PAGE_Y + 20 + k * 9, FR[2] - 34, PAGE_Y + 23 + k * 9], fill=(90, 84, 76))
        d.text((FR[2] - 80, PAGE_Y + 30), f"No. {i + 1}", font=F(20, "Medium"), fill=(120, 112, 100), anchor="rm")
        fact = ctx.fact_of(link, full=revealed)
        hf = F(24, "Bold")
        d.text((WX, 1428), clip(fact, hf, credit_left(i) - WX - 24), font=hf, fill=(40, 34, 30), anchor="ls")
        _cache[key] = im
        return im

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        trev = seg.get("rv", ctx.REVEAL) * seg["speech"] + ctx.THROW     # Buh lands on B
        base = page(i, local >= trev).copy()
        d = ImageDraw.Draw(base)
        # the page load bar runs under the address row whenever a new page (link) opens, and briefly at the reveal
        for t0_, dur in ((0.0, 0.55), (trev, 0.35)):
            u = (local - t0_) / dur
            if 0 <= u < 1.15:
                x1 = FR[0] + 6 + (FR[2] - FR[0] - 12) * min(1.0, ctx.ease(u))
                a = 255 if u < 1 else int(255 * (1.15 - u) / 0.15)
                bar = Image.new("RGBA", (int(x1 - FR[0] - 6) + 1, 5), (80, 150, 255, a))
                base.alpha_composite(bar, (FR[0] + 6, PAGE_Y))

        def mount(b, X0, Y0, w, h):                     # a printed photo on the archive page: thin border + shadow
            dr = ImageDraw.Draw(b)
            dr.rectangle([X0 - 2, Y0 + 6, X0 + w + 8, Y0 + h + 10], fill=(214, 208, 196))
            dr.rectangle([X0 - 6, Y0 - 6, X0 + w + 5, Y0 + h + 5], fill=(255, 255, 255))
        c = ctx.draw_core(base, seg, tt, WX, WY, SC, frame=mount)
        return ctx.core_finish(base, seg, tt, c)

    return frame
