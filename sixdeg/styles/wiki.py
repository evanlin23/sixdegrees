"""Encyclopedia skin around the show's core: each link is an article on A in "Buhpedia" (the show's own
encyclopedia, a tiny Buh pin as its logo), and its infobox photo IS the core window (camera, Buh, red string, name
tags). Beside it the infobox lists the date and who is pictured: B is "???" until Buh lands, then becomes a blue link.
The image caption under it is the link's fact, cut before B's name until the reveal. Under the infobox data a small
donation-style "ad" (Support Buhpedia: Like & Subscribe; no "Like" on sensitive links).

Link to link: the next article opens under a load bar (the owner dropped the scroll/blur/click-a-link transition
as confusing). The core is still drawn on every frame (then blurred with the page); the next link's B never appears."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFilter, ImageFont

NAME = "wiki"

SC = 0.85                              # core window scale (748 x 807)
WX, WY = 60, 574                       # core window top-left
PANEL = (44, 512, 1038, 1446)          # infobox panel
COLX, COLW = 826, 200                  # infobox data column (kept above mid-height: the feed's buttons sit below)
BLUE = (51, 102, 204)
INKT = (32, 33, 34)
GREY = (84, 89, 93)
RULE = (162, 169, 177)
SITE = "Buhpedia"
TR = 0.5                               # link-to-link transition: the last TR s of a link (scroll, blur, click)
CLICK = 0.1                            # the cursor clicks this long before the cut
REG = (400, 1460)                      # the scrolling article viewport (y0, y1)
SCROLL = 680                           # px the article scrolls down in the transition
LINK_Y = 366                           # the "See also" link's y inside the article body (below the infobox)


def make(ctx):
    _fonts, _cache = {}, {}

    def F(size, weight="Bold"):
        k = (size, weight)
        if k not in _fonts:
            _fonts[k] = ImageFont.truetype(str(ctx.FONTS / f"Inter-{weight}.otf"), size)
        return _fonts[k]

    def label(n):
        return ctx.C.labels.get(n, n)

    def wrap(text, font, maxw, max_lines=2):
        lines, cur = [], ""
        for w in text.split():
            t = f"{cur} {w}".strip()
            if cur and font.getlength(t) > maxw:
                lines.append(cur)
                cur = w
            else:
                cur = t
        if cur:
            lines.append(cur)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1].rstrip(",;:…") + "…"
        return lines

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
        """The article page for link i (everything but the core window)."""
        key = (i, revealed)
        if key in _cache:
            return _cache[key]
        an, bn = ctx.link_names(i)
        link = ctx.spec["links"][i]
        im = Image.new("RGBA", (ctx.W, ctx.H), (255, 255, 255, 255))
        d = ImageDraw.Draw(im)
        # article title + tabs + tagline
        title = label(an)
        tf = F(46, "Bold")
        while tf.size > 30 and tf.getlength(title) > 640:
            tf = F(tf.size - 2, "Bold")
        d.text((60, 450), title, font=tf, fill=INKT, anchor="ls")
        x = 1030
        for k, tab in enumerate(reversed(("Article", "Talk", "Edit"))):
            f = F(22, "Bold" if tab == "Article" else "Medium")
            x -= f.getlength(tab)
            d.text((x, 446), tab, font=f, fill=INKT if tab == "Article" else BLUE, anchor="ls")
            if tab == "Article":
                d.rectangle([x, 456, x + f.getlength(tab), 459], fill=INKT)
            x -= 26
        d.line([44, 466, 1038, 466], fill=RULE, width=2)
        logo = ctx.pin_sprite("smug", scale=1)                       # Buhpedia's logo: a tiny Buh pin
        im.alpha_composite(logo, (58, 467))
        d.text((58 + logo.width + 8, 492), f"From {SITE}, the free encyclopedia", font=F(21, "Medium"), fill=GREY,
               anchor="ls")
        # infobox panel + header strip
        d.rectangle(PANEL, fill=(248, 249, 250), outline=RULE, width=2)
        d.rectangle([PANEL[0] + 2, PANEL[1] + 2, PANEL[2] - 2, 562], fill=(222, 230, 245))
        yr = ctx.year_of(i)
        d.text(((PANEL[0] + PANEL[2]) / 2, 538), f"Photograph, {yr}" if yr else "Photograph", font=F(26, "Bold"),
               fill=INKT, anchor="mm")
        # data rows beside the photo
        y = 600
        rows = [("Date", str(yr) if yr else ""), ("Pictured", label(an)),
                ("", label(bn) if revealed else "???"), ("Chain", f"link {i + 1}")]
        for k, (lab, val) in enumerate(rows):
            if not val:                                  # no year known: no Date row
                continue
            if lab:
                d.text((COLX, y), lab, font=F(20, "Bold"), fill=INKT, anchor="ls")
                y += 32
            f = F(24, "Medium")
            lines = wrap(val, f, COLW, 2)
            blue = (k in (1, 2) and val != "???")
            for ln in lines:
                d.text((COLX, y), ln, font=f, fill=BLUE if blue else GREY, anchor="ls")
                if blue and k == 2:
                    d.line([COLX, y + 4, COLX + f.getlength(ln), y + 4], fill=BLUE, width=2)
                y += 32
            if k != 1:
                y += 10
                d.line([COLX, y - 12, PANEL[2] - 12, y - 12], fill=(226, 228, 232), width=1)
                y += 14
        # a small donation-style ad under the data rows (clear of the feed's buttons as far as the column allows)
        soft = ctx.sensitive(an) or ctx.sensitive(bn)
        ay = max(y + 4, 870)
        d.rectangle([COLX - 8, ay, PANEL[2] - 10, ay + 146], fill=(255, 248, 222), outline=(234, 206, 120), width=2)
        d.text((COLX + 2, ay + 22), "Support Buhpedia", font=F(19, "Bold"), fill=INKT, anchor="lm")
        for k, ln in enumerate(("Subscribe for more", "connections") if soft else ("Like & Subscribe for", "more connections")):
            d.text((COLX + 2, ay + 50 + 22 * k), ln, font=F(17, "Medium"), fill=GREY, anchor="lm")
        d.rounded_rectangle([COLX + 2, ay + 96, PANEL[2] - 22, ay + 132], 6, fill=BLUE)
        d.text(((COLX + PANEL[2] - 20) / 2, ay + 114), "Subscribe", font=F(19, "Bold"), fill=(255, 255, 255), anchor="mm")
        # image caption (the fact; B only after the reveal), left of the photo credit
        fact = ctx.fact_of(link, full=revealed)
        cf = F(22, "Medium")
        lines = wrap(fact, cf, credit_left(i) - 60 - 24, 1)
        for k, ln in enumerate(lines):
            d.text((60, 1416 + 30 * k), ln, font=cf, fill=GREY, anchor="ls")
        _cache[key] = im
        return im

    def body(i):
        """The article below the infobox (seen only while it scrolls by, blurred): text lines, then "See also"."""
        if ("body", i) in _cache:
            return _cache[("body", i)]
        h = SCROLL + 80
        im = Image.new("RGBA", (ctx.W, h), (255, 255, 255, 255))
        d = ImageDraw.Draw(im)
        y = 40
        for sec, n in (("Life", 4), ("See also", 0)):
            d.text((60, y + 30), sec, font=F(36, "Bold"), fill=INKT, anchor="ls")
            d.line([44, y + 44, 1038, y + 44], fill=RULE, width=2)
            y += 74
            for k in range(n):
                w = 900 - (k * 137) % 260 if k < n - 1 else 520
                d.rounded_rectangle([60, y, 60 + w, y + 14], 6, fill=(200, 203, 208))
                if k % 2 == 0:
                    d.rounded_rectangle([60 + w // 3, y, 60 + w // 3 + 150, y + 14], 6, fill=(150, 178, 230))
                y += 34
            y += 26
        _cache[("body", i)] = im
        return im

    def cursor_im():
        if "cur" not in _cache:
            im = Image.new("RGBA", (40, 56), (0, 0, 0, 0))
            ImageDraw.Draw(im).polygon([(3, 2), (3, 42), (13, 33), (21, 51), (29, 47), (21, 30), (35, 30)],
                                       fill=(255, 255, 255), outline=(0, 0, 0), width=3)
            _cache["cur"] = im
        return _cache["cur"]

    def transition(base, i, bn, u, clicked):
        """Scroll the article viewport down fast, blur it all, then draw B's link sharp on top (and the cursor)."""
        y0, y1 = REG
        vh = y1 - y0
        S = int(SCROLL * ctx.ease(min(1.0, u / 0.55)))
        view = base.crop((0, y0 + S, ctx.W, y1)) if S < vh else None
        out = Image.new("RGBA", (ctx.W, vh), (255, 255, 255, 255))
        if view is not None:
            out.paste(view, (0, 0))
        bd = body(i)
        top = PANEL[3] + 14 - y0 - S                                    # the article body follows the infobox
        out.paste(bd, (0, top))
        r = 9 * min(1.0, u / 0.2)
        if r > 0.5:                                                     # cheap blur: 1/4 size, blur, back up
            sm = out.resize((ctx.W // 4, vh // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(r / 4))
            out = sm.resize((ctx.W, vh), Image.BILINEAR)
            out.alpha_composite(Image.new("RGBA", out.size, (255, 255, 255, int(70 * min(1.0, u / 0.2)))))
        base.paste(out, (0, y0))
        d = ImageDraw.Draw(base)
        # the one sharp thing: B's article link (B is revealed already; the next link's A)
        ly = y0 + top + LINK_Y
        if ly + 36 > ctx.CAP_Y - 60:                                    # still low: keep clear of the captions
            return
        txt = label(bn)
        f = F(44, "Medium")
        while f.size > 26 and f.getlength(txt) > 880:
            f = F(f.size - 2, "Medium")
        x = 96
        wl = f.getlength(txt)
        glow = min(1.0, u / 0.25)
        d.rounded_rectangle([x - 18, ly - 36, x + wl + 18, ly + 36], 12,
                            fill=(255, 245, 190) if glow >= 1 else tuple(int(255 + (c - 255) * glow) for c in (255, 245, 190)),
                            outline=(240, 200, 60) if glow > 0.5 else None, width=3)
        d.ellipse([60, ly - 7, 74, ly + 7], fill=INKT)
        col = (110, 60, 170) if clicked else BLUE                       # visited purple once clicked
        d.text((x, ly), txt, font=f, fill=col, anchor="lm")
        d.line([x, ly + 26, x + wl, ly + 26], fill=col, width=4)
        # the cursor glides in from the lower right and clicks the link
        cu = ctx.ease(min(1.0, max(0.0, (u - 0.25) / 0.45)))
        if u > 0.25:
            cx0, cy0 = 980, 1240
            cx1, cy1 = x + min(wl * 0.6, 220), ly + 6
            cx, cy = cx0 + (cx1 - cx0) * cu, cy0 + (cy1 - cy0) * cu
            press = 0.85 if clicked and u < 1 - CLICK / TR + 0.14 else 1.0
            if clicked:
                rr = 14 + 120 * (u - (1 - CLICK / TR)) * TR
                d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(90, 140, 230), width=4)
            ci = cursor_im()
            ctx.paste(base, ci, cx + ci.width * press / 2 - 3, cy + ci.height * press / 2 - 2, scale=press,
                      alpha=min(1.0, (u - 0.25) / 0.08), nearest=False)

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        trev = seg.get("rv", ctx.REVEAL) * seg["speech"] + ctx.THROW     # Buh lands on B
        base = page(i, local >= trev).copy()

        def mount(b, X0, Y0, w, h):                    # the infobox image's thin border
            ImageDraw.Draw(b).rectangle([X0 - 3, Y0 - 3, X0 + w + 2, Y0 + h + 2], fill=(200, 204, 210))
        c = ctx.draw_core(base, seg, tt, WX, WY, SC, frame=mount)
        # owner (Review Room): "maybe don't need the link, it's a bit confusing" -> no scroll/blur/click transition;
        # the next article just loads under the blue load bar. (transition() kept, unused, in case it comes back.)
        if i > 0 and local < 0.4:                                       # the new article loads
            ImageDraw.Draw(base).rectangle([44, 402, 44 + int(994 * ctx.ease(min(1.0, local / 0.35))), 406],
                                           fill=BLUE)
        return ctx.core_finish(base, seg, tt, c)

    return frame
