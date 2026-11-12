"""Encyclopedia skin around the show's core: each link is a fictional encyclopedia article ("Openpedia") on A, and its
infobox photo IS the core window (camera, Buh, red string, name tags). Beside it the infobox lists the date and who
is pictured: B is "???" until Buh lands, then becomes a blue link. The image caption under it is the link's fact,
cut before B's name until the reveal."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

NAME = "wiki"

SC = 0.85                              # core window scale (748 x 807)
WX, WY = 60, 574                       # core window top-left
PANEL = (44, 512, 1038, 1446)          # infobox panel
COLX, COLW = 826, 200                  # infobox data column (kept above mid-height: the feed's buttons sit below)
BLUE = (51, 102, 204)
INKT = (32, 33, 34)
GREY = (84, 89, 93)
RULE = (162, 169, 177)
SITE = "Openpedia"


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
        d.text((60, 492), f"From {SITE}, the free encyclopedia", font=F(21, "Medium"), fill=GREY, anchor="ls")
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
        # image caption (the fact; B only after the reveal), left of the photo credit
        fact = ctx.fact_of(link, full=revealed)
        cf = F(22, "Medium")
        lines = wrap(fact, cf, credit_left(i) - 60 - 24, 1)
        for k, ln in enumerate(lines):
            d.text((60, 1416 + 30 * k), ln, font=cf, fill=GREY, anchor="ls")
        _cache[key] = im
        return im

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
        return ctx.core_finish(base, seg, tt, c)

    return frame
