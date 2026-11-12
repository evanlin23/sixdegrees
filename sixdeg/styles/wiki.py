"""Wikipedia rabbit hole: a browser shows an encyclopedia article on A; the link's photo is the infobox image and the
narrated line is a highlighted sentence with B as a blue link. The cursor hovers it (a preview card with B's face
pops), clicks, the page loads and slides over to B's article, which is the next link."""
from __future__ import annotations

import math
import re

from PIL import Image, ImageDraw, ImageFont

NAME = "wiki"

PAGE_Y0 = 360                     # top of the page viewport (below the browser chrome)
SCROLL = 150                      # how far the page scrolls to bring the highlighted sentence up
PW, PH = 830, 470                 # infobox photo
X0, X1 = 70, 915                  # article text column
BLUE = (51, 102, 204)
INKT = (32, 33, 34)
GREY = (114, 119, 125)
BAR = (216, 219, 223)
HILITE = (255, 236, 140)
SITE = "rabbithole.wiki"


def make(ctx):
    _fonts, _pages, _chrome = {}, {}, {}

    def F(size, weight="Bold"):
        k = (size, weight)
        if k not in _fonts:
            _fonts[k] = ImageFont.truetype(str(ctx.FONTS / f"Inter-{weight}.otf"), size)
        return _fonts[k]

    def label(n):
        return ctx.C.labels.get(n, n)

    def fit(text, size, maxw, weight="Bold", low=26):
        while size > low and F(size, weight).getlength(text) > maxw:
            size -= 2
        return F(size, weight)

    def wrap(text, font, maxw, max_lines=3):
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
            lines[-1] = lines[-1].rstrip(",;:") + "…"
        return lines

    def bars(d, x, y, widths, gap=40, h=16):
        for k, wd in enumerate(widths):
            d.rounded_rectangle([x, y + k * gap, x + wd, y + k * gap + h], 8, fill=BAR)
        return y + len(widths) * gap

    def b_span(line, b):
        """(start, end) of B's mention in the line: label, full name, or surname; None if absent."""
        for cand in (label(b), b, b.split()[-1], label(b).split()[-1]):
            m = re.search(r"\b" + re.escape(cand) + r"\b", line)
            if m:
                return m.span()
        return None

    def page(i):
        """Static article image for link i, plus the link's word boxes (page coords)."""
        if i in _pages:
            return _pages[i]
        an, bn = ctx.link_names(i)
        link = ctx.spec["links"][i]
        im = Image.new("RGBA", (ctx.W, 1900), (255, 255, 255, 255))
        d = ImageDraw.Draw(im)
        tf = fit(label(an), 66, X1 - X0, "ExtraBold")
        d.text((X0, 22), label(an), font=tf, fill=INKT)
        d.text((X0, 112), f"From {SITE[:-5].capitalize()}, the free encyclopedia", font=F(24), fill=GREY)
        d.line([X0, 152, X1, 152], fill=(200, 204, 209), width=2)
        # infobox: name band, photo, caption
        bx0, bx1 = X0, X1
        d.rectangle([bx0, 168, bx1, 168 + 48], fill=(234, 236, 240))
        d.text(((bx0 + bx1) / 2, 192), label(an), font=F(30), fill=INKT, anchor="mm")
        ph = ctx.view(i, *ctx.both_cam(i, PW, PH, 0.8), PW, PH)
        px = int((bx0 + bx1 - PW) / 2)
        im.alpha_composite(ph, (px, 224))
        yr = ctx.year_of(i)
        cap = wrap(ctx.fact_of(link), F(24), PW - 30, 2)
        cy = 224 + PH + 12
        for ln in cap:
            d.text(((bx0 + bx1) / 2, cy + 14), ln, font=F(24), fill=GREY, anchor="mm")
            cy += 32
        d.rectangle([bx0, 168, bx1, cy + 10], outline=(162, 169, 177), width=2)
        # body: placeholder lines, the highlighted sentence with B's link, more lines
        y = bars(d, X0, cy + 40, [845, 790])
        sf = F(36)
        line = link["line"]
        sp = b_span(line, bn)
        if sp is None:                                  # B not named in the line: add a "see also" link
            line = f"{line} See also: {label(bn)}."
            sp = b_span(line, bn)
        words, pos = [], 0
        for m in re.finditer(r"\S+", line):
            is_link = sp[0] <= m.start() < sp[1] or (m.start() < sp[0] < m.end())
            words.append((m.group(0), is_link))
        lines, cur, curw = [], [], 0.0
        space = sf.getlength(" ")
        for w, lk in words:
            wd = sf.getlength(w)
            if cur and curw + space + wd > X1 - X0:
                lines.append(cur)
                cur, curw = [], 0.0
            cur.append((w, lk, wd))
            curw += (space if len(cur) > 1 else 0) + wd
        if cur:
            lines.append(cur)
        boxes = []
        y += 6
        sent_top = y
        LH = 52
        for ln in lines:
            x = X0
            lw = sum(wd for _, _, wd in ln) + space * (len(ln) - 1)
            d.rectangle([x - 6, y - 4, x + lw + 6, y + LH - 8], fill=HILITE)
            for k, (w, lk, wd) in enumerate(ln):
                # keep trailing punctuation black, only the name is blue
                core = w.rstrip(".,;:!?") if lk else w
                d.text((x, y), core, font=sf, fill=BLUE if lk else INKT)
                if lk:
                    boxes.append((x, y, x + sf.getlength(core), y + 42))
                    if core != w:
                        d.text((x + sf.getlength(core), y), w[len(core):], font=sf, fill=INKT)
                x += wd + space
            y += LH
        y = bars(d, X0, y + 16, [820, 860, 610])
        d.text((X0, y + 30), "Legacy", font=F(40), fill=INKT)
        d.line([X0, y + 86, X1, y + 86], fill=(200, 204, 209), width=2)
        bars(d, X0, y + 110, [850, 800, 845, 700, 830, 860, 520, 840, 790, 845, 600])
        # merge the link's word boxes per line
        merged = []
        for b in boxes:
            if merged and abs(merged[-1][1] - b[1]) < 2:
                merged[-1] = (merged[-1][0], merged[-1][1], b[2], b[3])
            else:
                merged.append(b)
        _pages[i] = (im, merged, sent_top)
        return _pages[i]

    def chrome(i):
        if i in _chrome:
            return _chrome[i]
        an, _ = ctx.link_names(i)
        im = Image.new("RGBA", (ctx.W, PAGE_Y0 - 230), (222, 225, 230, 255))
        d = ImageDraw.Draw(im)
        # tabs: an inactive one behind, the active article tab
        d.rounded_rectangle([430, 12, 700, 60], 14, fill=(206, 210, 216))
        d.text((470, 36), "Search results", font=F(20), fill=(90, 94, 100), anchor="lm")
        d.rounded_rectangle([40, 8, 440, 70], 16, fill=(255, 255, 255))
        d.ellipse([60, 22, 88, 50], fill=(40, 40, 46))
        d.text((74, 36), "R", font=F(18, "Black"), fill=(255, 255, 255), anchor="mm")
        t = f"{label(an)} – Rabbithole"
        tf = F(22)
        while tf.getlength(t) > 290 and len(t) > 8:
            t = t[:-2].rstrip(" –") + "…"
            t = t.replace("……", "…")
        d.text((100, 37), t, font=tf, fill=INKT, anchor="lm")
        d.text((418, 37), "×", font=F(24), fill=GREY, anchor="mm")
        # toolbar: back / forward / reload, then the address field
        d.rectangle([0, 62, ctx.W, PAGE_Y0 - 230], fill=(255, 255, 255))
        cy = 62 + (PAGE_Y0 - 230 - 62) / 2
        for k, ch in enumerate(("←", "→", "↻")):
            d.text((60 + k * 46, cy), ch, font=F(32), fill=(95, 99, 104) if k != 1 else (190, 192, 196), anchor="mm")
        d.rounded_rectangle([200, cy - 26, 920, cy + 26], 26, fill=(241, 243, 244))
        d.rounded_rectangle([222, cy - 6, 236, cy + 9], 3, fill=(95, 99, 104))       # padlock
        d.arc([224, cy - 15, 234, cy - 1], 180, 360, fill=(95, 99, 104), width=3)
        url = f"{SITE}/wiki/{an.replace(' ', '_')}"
        uf = F(24)
        while uf.getlength(url) > 640 and len(url) > 12:
            url = url[:-2]
        d.text((252, cy), SITE, font=uf, fill=INKT, anchor="lm")
        d.text((252 + uf.getlength(SITE), cy), url[len(SITE):], font=uf, fill=GREY, anchor="lm")
        d.line([0, im.height - 1, ctx.W, im.height - 1], fill=(200, 204, 209), width=2)
        _chrome[i] = im
        return im

    _cur = {}

    def cursor(press=0.0):
        k = round(press, 1)
        if k not in _cur:
            s = 3.2 * (1 - 0.15 * k)
            pts = [(0, 0), (0, 17), (4, 13), (7, 20), (10, 19), (7, 12), (12, 12)]
            im = Image.new("RGBA", (int(16 * s) + 8, int(24 * s) + 8), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.polygon([(x * s + 5, y * s + 5) for x, y in pts], fill=(0, 0, 0, 70))
            d.polygon([(x * s + 3, y * s + 3) for x, y in pts], fill=(255, 255, 255), outline=(0, 0, 0))
            d.line([(x * s + 3, y * s + 3) for x, y in pts + pts[:1]], fill=(0, 0, 0), width=3)
            _cur[k] = im
        return _cur[k]

    _cards = {}

    def preview_card(bn):
        if bn not in _cards:
            cw, ch = 640, 200
            im = Image.new("RGBA", (cw + 20, ch + 20), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.rounded_rectangle([8, 10, cw + 12, ch + 14], 18, fill=(0, 0, 0, 60))
            d.rounded_rectangle([2, 2, cw + 2, ch + 2], 18, fill=(255, 255, 255), outline=(200, 204, 209), width=2)
            face = ctx.sharp[bn].convert("RGBA").resize((156, 156), Image.BICUBIC)
            m = Image.new("L", face.size, 0)
            ImageDraw.Draw(m).rounded_rectangle([0, 0, 155, 155], 14, fill=255)
            im.paste(face, (24, 24), m)
            nf = fit(label(bn), 44, cw - 220, "ExtraBold")
            d.text((200, 40), label(bn), font=nf, fill=INKT)
            bars(d, 200, 112, [400, 340, 250], gap=30, h=13)
            _cards[bn] = im
        return _cards[bn]

    def viewport(base, i, scroll, dx, hover=False, clicked=False):
        im, boxes, _ = page(i)
        top = int(scroll)
        crop = im.crop((0, top, ctx.W, top + ctx.H - PAGE_Y0))
        if hover or clicked:
            crop = crop.copy()
            d = ImageDraw.Draw(crop)
            for x0, y0, x1, y1 in boxes:
                if clicked:
                    d.rounded_rectangle([x0 - 4, y0 - top - 2, x1 + 4, y1 - top + 6], 6, outline=BLUE, width=2)
                d.line([x0, y1 - top + 2, x1, y1 - top + 2], fill=BLUE, width=3)
        base.alpha_composite(crop, (int(dx), PAGE_Y0))

    def scroll_at(local, trev, last=False):
        u = ctx.ease((local - 0.5) / max(0.6, trev - 1.3))
        return (SCROLL - 110 if last else SCROLL) * u + 14 * max(0.0, local - 0.5)          # eased scroll, then a slow reading drift

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        dur = seg["t1"] - seg["t0"]
        sp, R = seg["speech"], seg.get("rv", ctx.REVEAL)
        trev = R * sp + ctx.THROW
        t_click = min(max(trev + 0.55, dur - 1.1), dur - 0.35)
        an, bn = ctx.link_names(i)
        base = Image.new("RGBA", (ctx.W, ctx.H), (255, 255, 255, 255))
        last = i == ctx.n_links - 1                     # the chain ends here: no click, CONNECTED! lands instead
        scroll = scroll_at(local, trev, last)
        # page slide: the previous article leaves to the left as this one arrives from the right
        slide = 1.0
        if i > 0 and not ctx.C.fj["links"][i - 1].get("lineage"):
            slide = ctx.ease(local / 0.45)
            if slide < 1:
                ps = next((s for s in ctx.segs if s.get("kind") == "link" and s.get("i") == i - 1), None)
                pl = (ps["t1"] - ps["t0"]) if ps else 4.0
                ptrev = (ps["rv"] * ps["speech"] + ctx.THROW) if ps else 2.0
                viewport(base, i - 1, scroll_at(pl, ptrev), -ctx.W * 0.35 * slide, clicked=True)
                base.alpha_composite(Image.new("RGBA", (ctx.W, ctx.H - PAGE_Y0), (0, 0, 0, int(90 * slide))),
                                     (0, PAGE_Y0))
        hover = local > trev - 0.25
        if last:
            t_click = dur + 1
        viewport(base, i, scroll, ctx.W * (1 - slide), hover=hover, clicked=local > t_click)
        if slide < 1:
            ImageDraw.Draw(base).line([ctx.W * (1 - slide) - 1, PAGE_Y0, ctx.W * (1 - slide) - 1, ctx.H],
                                      fill=(160, 160, 160), width=3)
        base.alpha_composite(chrome(i), (0, 230))
        # loading bar under the address field once the link is clicked (and while this page arrives)
        d = ImageDraw.Draw(base)
        if local > t_click:
            u = min(1.0, (local - t_click) / max(0.3, dur - t_click))
            d.rectangle([0, PAGE_Y0 - 5, int(ctx.W * (0.15 + 0.8 * ctx.ease(u))), PAGE_Y0], fill=BLUE)
        elif slide < 1:
            d.rectangle([0, PAGE_Y0 - 5, int(ctx.W * (0.95 + 0.05 * slide)), PAGE_Y0], fill=BLUE)
        # where the link is on screen now
        _, boxes, sent_top = page(i)
        lx0, ly0, lx1, ly1 = boxes[0]
        tx, ty = (lx0 + lx1) / 2, (ly0 + ly1) / 2 - scroll + PAGE_Y0 + 6
        # preview card (B's face + name) pops above the link on hover
        if local > trev and not last:
            pop = ctx.ease_out_back(min(1.0, (local - trev) / 0.22))
            card = preview_card(bn)
            cx = min(max(tx, ctx.SAFE[0] + card.width / 2), ctx.SAFE[2] - card.width / 2)
            ctx.paste(base, card, cx, sent_top - scroll + PAGE_Y0 - card.height / 2 - 12 + 20 * (1 - pop),
                      scale=0.8 + 0.2 * pop, alpha=min(1.0, pop), nearest=False)
        # the cursor: idles and drifts, glides to the link, clicks
        wx = 760 + 70 * math.sin(local * 1.7) + 30 * math.sin(local * 3.1)
        wy = 1060 + 90 * math.sin(local * 1.1 + 1) - 40 * math.cos(local * 2.3)
        go0 = max(0.4, trev - 1.25)
        u = ctx.ease((local - go0) / max(0.3, trev - 0.2 - go0))
        mx, my = wx + (tx + 6 - wx) * u, wy + (ty - wy) * u
        if u >= 1:
            mx += 3 * math.sin(local * 5)
            my += 2 * math.cos(local * 4)
        press = max(0.0, 1 - abs(local - t_click) / 0.1)
        if local > t_click:
            r = 12 + 140 * (local - t_click)
            a = max(0, int(150 * (1 - (local - t_click) / 0.45)))
            if a:
                ring = Image.new("RGBA", (int(2 * r + 8),) * 2, (0, 0, 0, 0))
                ImageDraw.Draw(ring).ellipse([4, 4, 2 * r + 4, 2 * r + 4], outline=BLUE + (a,), width=5)
                ctx.paste(base, ring, mx, my)
        if slide >= 1 or i == 0:
            base.alpha_composite(cursor(press), (int(mx - 3), int(my - 3)))
        ctx.header(base, i)
        ctx.connected(base, seg, tt)
        seg["cap"].draw(base, tt, y=1400)
        return base

    return frame
