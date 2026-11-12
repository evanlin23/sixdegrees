"""Rabbit hole: a browser shows an encyclopedia article on A; the link's photo is a big infobox image and the narrated
line is a highlighted sentence that appears word by word in sync with the voice (it is the only caption: no pixel
captions on top). B's name only appears, as a blue link, at the reveal. Buh peeks out beside the infobox and jumps
at the reveal; the cursor glides to the link, a preview card with B's face pops, it clicks, the page loads and slides
over to B's article, which is the next link."""
from __future__ import annotations

import math
import re

from PIL import Image, ImageDraw, ImageFont

NAME = "wiki"

PAGE_Y0 = 360                     # top of the page viewport (below the browser chrome)
SCROLL = 150                      # how far the page scrolls to bring the highlighted sentence up
PW, PH = 841, 640                 # infobox photo
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

    _box = {}

    def cam(i):
        """(cx, cy, s, w, h): the infobox photo, cover-cropped (never view()'s blurred fill) and never upscaled
        past ~1.3x; a photo too small to cover PW x PH at 1.3x gets a smaller box instead of a blurred band."""
        if i in _box:
            return _box[i]
        cx, cy, sc = ctx.both_cam(i, PW, PH, 0.84)
        ih, iw = ctx.C.imgs[i].shape[:2]
        s = min(1.3, max(sc, PW / iw, PH / ih))
        w, h = int(min(PW, iw * s)), int(min(PH, ih * s))
        cx = min(max(cx, w / 2 / s), iw - w / 2 / s)
        cy = min(max(cy, h / 2 / s), ih - h / 2 / s)
        _box[i] = (cx, cy, s, w, h)
        return _box[i]

    def face_crop(i, role, sz):
        """A square crop around one face of photo i (inside the photo, <= 1.3x)."""
        fb = ctx.fbox(i, role)
        ih, iw = ctx.C.imgs[i].shape[:2]
        s = min(1.3, max(sz * 0.5 / max(1, fb[3]), sz / min(iw, ih)))
        fx_, fy_ = ctx.ctr(fb)
        fx_ = min(max(fx_, sz / 2 / s), iw - sz / 2 / s)
        fy_ = min(max(fy_ + fb[3] * 0.15, sz / 2 / s), ih - sz / 2 / s)
        im = ctx.view(i, fx_, fy_, s, sz, sz)
        m = Image.new("L", (sz, sz), 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, sz - 1, sz - 1], 14, fill=255)
        im.putalpha(m)
        return im

    def para(d, text, x, y, font, fill, maxw, lh, max_lines=3):
        for ln in wrap(text, font, maxw, max_lines):
            d.text((x, y), ln, font=font, fill=fill)
            y += lh
        return y

    _intro_top, _bslot = {}, {}

    def page(i):
        """Static article image for link i (without the narrated sentence), the sentence's word layout
        [(x, y, word, is_link, width)] in page coords, and the top of the sentence."""
        if i in _pages:
            return _pages[i]
        an, bn = ctx.link_names(i)
        link = ctx.spec["links"][i]
        im = Image.new("RGBA", (ctx.W, 2200), (255, 255, 255, 255))
        d = ImageDraw.Draw(im)
        tf = fit(label(an), 66, X1 - X0, "ExtraBold")
        d.text((X0, 22), label(an), font=tf, fill=INKT)
        d.text((X0, 112), "Rabbithole \u00b7 every page links to someone else", font=F(24), fill=GREY)
        d.line([X0, 152, X1, 152], fill=(200, 204, 209), width=2)
        # infobox: name band and a big photo
        bx0, bx1 = X0, X1
        d.rectangle([bx0, 168, bx1, 168 + 48], fill=(234, 236, 240))
        yr = ctx.year_of(i)
        d.text(((bx0 + bx1) / 2, 192), label(an) + (f"  \u00b7  {yr}" if yr else ""), font=F(30), fill=INKT, anchor="mm")
        ccx, ccy, cs, bw, bh = cam(i)
        ph = ctx.view(i, ccx, ccy, cs, bw, bh)
        px = int((bx0 + bx1 - bw) / 2)
        im.alpha_composite(ph, (px, 220))
        cy = 220 + bh + 2
        d.rectangle([bx0, 168, bx1, cy], outline=(162, 169, 177), width=2)
        # a short real paragraph before the sentence: how we got to this page (never names B)
        if i > 0:
            intro = ctx.before_b(ctx.spec["links"][i - 1]["line"], link)
        else:
            intro = f"{label(an)} is where this rabbit hole starts."
        _intro_top[i] = cy + 14
        y = para(d, intro, X0, cy + 22, F(28, "Regular"), (64, 66, 70), X1 - X0, 38, 2) + 14
        # the narrated sentence: laid out once (no reflow while it types), drawn per frame
        sf = F(38)
        line = link["line"]
        sp = b_span(line, bn)
        if sp is None:                                  # B not named in the line: add a "see also" link
            line = f"{line} See also: {label(bn)}."
            sp = b_span(line, bn)
        words = []
        for m in re.finditer(r"\S+", line):
            is_link = sp[0] <= m.start() < sp[1] or (m.start() < sp[0] < m.end())
            words.append((m.group(0), is_link))
        space = sf.getlength(" ")
        LH = 54
        lay, x, sent_top = [], X0, y
        for w, lk in words:
            wd = sf.getlength(w)
            if x > X0 and x + wd > X1:
                x, y = X0, y + LH
            lay.append((x, y, w, lk, wd))
            x += wd + space
        y += LH
        # the trail so far, in small print (real names, all before B)
        trail = " \u203a ".join(label(ctx.link_names(j)[0]) for j in range(max(0, i - 3), i + 1))
        if i > 3:
            trail = "\u2026 \u203a " + trail
        d.text((X0, y + 26), "Trail", font=F(30, "ExtraBold"), fill=INKT)
        d.line([X0, y + 70, X1, y + 70], fill=(200, 204, 209), width=2)
        y = para(d, trail + " \u203a ?", X0, y + 84, F(26, "Regular"), BLUE, X1 - X0, 36, 3)
        # "In this photo": both faces from this link's photo (B's name is drawn per frame: ??? until the reveal)
        y += 34
        d.text((X0, y), "In this photo", font=F(30, "ExtraBold"), fill=INKT)
        d.line([X0, y + 44, X1, y + 44], fill=(200, 204, 209), width=2)
        y += 64
        cw, fs = (X1 - X0 - 24) // 2, 176
        bslot = None
        for k, role in enumerate(("a", "b")):
            x = X0 + k * (cw + 24)
            d.rounded_rectangle([x, y, x + cw, y + fs + 24], 16, fill=(248, 249, 250), outline=(200, 204, 209),
                                width=2)
            im.alpha_composite(face_crop(i, role, fs), (x + 12, y + 12))
            tx = x + fs + 28
            if role == "a":
                for k2, ln in enumerate(wrap(label(an), F(30), cw - fs - 40, 2)):
                    d.text((tx, y + 30 + k2 * 38), ln, font=F(30), fill=INKT)
                d.text((tx, y + fs - 20), "This page", font=F(22, "Medium"), fill=GREY)
            else:
                bslot = (tx, y + 30, cw - fs - 40)
                d.text((tx, y + fs - 20), "Linked page", font=F(22, "Medium"), fill=GREY)
        y += fs + 24 + 40
        # "See also": everyone earlier in the trail (real names, all before B)
        prev = [ctx.link_names(j)[0] for j in range(i)]
        if prev:
            d.text((X0, y), "See also", font=F(30, "ExtraBold"), fill=INKT)
            d.line([X0, y + 44, X1, y + 44], fill=(200, 204, 209), width=2)
            y += 64
            n_ = len(prev)
            cell = (X1 - X0) / max(5, n_)
            for k, nm in enumerate(prev[-5:]):
                x = X0 + k * cell + cell / 2
                th = ctx.sharp[nm].convert("RGBA").resize((128, 128), Image.BICUBIC)
                m = Image.new("L", th.size, 0)
                ImageDraw.Draw(m).ellipse([0, 0, 127, 127], fill=255)
                im.paste(th, (int(x - 64), int(y)), m)
                d.ellipse([x - 65, y - 1, x + 65, y + 129], outline=(200, 204, 209), width=2)
                for k2, ln in enumerate(wrap(label(nm), F(22), cell - 12, 2)):
                    d.text((x, y + 146 + k2 * 28), ln, font=F(22), fill=BLUE, anchor="ma")
            y += 220
        # categories footer
        yr = ctx.year_of(i)
        cats = [f"{yr} photographs" if yr else "Photographs", "Rabbit holes", f"Link {i + 1} of {ctx.n_links}"]
        x = X0
        for c in cats:
            wd = F(22, "Medium").getlength(c) + 32
            d.rounded_rectangle([x, y, x + wd, y + 42], 21, fill=(241, 243, 244))
            d.text((x + wd / 2, y + 21), c, font=F(22, "Medium"), fill=GREY, anchor="mm")
            x += wd + 12
        _bslot[i] = bslot
        _pages[i] = (im, lay, sent_top)
        return _pages[i]

    def n_spoken(seg, tt, n):
        """How many of the line's n words the voice has started by tt (music-only: paced over the speech)."""
        g = getattr(seg["cap"], "groups", None)
        if g:
            ws = [w for grp in g for w in grp][:n]
            if ws:
                return sum(1 for _, a, _ in ws if a <= tt + 0.04) * n / len(ws)
        return n * min(1.0, max(0.0, (tt - seg["t0"]) / max(0.5, seg["speech"] * 0.92)))

    def draw_sentence(dst, lay, k, top, show_link, hover, clicked):
        """The first k words of the sentence onto dst (page y offset -top); link words only when show_link."""
        d = ImageDraw.Draw(dst)
        sf = F(38)
        sp_ = sf.getlength(" ")
        vis = []
        for j, (x, y, w, lk, wd) in enumerate(lay):
            if j >= k or (lk and not show_link):
                break
            vis.append((x, y - top, w, lk, wd))
        for j, (x, y, w, lk, wd) in enumerate(vis):
            ext = sp_ if j + 1 < len(vis) and vis[j + 1][1] == y else 0
            d.rectangle([x - 6, y - 4, x + wd + ext + 6, y + 46], fill=HILITE)
        for x, y, w, lk, wd in vis:
            core = w.rstrip(".,;:!?") if lk else w
            d.text((x, y), core, font=sf, fill=BLUE if lk else INKT)
            if core != w:
                d.text((x + sf.getlength(core), y), w[len(core):], font=sf, fill=INKT)
            if lk and (hover or clicked):
                cw = sf.getlength(core)
                if clicked:
                    d.rounded_rectangle([x - 4, y - 2, x + cw + 4, y + 50], 6, outline=BLUE, width=2)
                d.line([x, y + 46, x + cw, y + 46], fill=BLUE, width=3)
        return vis

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

    def preview_card(i):
        _, bn = ctx.link_names(i)
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
            para(d, ctx.fact_of(ctx.spec["links"][i], full=True), 200, 104, F(22, "Regular"), GREY, cw - 220, 30, 3)
            _cards[bn] = im
        return _cards[bn]

    def viewport(base, i, scroll, dx, k=10 ** 6, show_link=True, hover=False, clicked=False):
        im, lay, _ = page(i)
        top = int(scroll)
        crop = im.crop((0, top, ctx.W, top + ctx.H - PAGE_Y0)).copy()
        draw_sentence(crop, lay, k, top, show_link, hover, clicked)
        bx_, by_, bw_ = _bslot[i]
        _, bn = ctx.link_names(i)
        if show_link:
            dd = ImageDraw.Draw(crop)
            for k2, ln in enumerate(wrap(label(bn), F(30), bw_, 2)):
                dd.text((bx_, by_ - top + k2 * 38), ln, font=F(30), fill=BLUE)
        else:
            ImageDraw.Draw(crop).text((bx_, by_ - top), "???", font=F(30), fill=GREY)
        base.alpha_composite(crop, (int(dx), PAGE_Y0))

    def scroll_at(local, trev, last=False):
        u = ctx.ease((local - 0.5) / max(0.6, trev - 1.3))
        return SCROLL * u + 14 * max(0.0, local - 0.5)          # eased scroll (the heading fully out), then a slow drift

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
        revealed = local >= trev
        _, lay, sent_top = page(i)
        n_words = len(lay)
        first_link = next((j for j, w in enumerate(lay) if w[3]), n_words)
        k = int(math.ceil(n_spoken(seg, tt, n_words) - 1e-6))
        if revealed:                                     # B's name lands (blue) at the reveal, even if said earlier
            k = max(k, first_link + sum(1 for w in lay if w[3]))
        else:
            k = min(k, first_link)
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
        hover = revealed
        if last:
            t_click = dur + 1
        viewport(base, i, scroll, ctx.W * (1 - slide), k=k, show_link=revealed, hover=hover, clicked=local > t_click)
        if slide < 1:
            ImageDraw.Draw(base).line([ctx.W * (1 - slide) - 1, PAGE_Y0, ctx.W * (1 - slide) - 1, ctx.H],
                                      fill=(160, 160, 160), width=3)
        # Buh peeks out from behind the infobox's right edge (upper half: clear of the feed's buttons)
        if last and revealed:
            expr = "happy"
        elif revealed:
            expr = "shock" if local < trev + 0.5 else "happy"
        else:
            expr = "smug"
        hop = 0.0
        if revealed and local < trev + 0.35:
            hop = 50 * math.sin(math.pi * (local - trev) / 0.35)
        elif last and revealed:
            hop = abs(math.sin(local * 8)) * 34
        peek = ctx.ease(local / 0.4) if i == 0 else 1.0
        bxp = X1 + 78 + 60 * (1 - peek) + ctx.W * (1 - slide)
        byp = 220 + cam(i)[4] * 0.62 - scroll + PAGE_Y0 + 6 * math.sin(local * 2.2) - hop
        ctx.paste_pin(base, ctx.pin_sprite(expr, 8 + 4 * math.sin(local * 1.6), 0, 0, scale=3), bxp, byp)
        base.alpha_composite(chrome(i), (0, 230))
        # loading bar under the address field once the link is clicked (and while this page arrives)
        d = ImageDraw.Draw(base)
        if local > t_click:
            u = min(1.0, (local - t_click) / max(0.3, dur - t_click))
            d.rectangle([0, PAGE_Y0 - 5, int(ctx.W * (0.15 + 0.8 * ctx.ease(u))), PAGE_Y0], fill=BLUE)
        elif slide < 1:
            d.rectangle([0, PAGE_Y0 - 5, int(ctx.W * (0.95 + 0.05 * slide)), PAGE_Y0], fill=BLUE)
        # where the link is (or will be) on screen now
        lk = [w for w in lay if w[3]] or lay[-1:]
        lx0, ly0 = lk[0][0], lk[0][1]
        lx1 = lk[0][0] + lk[0][4]
        tx, ty = (lx0 + lx1) / 2, ly0 + 26 - scroll + PAGE_Y0
        # preview card (B's face + name) pops above the sentence on hover, after the reveal
        if revealed and not last:
            pop = ctx.ease_out_back(min(1.0, (local - trev) / 0.22))
            card = preview_card(i)
            cx = min(max(tx, ctx.SAFE[0] + card.width / 2), ctx.SAFE[2] - card.width / 2)
            ctx.paste(base, card, cx, _intro_top[i] - scroll + PAGE_Y0 - card.height / 2 + 20 * (1 - pop),
                      scale=0.8 + 0.2 * pop, alpha=min(1.0, pop), nearest=False)
        # the cursor: idles below the sentence, glides to the link, clicks
        wx = 760 + 60 * math.sin(local * 1.7) + 24 * math.sin(local * 3.1)
        wy = sent_top - scroll + PAGE_Y0 + 230 + 40 * math.sin(local * 1.1 + 1)
        go0 = max(0.4, trev - 1.0)
        u = ctx.ease((local - go0) / max(0.3, trev - go0))
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
        if i == 0 and local < 0.3:                      # fade in from the dim hook (no hard cut to white)
            a = int(200 * (1 - ctx.ease(local / 0.3)))
            if a > 0:
                base.alpha_composite(Image.new("RGBA", (ctx.W, ctx.H), (16, 10, 24, a)))
        ctx.header(base, i)
        if last:                                         # CONNECTED! across the photo, below both faces
            fy = []
            cx_, cy_, sc_, _, ph_h = cam(i)
            for role in ("a", "b"):
                b_ = ctx.fbox(i, role)
                fy.append(220 + ph_h / 2 + (b_[1] + b_[3] - cy_) * sc_)
            y = min(max(max(fy) + 90, 300), 220 + ph_h - 60) - scroll + PAGE_Y0
            ctx.connected(base, seg, tt, y=int(y))
        return base

    return frame
