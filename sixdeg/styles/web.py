"""Web browser skin around the show's core: a dark browser window (tab strip, address bar, a load bar on every new
link) shows a fictional photo-archive page whose hero picture IS the core window (camera, Buh, red string, name
tags). Beside it a display "ad" for the channel itself (Buh's avatar, Like & Subscribe, a SUBSCRIBE button; no
"Like" on sensitive links). The tab strip tells the chain as browsing: the previous page's tab sits to the left, the
active tab names A until Buh lands on B, then B is appended; a little after the landing B's page opens as a new
background tab, and in the last moments of the link a cursor clicks it; the next link opens on that tab (the strip
slides left, the favicon spins, the load bar runs). The cursor stays in the browser chrome (never over the photo) and
under Buh. The address is a playful hidden message on the show's own domain (buh.connects/subscribe-if-you-read-
this ...), different per link and never naming B. The headline under the photo is the link's fact, cut before B's
name until the reveal. Generic chrome: no real browser's or site's brand."""
from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

from ..common import ROOT

NAME = "web"

SC = 0.85                              # core window scale (748 x 807)
FR = (34, 412, 1046, 1448)             # browser window
WX, WY = FR[0] + 22, 578               # core window top-left (left of the page: the ad column is on the right)
TABS = (FR[1], FR[1] + 56)             # tab strip
TAB0 = FR[0] + 126                     # first tab x
TW = (220, 330, 260)                   # tab widths: previous page, active, new background tab (then "+")
ADDR = (FR[1] + 62, FR[1] + 114)       # address bar row
PAGE_Y = FR[1] + 122                   # page viewport top
AD = (826, 600, 1030, 940)             # sidebar ad (above mid-height: the feed's buttons sit below)
CLICK = 0.12                           # the cursor clicks the new tab this long before the link ends
CHROME = (36, 38, 46)
TAB_ON = (58, 61, 72)
FIELD = (24, 26, 32)
TXT = (232, 234, 240)
DIM = (150, 154, 166)
PAGE = (244, 241, 234)                 # warm archive paper
ACCENT = (190, 60, 50)
SITE = "buh.connects"
AVATAR = ROOT / "branding" / "avatar-purple.png"
PATHS = [                              # the address is a little hidden message, one per link (never B's name)
    "/subscribe-if-you-read-this", "/who-is-next/comment-below", "/the-pin-knows", "/buh-was-here",
    "/hi-mom/im-on-a-short", "/you-found-the-secret-url", "/six-degrees-or-less", "/pause-more-often",
    "/tell-a-friend-about-buh", "/the-red-string-never-lies",
]


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
        return WX + 748 - f.getlength(txt) - 4

    def avatar(n):
        try:
            av = Image.open(AVATAR).convert("RGBA")
            s_ = av.width
            av = av.crop((int(s_ * 0.16), int(s_ * 0.08), int(s_ * 0.84), int(s_ * 0.76))).resize((n, n), Image.LANCZOS)
        except OSError:                                  # no branding file: Buh's sprite on a purple disc
            av = Image.new("RGBA", (n, n), (92, 60, 220, 255))
            pin = ctx.pin_sprite("happy", scale=max(1, n // 40))
            av.alpha_composite(pin, ((n - pin.width) // 2, (n - pin.height) // 2))
        m = Image.new("L", (n, n), 0)
        ImageDraw.Draw(m).ellipse([0, 0, n - 1, n - 1], fill=255)
        av.putalpha(m)
        return av

    def ad_im(soft):
        """A display ad for the channel itself. soft: a sensitive link, so no "Like" (just Subscribe)."""
        if ("ad", soft) in _cache:
            return _cache[("ad", soft)]
        w, h = AD[2] - AD[0], AD[3] - AD[1]
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, w - 1, h - 1], fill=(40, 30, 70), outline=(214, 208, 196), width=2)
        for k in range(0, h, 18):                                  # a little diagonal sheen
            d.line([(0, k + 60), (w, k)], fill=(48, 36, 84), width=6)
        d.rectangle([w - 34, 0, w - 1, 20], fill=(214, 208, 196))
        d.text((w - 17, 10), "AD", font=F(14, "Bold"), fill=(90, 84, 76), anchor="mm")
        av = avatar(96)
        d.ellipse([w / 2 - 52, 22, w / 2 + 52, 126], fill=(255, 255, 255))
        im.alpha_composite(av, (int(w / 2 - 48), 26))
        lines = ["Subscribe to", "Buh Connects"] if soft else ["Like & Subscribe", "to Buh Connects"]
        d.text((w / 2, 156), lines[0], font=F(22, "Bold"), fill=(255, 255, 255), anchor="mm")
        d.text((w / 2, 184), lines[1], font=F(22, "Bold"), fill=(255, 214, 90), anchor="mm")
        d.text((w / 2, 216), "more connections daily", font=F(17, "Medium"), fill=(200, 196, 230), anchor="mm")
        _cache[("ad", soft)] = im
        return im

    def sub_btn():
        if "btn" not in _cache:
            bw, bh = AD[2] - AD[0] - 36, 48
            im = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.rounded_rectangle([0, 0, bw - 1, bh - 1], 24, fill=(204, 0, 0))
            d.text((bw / 2, bh / 2), "SUBSCRIBE", font=F(22, "Bold"), fill=(255, 255, 255), anchor="mm")
            _cache["btn"] = im
        return _cache["btn"]

    def cursor_im():
        if "cur" not in _cache:
            im = Image.new("RGBA", (40, 56), (0, 0, 0, 0))
            ImageDraw.Draw(im).polygon([(3, 2), (3, 42), (13, 33), (21, 51), (29, 47), (21, 30), (35, 30)],
                                       fill=(255, 255, 255), outline=(0, 0, 0), width=3)
            _cache["cur"] = im
        return _cache["cur"]

    def title_of(i, revealed):
        an, bn = ctx.link_names(i)
        yr = ctx.year_of(i)
        return f"{label(an)} and {label(bn) if revealed else '???'}" + (f" ({yr})" if yr else "")

    def tab(d, x, w, title, on, spin=None, alpha=1.0):
        """One tab at x (w wide); on = the active one; spin = loading spinner phase (instead of the favicon)."""
        cy = (TABS[0] + TABS[1]) / 2 + 2
        x1 = min(x + w - 6, FR[2] - 70)
        if x1 - x < 60:
            return
        if on:
            d.rounded_rectangle([x, TABS[0] + 10, x1, TABS[1] + 12], 14, fill=TAB_ON)
            d.rectangle([x, TABS[1] - 2, x1, TABS[1] + 6], fill=TAB_ON)
        else:
            d.line([x1 + 3, cy - 14, x1 + 3, cy + 14], fill=(70, 72, 82), width=2)
        if spin is not None:
            d.arc([x + 14, cy - 12, x + 38, cy + 12], spin * 360, spin * 360 + 270, fill=(80, 150, 255), width=4)
        else:
            d.rounded_rectangle([x + 16, cy - 11, x + 38, cy + 11], 5, fill=ACCENT if on else (120, 70, 66))
        tf = F(21, "Medium")
        d.text((x + 48, cy), clip(title, tf, x1 - x - 86), font=tf, fill=TXT if on else DIM, anchor="lm")
        d.text((x1 - 22, cy), "×", font=F(24, "Medium"), fill=DIM, anchor="mm")

    def page(i, revealed):
        key = (i, revealed)
        if key in _cache:
            return _cache[key]
        link = ctx.spec["links"][i]
        im = Image.new("RGBA", (ctx.W, ctx.H), (255, 255, 255, 255))
        d = ImageDraw.Draw(im)
        # browser window: shadow, frame, page (the tabs are drawn per frame: they move)
        d.rounded_rectangle([FR[0] + 8, FR[1] + 10, FR[2] + 8, FR[3] + 10], 26, fill=(200, 202, 208))
        d.rounded_rectangle(FR, 26, fill=CHROME)
        d.rectangle([FR[0] + 6, PAGE_Y, FR[2] - 6, FR[3] - 6], fill=PAGE)
        # tab strip: window dots, the active tab, a second tab, "+"
        cy = (TABS[0] + TABS[1]) / 2 + 2
        for k, col in enumerate(((255, 95, 87), (254, 188, 46), (40, 200, 64))):
            d.ellipse([FR[0] + 24 + k * 30, cy - 9, FR[0] + 42 + k * 30, cy + 9], fill=col)
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
        path = PATHS[i % len(PATHS)]
        uf = F(24, "Medium")
        path = clip(path, uf, FR[2] - 60 - (lx + 34) - uf.getlength(SITE))
        d.text((lx + 34, acy), SITE, font=uf, fill=TXT, anchor="lm")
        d.text((lx + 34 + uf.getlength(SITE), acy), path, font=uf, fill=DIM, anchor="lm")
        # page: site header (ours), then the hero picture slot, then the headline under it
        d.text((FR[0] + 30, PAGE_Y + 30), "THE PHOTO ARCHIVE", font=F(22, "Black"), fill=ACCENT, anchor="lm")
        for k in range(3):                                                                  # menu icon
            d.rectangle([FR[2] - 62, PAGE_Y + 20 + k * 9, FR[2] - 34, PAGE_Y + 23 + k * 9], fill=(90, 84, 76))
        d.text((FR[2] - 80, PAGE_Y + 30), f"No. {i + 1}", font=F(20, "Medium"), fill=(120, 112, 100), anchor="rm")
        fact = ctx.fact_of(link, full=revealed)
        hf = F(24, "Bold")
        d.text((WX, 1428), clip(fact, hf, credit_left(i) - WX - 24), font=hf, fill=(40, 34, 30), anchor="ls")
        an, bn = ctx.link_names(i)
        im.alpha_composite(ad_im(ctx.sensitive(an) or ctx.sensitive(bn)), AD[:2])
        _cache[key] = im
        return im

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        dur = seg["t1"] - seg["t0"]
        trev = seg.get("rv", ctx.REVEAL) * seg["speech"] + ctx.THROW     # Buh lands on B
        revealed = local >= trev
        last = i == ctx.n_links - 1
        bn = ctx.link_names(i)[1]
        base = page(i, revealed).copy()
        d = ImageDraw.Draw(base)
        # tabs: [previous page] [this link, active] [B's page, opened in the background after the landing]
        e = ctx.ease(min(1.0, local / 0.3)) if i > 0 else 1.0            # the strip slides left as a link opens
        lerp = lambda a, b: a + (b - a) * e
        prev = title_of(i - 1, True) if i > 0 and not ctx.C.fj["links"][i - 1].get("lineage") else "Old photos – archive"
        tab(d, lerp(TAB0 + TW[0], TAB0), int(lerp(TW[1], TW[0])), prev, False)
        t_new = trev + 0.6                                               # B's page opens as a background tab
        clicked = not last and local >= dur - CLICK
        x_act = lerp(TAB0 + TW[0] + TW[1], TAB0 + TW[0])
        tab(d, x_act, int(lerp(TW[2], TW[1])), title_of(i, revealed), not clicked,
            spin=(local * 1.6) % 1.0 if local < 0.45 else None)
        x_new = TAB0 + TW[0] + TW[1]
        if not last and local >= t_new:
            g = ctx.ease(min(1.0, (local - t_new) / 0.2))
            tab(d, x_new, int(TW[2] * g), f"{label(bn)} – photos", clicked, spin=None)
        # the cursor: glides from the address bar to the new tab and clicks it just before the cut; on the next
        # link it rides the strip left with the tab, then fades. Chrome only (never over the photo), under Buh.
        cur, ca, press = None, 0.0, 1.0
        tab_cy = (TABS[0] + TABS[1]) / 2 + 8
        if not last and local >= dur - 0.55:
            u = ctx.ease(min(1.0, (local - (dur - 0.55)) / 0.38))
            x0_, y0_ = FR[2] - 120, (ADDR[0] + ADDR[1]) / 2 + 6
            x1_, y1_ = x_new + 120, tab_cy
            cur = (x0_ + (x1_ - x0_) * u, y0_ + (y1_ - y0_) * u - 18 * math.sin(math.pi * u))
            ca = min(1.0, (local - (dur - 0.55)) / 0.1)
            press = 0.85 if dur - CLICK <= local < dur - CLICK + 0.07 else 1.0
        elif i > 0 and local < 0.5 and not ctx.C.fj["links"][i - 1].get("lineage"):
            cur = (x_act + 120, tab_cy)
            ca = 1.0 if local < 0.3 else max(0.0, 1 - (local - 0.3) / 0.2)
        if cur and ca > 0:
            if press < 1 or (i > 0 and local < 0.12):                    # click ripple
                rr = 10 + 60 * (0.07 - max(0.0, (dur - CLICK + 0.07) - local)) if press < 1 else 22 + 40 * local
                d.ellipse([cur[0] - rr, cur[1] - rr, cur[0] + rr, cur[1] + rr], outline=(120, 170, 255), width=4)
            ci = cursor_im()
            ctx.paste(base, ci, cur[0] + ci.width * press / 2 - 3, cur[1] + ci.height * press / 2 - 2, scale=press,
                      alpha=ca, nearest=False)
        # the ad's button pulses now and then
        ph = (local * 0.8) % 1.0
        pulse = 1.0 + 0.05 * max(0.0, math.sin(ph * 6.283)) if ph < 0.5 else 1.0
        ctx.paste(base, sub_btn(), (AD[0] + AD[2]) / 2, AD[3] - 50, scale=pulse, nearest=False)
        # the page load bar runs under the address row whenever a new page (link) opens, and briefly at the reveal
        for t0_, dur_ in ((0.0, 0.55), (trev, 0.35)):
            u = (local - t0_) / dur_
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
