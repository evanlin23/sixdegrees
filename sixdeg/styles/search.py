"""Search engine skin around the show's core: the show's own picture search ("Buh Search", the channel avatar,
Buh the red pin, as its logo). "<A> and " types into the search box, and the top picture result IS the core window
(camera, Buh, red string, name tags). At the reveal B's name finishes typing, the result gets a selection ring and
its title completes. Beside it: a sponsored "ad" for the channel itself (Buh's avatar, Like & Subscribe, a
SUBSCRIBE button; no "Like" on sensitive links), then a "Related" column that runs down the page and collects every
link photo seen so far (the newest slides in on top; future photos never appear). The results scroll up a little
over each link so the page never sits still. Nothing names B before the reveal."""
from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

from ..common import ROOT

NAME = "search"

SC = 0.85                              # core window scale (748 x 807)
WX, WY = 60, 572                       # core window top-left
BOX = (150, 418, 1024, 498)            # search field
TX0 = 60
COLX, COLW, COLGAP = 824, 206, 10     # right column: the sponsored tile, then every photo so far
AD_H = 236                             # sponsored tile height (top of the column, above the feed's buttons)
REL_Y = WY + AD_H + 46                 # first related thumb
REL_END = 1440                         # the column runs down to the bottom of the safe area
DRIFT = 26                             # px the results scroll up over one link
BLUE = (26, 95, 214)
PURPLE = (92, 60, 220)
INKT = (32, 33, 36)
GREY = (95, 99, 104)
LIGHT = (218, 220, 224)
AVATAR = ROOT / "branding" / "avatar-purple.png"


def make(ctx):
    _fonts, _cache = {}, {}

    def F(size, weight="Bold"):
        k = (size, weight)
        if k not in _fonts:
            _fonts[k] = ImageFont.truetype(str(ctx.FONTS / f"Inter-{weight}.otf"), size)
        return _fonts[k]

    def label(n):
        return ctx.C.labels.get(n, n)

    def rounded(im, r):
        m = Image.new("L", im.size, 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, im.width - 1, im.height - 1], r, fill=255)
        out = im.convert("RGBA")
        out.putalpha(m)
        return out

    def crop_cam(i, kind, w, h):
        """A w x h crop of photo i inside the photo: None = whole scene, "a"/"b" = a looser crop around one face."""
        ih, iw = ctx.C.imgs[i].shape[:2]
        cover = max(w / iw, h / ih) * 1.002
        if kind in ("a", "b"):
            fb = ctx.fbox(i, kind)
            cx, cy = ctx.ctr(fb)
            s = max(cover, min(1.3, h * 0.45 / max(1, fb[3])))
        else:
            cx, cy, s = iw / 2, ih / 2, cover
        return min(max(cx, w / 2 / s), iw - w / 2 / s), min(max(cy, h / 2 / s), ih - h / 2 / s), s

    def page_bg():
        if "bg" in _cache:
            return _cache["bg"]
        im = Image.new("RGBA", (ctx.W, ctx.H), (255, 255, 255, 255))
        d = ImageDraw.Draw(im)
        # our own mark: the channel avatar (Buh, the red pin) in a rounded tile
        n = BOX[3] - BOX[1] - 4
        try:
            av = Image.open(AVATAR).convert("RGBA")
            s_ = av.width
            av = av.crop((int(s_ * 0.16), int(s_ * 0.08), int(s_ * 0.84), int(s_ * 0.76)))
            av = av.resize((n, n), Image.NEAREST if av.width % n == 0 else Image.LANCZOS)
        except OSError:                                  # no branding file: Buh's sprite on the purple tile
            av = Image.new("RGBA", (n, n), PURPLE + (255,))
            pin = ctx.pin_sprite("happy", scale=1)
            av.alpha_composite(pin, ((n - pin.width) // 2, (n - pin.height) // 2))
        im.alpha_composite(rounded(av, 20), (TX0, BOX[1] + 2))
        d.rounded_rectangle([BOX[0] + 2, BOX[1] + 5, BOX[2] + 2, BOX[3] + 5], 40, fill=(226, 228, 232))
        d.rounded_rectangle(BOX, 40, fill=(255, 255, 255), outline=LIGHT, width=2)
        cy = (BOX[1] + BOX[3]) / 2
        d.ellipse([BOX[0] + 28, cy - 17, BOX[0] + 56, cy + 11], outline=GREY, width=4)      # magnifier
        d.line([BOX[0] + 52, cy + 7, BOX[0] + 64, cy + 19], fill=GREY, width=5)
        d.line([BOX[2] - 50, cy - 12, BOX[2] - 26, cy + 12], fill=GREY, width=4)           # clear x
        d.line([BOX[2] - 50, cy + 12, BOX[2] - 26, cy - 12], fill=GREY, width=4)
        x = TX0
        for k, tab in enumerate(("All", "Pictures", "Old photos", "Stories", "Clips")):    # tabs, ours
            f = F(24, "Bold" if k == 1 else "Medium")
            w = f.getlength(tab)
            d.text((x, 536), tab, font=f, fill=PURPLE if k == 1 else GREY, anchor="lm")
            if k == 1:
                d.rounded_rectangle([x - 4, 556, x + w + 4, 561], 2, fill=PURPLE)
            x += w + 34
        d.line([0, 562, ctx.W, 562], fill=(234, 236, 239), width=2)
        _cache["bg"] = im
        return im

    def avatar(n):
        if ("av", n) not in _cache:
            try:
                av = Image.open(AVATAR).convert("RGBA")
                s_ = av.width
                av = av.crop((int(s_ * 0.16), int(s_ * 0.08), int(s_ * 0.84), int(s_ * 0.76))).resize((n, n), Image.LANCZOS)
            except OSError:
                av = Image.new("RGBA", (n, n), PURPLE + (255,))
                pin = ctx.pin_sprite("happy", scale=max(1, n // 40))
                av.alpha_composite(pin, ((n - pin.width) // 2, (n - pin.height) // 2))
            m = Image.new("L", (n, n), 0)
            ImageDraw.Draw(m).ellipse([0, 0, n - 1, n - 1], fill=255)
            av.putalpha(m)
            _cache[("av", n)] = av
        return _cache[("av", n)]

    def ad_tile(soft):
        """A sponsored result for the channel itself. soft: a sensitive link, so no "Like" (just Subscribe)."""
        if ("ad", soft) in _cache:
            return _cache[("ad", soft)]
        im = Image.new("RGBA", (COLW, AD_H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([0, 0, COLW - 1, AD_H - 1], 16, fill=(248, 247, 253), outline=(222, 218, 240), width=2)
        d.rounded_rectangle([12, 12, 44, 34], 6, fill=INKT)
        d.text((28, 23), "Ad", font=F(16, "Bold"), fill=(255, 255, 255), anchor="mm")
        d.text((52, 23), "Sponsored", font=F(17, "Medium"), fill=GREY, anchor="lm")
        im.alpha_composite(avatar(62), (12, 46))
        d.text((84, 62), "Buh", font=F(23, "Bold"), fill=INKT, anchor="lm")
        d.text((84, 90), "Connects", font=F(23, "Bold"), fill=INKT, anchor="lm")
        lines = ["Subscribe for more", "connections"] if soft else ["Like & Subscribe for", "more connections"]
        for k, ln in enumerate(lines):
            d.text((14, 132 + 25 * k), ln, font=F(19, "Medium"), fill=(60, 64, 70), anchor="lm")
        _cache[("ad", soft)] = im
        return im

    def sub_btn(pressed):
        if ("btn", pressed) not in _cache:
            w, h = COLW - 28, 44
            im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(im).rounded_rectangle([0, 0, w - 1, h - 1], 22, fill=(150, 0, 0) if pressed else (204, 0, 0))
            ImageDraw.Draw(im).text((w / 2, h / 2), "SUBSCRIBE", font=F(21, "Bold"), fill=(255, 255, 255), anchor="mm")
            _cache[("btn", pressed)] = im
        return _cache[("btn", pressed)]

    def slot_h():
        """Related thumb height: room for every (non-lineage) link photo between REL_Y and REL_END."""
        n = max(1, sum(1 for L in ctx.C.fj["links"] if not L.get("lineage")))
        return int(max(60, min(128, (REL_END - REL_Y) / n - COLGAP)))

    def thumb(j, h):
        """Photo j (whole scene) as a related thumb."""
        key = ("th", j, h)
        if key not in _cache:
            _cache[key] = rounded(ctx.view(j, *crop_cam(j, None, COLW, h), COLW, h), 12)
        return _cache[key]

    def credit_left(i, right):
        """x where the core's photo credit (right-aligned at the window's right edge, under it) starts."""
        cr = ctx.C.fj["credits"][i]
        who = (cr.get("artist") or "").strip()
        if len(who) > 26 or "(" in who or "http" in who:
            who = "Wikimedia Commons" if cr.get("file") else ""
        txt = f"PHOTO: {who + ' / ' if who else ''}{cr.get('license', '')}".upper()
        f = ImageFont.truetype(str(ctx.FONTS / "Silkscreen-Bold.ttf"), 18)
        return right - f.getlength(txt) - 4

    def title_line(i, revealed):
        an, bn = ctx.link_names(i)
        yr = ctx.year_of(i)
        return f"{label(an)} and {label(bn) if revealed else '???'}" + (f" ({yr})" if yr else "")

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        sp, R = seg["speech"], seg.get("rv", ctx.REVEAL)
        trev = R * sp + ctx.THROW                       # Buh lands on B: B's name may appear from here on
        revealed = local >= trev
        an, bn = ctx.link_names(i)
        base = page_bg().copy()
        d = ImageDraw.Draw(base)
        # the query: "A and " types right away; B's name only from the landing on
        pa, pb = f"{label(an)} and ", label(bn)
        t_a = max(0.35, min(0.9, trev - 0.3))
        ka = len(pa) * min(1.0, max(0.0, (local - 0.05) / t_a))
        kb = len(pb) * min(1.0, max(0.0, (local - trev) / 0.32))
        typed = pa[:int(ka)] + pb[:int(kb)]
        qf = F(34, "Medium")
        cy = (BOX[1] + BOX[3]) / 2
        qx0, qmax = BOX[0] + 82, BOX[2] - 80 - (BOX[0] + 82)
        shown = typed
        while qf.getlength(shown) > qmax:                # long query: the field scrolls to keep the caret in view
            shown = shown[1:]
        d.text((qx0, cy), shown, font=qf, fill=INKT, anchor="lm")
        if local < trev + 0.45 or int(local * 2.4) % 2 == 0:
            cx = qx0 + qf.getlength(shown) + 4
            d.rectangle([cx, cy - 21, cx + 3, cy + 21], fill=BLUE)
        # the results scroll up a little over the link (the search box stays put); a new search starts low again
        dy = round(DRIFT * (1 - min(1.0, local / max(0.1, seg["t1"] - seg["t0"]))))
        # right column: the sponsored tile on top, its button pulsing now and then
        base.alpha_composite(ad_tile(ctx.sensitive(an) or ctx.sensitive(bn)), (COLX, WY + dy))
        ph = (local * 0.8) % 1.0
        pulse = 1.0 + 0.05 * max(0.0, math.sin(ph * 6.283)) if ph < 0.5 else 1.0
        ctx.paste(base, sub_btn(False), COLX + COLW / 2, WY + dy + AD_H - 36, scale=pulse, nearest=False)
        # then "Related": every link photo so far, newest on top (it slides in as the link opens); never a future one
        sh = slot_h()
        ry = REL_Y + dy
        d.text((COLX + 2, ry - 22), "Related", font=F(22, "Medium"), fill=(150, 154, 160), anchor="lm")
        # the current link's photo joins only once Buh has landed on B (its thumb would show B's face early)
        grow = ctx.ease(min(1.0, max(0.0, (local - trev - 0.1) / 0.4)))
        y = ry - (1 - grow) * (sh + COLGAP) if grow > 0 else ry
        for j in range(i, -1, -1):
            if ctx.C.fj["links"][j].get("lineage") or (j == i and grow <= 0):
                continue
            if y > REL_END:
                break
            th = thumb(j, sh)
            top = int(ry - y)                                # the newest slides in from under the "Related" label
            if top >= th.height:
                y += sh + COLGAP
                continue
            th = th.crop((0, max(0, top), th.width, int(min(th.height, REL_END - y))))
            if j == i and grow < 1:
                th.putalpha(th.getchannel("A").point(lambda v, g=grow: int(v * g)))
            base.alpha_composite(th, (COLX, int(max(y, ry))))
            y += sh + COLGAP
        sel = min(1.0, max(0.0, (local - trev) / 0.15))

        def mount(b, X0, Y0, w, h):                     # result tile: soft shadow, then a blue ring once picked
            dr = ImageDraw.Draw(b)
            dr.rounded_rectangle([X0 - 6, Y0 + 2, X0 + w + 6, Y0 + h + 10], 18, fill=(228, 230, 234))
            if sel > 0:
                dr.rounded_rectangle([X0 - 9, Y0 - 9, X0 + w + 9, Y0 + h + 9], 16,
                                     fill=tuple(int(255 + (c - 255) * sel) for c in BLUE))
        c = ctx.draw_core(base, seg, tt, WX, WY + dy, SC, frame=mount)
        # result title under the tile (left half: the photo credit is on the right)
        X0, Y0, w, h = c["rect"]
        tl = title_line(i, revealed)
        tf = F(30, "Bold")
        room = min(640, credit_left(i, X0 + w) - TX0 - 24)
        while tf.size > 20 and tf.getlength(tl) > room:
            tf = F(tf.size - 2, "Bold")
        if tf.getlength(tl) > room and " (" in tl:      # long credit: drop the year, then trim
            tl = tl.rsplit(" (", 1)[0]
        if tf.getlength(tl) > room:
            while len(tl) > 4 and tf.getlength(tl + "…") > room:
                tl = tl[:-1]
            tl = tl.rstrip() + "…"
        d.text((TX0, Y0 + h + 30), tl, font=tf, fill=(26, 13, 171) if revealed else (70, 60, 150), anchor="lm")
        return ctx.core_finish(base, seg, tt, c)

    return frame
