"""Search engine skin around the show's core: the show's own picture search ("Buh Search", the channel avatar,
Buh the red pin, as its logo). "<A> and " types into the search box, and the top picture result IS the core window
(camera, Buh, red string, name tags). At the reveal B's name finishes typing, the result gets a selection ring and
its title completes. A column of related pictures (this photo's crops and earlier links' photos) sits beside it.
Nothing names B before the reveal."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from ..common import ROOT

NAME = "search"

SC = 0.85                              # core window scale (748 x 807)
WX, WY = 60, 572                       # core window top-left
BOX = (150, 418, 1024, 498)            # search field
TX0 = 60
COLX, COLW, COLH, COLGAP = 824, 206, 128, 14   # related-pictures column (top half only: the feed's buttons sit below)
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
        d.text((COLX, 536), "Related", font=F(22, "Medium"), fill=(150, 154, 160), anchor="lm")
        _cache["bg"] = im
        return im

    def related(i):
        """Thumbs beside the result: crops of this photo (A's face, the scene) and earlier links' photos."""
        if ("rel", i) in _cache:
            return _cache[("rel", i)]
        out = []
        for j in (i - 1, i - 2):
            if j >= 0 and not ctx.C.fj["links"][j].get("lineage"):
                out.append(rounded(ctx.view(j, *crop_cam(j, None, COLW, COLH), COLW, COLH), 14))
        out.insert(0, rounded(ctx.view(i, *crop_cam(i, "a", COLW, COLH), COLW, COLH), 14))
        if len(out) < 3:
            out.insert(1, rounded(ctx.view(i, *crop_cam(i, None, COLW, COLH), COLW, COLH), 14))
        _cache[("rel", i)] = out[:3]
        return _cache[("rel", i)]

    def credit_left(i):
        """x where the core's photo credit (right-aligned at the card's right edge, under it) starts."""
        cr = ctx.C.fj["credits"][i]
        who = (cr.get("artist") or "").strip()
        if len(who) > 26 or "(" in who or "http" in who:
            who = "Wikimedia Commons" if cr.get("file") else ""
        txt = f"PHOTO: {who + ' / ' if who else ''}{cr.get('license', '')}".upper()
        f = ImageFont.truetype(str(ctx.FONTS / "Silkscreen-Bold.ttf"), 18)
        return ctx.CARD[2] - f.getlength(txt) - 4

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
        # related column (top half: clear of the feed's buttons and the captions)
        for k, th in enumerate(related(i)):
            base.alpha_composite(th, (COLX, WY + k * (COLH + COLGAP)))
        sel = min(1.0, max(0.0, (local - trev) / 0.15))

        def mount(b, X0, Y0, w, h):                     # result tile: soft shadow, then a blue ring once picked
            dr = ImageDraw.Draw(b)
            dr.rounded_rectangle([X0 - 6, Y0 + 2, X0 + w + 6, Y0 + h + 10], 18, fill=(228, 230, 234))
            if sel > 0:
                dr.rounded_rectangle([X0 - 9, Y0 - 9, X0 + w + 9, Y0 + h + 9], 16,
                                     fill=tuple(int(255 + (c - 255) * sel) for c in BLUE))
        c = ctx.draw_core(base, seg, tt, WX, WY, SC, frame=mount)
        # result title under the tile (left half: the photo credit is on the right)
        X0, Y0, w, h = c["rect"]
        tl = title_line(i, revealed)
        tf = F(30, "Bold")
        while tf.size > 20 and tf.getlength(tl) > min(640, credit_left(i) - TX0 - 24):
            tf = F(tf.size - 2, "Bold")
        d.text((TX0, Y0 + h + 30), tl, font=tf, fill=(26, 13, 171) if revealed else (70, 60, 150), anchor="lm")
        return ctx.core_finish(base, seg, tt, c)

    return frame
