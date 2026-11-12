"""Search engine: each link, "<A> and " types into a search box and the link's photo slides in as the big top result
(the previous result slides out); at the reveal B's name finishes typing, the result title completes and the cursor
clicks B's face. A fictional engine ("Searchly") with its own wording; nothing names B before the reveal."""
from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

NAME = "search"

X0, X1 = 40, 1040                      # page column (photo is full column width)
TX0 = 70                               # text column left edge (inside the safe area)
BOX = (TX0, 296, 915, 380)             # search field
IMG = (X0, 494, X1, 1116)              # top result image (1000 x 622)
TW, TH, TGAP = 322, 196, 17            # small result thumbs
THUMB_Y = 1380
CAP_Y = 1300                           # caption band (between the result text and the thumbs)
BLUE = (26, 95, 214)
INKT = (32, 33, 36)
GREY = (95, 99, 104)
LIGHT = (232, 234, 237)
BRAND = "Searchly"
SLIDE = 0.3                            # old result out / new result in, s


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

    def cover_cam(i, w, h, fill):
        """both_cam, but never showing view()'s blurred fill: scale up to cover and keep the crop inside the photo."""
        cx, cy, s = ctx.both_cam(i, w, h, fill)
        ih, iw = ctx.C.imgs[i].shape[:2]
        s = max(s, w / iw, h / ih) * 1.002
        cx = min(max(cx, w / 2 / s), iw - w / 2 / s)
        cy = min(max(cy, h / 2 / s), ih - h / 2 / s)
        return cx, cy, s

    def crop_cam(i, kind, w, h):
        """A w x h crop of photo i that stays inside the photo and never upscales past ~1.3x:
        None/"wide2" = the whole scene, "a2"/"b2" = a looser crop around one face."""
        ih, iw = ctx.C.imgs[i].shape[:2]
        cover = max(w / iw, h / ih) * 1.002
        if kind in ("a2", "b2"):
            fb = ctx.fbox(i, kind[0])
            cx, cy = ctx.ctr(fb)
            s = max(cover, min(1.3, h * 0.3 / max(1, fb[3])))
        else:
            cx, cy, s = iw / 2, ih / 2, cover * (1.15 if kind == "wide2" else 1.0)
        return min(max(cx, w / 2 / s), iw - w / 2 / s), min(max(cy, h / 2 / s), ih - h / 2 / s), s

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
            lines[-1] = lines[-1].rstrip(",;:") + "…"
        return lines

    def page_bg():
        """The page without anything query-specific: logo, empty field, our own filter chips."""
        if "bg" in _cache:
            return _cache["bg"]
        im = Image.new("RGBA", (ctx.W, ctx.H), (255, 255, 255, 255))
        d = ImageDraw.Draw(im)
        d.text((TX0, 258), BRAND, font=F(34, "Black"), fill=(92, 60, 220), anchor="lm")
        bw = F(34, "Black").getlength(BRAND)
        d.ellipse([TX0 + bw + 6, 264, TX0 + bw + 18, 276], fill=(255, 190, 0))
        d.ellipse([871, 238, 915, 282], fill=(92, 60, 220))                     # account avatar
        d.text((893, 260), "B", font=F(24, "Black"), fill=(255, 255, 255), anchor="mm")
        d.rounded_rectangle([BOX[0] + 2, BOX[1] + 5, BOX[2] + 2, BOX[3] + 5], 42, fill=(222, 224, 228))
        d.rounded_rectangle(BOX, 42, fill=(255, 255, 255), outline=(218, 220, 224), width=2)
        cy = (BOX[1] + BOX[3]) / 2
        d.ellipse([BOX[0] + 30, cy - 17, BOX[0] + 58, cy + 11], outline=GREY, width=4)  # magnifier
        d.line([BOX[0] + 54, cy + 7, BOX[0] + 66, cy + 19], fill=GREY, width=5)
        d.line([BOX[2] - 50, cy - 12, BOX[2] - 26, cy + 12], fill=GREY, width=4)       # clear ×
        d.line([BOX[2] - 50, cy + 12, BOX[2] - 26, cy - 12], fill=GREY, width=4)
        d.line([BOX[2] - 76, cy - 22, BOX[2] - 76, cy + 22], fill=LIGHT, width=2)
        x = TX0
        for k, chip in enumerate(("Pictures", "Old photos", "Stories", "Clips")):   # pill chips, ours
            f = F(24)
            w = f.getlength(chip) + 40
            on = k == 0
            d.rounded_rectangle([x, 404, x + w, 446], 21, fill=(232, 226, 252) if on else (255, 255, 255),
                                outline=(92, 60, 220) if on else (218, 220, 224), width=2)
            d.text((x + w / 2, 425), chip, font=f, fill=(70, 40, 190) if on else GREY, anchor="mm")
            x += w + 14
        _cache["bg"] = im
        return im

    def title_for(i, revealed):
        an, bn = ctx.link_names(i)
        yr = ctx.year_of(i)
        return f"{label(an)} and {label(bn) if revealed else '???'}" + (f" ({yr})" if yr else "")

    def text_block(i, revealed):
        """Title + fact under the image (B only after the reveal)."""
        k = ("txt", i, revealed)
        if k in _cache:
            return _cache[k]
        link = ctx.spec["links"][i]
        w = 860
        im = Image.new("RGBA", (w, 150), (255, 255, 255, 0))
        d = ImageDraw.Draw(im)
        title = title_for(i, revealed)
        tf = F(38)
        while tf.size > 26 and tf.getlength(title) > w:
            tf = F(tf.size - 2)
        d.text((0, 4), title, font=tf, fill=(26, 13, 171) if revealed else (60, 50, 150))
        fact = ctx.fact_of(link, full=revealed)
        for k2, ln in enumerate(wrap(fact, F(28, "Medium"), w, 2)):
            d.text((0, 56 + k2 * 36), ln, font=F(28, "Medium"), fill=(77, 81, 86))
        _cache[k] = im
        return im

    def results(i):
        """Static results for link i: big image (oversized for a slow zoom), its mask, thumbs, B's face on screen."""
        if ("res", i) in _cache:
            return _cache[("res", i)]
        iw, ih = IMG[2] - IMG[0], IMG[3] - IMG[1]
        bw, bh = int(iw * 1.08), int(ih * 1.08)
        cam = cover_cam(i, bw, bh, 0.82 * 1.08)
        big = ctx.view(i, *cam, bw, bh)
        bx, by = ctx.ctr(ctx.fbox(i, "b"))
        bface = ((bx - cam[0]) * cam[2] + bw / 2, (by - cam[1]) * cam[2] + bh / 2)   # in `big` px
        thumbs = []
        for kind in ("a", "wide", "b"):
            if kind == "wide":
                th = ctx.view(i, *cover_cam(i, TW, TH, 0.95), TW, TH)
            else:
                fb = ctx.fbox(i, kind)
                th = ctx.view(i, *crop_cam(i, kind + "2", TW, TH)[:2],
                              max(max(TW / ctx.C.imgs[i].shape[1], TH / ctx.C.imgs[i].shape[0]) * 1.002,
                                  min(1.3, TH * 0.5 / max(1, fb[3]))), TW, TH)
            thumbs.append(rounded(th, 16))
        mask = rounded(Image.new("RGBA", (iw, ih)), 24).split()[-1]
        # a second row: other crops of the chain's photos so far (this one and the ones before; never later ones)
        more = []
        for j in (i - 1, i - 2, i - 3):
            if j >= 0 and not ctx.C.fj["links"][j].get("lineage"):
                more.append(rounded(ctx.view(j, *crop_cam(j, None, TW, TH), TW, TH), 16))
        for kind in ("wide2", "a2", "b2"):
            if len(more) >= 3:
                break
            more.append(rounded(ctx.view(i, *crop_cam(i, kind, TW, TH), TW, TH), 16))
        _cache[("res", i)] = (big, mask, thumbs, bface, more)
        return _cache[("res", i)]

    _cur = {}

    def cursor(press=0.0):
        k = round(press, 1)
        if k not in _cur:
            s = 3.2 * (1 - 0.15 * k)
            pts = [(0, 0), (0, 17), (4, 13), (7, 20), (10, 19), (7, 12), (12, 12)]
            im = Image.new("RGBA", (int(16 * s) + 8, int(24 * s) + 8), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.polygon([(x * s + 5, y * s + 5) for x, y in pts], fill=(0, 0, 0, 70))
            d.polygon([(x * s + 3, y * s + 3) for x, y in pts], fill=(255, 255, 255))
            d.line([(x * s + 3, y * s + 3) for x, y in pts + pts[:1]], fill=(0, 0, 0), width=3)
            _cur[k] = im
        return _cur[k]

    def zoomed(i, z):
        big, mask, _, bface, _ = results(i)
        iw, ih = IMG[2] - IMG[0], IMG[3] - IMG[1]
        cw, ch = int(iw * z), int(ih * z)
        ox, oy = (big.width - cw) // 2, (big.height - ch) // 2
        ph = big.crop((ox, oy, ox + cw, oy + ch)).resize((iw, ih), Image.BILINEAR)
        ph.putalpha(mask)
        face = ((bface[0] - ox) * iw / cw + IMG[0], (bface[1] - oy) * ih / ch + IMG[1])
        return ph, face

    def draw_result(base, i, dx, z, revealed, alpha=1.0, ring=0.0, sc=1.0):
        """The result image + its text block + thumbs, shifted horizontally by dx."""
        iw, ih = IMG[2] - IMG[0], IMG[3] - IMG[1]
        ph, face = zoomed(i, z)
        icx, icy = (IMG[0] + IMG[2]) / 2 + dx, (IMG[1] + IMG[3]) / 2
        if ring > 0:
            rg = Image.new("RGBA", (iw + 14, ih + 14), (0, 0, 0, 0))
            ImageDraw.Draw(rg).rounded_rectangle([0, 0, iw + 13, ih + 13], 30, fill=BLUE + (int(255 * ring),))
            ctx.paste(base, rg, icx, icy, scale=sc, nearest=False)
        ctx.paste(base, ph, icx, icy, scale=sc, alpha=alpha, nearest=False)
        tb = text_block(i, revealed)
        ctx.paste(base, tb, TX0 + tb.width / 2 + dx, IMG[3] + 14 + tb.height / 2, alpha=alpha)
        for k, th in enumerate(results(i)[2]):
            ctx.paste(base, th, X0 + k * (TW + TGAP) + TW / 2 + dx, THUMB_Y + TH / 2, alpha=alpha)
        for k, th in enumerate(results(i)[4]):
            ctx.paste(base, th, X0 + k * (TW + TGAP) + TW / 2 + dx, THUMB_Y + TH + TGAP + TH / 2, alpha=alpha)
        return (icx + (face[0] - (IMG[0] + IMG[2]) / 2) * sc, icy + (face[1] - icy) * sc)

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        dur = seg["t1"] - seg["t0"]
        sp, R = seg["speech"], seg.get("rv", ctx.REVEAL)
        trev = R * sp + ctx.THROW
        an, bn = ctx.link_names(i)
        base = page_bg().copy()
        d = ImageDraw.Draw(base)
        # typing: "A and " right away; B's name only types from the reveal on (nothing names B before it)
        pa, pb = f"{label(an)} and ", label(bn)
        t_a = max(0.35, min(0.9, trev - 0.25))
        t_b = 0.32
        ka = len(pa) * min(1.0, max(0.0, (local - 0.05) / t_a))
        kb = len(pb) * min(1.0, max(0.0, (local - trev) / t_b))
        typed = pa[:int(ka)] + pb[:int(kb)]
        revealed = local >= trev
        qf = F(34)
        cy = (BOX[1] + BOX[3]) / 2
        qx0, qmax = BOX[0] + 84, BOX[2] - 100 - (BOX[0] + 84)
        shown = typed
        while qf.getlength(shown) > qmax:                     # long query: the field scrolls to keep the caret visible
            shown = shown[1:]
        d.text((qx0, cy), shown, font=qf, fill=INKT, anchor="lm")
        if local < trev + t_b + 0.1 or int(local * 2.4) % 2 == 0:
            cx = qx0 + qf.getlength(shown) + 4
            d.rectangle([cx, cy - 22, cx + 3, cy + 22], fill=BLUE)
        n = 10000 * int(170 + 230 * ((i * 7919 + len(an)) % 97) / 97)
        d.text((TX0, 464), f"Showing {n:,} pictures", font=F(22, "Medium"), fill=(112, 117, 122), anchor="lm")
        # Buh peeks over the top-right corner of the result (behind it), and reacts at the reveal
        sens = ctx.sensitive(an) or ctx.sensitive(bn)
        revealed_ = local >= trev
        if sens:                                   # neutral on a sensitive link: no grin, no celebration hop
            expr = "shock"
        else:
            expr = ("shock" if local < trev + 0.45 else "happy") if revealed_ else "smug"
        hop = 0.0
        if revealed_ and local < trev + 0.4 and not sens:
            hop = 30 * math.sin(math.pi * (local - trev) / 0.4)
        rise = ctx.ease((local - 0.3) / 0.35) if i == 0 else 1.0     # rises once the first result is opaque
        bx_ = 850 + 4 * math.sin(local * 1.3)
        by_ = IMG[1] + 50 + 80 * (1 - rise) - hop + 3 * math.sin(local * 2.1)
        if i > 0 or local >= 0.26:
            ctx.paste_pin(base, ctx.pin_sprite(expr, 6 * math.sin(local * 1.5), 0, -1, scale=3), bx_, by_)
        # the previous result slides out to the left while this one slides in from the right
        e = ctx.ease(min(1.0, local / SLIDE))
        has_prev = i > 0 and not ctx.C.fj["links"][i - 1].get("lineage")
        if has_prev and e < 1:
            draw_result(base, i - 1, -ctx.W * 0.9 * e, 1.01, True, alpha=1 - 0.6 * e)
        z = 1.08 - 0.07 * min(1.0, local / max(1.0, dur))
        t_click = trev + t_b + 0.12
        sel = local > t_click
        pop = 1.0 + (0.012 * math.sin(min(1.0, (local - t_click) / 0.3) * math.pi) if sel else 0.0)
        enter = (ctx.W * 0.9 * (1 - e)) if has_prev else 0.0
        a_in = 1.0 if has_prev else ctx.ease(local / 0.25)
        face = draw_result(base, i, enter, z, revealed, alpha=a_in,
                           ring=min(1.0, (local - t_click) / 0.15) if sel else 0.0, sc=pop)
        # cursor: rests on the result's title text (never on someone's body); glides to B's face at the reveal
        tf_w = min(820, F(38).getlength(title_for(i, False)))
        rx = TX0 + tf_w + 26 + 10 * math.sin(local * 1.4)
        ry = IMG[3] + 26 + 6 * math.sin(local * 2.0)
        fx_, fy_ = face
        fy_ = min(max(fy_, IMG[1] + 40), IMG[3] - 40)
        fx_ = min(max(fx_, IMG[0] + 40), 900)
        g = ctx.ease((local - (t_click - 0.4)) / 0.4)
        g *= 1 - ctx.ease((local - t_click - 0.45) / 0.4)          # after the click it glides back off the faces
        mx, my = rx + (fx_ - rx) * g, ry + (fy_ - ry) * g
        press = max(0.0, 1 - abs(local - t_click) / 0.1)
        if sel and local - t_click < 0.45:
            r = 12 + 150 * (local - t_click)
            a = int(170 * (1 - (local - t_click) / 0.45))
            rg = Image.new("RGBA", (int(2 * r + 8),) * 2, (0, 0, 0, 0))
            ImageDraw.Draw(rg).ellipse([4, 4, 2 * r + 4, 2 * r + 4], outline=(255, 255, 255, a), width=6)
            ctx.paste(base, rg, mx, my)
        if local > SLIDE * 0.8:
            base.alpha_composite(cursor(press), (int(mx - 3), int(my - 3)))
        # autocomplete while A's half types: completes A's name only (never suggests B)
        if 0.1 < local < t_a + 0.15 and typed.strip():
            yr = ctx.year_of(i)
            A = label(an)
            sugg = (f"{A} photos", f"{A} and …", f"{A} {yr}" if yr else f"{A} history")
            dy0 = BOX[3] + 6
            d.rounded_rectangle([BOX[0], dy0, BOX[2], dy0 + 24 + 66 * len(sugg)], 22, fill=(255, 255, 255),
                                outline=(218, 220, 224), width=2)
            sf = F(30)
            dark = len(typed.rstrip())
            for k, full in enumerate(sugg):
                ry_ = dy0 + 46 + 66 * k
                d.ellipse([BOX[0] + 34, ry_ - 12, BOX[0] + 54, ry_ + 8], outline=(154, 160, 166), width=3)
                d.line([BOX[0] + 51, ry_ + 5, BOX[0] + 59, ry_ + 13], fill=(154, 160, 166), width=4)
                cut = 0
                while sf.getlength(("…" if cut else "") + full[cut:]) > BOX[2] - BOX[0] - 130:
                    cut += 1
                pre = ("…" if cut else "") + full[cut:max(cut, min(dark, len(full)))]
                d.text((BOX[0] + 84, ry_), pre, font=sf, fill=INKT, anchor="lm")
                d.text((BOX[0] + 84 + sf.getlength(pre), ry_), full[max(cut, min(dark, len(full))):], font=sf,
                       fill=(130, 134, 140), anchor="lm")
        if i == 0 and local < 0.3:                      # fade in from the dim hook (no hard cut to white)
            a = int(200 * (1 - ctx.ease(local / 0.3)))
            if a > 0:
                base.alpha_composite(Image.new("RGBA", (ctx.W, ctx.H), (16, 10, 24, a)))
        ctx.header(base, i)
        ctx.connected(base, seg, tt, y=IMG[3] - 70)
        seg["cap"].draw(base, tt, y=CAP_Y)
        return base

    return frame
