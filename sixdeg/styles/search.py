"""Search engine: "<A> and <B>" types into a search box letter by letter, the image results load (skeleton shimmer
first), and the link's photo is the big top result with the year and fact under it; the cursor clicks it."""
from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

NAME = "search"

X0, X1 = 70, 915                       # page column
BOX = (X0, 318, X1, 410)               # search field
IMG = (X0, 540, X1, 1030)              # top result image
TW, TH, TGAP = 271, 160, 16            # small result thumbs
THUMB_Y = 1196
BLUE = (26, 95, 214)
INKT = (32, 33, 36)
GREY = (95, 99, 104)
LIGHT = (232, 234, 237)
BRAND = "Searchly"


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
        """The page without anything query-specific: logo, empty field, tabs."""
        if "bg" in _cache:
            return _cache["bg"]
        im = Image.new("RGBA", (ctx.W, ctx.H), (255, 255, 255, 255))
        d = ImageDraw.Draw(im)
        d.text((X0, 268), BRAND, font=F(36, "Black"), fill=(92, 60, 220), anchor="lm")
        bw = F(36, "Black").getlength(BRAND)
        d.ellipse([X0 + bw + 6, 274, X0 + bw + 18, 286], fill=(255, 190, 0))
        d.ellipse([X1 - 44, 246, X1, 290], fill=(92, 60, 220))                     # account avatar
        d.text((X1 - 22, 268), "B", font=F(24, "Black"), fill=(255, 255, 255), anchor="mm")
        d.rounded_rectangle([BOX[0] + 2, BOX[1] + 5, BOX[2] + 2, BOX[3] + 5], 46, fill=(222, 224, 228))
        d.rounded_rectangle(BOX, 46, fill=(255, 255, 255), outline=(218, 220, 224), width=2)
        cy = (BOX[1] + BOX[3]) / 2
        d.ellipse([BOX[0] + 30, cy - 17, BOX[0] + 58, cy + 11], outline=GREY, width=4)  # magnifier
        d.line([BOX[0] + 54, cy + 7, BOX[0] + 66, cy + 19], fill=GREY, width=5)
        d.line([BOX[2] - 50, cy - 12, BOX[2] - 26, cy + 12], fill=GREY, width=4)       # clear ×
        d.line([BOX[2] - 50, cy + 12, BOX[2] - 26, cy - 12], fill=GREY, width=4)
        d.line([BOX[2] - 76, cy - 22, BOX[2] - 76, cy + 22], fill=LIGHT, width=2)
        x = X0
        for k, tab in enumerate(("All", "Images", "Videos", "News", "Maps")):
            f = F(26)
            d.text((x, 458), tab, font=f, fill=BLUE if k == 1 else GREY, anchor="lm")
            if k == 1:
                d.rounded_rectangle([x, 482, x + f.getlength(tab), 488], 3, fill=BLUE)
            x += f.getlength(tab) + 44
        d.line([0, 489, ctx.W, 489], fill=LIGHT, width=2)
        _cache["bg"] = im
        return im

    def results(i):
        """Static results for link i: big image (oversized for a slow zoom), caption block, thumbs."""
        if i in _cache:
            return _cache[i]
        an, bn = ctx.link_names(i)
        link = ctx.spec["links"][i]
        iw, ih = IMG[2] - IMG[0], IMG[3] - IMG[1]
        big = ctx.view(i, *ctx.both_cam(i, int(iw * 1.08), int(ih * 1.08), 0.8 * 1.08), int(iw * 1.08), int(ih * 1.08))
        yr = ctx.year_of(i)
        cap = Image.new("RGBA", (iw, 150), (255, 255, 255, 0))
        d = ImageDraw.Draw(cap)
        d.ellipse([0, 4, 26, 30], fill=(92, 60, 220))
        src = f"photo archive · {yr}" if yr else "photo archive"
        d.text((36, 17), src, font=F(22), fill=GREY, anchor="lm")
        title = f"{label(an)} and {label(bn)}" + (f" ({yr})" if yr else "")
        tf = F(32)
        while tf.size > 24 and tf.getlength(title) > iw:
            tf = F(tf.size - 2)
        d.text((0, 42), title, font=tf, fill=(26, 13, 171))
        for k, ln in enumerate(wrap(ctx.fact_of(link), F(26), iw, 2)):
            d.text((0, 86 + k * 34), ln, font=F(26), fill=(77, 81, 86))
        thumbs = []
        for kind in ("a", "wide", "b"):
            if kind == "wide":
                th = ctx.view(i, *ctx.both_cam(i, TW, TH, 0.95), TW, TH)
            else:
                th = ctx.view(i, *ctx.face_cam(ctx.fbox(i, kind), TH, 0.55), TW, TH)
            thumbs.append(rounded(th, 14))
        _cache[i] = (big, rounded(Image.new("RGBA", (iw, ih)), 22).split()[-1], cap, thumbs)
        return _cache[i]

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

    def skeleton(d, t, boxes):
        for k, (x0, y0, x1, y1) in enumerate(boxes):
            v = int(232 + 14 * math.sin(t * 7 - k * 0.9))
            d.rounded_rectangle([x0, y0, x1, y1], 16, fill=(v, v + 1, v + 4))

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
        # typing: "A and " first, then B's name, finishing (Enter) around the reveal
        t_enter = min(max(1.0, trev - 0.05), 1.7)
        pa, pb = f"{label(an)} and ", label(bn)
        ka = len(pa) * min(1.0, max(0.0, (local - 0.08) / (0.55 * t_enter)))
        kb = len(pb) * min(1.0, max(0.0, (local - 0.08 - 0.55 * t_enter) / (0.42 * t_enter)))
        typed = (pa[:int(ka)] + pb[:int(kb)]) if local < t_enter else pa + pb
        qf = F(34)
        cy = (BOX[1] + BOX[3]) / 2
        qx0, qmax = BOX[0] + 84, BOX[2] - 100 - (BOX[0] + 84)
        shown = typed
        while qf.getlength(shown) > qmax:                     # long query: the field scrolls to keep the caret visible
            shown = shown[1:]
        d.text((qx0, cy), shown, font=qf, fill=INKT, anchor="lm")
        if local < t_enter + 0.1 or int(local * 2.4) % 2 == 0:
            cx = qx0 + qf.getlength(shown) + 4
            d.rectangle([cx, cy - 22, cx + 3, cy + 22], fill=BLUE)
        # results: skeleton until loaded, then the image and thumbs drop in
        t_res = t_enter + 0.22
        iw, ih = IMG[2] - IMG[0], IMG[3] - IMG[1]
        big, mask, cap, thumbs = results(i)
        if local < t_enter:
            pass
        else:
            n = 10000 * int(170 + 230 * ((i * 7919 + len(an)) % 97) / 97)
            d.text((X0, 516), f"About {n:,} results ({0.31 + 0.04 * (i % 4):.2f} seconds)", font=F(22),
                   fill=(112, 117, 122), anchor="lm")
        if local < t_res:
            skeleton(d, local, [IMG, (X0, IMG[3] + 20, X0 + 420, IMG[3] + 46), (X0, IMG[3] + 62, X1 - 120, IMG[3] + 92)]
                     + [(X0 + k * (TW + TGAP), THUMB_Y, X0 + k * (TW + TGAP) + TW, THUMB_Y + TH) for k in range(3)])
        else:
            u = local - t_res
            ap = ctx.ease(u / 0.3)
            # slow push-in on the photo (keeps moving the whole time)
            z = 1.08 - 0.07 * min(1.0, u / max(1.0, dur - t_res))
            cw, ch = int(iw * z), int(ih * z)
            ox, oy = (big.width - cw) // 2, (big.height - ch) // 2
            ph = big.crop((ox, oy, ox + cw, oy + ch)).resize((iw, ih), Image.BILINEAR)
            ph.putalpha(mask)
            t_click = max(trev + 0.1, t_res + 0.9)
            sel = local > t_click
            pop = 1.0 + (0.035 * math.sin(min(1.0, (local - t_click) / 0.3) * math.pi) if sel else 0.0)
            sc = (0.94 + 0.06 * ctx.ease_out_back(min(1.0, u / 0.35))) * pop
            if sel:
                ring = Image.new("RGBA", (iw + 20, ih + 20), (0, 0, 0, 0))
                ImageDraw.Draw(ring).rounded_rectangle([0, 0, iw + 19, ih + 19], 30, fill=BLUE)
                ctx.paste(base, ring, (IMG[0] + IMG[2]) / 2, (IMG[1] + IMG[3]) / 2, scale=sc, nearest=False)
            ctx.paste(base, ph, (IMG[0] + IMG[2]) / 2, (IMG[1] + IMG[3]) / 2 + 30 * (1 - ap), scale=sc,
                      alpha=ap, nearest=False)
            ctx.paste(base, cap, X0 + cap.width / 2, IMG[3] + 22 + cap.height / 2 + 20 * (1 - ap), alpha=ap)
            for k, th in enumerate(thumbs):
                v = ctx.ease((u - 0.12 - 0.08 * k) / 0.3)
                if v > 0:
                    ctx.paste(base, th, X0 + k * (TW + TGAP) + TW / 2, THUMB_Y + TH / 2 + 26 * (1 - v), alpha=v)
                else:
                    skeleton(d, local, [(X0 + k * (TW + TGAP), THUMB_Y, X0 + k * (TW + TGAP) + TW, THUMB_Y + TH)])
            # more results below the fold (under the Shorts UI) so the page doesn't end
            skeleton(d, local, [(X0 + k * (TW + TGAP), THUMB_Y + TH + TGAP, X0 + k * (TW + TGAP) + TW,
                                 THUMB_Y + 2 * TH + TGAP) for k in range(3)])
            # cursor glides in over the result, clicks it at the reveal
            wx = 760 + 60 * math.sin(local * 1.6) + 25 * math.sin(local * 3.3)
            wy = 1150 + 50 * math.sin(local * 1.2)
            gx, gy = 600 + 30 * math.sin(local * 2.1), 760 + 24 * math.cos(local * 1.7)
            g = ctx.ease((local - (t_click - 0.7)) / 0.6)
            mx, my = wx + (gx - wx) * g, wy + (gy - wy) * g
            press = max(0.0, 1 - abs(local - t_click) / 0.1)
            if sel and local - t_click < 0.45:
                r = 12 + 150 * (local - t_click)
                a = int(160 * (1 - (local - t_click) / 0.45))
                rg = Image.new("RGBA", (int(2 * r + 8),) * 2, (0, 0, 0, 0))
                ImageDraw.Draw(rg).ellipse([4, 4, 2 * r + 4, 2 * r + 4], outline=(255, 255, 255, a), width=6)
                ctx.paste(base, rg, mx, my)
            base.alpha_composite(cursor(press), (int(mx - 3), int(my - 3)))
        # autocomplete suggestions under the field while typing (drawn over the loading skeleton)
        if 0.15 < local < t_enter and typed.strip():
            yr = ctx.year_of(i)
            sfxs = (" photo", " meeting", f" {yr}" if yr else " history")
            dy0 = BOX[3] + 6
            d.rounded_rectangle([BOX[0], dy0, BOX[2], dy0 + 24 + 66 * len(sfxs)], 22, fill=(255, 255, 255),
                                outline=(218, 220, 224), width=2)
            sf = F(30)
            for k, sfx in enumerate(sfxs):
                ry = dy0 + 46 + 66 * k
                d.ellipse([BOX[0] + 34, ry - 12, BOX[0] + 54, ry + 8], outline=(154, 160, 166), width=3)
                d.line([BOX[0] + 51, ry + 5, BOX[0] + 59, ry + 13], fill=(154, 160, 166), width=4)
                # the suggestion completes what's typed: A's name first, then the whole query
                full = (label(an) if len(typed) <= len(label(an)) + 1 else pa + pb) + sfx
                dark, cut = len(typed.rstrip()), 0
                while sf.getlength(("…" if cut else "") + full[cut:]) > BOX[2] - BOX[0] - 130:
                    cut += 1
                pre = ("…" if cut else "") + full[cut:max(cut, dark)]
                d.text((BOX[0] + 84, ry), pre, font=sf, fill=INKT, anchor="lm")
                d.text((BOX[0] + 84 + sf.getlength(pre), ry), full[max(cut, dark):], font=sf, fill=(130, 134, 140),
                       anchor="lm")
        ctx.header(base, i)
        ctx.connected(base, seg, tt)
        seg["cap"].draw(base, tt, y=1400)
        return base

    return frame
