"""Police lineup: every person in the chain steps into a lineup in front of a height chart, holding a placard with
their name and year; the link photo pops up below as taped EVIDENCE; Buh inspects each suspect with a magnifying
glass and hops from A to B on the reveal."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

NAME = "lineup"


def make(ctx):
    SLOT = 280                              # distance between suspects
    HEAD_Y = 430                            # face centre
    FLOOR_Y = 880                           # lineup floor / counter top
    PLAC_Y = 668                            # placard centre
    EV_Y = 1118                             # evidence print centre
    EV_W, EV_H = 600, 380
    CAP_Y = 1410
    SPR_W, SPR_H, HEAD_C = 260, 560, 110    # suspect sprite and head centre inside it
    PAN = 0.6                               # camera slides one slot at the start of a link
    WALK = (0.12, 0.95)                     # B walks in during this window
    n = ctx.n_links

    def font(name, size):
        return ImageFont.truetype(str(ctx.FONTS / f"Inter-{name}.otf"), size)

    def person(s):
        """Suspect s: 0 is the chain's start (A of link 0), s >= 1 is B of link s - 1. -> (name, link, role)."""
        if s == 0:
            return ctx.link_names(0)[0], 0, "a"
        return ctx.link_names(s - 1)[1], s - 1, "b"

    def year_for(s):
        for j in range(max(0, s - 1), -1, -1):
            y = ctx.year_of(j)
            if y:
                return y
        return None

    # ---------- static background: height-chart wall, counter, evidence desk
    def background():
        im = Image.new("RGBA", (ctx.W, ctx.H), (0, 0, 0, 255))
        a = np.zeros((ctx.H, ctx.W, 3), np.float32)
        yy, xx = np.mgrid[0:ctx.H, 0:ctx.W].astype(np.float32)
        wall = np.array([86, 96, 108], np.float32)
        lightpool = 0.8 + 0.35 * np.exp(-((xx - ctx.CX) / 520) ** 2 - ((yy - 420) / 520) ** 2)
        a[:FLOOR_Y] = wall * lightpool[:FLOOR_Y, :, None]
        desk = np.array([34, 38, 36], np.float32)
        grain = 1 + 0.05 * np.sin(yy / 3.0 + np.sin(xx / 40) * 3)
        deskpool = 0.6 + 0.7 * np.exp(-((xx - ctx.CX) / 600) ** 2 - ((yy - EV_Y) / 420) ** 2)
        a[FLOOR_Y:] = desk * (grain * deskpool)[FLOOR_Y:, :, None]
        im = Image.fromarray(a.clip(0, 255).astype(np.uint8)).convert("RGBA")
        d = ImageDraw.Draw(im)
        ppf = 88                                         # pixels per foot on the chart
        for k in range(0, 40):
            y = FLOOR_Y - k * ppf / 2
            if y < 236:
                break
            major = k % 2 == 0
            d.line([(0, y), (ctx.W, y)], fill=(236, 238, 240) if major else (170, 180, 190), width=4 if major else 2)
            if major and k:
                for x in (ctx.SAFE[0] + 6, ctx.SAFE[2] - 64):
                    d.text((x, y - 34), f"{k // 2}'", font=font("Black", 30), fill=(236, 238, 240))
            elif k:
                for x in (ctx.SAFE[0] + 6, ctx.SAFE[2] - 64):
                    d.text((x, y - 24), "6\"", font=font("Bold", 18), fill=(190, 198, 206))
        d.rectangle([0, FLOOR_Y, ctx.W, FLOOR_Y + 44], fill=(28, 28, 32))
        d.rectangle([0, FLOOR_Y, ctx.W, FLOOR_Y + 5], fill=(120, 124, 130))
        d.rectangle([0, FLOOR_Y + 44, ctx.W, FLOOR_Y + 52], fill=(14, 14, 16))
        return im

    BG = background()

    # ---------- suspects
    _faces, _sprites, _plac = {}, {}, {}

    def face(s):
        if s in _faces:
            return _faces[s]
        nm, j, role = person(s)
        fw, fh = 168, 210
        try:
            if ctx.C.fj["links"][j].get("lineage"):
                raise ValueError
            b = ctx.fbox(j, role)
            cx, cy = ctx.ctr(b)
            im = ctx.view(j, cx, cy - b[3] * 0.08, fh * 0.56 / max(1, b[3]), fw, fh)
        except Exception:  # noqa: BLE001 - lineage people: the face thumbnail
            t = ctx.sharp[nm]
            im = t.resize((fh, fh), Image.BICUBIC).crop(((fh - fw) // 2, 0, (fh - fw) // 2 + fw, fh))
        _faces[s] = im.convert("RGBA")
        return _faces[s]

    def sprite(s):
        if s in _sprites:
            return _sprites[s]
        im = Image.new("RGBA", (SPR_W, SPR_H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        f = face(s)
        skin = tuple(int(v) for v in np.asarray(f.convert("RGB"))[150:200, 60:110].reshape(-1, 3).mean(0) * 0.8)
        d.rectangle([SPR_W / 2 - 30, HEAD_C + 60, SPR_W / 2 + 30, HEAD_C + 120], fill=skin)            # neck
        d.rounded_rectangle([8, 212, SPR_W - 8, SPR_H + 80], 80, fill=(226, 110, 36), outline=(40, 24, 16), width=4)
        d.rounded_rectangle([SPR_W / 2 + 20, 222, SPR_W - 14, SPR_H + 80], 70, fill=(200, 92, 28))     # shading
        d.polygon([(SPR_W / 2 - 34, 214), (SPR_W / 2 + 34, 214), (SPR_W / 2, 268)], fill=(236, 236, 230),
                  outline=(40, 24, 16))
        d.line([(SPR_W / 2, 268), (SPR_W / 2, SPR_H)], fill=(170, 80, 24), width=3)                 # zip
        mask = Image.new("L", f.size, 0)
        ImageDraw.Draw(mask).ellipse([2, 2, f.width - 3, f.height - 3], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(1.2))
        head = f.copy()
        head.putalpha(mask)
        hx, hy = int(SPR_W / 2 - f.width / 2), int(HEAD_C - f.height / 2)
        d.ellipse([hx - 4, hy - 4, hx + f.width + 3, hy + f.height + 3], fill=(30, 22, 18))
        im.alpha_composite(head, (hx, hy))
        _sprites[s] = im
        return im

    _dim = {}

    def dimmed(s):
        if s not in _dim:
            r, g, b, al = sprite(s).split()
            r, g, b = (c.point(lambda v: int(v * 0.72)) for c in (r, g, b))
            _dim[s] = Image.merge("RGBA", (r, g, b, al))
        return _dim[s]

    def placard(s, revealed):
        key = (s, revealed)
        if key in _plac:
            return _plac[key]
        PWd, PHt = 250, 136
        im = Image.new("RGBA", (PWd + 60, PHt + 20), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        ox, oy = 30, 6
        d.rectangle([ox + 6, oy + 8, ox + PWd + 6, oy + PHt + 8], fill=(0, 0, 0, 120))
        d.rectangle([ox, oy, ox + PWd, oy + PHt], fill=(22, 22, 26), outline=(235, 235, 235), width=4)
        d.text((ox + PWd / 2, oy + 10), f"BUH P.D.  ·  No. {s + 1}", font=font("Bold", 16), fill=(200, 200, 200),
               anchor="ma")
        nm = ctx.C.labels[person(s)[0]].upper() if revealed else "? ? ?"
        words, lines = nm.split(), []
        size = 32
        while size > 16:
            f = font("Black", size)
            if f.getlength(nm) <= PWd - 24:
                lines = [nm]
                break
            # two lines, split near the middle
            best = None
            for k in range(1, len(words)):
                l1, l2 = " ".join(words[:k]), " ".join(words[k:])
                wmax = max(f.getlength(l1), f.getlength(l2))
                if wmax <= PWd - 24 and (best is None or wmax < best[0]):
                    best = (wmax, [l1, l2])
            if best and size <= 28:
                lines = best[1]
                break
            size -= 2
        f = font("Black", size)
        y = oy + 36 + (0 if len(lines) == 2 else size * 0.5)
        for ln in lines:
            d.text((ox + PWd / 2, y), ln, font=f, fill=(255, 255, 255), anchor="ma")
            y += size + 2
        yr = year_for(s)
        if yr:
            d.text((ox + PWd / 2, oy + PHt - 34), str(yr), font=font("Black", 24), fill=ctx.YELLOW, anchor="ma")
        for hx in (ox - 6, ox + PWd + 6):                                         # sleeves gripping the board
            d.ellipse([hx - 20, oy + PHt / 2 - 22, hx + 20, oy + PHt / 2 + 22], fill=(214, 100, 32),
                      outline=(40, 24, 16), width=3)
        _plac[key] = im
        return im

    _ev = {}

    def evidence(i):
        if i not in _ev:
            ph = ctx.view(i, *ctx.both_cam(i, EV_W, EV_H, 0.8), EV_W, EV_H)
            _ev[i] = ctx.framed(ph, 0.0)
        return _ev[i]

    def stamp(i):
        key = ("stamp", i)
        if key not in _ev:
            txt = f"EVIDENCE #{i + 1}"
            f = font("Black", 46)
            w = int(f.getlength(txt)) + 40
            im = Image.new("RGBA", (w + 20, 100), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            red = (214, 30, 40, 235)
            d.rectangle([10, 10, w + 10, 90], fill=(250, 244, 236, 150), outline=red, width=6)
            d.rectangle([18, 18, w + 2, 82], outline=red, width=2)
            d.text((w / 2 + 10, 50), txt, font=f, fill=red, anchor="mm")
            _ev[key] = im.rotate(8, expand=True, resample=Image.BICUBIC)
        return _ev[key]

    def rot_paste(base, im, cx, cy, ang, sc=1.0, alpha=1.0):
        if sc <= 0.02:
            return
        if abs(sc - 1) > 1e-3:
            im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.BILINEAR)
        if abs(ang) > 0.05:
            im = im.rotate(ang, expand=True, resample=Image.BICUBIC)
        ctx.paste(base, im, cx, cy, alpha=alpha)

    def seg_of(j):
        return next(s for s in ctx.segs if s.get("i") == j)

    def cam_x(i, local):
        """World x at the screen centre: suspect i (link i's A) sits in the middle."""
        return (i - 1 + ctx.ease(min(1.0, local / PAN))) * SLOT if i else 0.0

    def magnifier(base, lx, ly, r=74, zoom=1.55):
        src = base.crop((int(lx - r / zoom), int(ly - r / zoom), int(lx + r / zoom), int(ly + r / zoom)))
        src = src.resize((2 * r, 2 * r), Image.BICUBIC)
        m = Image.new("L", (2 * r, 2 * r), 0)
        ImageDraw.Draw(m).ellipse([0, 0, 2 * r - 1, 2 * r - 1], fill=255)
        tint = Image.new("RGBA", src.size, (210, 235, 255, 40))
        src.alpha_composite(tint)
        src.putalpha(m)
        base.alpha_composite(src, (int(lx - r), int(ly - r)))
        d = ImageDraw.Draw(base)
        d.ellipse([lx - r - 8, ly - r - 8, lx + r + 8, ly + r + 8], outline=(40, 26, 14), width=12)
        d.ellipse([lx - r - 4, ly - r - 4, lx + r + 4, ly + r + 4], outline=(200, 160, 80), width=4)
        d.arc([lx - r * 0.7, ly - r * 0.7, lx + r * 0.7, ly + r * 0.7], 200, 250, fill=(255, 255, 255), width=6)

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        sp = seg["speech"]
        R = seg.get("rv", ctx.REVEAL) * sp                 # Buh leaves A
        RV = R + ctx.THROW                                 # ... and lands on B: B's name shows
        base = BG.copy()
        cx = cam_x(i, local) + 6 * math.sin(tt * 0.7)
        flick = 1.0 + 0.02 * math.sin(tt * 23) * (math.sin(tt * 1.7) > 0.6)

        def sx(s):
            return ctx.CX + s * SLOT - cx

        d = ImageDraw.Draw(base)
        for s in range(0, i + 2):                          # slot numbers on the wall
            x = sx(s)
            if -60 < x < ctx.W + 60:
                d.text((x, 262), str(s + 1), font=font("Black", 44), fill=(236, 238, 240), anchor="ma")
        # suspects: everyone so far stands still (breathing); B walks in from the right
        for s in range(max(0, i - 2), i + 2):
            x, y, ang, alpha = sx(s), 0.0, 0.0, 1.0
            if s == i + 1:
                u = (local - WALK[0]) / (WALK[1] - WALK[0])
                if u <= 0:
                    continue
                e = 1 - (1 - min(1.0, u)) ** 2
                x += (1 - e) * 640
                if u < 1:
                    y = -abs(math.sin(u * math.pi * 4)) * 16
                    ang = 4 * math.sin(u * math.pi * 4)
            if not (-SPR_W < x < ctx.W + SPR_W):
                continue
            y += 3 * math.sin(tt * 2.1 + s * 1.3)
            ang += 0.8 * math.sin(tt * 1.3 + s)
            spr = sprite(s)
            if s < i:                                      # earlier suspects step back into the shade
                spr = dimmed(s)
            rot_paste(base, spr, x, FLOOR_Y - SPR_H / 2 + y, ang)
            if s < i - 1:                                  # leaving the frame: no placard
                continue
            pl = placard(s, s <= i or local > RV)
            rot_paste(base, pl, x, PLAC_Y + y * 0.6, ang * 0.5 + 1.2 * math.sin(tt * 1.7 + s), alpha=alpha)
        if flick != 1.0:
            base.alpha_composite(Image.new("RGBA", (ctx.W, FLOOR_Y), (0, 0, 0, 14)), (0, 0))
        # the previous evidence slides away, the new print pops up
        if i > 0 and local < 0.45 and not ctx.C.fj["links"][i - 1].get("lineage"):
            v = ctx.ease(local / 0.45)
            rot_paste(base, evidence(i - 1), ctx.CX - 700 * v, EV_Y + 260 * v, (-1) ** (i - 1) * 2.5 + 25 * v)
        pop = (local - 0.3) / 0.35
        if pop > 0:
            k = ctx.ease_out_back(min(1.0, pop))
            ang = (-1) ** i * 2.5 + (1 - min(1.0, pop)) * -18 + 1.0 * math.sin(tt * 1.1)
            rot_paste(base, evidence(i), ctx.CX, EV_Y, ang, sc=max(0.02, k) * (1 + 0.012 * max(0.0, local - 0.65)))
        st = local - 0.75
        if st > 0:
            k = min(1.0, st / 0.14)
            dx, dy = ctx.shake(st - 0.14, 10, 0.25)
            ctx.paste(base, stamp(i), ctx.CX + 140 + dx, EV_Y + 178 + dy, scale=2.0 - ctx.ease(k), alpha=0.4 + 0.6 * k,
                      nearest=False)
        # Buh: floats beside the suspect's head, glass on their face; hops from A to B on the reveal
        ax, bx = sx(i) - 132, sx(i + 1) - 132
        tip = HEAD_Y + 132
        if local < R:
            px, py, spin, expr, look = ax, tip + 8 * math.sin(tt * 3.2), 0.0, "smug", 1
            if i == 0 and local < 0.5:                       # first link: Buh drops in from above
                py -= (1 - ctx.ease_out_back(local / 0.5)) * 500
        elif local < RV:
            k = (local - R) / ctx.THROW
            px = ax + (bx - ax) * k
            py = tip - 220 * 4 * k * (1 - k)
            spin, expr, look = 360 * k, "shock", 1
        else:
            k = local - RV
            px, py, spin, look = bx, tip + 8 * math.sin(tt * 3.2), 0.0, 1
            expr = "shock" if k < 0.35 else "happy"
        squash = 0.25 * max(0.0, 1 - (local - RV) / 0.2) if local > RV else 0.0
        lx, ly = px + 132, py - 132
        if local < R or local > RV:
            ImageDraw.Draw(base).line([(px + 40, py - 96), (lx - 58, ly + 58)], fill=(60, 36, 18), width=14)
            magnifier(base, lx, ly + 6 * math.sin(tt * 2.3))
        ctx.paste_pin(base, ctx.pin_sprite(expr, spin, squash, look, scale=4), px, py)
        ctx.header(base, i)
        ctx.connected(base, seg, tt)
        seg["cap"].draw(base, tt, y=CAP_Y)
        return base

    return frame
