"""Domino: every photo is a big domino tile standing on a long table. At each link the last tile tips over and slams
down, and the slam knocks the next tile up out of the table and into view; the camera tracks down the line and
Buh rides on top, hopping from tile to tile. Heavy: accelerating falls, a thud with camera shake and dust, an
overshoot-and-wobble landing. The wall scrolls behind at half speed and dust motes drift in the lamp light.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

NAME = "domino"

DW, DH = 560, 860          # tile size
PAD = 30                   # transparent margin around the tile image (outline/shadow room)
FLOOR = 1290               # table top (screen y)
SPACING = DW + DH + 60     # a fallen tile lies flat across DH, then a gap, then the next tile
FALL = 0.5                 # s for the last tile to tip over and slam
RISE = 0.42                # s for the new tile to shoot up after the slam
CAM = 0.95                 # s for the camera to settle on the new tile
PH_Y, PH_H = 28, 450       # photo inside the tile
TAG_A, TAG_B = 580, 668    # tag centres, tile-local y
CAP_Y = 1372


def make(ctx):
    INK = ctx.INK
    TX = ctx.CX - DW / 2   # screen x of a settled tile's left edge
    _cache = {}

    # ---------------------------------------------------------------- tiles
    def tile(i, tags):
        key = ("tile", i, tags)
        if key not in _cache:
            im = Image.new("RGBA", (DW + 2 * PAD, DH + 2 * PAD), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            x0, y0, x1, y1 = PAD, PAD, PAD + DW, PAD + DH
            d.rounded_rectangle([x0 - 6, y0 - 6, x1 + 6, y1 + 6], 46, fill=INK)
            d.rounded_rectangle([x0, y0, x1, y1], 40, fill=(244, 238, 222))
            d.rounded_rectangle([x0 + 8, y0 + 8, x1 - 8, y1 - 8], 34, outline=(255, 252, 244), width=4)
            d.rounded_rectangle([x0 + 10, y1 - 40, x1 - 10, y1 - 8], 20, fill=(214, 204, 184))   # thickness/shade
            ph = ctx.view(i, *ctx.both_cam(i, DW - 56, PH_H, 0.8), DW - 56, PH_H)
            m = Image.new("L", ph.size, 0)
            ImageDraw.Draw(m).rounded_rectangle([0, 0, ph.width - 1, ph.height - 1], 26, fill=255)
            d.rounded_rectangle([x0 + 22, y0 + PH_Y - 6, x1 - 22, y0 + PH_Y + PH_H + 6], 30, fill=INK)
            im.paste(ph, (x0 + 28, y0 + PH_Y), m)
            dy = y0 + PH_Y + PH_H + 34                   # the divider bar and its brass spinner
            d.rectangle([x0 + 30, dy - 7, x1 - 30, dy + 7], fill=INK)
            d.ellipse([x0 + DW / 2 - 20, dy - 20, x0 + DW / 2 + 20, dy + 20], fill=(212, 160, 60), outline=INK, width=5)
            d.ellipse([x0 + DW / 2 - 7, dy - 11, x0 + DW / 2 + 1, dy - 3], fill=(255, 230, 150))
            n = i + 1                                    # a row of pips counts the link
            for k in range(n):
                px = x0 + DW / 2 + (k - (n - 1) / 2) * 56
                py = y0 + 790
                d.ellipse([px - 17, py - 17, px + 17, py + 17], fill=INK)
                d.ellipse([px - 9, py - 11, px - 1, py - 3], fill=(90, 80, 100))
            if tags:
                an, bn = ctx.link_names(i)
                for nm, col, ty in ((an, ctx.CYAN, TAG_A), (bn, ctx.YELLOW, TAG_B)):
                    t = ctx.pixel_tag(ctx.C.labels[nm].upper(), col, 30)
                    im.alpha_composite(t, (int(x0 + DW / 2 - t.width / 2), int(y0 + ty - t.height / 2)))
            _cache[key] = im
        return _cache[key]

    def paste_rot(base, im, pivot_local, pivot_screen, deg):
        """Paste tile image rotated `deg` clockwise about pivot_local (tile px, PAD included) placed at pivot_screen."""
        if abs(deg) < 0.05:
            base.alpha_composite(im, (int(pivot_screen[0] - pivot_local[0]), int(pivot_screen[1] - pivot_local[1])))
            return
        r = im.rotate(-deg, resample=Image.BICUBIC, expand=True)
        th = math.radians(deg)
        cx, cy = im.width / 2, im.height / 2
        dx, dy = pivot_local[0] - cx, pivot_local[1] - cy
        rx, ry = dx * math.cos(th) - dy * math.sin(th), dx * math.sin(th) + dy * math.cos(th)
        px, py = r.width / 2 + rx, r.height / 2 + ry
        base.alpha_composite(r, (int(pivot_screen[0] - px), int(pivot_screen[1] - py)))

    def rot_pt(local, pivot_local, pivot_screen, deg):
        th = math.radians(deg)
        dx, dy = local[0] - pivot_local[0], local[1] - pivot_local[1]
        return (pivot_screen[0] + dx * math.cos(th) - dy * math.sin(th),
                pivot_screen[1] + dx * math.sin(th) + dy * math.cos(th))

    # ---------------------------------------------------------------- room
    WT = 160                                             # wallpaper repeat

    def wall():
        if "wall" not in _cache:
            im = Image.new("RGB", (ctx.W + WT, FLOOR), (58, 30, 44))
            d = ImageDraw.Draw(im)
            for x in range(0, ctx.W + WT, WT):
                d.rectangle([x, 0, x + 6, FLOOR], fill=(74, 40, 56))
                for y in range(40, FLOOR, WT):
                    d.polygon([(x + WT / 2, y - 26), (x + WT / 2 + 18, y), (x + WT / 2, y + 26), (x + WT / 2 - 18, y)],
                              fill=(96, 52, 70))
                    d.rectangle([x + WT / 2 - 4, y + 70, x + WT / 2 + 4, y + 78], fill=(120, 74, 80))
            d.rectangle([0, FLOOR - 160, ctx.W + WT, FLOOR - 150], fill=(34, 18, 28))   # dado rail
            d.rectangle([0, FLOOR - 150, ctx.W + WT, FLOOR], fill=(44, 24, 36))
            _cache["wall"] = im
        return _cache["wall"]

    FT = 240

    def table():
        if "table" not in _cache:
            h = ctx.H - FLOOR
            im = Image.new("RGB", (ctx.W + FT, h), (92, 50, 26))
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, ctx.W + FT, 46], fill=(176, 104, 52))                  # table top
            d.line([(0, 2), (ctx.W + FT, 2)], fill=(226, 156, 92), width=4)
            for x in range(0, ctx.W + FT, FT):
                d.line([(x, 0), (x, 46)], fill=(120, 66, 30), width=4)
                d.line([(x + 60, 16), (x + 150, 16)], fill=(150, 86, 40), width=3)
                d.line([(x + 30, 32), (x + 110, 32)], fill=(150, 86, 40), width=3)
            d.rectangle([0, 46, ctx.W + FT, 66], fill=(60, 30, 16))                  # front edge
            for y in range(100, h, 70):
                d.line([(0, y), (ctx.W + FT, y)], fill=(80, 42, 22), width=3)
            _cache["table"] = im
        return _cache["table"]

    def light():
        if "light" not in _cache:
            yy, xx = np.mgrid[0:ctx.H, 0:ctx.W].astype(np.float32)
            r = np.sqrt(((xx - ctx.CX) / 620) ** 2 + ((yy - 820) / 900) ** 2)
            v = np.clip(1.25 - 0.85 * r, 0.32, 1.0)
            _cache["light"] = Image.fromarray((v * 255).astype(np.uint8)).convert("RGB")
        return _cache["light"]

    def room(camx, sh):
        bg = Image.new("RGB", (ctx.W, ctx.H))
        o = int(camx * 0.45) % WT
        bg.paste(wall().crop((o, 0, o + ctx.W, FLOOR)), (0, int(sh[1] * 0.5)))
        o = int(camx) % FT
        bg.paste(table().crop((o, 0, o + ctx.W, ctx.H - FLOOR)), (0, FLOOR + int(sh[1])))
        return ImageChops.multiply(bg, light()).convert("RGBA")

    _motes = np.random.default_rng(4).random((40, 4))
    _dust = np.random.default_rng(9).random((26, 3))

    def motes(base, tt, camx):
        d = ImageDraw.Draw(base)
        for a, b, c, e in _motes:
            x = (a * ctx.W * 1.3 - camx * 0.2 + 20 * math.sin(tt * (0.5 + c) + e * 6)) % (ctx.W + 40) - 20
            y = (b * FLOOR - tt * (14 + 30 * c)) % FLOOR
            s = 2 + int(3 * c)
            d.rectangle([x, y, x + s, y + s], fill=(255, 226, 170, int(70 + 120 * e)))

    def shadow(base, x0, x1, alpha):
        if x1 - x0 < 4:
            return
        key = ("shadow", int(x1 - x0) // 8)
        if key not in _cache:
            w = max(8, int(x1 - x0) // 8 * 8)
            im = Image.new("RGBA", (w + 80, 70), (0, 0, 0, 0))
            ImageDraw.Draw(im).ellipse([40, 20, w + 40, 50], fill=(0, 0, 0, 150))
            _cache[key] = im.filter(ImageFilter.GaussianBlur(12))
        ctx.paste(base, _cache[key], (x0 + x1) / 2, FLOOR + 6, alpha=alpha)

    # ---------------------------------------------------------------- motion
    def wx(i):                                          # world x of tile i's left edge
        return i * SPACING

    def fall_deg(local):
        u = min(1.0, local / FALL)
        return 90 * u * u                                # gravity: slow start, fast finish

    def rise(local):
        """(height offset below standing, wobble deg) for the new tile; it shoots up after the slam."""
        k = local - FALL
        if k < 0:
            return DH + 2 * PAD + 40, 0.0
        if k < RISE:
            u = k / RISE
            return (1 - ctx.ease_out_back(u)) * (DH + 2 * PAD + 40), 0.0
        w = k - RISE
        return 0.0, 5.5 * math.exp(-w * 3.2) * math.sin(w * 15)

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        sp, R = seg["speech"], seg.get("rv", ctx.REVEAL)
        coin_t = R * sp + ctx.THROW
        last = i == ctx.n_links - 1
        first = i == 0
        lt = local + (FALL if first else 0.0)            # link 1: no tile to knock over, it just shoots up

        # camera: from the last tile to this one, eased; never still afterwards
        u = ctx.ease(min(1.0, local / CAM)) if not first else 1.0
        camx = wx(i - 1) + (wx(i) - wx(i - 1)) * u - TX
        camx += 18 * math.sin(tt * 0.8)
        sh = ctx.shake(lt - FALL, 26, 0.45)
        sh2 = ctx.shake(local - coin_t, 10, 0.3)
        sh = (sh[0] + sh2[0], sh[1] + sh2[1])
        camx -= sh[0]
        cy = sh[1]
        base = room(camx, sh)
        motes(base, tt, camx)

        def sx(w):
            return w - camx

        # the previous tile tipping over (and, once down, lying on the table)
        if not first and not ctx.C.fj["links"][i - 1].get("lineage"):
            deg = fall_deg(local)
            pl = (PAD + DW, PAD + DH)
            ps = (sx(wx(i - 1) + DW), FLOOR + cy)
            if deg >= 90:                                # bounce a touch on the slam
                deg = 90 - 4 * math.exp(-(local - FALL) * 9) * abs(math.sin((local - FALL) * 20))
            x_tip = ps[0] + DH * math.sin(math.radians(deg))
            shadow(base, ps[0] - DW * math.cos(math.radians(deg)), x_tip, 0.7)
            paste_rot(base, tile(i - 1, True), pl, ps, deg)

        # this tile: under the table until the slam, then up, overshoot, wobble
        down, wob = rise(lt)
        if lt > FALL:
            persistent = 0.8 * math.sin(tt * 1.7)        # the lamp-lit tile never sits dead still
            deg = wob + (persistent if lt - FALL > RISE else 0)
            pl = (PAD + DW / 2, PAD + DH)
            ps = (sx(wx(i) + DW / 2), FLOOR + cy + down)
            if down < 5:
                shadow(base, ps[0] - DW / 2, ps[0] + DW / 2, 0.8)
            layer = Image.new("RGBA", base.size, (0, 0, 0, 0)) if down > 0 else base
            paste_rot(layer, tile(i, False), pl, ps, deg)
            if down > 0:                                 # clip to above the table top
                layer = layer.crop((0, 0, ctx.W, int(FLOOR + cy)))
                base.alpha_composite(layer)
                ImageDraw.Draw(base).rectangle([0, FLOOR + cy, ctx.W, FLOOR + cy + 6], fill=(226, 156, 92))
            # dust burst where the old tile slammed
            k = lt - FALL
            if 0 <= k < 0.7 and not first:
                d = ImageDraw.Draw(base)
                cxs = sx(wx(i - 1) + DW + DH * 0.92)
                fade = 1 - k / 0.7
                for a_, (r0, r1, r2) in enumerate(_dust):
                    ang = math.pi * r0
                    rr = (60 + 300 * r1) * (1 - math.exp(-k * 7))
                    px = cxs + rr * math.cos(ang) * 1.4
                    py = FLOOR + cy - 10 - rr * math.sin(ang) * 0.5 - 90 * k * r2
                    s = int((6 + 16 * r2) * (0.4 + fade))
                    g = 150 + int(70 * r2)
                    d.ellipse([px - s, py - s, px + s, py + s], fill=(g, g - 14, g - 34, int(200 * fade)))

            # tags ride on the tile
            def at(ly):
                return rot_pt((PAD + DW / 2, PAD + ly), pl, ps, deg)
            an, bn = ctx.link_names(i)
            ax, ay = at(TAG_A)
            if down <= 0:
                ctx.paste(base, ctx.pixel_tag(ctx.C.labels[an].upper(), ctx.CYAN, 30), ax, ay)
            if local > coin_t and down <= 0:
                pop = ctx.ease_out_back(min(1.0, (local - coin_t) / 0.18))
                bx_, by_ = at(TAG_B)
                ctx.paste(base, ctx.pixel_tag(ctx.C.labels[bn].upper(), ctx.YELLOW, 30), bx_, by_ - 60 * (1 - pop),
                          scale=1.6 - 0.6 * pop)
            top = at(0)
        else:
            top = None

        # Buh rides the dominoes: on the old tile's top as it tips, a big leap, then onto the new one
        ptop = None
        if not first:
            deg0 = fall_deg(min(local, 0.31))            # Buh rides the tipping tile, then leaps off it
            ptop = rot_pt((PAD + DW / 2, PAD), (PAD + DW, PAD + DH), (sx(wx(i - 1) + DW), FLOOR + cy), deg0)
        jump0, jump1 = 0.31, FALL + RISE * 0.8
        if top is not None and lt >= jump1:
            bxy, expr = top, ("shock" if coin_t - 0.05 < local < coin_t + 0.5 else ("happy" if local > coin_t else "smug"))
            hop = abs(math.sin(local * 3.0)) * 14
            if last and local > coin_t + 0.05:
                hop = abs(math.sin(local * 8)) * 70
                expr = "happy"
            spr = ctx.pin_sprite(expr, 0, 0.3 * max(0.0, 1 - (lt - jump1) / 0.15), 1, scale=4)
            ctx.paste_pin(base, spr, bxy[0], bxy[1] - hop)
        elif not first and local >= jump0:
            v = (local - jump0) / (jump1 - jump0)
            tgt = (sx(wx(i) + DW / 2), FLOOR + cy - DH)
            x = ptop[0] + (tgt[0] - ptop[0]) * v
            y = ptop[1] + (tgt[1] - ptop[1]) * math.sin(v * math.pi / 2) - 40 * math.sin(math.pi * v)
            ctx.paste_pin(base, ctx.pin_sprite("shock", -30 * (1 - v), 0, 1, scale=4), x, y)
        elif first and top is not None:                  # link 1: Buh drops in from above as the tile lands
            v = min(1.0, max(0.0, (lt - FALL) / (jump1 - FALL)))
            ctx.paste_pin(base, ctx.pin_sprite("shock", 0, 0, 1, scale=4), top[0], top[1] - 700 * (1 - v * v))
        elif ptop is not None:
            ctx.paste_pin(base, ctx.pin_sprite("shock" if local > 0.1 else "smug", -deg0, 0, 1, scale=4), ptop[0], ptop[1])

        ctx.header(base, i)
        if last:                                         # CONNECTED! slams on the table edge, under the tags
            ov = Image.new("RGBA", (ctx.W, ctx.H), (0, 0, 0, 0))
            ctx.connected(ov, seg, tt)
            base.alpha_composite(ov.crop((0, 0, ctx.W, ctx.H - 215)), (0, 215))
        seg["cap"].draw(base, tt, y=CAP_Y)
        return base

    return frame
