"""Domino: every photo is a huge domino tile standing in a line on a games table. At each link the last tile tips
over toward the next one and slams flat in the gap (fast, under gravity, done in ~0.3 s), the thud jolts the new
tile into a wobble and kicks up dust, and the camera whip-pans down the line while Buh leaps from the falling tile's
top onto the new one. The fallen tile ends fully out of shot. Name tags sit under their own faces (arrows point up
at them) and are only drawn on a standing tile, never while it tumbles. The first caption of a link waits until the
slam is over.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

NAME = "domino"

DW, DH = 900, 880          # tile size (front view: it tips over in the screen plane)
PAD = 30                   # transparent margin around the tile image (outline/shadow room)
FLOOR = 1320               # table top (screen y); a standing tile's top is at FLOOR - DH
GAP = 170                  # gap between a fallen tile's top and the next tile's foot
SPACING = DW + DH + GAP    # a fallen tile lies flat across DH, then the gap, then the next tile
FALL = 0.3                 # s for the last tile to tip over and slam (gravity: slow start, fast finish)
RISE = 0.42                # s for the first tile to shoot up out of the table
CAM = 0.52                 # s for the camera to whip-pan to the new tile
CAP_HOLD = 0.55            # the link's first caption waits until the slam (and its dust) is over
DUST = 0.25                # s the slam's dust puff lives
ROW_Y, ROW_TH = 1548, 78   # the chain so far: little standing dominoes with faces, along the table front
PH_Y, PH_H = 26, 600       # photo inside the tile (DW - 50 = 850 px wide)
DIV_Y = 656                # divider bar, tile-local y
TAG_Y = 724                # tag centres, tile-local y
PIP_Y = 806                # pips, tile-local y
CAP_Y = 1398


def make(ctx):
    INK = ctx.INK
    TX = ctx.CX - DW / 2   # screen x of a settled tile's left edge
    TOP = FLOOR - DH       # screen y of a settled tile's top
    _cache = {}

    def cam(i):
        cx, cy, sc = ctx.both_cam(i, DW - 50, PH_H, 0.82)
        ih, iw = ctx.C.imgs[i].shape[:2]
        pw = DW - 50
        sc = max(min(sc, 1.3), pw / iw, PH_H / ih)    # never upscale past ~1.3x, but always cover the slot
        cx = max(pw / 2 / sc, min(iw - pw / 2 / sc, cx))   # cover-crop: the view never runs off the photo
        cy = max(PH_H / 2 / sc, min(ih - PH_H / 2 / sc, cy))
        return cx, cy, sc

    # ---------------------------------------------------------------- tiles
    def tile(i, pips=True):
        key = ("tile", i, pips)
        if key not in _cache:
            im = Image.new("RGBA", (DW + 2 * PAD, DH + 2 * PAD), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            x0, y0, x1, y1 = PAD, PAD, PAD + DW, PAD + DH
            d.rounded_rectangle([x0 - 7, y0 - 7, x1 + 7, y1 + 7], 52, fill=INK)
            d.rounded_rectangle([x0, y0, x1, y1], 46, fill=(246, 241, 228))
            d.rounded_rectangle([x0 + 8, y0 + 8, x1 - 8, y1 - 8], 40, outline=(255, 253, 246), width=4)
            d.rounded_rectangle([x0 + 10, y1 - 34, x1 - 10, y1 - 8], 20, fill=(216, 206, 186))   # thickness/shade
            ph = ctx.view(i, *cam(i), DW - 50, PH_H)
            m = Image.new("L", ph.size, 0)
            ImageDraw.Draw(m).rounded_rectangle([0, 0, ph.width - 1, ph.height - 1], 30, fill=255)
            d.rounded_rectangle([x0 + 19, y0 + PH_Y - 6, x1 - 19, y0 + PH_Y + PH_H + 6], 34, fill=INK)
            im.paste(ph, (x0 + 25, y0 + PH_Y), m)
            dy = y0 + DIV_Y                              # the divider bar and its brass spinner
            d.rectangle([x0 + 34, dy - 6, x1 - 34, dy + 6], fill=INK)
            d.ellipse([x0 + DW / 2 - 20, dy - 20, x0 + DW / 2 + 20, dy + 20], fill=(212, 160, 60), outline=INK, width=5)
            d.ellipse([x0 + DW / 2 - 7, dy - 11, x0 + DW / 2 + 1, dy - 3], fill=(255, 230, 150))
            if pips:
                n = i + 1                                # a row of pips counts the link
                for k in range(n):
                    px = x0 + DW / 2 + (k - (n - 1) / 2) * 64
                    py = y0 + PIP_Y
                    d.ellipse([px - 20, py - 20, px + 20, py + 20], fill=INK)
                    d.ellipse([px - 11, py - 13, px - 2, py - 4], fill=(90, 80, 100))
            _cache[key] = im
        return _cache[key]

    def face_x(i, role):
        """Tile-local x (from the tile's left edge) of a face's centre in the tile photo."""
        cx, _, sc = cam(i)
        b = ctx.fbox(i, role)
        return 25 + (DW - 50) / 2 + (b[0] + b[2] / 2 - cx) * sc

    def tag_layout(i):
        """(size, x_a, x_b, face_a, face_b), tile-local x: both tags on one row, each as close under its own face as
        the row allows. Laid out with both names from the start so A's tag never moves when B's appears."""
        if ("tags", i) in _cache:
            return _cache[("tags", i)]
        an, bn = ctx.link_names(i)
        fa, fb = face_x(i, "a"), face_x(i, "b")
        lo, hi = 40, DW - 40
        for size in (30, 28, 26, 24, 22, 20, 18):
            wa = ctx.pixel_tag(ctx.C.labels[an].upper(), ctx.CYAN, size).width
            wb = ctx.pixel_tag(ctx.C.labels[bn].upper(), ctx.YELLOW, size).width
            if wa + wb + 20 <= hi - lo:
                break
        (f1, w1, r1), (f2, w2, r2) = sorted([(fa, wa, "a"), (fb, wb, "b")])
        x1 = min(max(f1, lo + w1 / 2), hi - w2 - 20 - w1 / 2)
        x2 = min(max(f2, x1 + w1 / 2 + 20 + w2 / 2), hi - w2 / 2)
        if x2 - w2 / 2 < x1 + w1 / 2 + 20:
            x1 = x2 - w2 / 2 - 20 - w1 / 2
        pos = {r1: x1, r2: x2}
        _cache[("tags", i)] = (size, pos["a"], pos["b"], fa, fb)
        return _cache[("tags", i)]

    def draw_tag(base, text, bg, size, x, fx, y, scale=1.0):
        """A pixel tag with a little arrow on top pointing up at its face."""
        if scale <= 0.01:
            return
        tg = ctx.pixel_tag(text, bg, size)
        ax = min(max(fx, x - tg.width / 2 + 26), x + tg.width / 2 - 34)
        top = y - tg.height / 2 * scale
        s_ = 18 * scale
        d = ImageDraw.Draw(base)
        d.polygon([(ax - s_ - 4, top + 4), (ax + s_ + 4, top + 4), (ax, top - s_ - 6)], fill=INK)
        d.polygon([(ax - s_, top + 4), (ax + s_, top + 4), (ax, top - s_)], fill=bg)
        ctx.paste(base, tg, x, y, scale=scale)

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
        ox, oy = int(pivot_screen[0] - px), int(pivot_screen[1] - py)
        if ox > ctx.W or oy > ctx.H or ox + r.width < 0 or oy + r.height < 0:
            return
        base.alpha_composite(r, (ox, oy)) if ox >= 0 and oy >= 0 else _clip_comp(base, r, ox, oy)

    def _clip_comp(base, im, ox, oy):
        cx0, cy0 = max(0, -ox), max(0, -oy)
        base.alpha_composite(im.crop((cx0, cy0, im.width, im.height)), (max(0, ox), max(0, oy)))

    def rot_pt(local, pivot_local, pivot_screen, deg):
        th = math.radians(deg)
        dx, dy = local[0] - pivot_local[0], local[1] - pivot_local[1]
        return (pivot_screen[0] + dx * math.cos(th) - dy * math.sin(th),
                pivot_screen[1] + dx * math.sin(th) + dy * math.cos(th))

    # ---------------------------------------------------------------- room: a green-baize games room
    WT = 180                                             # wall panel repeat

    def wall():
        if "wall" not in _cache:
            im = Image.new("RGB", (ctx.W + WT, FLOOR), (26, 70, 74))
            d = ImageDraw.Draw(im)
            for x in range(0, ctx.W + WT, WT):           # tall wood-trimmed panels
                d.rectangle([x, 0, x + 10, FLOOR], fill=(18, 46, 50))
                d.rectangle([x + 28, 60, x + WT - 18, FLOOR - 200], outline=(44, 100, 104), width=4)
            d.rectangle([0, FLOOR - 170, ctx.W + WT, FLOOR - 158], fill=(120, 76, 40))   # dado rail
            d.rectangle([0, FLOOR - 158, ctx.W + WT, FLOOR], fill=(20, 52, 56))
            _cache["wall"] = im
        return _cache["wall"]

    FT = 240

    def table():
        if "table" not in _cache:
            h = ctx.H - FLOOR
            im = Image.new("RGB", (ctx.W + FT, h), (22, 96, 58))                     # green baize
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, ctx.W + FT, 40], fill=(40, 132, 80))
            d.line([(0, 2), (ctx.W + FT, 2)], fill=(90, 180, 120), width=4)
            r = np.random.default_rng(3)
            for _ in range(500):                         # felt speckle
                x, y = int(r.integers(0, ctx.W + FT)), int(r.integers(0, 40))
                d.point((x, y), fill=(30, 110, 66))
            d.rectangle([0, 40, ctx.W + FT, 70], fill=(110, 62, 28))                  # wooden rail
            d.line([(0, 44), (ctx.W + FT, 44)], fill=(170, 106, 56), width=3)
            d.rectangle([0, 70, ctx.W + FT, h], fill=(64, 34, 18))
            for y in range(110, h, 70):
                d.line([(0, y), (ctx.W + FT, y)], fill=(78, 42, 22), width=3)
            _cache["table"] = im
        return _cache["table"]

    def light():
        if "light" not in _cache:
            yy, xx = np.mgrid[0:ctx.H, 0:ctx.W].astype(np.float32)
            r = np.sqrt(((xx - ctx.CX) / 700) ** 2 + ((yy - 800) / 1000) ** 2)
            v = np.clip(1.3 - 0.8 * r, 0.4, 1.0)
            _cache["light"] = Image.fromarray((v * 255).astype(np.uint8)).convert("RGB")
        return _cache["light"]

    def room(camx, sh):
        bg = Image.new("RGB", (ctx.W, ctx.H))
        o = int(camx * 0.45) % WT
        bg.paste(wall().crop((o, 0, o + ctx.W, FLOOR)), (0, int(sh[1] * 0.5)))
        o = int(camx) % FT
        bg.paste(table().crop((o, 0, o + ctx.W, ctx.H - FLOOR)), (0, FLOOR + int(sh[1])))
        return ImageChops.multiply(bg, light()).convert("RGBA")

    _dust = np.random.default_rng(9).random((30, 3))

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

    def chain_row(base, i, revealed, cy):
        """The chain so far along the table front: a little standing domino per person, face on top ("?" until
        revealed); the ones already passed lie tipped over, the current pair is outlined."""
        people = ctx.C.people
        n = len(people)
        x0, x1 = ctx.SAFE[0] + 50, ctx.SAFE[2] - 50
        step = (x1 - x0) / max(1, n - 1)
        known = i + (2 if revealed else 1)
        tw, th = ROW_TH + 14, ROW_TH * 2 + 20
        for j, nm in enumerate(people):
            key = ("mini", nm, j < known)
            if key not in _cache:
                im = Image.new("RGBA", (tw + 16, th + 16), (0, 0, 0, 0))
                d = ImageDraw.Draw(im)
                d.rounded_rectangle([8, 10, tw + 8, th + 10], 12, fill=(0, 0, 0, 110))
                d.rounded_rectangle([0, 0, tw, th], 12, fill=(246, 241, 228), outline=INK, width=4)
                if j < known:
                    im.alpha_composite(ctx.sharp[nm].resize((ROW_TH, ROW_TH), Image.BICUBIC), (7, 7))
                else:
                    d.rectangle([7, 7, 7 + ROW_TH, 7 + ROW_TH], fill=(40, 60, 64))
                    q = ctx.pixel_text("?", 34, fill=ctx.WHITE, shadow=False)
                    im.alpha_composite(q, (7 + ROW_TH // 2 - q.width // 2, 7 + ROW_TH // 2 - q.height // 2))
                d.rectangle([12, ROW_TH + 12, tw - 12, ROW_TH + 17], fill=INK)
                for k in range(min(j + 1, 4)):
                    px = tw / 2 + (k - (min(j + 1, 4) - 1) / 2) * 18
                    d.ellipse([px - 6, ROW_TH + 40, px + 6, ROW_TH + 52], fill=INK)
                _cache[key] = im
            im = _cache[key]
            x = x0 + j * step
            if j < i:                                    # already toppled: lying tipped toward the next one
                im = im.rotate(-62, resample=Image.BICUBIC, expand=True)
                base.alpha_composite(im, (int(x - im.width / 2 + 30), int(ROW_Y + th / 2 + cy - im.height + 20)))
            else:
                base.alpha_composite(im, (int(x - tw / 2), int(ROW_Y - th / 2 + cy)))
                if j in (i, i + 1):
                    ImageDraw.Draw(base).rounded_rectangle([x - tw / 2 - 4, ROW_Y - th / 2 + cy - 4, x + tw / 2 + 4,
                                                            ROW_Y + th / 2 + cy + 4], 14,
                                                           outline=ctx.CYAN if j == i else ctx.YELLOW, width=5)

    # ---------------------------------------------------------------- motion
    def wx(i):                                          # world x of tile i's left edge
        return i * SPACING

    def fall_deg(local):
        u = min(1.0, max(0.0, local) / FALL)
        return 90 * u * u

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        sp, R = seg["speech"], seg.get("rv", ctx.REVEAL)
        coin_t = R * sp + ctx.THROW
        last = i == ctx.n_links - 1
        first = i == 0 or ctx.C.fj["links"][i - 1].get("lineage")

        # camera: whip-pan from the last tile to this one, eased; never quite still afterwards
        u = 1.0 if first else ctx.ease(min(1.0, local / CAM))
        camx = wx(i - 1) + (wx(i) - wx(i - 1)) * u - TX
        camx += 14 * math.sin(tt * 0.8)
        slam = (RISE if first else FALL)
        sh = ctx.shake(local - slam, 22, 0.4)
        sh2 = ctx.shake(local - coin_t, 8, 0.3)
        sh = (sh[0] + sh2[0], sh[1] + sh2[1])
        camx -= sh[0]
        cy = sh[1]
        base = room(camx, sh)

        def sx(w):
            return w - camx

        # the previous tile tipping over (no tags on it: nothing rotates while it's readable), then lying flat
        if not first:
            deg = fall_deg(local)
            if local >= FALL:                            # bounce a touch on the slam
                deg = 90 - 5 * math.exp(-(local - FALL) * 10) * abs(math.sin((local - FALL) * 22))
            pl = (PAD + DW, PAD + DH)
            ps = (sx(wx(i - 1) + DW), FLOOR + cy)
            x_tip = ps[0] + DH * math.sin(math.radians(deg))
            if ps[0] - DW < ctx.W and x_tip > 0:
                shadow(base, ps[0] - DW * math.cos(math.radians(deg)), x_tip, 0.7)
                paste_rot(base, tile(i - 1), pl, ps, deg)

        # this tile: already standing in the line (link 1: shoots up out of the table); the slam jolts it
        pl = (PAD + DW / 2, PAD + DH)
        down = 0.0
        if first:
            k = min(1.0, local / RISE)
            down = (1 - ctx.ease_out_back(k)) * (DH + 2 * PAD + 40)
            w = local - RISE
        else:
            w = local - FALL
        wob = 3.2 * math.exp(-w * 3.4) * math.sin(w * 15) if w > 0 else 0.0
        deg = wob + 0.5 * math.sin(tt * 1.7) * min(1.0, max(0.0, w) / 0.6)
        ps = (sx(wx(i) + DW / 2), FLOOR + cy + down)
        if down < 5:
            shadow(base, ps[0] - DW / 2, ps[0] + DW / 2, 0.8)
        if down > 0:
            layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
            paste_rot(layer, tile(i, not last), pl, ps, deg)
            base.alpha_composite(layer.crop((0, 0, ctx.W, int(FLOOR + cy))))
        else:
            paste_rot(base, tile(i, not last), pl, ps, deg)

        # dust burst where the old tile slammed
        k = local - FALL
        if not first and 0 <= k < DUST:
            d = ImageDraw.Draw(base)
            cxs = sx(wx(i - 1) + DW + DH * 0.9)
            fade = (1 - k / DUST) ** 1.5
            for r0, r1, r2 in _dust:
                ang = math.pi * r0
                rr = (60 + 340 * r1) * (1 - math.exp(-k * 12))
                px = cxs + rr * math.cos(ang) * 1.4
                py = FLOOR + cy - 10 - rr * math.sin(ang) * 0.5 - 90 * k * r2
                s = int((8 + 18 * r2) * (0.4 + fade))
                g = 170 + int(60 * r2)
                d.ellipse([px - s, py - s, px + s, py + s], fill=(g, g - 10, g - 30, int(200 * fade)))

        # tags under their own faces, only once the tile stands (and the camera has nearly arrived)
        def at(lx, ly):
            return rot_pt((PAD + lx, PAD + ly), pl, ps, deg)
        an, bn = ctx.link_names(i)
        size, xa, xb, fa, fb = tag_layout(i)
        t_show = RISE if first else CAM * 0.8
        if local > t_show and down <= 0:
            pa = ctx.ease_out_back(min(1.0, (local - t_show) / 0.2))
            ax, ay = at(xa, TAG_Y)
            draw_tag(base, ctx.C.labels[an].upper(), ctx.CYAN, size, ax, at(fa, TAG_Y)[0], ay, pa)
        if local > coin_t and down <= 0:
            pop = ctx.ease_out_back(min(1.0, (local - coin_t) / 0.18))
            bx_, by_ = at(xb, TAG_Y)
            draw_tag(base, ctx.C.labels[bn].upper(), ctx.YELLOW, size, bx_, at(fb, TAG_Y)[0], by_ - 40 * (1 - pop),
                     0.5 + 0.5 * pop)
        top = at(DW / 2, 0)

        # Buh rides the dominoes: on the old tile's top as it tips, a leap, then onto the new one
        bscale = 5
        jump0, jump1 = 0.12, FALL + 0.3
        bxy, land = top, 1.0
        if first:
            v = min(1.0, max(0.0, (local - RISE * 0.6) / 0.4))
            if v < 1:                                    # link 1: Buh drops in from above as the tile lands
                ctx.paste_pin(base, ctx.pin_sprite("shock", 0, 0, 1, scale=bscale), top[0], top[1] - 600 * (1 - v * v))
                bxy = None
        elif local < jump0:
            ptop = rot_pt((PAD + DW / 2, PAD), (PAD + DW, PAD + DH), (sx(wx(i - 1) + DW), FLOOR + cy), fall_deg(local))
            ctx.paste_pin(base, ctx.pin_sprite("smug", -fall_deg(local), 0, 1, scale=bscale), ptop[0], ptop[1])
            bxy = None
        elif local < jump1:
            ptop = rot_pt((PAD + DW / 2, PAD), (PAD + DW, PAD + DH), (sx(wx(i - 1) + DW), FLOOR + cy), fall_deg(jump0))
            v = (local - jump0) / (jump1 - jump0)
            x = ptop[0] + (top[0] - ptop[0]) * v
            y = ptop[1] + (top[1] - ptop[1]) * v - 260 * math.sin(math.pi * v)
            ctx.paste_pin(base, ctx.pin_sprite("shock", -30 * (1 - v), 0, 1, scale=bscale), x, y)
            bxy = None
        else:
            land = local - jump1
        if bxy is not None:
            expr = "shock" if coin_t - 0.05 < local < coin_t + 0.5 else ("happy" if local > coin_t else "smug")
            hop = abs(math.sin(local * 3.0)) * 12
            if last and local > coin_t + 0.05:
                hop = abs(math.sin(local * 8)) * 50
                expr = "happy"
            spr = ctx.pin_sprite(expr, 0, 0.3 * max(0.0, 1 - land / 0.15), 1, scale=bscale)
            ctx.paste_pin(base, spr, bxy[0], bxy[1] - hop)

        chain_row(base, i, local > coin_t, cy)
        ctx.header(base, i)
        if last:                                         # CONNECTED! across the tile's (pip-less) foot, under the tags
            ctx.connected(base, seg, tt, y=TOP + PIP_Y + 4)
        if first or local >= CAP_HOLD:                   # the first caption waits for the slam to finish
            seg["cap"].draw(base, tt, y=CAP_Y)
        return base

    return frame
