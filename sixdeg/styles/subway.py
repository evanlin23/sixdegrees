"""Subway: the chain as a metro line. Each person is a station (their face in a roundel), Buh rides a little train
from stop to stop, the link photo pops up as a "next stop" panel, and the finale pulls back to the whole line."""
from __future__ import annotations

import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

NAME = "subway"

PAPER = (241, 236, 224)
LINE = (226, 46, 58)                     # the Buh line
PALE = tuple(int(c * 0.35 + p * 0.65) for c, p in zip(LINE, PAPER))
# station layout (world px): a snake of straight runs and 45-degree bends that fits a portrait screen
PATTERN = [(0, 0), (520, 0), (1000, 330), (520, 700), (0, 700), (250, 1150), (900, 1150)]
CARD_TOP = 248                           # the next-stop card sits right under the title band
SIGN0, SIGN1 = 978, 1068                 # a dark platform sign under the card carries the captions
CAP_Y = (SIGN0 + SIGN1) // 2
MAP_Y0, MAP_Y1, MAP_AY = SIGN1 + 12, 1580, 1372   # the live map fills the lower half; labels stay above MAP_Y1
TRAIN_Y = (1268, 1640)                   # the camera keeps the train (and Buh on its roof) inside this band


def photo_fit(ctx, i, w, h, fill=0.8, maxup=1.3):
    """Link photo i framed on both faces in a w x h window, cover-cropped where the source allows, never upscaled
    beyond maxup (a small source comes back smaller than w x h: the caller mattes it, no blurred bars)."""
    ih, iw = ctx.C.imgs[i].shape[:2]
    cx, cy, s = ctx.both_cam(i, w, h, fill)
    ba, bb = ctx.fbox(i, "a"), ctx.fbox(i, "b")
    spread = max(ba[0] + ba[2], bb[0] + bb[2]) - min(ba[0], bb[0])
    s = min(maxup, max(s, max(w / iw, h / ih)), 0.95 * w / max(1, spread))
    ow, oh = min(w, int(iw * s)), min(h, int(ih * s))
    hw, hh = ow / 2 / s, oh / 2 / s
    cx, cy = min(max(cx, hw), iw - hw), min(max(cy, hh), ih - hh)
    return ctx.view(i, cx, cy, s, ow, oh)


def _route(p, q):
    """Metro-map path p -> q: straight, a 45-degree diagonal, straight (the bend sits in the middle)."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    ax, ay = abs(dx), abs(dy)
    sx, sy = (1 if dx >= 0 else -1), (1 if dy >= 0 else -1)
    d = min(ax, ay)
    if ax >= ay:
        s = (ax - d) / 2
        a = (p[0] + sx * s, p[1])
        b = (a[0] + sx * d, a[1] + sy * d)
    else:
        s = (ay - d) / 2
        a = (p[0], p[1] + sy * s)
        b = (a[0] + sx * d, a[1] + sy * d)
    pts = [p, a, b, q]
    out = [pts[0]]
    for x in pts[1:]:
        if math.dist(x, out[-1]) > 0.5:
            out.append(x)
    return out


def _positions(n):
    pos = []
    for k in range(n):
        blk, j = divmod(k, len(PATTERN))
        x, y = PATTERN[j]
        if blk % 2:
            x = 1000 - x                     # next block mirrored, so the snake carries on downwards
        pos.append((float(x), float(y + blk * 1600)))
    return pos


def make(ctx):
    people = list(ctx.C.people)
    N = len(people)
    idx = {nm: k for k, nm in enumerate(people)}
    ST = _positions(N)
    # one polyline through every station, with each station's distance along it
    poly, D = [ST[0]], [0.0]
    for k in range(1, N):
        D.append(D[-1] + sum(math.dist(a, b) for a, b in zip(_route(ST[k - 1], ST[k])[:-1], _route(ST[k - 1], ST[k])[1:])))
        poly += _route(ST[k - 1], ST[k])[1:]
    cum = [0.0]
    for a, b in zip(poly[:-1], poly[1:]):
        cum.append(cum[-1] + math.dist(a, b))

    def at(d):
        d = max(0.0, min(cum[-1], d))
        for k in range(1, len(poly)):
            if d <= cum[k] or k == len(poly) - 1:
                a, b = poly[k - 1], poly[k]
                L = max(1e-6, cum[k] - cum[k - 1])
                u = (d - cum[k - 1]) / L
                return a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u
        return poly[-1]

    def upto(d):
        """Polyline points from the first station to distance d."""
        pts = [poly[0]]
        for k in range(1, len(poly)):
            if cum[k] < d:
                pts.append(poly[k])
            else:
                break
        pts.append(at(d))
        return pts

    xs, ys = [p[0] for p in ST], [p[1] for p in ST]
    BB = (min(xs), min(ys), max(xs), max(ys))
    M = 2000
    OX, OY = BB[0] - M, BB[1] - M
    LW, LH = int(BB[2] - BB[0] + 2 * M), int(BB[3] - BB[1] + 2 * M)

    # ---------- static map layer: paper, grid, river, parks, other lines, the (pale) Buh line
    def build_layer():
        im = Image.new("RGB", (LW, LH), PAPER)
        d = ImageDraw.Draw(im)
        for x in range(0, LW, 90):
            d.line([(x, 0), (x, LH)], fill=(232, 226, 212), width=2)
        for y in range(0, LH, 90):
            d.line([(0, y), (LW, y)], fill=(232, 226, 212), width=2)
        rnd = random.Random(7)
        for _ in range(26):                                      # parks
            x, y = rnd.uniform(0, LW), rnd.uniform(0, LH)
            w, h = rnd.uniform(160, 420), rnd.uniform(120, 300)
            d.rounded_rectangle([x, y, x + w, y + h], 40, fill=(214, 228, 196))

        def L(x, y):
            return x - OX, y - OY
        # a river with 45-degree bends crossing the whole map
        cy = (BB[1] + BB[3]) / 2
        river = [L(BB[0] - M, cy - 900), L(BB[0] - 300, cy - 900), L(BB[0] + 300, cy - 300), L(BB[2] - 100, cy - 300),
                 L(BB[2] + 500, cy + 300), L(BB[2] + M, cy + 300)]
        d.line(river, fill=(188, 214, 236), width=120, joint="curve")
        for p in river:
            d.ellipse([p[0] - 60, p[1] - 60, p[0] + 60, p[1] + 60], fill=(188, 214, 236))
        # other lines, faded so the Buh line stays the hero
        cols = [(60, 120, 210), (40, 160, 90), (245, 165, 30), (150, 90, 200), (0, 160, 170), (120, 120, 120)]
        for k, col in enumerate(cols):
            col = tuple(int(c * 0.55 + p * 0.45) for c, p in zip(col, PAPER))
            x, y = rnd.uniform(BB[0] - 900, BB[2] + 900), BB[1] - M + 60
            pts = [L(x, y)]
            vertical = k % 2 == 0
            if not vertical:
                x, y = BB[0] - M + 60, rnd.uniform(BB[1] - 600, BB[3] + 600)
                pts = [L(x, y)]
            while 0 <= pts[-1][0] < LW and 0 <= pts[-1][1] < LH:
                step = rnd.uniform(250, 600)
                turn = rnd.choice([-1, 0, 0, 1])
                if vertical:
                    x += turn * step * 0.7
                    y += step
                else:
                    x += step
                    y += turn * step * 0.7
                pts.append(L(x, y))
            d.line(pts, fill=col, width=18, joint="curve")
            for a, b in zip(pts[:-1], pts[1:]):                     # station ticks
                for u in (0.33, 0.66):
                    px, py = a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u
                    d.ellipse([px - 13, py - 13, px + 13, py + 13], fill=(255, 255, 255), outline=col, width=5)
        pts = [L(*p) for p in poly]                               # the Buh line, not yet travelled
        d.line(pts, fill=PALE, width=34)
        for p in pts:
            d.ellipse([p[0] - 17, p[1] - 17, p[0] + 17, p[1] + 17], fill=PALE)
        return np.asarray(im)

    LAYER = build_layer()

    # ---------- sprites
    PARK = 150
    RR = 85                                                       # roundel radius at zoom 1

    def roundel(nm, known):
        S = 2 * RR + 8
        im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        c = S / 2
        d.ellipse([c - RR + 4, c - RR + 8, c + RR + 4, c + RR + 8], fill=(0, 0, 0, 90))
        d.ellipse([c - RR, c - RR, c + RR, c + RR], fill=ctx.INK)
        d.ellipse([c - RR + 5, c - RR + 5, c + RR - 5, c + RR - 5], fill=LINE if known else (150, 146, 140))
        r = RR - 18
        if known:
            face = ctx.sharp[nm].convert("RGBA").resize((2 * r, 2 * r), Image.BICUBIC)
            m = Image.new("L", face.size, 0)
            ImageDraw.Draw(m).ellipse([0, 0, 2 * r - 1, 2 * r - 1], fill=255)
            d.ellipse([c - r - 5, c - r - 5, c + r + 5, c + r + 5], fill=(255, 255, 255))
            im.paste(face, (int(c - r), int(c - r)), m)
        else:
            d.ellipse([c - r, c - r, c + r, c + r], fill=(222, 218, 210))
            q = ctx.pixel_text("?", 60, fill=(255, 255, 255))
            im.alpha_composite(q, (int(c - q.width / 2), int(c - q.height / 2)))
        return im

    ROUND = {(nm, kn): roundel(nm, kn) for nm in people for kn in (True, False)}

    def label(nm):
        words, lines, cur = ctx.C.labels[nm].upper().split(), [], ""
        for w in words:
            if cur and len(cur) + len(w) + 1 > 11:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        lines.append(cur)
        ims = [ctx.pixel_text(ln, 38, fill=ctx.WHITE) for ln in lines]
        w = max(i.width for i in ims)
        out = Image.new("RGBA", (w, sum(i.height - 8 for i in ims) + 8), (0, 0, 0, 0))
        y = 0
        for i in ims:
            out.alpha_composite(i, (int((w - i.width) / 2), y))
            y += i.height - 8
        return out

    LABEL = {nm: label(nm) for nm in people}

    def train_sprite():
        s = Image.new("RGBA", (64, 26), (0, 0, 0, 0))
        d = ImageDraw.Draw(s)
        d.rounded_rectangle([1, 2, 60, 21], 5, fill=(250, 250, 250), outline=ctx.INK, width=2)
        d.rectangle([2, 15, 59, 18], fill=LINE)
        for x in range(8, 48, 10):
            d.rectangle([x, 6, x + 6, 11], fill=(40, 60, 90))
        d.rectangle([52, 6, 58, 12], fill=(40, 60, 90))
        d.rectangle([58, 14, 60, 16], fill=ctx.YELLOW)
        for x in (12, 22, 40, 50):
            d.ellipse([x - 3, 19, x + 3, 25], fill=ctx.INK)
        return s.resize((s.width * 3, s.height * 3), Image.NEAREST)

    TRAIN = train_sprite()
    TRAIN_L = TRAIN.transpose(Image.FLIP_LEFT_RIGHT)


    # ---------- the next-stop card: one fixed place under the map strip, photo as big as the screen allows
    PW, PH, HEAD, FOOT = 860, 540, 74, 100
    CARD_H = HEAD + 6 + PH + FOOT
    CARD_CY = CARD_TOP + CARD_H / 2
    FOOT_CY = CARD_TOP + HEAD + 6 + PH + FOOT / 2         # CONNECTED! lands here on the last link
    _panels = {}

    def wrap(text, n):
        words, lines, cur = text.split(), [], ""
        for w in words:
            if cur and len(cur) + len(w) + 1 > n:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        lines.append(cur)
        if len(lines) > 2:
            lines = [lines[0], lines[1].rstrip(",;:…") + "…"]
        return lines

    def panel(i, revealed, fact=True):
        key = (i, revealed, fact)
        if key in _panels:
            return _panels[key]
        an, bn = ctx.link_names(i)
        im = Image.new("RGBA", (PW + 40, CARD_H + 20), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([12, 16, PW + 32, CARD_H + 16], 18, fill=(0, 0, 0, 110))
        d.rounded_rectangle([0, 0, PW + 20, CARD_H], 18, fill=(252, 250, 244), outline=ctx.INK, width=6)
        d.rounded_rectangle([0, 0, PW + 20, HEAD], 18, fill=LINE, outline=ctx.INK, width=6)
        d.rectangle([6, 40, PW + 14, HEAD], fill=LINE)
        d.line([(0, HEAD), (PW + 20, HEAD)], fill=ctx.INK, width=6)
        tag = ctx.pixel_text("NEXT STOP" if not revealed else "NOW ARRIVING", 24, fill=ctx.WHITE)
        im.alpha_composite(tag, (24, int(HEAD / 2 + 1 - tag.height / 2)))
        nm = ctx.C.labels[bn].upper() if revealed else "? ? ?"
        size = 28
        while size > 16 and ctx.pf(size).getlength(nm) > PW - tag.width - 90:
            size -= 2
        b = ctx.pixel_tag(nm, ctx.YELLOW if revealed else (210, 206, 198), size)
        im.alpha_composite(b, (int(PW + 8 - b.width), int(HEAD / 2 + 1 - b.height / 2)))
        d.rectangle([10, HEAD + 6, 10 + PW, HEAD + 6 + PH], fill=(40, 38, 44))   # matte for small photos
        ph = photo_fit(ctx, i, PW, PH)
        px, py = 10 + (PW - ph.width) // 2, HEAD + 6 + (PH - ph.height) // 2
        im.alpha_composite(ph, (px, py))
        d.rectangle([px, py, px + ph.width, py + ph.height], outline=ctx.INK, width=3)
        if fact:
            lines = wrap(ctx.fact_of(ctx.spec["links"][i], full=revealed).upper(), 32)
            f = ctx.pf(24)
            y0 = HEAD + 6 + PH + FOOT / 2 - (len(lines) * 38 - 8) / 2
            for k, ln in enumerate(lines):
                d.text((26, y0 + k * 38), ln, font=f, fill=ctx.INK)
        _panels[key] = im
        return im

    # ---------- the clock: where the train is at any moment of the video
    segs = [s for s in ctx.segs if s["kind"] == "link"]

    def arrive(s):
        return s.get("rv", ctx.REVEAL) * s["speech"] + ctx.THROW

    def train_d(T):
        """Distance of the train along the line at time T (s)."""
        for s in segs:
            if T < s["t0"]:
                return max(0.0, D[idx.get(ctx.link_names(s["i"])[0], 0)] - PARK)
            if T < s["t1"] or s is segs[-1]:
                a, b = ctx.link_names(s["i"])
                da, db = D[idx.get(a, 0)], D[idx.get(b, 0)]
                da, db = max(0.0, da - PARK), max(0.0, db - PARK)   # the train halts just short of the roundel
                u = (T - s["t0"] - 0.1) / max(0.2, arrive(s) - 0.1)
                return da + (db - da) * ctx.ease(max(0.0, min(1.0, u)))
        return 0.0

    def seg_at(T):
        for s in segs:
            if T < s["t1"] or s is segs[-1]:
                return s
        return segs[-1]

    def target(T):
        """Camera (x, y, zoom, screen y) framing A and B of the link in the map strip; on the last link it closes
        in on the destination once the train arrives."""
        s = seg_at(T)
        a, b = ST[idx.get(ctx.link_names(s["i"])[0], 0)], ST[idx.get(ctx.link_names(s["i"])[1], 0)]
        if s["i"] == ctx.n_links - 1 and T - s["t0"] > arrive(s) - 0.15:
            return b[0], b[1], 1.0, MAP_AY - 20
        dx, dy = abs(b[0] - a[0]) + 1, abs(b[1] - a[1]) + 1
        z = max(0.5, min(0.9, 640 / dx, 220 / dy))
        return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, z, MAP_AY

    def cam(T):
        n = 8
        acc = [0.0, 0.0, 0.0, 0.0]
        for k in range(n):
            x, y, z, ay = target(T - 0.6 * k / (n - 1))
            acc[0] += x
            acc[1] += y
            acc[2] += math.log(z)
            acc[3] += ay
        return acc[0] / n, acc[1] / n, math.exp(acc[2] / n), acc[3] / n

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        sp, R = seg["speech"], seg.get("rv", ctx.REVEAL)
        arr = arrive(seg)
        revealed = local > R * sp + ctx.THROW
        last = i == ctx.n_links - 1
        an, bn = ctx.link_names(i)
        ia, ib = idx.get(an, 0), idx.get(bn, 0)

        cx, cy, z, ay = cam(tt)
        cx += 10 * math.sin(tt * 0.9)
        cy += 6 * math.cos(tt * 0.7)
        ax = ctx.CX
        td = train_d(tt)
        wy = (at(td)[1] - cy) * z + ay                          # keep the train (and Buh) in the map band,
        ay += max(0.0, TRAIN_Y[0] - wy) - max(0.0, wy - TRAIN_Y[1])   # never up under the card or the sign

        def scr(p):
            return (p[0] - cx) * z + ax, (p[1] - cy) * z + ay

        Mx = np.float32([[z, 0, z * (OX - cx) + ax], [0, z, z * (OY - cy) + ay]])
        arr_img = cv2.warpAffine(LAYER, Mx, (ctx.W, ctx.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=PAPER)
        base = Image.fromarray(arr_img).convert("RGBA")
        d = ImageDraw.Draw(base)

        # the travelled part of the line, bold, with little white dashes flowing along it
        ld = td + PARK * ctx.ease(max(0.0, min(1.0, (local - arr) / 0.2)))   # line reaches the stop once arrived
        pts = [scr(p) for p in upto(ld)]
        lw = max(8, int(34 * z))
        if len(pts) > 1:
            d.line(pts, fill=LINE, width=lw)
            for p in pts:
                d.ellipse([p[0] - lw / 2, p[1] - lw / 2, p[0] + lw / 2, p[1] + lw / 2], fill=LINE)
            dash = 60
            k = (tt * 140) % dash
            while k < ld:
                p = scr(at(k))
                r = max(2, lw * 0.16)
                d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(255, 235, 235))
                k += dash

        # the train, Buh on the roof
        p0, p1 = at(td - 30), at(td + 30)
        hx, hy = p1[0] - p0[0], p1[1] - p0[1]
        moving = 0.12 < local < arr
        tx, ty = scr(at(td))
        bob = 2 * math.sin(tt * 18) if moving else 1.5 * math.sin(tt * 4)
        ts = max(0.55, min(1.0, z * 1.05))
        if abs(hx) < 1e-3 and abs(hy) < 1e-3:
            hx = 1.0
        left = hx < -1e-3
        ang = math.degrees(math.atan2(hy, -hx)) if left else -math.degrees(math.atan2(hy, hx))
        spr = (TRAIN_L if left else TRAIN).rotate(ang, expand=True, resample=Image.BICUBIC)
        ctx.paste(base, spr, tx, ty + bob, scale=ts, nearest=False)

        # stations
        rs = max(0.5, min(1.05, z * 1.05))
        boxes, shown = [], []
        for k, nm in enumerate(people):
            known = k <= ia or (k == ib and revealed) or (ia < k < ib)
            sx, sy = scr(ST[k])
            if not (-200 < sx < ctx.W + 200 and MAP_Y0 - 200 < sy < ctx.H + 200):
                continue
            s = rs
            if k == ib and revealed:                             # B's roundel pops when the train pulls in
                pop = (local - R * sp - ctx.THROW) / 0.3
                s *= 1 + 0.35 * math.sin(math.pi * max(0.0, min(1.0, pop)))
                if last:
                    s *= 1 + 0.3 * ctx.ease(min(1.0, pop))       # and stays big at the finale
            elif k == ib:
                s *= 1 + 0.05 * math.sin(tt * 7)                # the next stop pulses
            ctx.paste(base, ROUND[(nm, known)], sx, sy, scale=s, nearest=False)
            r = RR * s
            boxes.append((sx - r, sy - r, sx + r, sy + r))
            if known:
                shown.append((k, nm, sx, sy, r))
        # labels: B first, then A, then the rest; each goes under, over or beside its roundel where it is free
        tb = ts * 1.0
        boxes.append((tx - 100 * tb, ty - 60 * tb, tx + 100 * tb, ty + 40 * tb))
        boxes.append((tx - 70 * tb, ty - 200 * tb, tx + 70 * tb, ty))          # Buh on the roof
        order = sorted(shown, key=lambda q: (q[0] != ib, q[0] != ia, -q[0]))
        for k, nm, sx, sy, r in order:
            if not ctx.SAFE[0] - 30 < sx < ctx.SAFE[2] + 30:
                continue                                         # off to the side: no label away from its face
            lab = LABEL[nm]
            ls = 1.0 if (last and k == ib and revealed) else 0.9
            w2, h2 = lab.width * ls / 2, lab.height * ls / 2
            cx_ = max(ctx.SAFE[0] + w2, min(ctx.SAFE[2] - w2, sx))
            # under, over, then beside its own roundel (right, left), wherever it is free and inside the map band
            for lx, ly in ((cx_, sy + r + h2 + 2), (cx_, sy - r - h2 - 4), (sx + r + w2 + 8, sy), (sx - r - w2 - 8, sy)):
                rect = (lx - w2, ly - h2, lx + w2, ly + h2)
                if rect[1] < MAP_Y0 + 4 or rect[3] > MAP_Y1 or rect[0] < ctx.SAFE[0] - 20 or rect[2] > ctx.SAFE[2] + 20:
                    continue
                own = (sx - r, sy - r, sx + r, sy + r)
                if any(rect[0] < b[2] and b[0] < rect[2] and rect[1] < b[3] and b[1] < rect[3]
                       for b in boxes if b != own):
                    continue
                ctx.paste(base, lab, lx, ly, scale=ls)
                boxes.append(rect)
                break

        expr = "happy"
        squash = 0.0
        if revealed:
            ar = local - arr
            expr = "shock" if ar < 0.5 else "smug"
            squash = 0.25 * math.sin(math.pi * min(1.0, ar / 0.3)) if ar < 0.3 else 0.0
        sway = (10 * math.sin(tt * 9)) if moving else 4 * math.sin(tt * 3)
        ctx.paste_pin(base, ctx.pin_sprite(expr, sway, squash, -1 if left else 1, scale=4), tx, ty + bob - 22 * ts)

        # the platform sign the captions sit on
        d.rectangle([0, SIGN0, ctx.W, SIGN1], fill=ctx.INK)
        d.rectangle([0, SIGN0, ctx.W, SIGN0 + 8], fill=LINE)
        d.rectangle([0, SIGN1 - 4, ctx.W, SIGN1], fill=(70, 66, 80))

        # the card never moves: its text switches at once to this stop, only the photo cross-fades from the last
        fin = last and revealed
        pan = panel(i, revealed, fact=not fin)
        ps = 1.0
        if revealed and local - arr < 0.25:
            ps = 1 + 0.03 * math.sin(math.pi * (local - arr) / 0.25)
        k_in = ctx.ease(max(0.0, min(1.0, local / 0.3)))
        if i > 0 and k_in < 1 and not ctx.C.fj["links"][i - 1].get("lineage"):
            ctx.paste(base, pan, ctx.CX, CARD_CY + 10, nearest=False)
            old = panel(i - 1, True).crop((10, HEAD + 6, 11 + PW, HEAD + 7 + PH))
            x0 = int(ctx.CX - pan.width / 2)
            y0 = int(CARD_CY + 10 - pan.height / 2)
            ctx.paste(base, old, x0 + 10 + old.width / 2, y0 + HEAD + 6 + old.height / 2, alpha=1 - k_in)
        elif i == 0 and k_in < 1:
            ctx.paste(base, pan, ctx.CX, CARD_CY + 10 + 60 * (1 - k_in), scale=0.94 + 0.06 * k_in, alpha=k_in,
                      nearest=False)
        else:
            ctx.paste(base, pan, ctx.CX, CARD_CY + 10, scale=ps, nearest=False)
        ctx.header(base, i)
        ctx.connected(base, seg, tt, y=FOOT_CY)
        seg["cap"].draw(base, tt, y=CAP_Y)
        return base

    return frame
