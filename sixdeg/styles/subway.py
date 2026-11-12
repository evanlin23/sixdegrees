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
        ims = [ctx.pixel_text(ln, 28, fill=ctx.WHITE) for ln in lines]
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

    _panels = {}

    def panel(i, revealed):
        key = (i, revealed)
        if key in _panels:
            return _panels[key]
        PW, PH = 820, 400
        an, bn = ctx.link_names(i)
        im = Image.new("RGBA", (PW + 40, PH + 190), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([12, 16, PW + 32, PH + 186], 18, fill=(0, 0, 0, 110))
        d.rounded_rectangle([0, 0, PW + 20, PH + 170], 18, fill=(252, 250, 244), outline=ctx.INK, width=6)
        d.rounded_rectangle([0, 0, PW + 20, 74], 18, fill=LINE, outline=ctx.INK, width=6)
        d.rectangle([6, 40, PW + 14, 74], fill=LINE)
        d.line([(0, 74), (PW + 20, 74)], fill=ctx.INK, width=6)
        tag = ctx.pixel_text("NEXT STOP" if not revealed else "NOW ARRIVING", 24, fill=ctx.WHITE)
        im.alpha_composite(tag, (24, int(38 - tag.height / 2)))
        nm = ctx.C.labels[bn].upper() if revealed else "? ? ?"
        size = 28
        while size > 16 and ctx.pf(size).getlength(nm) > PW - tag.width - 90:
            size -= 2
        b = ctx.pixel_tag(nm, ctx.YELLOW if revealed else (210, 206, 198), size)
        im.alpha_composite(b, (int(PW + 8 - b.width), int(38 - b.height / 2)))
        ph = ctx.view(i, *ctx.both_cam(i, PW, PH, 0.8), PW, PH)
        im.alpha_composite(ph, (10, 80))
        d.rectangle([10, 80, 10 + PW, 80 + PH], outline=ctx.INK, width=3)
        fact = ctx.fact_of(ctx.spec["links"][i])
        words, lines, cur = fact.upper().split(), [], ""
        for w in words:
            if cur and len(cur) + len(w) + 1 > 38:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        lines.append(cur)
        lines = lines[:2]
        f = ctx.pf(19)
        for k, ln in enumerate(lines):
            d.text((24, PH + 98 + k * 30), ln, font=f, fill=ctx.INK)
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

    def dest_d(T):
        for s in segs:
            if T < s["t1"] or s is segs[-1]:
                return D[idx.get(ctx.link_names(s["i"])[1], 0)]
        return D[-1]

    def follow(T):
        tx, ty = at(train_d(T))
        dx, dy = at(dest_d(T))
        return tx + (dx - tx) * 0.38, ty + (dy - ty) * 0.38

    def cam_follow(T):
        """Smoothed: the average of the follow target over the last 0.7 s."""
        n = 8
        xs_, ys_ = 0.0, 0.0
        for k in range(n):
            x, y = follow(T - 0.7 * k / (n - 1))
            xs_ += x
            ys_ += y
        return xs_ / n, ys_ / n

    FIT_Z = min((ctx.SAFE[2] - ctx.SAFE[0] - 300) / max(1, BB[2] - BB[0]),
                (1240 - 300 - 220) / max(1, BB[3] - BB[1]), 0.9)
    # finale framing: the gap between the last two rows of stations sits behind the CONNECTED! stamp (y 1020)
    gap_y = (BB[3] + max([y for y in ys if y < BB[3] - 1] or [BB[1]])) / 2
    FIT_AY = 1020.0
    FIT_C = ((BB[0] + BB[2]) / 2, gap_y + 45)
    if FIT_AY - (gap_y - BB[1]) * FIT_Z < 330 or FIT_AY + (BB[3] - gap_y) * FIT_Z > 1220:
        FIT_AY, FIT_C = 760.0, ((BB[0] + BB[2]) / 2, (BB[1] + BB[3]) / 2)

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
        dur = seg["t1"] - seg["t0"]

        # finale: on the last link the camera pulls back to the whole line during the final approach
        fin = ctx.ease(max(0.0, min(1.0, (local - 0.45 * arr) / (0.5 * arr)))) if last else 0.0
        fx, fy = cam_follow(tt)
        drift = 18 * math.sin(tt * 0.9), 14 * math.cos(tt * 0.7)
        z_f = 0.92 + 0.03 * math.sin(tt * 1.3)
        z = math.exp(math.log(z_f) + (math.log(FIT_Z) - math.log(z_f)) * fin)
        cx = fx + drift[0] + (FIT_C[0] - fx - drift[0]) * fin
        cy = fy + drift[1] + (FIT_C[1] - fy - drift[1]) * fin
        ax, ay = ctx.CX, 470 + (FIT_AY - 470) * fin

        def scr(p):
            return (p[0] - cx) * z + ax, (p[1] - cy) * z + ay

        Mx = np.float32([[z, 0, z * (OX - cx) + ax], [0, z, z * (OY - cy) + ay]])
        arr_img = cv2.warpAffine(LAYER, Mx, (ctx.W, ctx.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=PAPER)
        base = Image.fromarray(arr_img).convert("RGBA")
        d = ImageDraw.Draw(base)

        # the travelled part of the line, bold, with little white dashes flowing along it
        td = train_d(tt)
        ld = td + PARK * ctx.ease(max(0.0, min(1.0, (local - arr) / 0.2)))   # line reaches the stop once arrived
        pts = [scr(p) for p in upto(ld)]
        lw = max(8, int(34 * z))
        if len(pts) > 1:
            d.line(pts, fill=LINE, width=lw)
            for p in pts:
                d.ellipse([p[0] - lw / 2, p[1] - lw / 2, p[0] + lw / 2, p[1] + lw / 2], fill=LINE)
            dash = 60
            off = (tt * 140) % dash
            k = off
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
        ts = max(0.6, min(1.0, z * 1.05))
        if abs(hx) < 1e-3 and abs(hy) < 1e-3:
            hx = 1.0
        left = hx < -1e-3
        ang = math.degrees(math.atan2(hy, -hx)) if left else -math.degrees(math.atan2(hy, hx))
        spr = (TRAIN_L if left else TRAIN).rotate(ang, expand=True, resample=Image.BICUBIC)
        ctx.paste(base, spr, tx, ty + bob, scale=ts, nearest=False)
        # stations
        rs = max(0.55, min(1.0, z * 1.05))
        for k, nm in enumerate(people):
            known = k <= ia or (k == ib and revealed) or (k < ib and k > ia)
            sx, sy = scr(ST[k])
            if not (-200 < sx < ctx.W + 200 and -200 < sy < ctx.H + 200):
                continue
            s = rs
            if k == ib and revealed:                             # B's roundel pops when the train pulls in
                pop = (local - R * sp - ctx.THROW) / 0.3
                s *= 1 + 0.35 * math.sin(math.pi * max(0.0, min(1.0, pop)))
            elif k == ib:
                s *= 1 + 0.05 * math.sin(tt * 7)                # the next stop pulses
            ctx.paste(base, ROUND[(nm, known)], sx, sy, scale=s, nearest=False)
        for k, nm in enumerate(people):                          # labels on top of every roundel
            known = k <= ia or (k == ib and revealed) or (k < ib and k > ia)
            if not known:
                continue
            sx, sy = scr(ST[k])
            if not ctx.SAFE[0] - 60 < sx < ctx.SAFE[2] + 60:
                continue                                         # off to the side: no label squeezed in
            lab = LABEL[nm]
            ls = 0.72 + 0.28 * (1 - fin)
            lx = max(ctx.SAFE[0] + lab.width * ls / 2, min(ctx.SAFE[2] - lab.width * ls / 2, sx))
            ly = sy + RR * rs + lab.height * ls / 2 + 2
            if ly - lab.height * ls / 2 > ctx.BAND - 30 and ly < ctx.SAFE[3] - 40:
                ctx.paste(base, lab, lx, ly, scale=ls)

        expr = "happy"
        squash = 0.0
        if revealed:
            ar = local - arr
            expr = "shock" if ar < 0.5 else "smug"
            squash = 0.25 * math.sin(math.pi * min(1.0, ar / 0.3)) if ar < 0.3 else 0.0
        sway = (10 * math.sin(tt * 9)) if moving else 4 * math.sin(tt * 3)
        ctx.paste_pin(base, ctx.pin_sprite(expr, sway, squash, -1 if left else 1, scale=4), tx, ty + bob - 22 * ts)

        # next-stop panel: pops up at the start of the link, flips to the arrival name, drops away at the end
        pin = ctx.ease_out_back(max(0.0, min(1.0, (local - 0.08) / 0.3)))
        pout = 1.0
        if last:
            pout = 1 - ctx.ease(max(0.0, min(1.0, (local - 0.45 * arr) / 0.3)))
        else:
            pout = 1 - ctx.ease(max(0.0, min(1.0, (local - (dur - 0.25)) / 0.22)))
        vis = min(pin, 1.0) * pout
        if vis > 0.01:
            pan = panel(i, revealed)
            py = 1015 + (1 - vis) * 700 + 7 * math.sin(tt * 2.4)
            ps = 0.9 + 0.1 * vis
            if revealed and local - arr < 0.25:
                ps *= 1 + 0.05 * math.sin(math.pi * (local - arr) / 0.25)
            ctx.paste(base, pan, ctx.CX, py, scale=ps, nearest=False)
            # a stem from the panel up to the train, like a speech bubble pointing at the stop
            if vis > 0.6 and fin < 0.05:
                top = py - pan.height * ps / 2
                if ty + 40 < top:
                    d.polygon([(ctx.CX - 30, top + 4), (ctx.CX + 30, top + 4), (tx, max(ty + 40, top - 60))],
                              fill=ctx.INK)
        ctx.header(base, i)
        ctx.connected(base, seg, tt)
        seg["cap"].draw(base, tt, y=1368)
        return base

    return frame
