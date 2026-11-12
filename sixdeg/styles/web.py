"""Web: "six degrees" as a network graph you fly through. Every link photo is a NODE of the graph: a big photo card
floating in a dark web of grey, unknown photo-nodes. The edge between two cards runs from a person's face medallion
under one photo to the same person's medallion over the next, and the camera flies along it with Buh from card to
card. Under each card B's medallion is "?" until the reveal; a ghost "?" node waits further down the web."""
from __future__ import annotations

import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

NAME = "web"

BG = (12, 16, 34)
DECOY = (70, 80, 110)
DECOY_EDGE = (34, 42, 70)
PATH = (40, 220, 255)
HOT = (255, 214, 10)
CALM = (196, 200, 212)                  # links with a dictator: neutral frame, no glow, no gold
PW, PH, STRIP, BORD = 860, 590, 72, 8   # photo, tag strip under it, border
CW, CH = PW + 2 * BORD, PH + STRIP + 2 * BORD
GL = 30                                 # glow margin around a card sprite
MR = 52                                 # medallion radius
MED_OFF = 24                            # gap between a card's border and its medallions
STEP_Y, STEP_X = 1240, 380              # world offset from one card to the next (x alternates)
SX, SY = 505, 780                       # screen point the current card is centred on
FLY = 0.5                               # seconds the camera flies from one card to the next
CAP_Y = 1404


def photo_params(ctx, i, w, h, fill=0.8, maxup=1.3):
    """(cx, cy, scale, ow, oh) for link photo i in a w x h window: both faces in, cover-cropped where the source
    allows, never upscaled beyond maxup (a small source gives ow, oh < w, h and gets a dark matte, not blur)."""
    ih, iw = ctx.C.imgs[i].shape[:2]
    cx, cy, s = ctx.both_cam(i, w, h, fill)
    ba, bb = ctx.fbox(i, "a"), ctx.fbox(i, "b")
    spread = max(ba[0] + ba[2], bb[0] + bb[2]) - min(ba[0], bb[0])
    s = min(maxup, max(s, max(w / iw, h / ih)), 0.95 * w / max(1, spread))
    ow, oh = min(w, int(iw * s)), min(h, int(ih * s))
    hw, hh = ow / 2 / s, oh / 2 / s
    cx, cy = min(max(cx, hw), iw - hw), min(max(cy, hh), ih - hh)
    return cx, cy, s, ow, oh


def bez(p0, p3, t):
    """Vertical S-curve from p0 to p3 (tangents straight down at both ends)."""
    dy = (p3[1] - p0[1]) * 0.55
    p1, p2 = (p0[0], p0[1] + dy), (p3[0], p3[1] - dy)
    u = 1 - t
    return (u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
            u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1])


def make(ctx):
    N = ctx.n_links
    rnd = random.Random(11)
    CC = [(float(STEP_X * (k % 2) - STEP_X / 2), float(k * STEP_Y)) for k in range(N + 1)]   # + the ghost slot

    def calm(i):
        a, b = ctx.link_names(i)
        return ctx.sensitive(a) or ctx.sensitive(b)

    _fits = {}

    def fit(i):
        if i not in _fits:
            _fits[i] = photo_params(ctx, i, PW, PH)
        return _fits[i]

    def face_dx(i, role):
        """x of person role's face relative to card i's centre (card px), kept clear of the card's corners."""
        cx, cy, s, ow, oh = fit(i)
        fx = (ctx.ctr(ctx.fbox(i, role))[0] - cx) * s
        return max(-CW / 2 + MR + 26, min(CW / 2 - MR - 26, fx))

    def med_a(i):
        return CC[i][0] + face_dx(i, "a"), CC[i][1] - CH / 2 - MED_OFF - MR

    def med_b(i):
        if i >= N:
            return CC[i][0], CC[i][1] - CH / 2 - MED_OFF - MR
        return CC[i][0] + face_dx(i, "b"), CC[i][1] + CH / 2 + MED_OFF + MR

    # ---------- the incoming edge of each card: from the previous card's B medallion to this card's A medallion
    def edge_pts(i, n=28):
        p0, p3 = med_b(i - 1), (med_a(i) if i < N else med_b(i))
        return [bez(p0, p3, k / (n - 1)) for k in range(n)]

    # ---------- decoys: grey unknown photo-nodes all around the chain, with faint edges
    xs = [c[0] for c in CC]
    DEC = []
    corridor = [p for k in range(1, N + 1) for p in edge_pts(k, 12)]
    tries = 0
    while len(DEC) < 90 and tries < 8000:
        tries += 1
        x = rnd.uniform(min(xs) - 1300, max(xs) + 1300)
        y = rnd.uniform(-1300, CC[-1][1] + 1300)
        if any(abs(x - c[0]) < CW / 2 + 110 and abs(y - c[1]) < CH / 2 + 190 for c in CC):
            continue
        if any(math.dist((x, y), p) < 150 for p in corridor) or any(math.dist((x, y), d) < 240 for d in DEC):
            continue
        DEC.append((x, y))
    DSIZE = [rnd.choice([0.55, 0.65, 0.75, 0.9]) for _ in DEC]
    DEDGE = set()
    for k, p in enumerate(DEC):
        near = sorted(range(len(DEC)), key=lambda j: math.dist(p, DEC[j]))[1:3]
        for j in near[: rnd.choice([1, 2, 2])]:
            if math.dist(p, DEC[j]) < 560:
                DEDGE.add((min(k, j), max(k, j)))
    DEDGE = sorted(DEDGE)
    CEDGE = []                                       # decoys wired to the side of a card (the graph goes on)
    for k, p in enumerate(DEC):
        for c in range(N):
            cx, cy = CC[c]
            if abs(p[1] - cy) < CH / 2 - 40 and CW / 2 < abs(p[0] - cx) < CW / 2 + 520 and rnd.random() < 0.7:
                CEDGE.append((k, c, (cx + math.copysign(CW / 2, p[0] - cx), p[1] + rnd.uniform(-60, 60))))

    # ---------- far layer: a much bigger web, dim and soft (parallax)
    M = 2600
    FOX, FOY = min(xs) - M, -M
    FW, FH = int(max(xs) - min(xs) + 2 * M), int(CC[-1][1] + 2 * M)

    def build_far():
        im = Image.new("RGB", (FW, FH), BG)
        d = ImageDraw.Draw(im)
        pts = [(rnd.uniform(0, FW), rnd.uniform(0, FH)) for _ in range(1100)]
        grid = {}
        for k, (x, y) in enumerate(pts):
            grid.setdefault((int(x // 300), int(y // 300)), []).append(k)
        for k, (x, y) in enumerate(pts):
            gx, gy = int(x // 300), int(y // 300)
            cand = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in grid.get((gx + dx, gy + dy), []) if j != k]
            cand.sort(key=lambda j: math.dist(pts[k], pts[j]))
            for j in cand[:2]:
                d.line([pts[k], pts[j]], fill=(24, 30, 54), width=3)
        for x, y in pts:
            r = rnd.uniform(6, 13)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(34, 42, 70))
        return np.asarray(im.filter(ImageFilter.GaussianBlur(2.5)))

    FAR = build_far()

    # ---------- sprites
    def decoy_img():
        w, h = 170, 124
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([0, 0, w - 1, h - 1], 14, fill=(28, 34, 58), outline=DECOY, width=4)
        for k, cx in enumerate((w * 0.36, w * 0.64)):          # two generic silhouettes: an unknown photo
            r = 17
            cy = h * 0.42
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(56, 64, 94))
            d.chord([cx - 34, cy + 16, cx + 34, cy + 80], 180, 360, fill=(56, 64, 94))
        d.rectangle([4, h - 10, w - 5, h - 5], fill=(28, 34, 58))
        return im

    DIMG = decoy_img()

    def ghost_img():
        im = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([0, 0, CW - 1, CH - 1], 26, fill=(18, 24, 46, 200))
        for x in range(0, CW, 40):                             # dashed outline
            d.line([(x, 2), (x + 20, 2)], fill=(110, 122, 160), width=5)
            d.line([(x, CH - 3), (x + 20, CH - 3)], fill=(110, 122, 160), width=5)
        for y in range(0, CH, 40):
            d.line([(2, y), (2, y + 20)], fill=(110, 122, 160), width=5)
            d.line([(CW - 3, y), (CW - 3, y + 20)], fill=(110, 122, 160), width=5)
        q = ctx.pixel_text("?", 150, fill=(110, 122, 160))
        im.alpha_composite(q, (int(CW / 2 - q.width / 2), int(CH * 0.3 - q.height / 2)))
        return im

    GHOST = ghost_img()

    _cards = {}

    def card(i, state):
        """state: 'cur' (before the reveal), 'rev' (B just named), 'done' (an earlier link)."""
        key = (i, state)
        if key in _cards:
            return _cards[key]
        plain = calm(i)
        col = CALM if plain else {"cur": PATH, "rev": HOT, "done": PATH}[state]
        glowing = not plain and state != "done"
        im = Image.new("RGBA", (CW + 2 * GL, CH + 2 * GL), (0, 0, 0, 0))
        if glowing:
            g = Image.new("RGBA", im.size, (0, 0, 0, 0))
            ImageDraw.Draw(g).rounded_rectangle([GL - 8, GL - 8, GL + CW + 8, GL + CH + 8], 34, fill=col + (170,))
            im.alpha_composite(g.filter(ImageFilter.GaussianBlur(14)))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([GL, GL, GL + CW, GL + CH], 26, fill=(10, 14, 30), outline=col, width=BORD)
        cx, cy, s, ow, oh = fit(i)
        ph = ctx.view(i, cx, cy, s, ow, oh)
        m = Image.new("L", (ow, oh), 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, ow - 1, oh - 1], 18, fill=255)
        px, py = GL + BORD + (PW - ow) // 2, GL + BORD + (PH - oh) // 2
        im.paste(ph, (px, py), m)
        # name tags in the strip under the photo, each under its own face, never overlapping
        an, bn = ctx.link_names(i)
        known_b = state != "cur"
        ta = ctx.pixel_tag(ctx.C.labels[an].upper(), ctx.CYAN, 22)
        tb = ctx.pixel_tag(ctx.C.labels[bn].upper() if known_b else "? ? ?",
                           (ctx.YELLOW if not plain else ctx.WHITE) if known_b else (150, 160, 190), 22)
        lo, hi = GL + BORD + 6, GL + BORD + PW - 6
        items = sorted([[ta, GL + CW / 2 + face_dx(i, "a")], [tb, GL + CW / 2 + face_dx(i, "b")]],
                       key=lambda q: q[1])
        for q in items:
            q[1] = max(lo + q[0].width / 2, min(hi - q[0].width / 2, q[1]))
        (l, xl), (r, xr) = items
        over = (l.width + r.width) / 2 + 10 - (xr - xl)
        if over > 0:
            xl, xr = xl - over / 2, xr + over / 2
            if xl - l.width / 2 < lo:
                xr += lo - (xl - l.width / 2)
                xl = lo + l.width / 2
            if xr + r.width / 2 > hi:
                xl -= xr + r.width / 2 - hi
                xr = hi - r.width / 2
        ty = GL + BORD + PH + STRIP / 2
        for t_, x_ in ((l, xl), (r, xr)):
            im.alpha_composite(t_, (int(x_ - t_.width / 2), int(ty - t_.height / 2)))
        _cards[key] = im
        return im

    _meds = {}

    def medal(nm, col):
        key = (nm, col)
        if key not in _meds:
            S = 2 * MR + 8
            im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.ellipse([0, 0, S - 1, S - 1], fill=col)
            r = MR - 4
            c = S / 2
            if nm is None:
                d.ellipse([c - r, c - r, c + r, c + r], fill=(34, 40, 66))
                q = ctx.pixel_text("?", 44, fill=ctx.WHITE)
                im.alpha_composite(q, (int(c - q.width / 2), int(c - q.height / 2)))
            else:
                face = ctx.sharp[nm].convert("RGBA").resize((2 * r, 2 * r), Image.BICUBIC)
                m = Image.new("L", face.size, 0)
                ImageDraw.Draw(m).ellipse([0, 0, 2 * r - 1, 2 * r - 1], fill=255)
                im.paste(face, (int(c - r), int(c - r)), m)
            _meds[key] = im
        return _meds[key]

    # ---------- the clock
    segs = [s for s in ctx.segs if s["kind"] == "link"]

    def rev_t(s):
        return s.get("rv", ctx.REVEAL) * s["speech"] + ctx.THROW

    def side(i, role):
        """Buh perches on the side of a medallion that faces the card's middle."""
        dx = face_dx(i, role)
        return 1 if dx < 0 else -1

    def perch_a(i):
        x, y = med_a(i)
        return x + side(i, "a") * (MR + 58), y + MR

    def perch_b(i):
        x, y = med_b(i)
        return x + side(i, "b") * (MR + 58), y + MR

    def hop_times(s):
        d = s["t1"] - s["t0"]
        hs = max(FLY + 0.2, min(rev_t(s) + 0.3, d - 0.6))
        return hs, min(hs + 0.55, d - 0.03)

    def buh_world(s, local):
        """Buh's needle-tip position (world) and whether he is in flight."""
        i = s["i"]
        if i > 0 and local < FLY:
            u = ctx.ease(max(0.0, local / FLY))
            return bez(perch_b(i - 1), perch_a(i), u), True
        if i == N - 1:
            return perch_a(i), False
        hs, he = hop_times(s)
        if local < hs:
            return perch_a(i), False
        if local >= he:
            return perch_b(i), False
        # down the outside of the card's right edge, clear of the photo
        rx = CC[i][0] + CW / 2 + 66
        a, b = perch_a(i), perch_b(i)
        pts = [a, (rx, a[1]), (rx, b[1]), b]
        L = [math.dist(p, q) for p, q in zip(pts[:-1], pts[1:])]
        u = ctx.ease((local - hs) / max(0.05, he - hs)) * sum(L)
        for p, q, l in zip(pts[:-1], pts[1:], L):
            if u <= l or l == L[-1]:
                v = u / max(1e-6, l)
                return (p[0] + (q[0] - p[0]) * v, p[1] + (q[1] - p[1]) * v), True
            u -= l
        return b, False

    def cam(seg, local):
        i = seg["i"]
        if i > 0 and local < FLY:
            u = ctx.ease(local / FLY)
            x = CC[i - 1][0] + (CC[i][0] - CC[i - 1][0]) * u
            y = CC[i - 1][1] + (CC[i][1] - CC[i - 1][1]) * u
            return x, y, 1 - 0.36 * math.sin(math.pi * u)
        return CC[i][0], CC[i][1], 1.0

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        rv = rev_t(seg)
        revealed = local > rv
        last = i == N - 1
        an, bn = ctx.link_names(i)
        cx, cy, z = cam(seg, local)
        cx += 10 * math.sin(tt * 0.5)                          # the camera never quite settles
        cy += 7 * math.cos(tt * 0.43)

        def scr(p):
            return (p[0] - cx) * z + SX, (p[1] - cy) * z + SY

        def on(p, m=200):
            return -m < p[0] < ctx.W + m and -m < p[1] < ctx.H + m

        # far web, parallax
        fz = z * 0.7
        fcx, fcy = cx * 0.55, cy * 0.55
        Mx = np.float32([[fz, 0, fz * (FOX - fcx) + SX], [0, fz, fz * (FOY - fcy) + SY]])
        base = Image.fromarray(cv2.warpAffine(FAR, Mx, (ctx.W, ctx.H), flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REFLECT)).convert("RGBA")
        d = ImageDraw.Draw(base)

        def flo(k, amp=1.0):
            return (amp * 12 * math.sin(tt * 0.8 + k * 1.7), amp * 10 * math.cos(tt * 0.67 + k * 2.3))

        dpos = [scr((p[0] + flo(k)[0], p[1] + flo(k)[1])) for k, p in enumerate(DEC)]
        ew = max(2, int(3 * z))
        for a, b in DEDGE:
            if on(dpos[a], 600) or on(dpos[b], 600):
                d.line([dpos[a], dpos[b]], fill=DECOY_EDGE, width=ew)
        for k, c, pt in CEDGE:
            if c <= i and (on(dpos[k], 600)):
                d.line([dpos[k], scr(pt)], fill=DECOY_EDGE, width=ew)
        for k, p in enumerate(dpos):
            if on(p, 120):
                ctx.paste(base, DIMG, p[0], p[1], scale=z * DSIZE[k], nearest=False)

        def edge(pts, col, w, glow=True):
            sp = [scr(p) for p in pts]
            if glow:
                d.line(sp, fill=tuple(int(c * 0.35 + b * 0.65) for c, b in zip(col, BG)), width=int(w * 2.6),
                       joint="curve")
            d.line(sp, fill=col, width=w, joint="curve")
            for q in range(3):                                  # pulses running down the edge
                v = (tt * 0.7 + q / 3) % 1.0
                p = sp[min(len(sp) - 1, int(v * (len(sp) - 1)))]
                r = w * 0.9
                d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(255, 255, 255))

        ew2 = max(4, int(9 * z))
        # the ghost "?" node further down the web and its dashed stub (where the chain goes next)
        if not last:
            g = scr(CC[i + 1])
            ga = min(1.0, max(0.0, (local - FLY) / 0.4)) if i > 0 else min(1.0, local / 0.4)
            if on(g, 600) and ga > 0:
                pts = [scr(p) for p in edge_pts(i + 1)]
                if revealed:
                    for k in range(0, len(pts) - 1, 2):
                        d.line([pts[k], pts[k + 1]], fill=(110, 122, 160), width=max(3, int(5 * z)))
                ctx.paste(base, GHOST, g[0], g[1], scale=z, alpha=0.85 * ga, nearest=False)
        # edges between the cards already reached
        for k in range(1, i + 1):
            pts = edge_pts(k)
            if on(scr(pts[0]), 900) or on(scr(pts[-1]), 900):
                edge(pts, PATH, ew2)

        # the cards (photo nodes): earlier ones, then this one
        for k in range(0, i + 1):
            c = scr(CC[k])
            if not on(c, CH):
                continue
            stt = "done" if k < i else ("rev" if revealed else "cur")
            ps = z
            if k == i and revealed and local - rv < 0.25:
                ps *= 1 + 0.025 * math.sin(math.pi * (local - rv) / 0.25)
            ctx.paste(base, card(k, stt), c[0], c[1], scale=ps, nearest=False)
            # medallions: A over the card, B under it
            ka, kb = ctx.link_names(k)
            kc = CALM if calm(k) else PATH
            pa = scr(med_a(k))
            ctx.paste(base, medal(ka, kc), pa[0], pa[1], scale=z, nearest=False)
            pb = scr(med_b(k))
            if k < i or revealed:
                colb = CALM if calm(k) else (HOT if k == i else PATH)
                s_ = z
                if k == i and local - rv < 0.3:
                    s_ *= 1 + 0.3 * math.sin(math.pi * (local - rv) / 0.3)
                ctx.paste(base, medal(kb, colb), pb[0], pb[1], scale=s_, nearest=False)
            else:
                ctx.paste(base, medal(None, (150, 160, 190)), pb[0], pb[1], scale=z * (1 + 0.04 * math.sin(tt * 8)),
                          nearest=False)

        # Buh, flying along the edge between cards, then perched by the medallions
        (wx, wy), flying = buh_world(seg, local)
        bx, by = scr((wx, wy))
        expr, squash = "happy", 0.0
        if revealed:
            ar = local - rv
            expr = "shock" if ar < 0.5 else "smug"
            squash = 0.25 * math.sin(math.pi * min(1.0, ar / 0.3)) if ar < 0.3 else 0.0
        bob = 0.0 if flying else 5 * abs(math.sin(tt * 4))
        tilt = 8 * math.sin(tt * 9) if flying else 4 * math.sin(tt * 3)
        spr = ctx.pin_sprite(expr, tilt, squash, 1 if bx < SX + 300 else -1, scale=3)
        tip = max(by - bob, 240 + spr.height * 0.93)          # never under the title band
        ctx.paste_pin(base, spr, max(70.0, min(ctx.W - 60.0, bx)), tip)

        # a soft band behind the captions, so the moving web never fights them
        if "band" not in _cards:
            bh = 150
            a = (np.clip(1 - np.abs(np.linspace(-1, 1, bh)) ** 3, 0, 1) * 165).astype(np.uint8)
            band = np.zeros((bh, ctx.W, 4), np.uint8)
            band[..., :3] = BG
            band[..., 3] = a[:, None]
            _cards["band"] = Image.fromarray(band)
        base.alpha_composite(_cards["band"], (0, int(CAP_Y - 75)))

        ctx.header(base, i)
        ctx.connected(base, seg, tt, y=1302)
        seg["cap"].draw(base, tt, y=CAP_Y)
        return base

    return frame
