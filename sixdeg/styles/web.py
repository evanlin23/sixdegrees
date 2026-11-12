"""Web: "six degrees" as a growing network graph. Each person is a glowing node; every link a new node flies in, an
edge draws to it and the photo hangs off the edge as a card before shrinking into it. Grey decoy nodes float all
around, and the finale pulls back to the highlighted path through the whole web."""
from __future__ import annotations

import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

NAME = "web"

BG = (12, 16, 34)
DECOY = (70, 80, 110)
DECOY_EDGE = (36, 44, 72)
PATH = (40, 220, 255)
HOT = (255, 214, 10)
# path node layout (world px): a compact meander that fits a portrait screen at the finale
PATTERN = [(0, 0), (560, 40), (980, 420), (520, 760), (30, 700), (260, 1180), (900, 1150)]


def _positions(n):
    pos = []
    for k in range(n):
        blk, j = divmod(k, len(PATTERN))
        x, y = PATTERN[j]
        if blk % 2:
            x = 980 - x
        pos.append((float(x), float(y + blk * 1650)))
    return pos


def make(ctx):
    people = list(ctx.C.people)
    N = len(people)
    idx = {nm: k for k, nm in enumerate(people)}
    P = _positions(N)
    xs, ys = [p[0] for p in P], [p[1] for p in P]
    BB = (min(xs), min(ys), max(xs), max(ys))
    rnd = random.Random(11)

    def flo(k, t, amp=1.0):
        """Gentle float of node k: two slow sines with per-node phases."""
        return (amp * 16 * math.sin(t * 0.8 + k * 1.7) + amp * 6 * math.sin(t * 1.9 + k * 0.6),
                amp * 13 * math.cos(t * 0.67 + k * 2.3) + amp * 5 * math.cos(t * 1.5 + k * 1.1))

    # ---------- near decoys: grey people-nodes floating around the path, with faint edges
    DEC = []
    tries = 0
    while len(DEC) < 70 and tries < 5000:
        tries += 1
        x = rnd.uniform(BB[0] - 900, BB[2] + 900)
        y = rnd.uniform(BB[1] - 1100, BB[3] + 1100)
        if min(math.dist((x, y), p) for p in P) < 230 or any(math.dist((x, y), d) < 190 for d in DEC):
            continue
        # keep the path's own edges clear of clutter
        ok = True
        for a, b in zip(P[:-1], P[1:]):
            ux, uy = b[0] - a[0], b[1] - a[1]
            L2 = ux * ux + uy * uy
            u = max(0.0, min(1.0, ((x - a[0]) * ux + (y - a[1]) * uy) / L2))
            if math.dist((x, y), (a[0] + ux * u, a[1] + uy * u)) < 130:
                ok = False
                break
        if ok:
            DEC.append((x, y))
    nodes_all = [("d", k, p) for k, p in enumerate(DEC)] + [("p", k, p) for k, p in enumerate(P)]
    DEDGE = set()
    for k, p in enumerate(DEC):
        near = sorted(range(len(nodes_all)), key=lambda j: math.dist(p, nodes_all[j][2]))[1:4]
        for j in near[: rnd.choice([1, 2, 2, 3])]:
            if math.dist(p, nodes_all[j][2]) < 520:
                DEDGE.add(tuple(sorted([("d", k), nodes_all[j][:2]])))
    DEDGE = sorted(DEDGE)
    DSIZE = [rnd.choice([0.5, 0.6, 0.7, 0.8]) for _ in DEC]

    # ---------- far layer: a much bigger web, dim and soft (parallax)
    M = 2600
    FOX, FOY = BB[0] - M, BB[1] - M
    FW, FH = int(BB[2] - BB[0] + 2 * M), int(BB[3] - BB[1] + 2 * M)

    def build_far():
        im = Image.new("RGB", (FW, FH), BG)
        d = ImageDraw.Draw(im)
        pts = [(rnd.uniform(0, FW), rnd.uniform(0, FH)) for _ in range(900)]
        grid = {}
        for k, (x, y) in enumerate(pts):
            grid.setdefault((int(x // 300), int(y // 300)), []).append(k)
        for k, (x, y) in enumerate(pts):
            gx, gy = int(x // 300), int(y // 300)
            cand = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in grid.get((gx + dx, gy + dy), []) if j != k]
            cand.sort(key=lambda j: math.dist(pts[k], pts[j]))
            for j in cand[:2]:
                d.line([pts[k], pts[j]], fill=(26, 32, 58), width=3)
        for x, y in pts:
            r = rnd.uniform(7, 15)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(38, 46, 76))
        im = im.filter(ImageFilter.GaussianBlur(2.5))
        return np.asarray(im)

    FAR = build_far()

    # ---------- sprites
    NR = 74                                                    # node radius at zoom 1

    def glow(col, r):
        S = int(r * 3.2)
        im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        ImageDraw.Draw(im).ellipse([S / 2 - r * 1.15, S / 2 - r * 1.15, S / 2 + r * 1.15, S / 2 + r * 1.15],
                                   fill=col + (150,))
        return im.filter(ImageFilter.GaussianBlur(r * 0.35))

    def node_img(nm, state):
        """state: 'face' (visited), 'hot' (just revealed), 'q' (incoming, unknown)."""
        col = {"face": PATH, "hot": HOT, "q": (150, 160, 190)}[state]
        g = glow(col, NR)
        c = g.width / 2
        d = ImageDraw.Draw(g)
        d.ellipse([c - NR, c - NR, c + NR, c + NR], fill=col)
        r = NR - 9
        if state == "q":
            d.ellipse([c - r, c - r, c + r, c + r], fill=(34, 40, 66))
            q = ctx.pixel_text("?", 56, fill=ctx.WHITE)
            g.alpha_composite(q, (int(c - q.width / 2), int(c - q.height / 2)))
        else:
            face = ctx.sharp[nm].convert("RGBA").resize((2 * r, 2 * r), Image.BICUBIC)
            m = Image.new("L", face.size, 0)
            ImageDraw.Draw(m).ellipse([0, 0, 2 * r - 1, 2 * r - 1], fill=255)
            g.paste(face, (int(c - r), int(c - r)), m)
        return g

    NODE = {(nm, st): node_img(nm, st) for nm in people for st in ("face", "hot", "q")}

    def decoy_img(r):
        S = 2 * r + 4
        im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.ellipse([2, 2, S - 2, S - 2], fill=DECOY)
        d.ellipse([7, 7, S - 7, S - 7], fill=(44, 52, 80))
        c = S / 2                                              # a generic head-and-shoulders silhouette
        d.ellipse([c - r * 0.28, c - r * 0.55, c + r * 0.28, c - r * 0.02], fill=DECOY)
        d.chord([c - r * 0.55, c + r * 0.05, c + r * 0.55, c + r * 1.0], 180, 360, fill=DECOY)
        m = Image.new("L", im.size, 0)
        ImageDraw.Draw(m).ellipse([2, 2, S - 2, S - 2], fill=255)
        out = Image.new("RGBA", im.size, (0, 0, 0, 0))
        out.paste(im, (0, 0), m)
        return out

    DIMG = decoy_img(60)

    _tags = {}

    def tag(nm, col):
        if (nm, col) not in _tags:
            _tags[(nm, col)] = ctx.pixel_tag(ctx.C.labels[nm].upper(), col, 24)
        return _tags[(nm, col)]

    _cards, _thumbs = {}, {}

    def card(i, revealed):
        key = (i, revealed)
        if key in _cards:
            return _cards[key]
        PW, PH = 800, 420
        an, bn = ctx.link_names(i)
        im = Image.new("RGBA", (PW + 60, PH + 190), (0, 0, 0, 0))
        g = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(g).rounded_rectangle([14, 14, PW + 46, PH + 176], 22, fill=PATH + (170,))
        im.alpha_composite(g.filter(ImageFilter.GaussianBlur(10)))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([20, 20, PW + 40, PH + 170], 18, fill=(18, 24, 48), outline=PATH, width=5)
        ta = tag(an, ctx.CYAN)
        tb = tag(bn, ctx.YELLOW) if revealed else ctx.pixel_tag("? ? ?", (150, 160, 190), 24)
        plus = ctx.pixel_text("+", 30, fill=ctx.WHITE)
        tot = ta.width + plus.width + tb.width + 10
        sc = min(1.0, (PW - 20) / tot)
        x = 30 + (PW + 20 - tot * sc) / 2
        for t_ in (ta, plus, tb):
            t2 = t_.resize((int(t_.width * sc), int(t_.height * sc)), Image.NEAREST) if sc < 1 else t_
            im.alpha_composite(t2, (int(x), int(62 - t2.height / 2)))
            x += t2.width + 5 * sc
        ph = ctx.view(i, *ctx.both_cam(i, PW, PH, 0.8), PW, PH)
        im.alpha_composite(ph, (30, 100))
        d.rectangle([30, 100, 30 + PW, 100 + PH], outline=PATH, width=2)
        words, lines, cur = ctx.fact_of(ctx.spec["links"][i]).upper().split(), [], ""
        for w in words:
            if cur and len(cur) + len(w) + 1 > 38:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        lines.append(cur)
        f = ctx.pf(19)
        for k, ln in enumerate(lines[:2]):
            d.text((44, PH + 114 + k * 28), ln, font=f, fill=(220, 230, 255))
        _cards[key] = im
        return im

    def thumb(i):
        if i not in _thumbs:
            ph = ctx.view(i, *ctx.both_cam(i, 150, 100, 0.8), 150, 100)
            im = Image.new("RGBA", (160, 110), PATH + (255,))
            im.alpha_composite(ph, (5, 5))
            _thumbs[i] = im
        return _thumbs[i]

    # ---------- the clock
    segs = [s for s in ctx.segs if s["kind"] == "link"]

    def rev_t(s):
        return s.get("rv", ctx.REVEAL) * s["speech"] + ctx.THROW

    def card_out(s):
        dur = s["t1"] - s["t0"]
        if s["i"] == ctx.n_links - 1:
            return max(0.9, rev_t(s) - 0.55)
        return max(rev_t(s) + 0.4, min(dur - 0.5, rev_t(s) + 1.3))

    def seg_at(T):
        for s in segs:
            if T < s["t1"] or s is segs[-1]:
                return s
        return segs[-1]

    # finale framing: the gap between the last two rows of nodes sits behind the CONNECTED! stamp (y 1020)
    gap_y = (BB[3] + max([y for y in ys if y < BB[3] - 200] or [BB[1]])) / 2 + 45
    FIT_Z = min((ctx.SAFE[2] - ctx.SAFE[0] - 260) / max(1, BB[2] - BB[0]),
                (1020 - 330) / max(1, gap_y - BB[1]), (1260 - 1020 - 90) / max(1, BB[3] - gap_y), 0.9)
    FIT = ((BB[0] + BB[2]) / 2, gap_y, FIT_Z, 1020.0)
    if FIT_Z < 0.4:
        FIT_Z = min((ctx.SAFE[2] - ctx.SAFE[0] - 260) / max(1, BB[2] - BB[0]), 900 / max(1, BB[3] - BB[1]))
        FIT = ((BB[0] + BB[2]) / 2, (BB[1] + BB[3]) / 2, FIT_Z, 760.0)

    def target(T):
        """Camera target (cx, cy, zoom, screen anchor y) at time T, before smoothing."""
        s = seg_at(T)
        local = T - s["t0"]
        if T < s["t0"]:
            a = idx.get(ctx.link_names(s["i"])[0], 0)
            return P[a][0], P[a][1], 0.95, 560.0
        a, b = (idx.get(n, 0) for n in ctx.link_names(s["i"]))
        A, B = P[a], P[b]
        if s["i"] == ctx.n_links - 1 and local > card_out(s) - 0.1:
            return FIT
        if local < card_out(s):
            dx, dy = abs(B[0] - A[0]) + 1, abs(B[1] - A[1]) + 1
            z = max(0.5, min(0.95, 560 / dx, 250 / dy))
            return (A[0] + B[0]) / 2, (A[1] + B[1]) / 2, z, 480.0
        return B[0], B[1], 1.0, 720.0

    def cam(T):
        n = 9
        acc = [0.0, 0.0, 0.0, 0.0]
        for k in range(n):
            x, y, z, ay = target(T - 0.75 * k / (n - 1))
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
        rv = rev_t(seg)
        revealed = local > rv
        last = i == ctx.n_links - 1
        an, bn = ctx.link_names(i)
        ia, ib = idx.get(an, 0), idx.get(bn, 0)
        cx, cy, z, ay = cam(tt)
        cx += 20 * math.sin(tt * 0.5)                         # the camera never quite settles
        cy += 14 * math.cos(tt * 0.43)
        z *= 1 + 0.025 * math.sin(tt * 0.9)
        ax = ctx.CX
        fin = 0.0
        if last:
            fin = ctx.ease(max(0.0, min(1.0, (local - card_out(seg) + 0.1) / 0.8)))

        def scr(p):
            return (p[0] - cx) * z + ax, (p[1] - cy) * z + ay

        # far web, parallax
        fz = z * 0.75
        fcx, fcy = cx * 0.6, cy * 0.6
        Mx = np.float32([[fz, 0, fz * (FOX - fcx) + ax], [0, fz, fz * (FOY - fcy) + ay]])
        base = Image.fromarray(cv2.warpAffine(FAR, Mx, (ctx.W, ctx.H), flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=BG)).convert("RGBA")
        d = ImageDraw.Draw(base)

        # positions of everything that exists now
        pos = {}
        for k, p in enumerate(DEC):
            f = flo(k + 40, tt, 1.3)
            pos[("d", k)] = scr((p[0] + f[0], p[1] + f[1]))
        for k in range(N):
            f = flo(k, tt)
            pos[("p", k)] = scr((P[k][0] + f[0], P[k][1] + f[1]))
        # the incoming node B flies in from beyond itself
        fly = ctx.ease_out_back(max(0.0, min(1.0, (local - 0.05) / 0.55)))
        A, B = P[ia], P[ib]
        ux, uy = B[0] - A[0], B[1] - A[1]
        L = math.hypot(ux, uy) or 1.0
        start = (B[0] + ux / L * 900 + 300, B[1] + uy / L * 900 - 500)
        fb = flo(ib, tt)
        bw = (start[0] + (B[0] - start[0]) * fly + fb[0], start[1] + (B[1] - start[1]) * fly + fb[1])
        pos[("p", ib)] = scr(bw)
        exists = set(range(0, ib + 1)) if ib > ia else set(range(0, ia + 1)) | {ib}

        # decoy edges (and decoys hooked onto existing path nodes)
        ew = max(2, int(3 * z))
        for a, b in DEDGE:
            if b[0] == "p" and b[1] not in exists:
                continue
            pa, pb = pos[a], pos[b]
            if (min(pa[0], pb[0]) > ctx.W + 50 or max(pa[0], pb[0]) < -50 or min(pa[1], pb[1]) > ctx.H + 50
                    or max(pa[1], pb[1]) < -50):
                continue
            d.line([pa, pb], fill=DECOY_EDGE, width=ew)
        ds = max(0.35, z * 0.75)
        for k in range(len(DEC)):
            x, y = pos[("d", k)]
            if -80 < x < ctx.W + 80 and -80 < y < ctx.H + 80:
                ctx.paste(base, DIMG, x, y, scale=ds * DSIZE[k], nearest=False)

        # path edges: done ones glow cyan, the new one draws itself from A to B
        def edge(pa, pb, col, w, u=1.0, pulse=True):
            pb2 = (pa[0] + (pb[0] - pa[0]) * u, pa[1] + (pb[1] - pa[1]) * u)
            d.line([pa, pb2], fill=tuple(int(c * 0.35 + b * 0.65) for c, b in zip(col, BG)), width=int(w * 2.4))
            d.line([pa, pb2], fill=col, width=w)
            if pulse:
                for q in range(3):
                    v = ((tt * 0.6 + q / 3) % 1.0) * u
                    if v < u:
                        px, py = pa[0] + (pb[0] - pa[0]) * v, pa[1] + (pb[1] - pa[1]) * v
                        r = w * 0.9
                        d.ellipse([px - r, py - r, px + r, py + r], fill=(255, 255, 255))

        pw = max(4, int(9 * z))
        for k in range(ctx.n_links):
            if k == i:
                continue
            a, b = (idx.get(n, 0) for n in ctx.link_names(k))
            if k < i:
                edge(pos[("p", a)], pos[("p", b)], HOT if fin > 0.3 else PATH, pw)
        u = ctx.ease(max(0.0, min(1.0, (local - 0.25) / 0.5)))
        if u > 0:
            edge(pos[("p", ia)], pos[("p", ib)], HOT, pw + 2, u)
        # photo thumbnails sitting on finished edges
        co = card_out(seg)
        for k in range(i + 1):
            if k == i and local < co + 0.3:
                continue
            a, b = (idx.get(n, 0) for n in ctx.link_names(k))
            pa, pb = pos[("p", a)], pos[("p", b)]
            mx, my = pa[0] + (pb[0] - pa[0]) * 0.55, pa[1] + (pb[1] - pa[1]) * 0.55
            ts = max(0.45, min(0.8, z * 0.8)) * (1 - 0.25 * fin)
            if k == i:
                ts *= ctx.ease_out_back(min(1.0, (local - co - 0.3) / 0.25))
            ctx.paste(base, thumb(k), mx, my, scale=ts, nearest=False)

        # path nodes and their names
        ns = max(0.6, min(1.0, z * 1.05))
        for k in sorted(exists):
            nm = people[k]
            x, y = pos[("p", k)]
            st = "face"
            s = ns
            if k == ib:
                st = "hot" if revealed else "q"
                if revealed:
                    s *= 1 + 0.3 * math.sin(math.pi * min(1.0, (local - rv) / 0.3))
                else:
                    s *= 0.85 + 0.15 * fly + 0.04 * math.sin(tt * 8)
            elif fin > 0.3:
                st = "hot"
            ctx.paste(base, NODE[(nm, st)], x, y, scale=s, nearest=False)
        for k in sorted(exists):
            if k == ib and not revealed or (k not in (ia, ib) and fin < 0.3):
                continue                                       # older nodes keep just their faces (less clutter)
            nm = people[k]
            x, y = pos[("p", k)]
            tg = tag(nm, ctx.YELLOW if (k == ib or fin > 0.3) else ctx.CYAN)
            ls = 0.8 + 0.2 * (1 - fin)
            ly = y + NR * ns + tg.height * ls / 2 - 2
            lx = max(ctx.SAFE[0] + tg.width * ls / 2, min(ctx.SAFE[2] - tg.width * ls / 2, x))
            if ctx.SAFE[0] - 80 < x < ctx.SAFE[2] + 80 and ctx.BAND - 20 < ly - tg.height * ls / 2 and ly < ctx.SAFE[3] - 30:
                ctx.paste(base, tg, lx, ly, scale=ls)

        # the photo card hangs off the new edge, then shrinks into it
        cin = ctx.ease_out_back(max(0.0, min(1.0, (local - 0.35) / 0.3)))
        shr = ctx.ease(max(0.0, min(1.0, (local - co) / 0.35)))
        if cin > 0.01 and shr < 0.99:
            pa, pb = pos[("p", ia)], pos[("p", ib)]
            em = (pa[0] + (pb[0] - pa[0]) * 0.55, pa[1] + (pb[1] - pa[1]) * 0.55)
            cd = card(i, revealed)
            home = (ctx.CX, 1028 + 6 * math.sin(tt * 2.2))
            cxp = home[0] + (em[0] - home[0]) * shr
            cyp = home[1] + (em[1] - home[1]) * shr
            sc = (0.92 + 0.08 * cin) * (1 - 0.88 * shr)
            if revealed and local - rv < 0.25:
                sc *= 1 + 0.05 * math.sin(math.pi * (local - rv) / 0.25)
            top = (cxp, cyp - cd.height * sc / 2 + 20 * sc)
            if shr < 0.05 and cin > 0.5:                      # a tether from the card to its edge
                d.line([em, top], fill=PATH, width=4)
                d.ellipse([em[0] - 9, em[1] - 9, em[0] + 9, em[1] + 9], fill=PATH)
            ctx.paste(base, cd, cxp, cyp, scale=sc * min(1.0, cin), nearest=False)

        ctx.header(base, i)
        ctx.connected(base, seg, tt)
        seg["cap"].draw(base, tt, y=1368)
        return base

    return frame
