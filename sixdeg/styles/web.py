"""Web: "six degrees" as a growing network graph. Each person is a glowing node in a strip of web under the title;
the link photo is one big glowing panel below it, and A's and B's nodes each run a tether down into their own face
in it. Buh rides the new edge from A to B as it draws; finished edges carry their photos as thumbnails, and the
finale closes in on the destination node while the last photo stays up."""
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
PW, PH, GL = 860, 580, 24                # the photo panel (and its glow margin)
PH_Y0 = 590                             # panel top; the web strip sits above it, under the title band
PH_Y1 = PH_Y0 + PH
WEB_AY = 405
CAP_Y = 1384


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
            _tags[(nm, col)] = ctx.pixel_tag(ctx.C.labels[nm].upper(), col, 28)
        return _tags[(nm, col)]


    # ---------- the link photo: one big glowing "evidence" panel under the web; edges run from the nodes into it
    _fits, _cards, _thumbs = {}, {}, {}

    def fit(i):
        if i not in _fits:
            _fits[i] = photo_params(ctx, i, PW, PH)
        return _fits[i]

    def card(i, revealed):
        key = (i, revealed)
        if key in _cards:
            return _cards[key]
        cx, cy, s, ow, oh = fit(i)
        col = HOT if revealed else PATH
        im = Image.new("RGBA", (PW + 2 * GL, PH + 2 * GL), (0, 0, 0, 0))
        g = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(g).rounded_rectangle([GL - 10, GL - 10, GL + PW + 10, GL + PH + 10], 34, fill=col + (190,))
        im.alpha_composite(g.filter(ImageFilter.GaussianBlur(14)))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([GL - 6, GL - 6, GL + PW + 6, GL + PH + 6], 32, fill=(10, 14, 30), outline=col, width=6)
        ph = ctx.view(i, cx, cy, s, ow, oh)
        m = Image.new("L", (ow, oh), 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, ow - 1, oh - 1], 26, fill=255)
        im.paste(ph, (GL + (PW - ow) // 2, GL + (PH - oh) // 2), m)
        _cards[key] = im
        return im

    def face_x(i, role):
        """Screen x of person role's face inside the photo panel."""
        cx, cy, s, ow, oh = fit(i)
        fx = ctx.ctr(ctx.fbox(i, role))[0]
        return ctx.CX + (fx - cx) * s

    def thumb(i):
        if i not in _thumbs:
            cx, cy, s, ow, oh = photo_params(ctx, i, 150, 100)
            ph = ctx.view(i, cx, cy, s, ow, oh)
            im = Image.new("RGBA", (160, 110), PATH + (255,))
            im.paste((10, 14, 30), [5, 5, 155, 105])
            im.alpha_composite(ph, (5 + (150 - ow) // 2, 5 + (100 - oh) // 2))
            _thumbs[i] = im
        return _thumbs[i]

    # ---------- the clock
    segs = [s for s in ctx.segs if s["kind"] == "link"]

    def rev_t(s):
        return s.get("rv", ctx.REVEAL) * s["speech"] + ctx.THROW

    def seg_at(T):
        for s in segs:
            if T < s["t1"] or s is segs[-1]:
                return s
        return segs[-1]

    def target(T):
        """Camera (x, y, zoom, screen y) over the web strip: A and B of the link; on the last link it closes in on
        the destination node after the reveal."""
        s = seg_at(T)
        a, b = (idx.get(n, 0) for n in ctx.link_names(s["i"]))
        A, B = P[a], P[b]
        if s["i"] == ctx.n_links - 1 and T - s["t0"] > rev_t(s) - 0.1:
            return B[0], B[1], 1.15, WEB_AY - 10
        dx, dy = abs(B[0] - A[0]) + 1, abs(B[1] - A[1]) + 1
        z = max(0.5, min(0.85, 600 / dx, 235 / dy))
        return (A[0] + B[0]) / 2, (A[1] + B[1]) / 2, z, WEB_AY

    def cam(T):
        n = 9
        acc = [0.0, 0.0, 0.0, 0.0]
        for k in range(n):
            x, y, z, ay = target(T - 0.7 * k / (n - 1))
            acc[0] += x
            acc[1] += y
            acc[2] += math.log(z)
            acc[3] += ay
        return acc[0] / n, acc[1] / n, math.exp(acc[2] / n), acc[3] / n

    def place_tags(items):
        """items: [(img, face_x)] -> [(img, x, y)] under their faces, pushed apart, kept inside SAFE."""
        y = PH_Y1 + 36
        out = []
        items = sorted(items, key=lambda q: q[1])
        xs = [max(ctx.SAFE[0] + im.width / 2, min(ctx.SAFE[2] - im.width / 2, fx)) for im, fx in items]
        if len(items) == 2:
            (l, _), (r, _) = items
            gap = xs[1] - xs[0] - (l.width + r.width) / 2 - 14
            if gap < 0:
                xs[0] -= -gap / 2
                xs[1] += -gap / 2
                xs[0] = max(ctx.SAFE[0] + l.width / 2, xs[0])
                xs[1] = min(ctx.SAFE[2] - r.width / 2, xs[1])
        for k, (im, _) in enumerate(items):
            yy = y
            if k == 1 and xs[1] - xs[0] < (items[0][0].width + im.width) / 2 + 8:
                yy = y + im.height + 6                        # still too close: stack it below
            out.append((im, xs[k], yy))
        return out

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        rv = rev_t(seg)
        revealed = local > rv
        last = i == ctx.n_links - 1
        fin = last and revealed
        an, bn = ctx.link_names(i)
        ia, ib = idx.get(an, 0), idx.get(bn, 0)
        cx, cy, z, ay = cam(tt)
        cx += 14 * math.sin(tt * 0.5)                         # the camera never quite settles
        cy += 8 * math.cos(tt * 0.43)
        ax = ctx.CX

        def scr(p):
            return (p[0] - cx) * z + ax, (p[1] - cy) * z + ay

        # far web, parallax (fills the whole screen, top to bottom)
        fz = z * 0.75
        fcx, fcy = cx * 0.6, cy * 0.6
        Mx = np.float32([[fz, 0, fz * (FOX - fcx) + ax], [0, fz, fz * (FOY - fcy) + ay]])
        base = Image.fromarray(cv2.warpAffine(FAR, Mx, (ctx.W, ctx.H), flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REFLECT)).convert("RGBA")
        d = ImageDraw.Draw(base)

        pos = {}
        for k, p in enumerate(DEC):
            f = flo(k + 40, tt, 1.3)
            pos[("d", k)] = scr((p[0] + f[0], p[1] + f[1]))
        for k in range(N):
            f = flo(k, tt)
            pos[("p", k)] = scr((P[k][0] + f[0], P[k][1] + f[1]))
        exists = set(range(0, max(ia, ib) + 1))

        ew = max(2, int(3 * z))
        for a, b in DEDGE:
            if b[0] == "p" and b[1] not in exists:
                continue
            pa, pb = pos[a], pos[b]
            if min(pa[1], pb[1]) > PH_Y0 + 40 or max(pa[1], pb[1]) < 150:
                continue
            d.line([pa, pb], fill=DECOY_EDGE, width=ew)
        ds = max(0.35, z * 0.75)
        for k in range(len(DEC)):
            x, y = pos[("d", k)]
            if -80 < x < ctx.W + 80 and 150 < y < PH_Y0 + 80:
                ctx.paste(base, DIMG, x, y, scale=ds * DSIZE[k], nearest=False)

        def edge(pa, pb, col, w, u=1.0, pulse=True):
            pb2 = (pa[0] + (pb[0] - pa[0]) * u, pa[1] + (pb[1] - pa[1]) * u)
            d.line([pa, pb2], fill=tuple(int(c * 0.35 + b * 0.65) for c, b in zip(col, BG)), width=int(w * 2.4))
            d.line([pa, pb2], fill=col, width=w)
            if pulse:
                for q in range(3):
                    v = ((tt * 0.6 + q / 3) % 1.0) * u
                    px, py = pa[0] + (pb[0] - pa[0]) * v, pa[1] + (pb[1] - pa[1]) * v
                    r = w * 0.9
                    d.ellipse([px - r, py - r, px + r, py + r], fill=(255, 255, 255))

        pw = max(4, int(9 * z))
        for k in range(i):
            a, b = (idx.get(n, 0) for n in ctx.link_names(k))
            edge(pos[("p", a)], pos[("p", b)], HOT if fin else PATH, pw)
        u = ctx.ease(max(0.0, min(1.0, (local - 0.1) / max(0.3, rv - 0.1))))   # the new edge reaches B at the reveal
        pa_, pb_ = pos[("p", ia)], pos[("p", ib)]
        if u > 0:
            edge(pa_, pb_, HOT, pw + 2, u)
        # finished edges carry their photos as little thumbnails
        ts = max(0.45, min(0.75, z * 0.8))
        for k in range(i):
            a, b = (idx.get(n, 0) for n in ctx.link_names(k))
            pa, pb = pos[("p", a)], pos[("p", b)]
            ctx.paste(base, thumb(k), pa[0] + (pb[0] - pa[0]) * 0.5, pa[1] + (pb[1] - pa[1]) * 0.5, scale=ts,
                      nearest=False)

        # tethers: node A and node B each run a line down into their own face in the photo
        k_in = ctx.ease(max(0.0, min(1.0, local / 0.3)))
        tk = ctx.ease(max(0.0, min(1.0, (local - 0.15) / 0.35)))
        for role, k, col in (("a", ia, PATH), ("b", ib, HOT if revealed else (150, 160, 190))):
            nx, ny = pos[("p", k)]
            fx = face_x(i, role)
            ex, ey = nx + (fx - nx) * tk, ny + (PH_Y0 - ny) * tk
            if tk > 0.02:
                d.line([(nx, ny), (ex, ey)], fill=tuple(int(c * 0.4 + b * 0.6) for c, b in zip(col, BG)), width=10)
                d.line([(nx, ny), (ex, ey)], fill=col, width=4)
                d.ellipse([ex - 10, ey - 10, ex + 10, ey + 10], fill=col)

        # path nodes (faces only: the names live under the photo)
        ns = max(0.6, min(1.05, z * 1.05))
        for k in sorted(exists):
            nm = people[k]
            x, y = pos[("p", k)]
            st_ = "face"
            s = ns
            if k == ib:
                st_ = "hot" if revealed else "q"
                if revealed:
                    s *= 1 + 0.3 * math.sin(math.pi * min(1.0, (local - rv) / 0.3))
                    if last:
                        s *= 1 + 0.35 * ctx.ease(min(1.0, (local - rv) / 0.4))
                else:
                    s *= 1 + 0.04 * math.sin(tt * 8)
            elif fin:
                st_ = "hot"
            ctx.paste(base, NODE[(nm, st_)], x, y, scale=s, nearest=False)

        # Buh rides the tip of the new edge toward B, then hops beside the node
        ub = 0.2 + 0.6 * u
        bx, by = pa_[0] + (pb_[0] - pa_[0]) * ub, pa_[1] + (pb_[1] - pa_[1]) * ub
        expr, squash = "happy", 0.0
        if revealed:
            ar = local - rv
            expr = "shock" if ar < 0.5 else "smug"
            squash = 0.25 * math.sin(math.pi * min(1.0, ar / 0.3)) if ar < 0.3 else 0.0
        bob = 6 * abs(math.sin(tt * 5))
        look = 1 if pb_[0] >= pa_[0] else -1
        ctx.paste_pin(base, ctx.pin_sprite(expr, 6 * math.sin(tt * 3), squash, look, scale=3), bx, by - bob + 10)

        # the photo panel: always on screen, cross-fading from the last link's photo
        hy = PH_Y0 + PH / 2
        if i > 0 and k_in < 1 and not ctx.C.fj["links"][i - 1].get("lineage"):
            ctx.paste(base, card(i - 1, True), ctx.CX, hy, nearest=False)
            ctx.paste(base, card(i, revealed), ctx.CX, hy, alpha=k_in, nearest=False)
        elif i == 0 and k_in < 1:
            ctx.paste(base, card(i, revealed), ctx.CX, hy + 50 * (1 - k_in), scale=0.94 + 0.06 * k_in, alpha=k_in,
                      nearest=False)
        else:
            ps = 1 + (0.03 * math.sin(math.pi * (local - rv) / 0.25) if revealed and local - rv < 0.25 else 0)
            ctx.paste(base, card(i, revealed), ctx.CX, hy, scale=ps, nearest=False)

        # name tags under each face
        ta = tag(an, ctx.CYAN)
        tb = tag(bn, ctx.YELLOW) if revealed else ctx.pixel_tag("? ? ?", (150, 160, 190), 28)
        for im, x, y in place_tags([(ta, face_x(i, "a")), (tb, face_x(i, "b"))]):
            ctx.paste(base, im, x, y, alpha=k_in)

        ctx.header(base, i)
        ctx.connected(base, seg, tt, y=PH_Y0 + PH * 0.8)
        seg["cap"].draw(base, tt, y=CAP_Y)
        return base

    return frame
