"""Social feed: each link is a post by A (avatar, name, the photo, the fact as the caption). B gets tagged ("with B"
plus a name tag on the photo) at the reveal, a big heart pops and the likes jump; the feed scrolls to the next post,
which is B's."""
from __future__ import annotations

import hashlib
import math

from PIL import Image, ImageDraw, ImageFont

NAME = "feed"

BG = (14, 14, 17)
FG = (245, 245, 247)
MUTED = (150, 152, 160)
LIKE = (255, 48, 84)
X0, X1 = 64, 926                    # post column (inside the safe zone)
PW, PH = X1 - X0, 640               # photo
PHOTO_Y = 108                       # photo top inside a post
POST_H = 1620                       # post pitch (the next post stays off-screen until the scroll)
TOP = 342                           # where the current post's top sits on screen
APPBAR = (230, 318)
BRAND = "chainline"


def make(ctx):
    _fonts, _posts, _sprites = {}, {}, {}

    def F(size, weight="Bold"):
        k = (size, weight)
        if k not in _fonts:
            _fonts[k] = ImageFont.truetype(str(ctx.FONTS / f"Inter-{weight}.otf"), size)
        return _fonts[k]

    def label(n):
        return ctx.C.labels.get(n, n)

    def seed(i, k=""):
        return int(hashlib.sha1(f"{ctx.spec['from']}{i}{k}".encode()).hexdigest(), 16)

    def heart_pts(cx, cy, s, n=60):
        pts = []
        for k in range(n):
            a = 2 * math.pi * k / n
            x = 16 * math.sin(a) ** 3
            y = 13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a)
            pts.append((cx + x * s / 32, cy - y * s / 32 - s * 0.05))
        return pts

    def heart(size, fill, outline=None, width=0):
        k = (size, fill, outline, width)
        if k not in _sprites:
            ss = 4                                                 # supersample for smooth edges
            im = Image.new("RGBA", (size * ss + 8, size * ss + 8), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            pts = heart_pts(im.width / 2, im.height / 2, size * ss)
            if fill:
                d.polygon(pts, fill=fill)
            if outline:
                d.line(pts + pts[:1], fill=outline, width=width * ss, joint="curve")
            _sprites[k] = im.resize((size + 2, size + 2), Image.LANCZOS)
        return _sprites[k]

    def circle_face(name, size):
        face = ctx.sharp[name].convert("RGBA").resize((size, size), Image.BICUBIC)
        m = Image.new("L", (size * 4, size * 4), 0)
        ImageDraw.Draw(m).ellipse([0, 0, size * 4 - 1, size * 4 - 1], fill=255)
        face.putalpha(m.resize((size, size), Image.LANCZOS))
        return face

    def avatar(name, size=78):
        """Face in a gradient ring (the 'has a story' ring)."""
        k = ("av", name, size)
        if k not in _sprites:
            R = size + 14
            im = Image.new("RGBA", (R * 4, R * 4), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            for a in range(0, 360, 3):                         # warm gradient ring
                u = a / 360
                col = (255, int(60 + 150 * u), int(120 - 90 * u))
                d.pieslice([0, 0, R * 4 - 1, R * 4 - 1], a - 90, a - 87 + 1, fill=col)
            d.ellipse([12, 12, R * 4 - 13, R * 4 - 13], fill=BG)
            im = im.resize((R, R), Image.LANCZOS)
            im.alpha_composite(circle_face(name, size), (7, 7))
            _sprites[k] = im
        return _sprites[k]

    def wrap(text, font, maxw, first_indent=0, max_lines=2):
        lines, cur, ind = [], "", first_indent
        for w in text.split():
            t = f"{cur} {w}".strip()
            if cur and font.getlength(t) > maxw - ind:
                lines.append(cur)
                cur, ind = w, 0
            else:
                cur = t
        if cur:
            lines.append(cur)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1].rstrip(",;:") + "…"
        return lines

    def icons(d, y, liked=False):
        """Comment and share icons (the heart is drawn per frame)."""
        x = X0 + 74
        d.ellipse([x, y - 21, x + 42, y + 19], outline=FG, width=4)               # comment bubble
        d.polygon([(x + 6, y + 10), (x + 2, y + 24), (x + 16, y + 17)], fill=FG)
        x += 72
        d.polygon([(x, y - 18), (x + 44, y - 20), (x + 20, y + 22), (x + 16, y + 2)], outline=FG, width=4)
        d.line([(x + 16, y + 2), (x + 44, y - 20)], fill=FG, width=4)               # share plane
        d.polygon([(X1 - 34, y - 21), (X1, y - 21), (X1, y + 22), (X1 - 17, y + 8), (X1 - 34, y + 22)],
                  outline=FG, width=4)                                              # bookmark

    def post(i):
        """Static post image (opaque, BG) and its geometry."""
        if i in _posts:
            return _posts[i]
        an, bn = ctx.link_names(i)
        link = ctx.spec["links"][i]
        im = Image.new("RGBA", (ctx.W, POST_H), BG + (255,))
        d = ImageDraw.Draw(im)
        im.alpha_composite(avatar(an), (X0 - 4, 6))
        nf = F(32)
        name = label(an)
        while nf.size > 24 and nf.getlength(name) > 560:
            nf = F(nf.size - 2)
        d.text((X0 + 102, 34), name, font=nf, fill=FG, anchor="lm")
        bx = X0 + 102 + nf.getlength(name) + 12                              # verified badge
        d.ellipse([bx, 20, bx + 28, 48], fill=(56, 151, 240))
        d.line([(bx + 7, 34), (bx + 12, 40), (bx + 21, 28)], fill=(255, 255, 255), width=4)
        for k in range(3):
            d.ellipse([X1 - 40 + k * 14, 32, X1 - 32 + k * 14, 40], fill=FG)
        cam = ctx.both_cam(i, PW, PH, 0.8)
        ph = ctx.view(i, *cam, PW, PH)
        m = Image.new("L", (PW, PH), 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, PW - 1, PH - 1], 26, fill=255)
        im.paste(ph, (X0, PHOTO_Y), m)
        ay = PHOTO_Y + PH + 50
        icons(d, ay)
        # carousel dots
        for k in range(3):
            d.ellipse([ctx.W / 2 - 50 + k * 22, ay - 5, ctx.W / 2 - 40 + k * 22, ay + 5],
                      fill=(56, 151, 240) if k == 0 else (80, 82, 90))
        # caption: name in bold + the fact
        cf = F(28)
        fact = ctx.fact_of(link)
        lead = label(an) + "  "
        lines = wrap(fact, cf, X1 - X0, cf.getlength(lead), 2)
        cy = ay + 108
        d.text((X0, cy), lead, font=cf, fill=FG)
        for k, ln in enumerate(lines):
            d.text((X0 + (cf.getlength(lead) if k == 0 else 0), cy + k * 40), ln, font=cf,
                   fill=(214, 216, 222))
        cy += 40 * len(lines) + 12
        nc = 200 + seed(i, "c") % 2400
        d.text((X0, cy), f"View all {nc:,} comments", font=F(26), fill=MUTED)
        # a few comments (below the fold, under the Shorts UI)
        yy = cy + 64
        for k in range(4):
            r = seed(i, f"cm{k}")
            col = ((r >> 3) % 160 + 60, (r >> 9) % 160 + 60, (r >> 15) % 160 + 60)
            d.ellipse([X0, yy, X0 + 48, yy + 48], fill=col)
            d.rounded_rectangle([X0 + 64, yy + 4, X0 + 64 + 120 + r % 80, yy + 20], 8, fill=(70, 72, 80))
            d.rounded_rectangle([X0 + 64, yy + 30, X0 + 64 + 300 + (r >> 4) % 420, yy + 44], 7, fill=(46, 48, 54))
            yy += 82
        # where the faces are on the photo (for the name tags)
        tags = {}
        for role in ("a", "b"):
            fb = ctx.fbox(i, role)
            fx, fy = ctx.ctr(fb)
            tags[role] = (X0 + PW / 2 + (fx - cam[0]) * cam[2], PHOTO_Y + PH / 2 + (fy - cam[1]) * cam[2],
                          fb[3] * cam[2])
        _posts[i] = {"im": im, "ay": ay, "tags": tags, "likes": 8000 + seed(i, "l") % 90000}
        return _posts[i]

    def tag_bubble(text):
        k = ("tag", text)
        if k not in _sprites:
            f = F(28)
            w = int(f.getlength(text)) + 36
            im = Image.new("RGBA", (w, 70), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.polygon([(w / 2 - 12, 14), (w / 2 + 12, 14), (w / 2, 0)], fill=(10, 10, 12, 215))
            d.rounded_rectangle([0, 13, w - 1, 69], 14, fill=(10, 10, 12, 215))
            d.text((w / 2, 42), text, font=f, fill=(255, 255, 255), anchor="mm")
            _sprites[k] = im
        return _sprites[k]

    def appbar():
        if "bar" not in _sprites:
            im = Image.new("RGBA", (ctx.W, APPBAR[1] - APPBAR[0]), BG + (255,))
            d = ImageDraw.Draw(im)
            cy = im.height / 2
            d.text((X0, cy), BRAND, font=F(44, "Black"), fill=FG, anchor="lm")
            hx = X1 - 120
            im.alpha_composite(heart(42, None, FG, 3), (hx - 21, int(cy - 21)))
            d.ellipse([hx + 22, cy - 26, hx + 36, cy - 12], fill=LIKE)               # notification dot
            px = X1 - 40
            d.polygon([(px - 22, cy - 18), (px + 22, cy - 20), (px - 2, cy + 22), (px - 6, cy + 2)], outline=FG,
                      width=4)
            d.line([0, im.height - 1, ctx.W, im.height - 1], fill=(38, 38, 44), width=2)
            _sprites["bar"] = im
        return _sprites["bar"]

    def seg_of(j):
        return next((s for s in ctx.segs if s.get("kind") == "link" and s.get("i") == j), None)

    def draw_post(base, j, y, local, seg_j, active):
        P = post(j)
        base.paste(P["im"], (0, int(y)))
        sp, R = seg_j["speech"], seg_j.get("rv", ctx.REVEAL)
        trev = R * sp + ctx.THROW
        dur = seg_j["t1"] - seg_j["t0"]
        an, bn = ctx.link_names(j)
        d = ImageDraw.Draw(base)
        revealed = local > trev
        # subline under the poster's name: year, then "with B" once B is tagged
        yr = ctx.year_of(j)
        sub = f"with {label(bn)}" if revealed else (f"{yr} · Public" if yr else "Public")
        sx = X0 + 102
        if revealed:
            flash = max(0.0, 1 - (local - trev) / 0.5)              # the new tag flashes in the like colour
            col = tuple(int(FG[c] + (LIKE[c] - FG[c]) * flash) for c in range(3))
            f = F(26)
            d.text((sx, y + 74), "with ", font=f, fill=MUTED, anchor="lm")
            w0 = f.getlength("with ")
            d.text((sx + w0, y + 74), label(bn), font=f, fill=col, anchor="lm")
            if yr:
                d.text((sx + w0 + f.getlength(label(bn)) + 10, y + 74), f"· {yr}", font=f, fill=MUTED, anchor="lm")
        else:
            d.text((sx, y + 74), sub, font=F(26), fill=MUTED, anchor="lm")
        # name tags on the photo: A's early, B's at the reveal
        for role, t_on, nm in (("a", 0.35, an), ("b", trev, bn)):
            if local > t_on:
                fx, fy, fh = P["tags"][role]
                tg = tag_bubble(label(nm))
                v = ctx.ease_out_back(min(1.0, (local - t_on) / 0.25))
                tx = min(max(fx, X0 + tg.width / 2 + 10), X1 - tg.width / 2 - 10)
                ty = min(fy + fh * 0.62 + tg.height / 2, PHOTO_Y + PH - tg.height / 2 - 10)
                ctx.paste(base, tg, tx, y + ty + 4 * math.sin(local * 3 + (role == "b")), scale=0.6 + 0.4 * v,
                          alpha=min(1.0, v), nearest=False)
        # big double-tap heart over the photo at the reveal
        hp = local - trev - 0.05
        if 0 < hp < 0.9:
            s = ctx.ease_out_back(min(1.0, hp / 0.22)) * (1 + 0.15 * max(0.0, hp - 0.6) / 0.3)
            a = 1.0 if hp < 0.6 else 1 - (hp - 0.6) / 0.3
            ctx.paste(base, heart(260, (255, 255, 255, 235)), ctx.CX, y + PHOTO_Y + PH * 0.45, scale=s, alpha=a,
                      nearest=False)
        # like button + count ticking up (a jump at the heart)
        ay = y + P["ay"]
        liked = revealed
        hx, hy = X0 + 22, ay
        if liked:
            bump = 1 + 0.35 * math.sin(min(1.0, (local - trev) / 0.25) * math.pi)
            ctx.paste(base, heart(48, LIKE), hx, hy, scale=bump, nearest=False)
            for k in range(5):                                       # little hearts floating up
                q = (local - trev - 0.05 * k) / 1.4
                if 0 < q < 1:
                    ctx.paste(base, heart(22, LIKE), hx + 20 * math.sin(q * 6 + k * 1.7) + 6 * k,
                              hy - 30 - 260 * q, alpha=1 - q, nearest=False)
        else:
            ctx.paste(base, heart(48, None, FG, 4), hx, hy, nearest=False)
        n = P["likes"] + int(37 * local + 9 * local * local)
        if liked:
            n += int(P["likes"] * 0.18 * ctx.ease(min(1.0, (local - trev) / 0.6)))
        d.text((X0, ay + 58), f"{n:,} likes", font=F(29), fill=FG, anchor="lm")

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        dur = seg["t1"] - seg["t0"]
        base = Image.new("RGBA", (ctx.W, ctx.H), BG + (255,))
        # scroll: flick from the previous post to this one, then keep drifting slowly while reading
        prev_ok = i > 0 and not ctx.C.fj["links"][i - 1].get("lineage")
        flick = ctx.ease(local / 0.5) if prev_ok else 1.0
        drift = 36 * local / max(1.0, dur)
        y = TOP + POST_H * (1 - flick) - drift
        if prev_ok and flick < 1:
            ps = seg_of(i - 1)
            pl = ps["t1"] - ps["t0"]
            draw_post(base, i - 1, y - POST_H - 36, pl + local, ps, False)    # where it was when its link ended
        draw_post(base, i, y, local, seg, True)
        base.alpha_composite(appbar(), (0, APPBAR[0]))
        ctx.header(base, i)
        ctx.connected(base, seg, tt)
        seg["cap"].draw(base, tt, y=1400)
        return base

    return frame
