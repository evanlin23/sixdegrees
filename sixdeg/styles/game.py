"""Game: Buh dashes through a side-scrolling pixel platformer; each photo is a checkpoint billboard in the level.

He runs in from the last checkpoint (hopping a pipe on the way), stops under a "?" block, and at the reveal jumps,
bonks it and a coin pops out (the B tag lands with the coin). Parallax sky/hills/ground are drawn once at quarter
resolution and scrolled; a WORLD / COINS / SCORE HUD ticks up. The last link ends on LEVEL CLEAR + CONNECTED!.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

NAME = "game"

Q = 4                     # background pixel size (drawn at 1/Q resolution, scaled up NEAREST)
SPACING = 1400            # world px between checkpoints
WALK = 0.85               # s for Buh's dash from one checkpoint to the next
GY = 1180                 # ground top (screen y)
STOP_DX = -440            # Buh stops this far left of a billboard centre (world px)
BOARD_SX = 560            # billboard centre on screen once Buh has arrived
BOARD_CY = 640            # photo centre y
PW, PH = 640, 470         # photo size
BLOCK_Y = 770             # "?" block centre y
TAG_A_Y, TAG_B_Y = 962, 1040
CAP_Y = 1335


def _rng(seed):
    return np.random.default_rng(seed)


def make(ctx):
    QW, QH = ctx.W // Q, ctx.H // Q
    qGY = GY // Q
    INK = ctx.INK
    _cache = {}

    # ---------------------------------------------------------------- static quarter-res layers
    def sky():
        if "sky" not in _cache:
            a = np.zeros((QH, QW, 3), np.uint8)
            top, bot = np.array([72, 132, 236]), np.array([170, 214, 255])
            for y in range(QH):
                k = min(1.0, y / (qGY * 0.95))
                k = round(k * 9) / 9                      # banded, 8-bit style gradient
                a[y] = top + (bot - top) * k
            _cache["sky"] = Image.fromarray(a).convert("RGBA")
        return _cache["sky"]

    def clouds():
        if "clouds" not in _cache:
            TW, TH = 540, 130
            im = Image.new("RGBA", (TW, TH), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            r = _rng(7)
            for k in range(6):
                x, y = int(k * TW / 6 + r.integers(0, 40)), int(r.integers(12, TH - 40))
                for dx, dy, rr in ((0, 8, 12), (12, 0, 15), (26, 6, 12), (36, 10, 9)):
                    d.ellipse([x + dx - rr, y + dy - rr, x + dx + rr, y + dy + rr], fill=(40, 70, 140, 255))
                for dx, dy, rr in ((0, 8, 11), (12, 0, 14), (26, 6, 11), (36, 10, 8)):
                    d.ellipse([x + dx - rr, y + dy - rr, x + dx + rr, y + dy + rr], fill=(255, 255, 255, 255))
                d.rectangle([x - 6, y + 14, x + 40, y + 18], fill=(214, 230, 250, 255))
            _cache["clouds"] = im
        return _cache["clouds"]

    def far_hills():
        if "far" not in _cache:
            TW = 600
            im = Image.new("RGBA", (TW, QH), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            r = _rng(11)
            for k in range(5):
                cx, h = k * TW / 5 + r.integers(0, 50), int(r.integers(70, 120))
                hw = int(r.integers(60, 95))
                for ox in (-TW, 0, TW):
                    pts = [(cx + ox - hw, qGY), (cx + ox, qGY - h), (cx + ox + hw, qGY)]
                    d.polygon(pts, fill=(110, 112, 190, 255), outline=(60, 58, 120, 255))
                    cap = [(cx + ox - hw * 0.22, qGY - h * 0.78), (cx + ox, qGY - h), (cx + ox + hw * 0.22, qGY - h * 0.78)]
                    d.polygon(cap, fill=(236, 240, 255, 255))
            _cache["far"] = im
        return _cache["far"]

    def near_hills():
        if "near" not in _cache:
            TW = 480
            im = Image.new("RGBA", (TW, QH), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            r = _rng(23)
            for k in range(4):
                cx, rr = k * TW / 4 + r.integers(0, 40), int(r.integers(38, 62))
                for ox in (-TW, 0, TW):
                    d.ellipse([cx + ox - rr * 1.5, qGY - rr, cx + ox + rr * 1.5, qGY + rr], fill=(46, 160, 72, 255), outline=(20, 80, 40, 255), width=2)
                    for s in range(3):
                        sx, sy = cx + ox - rr * 0.6 + s * rr * 0.55, qGY - rr * 0.55 + (s % 2) * 8
                        d.rectangle([sx, sy, sx + 3, sy + 5], fill=(26, 110, 52, 255))
            _cache["near"] = im
        return _cache["near"]

    def wx(i):
        return i * SPACING

    X_MIN = wx(-1) - 1400
    X_MAX = wx(ctx.n_links) + 1400

    def level():
        """The whole level's ground and props at quarter res: grass, brick dirt, pipes, bushes, floating bricks."""
        if "level" not in _cache:
            LW = (X_MAX - X_MIN) // Q
            im = Image.new("RGBA", (LW, QH), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.rectangle([0, qGY, LW, QH], fill=(150, 84, 40, 255))
            for y in range(qGY + 6, QH, 10):          # bricks
                off = 0 if ((y - qGY) // 10) % 2 else 8
                d.line([(0, y), (LW, y)], fill=(96, 50, 24, 255))
                for x in range(-off, LW, 16):
                    d.line([(x, y), (x, y + 9)], fill=(96, 50, 24, 255))
                    d.point((x + 2, y + 2), fill=(196, 126, 70, 255))
            d.rectangle([0, qGY, LW, qGY + 5], fill=(80, 200, 70, 255))
            d.line([(0, qGY), (LW, qGY)], fill=(170, 240, 120, 255))
            d.line([(0, qGY + 6), (LW, qGY + 6)], fill=(30, 90, 30, 255))
            for x in range(0, LW, 6):                  # grass tufts
                if (x * 7) % 11 < 4:
                    d.rectangle([x, qGY - 2, x + 1, qGY - 1], fill=(80, 200, 70, 255))
            uy = qGY + 72                              # the underground level below (blue bricks, buried coins)
            d.rectangle([0, uy, LW, QH], fill=(16, 20, 44, 255))
            d.rectangle([0, uy, LW, uy + 3], fill=(70, 40, 20, 255))
            for y in range(uy + 34, QH, 8):
                off = 0 if ((y - uy) // 8) % 2 else 6
                for x in range(-off, LW, 12):
                    d.rectangle([x, y, x + 11, y + 7], fill=(40, 70, 150, 255), outline=(12, 20, 60, 255))
            rc = _rng(17)
            for x in range(20, LW, 34):
                cy_ = uy + 12 + int(rc.integers(0, 10))
                d.ellipse([x, cy_, x + 6, cy_ + 9], fill=(255, 200, 30, 255), outline=(60, 30, 0, 255))
            r = _rng(5)
            for i in range(-1, ctx.n_links + 1):
                mid = (wx(i) + STOP_DX + wx(i + 1) + STOP_DX) / 2 - X_MIN   # pipe Buh jumps over
                if i < ctx.n_links - 1:
                    px, ph = int(mid / Q), 36
                    d.rectangle([px - 10, qGY - ph, px + 10, qGY], fill=(60, 190, 70, 255), outline=(16, 70, 24, 255), width=2)
                    d.rectangle([px - 13, qGY - ph - 10, px + 13, qGY - ph], fill=(80, 214, 90, 255), outline=(16, 70, 24, 255), width=2)
                    d.rectangle([px - 7, qGY - ph - 8, px - 4, qGY - 2], fill=(170, 250, 160, 255))
                for _ in range(2):                     # bushes
                    bx = int((wx(i) + r.integers(250, 700) - X_MIN) / Q)
                    for dx, rr in ((0, 7), (8, 9), (17, 7)):
                        d.ellipse([bx + dx - rr, qGY - rr - 2, bx + dx + rr, qGY + rr - 2], fill=(40, 170, 60, 255), outline=(16, 70, 24, 255))
                bx = int((wx(i) + 480 + r.integers(0, 120) - X_MIN) / Q)   # floating bricks, out of shot when stopped
                by = qGY - 70
                for k in range(3):
                    x0 = bx + k * 14
                    d.rectangle([x0, by, x0 + 13, by + 13], fill=(184, 96, 44, 255), outline=(70, 30, 10, 255))
                    d.line([(x0, by + 6), (x0 + 13, by + 6)], fill=(70, 30, 10, 255))
                    d.line([(x0 + 6, by), (x0 + 6, by + 6)], fill=(70, 30, 10, 255))
            _cache["level"] = im
        return _cache["level"]

    def tile_paste(dst, strip, off):
        TW = strip.width
        off = int(off) % TW
        x = -off
        while x < QW:
            dst.alpha_composite(strip, (x, 0) if strip.height == QH else (x, 18))
            x += TW

    def background(camx, tt):
        bg = sky().copy()
        tile_paste(bg, clouds(), camx * 0.12 / Q + tt * 5)
        tile_paste(bg, far_hills(), camx * 0.3 / Q)
        tile_paste(bg, near_hills(), camx * 0.55 / Q)
        lx = int((camx - X_MIN) / Q)
        bg.alpha_composite(level().crop((lx, 0, lx + QW, QH)))
        return bg.resize((ctx.W, ctx.H), Image.NEAREST)

    # ---------------------------------------------------------------- full-res sprites
    def sign(i):
        """Checkpoint billboard: wooden board with the framed photo, a CHECKPOINT plate and posts to the ground."""
        if ("sign", i) not in _cache:
            ph = ctx.view(i, *ctx.both_cam(i, PW, PH, 0.8), PW, PH)
            bw, bh = PW + 70, PH + 70
            top = 70                                      # room for the plate
            im = Image.new("RGBA", (bw + 20, top + bh + (GY - (BOARD_CY + bh // 2))), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            for px in (bw // 2 - 200, bw // 2 + 200):     # posts
                d.rectangle([px - 18, top + bh - 10, px + 18, im.height], fill=(120, 70, 34), outline=INK, width=5)
                d.rectangle([px - 10, top + bh - 10, px - 4, im.height - 4], fill=(170, 108, 56))
            d.rectangle([10, top + 8, bw + 10, top + bh + 8], fill=INK)       # drop shadow
            d.rectangle([0, top, bw, top + bh], fill=(170, 104, 50), outline=INK, width=6)
            for y in range(top + 18, top + bh, 26):
                d.line([(8, y), (bw - 8, y)], fill=(140, 82, 38), width=3)
            for x, y in ((14, top + 14), (bw - 22, top + 14), (14, top + bh - 22), (bw - 22, top + bh - 22)):
                d.rectangle([x, y, x + 8, y + 8], fill=(230, 220, 200), outline=INK, width=2)
            pl = ctx.pixel_tag(f"CHECKPOINT {i + 1}", ctx.WHITE, 24)
            im.alpha_composite(pl, (bw // 2 - pl.width // 2, top - pl.height + 18))
            _cache[("sign", i)] = (im, ph, top, bw, bh)
        return _cache[("sign", i)]

    def draw_sign(base, i, sx, local, bump):
        im, ph, top, bw, bh = sign(i)
        y0 = BOARD_CY - bh // 2 - top - int(bump)
        base.alpha_composite(im, (int(sx - bw / 2), y0))
        card = ctx.framed(ph, tilt=1.4 * math.sin(local * 1.6 + i), border=10, tape=False)
        ctx.paste(base, card, sx, BOARD_CY - bump + 6, nearest=False)

    def qblock(used, sz=78):
        key = ("blk", used, sz)
        if key not in _cache:
            im = Image.new("RGBA", (sz + 8, sz + 8), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.rectangle([8, 8, sz + 7, sz + 7], fill=INK)
            col, hi = ((150, 96, 60), (190, 130, 80)) if used else ((250, 180, 30), (255, 230, 120))
            d.rectangle([0, 0, sz, sz], fill=col, outline=INK, width=6)
            d.rectangle([6, 6, sz - 6, 12], fill=hi)
            for x, y in ((10, 10), (sz - 16, 10), (10, sz - 16), (sz - 16, sz - 16)):
                d.rectangle([x, y, x + 5, y + 5], fill=INK)
            if not used:
                q = ctx.pixel_text("?", 44, fill=ctx.WHITE, shadow=False)
                im.alpha_composite(q, (sz // 2 - q.width // 2, sz // 2 - q.height // 2))
            _cache[key] = im
        return _cache[key]

    def coin(spin, size=64):
        w = max(6, int(size * 0.8 * abs(math.cos(spin))))
        im = Image.new("RGBA", (int(size * 0.8) + 12, size + 12), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        cx = im.width // 2
        d.ellipse([cx - w // 2 - 4, 2, cx + w // 2 + 4, size + 8], fill=INK)
        d.ellipse([cx - w // 2, 6, cx + w // 2, size + 4], fill=(255, 200, 30))
        if w > 20:
            d.rectangle([cx - 4, 18, cx + 4, size - 8], fill=(255, 240, 150))
        return im

    # ---------------------------------------------------------------- timing helpers
    def stop_x(i):
        return wx(i) + STOP_DX

    def buh_state(i, local, sp, R):
        """(world x, height above ground, angle, squash, expr) for Buh at a segment-local time."""
        x0, x1 = stop_x(i - 1), stop_x(i)
        hit = R * sp                                   # jump starts here, head meets the block at hit + THROW
        if local < WALK:
            u = local / WALK
            x = x0 + (x1 - x0) * ctx.ease(u)
            h = abs(math.sin(u * math.pi * 6)) * 26       # running bounce
            if 0.3 < u < 0.7:                            # big jump over the pipe
                h = max(h, 230 * math.sin(math.pi * (u - 0.3) / 0.4))
            return x, h, -10 + 6 * math.sin(u * 30), 0.0, "happy"
        a = local - WALK
        if a < 0.18:                                   # landing squash
            return x1, 0, 0, 0.35 * (1 - a / 0.18), "smug"
        if hit <= local < hit + ctx.THROW + 0.32:
            k = local - hit
            up = ctx.THROW
            h = 200 * math.sin(min(1.0, k / up) * math.pi / 2) if k < up else 200 * max(0.0, 1 - ((k - up) / 0.32) ** 2)
            return x1, h, 0, 0.0, "shock"
        idle = 8 * abs(math.sin(local * 3.2))            # idle bob, a little hop every so often
        if (local % 1.7) < 0.35 and local > hit + 0.7:
            idle = 60 * math.sin(math.pi * (local % 1.7) / 0.35)
        return x1, idle, 4 * math.sin(local * 2.4), 0.0, ("smug" if local < hit else "happy")

    def score_at(i, local, hit_t):
        base_s = 1000 * i
        if local > hit_t:
            base_s += int(1000 * min(1.0, (local - hit_t) / 0.4))
        return base_s + int(min(local, WALK) / WALK * 150)

    # ---------------------------------------------------------------- frame
    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        sp, R = seg["speech"], seg.get("rv", ctx.REVEAL)
        hit = R * sp
        coin_t = hit + ctx.THROW                         # B is revealed: the coin pops
        last = i == ctx.n_links - 1
        land = local - coin_t                            # time since CONNECTED (last link)

        bx, bh, bang, bsq, expr = buh_state(i, local, sp, R)
        if last and land > 0:                           # celebrate: keep hopping
            expr = "happy"
            if land > 0.1:
                bh = max(bh, 90 * abs(math.sin(land * 7)))
        camx = bx - STOP_DX - BOARD_SX + 22 * math.sin(tt * 0.9)    # camera leads Buh; never quite still
        to_sx = lambda wxx: wxx - camx
        base = background(camx, tt)

        # fireworks behind everything in the sky on the final link
        if last and land > 0:
            d = ImageDraw.Draw(base)
            for k, (fx, fy) in enumerate(((140, 420), (860, 380), (300, 330), (720, 300), (120, 760), (900, 720))):
                t_ = land - 0.07 * k
                if t_ <= 0:
                    continue
                rr = 30 + 190 * (1 - math.exp(-t_ * 6))
                for a_ in range(18):
                    ang = a_ / 18 * 2 * math.pi + k
                    px, py = fx + rr * math.cos(ang), fy + rr * math.sin(ang) + 40 * t_ ** 2
                    s = 9
                    d.rectangle([px - s, py - s, px + s, py + s], fill=(ctx.YELLOW, ctx.RED, ctx.CYAN, ctx.WHITE)[(a_ + k) % 4])

        # previous and current checkpoints scroll by
        for j in (i - 1, i):
            if 0 <= j < ctx.n_links and not ctx.C.fj["links"][j].get("lineage"):
                sx = to_sx(wx(j))
                if -500 < sx < ctx.W + 500:
                    bump = 0
                    if j == i and coin_t <= local < coin_t + 0.25:
                        bump = 10 * math.sin(math.pi * (local - coin_t) / 0.25)
                    draw_sign(base, j, sx, local + j, bump)

        # "?" blocks above each stop point (the previous one is spent)
        for j in (i - 1, i):
            if j < 0:
                continue
            sx = to_sx(stop_x(j))
            if -100 < sx < ctx.W + 100:
                used = j < i or local >= coin_t
                yb = BLOCK_Y + (0 if used else 5 * math.sin(tt * 4))
                if j == i and coin_t <= local < coin_t + 0.2:
                    yb -= 26 * math.sin(math.pi * (local - coin_t) / 0.2)
                ctx.paste(base, qblock(used), sx, yb)

        # Buh
        sx = to_sx(bx)
        look = 1 if local < WALK + 0.3 else (1 if (local // 2.3) % 2 == 0 else 0)
        spr = ctx.pin_sprite(expr, bang, bsq, look, scale=5)
        ctx.paste_pin(base, spr, sx, GY - bh)

        # coin + points
        if coin_t <= local < coin_t + 0.9:
            k = (local - coin_t) / 0.9
            cy = BLOCK_Y - 70 - 260 * math.sin(min(1.0, k * 1.6) * math.pi / 2) + (180 * max(0.0, k - 0.62) ** 2 * 6)
            ctx.paste(base, coin(local * 18), to_sx(stop_x(i)), cy, alpha=1.0 if k < 0.75 else (1 - k) * 4)
            ctx.paste(base, ctx.pixel_text("+1000", 30, fill=ctx.WHITE), to_sx(stop_x(i)) + 110, BLOCK_Y - 90 - 120 * k,
                      alpha=min(1.0, (1 - k) * 3))
            d = ImageDraw.Draw(base)
            for a_ in range(8):                          # sparkle burst
                ang = a_ * math.pi / 4
                rr = 40 + 160 * k
                px, py = to_sx(stop_x(i)) + rr * math.cos(ang), BLOCK_Y - 40 + rr * math.sin(ang) * 0.6
                s = int(8 * (1 - k)) + 2
                d.rectangle([px - s, py - s, px + s, py + s], fill=ctx.YELLOW, outline=INK)

        # name tags (pixel tags on the sign's posts)
        an, bn = ctx.link_names(i)
        slide = ctx.ease_out_back(min(1.0, max(0.0, (local - WALK + 0.2) / 0.3)))
        ta = ctx.pixel_tag(ctx.C.labels[an].upper(), ctx.CYAN, 30)
        ctx.paste(base, ta, BOARD_SX + 30 + 700 * (1 - slide), TAG_A_Y + 3 * math.sin(tt * 3))
        if local > coin_t:
            pop = ctx.ease_out_back(min(1.0, (local - coin_t) / 0.18))
            tb = ctx.pixel_tag(ctx.C.labels[bn].upper(), ctx.YELLOW, 30)
            ctx.paste(base, tb, BOARD_SX + 30, TAG_B_Y + 3 * math.sin(tt * 3 + 1), scale=0.4 + 0.6 * pop)

        # HUD
        if last and land > 0.03:
            st = ctx.ease_out_back(min(1.0, (land - 0.03) / 0.18))
            ctx.paste(base, ctx.pixel_text("LEVEL CLEAR!", 54, fill=ctx.YELLOW), ctx.CX, 286, scale=0.3 + 0.7 * st)
        else:
            y = 290
            ctx.paste(base, ctx.pixel_text(f"WORLD 1-{i + 1}", 26, fill=ctx.WHITE), ctx.SAFE[0] + 130, y)
            coins = i + (1 if local >= coin_t else 0)
            ctx.paste(base, coin(tt * 3, 34), ctx.CX - 30, y)
            ctx.paste(base, ctx.pixel_text(f"x{coins:02d}", 26, fill=ctx.WHITE), ctx.CX + 30, y)
            ctx.paste(base, ctx.pixel_text(f"{score_at(i, local, coin_t):06d}", 26, fill=ctx.YELLOW), ctx.SAFE[2] - 110, y)

        ctx.header(base, i)
        if last:                                         # CONNECTED! drops a little lower, under the tags
            ov = Image.new("RGBA", (ctx.W, ctx.H), (0, 0, 0, 0))
            ctx.connected(ov, seg, tt)
            base.alpha_composite(ov.crop((0, 0, ctx.W, ctx.H - 100)), (0, 100))
        seg["cap"].draw(base, tt, y=CAP_Y)
        return base

    return frame
