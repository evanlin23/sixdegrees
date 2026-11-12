"""Game: Buh's "Case Run", an original side-scrolling pixel platformer at dusk. Each photo is a framed exhibit standing
on a floating stack of evidence crates; Buh dashes in from the last exhibit, hopping across a loose crate over the
gap, stops under a floating magnifier lens, and at the reveal jumps and bonks it: a little photo chip pops out
(the B tag lands with it). The skyline, fog and crates are drawn once at quarter resolution and scrolled.

HUD: STAGE n and LINKS n (no coins, no score), shown only from link 2 once its photo is in, and never on a link
with a dictator/killer (there the lens just thuds, dust only). The bottom of the frame carries the chain so far as
a row of faces. The last link ends on CONNECTED! in the HUD row (the HUD gives way).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

NAME = "game"

Q = 4                     # background pixel size (drawn at 1/Q resolution, scaled up NEAREST)
SPACING = 1400            # world px between exhibits
WALK = 0.45               # s for Buh's dash from one exhibit to the next (fast: the photo is in before the caption)
GY = 1275                 # platform top (screen y)
BUH_SX = 108              # Buh's screen x once he has arrived (left of the exhibit)
BOARD_SX = 640            # exhibit centre on screen once Buh has arrived
STOP_DX = BUH_SX - BOARD_SX   # Buh stops this far left of an exhibit centre (world px)
BOARD_CY = 720            # photo centre y
PW, PH = 760, 560         # photo size
BLOCK_Y = GY - 410        # lens centre y
TAG_Y = 1092              # name tags, each under its own face
TAG_X = (205, 935)        # tag row stays right of Buh and inside the safe area
CAP_Y = 1372
HUD_Y = 300
ROW_Y, ROW_TH = 1520, 84  # the chain-so-far row of faces (bottom band)
PLAT_L, PLAT_R = -660, 430   # platform extent around each exhibit centre (world px)
SLAB = 76                 # platform (crate row) thickness, screen px

SKY_TOP, SKY_MID, SKY_LOW = np.array([22, 14, 58]), np.array([110, 44, 128]), np.array([255, 146, 116])
FAR, FAR_WIN = (66, 40, 108, 255), (255, 206, 120, 255)
NEAR, NEAR_WIN = (38, 22, 68, 255), (255, 116, 170, 255)
VOID = np.array([28, 14, 48])
CRATE, CRATE_HI, CRATE_DK = (196, 132, 64, 255), (232, 176, 98, 255), (110, 62, 30, 255)
TEAL, TEAL_HI = (34, 88, 104), (70, 150, 160)
MINT = (120, 255, 200)


def _rng(seed):
    return np.random.default_rng(seed)


def make(ctx):
    QW, QH = ctx.W // Q, ctx.H // Q
    qGY = GY // Q
    INK = ctx.INK
    _cache = {}

    # ---------------------------------------------------------------- static quarter-res layers
    def sky():
        """Banded dusk gradient down to the platform line, then a dark void with drifting fog bands below."""
        if "sky" not in _cache:
            a = np.zeros((QH, QW, 3), np.uint8)
            for y in range(QH):
                if y < qGY:
                    k = round(min(1.0, y / (qGY * 0.98)) * 10) / 10
                    c = SKY_TOP + (SKY_MID - SKY_TOP) * (k / 0.6) if k < 0.6 else SKY_MID + (SKY_LOW - SKY_MID) * ((k - 0.6) / 0.4)
                else:
                    k = round(min(1.0, (y - qGY) / (QH - qGY)) * 6) / 6
                    c = SKY_LOW * 0.35 * (1 - k) + VOID * (0.65 + 0.35 * k)
                a[y] = np.clip(c, 0, 255)
            r = _rng(2)
            for _ in range(70):                            # stars in the upper sky
                x, y = int(r.integers(0, QW)), int(r.integers(0, qGY // 2))
                a[y, x] = (255, 240, 220)
            _cache["sky"] = Image.fromarray(a).convert("RGBA")
        return _cache["sky"]

    def fog():
        """Soft horizontal fog streaks (tileable), used just under the platforms and in the sky."""
        if "fog" not in _cache:
            TW, TH = 520, 60
            im = Image.new("RGBA", (TW, TH), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            r = _rng(7)
            for k in range(7):
                x, y, w = int(k * TW / 7 + r.integers(0, 30)), int(r.integers(4, TH - 8)), int(r.integers(40, 110))
                for ox in (-TW, 0, TW):
                    d.rectangle([x + ox, y, x + ox + w, y + 2], fill=(255, 170, 200, 70))
                    d.rectangle([x + ox + 8, y + 3, x + ox + w - 14, y + 4], fill=(255, 210, 230, 50))
            _cache["fog"] = im
        return _cache["fog"]

    def skyline(key, TW, col, win, hmin, hmax, wmin, wmax, seed, antenna):
        if key not in _cache:
            im = Image.new("RGBA", (TW, QH), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            r = _rng(seed)
            x = 0
            while x < TW:
                w, h = int(r.integers(wmin, wmax)), int(r.integers(hmin, hmax))
                top = qGY - h
                d.rectangle([x, top, x + w - 2, qGY], fill=col)
                if antenna and r.random() < 0.4:
                    d.rectangle([x + w // 2, top - 10, x + w // 2 + 1, top], fill=col)
                    d.point((x + w // 2, top - 11), fill=(255, 80, 90, 255))
                for wy in range(top + 4, qGY - 3, 6):
                    for wxx in range(x + 3, x + w - 4, 5):
                        if r.random() < 0.35:
                            d.rectangle([wxx, wy, wxx + 1, wy + 2], fill=win)
                x += w + int(r.integers(0, 6))
            _cache[key] = im
        return _cache[key]

    def wx(i):
        return i * SPACING

    X_MIN = wx(-1) - 1400
    X_MAX = wx(ctx.n_links) + 1400

    def crate(d, x0, y0, s):
        """One evidence crate at quarter res, s px square: planks, cross brace, a stencilled tag."""
        d.rectangle([x0, y0, x0 + s - 1, y0 + s - 1], fill=CRATE, outline=INK)
        d.line([(x0 + 1, y0 + 1), (x0 + s - 2, y0 + 1)], fill=CRATE_HI)
        d.line([(x0 + 2, y0 + 2), (x0 + s - 3, y0 + s - 3)], fill=CRATE_DK)
        d.line([(x0 + s - 3, y0 + 2), (x0 + 2, y0 + s - 3)], fill=CRATE_DK)
        d.rectangle([x0 + s // 2 - 3, y0 + s // 2 - 2, x0 + s // 2 + 3, y0 + s // 2 + 1], fill=(236, 230, 210, 255), outline=CRATE_DK)

    def level():
        """The whole level at quarter res: a crate platform under each exhibit, a loose crate over each gap."""
        if "level" not in _cache:
            LW = (X_MAX - X_MIN) // Q
            im = Image.new("RGBA", (LW, QH), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            cs = SLAB // Q
            for i in range(-1, ctx.n_links + 1):
                x0, x1 = int((wx(i) + PLAT_L - X_MIN) / Q), int((wx(i) + PLAT_R - X_MIN) / Q)
                d.rectangle([x0 + 3, qGY + 3, x1 + 3, qGY + cs + 3], fill=(12, 6, 24, 160))     # shadow
                for x in range(x0, x1 - cs + 1, cs):
                    crate(d, x, qGY, cs)
                d.rectangle([x0 - 2, qGY - 3, x1 + 1, qGY], fill=(90, 220, 190, 255), outline=INK)   # mint plank top
                for k in range(3):                         # hanging cables into the fog
                    cx_ = x0 + (k + 1) * (x1 - x0) // 4
                    d.line([(cx_, qGY + cs), (cx_, qGY + cs + 26 + 8 * k)], fill=(60, 36, 90, 255))
                if i < ctx.n_links:                         # the loose crate over the gap Buh hops across
                    gx = int((wx(i) + PLAT_R + (SPACING + PLAT_L - PLAT_R) / 2 - X_MIN) / Q)
                    crate(d, gx - cs // 2, qGY + 6, cs)
            _cache["level"] = im
        return _cache["level"]

    def tile_paste(dst, strip, off, y=0):
        TW = strip.width
        off = int(off) % TW
        x = -off
        while x < QW:
            dst.alpha_composite(strip, (x, y))
            x += TW

    def background(camx, tt):
        bg = sky().copy()
        tile_paste(bg, fog(), camx * 0.08 / Q + tt * 4, qGY // 3)
        tile_paste(bg, skyline("far", 640, FAR, FAR_WIN, 60, 140, 18, 40, 11, True), camx * 0.25 / Q)
        tile_paste(bg, skyline("near", 560, NEAR, NEAR_WIN, 30, 80, 24, 46, 23, False), camx * 0.5 / Q)
        lx = int((camx - X_MIN) / Q)
        bg.alpha_composite(level().crop((lx, 0, lx + QW, QH)))
        tile_paste(bg, fog(), camx * 0.9 / Q + tt * 9, qGY + SLAB // Q + 14)
        return bg.resize((ctx.W, ctx.H), Image.NEAREST)

    # ---------------------------------------------------------------- full-res sprites
    def cam(i):
        cx, cy, sc = ctx.both_cam(i, PW, PH, 0.8)
        return cx, cy, min(sc, 1.3)                   # never upscale a photo past ~1.3x

    def sign(i, plate=True):
        """Exhibit stand: a teal metal frame with rivets, an EXHIBIT plate and two struts down to the crates."""
        if ("sign", i, plate) not in _cache:
            ph = ctx.view(i, *cam(i), PW, PH)
            bw, bh = PW + 70, PH + 70
            top = 70                                      # room for the plate
            im = Image.new("RGBA", (bw + 20, top + bh + (GY - (BOARD_CY + bh // 2))), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            for px in (bw // 2 - 220, bw // 2 + 220):     # struts (A-frame)
                d.polygon([(px - 14, top + bh - 10), (px + 14, top + bh - 10), (px + 26, im.height), (px - 26, im.height)],
                          fill=TEAL, outline=INK)
                d.line([(px - 4, top + bh), (px - 10, im.height - 4)], fill=TEAL_HI, width=4)
            d.rectangle([10, top + 8, bw + 10, top + bh + 8], fill=INK)       # drop shadow
            d.rectangle([0, top, bw, top + bh], fill=TEAL, outline=INK, width=6)
            d.rectangle([8, top + 8, bw - 8, top + 14], fill=TEAL_HI)
            for x in range(22, bw - 14, 64):                # rivets along top and bottom
                for y in (top + 22, top + bh - 26):
                    d.rectangle([x, y, x + 6, y + 6], fill=(200, 230, 225), outline=INK, width=1)
            if plate:
                pl = ctx.pixel_tag(f"EXHIBIT {i + 1}", (255, 200, 150), 24)
                im.alpha_composite(pl, (bw // 2 - pl.width // 2, top - pl.height + 18))
            _cache[("sign", i, plate)] = (im, ph, top, bw, bh)
        return _cache[("sign", i, plate)]

    def face_sx(i, role, sx):
        """Screen x of a face on exhibit i whose centre is at screen x sx."""
        cx, _, sc = cam(i)
        b = ctx.fbox(i, role)
        return sx + (b[0] + b[2] / 2 - cx) * sc

    def tag_layout(i):
        """(size, x_a, x_b, face_a, face_b) relative to the board centre: both tags on one row, each as close under its own face as the row allows.
        Laid out with both names from the start so A's tag never moves when B's appears."""
        if ("tags", i) in _cache:
            return _cache[("tags", i)]
        an, bn = ctx.link_names(i)
        fa, fb = face_sx(i, "a", BOARD_SX), face_sx(i, "b", BOARD_SX)
        lo, hi = TAG_X
        for size in (28, 26, 24, 22, 20, 18):
            wa = ctx.pixel_tag(ctx.C.labels[an].upper(), ctx.CYAN, size).width
            wb = ctx.pixel_tag(ctx.C.labels[bn].upper(), ctx.YELLOW, size).width
            if wa + wb + 16 <= hi - lo:
                break
        items = sorted([(fa, wa, "a"), (fb, wb, "b")])
        (f1, w1, r1), (f2, w2, r2) = items
        x1 = min(max(f1, lo + w1 / 2), hi - w2 - 16 - w1 / 2)
        x2 = min(max(f2, x1 + w1 / 2 + 16 + w2 / 2), hi - w2 / 2)
        if x2 - w2 / 2 < x1 + w1 / 2 + 16:              # still touching: push the left one back
            x1 = x2 - w2 / 2 - 16 - w1 / 2
        pos = {r1: x1, r2: x2}
        _cache[("tags", i)] = (size, pos["a"] - BOARD_SX, pos["b"] - BOARD_SX, fa - BOARD_SX, fb - BOARD_SX)
        return _cache[("tags", i)]

    def draw_tag(base, text, bg, size, x, face_x, y, scale=1.0):
        """A pixel tag with a little arrow on top pointing up at its face."""
        tg = ctx.pixel_tag(text, bg, size)
        if scale <= 0.01:
            return
        ax = min(max(face_x, x - tg.width / 2 + 26), x + tg.width / 2 - 34)
        top = y - tg.height / 2 * scale
        d = ImageDraw.Draw(base)
        s_ = 18 * scale
        d.polygon([(ax - s_ - 4, top + 4), (ax + s_ + 4, top + 4), (ax, top - s_ - 6)], fill=INK)
        d.polygon([(ax - s_, top + 4), (ax + s_, top + 4), (ax, top - s_)], fill=bg)
        ctx.paste(base, tg, x, y, scale=scale)

    def draw_sign(base, i, sx, local, bump, plate=True):
        im, ph, top, bw, bh = sign(i, plate)
        y0 = BOARD_CY - bh // 2 - top - int(bump)
        base.alpha_composite(im, (int(sx - bw / 2), y0))
        card = ctx.framed(ph, tilt=1.4 * math.sin(local * 1.6 + i), border=10, tape=False)
        ctx.paste(base, card, sx, BOARD_CY - bump + 6, nearest=False)

    def lens(used, sz=84):
        """The floating magnifier Buh bonks: glass disc with a glint and a short handle; dims once used."""
        key = ("lens", used, sz)
        if key not in _cache:
            im = Image.new("RGBA", (sz + 40, sz + 40), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            glass = (70, 90, 120) if used else (110, 230, 255)
            d.polygon([(sz - 6, sz - 14), (sz + 6, sz - 26), (sz + 34, sz + 2), (sz + 22, sz + 14)], fill=INK)
            d.polygon([(sz - 2, sz - 14), (sz + 6, sz - 22), (sz + 28, sz + 2), (sz + 20, sz + 10)], fill=(150, 90, 50) if used else (255, 120, 170))
            d.ellipse([6, 6, sz + 6, sz + 6], fill=INK)
            d.ellipse([0, 0, sz, sz], fill=INK)
            d.ellipse([9, 9, sz - 9, sz - 9], fill=glass)
            if not used:
                d.rectangle([22, 20, 34, 26], fill=(240, 255, 255))
                d.rectangle([18, 28, 24, 40], fill=(240, 255, 255))
            _cache[key] = im
        return _cache[key]

    def chip(spin, size=70):
        """The photo chip that pops from the lens: a tiny white print, flipping."""
        w = max(6, int(size * abs(math.cos(spin))))
        im = Image.new("RGBA", (size + 12, size + 12), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        cx = im.width // 2
        d.rectangle([cx - w // 2 - 3, 2, cx + w // 2 + 3, size + 6], fill=INK)
        d.rectangle([cx - w // 2, 5, cx + w // 2, size + 3], fill=(250, 248, 240))
        if w > 24:
            d.rectangle([cx - w // 2 + 6, 11, cx + w // 2 - 6, size - 14], fill=(110, 230, 255))
        return im

    def chain_row(base, i, revealed):
        """The chain so far: a row of faces ("?" for whoever isn't revealed yet); the current pair outlined."""
        people = ctx.C.people
        n = len(people)
        x0, x1 = ctx.SAFE[0] + 60, ctx.SAFE[2] - ROW_TH / 2 - 10
        step = (x1 - x0) / max(1, n - 1)
        known = i + (2 if revealed else 1)
        d = ImageDraw.Draw(base)
        for j, nm in enumerate(people):
            x, y = x0 + j * step, ROW_Y
            if j + 1 < n:
                d.rectangle([x + ROW_TH / 2, y - 4, x + step - ROW_TH / 2, y + 4], fill=MINT if j + 1 < known else (80, 60, 120))
            box = [x - ROW_TH / 2, y - ROW_TH / 2, x + ROW_TH / 2, y + ROW_TH / 2]
            d.rectangle([box[0] + 6, box[1] + 6, box[2] + 6, box[3] + 6], fill=INK)
            if j < known:
                base.alpha_composite(ctx.sharp[nm].resize((ROW_TH, ROW_TH), Image.BICUBIC), (int(box[0]), int(box[1])))
            else:
                d.rectangle(box, fill=(60, 40, 100))
                ctx.paste(base, ctx.pixel_text("?", 36, fill=ctx.WHITE, shadow=False), x, y)
            col = ctx.CYAN if j == i else (ctx.YELLOW if j == i + 1 else INK)
            d.rectangle(box, outline=col, width=5 if j in (i, i + 1) else 3)

    # ---------------------------------------------------------------- timing helpers
    def stop_x(i):
        return wx(i) + STOP_DX

    def buh_state(i, local, sp, R):
        """(world x, height above platform, angle, squash, expr) for Buh at a segment-local time."""
        x0, x1 = stop_x(i - 1), stop_x(i)
        hit = R * sp                                   # jump starts here, head meets the lens at hit + THROW
        if local < WALK:
            u = local / WALK
            x = x0 + (x1 - x0) * ctx.ease(u)
            h = abs(math.sin(u * math.pi * 3)) * 26       # running bounce
            if 0.2 < u < 0.8:                            # big hop across the gap (over the loose crate)
                h = max(h, 240 * math.sin(math.pi * (u - 0.2) / 0.6))
            return x, h, -10 + 6 * math.sin(u * 20), 0.0, "happy"
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

    def sens(j):
        a, b = ctx.link_names(j)
        return ctx.sensitive(a) or ctx.sensitive(b)

    # ---------------------------------------------------------------- frame
    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        sp, R = seg["speech"], seg.get("rv", ctx.REVEAL)
        hit = R * sp
        coin_t = hit + ctx.THROW                         # B is revealed: the chip pops
        last = i == ctx.n_links - 1
        land = local - coin_t                            # time since CONNECTED (last link)
        sens_i = sens(i)

        bx, bh, bang, bsq, expr = buh_state(i, local, sp, R)
        if last and land > 0:                           # celebrate: keep hopping
            expr = "happy"
            if land > 0.1:
                bh = max(bh, 90 * abs(math.sin(land * 7)))
        camx = bx - STOP_DX - BOARD_SX + 22 * math.sin(tt * 0.9)    # camera leads Buh; never quite still
        to_sx = lambda wxx: wxx - camx
        base = background(camx, tt)

        # pixel fireworks in the sky behind everything on the final link
        if last and land > 0:
            d = ImageDraw.Draw(base)
            for k, (fx, fy) in enumerate(((140, 440), (860, 400), (300, 360), (720, 330), (120, 780), (900, 740))):
                t_ = land - 0.07 * k
                if t_ <= 0:
                    continue
                rr = 30 + 190 * (1 - math.exp(-t_ * 6))
                for a_ in range(18):
                    ang = a_ / 18 * 2 * math.pi + k
                    px, py = fx + rr * math.cos(ang), fy + rr * math.sin(ang) + 40 * t_ ** 2
                    s = 9
                    d.rectangle([px - s, py - s, px + s, py + s], fill=(MINT, (255, 120, 170), ctx.CYAN, ctx.WHITE)[(a_ + k) % 4])

        # previous and current exhibits scroll by
        for j in (i - 1, i):
            if 0 <= j < ctx.n_links and not ctx.C.fj["links"][j].get("lineage"):
                sx = to_sx(wx(j))
                if -500 < sx < ctx.W + 500:
                    bump = 0
                    if j == i and coin_t <= local < coin_t + 0.25:
                        bump = 10 * math.sin(math.pi * (local - coin_t) / 0.25)
                    draw_sign(base, j, sx, local + j, bump, plate=not (last and j == i and land > 0))

        # the floating lens above each stop point (the previous one is spent)
        for j in (i - 1, i):
            if j < 0:
                continue
            sx = to_sx(stop_x(j))
            if -100 < sx < ctx.W + 100:
                used = j < i or local >= coin_t
                yb = BLOCK_Y + (0 if used else 6 * math.sin(tt * 4))
                if j == i and coin_t <= local < coin_t + 0.2:
                    yb -= 26 * math.sin(math.pi * (local - coin_t) / 0.2)
                ctx.paste(base, lens(used), sx + 14, yb + 14)

        # Buh
        sx = to_sx(bx)
        look = 1 if local < WALK + 0.3 else (1 if (local // 2.3) % 2 == 0 else 0)
        spr = ctx.pin_sprite(expr, bang, bsq, look, scale=5)
        ctx.paste_pin(base, spr, sx, GY - bh)

        if coin_t <= local < coin_t + 0.9:
            k = (local - coin_t) / 0.9
            if sens_i:                                   # a plain thud: dust puffs off the lens, nothing pops
                d = ImageDraw.Draw(base)
                for a_ in (-1, 1):
                    px = to_sx(stop_x(i)) + a_ * (56 + 60 * k)
                    py = BLOCK_Y + 20 - 30 * k
                    r_ = int(14 * (1 - k)) + 2
                    d.rectangle([px - r_, py - r_, px + r_, py + r_], fill=(220, 214, 200))
            else:                                        # a photo chip flips out + "+1 LINK"
                cy = BLOCK_Y - 80 - 240 * math.sin(min(1.0, k * 1.6) * math.pi / 2) + (180 * max(0.0, k - 0.62) ** 2 * 6)
                ctx.paste(base, chip(local * 14), to_sx(stop_x(i)), cy, alpha=1.0 if k < 0.75 else (1 - k) * 4)
                ctx.paste(base, ctx.pixel_text("+1 LINK", 28, fill=MINT), to_sx(stop_x(i)) + 150, BLOCK_Y - 90 - 120 * k,
                          alpha=min(1.0, (1 - k) * 3))
                d = ImageDraw.Draw(base)
                for a_ in range(8):                      # sparkle burst
                    ang = a_ * math.pi / 4
                    rr = 40 + 160 * k
                    px, py = to_sx(stop_x(i)) + rr * math.cos(ang), BLOCK_Y - 40 + rr * math.sin(ang) * 0.6
                    s = int(8 * (1 - k)) + 2
                    d.rectangle([px - s, py - s, px + s, py + s], fill=MINT, outline=INK)

        # name tags under the exhibit, each under (and pointing at) its own face; they ride with the board
        an, bn = ctx.link_names(i)
        size, xa, xb, fa, fb = tag_layout(i)
        bsx = to_sx(wx(i))
        pop_a = ctx.ease_out_back(min(1.0, max(0.0, (local - WALK + 0.05) / 0.25)))
        draw_tag(base, ctx.C.labels[an].upper(), ctx.CYAN, size, bsx + xa, bsx + fa, TAG_Y + 3 * math.sin(tt * 3), pop_a)
        if local > coin_t:
            pop = ctx.ease_out_back(min(1.0, (local - coin_t) / 0.18))
            draw_tag(base, ctx.C.labels[bn].upper(), ctx.YELLOW, size, bsx + xb, bsx + fb, TAG_Y + 3 * math.sin(tt * 3 + 1),
                     0.4 + 0.6 * pop)

        # HUD: from link 2 once its photo is in; never on a dictator/killer link; gives way to CONNECTED! at the end
        if i >= 1 and not sens_i and not (last and land > 0.0):
            a_hud = min(1.0, max(0.0, (local - WALK - 0.1) / 0.2))
            if a_hud > 0:
                found = i + (1 if local >= coin_t else 0)
                ctx.paste(base, ctx.pixel_text(f"STAGE {i + 1}", 26, fill=ctx.WHITE), ctx.SAFE[0] + 110, HUD_Y, alpha=a_hud)
                ctx.paste(base, ctx.pixel_text(f"LINKS {found}/{ctx.n_links}", 26, fill=MINT), ctx.SAFE[2] - 150, HUD_Y,
                          alpha=a_hud)

        chain_row(base, i, local > coin_t)
        ctx.header(base, i)
        if last:                                         # CONNECTED! takes the (now empty) HUD row above the exhibit
            ctx.connected(base, seg, tt, y=HUD_Y + 6)
        if i == 0 or local >= WALK + 0.05:                # the link's first caption waits until its photo has arrived
            seg["cap"].draw(base, tt, y=CAP_Y)
        return base

    return frame
