"""Game: a skin around the core. The core photo window is the play screen of an original pixel handheld (grey bezel,
power LED, D-pad and A/B buttons below), over a night sky with drifting, twinkling pixel stars (an odd shooting star) and scrolling pixel hills. The bezel's lower lip carries
the HUD: LEVEL n, and on links without a dictator a coin score that ticks up when Buh lands, plus a small CLEAR!.
On a sensitive link the HUD shows only the level and the year: no coins, no score, no celebration. When a viewer
requested the episode ("requested_by": "@handle") they are the player: "P1 @HANDLE" under LEVEL n on the lip, and a
"PLAYER @HANDLE" level-start banner along the bottom of the screen as link 1 opens. No handle: nothing extra."""
from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw

NAME = "game"

S = 0.76                                 # core window scale (880x950 -> 669x722)
Y0 = 438                                 # window (screen) top
SKY = ((18, 14, 40), (34, 26, 72))
SHELL = (178, 174, 192)
SHELL_D = (128, 122, 146)
GLASS = (36, 34, 48)


def make(ctx):
    W, H, CX, INK, WHITE, YELLOW = ctx.W, ctx.H, ctx.CX, ctx.INK, ctx.WHITE, ctx.YELLOW
    w, h = int(880 * S), int(950 * S)
    X0 = int(CX - w / 2)
    LIP = 74                                 # bezel lip under the screen (the HUD lives here)
    rq = str(ctx.spec.get("requested_by") or "").strip()
    PLAYER = rq.upper() if rq.startswith("@") and len(rq) > 1 else ""     # the viewer who asked for this episode
    BANNER = 1.8                             # level-start banner on link 1, s
    _c = {}

    def sens(i):
        a, b = ctx.link_names(i)
        return ctx.sensitive(a) or ctx.sensitive(b)

    def sky():
        """Banded pixel gradient (cached); the stars are drawn per frame on top of it."""
        if "sky" not in _c:
            im = Image.new("RGBA", (W, H), SKY[0] + (255,))
            d = ImageDraw.Draw(im)
            for y in range(0, H, 8):                         # banded pixel gradient
                u = y / H
                d.rectangle([0, y, W, y + 8], fill=tuple(int(a + (b - a) * u) for a, b in zip(*SKY)))
            _c["sky"] = im
        return _c["sky"].copy()

    # stars: three parallax layers drifting left (slower than the hills), each one twinkling on its own clock
    rnd = random.Random(11)
    STARS = []
    for layer, (n, size, spd) in enumerate(((150, 4, 5.0), (80, 6, 11.0), (26, 6, 19.0))):
        for _ in range(n):
            STARS.append((rnd.uniform(0, W + 24), rnd.randrange(0, 1560, 6), size, spd, layer,
                          rnd.uniform(0, 6.283), rnd.uniform(1.2, 3.4), rnd.random() < 0.12))

    def stars(base, tt):
        d = ImageDraw.Draw(base)
        for x0, y, size, spd, layer, ph, fr, gold in STARS:
            x = int((x0 - tt * spd) % (W + 24) - 12) // 2 * 2
            tw = 0.5 + 0.5 * math.sin(tt * fr + ph)                         # 0..1 twinkle
            v = 0.35 + 0.65 * tw if layer else 0.25 + 0.55 * tw
            col = (255, 214, 60) if gold else (220, 220, 255)
            col = tuple(int(c * v + s_ * (1 - v)) for c, s_ in zip(col, SKY[1]))
            d.rectangle([x, y, x + size - 1, y + size - 1], fill=col)
            if layer == 2 and tw > 0.82:                                    # the near ones flash a little cross
                arm = 6 if tw > 0.93 else 3
                d.rectangle([x - arm, y + size // 2 - 1, x + size - 1 + arm, y + size // 2], fill=col)
                d.rectangle([x + size // 2 - 1, y - arm, x + size // 2, y + size - 1 + arm], fill=col)
        # a shooting star now and then, down one side of the handheld (never across the screen)
        per = 6.5
        k, u = int(tt // per), (tt % per) / 0.7
        if u < 1:
            side = k % 2
            sx = (60 + 50 * (k % 3)) if side == 0 else (W - 40 - 50 * (k % 3))
            sy = 520 + 230 * (k % 4)
            dx = -1 if side else 1
            for j in range(6):
                q = u - j * 0.05
                if 0 <= q <= 1:
                    px, py = int(sx + dx * q * 150) // 3 * 3, int(sy + q * 260) // 3 * 3
                    a = (1 - j / 6) * (1 - max(0.0, u - 0.7) / 0.3)
                    col = tuple(int(255 * a + s_ * (1 - a)) for s_ in SKY[1])
                    sz = 6 if j == 0 else 3
                    d.rectangle([px, py, px + sz, py + sz], fill=col)

    def hills(base, tt):
        """Two parallax rows of stepped pixel hills along the bottom, scrolling left."""
        d = ImageDraw.Draw(base)
        for row, (col, top, amp, per, spd) in enumerate((((44, 36, 96), 1500, 110, 420, 30),
                                                          ((30, 74, 70), 1640, 80, 300, 70))):
            off = (tt * spd) % per
            pts = [(-20, H)]
            for x in range(-24, W + 48, 24):
                y = top + amp * (0.5 + 0.5 * math.sin((x + off) / per * 2 * math.pi + row))
                pts += [(x, int(y // 12 * 12)), (x + 24, int(y // 12 * 12))]
            pts.append((W + 20, H))
            d.polygon(pts, fill=col)
        d.rectangle([0, 1820, W, H], fill=(96, 62, 40))     # ground blocks
        off = int(tt * 70) % 60
        for x in range(-off, W, 60):
            d.rectangle([x, 1820, x + 58, 1878], outline=(60, 38, 24), width=4)

    def bezel(b, X, Y, ww, hh):
        d = ImageDraw.Draw(b)
        d.rounded_rectangle([X - 36 + 12, Y - 28 + 12, X + ww + 36 + 12, Y + hh + LIP + 12], 34, fill=INK)
        d.rounded_rectangle([X - 36, Y - 28, X + ww + 36, Y + hh + LIP], 34, fill=SHELL, outline=INK, width=6)
        d.rounded_rectangle([X - 30, Y - 22, X + ww + 30, Y + hh + LIP - 6], 30, outline=(214, 210, 226), width=3)
        d.rectangle([X - 12, Y - 12, X + ww + 12, Y + hh + 12], fill=GLASS)
        d.rectangle([X - 4, Y - 4, X + ww + 4, Y + hh + 4], fill=INK)

    def controls(base, tt, k):
        """D-pad and A/B buttons under the device (decoration; A blinks on the reveal)."""
        d = ImageDraw.Draw(base)
        cx, cy = 215, 1650
        for dx, dy, ww, hh in ((-66, -22, 132, 44), (-22, -66, 44, 132)):
            d.rectangle([cx + dx + 8, cy + dy + 8, cx + dx + ww + 8, cy + dy + hh + 8], fill=INK)
        for dx, dy, ww, hh in ((-66, -22, 132, 44), (-22, -66, 44, 132)):
            d.rectangle([cx + dx, cy + dy, cx + dx + ww, cy + dy + hh], fill=SHELL_D, outline=INK, width=5)
        for j, (bx, by, lab) in enumerate(((730, 1670, "B"), (840, 1610, "A"))):
            press = 6 if (j == 1 and k["press"]) else 0
            d.ellipse([bx - 44 + 8, by - 44 + 8, bx + 44 + 8, by + 44 + 8], fill=INK)
            d.ellipse([bx - 44 + press, by - 44 + press, bx + 44 + press, by + 44 + press], fill=(204, 40, 70), outline=INK, width=5)
            ctx.paste(base, ctx.pixel_text(lab, 28, fill=WHITE, shadow=False), bx + press, by + press)

    def hud(base, i, c, tt):
        yy = Y0 + h + 14 + (LIP - 14) / 2
        if PLAYER:                                           # two rows: LEVEL n over P1 @HANDLE
            lv = ctx.pixel_text(f"LEVEL {i + 1}", 22, fill=INK, outline=SHELL, shadow=False)
            base.alpha_composite(lv, (X0 - 6, int(yy - lv.height + 4)))
            p1 = player_tag()
            base.alpha_composite(p1, (X0 - 6, int(yy - 2)))
        else:
            lv = ctx.pixel_text(f"LEVEL {i + 1}", 26, fill=INK, outline=SHELL, shadow=False)
            base.alpha_composite(lv, (X0 - 6, int(yy - lv.height / 2)))
        d = ImageDraw.Draw(base)
        led = (230, 40, 60) if int(tt * 2) % 2 else (120, 30, 40)
        d.rectangle([X0 + w + 4, yy - 8, X0 + w + 20, yy + 8], fill=led, outline=INK, width=3)   # power LED
        land = c["land"]
        if sens(i):                                          # no coins, no score, no celebration
            yr = ctx.year_of(i)
            if yr:
                t_ = ctx.pixel_text(str(yr), 26, fill=INK, outline=SHELL, shadow=False)
                base.alpha_composite(t_, (int(X0 + w - 14 - t_.width), int(yy - t_.height / 2)))
            return
        score = 100 * sum(1 for j in range(i) if not sens(j) and not ctx.C.fj["links"][j].get("lineage"))
        if land >= 0:
            score += int(100 * min(1.0, land / 0.4))
        sc = ctx.pixel_text(f"{score:05d}", 26, fill=INK, outline=SHELL, shadow=False)
        sx = X0 + w - 14 - sc.width
        base.alpha_composite(sc, (int(sx), int(yy - sc.height / 2)))
        coin(base, sx - 26, yy, tt)
        if 0 <= land < 0.3:                                  # +100 pops in the middle of the lip, then CLEAR!
            ctx.paste(base, ctx.pixel_text("+100", 26, fill=YELLOW), X0 + w / 2, yy,
                      scale=ctx.ease_out_back(min(1.0, land / 0.12)))
        elif land >= 0.3 and not c["last"]:
            cl = ctx.pixel_text("CLEAR!", 26, fill=YELLOW, outline=INK, shadow=False)
            ctx.paste(base, cl, X0 + w / 2, yy, scale=ctx.ease_out_back(min(1.0, (land - 0.3) / 0.15)))

    def player_tag():
        if "p1" not in _c:
            name, size = f"P1 {PLAYER}", 18
            im = ctx.pixel_text(name, size, fill=(170, 20, 50), outline=SHELL, shadow=False)
            while im.width > 300 and len(name) > 6:              # a long handle is cut to fit the lip
                name = name[:-1]
                im = ctx.pixel_text(name.rstrip(".") + "..", size, fill=(170, 20, 50), outline=SHELL, shadow=False)
            _c["p1"] = im
        return _c["p1"]

    def banner(base, local, quiet):
        """Level-start banner along the bottom of the screen on link 1 (clear of the faces and of Buh up top):
        PLAYER @HANDLE / LEVEL 1 START. It rises out of the bezel lip, holds, and sinks back."""
        if local >= BANNER:
            return
        if "ban" not in _c:
            l1 = ctx.pixel_text(f"PLAYER {PLAYER}", 34, fill=YELLOW)
            if l1.width > w - 40:
                l1 = l1.resize((w - 40, int(l1.height * (w - 40) / l1.width)), Image.NEAREST)
            l2 = ctx.pixel_text("LEVEL 1  START" + ("" if quiet else "!"), 24, fill=WHITE)
            bh = l1.height + l2.height + 18
            im = Image.new("RGBA", (w, bh), (12, 10, 30, 210))
            im.alpha_composite(l1, ((w - l1.width) // 2, 8))
            im.alpha_composite(l2, ((w - l2.width) // 2, 8 + l1.height + 2))
            _c["ban"] = im
        im = _c["ban"]
        u = max(0.0, min(1.0, local / 0.25, (BANNER - local) / 0.25))
        shown = int(im.height * ctx.ease(u))                   # rows visible above the screen's bottom edge
        if shown > 0:
            base.alpha_composite(im.crop((0, 0, w, shown)), (X0, Y0 + h - shown))

    def coin(base, x, y, tt):
        sq = abs(math.cos(tt * 4))                           # spinning
        cw = max(4, int(26 * sq))
        d = ImageDraw.Draw(base)
        d.ellipse([x - cw / 2, y - 14, x + cw / 2, y + 14], fill=YELLOW, outline=INK, width=4)

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        base = sky()
        stars(base, tt)
        hills(base, tt)
        c = ctx.draw_core(base, seg, tt, X0, Y0, S, frame=bezel)
        if PLAYER and i == 0:
            banner(base, tt - seg["t0"], sens(i))
        hud(base, i, c, tt)
        controls(base, tt, {"press": 0 <= c["since"] < 0.25})
        return ctx.core_finish(base, seg, tt, c)

    return frame
