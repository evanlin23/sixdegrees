"""Museum: a gallery walk. The show's core photo window (draw_core: camera, Buh, string, name tags) hangs in a gilt
frame on a red damask wall, under a picture light, with a wall label above it (names, year, fact; B stays "? ? ?"
until the reveal, then a glint runs across the label). Between links the camera walks along the wall: the finished
picture slides off to the left as a still in its frame and the next one comes in from the right. The wall is ONE
long wall for the whole video (owner, 2026-11-13: continuity across slides): link j's picture hangs at its own fixed
spot j * STEP along it, every finished picture stays hung there as a still, and the decorative paintings, plinths,
ropes and benches between them are fixed to the wall too, so nothing reshuffles from link to link. A link with a
sensitive person (dictators etc.) hangs in a plain dark archive frame with no picture light and a grey label: no
honour framing, no gold.
The room is furnished like a real gallery, all on the wall strip so it walks with the camera: neighbouring paintings
(made-up landscapes, never people) peek in at both edges, a marble plinth with a vase and a small room plaque stands
on the right, and on the floor a red velvet rope on brass stanchions and a gallery bench, all below the captions and
the Subscribe button and never over the photo window."""
from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

NAME = "museum"

SC = 0.75                         # core window scale: 660 x 712
MAT, T = 18, 40                   # passe-partout and moulding widths
LBL_Y0, LBL_H = 1344, 142         # wall label (under the frame, left: clear of the captions and the credit)
FRAME_TOP = 405                   # outer top of the frame
STEP = 1120                       # distance between pictures along the wall (px)
DOLLY = 0.6                       # walk time between pictures, s
FLOOR = 1600                      # wall meets floor (screen y)
WALL = np.array([96, 22, 32], np.float32)


def make(ctx):
    W, H = ctx.W, ctx.H
    n = ctx.n_links
    PW, PH = int(880 * SC), int(950 * SC)
    OW, OH = PW + 2 * (MAT + T), PH + 2 * (MAT + T)
    LBL_W = 380
    st = {}

    def font(name, size):
        key = ("f", name, size)
        if key not in st:
            st[key] = ImageFont.truetype(str(ctx.FONTS / f"Inter-{name}.otf"), size)
        return st[key]

    def sens(j):
        an, bn = ctx.link_names(j)
        return ctx.sensitive(an) or ctx.sensitive(bn)

    # ---------- frames (one sprite each, empty window)
    def archive_frame():
        """A plain, flat dark-wood frame with a grey mat (sensitive links)."""
        yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
        d = np.minimum(np.minimum(xx, OW - 1 - xx), np.minimum(yy, OH - 1 - yy))
        p = np.clip(d / T, 0, 1)
        grain = 0.06 * np.sin(np.where(np.minimum(xx, OW - 1 - xx) < np.minimum(yy, OH - 1 - yy), yy, xx) / 3.3)
        v = 0.75 + 0.25 * np.sin(np.pi * p) + grain
        v = np.where((p > 0.02) & (p < 0.06), 1.25, v)
        v = np.where(p > 0.95, 0.6, v)
        out = np.zeros((OH, OW, 4), np.uint8)
        out[..., :3] = (np.array([34, 28, 24], np.float32) * v[..., None]).clip(0, 255)
        out[..., 3] = 255
        out[T:OH - T, T:OW - T, :3] = (196, 194, 188)
        out[T:T + 3, T:OW - T, :3] = (70, 66, 60)
        out[T:OH - T, T:T + 3, :3] = (70, 66, 60)
        return out

    def gold_frame():
        yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
        dl, dr, dt, db = xx, OW - 1 - xx, yy, OH - 1 - yy
        d = np.minimum(np.minimum(dl, dr), np.minimum(dt, db))
        p = np.clip(d / T, 0, 1)
        side = np.argmin(np.stack([dl, dr, dt, db]), 0)
        along = np.where(side < 2, yy, xx)
        h = 0.55 * np.clip(np.sin(np.pi * np.clip(p * 1.05, 0, 1)), 0, 1) ** 0.6 + 0.18 * np.sin(3 * np.pi * p) \
            + 0.22 * np.cos(2 * np.pi * along / 15) ** 2 * np.exp(-((p - 0.58) / 0.07) ** 2) \
            + 0.12 * np.exp(-((p - 0.16) / 0.05) ** 2) + 0.1 * np.exp(-((p - 0.88) / 0.04) ** 2)
        for cx, cy in ((T / 2, T / 2), (OW - T / 2, T / 2), (T / 2, OH - T / 2), (OW - T / 2, OH - T / 2)):
            r = np.hypot(xx - cx, yy - cy) / (T * 0.62)
            rose = np.clip(1 - r ** 2, 0, 1) ** 0.5 * (0.8 + 0.25 * np.cos(np.arctan2(yy - cy, xx - cx) * 8))
            h = np.maximum(h, rose * (r < 1))
        gy, gx = np.gradient(h)
        shade = np.clip(0.55 + 9.0 * (-0.6 * gx - 0.8 * gy) + 0.25 * h, 0, 1.3)
        ramp = np.array([[34, 20, 8], [92, 62, 24], [146, 108, 46], [186, 148, 78], [222, 194, 128]], np.float32)
        k = np.clip(shade * 0.92, 0, 1.0) * (len(ramp) - 1)
        k0 = np.floor(k).astype(int).clip(0, len(ramp) - 2)
        rgb = ramp[k0] + (ramp[k0 + 1] - ramp[k0]) * (k - k0)[..., None]
        rng = np.random.default_rng(7)
        wear = cv2.GaussianBlur(rng.normal(0, 1, (OH, OW)).astype(np.float32), (0, 0), 3)
        recess = np.clip(0.5 - h, 0, 0.5)[..., None] * 2
        rgb = rgb * (1 - 0.35 * recess) + np.array([120, 46, 26], np.float32) * 0.35 * recess
        rgb = rgb * (1 + 0.07 * wear[..., None])
        rgb[(p > 0.97)] *= 0.55
        rgb[(p < 0.03)] *= 0.6
        out = np.zeros((OH, OW, 4), np.uint8)
        out[..., :3] = rgb.clip(0, 255)
        out[..., 3] = 255
        m0, m1 = T, T + MAT
        out[m0:OH - m0, m0:OW - m0, :3] = (234, 226, 206)
        out[m1 - 3:OH - m1 + 3, m1 - 3:OW - m1 + 3, :3] = (246, 240, 226)
        out[m0:m0 + 3, m0:OW - m0, :3] = (90, 70, 50)
        out[m0:OH - m0, m0:m0 + 3, :3] = (90, 70, 50)
        return out

    def frame_sprite(j):
        key = "arch" if sens(j) else "gold"
        if key not in st:
            st[key] = Image.fromarray(archive_frame() if key == "arch" else gold_frame())
        return st[key]

    def hung(j):
        """Link j's finished picture as it leaves: frame + a still of the photo (both faces)."""
        key = ("hung", j)
        if key not in st:
            im = frame_sprite(j).copy()
            im.alpha_composite(ctx.view(j, *ctx.both_cam(j, PW, PH, 0.75), PW, PH).convert("RGBA"), (T + MAT, T + MAT))
            st[key] = im
        return st[key]

    # ---------- wall label
    def wrap(text, f, width, maxl):
        lines, cur = [], ""
        for w in text.split():
            t = f"{cur} {w}".strip()
            if f.getlength(t) > width and cur:
                lines.append(cur)
                cur = w
            else:
                cur = t
        lines.append(cur)
        if len(lines) > maxl:
            lines = lines[:maxl]
            lines[-1] = lines[-1].rstrip(",;:…") + "…"
        return lines

    def label(j, revealed):
        key = ("lbl", j, revealed)
        if key in st:
            return st[key]
        an, bn = ctx.link_names(j)
        s = sens(j)
        im = Image.new("RGBA", (LBL_W + 12, LBL_H + 12), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rectangle([8, 10, LBL_W + 10, LBL_H + 10], fill=(0, 0, 0, 90))
        rim = (110, 108, 104) if s else (150, 116, 52)
        d.rectangle([0, 0, LBL_W, LBL_H], fill=(226, 224, 218) if s else (238, 230, 212), outline=rim, width=3)
        A = ctx.C.labels[an]
        B = ctx.C.labels[bn] if revealed else "?  ?  ?"
        y = 10
        for k, txt in enumerate((A, f"& {B}")):
            size = 30
            while font("Black", size).getlength(txt) > LBL_W - 36 and size > 16:
                size -= 1
            d.text((18, y), txt, font=font("Black", size), fill=(30, 22, 18) if k == 0 or revealed
                   else (150, 120, 90), anchor="la")
            y += 34
        d.line([18, y + 2, 140, y + 2], fill=rim, width=2)
        y += 10
        yr = ctx.year_of(j)
        fact = ctx.fact_of(ctx.spec["links"][j], full=revealed)
        if yr and fact.startswith(str(yr)):
            fact = fact[len(str(yr)):].lstrip(": ,")
        line = f"{yr}  ·  {fact}" if yr else fact
        for ln in wrap(line, font("SemiBold", 19), LBL_W - 36, 2):
            d.text((18, y), ln, font=font("SemiBold", 19), fill=(84, 66, 52), anchor="la")
            y += 23
        st[key] = im
        return im

    # ---------- gallery furniture, drawn into the wall strip (it is dimmed x0.55 at frame time: colours run bright)
    def neighbour(kind):
        """A small made-up painting in a thin gilt frame (no people): 0 dusk landscape, 1 seascape, 2 still life."""
        fw, fh, b = 300, 380, 22
        im = Image.new("RGB", (fw, fh), (60, 40, 20))
        d = ImageDraw.Draw(im)
        for k, col in enumerate(((120, 80, 30), (230, 190, 110), (200, 156, 80), (150, 104, 44))):
            d.rectangle([k * 5, k * 5, fw - 1 - k * 5, fh - 1 - k * 5], outline=col, width=5)
        x0, y0, x1, y1 = b, b, fw - b, fh - b
        cw, ch = x1 - x0, y1 - y0
        cv = Image.new("RGB", (cw, ch))
        cd = ImageDraw.Draw(cv)
        if kind == 0:
            for yy in range(ch):                       # dusk sky into hills
                u = yy / ch
                cd.line([(0, yy), (cw, yy)], fill=(int(250 - 90 * u), int(180 - 60 * u), int(120 + 40 * u)))
            cd.ellipse([cw * 0.55, ch * 0.30, cw * 0.80, ch * 0.30 + cw * 0.25], fill=(255, 236, 170))
            cd.polygon([(0, ch * 0.62), (cw * 0.35, ch * 0.48), (cw * 0.7, ch * 0.6), (cw, ch * 0.52), (cw, ch), (0, ch)], fill=(110, 130, 80))
            cd.polygon([(0, ch * 0.78), (cw * 0.5, ch * 0.66), (cw, ch * 0.8), (cw, ch), (0, ch)], fill=(70, 96, 54))
            for tx in (0.18, 0.26, 0.8):               # cypress trees
                cd.ellipse([cw * tx - 9, ch * 0.52, cw * tx + 9, ch * 0.74], fill=(40, 64, 36))
        elif kind == 1:
            cd.rectangle([0, 0, cw, ch * 0.55], fill=(150, 190, 220))
            cd.ellipse([cw * 0.1, ch * 0.12, cw * 0.5, ch * 0.24], fill=(236, 240, 244))
            cd.rectangle([0, ch * 0.55, cw, ch], fill=(40, 90, 130))
            for k in range(7):
                yy = ch * (0.6 + 0.055 * k)
                cd.arc([cw * (k % 3) * 0.3 - 20, yy - 8, cw * (k % 3) * 0.3 + 70, yy + 12], 200, 340, fill=(200, 226, 240), width=3)
            cd.polygon([(cw * 0.55, ch * 0.54), (cw * 0.62, ch * 0.3), (cw * 0.62, ch * 0.54)], fill=(244, 236, 220))
            cd.rectangle([cw * 0.5, ch * 0.54, cw * 0.72, ch * 0.58], fill=(110, 60, 30))
        else:
            cd.rectangle([0, 0, cw, ch], fill=(70, 46, 40))
            cd.rectangle([0, ch * 0.66, cw, ch], fill=(140, 96, 60))
            cd.polygon([(cw * 0.3, ch * 0.66), (cw * 0.36, ch * 0.4), (cw * 0.5, ch * 0.32), (cw * 0.64, ch * 0.4), (cw * 0.7, ch * 0.66)], fill=(70, 110, 150))
            for fx_, fy_, col in ((0.42, 0.24, (230, 80, 60)), (0.56, 0.2, (250, 210, 90)), (0.5, 0.28, (240, 240, 230))):
                cd.ellipse([cw * fx_ - 22, ch * fy_ - 22, cw * fx_ + 22, ch * fy_ + 22], fill=col)
            for fx_ in (0.2, 0.8):
                cd.ellipse([cw * fx_ - 26, ch * 0.62, cw * fx_ + 26, ch * 0.72], fill=(220, 150, 50))
        im.paste(cv, (x0, y0))
        return im

    def furnish(arr):
        """Neighbour paintings, a plinth with a vase + plaque, velvet rope on stanchions and a bench, every STEP.
        Fixed to the wall: the painting between pictures k and k+1 is always the same one (kind (k + 1) % 3)."""
        im = Image.fromarray(arr)
        d = ImageDraw.Draw(im)
        brass = [(140, 96, 30), (230, 180, 80), (255, 230, 150)]
        for k in range(-1, n + 1):
            bx = k * STEP + W // 2                     # where main picture k hangs on the wall
            # neighbour painting between two main pictures: peeks in at the screen edges when the camera rests
            npic = neighbour((k + 1) % 3)
            nx = bx + STEP // 2 - npic.width // 2
            d.rectangle([nx + 10, 470 + 14, nx + npic.width + 14, 470 + npic.height + 18], fill=(30, 8, 12))
            im.paste(npic, (nx, 470))
            # plinth + vase on the right, against the wall
            px0, px1, pt = bx + 395, bx + 525, FLOOR - 170
            d.rectangle([px0 + 10, pt + 10, px1 + 14, FLOOR + 8], fill=(30, 10, 14))
            d.rectangle([px0, pt, px1, FLOOR], fill=(236, 230, 220))
            d.rectangle([px0 - 8, pt - 14, px1 + 8, pt], fill=(250, 246, 238))
            d.rectangle([px0 - 8, FLOOR - 14, px1 + 8, FLOOR], fill=(214, 206, 194))
            d.line([px1 - 10, pt, px1 - 10, FLOOR - 14], fill=(200, 194, 184), width=6)        # shaded side
            d.rectangle([px0 + 22, pt + 40, px1 - 22, pt + 72], fill=brass[0])                 # room plaque
            d.rectangle([px0 + 25, pt + 43, px1 - 25, pt + 69], fill=brass[1])
            d.text(((px0 + px1) / 2, pt + 56), "HALL OF LINKS", font=font("Bold", 11), fill=(70, 44, 10), anchor="mm")
            vx, vb = (px0 + px1) / 2, pt - 14          # an amphora, terracotta with a black band
            d.polygon([(vx - 16, vb), (vx - 34, vb - 40), (vx - 38, vb - 70), (vx - 24, vb - 100), (vx - 12, vb - 112),
                       (vx - 16, vb - 124), (vx + 16, vb - 124), (vx + 12, vb - 112), (vx + 24, vb - 100), (vx + 38, vb - 70),
                       (vx + 34, vb - 40), (vx + 16, vb)], fill=(214, 120, 64), outline=(60, 24, 10))
            d.rectangle([vx - 37, vb - 74, vx + 37, vb - 58], fill=(40, 24, 20))
            for hx in (-1, 1):
                d.arc([vx + hx * 30 - 14, vb - 118, vx + hx * 30 + 14, vb - 86], 0, 360, fill=(60, 24, 10), width=4)
            # velvet rope on brass stanchions, in front of the main picture
            posts = (bx - 330, bx + 330)
            top_y, base_y = FLOOR - 40, FLOOR + 110
            for sx_ in posts:
                d.ellipse([sx_ - 36, base_y - 10, sx_ + 36, base_y + 14], fill=(40, 20, 10))   # shadow
                d.ellipse([sx_ - 30, base_y - 14, sx_ + 30, base_y + 6], fill=brass[1], outline=brass[0], width=3)
                d.rectangle([sx_ - 7, top_y, sx_ + 7, base_y - 6], fill=brass[1], outline=brass[0], width=2)
                d.line([sx_ - 2, top_y + 4, sx_ - 2, base_y - 10], fill=brass[2], width=3)
                d.ellipse([sx_ - 13, top_y - 22, sx_ + 13, top_y + 4], fill=brass[1], outline=brass[0], width=2)
                d.ellipse([sx_ - 6, top_y - 17, sx_ + 1, top_y - 10], fill=brass[2])
            rope = [(posts[0] + (posts[1] - posts[0]) * q / 40,
                     top_y + 4 + 70 * math.sin(math.pi * q / 40)) for q in range(41)]
            d.line(rope, fill=(70, 6, 16), width=16, joint="curve")
            d.line(rope, fill=(196, 30, 52), width=10, joint="curve")
            d.line([(x, y - 3) for x, y in rope], fill=(240, 90, 100), width=2)
            # a gallery bench, front centre (dark leather on a wooden frame)
            bl, br, by = bx - 250, bx + 250, H - 150
            d.rectangle([bl + 14, by + 20, br + 14, by + 64], fill=(40, 22, 12))
            for lx_ in (bl + 26, br - 46):
                d.rectangle([lx_, by + 30, lx_ + 20, by + 120], fill=(110, 66, 34), outline=(50, 28, 14), width=2)
            d.rounded_rectangle([bl, by, br, by + 44], 14, fill=(64, 48, 40), outline=(26, 16, 12), width=3)
            d.line([bl + 16, by + 10, br - 16, by + 10], fill=(120, 96, 82), width=4)
            for tx_ in range(bl + 60, br - 40, 80):    # tufting buttons
                d.ellipse([tx_ - 4, by + 22, tx_ + 4, by + 30], fill=(30, 20, 16))
        return np.asarray(im.convert("RGB")).copy()

    # ---------- wall + floor: one long strip for the whole video (picture j centred at x = j * STEP + W / 2)
    def wall():
        if "wall" in st:
            return st["wall"]
        WW = W + STEP * max(1, n - 1)
        yy, xx = np.mgrid[0:H, 0:WW].astype(np.float32)
        tex = 1 + 0.045 * (np.sin(xx / 9) * np.sin(yy / 9) > 0.35) + 0.02 * np.sin(xx / 3.1)   # faint damask
        img = WALL[None, None] * tex[..., None] * np.ones((H, WW, 1), np.float32)
        DADO = FLOOR - 190
        img[DADO + 12:FLOOR] *= 0.72
        img[DADO:DADO + 16] = (120, 84, 40)
        img[DADO - 2:DADO] = (176, 140, 80)
        img[FLOOR - 40:FLOOR] = (52, 30, 18)
        img[FLOOR - 42:FLOOR - 38] = (90, 58, 34)
        for k in range(-1, n + 1):                      # panel mouldings, repeating every STEP
            x0 = k * STEP + W // 2 - 380
            for (a, b) in ((x0, x0 + 360), (x0 + 400, x0 + 760)):
                a, b = max(0, a), min(WW - 1, b)
                if b > a:
                    p0, p1 = DADO + 30, FLOOR - 54
                    img[p0:p0 + 4, a:b] *= 1.25
                    img[p1 - 4:p1, a:b] *= 0.7
                    img[p0:p1, a:a + 4] *= 1.25
                    img[p0:p1, b - 4:b] *= 0.7
        # floor: polished boards parallel to the wall, darker near the wall
        PL = 46
        fy = np.arange(FLOOR, H)
        rows = ((fy - FLOOR) // PL).astype(int)
        tones = np.array([(128, 78, 44), (112, 66, 36), (138, 86, 48), (118, 72, 40)], np.float32)
        fl = tones[rows % 4][:, None, :] * np.ones((1, WW, 1), np.float32)
        joints = ((fy - FLOOR) % PL < 2)
        fl[joints] = (60, 34, 18)
        for r in range(rows.max() + 1):                 # butt joints, staggered per board
            ys = slice(FLOOR + r * PL, min(H, FLOOR + (r + 1) * PL))
            for x in range((r * 397) % 520, WW, 520):
                fl[ys.start - FLOOR:ys.stop - FLOOR, x:x + 2] = (70, 40, 22)
        shade = (0.55 + 0.45 * ((fy - FLOOR) / (H - FLOOR)) ** 0.7)[:, None, None]
        img[FLOOR:] = fl * shade
        st["wall"] = furnish(img.clip(0, 255).astype(np.uint8))
        # picture light: a scalloped pool of light over a frame (gain map, centred, moves with the frame)
        gy, gx = np.mgrid[0:H, -700:700].astype(np.float32)
        top = 300 + 0.0012 * gx ** 2
        edge = np.clip((gy - top) / 60, 0, 1)
        g = 1.25 * np.exp(-(gx / 470) ** 2 - ((gy - 820) / 640) ** 2) * edge
        g[FLOOR:] = 0.35 * np.exp(-(gx[FLOOR:] / 300) ** 2 - ((gy[FLOOR:] - FLOOR - 40) / 60) ** 2)  # sheen
        st["light"] = g
        return st["wall"]

    def drop():
        """The soft shadow a hung frame casts on the wall (cached)."""
        if "drop" not in st:
            sh = Image.new("RGBA", (OW + 60, OH + 60), (0, 0, 0, 0))
            ImageDraw.Draw(sh).rectangle([30, 30, OW + 30, OH + 30], fill=(0, 0, 0, 120))
            st["drop"] = sh.filter(ImageFilter.GaussianBlur(14))
        return st["drop"]

    def walk(i, local):
        """Camera position along the wall (in steps) and the walk progress (1 = settled)."""
        if i == 0:
            return 0.0, 1.0
        u = min(1.0, local / DOLLY)
        return i - 1 + ctx.ease(u), u

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        pos, u = walk(i, local)
        bob = 6 * math.sin(u * math.pi * 4) * math.sin(math.pi * u) if u < 1 else 0.0   # footsteps
        wl = wall()
        off = max(0, min(wl.shape[1] - W, int(round(pos * STEP))))   # the one long wall: no wrap, no reshuffle
        arr = wl[:, off:off + W].astype(np.float32) * 0.55
        # picture lights on the frames in view (none on a sensitive link)
        for j in (i - 1, i):
            if 0 <= j < n and not sens(j):
                fx = int(ctx.CX + (j - pos) * STEP)
                a0, a1 = max(0, fx - 700), min(W, fx + 700)
                if a1 > a0:
                    arr[:, a0:a1] *= 1 + st["light"][:, a0 - (fx - 700):a1 - (fx - 700), None]
        if bob:
            arr = np.roll(arr, int(bob), axis=0)
        base = Image.fromarray(arr.clip(0, 255).astype(np.uint8)).convert("RGBA")
        R = seg.get("rv", ctx.REVEAL) * seg["speech"] + ctx.THROW
        revealed = local > R
        # the pictures we already walked past stay hung at their own spots (stills), wherever they are in view
        for j in range(i):
            fx = ctx.CX + (j - pos) * STEP
            if ctx.C.fj["links"][j].get("lineage") or not (-OW / 2 - 40 < fx < W + OW / 2 + 40):
                continue
            base.alpha_composite(drop(), (int(fx - OW / 2 - 16), FRAME_TOP - 10 + int(bob)))
            base.alpha_composite(hung(j), (int(fx - OW / 2), FRAME_TOP + int(bob)))
            base.alpha_composite(label(j, True), (int(fx - OW / 2 + 6), LBL_Y0 + int(bob)))
        fx = ctx.CX + (i - pos) * STEP
        lx = int(fx - OW / 2 + 6)
        since = local - R
        lb = label(i, revealed)
        if 0 < since < 0.5:                             # a glint across the label as B's name lands (on it only)
            gl = Image.new("L", lb.size, 0)
            gx = (LBL_W + 200) * since / 0.5 - 100
            ImageDraw.Draw(gl).polygon([(gx + 70, 0), (gx + 110, 0), (gx + 60, LBL_H), (gx + 20, LBL_H)], fill=120)
            gl = gl.filter(ImageFilter.GaussianBlur(8))
            card = Image.new("L", lb.size, 0)
            ImageDraw.Draw(card).rectangle([0, 0, LBL_W, LBL_H], fill=255)
            layer = Image.new("RGBA", lb.size, (255, 252, 235, 0))
            layer.putalpha(Image.fromarray((np.asarray(gl, np.float32) * np.asarray(card, np.float32) / 255)
                                           .astype(np.uint8)))
            lb = lb.copy()
            lb.alpha_composite(layer)
        if not (i == n - 1 and since > 0.45):              # the Subscribe button takes this spot at the end
            base.alpha_composite(lb, (lx, LBL_Y0 + int(bob)))
        fs = frame_sprite(i)

        def mount(b, X0, Y0, w, h):
            b.alpha_composite(drop(), (int(X0 - MAT - T - 16), int(Y0 - MAT - T - 10)))
            b.alpha_composite(fs, (int(X0 - MAT - T), int(Y0 - MAT - T)))

        c = ctx.draw_core(base, seg, tt, fx - PW / 2, FRAME_TOP + T + MAT + bob, SC, frame=mount)
        X0, Y0, w, h = c["rect"]
        return ctx.core_finish(base, seg, tt, c, connected_y=int(Y0 + h - 150))

    return frame
