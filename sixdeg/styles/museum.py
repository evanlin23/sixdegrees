"""Museum: a gallery walk. The show's core photo window (draw_core: camera, Buh, string, name tags) hangs in a gilt
frame on a red damask wall, under a picture light, with a wall label above it (names, year, fact; B stays "? ? ?"
until the reveal, then a glint runs across the label). Between links the camera walks along the wall: the finished
picture slides off to the left as a still in its frame and the next one comes in from the right. A link with a
sensitive person (dictators etc.) hangs in a plain dark archive frame with no picture light and a grey label: no
honour framing, no gold."""
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

    # ---------- wall + floor: a wide strip that scrolls as the camera walks
    def wall():
        if "wall" in st:
            return st["wall"]
        WW = W + STEP * 2
        yy, xx = np.mgrid[0:H, 0:WW].astype(np.float32)
        tex = 1 + 0.045 * (np.sin(xx / 9) * np.sin(yy / 9) > 0.35) + 0.02 * np.sin(xx / 3.1)   # faint damask
        img = WALL[None, None] * tex[..., None] * np.ones((H, WW, 1), np.float32)
        DADO = FLOOR - 190
        img[DADO + 12:FLOOR] *= 0.72
        img[DADO:DADO + 16] = (120, 84, 40)
        img[DADO - 2:DADO] = (176, 140, 80)
        img[FLOOR - 40:FLOOR] = (52, 30, 18)
        img[FLOOR - 42:FLOOR - 38] = (90, 58, 34)
        for k in range(WW // STEP + 1):                 # panel mouldings, repeating every STEP
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
        st["wall"] = img.clip(0, 255).astype(np.uint8)
        # picture light: a scalloped pool of light over a frame (gain map, centred, moves with the frame)
        gy, gx = np.mgrid[0:H, -700:700].astype(np.float32)
        top = 300 + 0.0012 * gx ** 2
        edge = np.clip((gy - top) / 60, 0, 1)
        g = 1.25 * np.exp(-(gx / 470) ** 2 - ((gy - 820) / 640) ** 2) * edge
        g[FLOOR:] = 0.35 * np.exp(-(gx[FLOOR:] / 300) ** 2 - ((gy[FLOOR:] - FLOOR - 40) / 60) ** 2)  # sheen
        st["light"] = g
        return st["wall"]

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
        off = int(pos * STEP) % STEP
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
        # the picture we just walked away from
        if i > 0 and u < 1 and not ctx.C.fj["links"][i - 1].get("lineage"):
            fx = ctx.CX + (i - 1 - pos) * STEP
            sh = Image.new("RGBA", (OW + 60, OH + 60), (0, 0, 0, 0))
            ImageDraw.Draw(sh).rectangle([30, 30, OW + 30, OH + 30], fill=(0, 0, 0, 120))
            base.alpha_composite(sh.filter(ImageFilter.GaussianBlur(14)), (int(fx - OW / 2 - 16), FRAME_TOP - 10 + int(bob)))
            base.alpha_composite(hung(i - 1), (int(fx - OW / 2), FRAME_TOP + int(bob)))
            base.alpha_composite(label(i - 1, True), (int(fx - OW / 2 + 6), LBL_Y0 + int(bob)))
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
            sh = Image.new("RGBA", (OW + 60, OH + 60), (0, 0, 0, 0))
            ImageDraw.Draw(sh).rectangle([30, 30, OW + 30, OH + 30], fill=(0, 0, 0, 120))
            b.alpha_composite(sh.filter(ImageFilter.GaussianBlur(14)), (int(X0 - MAT - T - 16), int(Y0 - MAT - T - 10)))
            b.alpha_composite(fs, (int(X0 - MAT - T), int(Y0 - MAT - T)))

        c = ctx.draw_core(base, seg, tt, fx - PW / 2, FRAME_TOP + T + MAT + bob, SC, frame=mount)
        X0, Y0, w, h = c["rect"]
        return ctx.core_finish(base, seg, tt, c, connected_y=int(Y0 + h - 150))

    return frame
