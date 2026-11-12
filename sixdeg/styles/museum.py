"""Museum: a gallery walk. Each link photo hangs in a gilt frame on a long red wall, with a plaque (names, year,
fact) under it; picture lights pool on the wall over a wooden floor in perspective. A link with a sensitive person
(dictators etc.) hangs in a plain dark archive frame with no picture light and a grey plaque: no honour framing. The camera dollies along
the wall from frame to frame and slowly pushes in on the current one."""
from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

NAME = "museum"


def make(ctx):
    SP = 1150                               # distance between frames along the wall (world px)
    X0 = 700                                # first frame's centre x
    PW, PH = 790, 580                       # photo inside the frame
    MAT, T = 16, 42                         # passe-partout and moulding widths
    OW, OH = PW + 2 * (MAT + T), PH + 2 * (MAT + T)
    FY = 302 + OH // 2                      # frame centre y (world == screen at zoom 1)
    PLQ_W, PLQ_H = 800, 214                 # plaque size
    PLQ_Y = FY + OH // 2 + 24 + PLQ_H // 2  # plaque centre y
    FLOOR = 1392                            # wall meets floor (world y)
    DADO = PLQ_Y + PLQ_H // 2 + 22          # dado rail (world y)
    EYE = 760                               # camera pivot / horizon (screen y)
    DOLLY = 0.95                            # walk time between frames, s
    CAP_Y = 1346
    WALL = np.array([96, 22, 32], np.float32)
    n = ctx.n_links
    WW = 2 * X0 + (n - 1) * SP

    def font(name, size):
        return ImageFont.truetype(str(ctx.FONTS / f"Inter-{name}.otf"), size)

    def fx(j):
        return X0 + j * SP

    # ---------- the gold frame (one sprite, reused for every picture)
    def sensitive_link(j):
        an, bn = ctx.link_names(j)
        return ctx.sensitive(an) or ctx.sensitive(bn)

    def cover_cam(i, w, h, fill):
        """both_cam, but scaled to cover so view()'s blurred fill never shows inside the frame."""
        cx, cy, s = ctx.both_cam(i, w, h, fill)
        ih, iw = ctx.C.imgs[i].shape[:2]
        s = max(s, w / iw, h / ih) * 1.002
        return min(max(cx, w / 2 / s), iw - w / 2 / s), min(max(cy, h / 2 / s), ih - h / 2 / s), s

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
        m0 = T
        out[m0:OH - m0, m0:OW - m0, :3] = (196, 194, 188)
        out[m0:m0 + 3, m0:OW - m0, :3] = (70, 66, 60)
        out[m0:OH - m0, m0:m0 + 3, :3] = (70, 66, 60)
        return out

    def gold_frame():
        yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
        dl, dr, dt, db = xx, OW - 1 - xx, yy, OH - 1 - yy
        d = np.minimum(np.minimum(dl, dr), np.minimum(dt, db))
        p = np.clip(d / T, 0, 1)
        side = np.argmin(np.stack([dl, dr, dt, db]), 0)               # nearest outer edge: 0 L, 1 R, 2 T, 3 B
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
        recess = np.clip(0.5 - h, 0, 0.5)[..., None] * 2                     # red bole showing in the hollows
        rgb = rgb * (1 - 0.35 * recess) + np.array([120, 46, 26], np.float32) * 0.35 * recess
        rgb = rgb * (1 + 0.07 * wear[..., None])
        rgb[(p > 0.97)] *= 0.55                                              # dark sight edge
        rgb[(p < 0.03)] *= 0.6                                               # dark outer edge
        out = np.zeros((OH, OW, 4), np.uint8)
        out[..., :3] = rgb.clip(0, 255)
        out[..., 3] = 255
        # mat with a bevelled inner window
        m0, m1 = T, T + MAT
        out[m0:OH - m0, m0:OW - m0, :3] = (234, 226, 206)
        out[m1 - 3:OH - m1 + 3, m1 - 3:OW - m1 + 3, :3] = (246, 240, 226)
        out[m0:m0 + 3, m0:OW - m0, :3] = (90, 70, 50)
        out[m0:OH - m0, m0:m0 + 3, :3] = (90, 70, 50)
        return out

    def plaque(j, revealed):
        an, bn = ctx.link_names(j)
        im = Image.new("RGBA", (PLQ_W + 16, PLQ_H + 16), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rectangle([10, 12, PLQ_W + 14, PLQ_H + 14], fill=(0, 0, 0, 90))
        sens = sensitive_link(j)
        rim, rim2 = ((110, 108, 104), (150, 148, 142)) if sens else ((150, 116, 52), (196, 166, 108))
        d.rectangle([0, 0, PLQ_W, PLQ_H], fill=(226, 224, 218) if sens else (238, 230, 212), outline=rim, width=4)
        d.rectangle([9, 9, PLQ_W - 9, PLQ_H - 9], outline=rim2, width=1)
        A = ctx.C.labels[an]
        B = ctx.C.labels[bn] if revealed else "?  ?  ?"
        y = 16
        for k, txt in enumerate((A, f"& {B}")):
            size = 44
            while font("Black", size).getlength(txt) > PLQ_W - 50 and size > 24:
                size -= 1
            d.text((PLQ_W / 2, y), txt, font=font("Black", size), fill=(30, 22, 18) if k == 0 or revealed
                   else (150, 120, 90), anchor="ma")
            y += 50
        y += 4
        d.line([PLQ_W / 2 - 120, y, PLQ_W / 2 + 120, y], fill=rim, width=2)
        y += 9
        yr = ctx.year_of(j)
        fact = ctx.fact_of(ctx.spec["links"][j], full=revealed)
        if yr and fact.startswith(str(yr)):
            fact = fact[len(str(yr)):].lstrip(": ,")
        line = f"{yr}  ·  {fact}" if yr else fact
        f = font("Bold", 30)
        words, lines, cur = line.split(), [], ""
        for w in words:
            t = f"{cur} {w}".strip()
            if f.getlength(t) > PLQ_W - 50 and cur:
                lines.append(cur)
                cur = w
            else:
                cur = t
        lines.append(cur)
        for ln in lines[:2]:
            d.text((PLQ_W / 2, y), ln, font=f, fill=(84, 66, 52), anchor="ma")
            y += 36
        return np.asarray(im)

    def cover():
        im = Image.new("RGBA", (PW, PH), (44, 12, 20, 255))
        a = np.asarray(im).copy()
        yy, xx = np.mgrid[0:PH, 0:PW].astype(np.float32)
        fold = 0.8 + 0.2 * np.sin(xx / 23 + np.sin(yy / 60) * 2) ** 2          # draped velvet
        a[..., :3] = (a[..., :3] * fold[..., None]).astype(np.uint8)
        im = Image.fromarray(a)
        ImageDraw.Draw(im).text((PW / 2, PH / 2), "?", font=font("Black", 220), fill=(214, 168, 70), anchor="mm")
        return np.asarray(im)

    _w = {}

    def wall():
        if "w" in _w:
            return _w["w"]
        Hh = FLOOR + 10
        yy, xx = np.mgrid[0:Hh, 0:WW].astype(np.float32)
        tex = 1 + 0.045 * (np.sin(xx / 9) * np.sin(yy / 9) > 0.35) + 0.02 * np.sin(xx / 3.1)   # faint damask
        img = WALL[None, None] * tex[..., None]
        # lower panel, dado rail, baseboard
        img[DADO + 12:FLOOR] *= 0.72
        img[DADO:DADO + 16] = (120, 84, 40)
        img[DADO - 2:DADO] = (176, 140, 80)
        img[FLOOR - 52:FLOOR] = (52, 30, 18)
        img[FLOOR - 54:FLOOR - 50] = (90, 58, 34)
        for k in range(-1, n + 1):                                  # panel mouldings under each frame
            x0 = fx(k) - 380
            for (a, b) in ((x0, x0 + 360), (x0 + 400, x0 + 760)):
                a, b = max(0, a), min(WW - 1, b)
                if b > a:
                    p0, p1 = DADO + 34, FLOOR - 66
                    img[p0:p0 + 4, a:b] *= 1.25
                    img[p1 - 4:p1, a:b] *= 0.7
                    img[p0:p1, a:a + 4] *= 1.25
                    img[p0:p1, b - 4:b] *= 0.7
        # frame shadows
        sh = np.zeros((Hh, WW), np.float32)
        for j in range(n):
            sh[FY - OH // 2 + 22:FY + OH // 2 + 26, fx(j) - OW // 2 + 14:fx(j) + OW // 2 + 18] = 0.6
            sh[PLQ_Y - PLQ_H // 2 + 8:PLQ_Y + PLQ_H // 2 + 12, fx(j) - PLQ_W // 2 + 8:fx(j) + PLQ_W // 2 + 12] = 0.4
        sh = cv2.GaussianBlur(sh, (0, 0), 16)
        img *= (1 - sh)[..., None]
        # picture lights: a scalloped pool of light over every frame
        gain = np.full((Hh, WW), 0.55, np.float32)
        for j in range(-1, n + 1):
            dx = (xx[0] - fx(j))[None, :]
            dy = yy[:, :1] - 640
            top = 250 + 0.0012 * dx ** 2
            edge = np.clip((yy[:, :1] - top) / 60, 0, 1)
            lamp = 0.45 if 0 <= j < n and sensitive_link(j) else 1.35          # no spotlight on a sensitive link
            gain += lamp * np.exp(-(dx / 470) ** 2 - (dy / 620) ** 2) * edge
        img *= gain[..., None]
        img = img.clip(0, 255).astype(np.uint8)
        gf, af = gold_frame(), archive_frame()
        cv_ = cover()
        for j in range(n):
            x0, y0 = fx(j) - OW // 2, FY - OH // 2
            g = (af if sensitive_link(j) else gf).copy()
            ph = np.asarray(ctx.view(j, *cover_cam(j, PW, PH, 0.8), PW, PH))
            g[T + MAT:T + MAT + PH, T + MAT:T + MAT + PW] = ph
            lit = 0.85 + 0.3 * np.exp(-((np.arange(OH) - OH * 0.3) / (OH * 0.7)) ** 2)[:, None, None]
            img[y0:y0 + OH, x0:x0 + OW] = (g[..., :3] * lit).clip(0, 255).astype(np.uint8)
        # picture lights: a brass lamp on an arm over each frame
        for j in range(n):
            if sensitive_link(j):
                continue
            x, top = fx(j), FY - OH // 2
            for dx in (-150, 150):
                cv2.line(img, (x + dx, top + 4), (x + dx // 3, 262), (120, 84, 30), 7)
                cv2.line(img, (x + dx, top + 4), (x + dx // 3, 262), (210, 170, 90), 2)
            cv2.rectangle(img, (x - 120, 248), (x + 120, 270), (150, 108, 40), -1)
            cv2.rectangle(img, (x - 120, 248), (x + 120, 254), (240, 210, 140), -1)
            cv2.rectangle(img, (x - 114, 270), (x + 114, 276), (255, 246, 214), -1)
        _w["w"] = img
        _w["plq"] = {}
        _w["cover"] = cv_
        return img

    def plq(j, rev):
        wall()
        if (j, rev) not in _w["plq"]:
            _w["plq"][(j, rev)] = plaque(j, rev)
        return _w["plq"][(j, rev)]

    def blit(base, spr, x0, y0, z):
        """Paste an RGBA sprite whose world top-left is (x0, y0) through the camera (affine, zoom z)."""
        h, w = spr.shape[:2]
        sx0, sy0 = scr(x0, y0)
        sw, shh = w * z, h * z
        a0, b0 = max(0, int(sx0)), max(0, int(sy0))
        a1, b1 = min(ctx.W, int(math.ceil(sx0 + sw))), min(ctx.H, int(math.ceil(sy0 + shh)))
        if a1 <= a0 or b1 <= b0:
            return
        M = np.float32([[z, 0, sx0 - a0], [0, z, sy0 - b0]])
        out = cv2.warpAffine(spr, M, (a1 - a0, b1 - b0), flags=cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        al = out[..., 3:4].astype(np.float32) / 255
        reg = base[b0:b1, a0:a1].astype(np.float32)
        base[b0:b1, a0:a1] = (out[..., :3] * al + reg * (1 - al)).astype(np.uint8)

    cam = {}

    def scr(x, y):
        return ctx.CX + (x - cam["x"]) * cam["z"], EYE + (y - EYE) * cam["z"]

    def zoom_at(seg, local):
        dur = seg["t1"] - seg["t0"]
        u = ctx.ease(max(0.0, local - (DOLLY if seg["i"] else 0)) / max(0.5, dur - (DOLLY if seg["i"] else 0)))
        return 0.98 + 0.13 * u

    def camera(seg, local):
        i = seg["i"]
        if i == 0 or local >= DOLLY:
            x, z = fx(i), zoom_at(seg, local)
            x += 14 * math.sin(local * 0.8)
            bob = 0.0
        else:
            prev = next(s for s in ctx.segs if s.get("i") == i - 1)
            u = local / DOLLY
            e = ctx.ease(u)
            x = fx(i - 1) + (fx(i) - fx(i - 1)) * e
            z0 = zoom_at(prev, prev["t1"] - prev["t0"])
            z = z0 + (0.98 - z0) * e - 0.2 * math.sin(math.pi * u)              # step back while walking
            bob = 7 * math.sin(u * math.pi * 4) * math.sin(math.pi * u)           # footsteps
        return x, z, bob

    # floor
    PLANK = 92
    tones = [np.array(c, np.float32) for c in ((128, 78, 44), (112, 66, 36), (138, 86, 48), (118, 72, 40))]
    glow = None

    def floor(base, ys):
        nonlocal glow
        ys = int(ys)
        if ys >= ctx.H:
            return
        hgt = ctx.H - ys
        z = cam["z"]
        span = (ctx.H - EYE) / max(1.0, ys - EYE)                  # how much wider the near edge is
        wl = cam["x"] - (ctx.CX / z)
        wr = cam["x"] + ((ctx.W - ctx.CX) / z)
        k0 = int(math.floor((cam["x"] - (ctx.CX + 50) / (z * 1.0)) / PLANK)) - 1
        k1 = int(math.ceil((cam["x"] + (ctx.W - ctx.CX + 50) / (z * 1.0)) / PLANK)) + 1
        sub = np.zeros((hgt, ctx.W, 3), np.uint8)
        for k in range(k0, k1):
            col = tuple(int(v) for v in tones[(k * 7 + 3) % 4])
            xa, xb = k * PLANK, (k + 1) * PLANK
            pa = ctx.CX + (xa - cam["x"]) * z
            pb = ctx.CX + (xb - cam["x"]) * z
            qa = ctx.CX + (xa - cam["x"]) * z * span
            qb = ctx.CX + (xb - cam["x"]) * z * span
            if max(pa, qa) < -50 and max(pb, qb) < -50 or min(pa, qa) > ctx.W + 50 and min(pb, qb) > ctx.W + 50:
                continue
            poly = np.array([[pa, 0], [pb, 0], [qb, hgt], [qa, hgt]], np.int32)
            cv2.fillPoly(sub, [poly], col)
            cv2.line(sub, (int(pa), 0), (int(qa), hgt), (60, 34, 18), 2)
            dep = 0.55 + 0.4 * ((k * 0.37) % 1)                     # a butt joint somewhere along the board
            yj = (ys - EYE) / dep + EYE - ys
            if 0 < yj < hgt:
                f = (yj + ys - EYE) / (ys - EYE)
                cv2.line(sub, (int(ctx.CX + (xa - cam["x"]) * z * f), int(yj)),
                         (int(ctx.CX + (xb - cam["x"]) * z * f), int(yj)), (70, 40, 22), 2)
        shade = (0.55 + 0.45 * (np.arange(hgt, dtype=np.float32) / max(1, hgt)) ** 0.7)[:, None, None]
        f = sub.astype(np.float32) * shade
        if glow is None:
            gy, gx = np.mgrid[-90:90, -260:260].astype(np.float32)
            glow = np.exp(-(gx / 160) ** 2 - (gy / 45) ** 2)[..., None] * np.array([90, 60, 30], np.float32)
        for j in range(n):                                          # the lamps' sheen on the polished floor
            gx_ = int(ctx.CX + (fx(j) - cam["x"]) * z * 1.15)
            gy_ = int(min(hgt - 91, 60 * z + 40))
            x0, x1 = max(0, gx_ - 260), min(ctx.W, gx_ + 260)
            if x1 > x0 and gy_ > 90:
                f[gy_ - 90:gy_ + 90, x0:x1] += glow[:, x0 - (gx_ - 260):x1 - (gx_ - 260)]
        f[:10] *= np.linspace(0.4, 1, 10, dtype=np.float32)[:, None, None]
        base[ys:] = f.clip(0, 255).astype(np.uint8)

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        x, z, bob = camera(seg, local)
        cam.update(x=x, z=z)
        w = wall()
        M = np.float32([[z, 0, ctx.CX - x * z], [0, z, EYE - EYE * z + bob]])
        base = cv2.warpAffine(w, M, (ctx.W, ctx.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        R = seg.get("rv", ctx.REVEAL) * seg["speech"] + ctx.THROW
        for j in range(n):
            sx, _ = scr(fx(j), FY)
            if abs(sx - ctx.CX) > ctx.W:
                continue
            if j > i:                                                # pictures still to come: draped
                blit(base, _w["cover"], fx(j) - PW // 2, FY - PH // 2 + bob / z, z)
            p = plq(j, j < i or (j == i and local > R))
            blit(base, p, fx(j) - PLQ_W // 2, PLQ_Y - PLQ_H // 2 + bob / z, z)
        floor(base, EYE + (FLOOR - EYE) * z + bob)
        out = Image.fromarray(base).convert("RGBA")
        since = local - R
        if 0 < since < 0.5:                             # a glint across the plaque as B's name lands
            gx = int(scr(fx(i) - PLQ_W / 2 + (PLQ_W + 200) * since / 0.5 - 100, PLQ_Y)[0])
            py0, py1 = scr(0, PLQ_Y - PLQ_H / 2)[1], scr(0, PLQ_Y + PLQ_H / 2)[1]
            hh = int(py1 - py0)
            gl = Image.new("L", (160, hh), 0)
            ImageDraw.Draw(gl).polygon([(70, 0), (110, 0), (60, hh), (20, hh)], fill=120)
            gl = gl.filter(ImageFilter.GaussianBlur(8))
            layer = Image.new("RGBA", (160, hh), (255, 252, 235, 0))
            layer.putalpha(gl)
            if 0 <= gx < ctx.W - 160:
                out.alpha_composite(layer, (gx, int(py0 + bob)))
        ctx.header(out, i)
        ctx.connected(out, seg, tt, y=930)                 # over the photo, below the faces
        seg["cap"].draw(out, tt, y=CAP_Y)
        return out

    return frame
