"""Film reel: the show's core photo window (draw_core: camera, Buh, string, name tags) is the frame in the gate of a
strip of 35 mm film running down the screen: sprocket holes, frame lines, orange edge markings (frame number, year)
and a warm projector beam. At each new link the strip pulls the next frame into the gate; the frame that just
played slides up out of the gate as a still, the frames still to come wait below as dark unexposed leader (never a
spoiler of the next photo).

Around the strip the booth is alive: the projector beam sways slowly, dust motes drift up through it, a warm light
leak breathes along one film edge every few seconds and film grain flickers on the margins (never over the picture)."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

NAME = "filmreel"

SC = 0.8                       # core window scale (880x950 -> 704x760)
GATE_Y = 438                   # top of the frame in the gate
LINE = 46                      # frame line (gap between frames)
MARGIN = 74                    # film edge outside the picture (sprocket holes live here)
PULL = 0.42                    # seconds for the strip to pull the next frame into the gate
FILM = (30, 22, 16)            # film base (dark amber-brown)
EDGE = (255, 156, 40)          # edge-print orange


def make(ctx):
    W, H = ctx.W, ctx.H
    fw, fh = int(880 * SC), int(950 * SC)
    PITCH = fh + LINE
    X0 = int(ctx.CX - fw / 2)
    SX0, SX1 = X0 - MARGIN, X0 + fw + MARGIN            # the strip's outer edges
    st = {}

    def font(name, size):
        key = ("f", name, size)
        if key not in st:
            st[key] = ImageFont.truetype(str(ctx.FONTS / f"Inter-{name}.otf"), size)
        return st[key]

    SW = 70                                             # backdrop overscan each way (room for the beam to sway)

    def backdrop(tt):
        """Projection-booth dark with a warm beam behind the strip. The big gradient is cached once; per frame it
        is cropped at a slowly swaying offset (the beam drifts and the booth air never stands still)."""
        if "bg" not in st:
            BW_, BH_ = W + 2 * SW, H + 2 * SW
            yy, xx = np.mgrid[0:BH_, 0:BW_].astype(np.float32)
            r = np.hypot((xx - BW_ / 2) / W, (yy - 820 - SW) / H)
            v = np.clip(1 - r * 1.6, 0, 1)[..., None] ** 1.5
            c = np.array([10, 8, 14], np.float32) * (1 - v) + np.array([70, 48, 30], np.float32) * v
            st["bg"] = Image.fromarray(np.dstack([c.clip(0, 255).astype(np.uint8), np.full((BH_, BW_), 255, np.uint8)]))
        ox = SW + SW * 0.8 * math.sin(tt * 0.37) + 3 * math.sin(tt * 2.1)
        oy = SW + SW * 0.6 * math.sin(tt * 0.27 + 1.1)
        return st["bg"].crop((int(ox), int(oy), int(ox) + W, int(oy) + H))

    def grain(k):
        """One of a few cycling film-grain layers for the side (booth + film edge) strips: speckles, no picture."""
        key = ("grain", k)
        if key not in st:
            rng = np.random.default_rng(100 + k)
            gw, gh = X0 // 2, H // 2
            n = rng.normal(0, 1, (gh, gw))
            a = np.zeros((gh, gw, 4), np.uint8)
            a[..., :3] = np.where(n[..., None] > 0, 255, 0).astype(np.uint8)
            a[..., 3] = (np.clip(np.abs(n) - 1.0, 0, 2) * 13).astype(np.uint8)
            for _ in range(3):                                           # the odd hair / scratch
                x, y, l = int(rng.integers(4, gw - 4)), int(rng.integers(0, gh)), int(rng.integers(20, 120))
                a[y:y + l, x, :3] = 230 if rng.random() < 0.5 else 0
                a[y:y + l, x, 3] = 50
            st[key] = Image.fromarray(a).resize((gw * 2, gh * 2), Image.NEAREST)
        return st[key]

    def leak():
        """A soft warm light leak (orange-red, fading out from its middle)."""
        if "leak" not in st:
            lw, lh = 460, 980
            yy, xx = np.mgrid[0:lh, 0:lw].astype(np.float32)
            q = np.clip(1 - np.hypot((xx - lw / 2) / (lw / 2), (yy - lh / 2) / (lh / 2)), 0, 1) ** 1.6
            a = np.zeros((lh, lw, 4), np.uint8)
            a[..., 0] = 255
            a[..., 1] = (90 + 90 * q).astype(np.uint8)
            a[..., 2] = 30
            a[..., 3] = (q * 255).astype(np.uint8)
            st["leak"] = Image.fromarray(a)
        return st["leak"]

    MOTES = [(float(m[0]), float(m[1]), float(m[2]), float(m[3]), int(m[4]))
             for m in np.random.default_rng(9).uniform((0, 0, 0, 0, 3), (W, H, 6.283, 1, 7), (46, 5))]

    def mote_im(r):
        key = ("mote", r)
        if key not in st:
            m = Image.new("RGBA", (r * 4, r * 4), (255, 226, 180, 0))
            a = Image.new("L", (r * 4, r * 4), 0)
            ImageDraw.Draw(a).ellipse([r, r, r * 3, r * 3], fill=210)
            m.putalpha(a.filter(ImageFilter.GaussianBlur(r * 0.6)))
            st[key] = m
        return st[key]

    def booth_air(base, tt):
        """Dust motes floating up through the beam, a light leak breathing along one film edge, and film grain on
        the strips either side of the picture. Drawn under the core, so never over the photo window."""
        for x0, y0, ph, sp, r in MOTES:
            y = (y0 - tt * (10 + 14 * sp)) % (H + 40) - 20
            x = x0 + 26 * math.sin(tt * (0.4 + 0.5 * sp) + ph)
            if X0 - 4 < x < X0 + fw + 4 and GATE_Y - 60 < y < GATE_Y + fh + 60:
                continue                                                  # not over the gate
            al = 0.55 + 0.45 * math.sin(tt * 1.3 + ph * 3)
            if al > 0.15:
                ctx.paste(base, mote_im(r), x, y, alpha=al, nearest=False)
        per = 5.2                                                         # light leak: one every few seconds
        k, u = int(tt // per), (tt % per) / per
        a = math.sin(math.pi * u) ** 2
        if a > 0.02:
            lk = leak()
            lk = Image.fromarray((np.asarray(lk) * np.array([1, 1, 1, 0.34 * a])).astype(np.uint8))
            side = k % 2
            lx = (SX0 - lk.width * 0.55) if side == 0 else (SX1 - lk.width * 0.45)
            ly = 520 + ((k * 397) % 520) + 140 * (u - 0.5)
            base.alpha_composite(lk, (int(lx), int(ly)))
        g = grain(int(tt * 24) % 4)
        base.alpha_composite(g, (0, 0))
        base.alpha_composite(g.transpose(Image.FLIP_LEFT_RIGHT), (W - g.width, 0))

    def leader(j):
        """An unexposed frame for a link still to come: dark amber with a countdown-style mark (no photo)."""
        key = ("lead", j)
        if key not in st:
            im = Image.new("RGBA", (fw, fh), (44, 30, 20, 255))
            d = ImageDraw.Draw(im)
            cx, cy = fw / 2, fh / 2
            d.ellipse([cx - 230, cy - 230, cx + 230, cy + 230], outline=(92, 66, 44), width=6)
            d.ellipse([cx - 190, cy - 190, cx + 190, cy + 190], outline=(80, 56, 38), width=3)
            d.line([cx, cy - 300, cx, cy + 300], fill=(80, 56, 38), width=3)
            d.line([cx - 300, cy, cx + 300, cy], fill=(80, 56, 38), width=3)
            d.text((cx, cy), "?", font=font("Black", 220), fill=(110, 80, 52), anchor="mm")
            st[key] = im
        return st[key]

    def still(j):
        """The frame link j left in the gate: its photo framed on both faces, B named (it was revealed)."""
        key = ("still", j)
        if key not in st:
            im = ctx.view(j, *ctx.both_cam(j, fw, fh, 0.8), fw, fh).convert("RGBA")
            a = np.asarray(im).astype(np.float32)
            a[..., :3] = a[..., :3] * np.array([1.0, 0.9, 0.74]) * 0.62     # an amber, darker played frame
            st[key] = Image.fromarray(a.clip(0, 255).astype(np.uint8))
        return st[key]

    def strip(base, pos, i, tt):
        """The film strip itself: base, sprocket holes and edge print scroll with pos (frames advanced)."""
        d = ImageDraw.Draw(base)
        d.rectangle([SX0, 0, SX1, H], fill=FILM)
        d.line([SX0, 0, SX0, H], fill=(70, 52, 36), width=3)
        d.line([SX1, 0, SX1, H], fill=(70, 52, 36), width=3)
        hole_pitch = PITCH / 8
        off = (pos * PITCH) % hole_pitch
        hx = (SX0 + 16, SX1 - 16 - 34)
        k = -2
        while True:
            y = GATE_Y - 40 + k * hole_pitch - off
            if y > H + 40:
                break
            for x in hx:
                d.rounded_rectangle([x, y, x + 34, y + 26], 6, fill=(8, 6, 8))
                d.rounded_rectangle([x, y, x + 34, y + 4], 3, fill=(0, 0, 0))
            k += 1
        # edge print: frame numbers + year on the left margin, film stock on the right (rotated, like real film)
        for j in range(max(0, i - 2), min(ctx.n_links, i + 3)):
            y = GATE_Y + (j - pos) * PITCH
            if -PITCH < y < H + PITCH:
                lab = f"{j + 1:02d}A"
                yr = ctx.year_of(j)
                txt = f"▶ {lab}   {yr}" if yr and j <= i else f"▶ {lab}"
                t_im = Image.new("RGBA", (460, 30), (0, 0, 0, 0))
                ImageDraw.Draw(t_im).text((0, 15), txt, font=font("Black", 22), fill=EDGE, anchor="lm")
                t_im = t_im.rotate(90, expand=True)
                base.alpha_composite(t_im, (int(SX0 + 52), int(y + fh - 470)))
                r_im = Image.new("RGBA", (420, 26), (0, 0, 0, 0))
                ImageDraw.Draw(r_im).text((0, 13), "BUH SAFETY FILM  ·  35", font=font("Bold", 18),
                                          fill=(200, 128, 40), anchor="lm")
                r_im = r_im.rotate(-90, expand=True)
                base.alpha_composite(r_im, (int(SX1 - 76), int(y + 40)))

    def frame_lines(base, y0, w, h):
        """Black frame lines above and below the picture in the gate + a thin keyline."""
        d = ImageDraw.Draw(base)
        d.rectangle([X0 - 6, y0 - LINE, X0 + w + 6, y0 - 1], fill=(6, 4, 4))
        d.rectangle([X0 - 6, y0 + h, X0 + w + 6, y0 + h + LINE], fill=(6, 4, 4))

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        # the strip pulls one frame down at the start of each link (a quick, eased pull with a tiny overshoot)
        u = min(1.0, local / PULL)
        pos = (i - 1 + ctx.ease(u) + 0.03 * math.sin(math.pi * u)) if i else 0.0
        if u >= 1:
            pos = float(i)
        base = backdrop(tt)
        strip(base, pos, i, tt)
        # neighbours: the played frame above (still), unexposed leader below
        for j in range(max(0, i - 1), min(ctx.n_links, i + 2)):
            if j == i:
                continue
            y = GATE_Y + (j - pos) * PITCH
            if -fh < y < H:
                im = still(j) if j < i else leader(j)
                base.alpha_composite(im, (X0, int(y)))
                frame_lines(base, int(y), fw, fh)
        booth_air(base, tt)
        gy = GATE_Y + (i - pos) * PITCH

        def mount(b, x, y, w, h):
            frame_lines(b, y, w, h)
            ImageDraw.Draw(b).rectangle([x - 3, y - 3, x + w + 2, y + h + 2], outline=(250, 214, 150), width=3)

        c = ctx.draw_core(base, seg, tt, X0, gy, SC, frame=mount)
        # projector flicker: the light breathes on the film around the picture (never darkens the faces)
        fl = 0.5 + 0.5 * math.sin(tt * 31) * math.sin(tt * 7.3)
        if fl > 0.6:
            ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            od = ImageDraw.Draw(ov)
            a = int(36 * (fl - 0.6) / 0.4)
            od.rectangle([0, 0, SX0 - 1, H], fill=(0, 0, 0, a))
            od.rectangle([SX1 + 1, 0, W, H], fill=(0, 0, 0, a))
            base.alpha_composite(ov)
        return ctx.core_finish(base, seg, tt, c, connected_y=int(gy + fh - 120))

    return frame
