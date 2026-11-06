"""Pixel-arcade style renderer (the channel look).

Photos and paintings in a white card with a hard shadow, on a scrolling two-tone pixel checkerboard;
chunky pixel name tags and captions (Press Start 2P); Pin (pixel pushpin in a detective hat)
thrown along the red evidence string; "?" blocks that fill in as people are revealed;
pixelate-dissolve cuts; 8-bit SFX; CONNECTED! on the final reveal.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import assets, audio, pixelpin
from .common import FONTS
from .render import (END_HOLD, PAD, REVEAL, THROW, Ctx, both_camera, camera_for, ease, ease_out_back, face_crop,
                     grade, lerp_cam, seg_reveal, shake, short_name, to_screen, warp_photo)

W, H, FPS = 1080, 1920, 30
INK = (24, 14, 32)
WHITE = (255, 255, 255)
YELLOW = (255, 214, 10)
CYAN = (40, 220, 255)
RED = (236, 44, 64)
PURPLE_Q = (60, 50, 110)
# per-photo background tones (light, dark)
TONES = [((255, 46, 99), (226, 26, 82)), ((40, 170, 255), (20, 130, 225)), ((255, 170, 0), (236, 140, 0)),
         ((140, 70, 255), (112, 46, 220)), ((0, 205, 130), (0, 170, 105)), ((255, 90, 200), (225, 60, 170))]
FF, RW = 0.0, 0.0                     # hook: no flash-forward/rewind; the two faces are on screen from frame 0
# Shorts UI safe zone: the feed covers the top ~140 px (search/menu), the right ~140 px from mid-height down
# (like/comment/share buttons) and the bottom ~470 px (title, channel, sound). Everything that matters stays inside.
SAFE = (60, 150, 930, 1440)           # x0, y0, x1, y1
CX = (SAFE[0] + SAFE[2]) / 2          # content centre (left of the button column)
CARD = (50, 430, 930, 1380)           # photo window: x0, y0, x1, y1
BAND = 270                            # top title band height
BAR_Y = 352                           # chain bar (faces) centre line
CAP_Y = 1300                          # captions


def pf(size):  # pixel font
    return ImageFont.truetype(str(FONTS / "PressStart2P-Regular.ttf"), size)


def sf(size):
    return ImageFont.truetype(str(FONTS / "Silkscreen-Bold.ttf"), size)


def ascii_fold(text: str) -> str:
    """The pixel font has no accents: Raúl -> Raul, Göring -> Goring."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def pixel_text(text, size, fill=WHITE, outline=INK, shadow=True, font=None) -> Image.Image:
    text = ascii_fold(text)
    f = font or pf(size)
    l, t, r, b = f.getbbox(text)
    o = max(3, size // 9)
    im = Image.new("RGBA", (r - l + 4 * o + 8, b - t + 4 * o + 8), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x, y = 2 * o - l, 2 * o - t
    if shadow:
        for dx in range(-o, o + 1, o):
            for dy in range(-o, o + 1, o):
                d.text((x + dx + o, y + dy + o), text, font=f, fill=INK)
    for dx in range(-o, o + 1, o):
        for dy in range(-o, o + 1, o):
            d.text((x + dx, y + dy), text, font=f, fill=outline)
    d.text((x, y), text, font=f, fill=fill)
    return im


def pixel_tag(text, bg, size=26) -> Image.Image:
    text = ascii_fold(text)
    f = pf(size)
    l, t, r, b = f.getbbox(text)
    w, h = r - l + 36, b - t + 30
    im = Image.new("RGBA", (w + 10, h + 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([8, 8, w + 8, h + 8], fill=INK)
    d.rectangle([0, 0, w, h], fill=bg, outline=INK, width=6)
    d.text((18 - l, 15 - t), text, font=f, fill=INK)
    return im


def paste(base, im, cx, cy, scale=1.0, alpha=1.0, nearest=True):
    if scale <= 0.01 or alpha <= 0.01:
        return
    if abs(scale - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                       Image.NEAREST if nearest else Image.BICUBIC)
    if alpha < 1:
        im = im.copy()
        im.putalpha(im.split()[-1].point(lambda v: int(v * alpha)))
    base.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))


def checker(t: float, tone, tile=60) -> Image.Image:
    off = int(t * 40) % (2 * tile)
    yy, xx = np.mgrid[0:H, 0:W]
    m = (((xx + off) // tile + (yy + off) // tile) % 2).astype(bool)
    a = np.empty((H, W, 4), np.uint8)
    a[..., 3] = 255
    a[m, :3] = tone[0]
    a[~m, :3] = tone[1]
    return Image.fromarray(a)


def mosaic(im: Image.Image, block: int) -> Image.Image:
    if block <= 1:
        return im
    small = im.resize((max(1, im.width // block), max(1, im.height // block)), Image.BILINEAR)
    return small.resize(im.size, Image.NEAREST)


def pin_sprite(expr, angle=0.0, squash=0.0, look=0, scale=6):
    return pixelpin.draw(expr, scale, angle, squash, look, arms=True, hat=True)


def paste_pin(base, spr, x, tip_y):
    # needle tip is ~41/44 down the sprite
    base.alpha_composite(spr, (int(x - spr.width / 2), int(tip_y - spr.height * 0.93)))


class PixelCaptions:
    def __init__(self, words, t0):
        self.groups, cur = [], []
        for w, a, b in words:
            cur.append((w, a + t0, b + t0))
            if len(cur) >= 3 or w.endswith((",", ".", "?", "!", ":")) or len(" ".join(x[0] for x in cur)) > 11:
                self.groups.append(cur)
                cur = []
        if cur:
            self.groups.append(cur)

    def draw(self, base, t, y=CAP_Y):
        for g in self.groups:
            if g[0][1] - 0.02 <= t < g[-1][2] + 0.12:
                parts = [w.upper().strip(",.:;") or w.upper() for w, _, _ in g]
                size = 54
                while size > 30 and pf(size).getlength(" ".join(parts)) > SAFE[2] - SAFE[0] - 20:
                    size -= 4
                f = pf(size)
                total = f.getlength(" ".join(parts))
                x = CX - total / 2
                cur = max([j for j, (_, a, _) in enumerate(g) if a <= t] or [0])
                pop = ease_out_back((t - g[0][1]) / 0.1)
                for j, p in enumerate(parts):
                    im = pixel_text(p, size, fill=YELLOW if j == cur else WHITE)
                    wd = f.getlength(p)
                    paste(base, im, x + wd / 2, y, scale=0.85 + 0.15 * pop)
                    x += wd + f.getlength(" ")
                return


def render(slug: str, log=print) -> Path:
    C = Ctx(slug)
    spec = C.spec
    out_dir = C.dir / "out"
    out_dir.mkdir(exist_ok=True)
    voice = spec.get("voice")
    n_links = len(spec["links"])

    # ---------- audio + timeline
    segs, voice_clips, sfx = [], [], []
    t = 0.0
    n_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    hook = spec.get("hook") or f"How is {short_name(spec['from'])} connected to {short_name(spec['to'])}?"
    audio.prefetch([hook] + [l["line"] for l in spec["links"]], voice, log)   # Gemini: one request per episode
    a, words = audio.say(hook, voice)
    HOOK_VO = 0.0    # the question starts on the very first frame
    d = max(len(a) / audio.SR + 0.3 + HOOK_VO, FF + RW + 1.3)
    segs.append({"kind": "hook", "t0": t, "t1": t + d, "cap": PixelCaptions(words, t + HOOK_VO)})
    voice_clips.append((t + HOOK_VO, a))
    sfx += [(0.0, assets.sfx("impact", 0.4)),                              # A slams in on frame 0
            (0.18, assets.sfx("bass_hit", 0.4)),                            # B lands
            (0.5, assets.sfx("px_coin", 0.3))]
    t += d
    for i, link in enumerate(spec["links"]):
        a, words = audio.say(link["line"], voice)
        last = i == n_links - 1
        speech = len(a) / audio.SR + 0.08
        # last photo: cut hard right after the final word / the CONNECTED! slam, for a seamless loop
        d = max(speech + 0.05, seg_reveal(speech) + THROW + 0.5) if last else speech + PAD
        segs.append({"kind": "link", "i": i, "t0": t, "t1": t + d, "cap": PixelCaptions(words, t + 0.08),
                     "speech": speech})
        voice_clips.append((t + 0.08, a))
        reveal = t + (speech * 0.9 - THROW if link.get("lineage") else seg_reveal(speech))
        sfx += [(t, assets.sfx("px_cut", 0.45)), (t, assets.sfx("shutter", 0.35)),
                (reveal, assets.sfx("px_jump", 0.35)),
                (reveal + THROW, assets.sfx("px_coin", 0.45))]
        if last:
            sfx += [(max(0, reveal - 1.6), assets.sfx("riser", 0.3, max_len=1.6)),
                    (reveal + THROW, assets.sfx("px_bomb", 0.5)),
                    (reveal + THROW + 0.35, assets.sfx("px_pop", 0.4))]
        t += d
    total = t
    track = assets.pick_track(slug, spec.get("music"))
    wav = audio.mix(total, voice_clips, sfx, seed=0, out=out_dir / "mix.wav", music=assets.music(track, total))
    log(f"music: {track['title']} (Mixkit); timeline {total:.1f}s")

    # ---------- cameras (photo window is the card, 1000x1300)
    cams = []
    for i, l in enumerate(C.fj["links"]):
        img = C.imgs[i]
        ih, iw = img.shape[:2]
        ba = l["a"]["box"] or [iw * 0.25, ih * 0.2, iw * 0.15, iw * 0.15]
        bb = l["b"]["box"] or [iw * 0.6, ih * 0.2, iw * 0.15, iw * 0.15]
        cap = 3.2
        ca, cb, both = camera_for(ba, img.shape, 0.30), camera_for(bb, img.shape, 0.30), both_camera(ba, bb, img.shape)
        cams.append({"a": (ca[0], ca[1], min(ca[2], cap)), "b": (cb[0], cb[1], min(cb[2], cap)),
                     "both": (both[0], both[1], min(both[2], cap)), "ba": ba, "bb": bb})

    # blurred, darkened copy of each photo: fills the card where the photo doesn't reach
    card_bg = []
    for img in C.imgs:
        cw, ch = CARD[2] - CARD[0], CARD[3] - CARD[1]
        ih, iw = img.shape[:2]
        sc = max(cw / iw, ch / ih)
        b = cv2.resize(img, (int(iw * sc) + 2, int(ih * sc) + 2))
        y0, x0 = (b.shape[0] - ch) // 2, (b.shape[1] - cw) // 2
        b = cv2.GaussianBlur(b[y0:y0 + ch, x0:x0 + cw], (0, 0), 25)
        card_bg.append(Image.fromarray(cv2.cvtColor((b * 0.5).astype(np.uint8), cv2.COLOR_BGR2RGB)).convert("RGBA"))

    # ---------- static bits
    def short_label(n):
        lab = C.labels[n].upper()
        last = lab.split()[-1]
        return lab if (len(lab) <= 12 or len(last) <= 3) else last
    title = f"{short_label(spec['from'])} > {short_label(spec['to'])}"
    tsize = 46
    while pf(tsize).getlength(title) > SAFE[2] - SAFE[0] and tsize > 24:
        tsize -= 2
    title_im = pixel_text(title, tsize, shadow=False)
    def _flat(im):
        bg = Image.new("RGBA", im.size, (40, 30, 60, 255)); bg.alpha_composite(im.convert("RGBA")); return bg
    sharp = {n: _flat(C.thumbs[n].resize((120, 120), Image.BICUBIC)) for n in C.people}
    blurred = {n: sharp[n].filter(ImageFilter.GaussianBlur(9)) for n in C.people}

    def band(base, i=None):
        dr = ImageDraw.Draw(base)
        dr.rectangle([0, 0, W, BAND], fill=INK)
        paste(base, title_im, CX, 165)
        if i is not None:
            paste(base, pixel_text(f"LINK {i + 1}/{n_links}", 22, fill=YELLOW, shadow=False), CX, 232)

    def blocks(base, upto, active, tt, sharpen=1.0):
        """Chain bar: revealed faces sharp, upcoming faces blurred; the newest one sharpens in (sharpen 0..1)."""
        n = len(C.people)
        span = SAFE[2] - SAFE[0]
        size = min(92, int(span / n) - 16)
        gap = span / n
        y = BAR_Y
        dr = ImageDraw.Draw(base)
        for k, nm in enumerate(C.people):
            cx = SAFE[0] + gap * (k + 0.5)
            bob = -6 * abs(math.sin(tt * 6)) if k in active else 0
            x0, y0 = cx - size / 2, y - size / 2 + bob
            dr.rectangle([x0 + 8, y0 + 8, x0 + size + 8, y0 + size + 8], fill=INK)
            dr.rectangle([x0, y0, x0 + size, y0 + size], fill=YELLOW if k in active else WHITE, outline=INK, width=6)
            inner = size - 12
            if k < upto or (k == upto and sharpen >= 1):
                th = sharp[nm].resize((inner, inner), Image.BICUBIC)
            elif k == upto:
                th = Image.blend(blurred[nm].resize((inner, inner), Image.BICUBIC),
                                 sharp[nm].resize((inner, inner), Image.BICUBIC), sharpen)
            else:
                th = blurred[nm].resize((inner, inner), Image.BICUBIC)
            base.alpha_composite(th.convert("RGBA"), (int(x0 + 6), int(y0 + 6)))
            dr.rectangle([x0, y0, x0 + size, y0 + size], outline=INK, width=6)

    def credit(base, i):
        cr = C.fj["credits"][i]
        who = (cr.get("artist") or "").strip()
        if len(who) > 26 or "(" in who or "http" in who:
            who = "Wikimedia Commons" if cr.get("file") else ""
        txt = f"PHOTO: {who + ' / ' if who else ''}{cr.get('license', '')}".upper()
        ImageDraw.Draw(base).text((CARD[2], CARD[3] + 32), txt, font=sf(18), fill=(255, 255, 255), anchor="rm",
                                  stroke_width=2, stroke_fill=INK)

    # ---------- frames
    # ---- lineage links ("line of popes"): one portrait card per holder, scrolled fast
    lineage_cards = {}

    def lineage_card(i, j):
        key = (i, j)
        if key not in lineage_cards:
            it = C.fj["links"][i]["lineage"][j]
            img = grade(cv2.imread(str(C.work / it["path"])))
            ih, iw = img.shape[:2]
            x, y, w, h = it["box"]
            cw = min(iw, max(w * 2.6, iw * 0.5)); ch = min(ih, cw * 1.25)
            cx, cy = x + w / 2, y + h * 0.9
            x0 = int(max(0, min(iw - cw, cx - cw / 2))); y0 = int(max(0, min(ih - ch, cy - ch * 0.42)))
            crop = cv2.cvtColor(img[y0:int(y0 + ch), x0:int(x0 + cw)], cv2.COLOR_BGR2RGB)
            ph = Image.fromarray(crop).resize((560, 700), Image.BICUBIC).convert("RGBA")
            card = Image.new("RGBA", (600, 800), (0, 0, 0, 0))
            cd = ImageDraw.Draw(card)
            cd.rectangle([16, 16, 600, 760], fill=INK)
            cd.rectangle([0, 0, 584, 744], fill=WHITE, outline=INK, width=6)
            card.alpha_composite(ph, (12, 12))
            tag = pixel_tag(it["name"].upper(), YELLOW if j == len(C.fj["links"][i]["lineage"]) - 1 else CYAN,
                            24 if len(it["name"]) < 18 else 18)
            card.alpha_composite(tag, (int(292 - tag.width / 2), 640))
            if it.get("label"):
                lab = pixel_text(it["label"].upper(), 18, fill=WHITE)
                card.alpha_composite(lab, (int(292 - lab.width / 2), 24))
            lineage_cards[key] = card
        return lineage_cards[key]

    def frame_lineage(seg, tt):
        i = seg["i"]
        lin = C.fj["links"][i]["lineage"]
        n = len(lin)
        last = i == n_links - 1
        k = min(1.0, (tt - seg["t0"]) / seg["speech"])
        END = 0.9                                     # the last holder settles at 90% of the line
        p = ease(min(1.0, max(0.0, (k - 0.06) / (END - 0.06)))) * (n - 1)
        land = (k - END) * seg["speech"]
        base = checker(tt * 2, TONES[i % len(TONES)])
        STEP = 860
        for j in range(n):
            y = 960 + (j - p) * STEP
            if -500 < y < H + 500:
                paste(base, lineage_card(i, j), CX - 20, y, scale=0.9)
        # Buh rides along the right edge
        bob = abs(math.sin(tt * 9)) * 30
        ex = "shock" if 0.2 < k < END else ("smug" if land > 0.3 else "happy")
        paste_pin(base, pin_sprite(ex, 15 * math.sin(tt * 6) if 0.1 < k < END else 0, 0, -1), SAFE[2] - 40, 720 - bob)
        dr = ImageDraw.Draw(base)
        dr.rectangle([0, 0, W, BAND], fill=INK)
        paste(base, title_im, CX, 165)
        lab = C.fj["links"][i].get("label", "LINE OF SUCCESSION").upper()
        paste(base, pixel_text(f"{lab}  {min(n, int(p) + 1)}/{n}", 22, fill=YELLOW, shadow=False), CX, 232)
        if last and land > 0.35:
            st = min(1.0, (land - 0.35) / 0.12)
            paste(base, pixel_text("CONNECTED!", 72, fill=RED), CX, 1020, scale=2.0 - ease(st))
        upto = i + (1 if land >= 0 else 0)
        blocks(base, upto, {upto} if land >= 0 else {i}, tt, sharpen=min(1.0, land / 0.35) if land >= 0 else 1.0)
        seg["cap"].draw(base, tt)
        incut = tt - seg["t0"]
        if incut < 0.10:
            base.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(220 * (1 - incut / 0.10)))))
        return base

    def frame_link(seg, tt):
        i = seg["i"]
        if C.fj["links"][i].get("lineage"):
            return frame_lineage(seg, tt)
        img, cm = C.imgs[i], cams[i]
        last = i == n_links - 1
        k = min(1.0, (tt - seg["t0"]) / seg["speech"])
        if k < 0.10:
            cam = cm["a"]
        elif k < REVEAL:
            cam = lerp_cam(cm["a"], cm["both"], ease((k - 0.10) / (REVEAL - 0.10)))
        elif k < 0.62:
            cam = cm["both"]
        else:
            cam = lerp_cam(cm["both"], cm["b"], ease((k - 0.62) / 0.30))
        since = (k - REVEAL) * seg["speech"]
        land = since - THROW
        incut = tt - seg["t0"]
        punch = 0.10 * math.exp(-land * 8) if land >= 0 else 0.0
        cam = (cam[0], cam[1], cam[2] * (1 + 0.02 * math.sin(tt * 1.3) + punch + 0.15 * math.exp(-incut * 10)))
        sx, sy = shake(land, 26 if last else 12, 0.45 if last else 0.3)
        base = checker(tt, TONES[i % len(TONES)])
        # photo card
        ph, M = warp_photo(img, cam, center_y=(CARD[1] + CARD[3]) / 2)
        cw, ch = CARD[2] - CARD[0], CARD[3] - CARD[1]
        crop = ph.crop((CARD[0], CARD[1], CARD[2], CARD[3]))
        blk = int(46 * (1 - incut / 0.2)) if incut < 0.2 else 0
        if blk > 1:
            crop = mosaic(crop, blk)
        x0, y0 = int(CARD[0] + sx), int(CARD[1] + sy)
        dr = ImageDraw.Draw(base)
        dr.rectangle([x0 - 14 + 16, y0 - 14 + 16, x0 + cw + 14 + 16, y0 + ch + 14 + 16], fill=INK)
        dr.rectangle([x0 - 14, y0 - 14, x0 + cw + 14, y0 + ch + 14], fill=WHITE)
        flat = card_bg[i].copy()          # blurred photo fills the card when zoomed out
        flat.alpha_composite(crop)
        base.alpha_composite(flat, (x0, y0))
        M = M.copy(); M[0, 2] += sx; M[1, 2] += sy

        def scr(x, y):
            px, py = to_screen(M, x, y)
            return max(CARD[0] + 30, min(CARD[2] - 30, px)), max(CARD[1] + 60, min(CARD[3] - 40, py))
        pa = scr(cm["ba"][0] + cm["ba"][2] / 2, cm["ba"][1] - cm["ba"][3] * 0.08)
        pb = scr(cm["bb"][0] + cm["bb"][2] / 2, cm["bb"][1] - cm["bb"][3] * 0.08)
        # string
        start = (pa[0] + 60, pa[1] - 30)
        if since >= 0:
            if since < THROW:
                x, y, _ = pixelpin_path(start, pb, since / THROW)
                pe = (x, y)
            else:
                pe = pb
            dr.line([pa, pe], fill=INK, width=14)
            dr.line([pa, pe], fill=RED, width=8)
            dr.rectangle([pa[0] - 10, pa[1] - 10, pa[0] + 10, pa[1] + 10], fill=RED, outline=INK, width=4)
        # name tags
        names = (spec["links"][i]["a"], spec["links"][i]["b"])
        for role, nm, appear, col in (("ba", names[0], 0.0, CYAN), ("bb", names[1], REVEAL + THROW / seg["speech"], YELLOW)):
            bx = cm[role]
            fx, fy = to_screen(M, bx[0] + bx[2] / 2, bx[1] + bx[3] / 2)
            if not (CARD[0] < fx < CARD[2] and CARD[1] < fy < CARD[3]):
                continue
            if k >= appear:
                pop = ease_out_back((k - appear) * seg["speech"] / 0.12)
                tag = pixel_tag(C.labels[nm].upper(), col, 26 if len(C.labels[nm]) < 15 else 22)
                tx, ty = to_screen(M, bx[0] + bx[2] / 2, bx[1] + bx[3])   # under the chin
                tx = max(SAFE[0] + tag.width / 2, min(SAFE[2] - tag.width / 2, tx))
                ty = max(CARD[1] + 60, min(CARD[3] - 230, ty + 34))
                paste(base, tag, tx, ty, scale=pop)
        # Pin
        if since < 0:
            bob = int(8 * abs(math.sin(tt * 7)))
            spr = pin_sprite("happy", 0, 0, 1 if pb[0] > pa[0] else -1)
            paste_pin(base, spr, start[0], start[1] - bob)
        elif since < THROW:
            x, y, ang = pixelpin_path(start, pb, since / THROW)
            paste_pin(base, pin_sprite("shock", ang, -0.4), x, y)
        else:
            sq = 0.8 * math.exp(-land * 10) * math.cos(land * 30)
            ex = "shock" if land < 0.5 else ("happy" if last else "smug")
            hop = int(-abs(math.sin(land * 9)) * 30 * math.exp(-land * 5))
            if last and land > 0.6:
                hop = int(-abs(math.sin(land * 7)) * 40)  # celebrates on the final reveal
            paste_pin(base, pin_sprite(ex, 0, sq), pb[0], pb[1] + hop)
        # CONNECTED!
        if last and land > 0.35:
            st = min(1.0, (land - 0.35) / 0.12)
            paste(base, pixel_text("CONNECTED!", 72, fill=RED), CX, 1020, scale=2.0 - ease(st))
        band(base, i)
        upto = i + (1 if land >= 0 else 0)
        blocks(base, upto, {upto} if land >= 0 else {i}, tt, sharpen=min(1.0, land / 0.35) if land >= 0 else 1.0)
        credit(base, i)
        seg["cap"].draw(base, tt)
        if incut < 0.10:
            base.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(220 * (1 - incut / 0.10)))))
        if 0 <= land < 0.07:
            base.alpha_composite(Image.new("RGBA", (W, H), TONES[i % len(TONES)][0] + (int(120 * (1 - land / 0.07)),)))
        return base

    class _NoCap:
        def draw(self, *a, **k):
            pass

    def fake_link_frame(i, local_k):
        """Frame of link i at progress local_k (0..1+), without its captions: used by the hook."""
        sg = next(x for x in segs if x.get("i") == i)
        fake = {**sg, "cap": _NoCap()}
        return frame_link(fake, sg["t0"] + sg["speech"] * local_k)

    _strip = {}

    def strip_frame(i):
        if i not in _strip:
            _strip[i] = fake_link_frame(i, 0.75)
        return _strip[i]

    def frame_hook(seg, tt):
        k = tt - seg["t0"]
        last = n_links - 1
        if k < FF:                                   # 1) flash-forward: the payoff first
            land_k = REVEAL + (THROW + 0.45 + k * 0.5) / segs[-1]["speech"]
            base = fake_link_frame(last, land_k)
            z = 1.45 - 0.45 * ease(k / FF)                    # continuous dolly-out from tight to full frame
            cw, ch = int(W / z), int(H / z)
            x0, y0 = (W - cw) // 2, int((H - ch) * 0.42)
            base = base.crop((x0, y0, x0 + cw, y0 + ch)).resize((W, H), Image.BILINEAR)
            if k < 0.12:   # soft fade-in from white, not a hard strobe
                base.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(150 * (1 - k / 0.12)))))
        elif k < FF + RW:                            # 2) rewind scrub back through the chain
            r = ease((k - FF) / RW)                          # film strip whips back through the chain
            pos = r * last                                   # 0 = last photo ... last = first photo
            j = min(last - 1, int(pos)) if last > 0 else 0
            frac = pos - j if last > 0 else 0.0
            a_im = strip_frame(last - j)
            b_im = strip_frame(max(0, last - j - 1))
            off = int(frac * H)
            base = Image.new("RGBA", (W, H), (0, 0, 0, 255))
            base.alpha_composite(a_im, (0, off))
            base.alpha_composite(b_im, (0, off - H))
            paste(base, pixel_text("<< REWIND", 56, fill=WHITE), 270, 360)
        else:                                        # 3) versus: A and B slam in, string snaps, count pops
            kk = k - FF - RW
            sx, sy = shake(kk, 12, 0.25)
            sx2, sy2 = shake(kk - 0.2, 12, 0.25)
            base = checker(tt * 3, ((90, 40, 200), (70, 26, 170)))   # faster scroll keeps the screen moving
            dr = ImageDraw.Draw(base)
            dr.rectangle([0, 0, W, BAND], fill=INK)
            paste(base, title_im, CX, 190)            # same "A > B" header as the links, no count
            centers = {}
            for nm, cy, delay, col, (dx, dy) in ((spec["from"], 590, 0.0, CYAN, (sx, sy)),
                                                 (spec["to"], 1060, 0.18, YELLOW, (sx2, sy2))):
                if kk < delay:
                    continue
                sc = 1.6 - 0.6 * ease(min(1, (kk - delay) / 0.12))     # slams down from big to size
                sc *= 1 + 0.06 * min(1.0, kk / 2.0)                       # then a slow push-in
                dy += 10 * math.sin(kk * 5 + (0 if nm == spec["from"] else 2))   # and keeps bobbing
                size = 360
                face = C.thumbs[nm].resize((size, size), Image.BICUBIC).convert("RGBA")
                card = Image.new("RGBA", (size + 44, size + 44), (0, 0, 0, 0))
                cd = ImageDraw.Draw(card)
                cd.rectangle([16, 16, size + 44, size + 44], fill=INK)
                cd.rectangle([0, 0, size + 28, size + 28], fill=WHITE, outline=INK, width=6)
                flat = Image.new("RGBA", (size, size), (30, 30, 30, 255)); flat.alpha_composite(face)
                card.alpha_composite(flat, (14, 14))
                paste(base, card, CX + dx, cy + dy, scale=sc, nearest=False)
                paste(base, pixel_tag(C.labels[nm].upper(), col, 30), CX + dx, cy + 222 + dy, scale=min(1, sc))
                centers[nm] = (CX + dx, cy + dy)
            if len(centers) == 2 and kk > 0.3:                          # the string snaps between them
                (ax, ay), (bx, by) = centers[spec["from"]], centers[spec["to"]]
                g = min(1.0, (kk - 0.3) / 0.12)
                p0, p1 = (ax + 210, ay + 110), (bx + 210, by - 190)
                pe = (p0[0] + (p1[0] - p0[0]) * g, p0[1] + (p1[1] - p0[1]) * g)
                dr = ImageDraw.Draw(base)
                dr.line([p0, pe], fill=INK, width=14); dr.line([p0, pe], fill=RED, width=8)
                if g >= 1:   # a glow travels along the string
                    q = (kk * 1.6) % 1.0
                    gx, gy = p0[0] + (p1[0] - p0[0]) * q, p0[1] + (p1[1] - p0[1]) * q
                    dr.rectangle([gx - 9, gy - 9, gx + 9, gy + 9], fill=YELLOW, outline=INK, width=3)
                hop = abs(math.sin(kk * 8)) * 40
                walk = 120 * math.sin(kk * 2.2)                           # Buh paces along the string
                paste_pin(base, pin_sprite("shock" if kk < 0.8 else ("smug" if int(kk * 2) % 2 else "angry"),
                                           0, 0, 1 if math.cos(kk * 2.2) > 0 else -1), min(SAFE[2] - 60, p1[0] + walk * 0.4), p1[1] - hop)
            if spec.get("requested_by"):
                req = f'{spec["requested_by"]}: "{short_name(spec["from"]).upper()} TO {short_name(spec["to"]).upper()}"'
                fsz = 24
                while pf(fsz).getlength(req) > SAFE[2] - SAFE[0] - 40 and fsz > 14:
                    fsz -= 2
                paste(base, pixel_tag(req, WHITE, fsz), CX, 340, scale=1)   # readable from frame 0
        seg["cap"].draw(base, tt, y=1370)
        return base

    draw = {"hook": frame_hook, "link": frame_link}

    def frame_at(tt):
        for j, sg in enumerate(segs):
            if sg["t0"] <= tt < sg["t1"] or j == len(segs) - 1:
                return draw[sg["kind"]](sg, tt)
        raise AssertionError

    out = out_dir / f"{slug}.mp4"
    nframes = int(total * FPS)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", str(wav), "-c:v", "libx264", "-preset", "medium", "-crf", "21",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest",
           "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(nframes):
        proc.stdin.write(frame_at(fi / FPS).convert("RGB").tobytes())
        if fi % 150 == 0:
            log(f"  frame {fi}/{nframes}")
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    (out_dir / "timeline.json").write_text(json.dumps({"duration": total, "style": "pixel", "segments": [
        {"kind": s["kind"], "t0": round(s["t0"], 2), "t1": round(s["t1"], 2)} for s in segs]}, indent=1))
    log(f"wrote {out}")
    return out


def pixelpin_path(p0, p1, k, height=280.0):
    x = p0[0] + (p1[0] - p0[0]) * k
    y = p0[1] + (p1[1] - p0[1]) * k - height * 4 * k * (1 - k)
    return x, y, 360 * k * (1 if p1[0] >= p0[0] else -1)
