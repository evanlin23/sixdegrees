"""Render an episode to a 1080x1920 Short.

Structure:  hook (two faces + "?")  ->  one segment per photo link (camera pans
from person A's face to B's face, name tags pop, narration with captions)
->  ends on the final reveal (no outro: Shorts live on retention, and a hard end loops into the hook).
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import assets, audio, mascot
from .common import FONTS, slugify
from .episode import load, people_order

W, H, FPS = 1080, 1920, 30
CYAN = (34, 211, 238)
YELLOW = (255, 214, 10)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
HUES = [(255, 46, 99), (0, 200, 255), (255, 170, 0), (140, 60, 255), (0, 230, 120), (255, 80, 200)]
RED_STRING = (235, 30, 50)
REVEAL = 0.32       # fraction of a line at which B is revealed


def seg_reveal(speech: float) -> float:
    return speech * REVEAL


THROW = 0.30        # seconds Pin flies from A to B
PAD = 0.12          # silence after each narration line
XFADE = 0.10        # cross-fade between segments (snappy)
END_HOLD = 0.7      # hold on the final reveal


def font(size, w="Black"):
    return ImageFont.truetype(str(FONTS / f"Inter-{w}.otf"), size)


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ease_out_back(x):
    x = min(1.0, max(0.0, x))
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


# ------------------------------------------------------------------ helpers

def face_crop(img: np.ndarray, box, size=256, pad=0.9) -> Image.Image:
    """Square crop around a face box, as a circular RGBA thumbnail."""
    h, w = img.shape[:2]
    if box is None:
        x, y, bw, bh = w * 0.3, h * 0.15, w * 0.4, w * 0.4
    else:
        x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2 - bh * 0.05
    r = max(bw, bh) * (0.5 + pad)
    M = np.float32([[size / (2 * r), 0, size / 2 - cx * size / (2 * r)],
                    [0, size / (2 * r), size / 2 - cy * size / (2 * r)]])
    crop = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, size - 1, size - 1], fill=255)
    pil.putalpha(mask)
    return pil


def ring(img: Image.Image, color, width=8) -> Image.Image:
    s = img.size[0]
    out = Image.new("RGBA", (s + 2 * width, s + 2 * width), (0, 0, 0, 0))
    d = ImageDraw.Draw(out)
    d.ellipse([0, 0, s + 2 * width - 1, s + 2 * width - 1], fill=color + (255,))
    out.alpha_composite(img, (width, width))
    return out


def pill(text, size=46, fg=BLACK, bg=CYAN, padx=22, pady=12) -> Image.Image:
    f = font(size)
    l, t, r, b = f.getbbox(text)
    w, h = r - l + 2 * padx, b - t + 2 * pady
    im = Image.new("RGBA", (w + 8, h + 8), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([4, 6, w + 4, h + 6], radius=h // 2, fill=(0, 0, 0, 120))  # shadow
    d.rounded_rectangle([2, 2, w + 2, h + 2], radius=h // 2, fill=bg + (255,))
    d.text((2 + padx - l, 2 + pady - t), text, font=f, fill=fg)
    return im


def text_img(text, size, fill=WHITE, stroke=8, w="Black") -> Image.Image:
    f = font(size, w)
    l, t, r, b = f.getbbox(text, stroke_width=stroke)
    im = Image.new("RGBA", (r - l + 4, b - t + 4), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((2 - l, 2 - t), text, font=f, fill=fill, stroke_width=stroke, stroke_fill=BLACK)
    return im


def paste(base: Image.Image, im: Image.Image, cx, cy, scale=1.0, alpha=1.0):
    if scale <= 0.01 or alpha <= 0.01:
        return
    if abs(scale - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BICUBIC)
    if alpha < 1:
        a = im.split()[-1].point(lambda v: int(v * alpha))
        im = im.copy()
        im.putalpha(a)
    base.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2)))


def short_name(name: str) -> str:
    import re
    return re.sub(r"\s*\(.*?\)", "", name)


# ------------------------------------------------------------------ segments

class Captions:
    """Word-timed captions: show 1-3 words at a time, current word highlighted."""

    def __init__(self, words, t0):
        self.groups = []
        cur = []
        for w, a, b in words:
            cur.append((w, a + t0, b + t0))
            if len(cur) >= 3 or w.endswith((",", ".", "?", "!", ":")) or len(" ".join(x[0] for x in cur)) > 13:
                self.groups.append(cur)
                cur = []
        if cur:
            self.groups.append(cur)

    def draw(self, base: Image.Image, t: float, y=1390):
        for g in self.groups:
            if g[0][1] - 0.02 <= t < g[-1][2] + 0.12:
                fs_ = 84
                parts = [w.upper().strip(",.:;") if w.upper().strip(",.:;") else w.upper() for w, _, _ in g]
                while True:  # shrink to fit inside the safe width
                    f = font(fs_)
                    widths = [f.getlength(p + " ") for p in parts]
                    total = sum(widths)
                    if total <= W - 120 or fs_ <= 56:
                        break
                    fs_ -= 4
                x = W / 2 - total / 2
                pop = ease_out_back((t - g[0][1]) / 0.12)
                cur = max([j for j, (_, a, _) in enumerate(g) if a <= t] or [0])
                for j, ((w, a, b), p, wd) in enumerate(zip(g, parts, widths)):
                    col = YELLOW if j == cur else WHITE
                    im = text_img(p, fs_, fill=col, stroke=10)
                    paste(base, im, x + (wd - f.getlength(" ")) / 2, y, scale=0.85 + 0.15 * pop)
                    x += wd
                return


class Ctx:
    def __init__(self, slug):
        self.dir, self.spec = load(slug)
        self.work = self.dir / "work"
        self.fj = json.loads((self.work / "faces.json").read_text())
        self.people = people_order(self.spec)
        self.imgs = [grade(cv2.imread(str(self.work / l["image"]))) for l in self.fj["links"]]
        # face thumbnails: prefer the person's Wikipedia portrait (big, frontal); fall back to a link photo
        self.thumbs = {}
        self.hero_src = {}   # name -> (image, face box): the hook crops big rectangular faces from these
        from .faces import detect
        for name in self.people:
            refs = self.fj["people"][name]["refs"]
            if not refs:
                continue
            im = cv2.imread(str(self.work / "refs" / refs[0]))
            if im is None:
                continue
            fs = detect(im)
            if fs:
                big = max(fs, key=lambda f: f["box"][2] * f["box"][3])
                if big["box"][3] >= 60:
                    self.thumbs[name] = face_crop(im, big["box"], 220, pad=0.55)
                    self.hero_src[name] = (im, big["box"])
        for i, l in enumerate(self.fj["links"]):
            if l.get("lineage"):   # lineage boxes belong to individual portraits, not to links[i]["image"]
                continue
            for role in ("a", "b"):
                name = self.spec["links"][i][role]
                bx = l[role]["box"]
                if name not in self.thumbs and bx:
                    self.thumbs[name] = face_crop(self.imgs[i], bx, 220)
                    self.hero_src[name] = (self.imgs[i], bx)
        for name in self.people:
            if name not in self.thumbs:  # no usable portrait and never boxed in a photo
                refs = self.fj["people"][name]["refs"]
                im = cv2.imread(str(self.work / "refs" / refs[0])) if refs else np.full((300, 300, 3), 60, np.uint8)
                self.thumbs[name] = face_crop(im, None, 220)
                self.hero_src[name] = (im, None)
        self.labels = {n: self.spec.get("people", {}).get(n, {}).get("label", short_name(n)) for n in self.people}


def grade(img: np.ndarray) -> np.ndarray:
    """High-dopamine grade: more saturation and contrast (B&W photos just get contrast)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    if hsv[..., 1].mean() > 18:  # colour photo
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.55, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def blurred_bg(img: np.ndarray, t: float, hue=None) -> Image.Image:
    h, w = img.shape[:2]
    s = max(W / w, H / h) * (1.15 + 0.03 * t)
    M = np.float32([[s, 0, W / 2 - w * s / 2], [0, s, H / 2 - h * s / 2]])
    small = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    small = cv2.GaussianBlur(cv2.resize(small, (W // 8, H // 8)), (0, 0), 6)
    small = cv2.resize(small, (W, H))
    small = (small * 0.45).astype(np.uint8)
    out = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).convert("RGBA")
    if hue is not None:  # vivid colour wash, drifting diagonally
        g = np.linspace(0, 1, H)[:, None] * 0.6 + np.linspace(0, 1, W)[None, :] * 0.4
        g = (np.sin((g + t * 0.25) * math.pi * 2) * 0.5 + 0.5)[..., None]
        c1 = np.array(hue, np.float32); c2 = c1[::-1] * 0.6 + 60
        col = (c1 * g + c2 * (1 - g)).astype(np.uint8)
        wash = Image.fromarray(np.dstack([col, np.full((H, W, 1), 150, np.uint8)]))
        out.alpha_composite(wash)
    return out


def rgb_split(im: Image.Image, px: int) -> Image.Image:
    a = np.array(im)
    a[..., 0] = np.roll(a[..., 0], px, axis=1)
    a[..., 2] = np.roll(a[..., 2], -px, axis=1)
    return Image.fromarray(a)


def shake(t_since: float, amp: float, dur: float = 0.35):
    if t_since < 0 or t_since > dur:
        return 0.0, 0.0
    a = amp * (1 - t_since / dur) ** 2
    return a * math.sin(t_since * 95), a * math.cos(t_since * 77)


def camera_for(box, img_shape, view_h_frac):
    """(cx, cy, scale) so that the box sits centred and is view_h_frac of the viewport height."""
    x, y, w, h = box
    vh = 1300  # viewport (photo area) height in px
    scale = vh * view_h_frac / max(h, 1)
    return (x + w / 2, y + h / 2 + h * 0.6, scale)  # aim a bit below the face (show shoulders)


def both_camera(a, b, img_shape):
    ih, iw = img_shape[:2]
    xs = [a[0], a[0] + a[2], b[0], b[0] + b[2]]
    ys = [a[1], a[1] + a[3], b[1], b[1] + b[3]]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    fw = max(a[2], b[2])
    x0 -= fw * 0.9; x1 += fw * 0.9; y0 -= fw * 0.8; y1 += fw * 1.6
    scale = min(W * 0.98 / (x1 - x0), 1250 / (y1 - y0))
    return ((x0 + x1) / 2, (y0 + y1) / 2, scale)


def lerp_cam(c0, c1, k):
    # interpolate scale geometrically for a natural zoom
    return (c0[0] + (c1[0] - c0[0]) * k, c0[1] + (c1[1] - c0[1]) * k, c0[2] * (c1[2] / c0[2]) ** k)


def warp_photo(img, cam, center_y=980, shift=(0.0, 0.0)):
    cx, cy, s = cam
    M = np.float32([[s, 0, W / 2 - cx * s + shift[0]], [0, s, center_y - cy * s + shift[1]]])
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    rgb = cv2.warpAffine(img, M, (W, H), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpAffine(np.full(img.shape[:2], 255, np.uint8), M, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    pil = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)).convert("RGBA")
    pil.putalpha(Image.fromarray(mask))
    return pil, M


def to_screen(M, x, y):
    return M[0, 0] * x + M[0, 2], M[1, 1] * y + M[1, 2]


# ------------------------------------------------------------------ main render

def render(slug: str, log=print) -> Path:
    C = Ctx(slug)
    spec = C.spec
    out_dir = C.dir / "out"
    out_dir.mkdir(exist_ok=True)
    voice = spec.get("voice")  # optional per-episode override of data/voice.json

    # ---------- audio + timeline
    segs = []  # dicts: kind, t0, t1, ...
    voice_clips, sfx = [], []
    t = 0.0
    hook = spec.get("hook") or f"How is {short_name(spec['from'])} connected to {short_name(spec['to'])}?"
    a, words = audio.say(hook, voice)
    HOOK_VO = 0.35  # the boom lands first, then the question (overlapping them masked the names)
    d = len(a) / audio.SR + 0.15 + HOOK_VO
    segs.append({"kind": "hook", "t0": t, "t1": t + d, "cap": Captions(words, t + HOOK_VO)})
    voice_clips.append((t + HOOK_VO, a))
    sfx.append((0.0, assets.sfx("boom", 0.7))); sfx.append((0.12, assets.sfx("bass_hit", 0.6)))
    t += d
    for i, link in enumerate(spec["links"]):
        a, words = audio.say(link["line"], voice)
        last = i == len(spec["links"]) - 1
        d = len(a) / audio.SR + PAD + 0.08 + (END_HOLD if last else 0)
        segs.append({"kind": "link", "i": i, "t0": t, "t1": t + d, "cap": Captions(words, t + 0.08),
                     "speech": len(a) / audio.SR + 0.08})
        voice_clips.append((t + 0.08, a))
        sfx.append((t, assets.sfx("shutter", 0.6)))                                  # snapshot on every new photo
        reveal = t + seg_reveal(len(a) / audio.SR + 0.08)
        sfx.append((reveal - 0.05, assets.sfx("swipe", 0.35)))                        # Pin gets thrown
        sfx.append((reveal + THROW, assets.sfx("bass_hit", 0.55)))                    # ...and lands: B revealed
        sfx.append((reveal + THROW, assets.sfx("pop", 0.3)))
        if last:
            sfx.append((max(0, reveal - 1.6), assets.sfx("riser", 0.4, max_len=1.6)))  # build-up
            sfx.append((reveal + THROW, assets.sfx("big_impact", 0.75)))               # final reveal hit
            sfx.append((reveal + THROW + 0.35, assets.sfx("impact", 0.5)))             # CONNECTED! stamp
        t += d
    n_links = len(spec["links"])
    total = t
    track = assets.pick_track(slug, spec.get("music"))
    wav = audio.mix(total, voice_clips, sfx, seed=sum(map(ord, slug)), out=out_dir / "mix.wav",
                    music=assets.music(track, total))
    log(f"music: {track['title']} (Mixkit)")
    log(f"timeline {total:.1f}s, {len(segs)} segments")

    # ---------- precompute per-link cameras
    cams = []
    for i, l in enumerate(C.fj["links"]):
        img = C.imgs[i]
        ba, bb = l["a"]["box"], l["b"]["box"]
        ih, iw = img.shape[:2]
        if ba is None:
            ba = [iw * 0.25, ih * 0.2, iw * 0.15, iw * 0.15]
        if bb is None:
            bb = [iw * 0.6, ih * 0.2, iw * 0.15, iw * 0.15]
        both = both_camera(ba, bb, img.shape)
        ca = camera_for(ba, img.shape, 0.30)
        cb = camera_for(bb, img.shape, 0.30)
        # never zoom so far that the source is upscaled more than ~3.2x
        cap_s = 3.2
        ca = (ca[0], ca[1], min(ca[2], cap_s)); cb = (cb[0], cb[1], min(cb[2], cap_s))
        both = (both[0], both[1], min(both[2], cap_s))
        cams.append({"a": ca, "b": cb, "both": both, "ba": ba, "bb": bb})

    # ---------- static overlays
    title_im = text_img(f"{C.labels[spec['from']].upper()}  →  {C.labels[spec['to']].upper()}", 50, stroke=6)
    if title_im.width > W - 80:
        title_im = title_im.resize((W - 80, int(title_im.height * (W - 80) / title_im.width)), Image.BICUBIC)

    def chain_bar(base, upto, active, alpha=1.0):
        """Row of face circles at the bottom: revealed up to index `upto`."""
        n = len(C.people)
        size = min(130, int((W - 80) / n * 0.78))
        gap = (W - 80) / n
        y = 1745
        d = ImageDraw.Draw(base)
        for k in range(n - 1):
            x0 = 40 + gap * (k + 0.5) + size / 2 + 4
            x1 = 40 + gap * (k + 1.5) - size / 2 - 4
            col = CYAN if k < upto else (90, 90, 90)
            d.line([(x0, y), (x1, y)], fill=col + (int(255 * alpha),), width=6)
        for k, name in enumerate(C.people):
            cx = 40 + gap * (k + 0.5)
            if k <= upto or k == n - 1:
                th = C.thumbs[name].resize((size, size), Image.BICUBIC)
                col = YELLOW if k in active else (CYAN if k <= upto else (120, 120, 120))
                im = ring(th, col, 6 if k in active else 4)
                if k > upto:  # destination shown dimmed
                    arr = np.array(im); arr[..., :3] = (arr[..., :3] * 0.45).astype(np.uint8); im = Image.fromarray(arr)
                paste(base, im, cx, y, alpha=alpha)
            else:
                q = Image.new("RGBA", (size, size), (0, 0, 0, 0))
                qd = ImageDraw.Draw(q)
                qd.ellipse([0, 0, size - 1, size - 1], fill=(40, 40, 40, 230), outline=(110, 110, 110, 255), width=4)
                f = font(int(size * 0.5))
                qd.text((size / 2, size / 2), "?", font=f, fill=(170, 170, 170), anchor="mm")
                paste(base, q, cx, y, alpha=alpha)

    def credit_line(base, i):
        cr = C.fj["credits"][i]
        who = (cr.get("artist") or "").replace("Unknown", "").strip()
        if len(who) > 28 or "(" in who or "http" in who:
            who = "Wikimedia Commons" if cr.get("file") else ""
        txt = f"Photo: {who + ' · ' if who else ''}{cr.get('license', '')}"
        f = font(24, "Bold")
        ImageDraw.Draw(base).text((W - 30, 1638), txt, font=f, fill=(200, 200, 200, 200), anchor="rm",
                                  stroke_width=2, stroke_fill=BLACK)

    # ---------- per-frame drawing
    def frame_link(seg, tt):
        i = seg["i"]
        img = C.imgs[i]
        cm = cams[i]
        k = min(1.0, (tt - seg["t0"]) / seg.get("speech", seg["t1"] - seg["t0"]))
        # camera: A close -> both -> B close
        if k < 0.10:
            cam = cm["a"]
        elif k < 0.32:
            cam = lerp_cam(cm["a"], cm["both"], ease((k - 0.10) / 0.22))
        elif k < 0.62:
            cam = cm["both"]
        else:
            cam = lerp_cam(cm["both"], cm["b"], ease((k - 0.62) / 0.30))
        # drift, punch-in on the cut, punch + shake when Pin lands on B
        since = (k - REVEAL) * seg.get("speech", 1.0)           # seconds since the throw started
        land = since - THROW                                      # seconds since Pin landed
        incut = tt - seg["t0"]
        punch = 0.10 * math.exp(-land * 8) if land >= 0 else 0.0
        cutin = 0.18 * math.exp(-incut * 10)
        cam = (cam[0], cam[1], cam[2] * (1 + 0.02 * math.sin(tt * 1.3) + punch + cutin))
        last = i == n_links - 1
        sh = shake(land, 30 if last else 12, 0.45 if last else 0.3)
        base = blurred_bg(img, tt - seg["t0"], HUES[i % len(HUES)])
        ph, M = warp_photo(img, cam, shift=sh)
        base.alpha_composite(ph)
        # name tags
        names = (spec["links"][i]["a"], spec["links"][i]["b"])
        # Pin (the mascot) + the red evidence string it drags from A to B
        pa = to_screen(M, cm["ba"][0] + cm["ba"][2] / 2, cm["ba"][1] - cm["ba"][3] * 0.08)
        pb = to_screen(M, cm["bb"][0] + cm["bb"][2] / 2, cm["bb"][1] - cm["bb"][3] * 0.08)
        pa = (max(60, min(W - 60, pa[0])), max(320, min(1560, pa[1])))
        pb = (max(60, min(W - 60, pb[0])), max(320, min(1560, pb[1])))
        mascot_draw = None
        if since < 0:          # waiting on A, bobbing, eyeing B
            bob = 10 * math.sin(tt * 7)
            mascot_draw = (pa[0] + 70, pa[1] - 40 + bob, "happy", 0.0, -8 + 6 * math.sin(tt * 5), 1 if pb[0] > pa[0] else -1)
        elif since < THROW:    # thrown along an arc, spinning, stretched
            x, y, ang = mascot.throw_path((pa[0] + 70, pa[1] - 40), pb, since / THROW, 300)
            mascot_draw = (x, y, "shock", -0.6, ang, 0)
        else:                  # landed on B: squash, shock, then smug
            sq = 0.9 * math.exp(-land * 10) * math.cos(land * 30)
            ex = "shock" if land < 0.5 else "smug"
            hop = -abs(math.sin(land * 9)) * 25 * math.exp(-land * 5)
            spin = 360 * min(1, (land - 0.5) / 0.4) if (last and land > 0.5) else 0
            mascot_draw = (pb[0], pb[1] + hop, ex, sq, spin, 0)
        if since >= 0:
            pe = pb if since >= THROW else mascot.throw_path((pa[0] + 70, pa[1] - 40), pb, since / THROW, 300)[:2]
            sd = ImageDraw.Draw(base)
            sd.line([pa, pe], fill=(0, 0, 0, 140), width=13)
            sd.line([pa, pe], fill=RED_STRING + (255,), width=8)
            sd.ellipse([pa[0] - 14, pa[1] - 14, pa[0] + 14, pa[1] + 14], fill=RED_STRING + (255,), outline=(255, 255, 255, 255), width=4)
        for role, nm, appear in (("ba", names[0], 0.0), ("bb", names[1], REVEAL)):
            bx = cm[role]
            sx, sy = to_screen(M, bx[0] + bx[2] / 2, bx[1])
            pop = ease_out_back((k - appear) / 0.06) if k >= appear else 0
            fy = to_screen(M, 0, bx[1] + bx[3] / 2)[1]
            if not (40 < sx < W - 40 and 150 < fy < 1650):  # face is off-screen: no floating tag
                pop = 0
            if pop > 0:
                p = pill(C.labels[nm], 50 if len(C.labels[nm]) < 16 else 42,
                         bg=CYAN if role == "ba" else YELLOW)
                sy = max(330, min(1500, sy - 40))
                sx = max(p.width / 2 + 10, min(W - p.width / 2 - 10, sx))
                paste(base, p, sx, sy, scale=pop)
        if mascot_draw:
            x, y, ex, sq, ang, look = mascot_draw
            spr = mascot.draw(ex, 0.62 * (1.35 if (last and since >= THROW + 0.5) else 1.0), ang, sq, look)
            paste(base, spr, x, y - spr.height * 0.42)   # needle tip sits on the point
        if last and land > 0.35:  # CONNECTED! stamp slams down
            st = min(1.0, (land - 0.35) / 0.12)
            stamp = text_img("CONNECTED!", 120, fill=RED_STRING, stroke=12).rotate(-8, expand=True, resample=Image.BICUBIC)
            paste(base, stamp, W / 2, 1180, scale=2.2 - 1.2 * ease(st), alpha=min(1, st * 1.5))
        # header + progress
        paste(base, title_im, W / 2, 150)
        step = pill(f"PHOTO {i + 1} / {n_links}", 34, fg=WHITE, bg=(30, 30, 30))
        paste(base, step, W / 2, 235)
        revealed = i + (1 if k >= REVEAL else 0)
        chain_bar(base, revealed, {i, i + 1} if k >= REVEAL else {i})
        credit_line(base, i)
        seg["cap"].draw(base, tt)
        flash = tt - seg["t0"]
        if flash < 0.14:  # camera-flash cut into every new photo
            base.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(235 * (1 - flash / 0.14)))))
        if flash < 0.12:
            base = rgb_split(base, int(28 * (1 - flash / 0.12)))
        if 0 <= land < 0.08:  # landing: RGB split + colour flash
            base = rgb_split(base, int(20 * (1 - land / 0.08)))
            base.alpha_composite(Image.new("RGBA", (W, H), HUES[i % len(HUES)] + (int(110 * (1 - land / 0.08)),)))
        return base

    def frame_hook(seg, tt):
        k = tt - seg["t0"]
        base = Image.new("RGBA", (W, H), (20, 8, 40, 255))
        # vivid glow
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.ellipse([W / 2 - 650, H / 2 - 650, W / 2 + 650, H / 2 + 650], fill=HUES[3] + (170,))
        gd.ellipse([W / 2 - 300, 200, W / 2 + 500, 900], fill=HUES[0] + (140,))
        base.alpha_composite(glow.filter(ImageFilter.GaussianBlur(120)))
        a_name, b_name = spec["from"], spec["to"]
        for nm, cy, delay, col in ((a_name, 560, 0.0, CYAN), (b_name, 1300, 0.12, YELLOW)):
            sc = ease_out_back((k - delay) / 0.2)
            th = ring(C.thumbs[nm].resize((400, 400), Image.BICUBIC), col, 10)
            paste(base, th, W / 2, cy, scale=sc)
            lab = pill(C.labels[nm], 52, bg=col)
            paste(base, lab, W / 2, cy + 245, scale=sc)
        q = text_img("?", 200, fill=YELLOW, stroke=12)
        paste(base, q, W / 2 - 120, 935, scale=ease_out_back((k - 0.25) / 0.2) * (1 + 0.08 * math.sin(k * 10)))
        # Pin drops in from the top and bounces
        dk = min(1.0, k / 0.45)
        y = -200 + (935 + 200) * (1 - (1 - dk) ** 2) - (abs(math.sin((k - 0.45) * 10)) * 60 * math.exp(-(k - 0.45) * 4) if k > 0.45 else 0)
        spr = mascot.draw("shock", 0.7, 10 * math.sin(k * 6), 0.5 * math.exp(-(k - 0.45) * 8) if k > 0.45 else -0.4, -1)
        paste(base, spr, W / 2 + 140, y - 40)
        sx, sy = shake(k - 0.12, 16, 0.3)
        if sx or sy:
            base = Image.fromarray(np.roll(np.roll(np.array(base), int(sy), 0), int(sx), 1))
        top = text_img("HOW ARE THEY CONNECTED?", 60, fill=WHITE, stroke=8)
        paste(base, top, W / 2, 200, scale=min(1, ease(k / 0.12)))
        seg["cap"].draw(base, tt, y=1720)
        return base

    def frame_outro(seg, tt):
        k = tt - seg["t0"]
        base = Image.new("RGBA", (W, H), (12, 12, 16, 255))
        n = len(C.people)
        top = text_img(f"{n_links} PHOTOS. {n} PEOPLE.", 66, fill=YELLOW, stroke=8)
        paste(base, top, W / 2, 190, scale=min(1, ease_out_back(k / 0.3)))
        size = min(170, int(1250 / n * 0.8))
        y0, y1 = 360, 1500
        step = (y1 - y0) / max(1, n - 1)
        d = ImageDraw.Draw(base)
        for j, nm in enumerate(C.people):
            appear = 0.12 * j
            sc = ease_out_back((k - appear) / 0.25)
            y = y0 + step * j
            x = W / 2 - 170 if j % 2 == 0 else W / 2 + 170
            if j < n - 1 and k > appear + 0.1:
                y2 = y0 + step * (j + 1)
                x2 = W / 2 - 170 if (j + 1) % 2 == 0 else W / 2 + 170
                d.line([(x, y), (x2, y2)], fill=CYAN + (255,), width=8)
        for j, nm in enumerate(C.people):
            appear = 0.12 * j
            sc = ease_out_back((k - appear) / 0.25)
            y = y0 + step * j
            x = W / 2 - 170 if j % 2 == 0 else W / 2 + 170
            col = YELLOW if j in (0, n - 1) else CYAN
            paste(base, ring(C.thumbs[nm].resize((size, size), Image.BICUBIC), col, 6), x, y, scale=sc)
            lab = pill(C.labels[nm], 34, bg=col)
            lx = x + (size / 2 + lab.width / 2 + 10) * (1 if j % 2 == 0 else -1)
            lx = max(lab.width / 2 + 8, min(W - lab.width / 2 - 8, lx))
            paste(base, lab, lx, y, scale=sc)
        seg["cap"].draw(base, tt, y=1640)
        cta = pill("COMMENT TWO NAMES  ↓", 46, fg=BLACK, bg=YELLOW)
        if k > 1.0:
            paste(base, cta, W / 2, 1800, scale=ease_out_back((k - 1.0) / 0.3))
        return base

    draw = {"hook": frame_hook, "link": frame_link, "outro": frame_outro}

    def frame_at(tt):
        for j, sg in enumerate(segs):
            if sg["t0"] <= tt < sg["t1"] or j == len(segs) - 1:
                im = draw[sg["kind"]](sg, tt)
                # crossfade from previous segment's last frame
                if j > 0 and tt - sg["t0"] < XFADE and sg["kind"] != "link":
                    prev = segs[j - 1]
                    pim = draw[prev["kind"]](prev, prev["t1"] - 1e-3)
                    a = ease((tt - sg["t0"]) / XFADE)
                    im = Image.blend(pim, im, a)
                return finish(im, tt)
        raise AssertionError

    # film grain + vignette: cheap texture that makes still photos feel alive
    rng = np.random.default_rng(7)
    grains = [Image.fromarray(np.dstack([np.full((H // 2, W // 2), v, np.uint8) for v in (128, 128, 128)] +
                                        [(rng.random((H // 2, W // 2)) * 22).astype(np.uint8)])).resize((W, H))
              for _ in range(4)]
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
    vign = Image.fromarray(np.dstack([np.zeros((H, W), np.uint8)] * 3 + [np.clip((dist - 0.55) * 260, 0, 150).astype(np.uint8)]))

    def finish(im, tt):
        im = im.copy()
        im.alpha_composite(vign)
        im.alpha_composite(grains[int(tt * FPS) % 4])
        return im

    # ---------- encode
    out = out_dir / f"{slug}.mp4"
    nframes = int(total * FPS)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", str(wav),
           "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
           "-shortest", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(nframes):
        im = frame_at(fi / FPS).convert("RGB")
        proc.stdin.write(im.tobytes())
        if fi % 150 == 0:
            log(f"  frame {fi}/{nframes}")
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    timeline = [{"kind": s["kind"], "t0": round(s["t0"], 2), "t1": round(s["t1"], 2), **({"i": s["i"]} if "i" in s else {})}
                for s in segs]
    (out_dir / "timeline.json").write_text(json.dumps({"duration": total, "segments": timeline}, indent=1))
    log(f"wrote {out}")
    return out
