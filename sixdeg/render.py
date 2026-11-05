"""Render an episode to a 1080x1920 Short.

Structure:  hook (two faces + "?")  ->  one segment per photo link (camera pans
from person A's face to B's face, name tags pop, narration with captions)
->  outro (whole chain recap + call to comment).
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import audio
from .common import FONTS, slugify
from .episode import load, people_order

W, H, FPS = 1080, 1920, 30
CYAN = (34, 211, 238)
YELLOW = (255, 214, 10)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PAD = 0.35          # silence after each narration line
XFADE = 0.22        # cross-fade between segments


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
        self.imgs = [cv2.imread(str(self.work / l["image"])) for l in self.fj["links"]]
        # face thumbnails: prefer the person's Wikipedia portrait (big, frontal); fall back to a link photo
        self.thumbs = {}
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
        for i, l in enumerate(self.fj["links"]):
            for role in ("a", "b"):
                name = self.spec["links"][i][role]
                bx = l[role]["box"]
                if name not in self.thumbs and bx:
                    self.thumbs[name] = face_crop(self.imgs[i], bx, 220)
        for name in self.people:
            if name not in self.thumbs:  # no usable portrait and never boxed in a photo
                refs = self.fj["people"][name]["refs"]
                im = cv2.imread(str(self.work / "refs" / refs[0])) if refs else np.full((300, 300, 3), 60, np.uint8)
                self.thumbs[name] = face_crop(im, None, 220)
        self.labels = {n: self.spec.get("people", {}).get(n, {}).get("label", short_name(n)) for n in self.people}


def blurred_bg(img: np.ndarray, t: float) -> Image.Image:
    h, w = img.shape[:2]
    s = max(W / w, H / h) * (1.15 + 0.03 * t)
    M = np.float32([[s, 0, W / 2 - w * s / 2], [0, s, H / 2 - h * s / 2]])
    small = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    small = cv2.GaussianBlur(cv2.resize(small, (W // 8, H // 8)), (0, 0), 6)
    small = cv2.resize(small, (W, H))
    small = (small * 0.38).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).convert("RGBA")


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


def warp_photo(img, cam, center_y=980):
    cx, cy, s = cam
    M = np.float32([[s, 0, W / 2 - cx * s], [0, s, center_y - cy * s]])
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

def render(slug: str, voice: str = "am_michael", log=print) -> Path:
    C = Ctx(slug)
    spec = C.spec
    out_dir = C.dir / "out"
    out_dir.mkdir(exist_ok=True)

    # ---------- audio + timeline
    segs = []  # dicts: kind, t0, t1, ...
    voice_clips, sfx = [], []
    t = 0.0
    hook = spec.get("hook") or f"How is {short_name(spec['from'])} connected to {short_name(spec['to'])}?"
    a, words = audio.say(hook, voice)
    d = len(a) / audio.SR + 0.5
    segs.append({"kind": "hook", "t0": t, "t1": t + d, "cap": Captions(words, t + 0.1)})
    voice_clips.append((t + 0.1, a)); sfx.append((0.0, audio.boom()))
    t += d
    for i, link in enumerate(spec["links"]):
        a, words = audio.say(link["line"], voice)
        d = len(a) / audio.SR + PAD + 0.25
        segs.append({"kind": "link", "i": i, "t0": t, "t1": t + d, "cap": Captions(words, t + 0.2)})
        voice_clips.append((t + 0.2, a))
        sfx.append((max(0, t - 0.15), audio.whoosh(seed=i)))
        sfx.append((t + d * 0.42, audio.ding(1320 if i % 2 == 0 else 1480)))
        t += d
    n_links = len(spec["links"])
    outro = spec.get("outro") or (f"{short_name(spec['from'])} to {short_name(spec['to'])} in {n_links} photos. "
                                  f"Comment two names and I'll connect them next.")
    a, words = audio.say(outro, voice)
    d = len(a) / audio.SR + 1.6
    segs.append({"kind": "outro", "t0": t, "t1": t + d, "cap": Captions(words, t + 0.2)})
    voice_clips.append((t + 0.2, a)); sfx.append((t, audio.boom()))
    t += d
    total = t
    wav = audio.mix(total, voice_clips, sfx, seed=sum(map(ord, slug)), out=out_dir / "mix.wav")
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
        k = (tt - seg["t0"]) / (seg["t1"] - seg["t0"])
        # camera: A close -> both -> B close
        if k < 0.18:
            cam = cm["a"]
        elif k < 0.45:
            cam = lerp_cam(cm["a"], cm["both"], ease((k - 0.18) / 0.27))
        elif k < 0.72:
            cam = cm["both"]
        else:
            cam = lerp_cam(cm["both"], cm["b"], ease((k - 0.72) / 0.28))
        # gentle drift so it never freezes
        cam = (cam[0], cam[1], cam[2] * (1 + 0.02 * math.sin(tt * 1.3)))
        base = blurred_bg(img, tt - seg["t0"])
        ph, M = warp_photo(img, cam)
        base.alpha_composite(ph)
        # name tags
        names = (spec["links"][i]["a"], spec["links"][i]["b"])
        for role, nm, appear in (("ba", names[0], 0.0), ("bb", names[1], 0.40)):
            bx = cm[role]
            sx, sy = to_screen(M, bx[0] + bx[2] / 2, bx[1])
            pop = ease_out_back((k - appear) / 0.06) if k >= appear else 0
            if pop > 0:
                p = pill(C.labels[nm], 50 if len(C.labels[nm]) < 16 else 42,
                         bg=CYAN if role == "ba" else YELLOW)
                sy = max(330, min(1500, sy - 40))
                sx = max(p.width / 2 + 10, min(W - p.width / 2 - 10, sx))
                paste(base, p, sx, sy, scale=pop)
        # header + progress
        paste(base, title_im, W / 2, 150)
        step = pill(f"PHOTO {i + 1} / {n_links}", 34, fg=WHITE, bg=(30, 30, 30))
        paste(base, step, W / 2, 235)
        revealed = i + (1 if k >= 0.40 else 0)
        chain_bar(base, revealed, {i, i + 1} if k >= 0.40 else {i})
        credit_line(base, i)
        seg["cap"].draw(base, tt)
        return base

    def frame_hook(seg, tt):
        k = tt - seg["t0"]
        base = Image.new("RGBA", (W, H), (12, 12, 16, 255))
        # subtle radial glow
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.ellipse([W / 2 - 600, H / 2 - 600, W / 2 + 600, H / 2 + 600], fill=(34, 90, 120, 90))
        base.alpha_composite(glow.filter(ImageFilter.GaussianBlur(120)))
        a_name, b_name = spec["from"], spec["to"]
        for nm, cy, delay, col in ((a_name, 560, 0.0, CYAN), (b_name, 1300, 0.25, YELLOW)):
            sc = ease_out_back((k - delay) / 0.35)
            th = ring(C.thumbs[nm].resize((400, 400), Image.BICUBIC), col, 10)
            paste(base, th, W / 2, cy, scale=sc)
            lab = pill(C.labels[nm], 52, bg=col)
            paste(base, lab, W / 2, cy + 245, scale=sc)
        q = text_img("?", 200, fill=WHITE, stroke=12)
        paste(base, q, W / 2, 935, scale=ease_out_back((k - 0.5) / 0.3) * (1 + 0.05 * math.sin(k * 8)))
        top = text_img("HOW ARE THEY CONNECTED?", 60, fill=WHITE, stroke=8)
        paste(base, top, W / 2, 200, scale=min(1, ease(k / 0.25)))
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
                if j > 0 and tt - sg["t0"] < XFADE:
                    prev = segs[j - 1]
                    pim = draw[prev["kind"]](prev, prev["t1"] - 1e-3)
                    a = ease((tt - sg["t0"]) / XFADE)
                    im = Image.blend(pim, im, a)
                return im
        raise AssertionError

    # ---------- encode
    out = out_dir / f"{slug}.mp4"
    nframes = int(total * FPS)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", str(wav),
           "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
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
