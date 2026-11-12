"""Pixel-arcade style renderer (the channel look).

Photos and paintings in a white card with a hard shadow, on a scrolling two-tone pixel checkerboard;
chunky pixel name tags and captions (Press Start 2P); Pin (pixel pushpin in a detective hat)
thrown along the red evidence string; "?" blocks that fill in as people are revealed;
pixelate-dissolve cuts; 8-bit SFX; CONNECTED! on the final reveal.
"""
from __future__ import annotations

import json
import math
import re
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
HERO_A = (0, 790, 470)                # hook: face A photo spans y0..y1 (full bleed), face centred at y=fy within it
HERO_B = (900, H, 340)                # face B photo; the 790..900 seam between them carries the spoken question


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

    class _NoCapEarly:
        def draw(self, *a, **k):
            pass

    class FactCap:
        """Music-only: the link's fact as 1-2 lines of pixel text, popping in just after the cut."""
        def __init__(self, text, t0):
            self.t0 = t0
            words, lines, cur = text.upper().split(), [], ""
            for w in words:
                if len(cur) + len(w) + 1 > 22 and cur:
                    lines.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}".strip()
            if cur:
                lines.append(cur)
            if len(lines) > 2:
                lines = lines[:2]
                lines[1] = lines[1].rstrip(",;:…") + "…"
            self.lines = [ln.rstrip(",;:") if k == len(lines) - 1 and not ln.endswith("…") else ln
                          for k, ln in enumerate(lines)]

        def draw(self, base, t, y=CAP_Y):
            if t < self.t0 + 0.15 or not self.lines:
                return
            pop = ease_out_back(min(1.0, (t - self.t0 - 0.15) / 0.12))
            for k, ln in enumerate(self.lines):
                size = 38
                while size > 24 and pf(size).getlength(ln) > SAFE[2] - SAFE[0] - 30:
                    size -= 2
                paste(base, pixel_text(ln, size, fill=YELLOW if k == 0 else WHITE), CX,
                      y - 26 * (len(self.lines) - 1) + k * 54, scale=0.85 + 0.15 * pop)

    def fact_of(link):
        """The link's short on-screen fact: its "fact" field, else the year plus the first words of the line."""
        if link.get("fact"):
            return link["fact"]
        line = link["line"].rstrip(".")
        m = re.match(r"(?i)(?:And |Then |So )?(?:(?:In|By|Around|On|Back in)\s+)?(\d{4}s?),\s*(.*)", line)
        year, rest = (m.group(1), m.group(2)) if m else ("", re.sub(r"^(And |Then |So )", "", line))
        words = rest.split()
        short = " ".join(words[:6]) + ("…" if len(words) > 6 else "")
        return f"{year}: {short}" if year else short

    t = 0.0
    n_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    hook = spec.get("hook") or f"How is {short_name(spec['from'])} connected to {short_name(spec['to'])}?"
    # A/B test: half the episodes end with a spoken "And that's how." over the CONNECTED! stamp (loops into the
    # opening question on replay). Deterministic per slug; an episode can force it with "outro": true/false.
    import hashlib
    OUTRO = "And that's how."
    use_outro = spec["outro"] if "outro" in spec else int(hashlib.sha1(slug.encode()).hexdigest(), 16) % 2 == 0
    # A/B test 2: a third of the episodes are music-only (no narration, just names on screen and the opening question
    # as text), like our 40k-view 2025 Da Vinci -> Naruto. The voice still sets the timing, it just isn't mixed in.
    narration = spec["narration"] if "narration" in spec else int(hashlib.sha1((slug + ":narration").encode()).hexdigest(), 16) % 3 != 0
    if not narration:
        use_outro = False
    # presentation style pilot: episode "style" wins; otherwise "cards" until the rotation is switched on
    STYLE = spec.get("style") or "cards"
    script = [hook] + [l["line"] for l in spec["links"]] + ([OUTRO] if use_outro else [])
    audio.prefetch(script, voice, log)                                       # Gemini: one request per episode
    vinfo = {"engine": audio.narrator(script, voice), "outro": use_outro, "rev": audio.VOICE_REV,
             "narration": narration, "links": len(spec["links"]), "style": STYLE,
             "lineage": sum(len(l["lineage"]["people"]) for l in spec["links"] if l.get("lineage"))}
    vinfo["clipped"] = [t[:60] for t in script if audio.clipped_end(audio.say(t, voice)[0])]
    (out_dir / "voice.json").write_text(json.dumps(vinfo))
    a, words = audio.say(hook, voice)
    HOOK_VO = 0.0    # the question starts on the very first frame
    d = max(len(a) / audio.SR + 0.3 + HOOK_VO, FF + RW + 1.3)
    if not narration:
        d = 3.0                     # the reference hits all hold the intro card for about 3 s
    segs.append({"kind": "hook", "t0": t, "t1": t + d, "cap": PixelCaptions(words, t + HOOK_VO)})
    voice_clips.append((t + HOOK_VO, a))
    sfx += [(0.0, assets.sfx("impact", 0.4)),                              # A slams in on frame 0
            (0.18, assets.sfx("bass_hit", 0.4)),                            # B lands
            (0.5, assets.sfx("px_coin", 0.3))]
    t += d
    # music-only: short chains get slower links so the video still runs >= 18 s (Shorts need 15 s; QA checks)
    n_fast = 2 if n_links > 3 else 0
    n_lin = sum(1 for l in spec["links"] if l.get("lineage"))
    mid_n = max(1, n_links - n_fast - n_lin)
    mid_pace = max(3.3, (18.0 - 3.0 - 2.2 * n_fast - 4.0 * n_lin) / mid_n)
    for i, link in enumerate(spec["links"]):
        a, words = audio.say(link["line"], voice)
        last = i == n_links - 1
        speech = len(a) / audio.SR + 0.08
        if not narration:   # music-only: the reference hits' pace, ~3.3 s a link, a sprint at the end, fast lineages
            speech = 4.0 if link.get("lineage") else (2.2 if i >= n_links - 2 and n_links > 3 else mid_pace)
        # last photo: cut hard right after the final word / the CONNECTED! slam, for a seamless loop
        # the final destination stays hidden (camera tight on A) until the END of the last line, then lands and we
        # cut almost at once: "ends super fast after the last link is shown" (what the 40k-view 2025 hit did)
        rv = max(REVEAL, (speech - THROW - 0.3) / speech) if last and not link.get("lineage") else REVEAL
        d = max(speech + 0.05, rv * speech + THROW + 0.3) if last else speech + PAD
        if last and STYLE == "board":
            d += 1.1                                     # time to zoom out to the whole evidence board
        reveal = t + (speech * 0.9 - THROW if link.get("lineage") else rv * speech)
        if last and use_outro:   # "And that's how." lands with the CONNECTED! stamp, then a hard cut
            oa, ow = audio.say(OUTRO, voice)
            ot = max(t + 0.08 + speech, reveal + THROW + 0.3)
            voice_clips.append((ot, oa))
            words = list(words) + [(w, a0 + ot - (t + 0.08), b0 + ot - (t + 0.08)) for w, a0, b0 in ow]
            d = max(d, ot - t + len(oa) / audio.SR + 0.12)
        segs.append({"kind": "link", "i": i, "t0": t, "t1": t + d, "cap": PixelCaptions(words, t + 0.08),
                     "speech": speech, "rv": rv})
        voice_clips.append((t + 0.08, a))
        sfx += [(t, assets.sfx("px_cut", 0.45)), (t, assets.sfx("shutter", 0.35)),
                (reveal, assets.sfx("px_jump", 0.35)),
                (reveal + THROW, assets.sfx("px_coin", 0.45))]
        if last:
            sfx += [(max(0, reveal - 1.6), assets.sfx("riser", 0.3, max_len=1.6)),
                    (reveal + THROW, assets.sfx("px_bomb", 0.5)),
                    (reveal + THROW + 0.35, assets.sfx("px_pop", 0.4))]
        t += d
    total = t
    # music-only uses a trending song (an episode's "music" pick only applies to narrated renders); opt out with
    # "trending": false
    track = (assets.pick_trending(slug) if not narration and spec.get("trending", True) else None) \
        or assets.pick_track(slug, spec.get("music"))
    vinfo.update(music=f'{track.get("artist", "")} - {track["title"]}'.strip(" -"), trending=bool(track.get("trending")))
    (out_dir / "voice.json").write_text(json.dumps(vinfo))
    if not narration:          # music-only: keep the hook's caption (the question as text), drop the spoken lines,
        voice_clips = []       # and show a short fact per link instead (the visible research)
        for sg in segs:
            if sg["kind"] != "hook":
                sg["cap"] = FactCap(fact_of(spec["links"][sg["i"]]), sg["t0"])
    wav = audio.mix(total, voice_clips, sfx, seed=0, out=out_dir / "mix.wav", music=assets.music(track, total))
    log(f"music: {track['title']} ({'trending' if track.get('trending') else 'Mixkit'}); timeline {total:.1f}s")

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
        """Header name: the full label when it fits (COOKIE MONSTER, KERMIT THE FROG, MARILYN MONROE); for long
        ones the episode's people[n]["short"] if set (SPONGEBOB), else the surname."""
        lab = C.labels[n].upper()
        override = spec.get("people", {}).get(n, {}).get("short")
        if override:
            return override.upper()
        last = lab.split()[-1]
        return lab if (len(lab) <= 15 or len(last) <= 3) else last
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
            paste(base, pixel_text(f"LINK {i + 1}", 22, fill=YELLOW, shadow=False), CX, 232)   # no total: the end stays a surprise

    def blocks(base, upto, active, tt, sharpen=1.0):
        """Chain bar: the faces revealed so far, then a "?" slot, with the destination (blurred until reached) at the
        far right. It never shows how many links are left, so viewers can't tell the end is near."""
        n = len(C.people)
        shown = list(range(min(upto + 1, n)))                 # revealed so far (the newest one sharpens in)
        reached_end = shown and shown[-1] == n - 1
        slots = shown + ([] if reached_end else ["?", n - 1])
        span = SAFE[2] - SAFE[0]
        gap = span / max(6, len(slots))
        size = min(92, int(gap) - 16)
        y = BAR_Y
        dr = ImageDraw.Draw(base)
        for pos, k in enumerate(slots):
            cx = SAFE[0] + gap * (pos + 0.5) if not (k == n - 1 and not reached_end) else SAFE[2] - gap * 0.5
            bob = -6 * abs(math.sin(tt * 6)) if k in active else 0
            x0, y0 = cx - size / 2, y - size / 2 + bob
            dr.rectangle([x0 + 8, y0 + 8, x0 + size + 8, y0 + size + 8], fill=INK)
            dr.rectangle([x0, y0, x0 + size, y0 + size], fill=YELLOW if k in active else WHITE, outline=INK, width=6)
            inner = size - 12
            if k == "?":
                q = pixel_text("?", max(20, inner // 2), fill=INK, outline=WHITE, shadow=False)
                paste(base, q, cx, y + bob)
            else:
                nm = C.people[k]
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
        R = seg.get("rv", REVEAL)                       # this link's reveal point (late on the final link)
        hold = max(0.62, R + 0.08)
        if k < 0.10:
            cam = cm["a"]
        elif k < R:
            pan = 0.10 if R <= REVEAL else R - 0.12     # final link: stay on A until just before the reveal
            cam = cm["a"] if k < pan else lerp_cam(cm["a"], cm["both"], ease((k - pan) / (R - pan)))
        elif k < hold:
            cam = cm["both"]
        else:
            cam = lerp_cam(cm["both"], cm["b"], ease(min(1.0, (k - hold) / 0.30)))
        since = (k - R) * seg["speech"]
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
        for role, nm, appear, col in (("ba", names[0], 0.0, CYAN), ("bb", names[1], R + THROW / seg["speech"], YELLOW)):
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

    _hero = {}

    def hero(nm, h, fy):
        """A full-width rectangular photo of nm's face (face ~340 px tall, centred at (CX, fy)); gaps are a blur."""
        key = (nm, h, fy)
        if key not in _hero:
            img, box = C.hero_src[nm]
            ih, iw = img.shape[:2]
            if box is None:
                box = (iw * 0.3, ih * 0.15, iw * 0.4, iw * 0.4)
            x, y, bw, bh = box
            s_ = 340 / max(bh, bw * 1.1)
            fx, fyy = x + bw / 2, y + bh * 0.45
            M = np.float32([[s_, 0, CX - fx * s_], [0, s_, fy - fyy * s_]])
            crop = cv2.warpAffine(img, M, (W, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            mask = cv2.warpAffine(np.full((ih, iw), 255, np.uint8), M, (W, h), flags=cv2.INTER_LINEAR)
            cover = max(W / iw, h / ih)                                   # blurred cover fill behind the photo
            bg = cv2.resize(img, (int(iw * cover) + 1, int(ih * cover) + 1))
            by0, bx0 = (bg.shape[0] - h) // 2, (bg.shape[1] - W) // 2
            bg = cv2.GaussianBlur(bg[by0:by0 + h, bx0:bx0 + W], (0, 0), 25) * 0.55
            m = (mask.astype(np.float32) / 255)[..., None]
            out = (crop * m + bg * (1 - m)).astype(np.uint8)
            _hero[key] = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).convert("RGBA")
        return _hero[key]

    def make_comment_card():
        """A YouTube-style comment card: avatar, @handle, the request in the viewer's words."""
        who = spec.get("requested_by", "")
        handle = who if who.startswith("@") else (who or "a viewer")
        text = spec.get("request_text") or f'{short_name(spec["from"])} to {short_name(spec["to"])}'
        fh = ImageFont.truetype(str(FONTS / "Inter-Bold.otf"), 30)
        ft = ImageFont.truetype(str(FONTS / "Inter-ExtraBold.otf"), 46)
        while ft.getlength(text) > 700 and ft.size > 28:
            ft = ImageFont.truetype(str(FONTS / "Inter-ExtraBold.otf"), ft.size - 2)
        w = int(max(560, min(860, 150 + max(fh.getlength(handle), ft.getlength(text)))))
        h = 150
        im = Image.new("RGBA", (w + 12, h + 12), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([12, 12, w + 11, h + 11], 26, fill=(0, 0, 0, 120))           # shadow
        d.rounded_rectangle([0, 0, w, h], 26, fill=(255, 255, 255, 255), outline=INK, width=4)
        d.ellipse([22, 30, 92, 100], fill=(120, 130, 150, 255))                             # avatar
        d.ellipse([44, 44, 70, 70], fill=(225, 228, 235, 255))
        d.pieslice([32, 70, 82, 120], 180, 360, fill=(225, 228, 235, 255))
        d.text((112, 26), handle, font=fh, fill=(96, 96, 96))
        d.text((112, 64), text, font=ft, fill=(15, 15, 15))
        return im

    comment_im = make_comment_card() if spec.get("requested_by") else None

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
        else:                                        # 3) versus: two big faces, full width, from frame 0
            kk = k - FF - RW
            base = Image.new("RGBA", (W, H), INK + (255,))
            for nm, (y0, y1, fy), col, sgn in ((spec["from"], HERO_A, CYAN, -1), (spec["to"], HERO_B, YELLOW, 1)):
                sx, sy = shake(kk - (0 if sgn < 0 else 0.06), 14, 0.22)
                z = 1.0 + 0.07 * min(1.0, kk / 3.0)                      # slow push-in on the face
                im = hero(nm, y1 - y0, fy)
                if z > 1.001:
                    im = im.resize((int(W * z), int((y1 - y0) * z)), Image.BICUBIC)
                    ox, oy = int(CX * (z - 1)), int(fy * (z - 1))
                    im = im.crop((ox, oy, ox + W, oy + (y1 - y0)))
                slide = int(sgn * 70 * (1 - ease(min(1.0, kk / 0.22))))  # A slides in from the left, B from the right
                base.alpha_composite(im, (slide + int(sx), y0 + int(sy)))
                tag = pixel_tag(C.labels[nm].upper(), col, 44)
                ty = y1 - tag.height // 2 - 24 if sgn < 0 else y0 + tag.height // 2 + 30   # A under its face, B above
                paste(base, tag, CX + int(sx), ty + int(sy))
            dr = ImageDraw.Draw(base)
            dr.rectangle([0, HERO_A[1], W, HERO_B[0]], fill=INK)               # the seam carries the question
            dr.rectangle([0, HERO_A[1] - 6, W, HERO_A[1]], fill=CYAN)
            dr.rectangle([0, HERO_B[0], W, HERO_B[0] + 6], fill=YELLOW)
            hop = abs(math.sin(kk * 8)) * 26
            paste_pin(base, pin_sprite("shock" if kk < 0.8 else ("smug" if int(kk * 2) % 2 else "angry"), 0, 0, -1),
                      SAFE[2] - 50, HERO_A[1] - 6 - hop)                 # Buh stands on the seam
            if spec.get("requested_by"):   # the viewer's comment IS the hook (every reference hit opens on one)
                pop = ease_out_back(min(1.0, kk / 0.25))
                paste(base, comment_im, CX, (HERO_A[1] + HERO_B[0]) // 2 + 10, scale=0.6 + 0.4 * pop, nearest=False)
                ax = CX + comment_im.width // 2 - 70
                ay = (HERO_A[1] + HERO_B[0]) // 2 + comment_im.height // 2 + 10
                dr = ImageDraw.Draw(base)
                dr.line([(ax, ay), (ax + 30, ay + 70), (ax + 10, ay + 150)], fill=RED, width=14, joint="curve")
                dr.polygon([(ax - 18, ay + 132), (ax + 38, ay + 140), (ax + 6, ay + 190)], fill=RED)
                return base
        seg["cap"].draw(base, tt, y=(HERO_A[1] + HERO_B[0]) // 2)
        return base

    # ---------- presentation styles (pilot). "cards" is the original; the others reuse the same photos, face boxes,
    # timings and captions. Chosen by episode "style" or the STYLES rotation; recorded per upload.
    _rgb, _vbg = {}, {}

    def rgb(i):
        if i not in _rgb:
            _rgb[i] = cv2.cvtColor(C.imgs[i], cv2.COLOR_BGR2RGB)
        return _rgb[i]

    def view(i, cx, cy, s, ow, oh):
        """Photo i scaled by s with image point (cx, cy) at the centre of an ow x oh frame; blurred photo outside."""
        img = rgb(i)
        ih, iw = img.shape[:2]
        M = np.float32([[s, 0, ow / 2 - cx * s], [0, s, oh / 2 - cy * s]])
        out = cv2.warpAffine(img, M, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.warpAffine(np.full((ih, iw), 255, np.uint8), M, (ow, oh)).astype(np.float32)[..., None] / 255
        if (i, ow, oh) not in _vbg:
            cs = max(ow / iw, oh / ih)
            b = cv2.resize(img, (int(iw * cs) + 2, int(ih * cs) + 2))
            y0, x0 = (b.shape[0] - oh) // 2, (b.shape[1] - ow) // 2
            _vbg[(i, ow, oh)] = cv2.GaussianBlur(b[y0:y0 + oh, x0:x0 + ow], (0, 0), 25).astype(np.float32) * 0.5
        out = out * mask + _vbg[(i, ow, oh)] * (1 - mask)
        return Image.fromarray(out.astype(np.uint8)).convert("RGBA")

    def fbox(i, role):
        return cams[i]["ba" if role == "a" else "bb"]

    def ctr(b):
        return b[0] + b[2] / 2, b[1] + b[3] / 2

    def both_cam(i, ow, oh, fill=0.7):
        ba, bb = fbox(i, "a"), fbox(i, "b")
        x0, x1 = min(ba[0], bb[0]), max(ba[0] + ba[2], bb[0] + bb[2])
        y0, y1 = min(ba[1], bb[1]), max(ba[1] + ba[3], bb[1] + bb[3])
        sc = min(ow * fill / max(1, x1 - x0), oh * fill / max(1, (y1 - y0) * 2.0))
        return (x0 + x1) / 2, (y0 + y1) / 2 + (y1 - y0) * 0.35, sc

    def face_cam(b, oh, frac=0.42):
        cx, cy = ctr(b)
        return cx, cy, oh * frac / max(b[3], 1)

    def lerp_cam3(p, q, u):
        u = max(0.0, min(1.0, u))
        return (p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u, math.exp(math.log(p[2]) + (math.log(q[2]) - math.log(p[2])) * u))

    def link_names(i):
        return spec["links"][i]["a"], spec["links"][i]["b"]

    def header(base, i):
        dr = ImageDraw.Draw(base)
        dr.rectangle([0, 0, W, BAND - 40], fill=INK)
        paste(base, title_im, CX, 165)
        paste(base, pixel_text(f"LINK {i + 1}", 22, fill=YELLOW, shadow=False), CX, 202)

    def connected(base, seg, tt):
        i = seg["i"]
        if i != n_links - 1:
            return
        land = tt - seg["t0"] - seg.get("rv", REVEAL) * seg["speech"] - THROW
        if land > 0.05:
            st = min(1.0, (land - 0.05) / 0.12)
            paste(base, pixel_text("CONNECTED!", 72, fill=RED), CX, 1020, scale=2.0 - ease(st))

    def year_of(i):
        l = spec["links"][i]
        m = re.search(r"\b(1\d{3}|20\d{2})\b", (l.get("fact") or "") + " " + l["line"])
        return int(m.group(1)) if m else None

    # --- zoom: one continuous camera, A's face -> both -> into B's face -> (B is next photo's A) ...
    def frame_zoom(seg, tt):
        i = seg["i"]
        if C.fj["links"][i].get("lineage"):
            return frame_lineage(seg, tt)
        sp, R, last = seg["speech"], seg.get("rv", REVEAL), i == n_links - 1
        k = (tt - seg["t0"]) / sp
        fa, fb, bc = face_cam(fbox(i, "a"), H), face_cam(fbox(i, "b"), H), both_cam(i, W, H, 0.78)
        if k < max(R, 0.3):
            cam = lerp_cam3(fa, bc, ease(k / max(R, 0.3)))
        elif last or k < 0.82:
            cam = bc
        else:
            cam = lerp_cam3(bc, fb, ease((k - 0.82) / 0.18))
        base = view(i, *cam, W, H)
        incut = tt - seg["t0"]
        if i > 0 and incut < 0.16 and not C.fj["links"][i - 1].get("lineage"):
            prev = view(i - 1, *face_cam(fbox(i - 1, "b"), H), W, H)
            base = Image.blend(prev, base, incut / 0.16)
        an, bn = link_names(i)
        for role, nm, appear, col in (("a", an, 0.0, CYAN), ("b", bn, R + THROW / sp, YELLOW)):
            if k < appear:
                continue
            fx, fy = ctr(fbox(i, role))
            x, y = (fx - cam[0]) * cam[2] + W / 2, (fy - cam[1]) * cam[2] + H / 2 + fbox(i, role)[3] * cam[2] * 0.62
            if 0 < x < W and 0 < y < H:
                paste(base, pixel_tag(C.labels[nm].upper(), col, 30), max(SAFE[0] + 150, min(SAFE[2] - 150, x)),
                      max(SAFE[1] + 120, min(SAFE[3] - 120, y)))
        header(base, i)
        connected(base, seg, tt)
        seg["cap"].draw(base, tt)
        return base

    # --- board: Buh pins each photo to a corkboard; red string from photo to photo; zoom out at the end
    BW, PW, PH = 1700, 600, 660
    bpos = [(BW / 2 + (-330 if j % 2 == 0 else 330), 560 + j * 620) for j in range(n_links)]
    BH = int(bpos[-1][1] + 760) if n_links else 1920
    _board = {}

    def polaroid(i):
        card = Image.new("RGBA", (PW, PH), (250, 248, 240, 255))
        card.alpha_composite(view(i, *both_cam(i, PW - 40, PH - 150, 0.8), PW - 40, PH - 150), (20, 20))
        dr = ImageDraw.Draw(card)
        an, bn = link_names(i)
        f = ImageFont.truetype(str(FONTS / "Inter-ExtraBold.otf"), 34)
        txt = f"{C.labels[an]} + {C.labels[bn]}"
        while f.getlength(txt) > PW - 40 and f.size > 20:
            f = ImageFont.truetype(str(FONTS / "Inter-ExtraBold.otf"), f.size - 2)
        dr.text((PW / 2, PH - 72), txt, font=f, fill=(30, 30, 30), anchor="mm")
        yr = year_of(i)
        if yr:
            dr.text((PW / 2, PH - 32), str(yr), font=ImageFont.truetype(str(FONTS / "Inter-Bold.otf"), 26), fill=(120, 110, 100), anchor="mm")
        return card.rotate(4 if i % 2 else -4, expand=True, resample=Image.BICUBIC)

    def board_layer(k):
        """Cork + the first k polaroids pinned, with strings between them."""
        if k not in _board:
            if k == 0:
                rng = np.random.default_rng(7)
                cork = np.zeros((BH, BW, 3), np.float32) + np.array([168, 120, 74], np.float32)
                cork += rng.normal(0, 14, (BH, BW, 1)).astype(np.float32)
                cork = cv2.GaussianBlur(cork, (0, 0), 1.2)
                _board[0] = Image.fromarray(np.clip(cork, 0, 255).astype(np.uint8)).convert("RGBA")
            else:
                im = board_layer(k - 1).copy()
                pc = polaroid(k - 1)
                x, y = bpos[k - 1]
                im.alpha_composite(pc, (int(x - pc.width / 2), int(y - pc.height / 2)))
                dr = ImageDraw.Draw(im)
                if k >= 2:
                    (x0, y0), (x1, y1) = bpos[k - 2], bpos[k - 1]
                    dr.line([(x0, y0 - PH / 2 + 20), (x1, y1 - PH / 2 + 20)], fill=(200, 20, 30), width=8)
                for j in range(k):
                    px, py = bpos[j]
                    dr.ellipse([px - 16, py - PH / 2 + 4, px + 16, py - PH / 2 + 36], fill=(220, 30, 40), outline=INK, width=4)
                _board[k] = im
        return _board[k]

    def board_view(layer, cx, cy, v):
        rw, rh = W / v, H / v
        box = (int(cx - rw / 2), int(cy - rh / 2), int(cx + rw / 2), int(cy + rh / 2))
        return layer.crop(box).resize((W, H), Image.BILINEAR)

    def frame_board(seg, tt):
        i = seg["i"]
        if C.fj["links"][i].get("lineage"):
            return frame_lineage(seg, tt)
        sp, R, last = seg["speech"], seg.get("rv", REVEAL), i == n_links - 1
        local = tt - seg["t0"]
        v = 1.25
        cur = bpos[i]
        prev = bpos[i - 1] if i > 0 else (cur[0], cur[1] - 620)
        u = ease(min(1.0, local / 0.55))
        cx, cy = prev[0] + (cur[0] - prev[0]) * u, prev[1] + (cur[1] - prev[1]) * u + 120
        land = local - R * sp - THROW
        if last and land > 0.25:                       # zoom out to the whole board
            z = ease(min(1.0, (land - 0.25) / 0.7))
            vfit = min(W / BW, H / BH) * 0.98
            v = math.exp(math.log(1.25) + (math.log(vfit) - math.log(1.25)) * z)
            cx, cy = cx + (BW / 2 - cx) * z, cy + (BH / 2 - cy) * z
        layer = board_layer(i + 1 if (last and land > 0.25) else i)
        base = board_view(layer, cx, cy, v)
        to_s = lambda bx, by: ((bx - cx) * v + W / 2, (by - cy) * v + H / 2)
        if not (last and land > 0.25):
            drop = ease_out_back(min(1.0, local / 0.3))
            pc = polaroid_cache.setdefault(i, polaroid(i))
            sx, sy = to_s(*cur)
            paste(base, pc, sx, sy - 80 * (1 - drop), scale=v * (1.25 - 0.25 * drop), nearest=False)
            if i > 0:                                  # the red string runs out from the last photo
                g = min(1.0, max(0.0, (local - 0.15) / 0.4))
                ax, ay = to_s(prev[0], prev[1] - PH / 2 + 20)
                bx, by = to_s(cur[0], cur[1] - PH / 2 + 20)
                ImageDraw.Draw(base).line([(ax, ay), (ax + (bx - ax) * g, ay + (by - ay) * g)], fill=(200, 20, 30), width=int(8 * v))
            px, py = to_s(cur[0], cur[1] - PH / 2 + 20)
            hop = abs(math.sin(local * 7)) * 18
            paste_pin(base, pin_sprite("shock" if local < 0.5 else ("smug" if k_even(local) else "happy"), 0, 0, 1), px, py - hop)
        header(base, i)
        connected(base, seg, tt)
        seg["cap"].draw(base, tt)
        return base

    polaroid_cache = {}

    def k_even(t):
        return int(t * 2) % 2 == 0

    # --- timeline: the photo above, a big year counter and a ruler collecting faces below
    years = [year_of(j) for j in range(n_links)]
    for j in range(n_links):                           # links without a year inherit the previous one
        if years[j] is None:
            years[j] = years[j - 1] if j and years[j - 1] else None
    known = [y for y in years if y]
    ymin, ymax = (min(known), max(known)) if known else (1900, 2025)
    if ymax - ymin < 10:
        ymin, ymax = ymin - 5, ymax + 5
    pad = (ymax - ymin) * 0.08
    RX0, RX1, RY = SAFE[0] + 40, SAFE[2] - 40, 1395

    def rx(y):
        return RX0 + (RX1 - RX0) * ((y - ymin + pad) / (ymax - ymin + 2 * pad))

    def frame_timeline(seg, tt):
        i = seg["i"]
        if C.fj["links"][i].get("lineage"):
            return frame_lineage(seg, tt)
        local = tt - seg["t0"]
        base = checker(tt * 2, ((28, 24, 52), (22, 18, 44)))
        ph = view(i, *both_cam(i, 900, 760, 0.8), 900, 760)
        dr = ImageDraw.Draw(base)
        dr.rectangle([CX - 466, 286, CX + 466, 1066], fill=WHITE, outline=INK, width=6)
        base.alpha_composite(ph, (int(CX - 450), 296))
        an, bn = link_names(i)
        sp, R = seg["speech"], seg.get("rv", REVEAL)
        paste(base, pixel_tag(C.labels[an].upper(), CYAN, 28), CX - 220, 1030)
        if local > R * sp + THROW:
            paste(base, pixel_tag(C.labels[bn].upper(), YELLOW, 28), CX + 220, 1030)
        y_now = years[i] or ymin
        y_prev = (years[i - 1] or y_now) if i else y_now
        u = ease(min(1.0, local / 0.6))
        yv = y_prev + (y_now - y_prev) * u
        paste(base, pixel_text(str(int(round(yv))), 110, fill=YELLOW), CX, 1240)
        dr = ImageDraw.Draw(base)
        dr.rectangle([RX0, RY - 5, RX1, RY + 5], fill=WHITE)
        step = 10 if ymax - ymin <= 120 else (50 if ymax - ymin <= 600 else 250)
        for yy in range(int(ymin // step * step), int(ymax) + step, step):
            x = rx(yy)
            if RX0 <= x <= RX1:
                dr.rectangle([x - 2, RY - 16, x + 2, RY + 16], fill=WHITE)
        for j in range(i + 1):                        # faces collected on the ruler
            if years[j] is None:
                continue
            xj = rx(years[j]) if j < i else rx(yv)
            th = sharp[C.people[j + 1]] if j + 1 < len(C.people) else None
            if th is not None:
                paste(base, th.resize((64, 64), Image.BICUBIC), xj, RY - 52 - (j % 2) * 30, nearest=False)
        header(base, i)
        connected(base, seg, tt)
        seg["cap"].draw(base, tt, y=1150)
        return base

    # --- map: Buh flies to where each photo was taken (links need "where": {"lat", "lon", "place"})
    _map = {}

    def world_map():
        if "img" not in _map:
            m = Image.open(FONTS.parent / "world_equirect.png").convert("LA")
            a = np.array(m.split()[-1].resize((3840, 1950), Image.NEAREST)) > 128
            img = np.zeros((1950, 3840, 3), np.uint8)
            img[:] = (24, 44, 86)
            img[a] = (70, 150, 96)
            yy, xx = np.mgrid[0:1950, 0:3840]
            grid = ((xx % 40 == 0) | (yy % 40 == 0)) & ~a
            img[grid] = (30, 54, 100)
            _map["img"] = Image.fromarray(img).convert("RGBA")
        return _map["img"]

    def geo(i):
        w = spec["links"][i].get("where") or {}
        if "lat" not in w:
            return None
        return (w["lon"] + 180) / 360 * 3840, (90 - w["lat"]) / 180 * 1950

    def frame_map(seg, tt):
        i = seg["i"]
        if C.fj["links"][i].get("lineage") or geo(i) is None:
            return frame_timeline(seg, tt)
        local = tt - seg["t0"]
        cur = geo(i)
        prev = next((geo(j) for j in range(i - 1, -1, -1) if geo(j)), None) or cur
        fly = ease(min(1.0, local / 0.9))
        dist = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        vmid = min(1.4, max(0.28, W * 0.7 / max(dist, 1)))     # zoomed out mid-flight, back in on landing
        v = 1.4 + (vmid - 1.4) * math.sin(math.pi * fly) if dist > 50 else 1.4
        cx, cy = prev[0] + (cur[0] - prev[0]) * fly, prev[1] + (cur[1] - prev[1]) * fly
        mp = world_map()
        rw, rh = W / v, H / v
        base = mp.crop((int(cx - rw / 2), int(cy - rh / 2 - 200 / v), int(cx + rw / 2), int(cy + rh / 2 - 200 / v))).resize((W, H), Image.NEAREST)
        to_s = lambda px, py: ((px - cx) * v + W / 2, (py - cy) * v + H / 2 + 200)
        dr = ImageDraw.Draw(base)
        for j in range(i + 1):                          # visited places and the route
            g = geo(j)
            if not g:
                continue
            gx, gy = to_s(*g)
            dr.ellipse([gx - 12, gy - 12, gx + 12, gy + 12], fill=RED, outline=INK, width=4)
        ax, ay = to_s(*prev)
        bx, by = to_s(cx, cy)
        n_d = int(math.hypot(bx - ax, by - ay) / 24)
        for d in range(n_d):
            px, py = ax + (bx - ax) * d / max(n_d, 1), ay + (by - ay) * d / max(n_d, 1)
            dr.rectangle([px - 4, py - 4, px + 4, py + 4], fill=WHITE)
        hop = abs(math.sin(local * 9)) * 16
        paste_pin(base, pin_sprite("shock" if fly < 1 else "happy", 0, 0, 1 if cur[0] >= prev[0] else -1), bx, by - hop)
        place = (spec["links"][i].get("where") or {}).get("place", "")
        if fly >= 1 and place:
            paste(base, pixel_tag(place.upper(), WHITE, 26), bx, by + 60)
        if local > 0.75:                                # the photo pops up over the landing spot
            pop = ease_out_back(min(1.0, (local - 0.75) / 0.25))
            card = Image.new("RGBA", (720, 620), WHITE + (255,))
            card.alpha_composite(view(i, *both_cam(i, 700, 600, 0.8), 700, 600), (10, 10))
            paste(base, card, CX, 640, scale=0.5 + 0.5 * pop, nearest=False)
            an, bn = link_names(i)
            sp, R = seg["speech"], seg.get("rv", REVEAL)
            paste(base, pixel_tag(C.labels[an].upper(), CYAN, 28), CX - 200, 990)
            if local > R * sp + THROW:
                paste(base, pixel_tag(C.labels[bn].upper(), YELLOW, 28), CX + 200, 990)
        header(base, i)
        connected(base, seg, tt)
        seg["cap"].draw(base, tt)
        return base

    style_frames = {"cards": frame_link, "zoom": frame_zoom, "board": frame_board, "timeline": frame_timeline,
                    "map": frame_map}
    draw = {"hook": frame_hook, "link": style_frames.get(STYLE, frame_link)}

    def frame_at(tt):
        for j, sg in enumerate(segs):
            if sg["t0"] <= tt < sg["t1"] or j == len(segs) - 1:
                return draw[sg["kind"]](sg, tt)
        raise AssertionError

    # thumbnail: the settled opening (both faces, names, Buh) without captions, plus a yellow "?" on the seam
    try:
        hs = segs[0]
        g = globals()
        saved = g["CX"]
        g["CX"] = W / 2                      # no feed buttons over a thumbnail: centre it properly
        try:
            th = frame_hook({**hs, "cap": _NoCap()}, hs["t0"] + 1.1)
        finally:
            g["CX"] = saved
        q = pixel_text("?", 170, fill=YELLOW)                       # on the seam, between the two faces
        th.alpha_composite(q, (int(W / 2 - q.width / 2), int((HERO_A[1] + HERO_B[0]) / 2 - q.height / 2)))
        th.convert("RGB").save(out_dir / "thumb.jpg", quality=92)
    except Exception as e:  # noqa: BLE001 - kit falls back to a video frame
        log(f"thumbnail skipped: {e}")
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
