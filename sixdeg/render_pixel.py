"""Pixel-arcade style renderer (the channel look).

Photos and paintings in a white card with a hard shadow, on a scrolling two-tone pixel checkerboard;
chunky pixel name tags and captions (Press Start 2P); Pin (pixel pushpin in a detective hat)
thrown along the red evidence string; "?" blocks that fill in as people are revealed;
pixelate-dissolve cuts; 8-bit SFX; CONNECTED! on the final reveal.
"""
from __future__ import annotations

import itertools
import json
import math
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import assets, audio, pixelpin
from .common import DATA, FONTS
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
# people for whom nothing celebratory may appear (hearts, likes, matches, coins, scores, trophies, rarity, gold
# frames, first-person posts): styles check sensitive(name) per link
SENSITIVE = re.compile(r"(?i)hitler|stalin|mussolini|mao zedong|pol pot|saddam|gaddafi|idi amin|bin laden|"
                       r"kim jong|himmler|goebbels|goering|göring|eichmann|mengele|lenin|franco|pinochet|escobar|"
                       r"capone|manson|bundy|dahmer|epstein|genghis|attila")


def sensitive(name) -> bool:
    return bool(SENSITIVE.search(name or ""))


MIN_LINK = 3.4                        # seconds on screen per link, at least (even pace to the end)
CAP_Y = 1300                          # captions
HERO_A = (0, 790, 470)                # hook: face A photo spans y0..y1 (full bleed), face centred at y=fy within it
HERO_B = (900, H, 340)                # face B photo; the 790..900 seam between them carries the spoken question
# A/B test "text_variant": "T1" (research 2026-11-13, owner-approved direction): the whole match-up as ONE static bold
# line ("HITLER → BIG BIRD?") from frame 0, in a light band between the faces; no chin tags; B's face bigger than A's;
# a brighter grade that keeps the blacks. Layout measured on the owner's Android app (video px): top icons to y ≈ 97,
# the like column at x ≥ 927 from y ≈ 1050, the viewer's channel row from y ≈ 1533. "T0" (default) is the opening above.
T1_A = (0, 780, 430)                  # A photo spans y 0..780, face centred at y 430
T1_BAND = (780, 1020)                 # the caption band ends above the like column
T1_B = (1020, H, 230)                 # B photo from y 1020, face centred at y ≈ 1250 (chin above ≈ 1450)
T1_FACE = (340, 400)                  # face heights (A, B): B reads bigger than A
T1_INK = (16, 16, 20)                 # near-black band text (≈ 13:1 on the yellow band)


def t1_grade(bgr: np.ndarray) -> np.ndarray:
    """T1 grade: brighter, never faded. Lifts the mid-tones on a gamma curve (black stays black, white stays white),
    then adds vibrance (muted colours gain the most; black-and-white photos stay black and white)."""
    lut = np.clip(255 * (np.arange(256) / 255) ** 0.78, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.LUT(lab[..., 0], lut)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[..., 1]
    if s.mean() > 18:                 # a colour photo
        hsv[..., 1] = np.clip(s + 0.6 * s * (1 - s / 255), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


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


_SUB = {}


def subscribe_button(base, t):
    """A pixel SUBSCRIBE button under the photo: pops in, a pointer clicks it at 0.55 s and it turns into
    SUBSCRIBED + a ringing bell. Lives in the gap between the photo card and the Shorts UI (y ~1470)."""
    if "on" not in _SUB:
        for k, (txt, bg, fg) in {"on": ("SUBSCRIBE", (204, 0, 0), WHITE), "off": ("SUBSCRIBED", (60, 60, 60), WHITE)}.items():
            lab = pixel_text(txt, 34, fill=fg, shadow=False)
            w, h = lab.width + 120, lab.height + 34
            im = Image.new("RGBA", (w + 10, h + 10), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            d.rounded_rectangle([10, 10, w + 9, h + 9], 18, fill=INK)                     # pixel drop shadow
            d.rounded_rectangle([0, 0, w, h], 18, fill=bg, outline=INK, width=5)
            bx, by = 26, h // 2                                                        # bell
            d.ellipse([bx - 14, by - 16, bx + 14, by + 10], fill=fg)
            d.rectangle([bx - 18, by + 4, bx + 18, by + 10], fill=fg)
            d.ellipse([bx - 5, by + 10, bx + 5, by + 18], fill=fg)
            im.alpha_composite(lab, (60, (h - lab.height) // 2))
            _SUB[k] = im
        hand = Image.new("RGBA", (44, 56), (0, 0, 0, 0))                               # pixel pointer
        d = ImageDraw.Draw(hand)
        d.polygon([(4, 2), (4, 44), (14, 34), (22, 52), (30, 48), (22, 30), (36, 30)], fill=WHITE, outline=INK)
        _SUB["hand"] = hand
    pop = ease_out_back(min(1.0, t / 0.18))
    clicked = t > 1.0
    btn = _SUB["off" if clicked else "on"]
    press = 0.92 if 0.95 < t < 1.07 else 1.0
    y = 1478
    paste(base, btn, CX, y, scale=pop * press)
    if t > 0.45:                                           # the pointer glides in from the lower right and taps
        u = ease(min(1.0, (t - 0.45) / 0.5))
        hx, hy = CX + 300 - 230 * u, y + 120 - 95 * u
        paste(base, _SUB["hand"], hx, hy, scale=1.0)


def _avatar(url):
    """Download (and cache) a commenter's YouTube profile picture as a 70x70 RGB image; None if unavailable."""
    if not url:
        return None
    import hashlib
    import urllib.request
    from .common import ROOT
    cache = ROOT / ".cache" / "avatars"
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / (hashlib.sha1(url.encode()).hexdigest() + ".img")
    try:
        if not f.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "sixdegrees-render/1.0"})
            f.write_bytes(urllib.request.urlopen(req, timeout=20).read())
        return Image.open(f).convert("RGB").resize((70, 70), Image.LANCZOS)
    except Exception:  # noqa: BLE001 - fall back to the default silhouette
        return None

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

    def before_b(text, link):
        """text cut just before B's name first appears (nothing on screen may name B before the reveal)."""
        a, b = link["a"], link["b"]
        a_words = set(re.findall(r"[A-Z][\w'-]{2,}", f"{a} {C.labels.get(a, '')}"))
        names = [w for w in set(re.findall(r"[A-Z][\w'-]{2,}", f"{b} {C.labels.get(b, '')}")) if w not in a_words]
        m = re.search(r"\b(" + "|".join(map(re.escape, names)) + r")\b", text) if names else None
        if not m:
            return text
        return text[:m.start()].rstrip(" ,;:-'’") + "…"

    def fact_of(link, full=False):
        """The link's short on-screen fact: its "fact" field, else the year plus the first words of the line.
        Cut before B's name unless full=True (use full=True only after the reveal)."""
        if link.get("fact"):
            return link["fact"] if full else before_b(link["fact"], link)
        line = link["line"].rstrip(".")
        m = re.match(r"(?i)(?:And |Then |So )?(?:(?:In|By|Around|On|Back in)\s+)?(\d{4}s?),\s*(.*)", line)
        year, rest = (m.group(1), m.group(2)) if m else ("", re.sub(r"^(And |Then |So )", "", line))
        if not full:
            rest = before_b(rest, link)
        words = rest.split()
        short = " ".join(words[:6]) + ("…" if len(words) > 6 else "")
        short = short.replace("……", "…")
        if short in ("…", ""):
            return year
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
    narration = spec.get("narration", True)   # creator's call 2026-11-12: the voice is our signature; music-only test ended
    if not narration:
        use_outro = False
    # presentation style pilot: episode "style" wins; otherwise "cards" until the rotation is switched on
    import os as _os
    STYLE = _os.environ.get("SIXDEG_STYLE") or spec.get("style")   # env: test a style without editing the episode
    if not STYLE:          # rotation from data/style.json "rotation", by slug hash; globe only with a place per link
        _sj = DATA / "style.json"
        _rot = (json.loads(_sj.read_text()).get("rotation") if _sj.exists() else None) or ["cards"]
        _placed = all(l.get("lineage") or (l.get("where") or {}).get("lat") is not None for l in spec["links"])
        _rot = [x for x in _rot if x != "globe" or _placed] or ["cards"]
        STYLE = _rot[int(hashlib.sha1((slug + ":style").encode()).hexdigest(), 16) % len(_rot)]
    # opening text A/B test (T0 = today's opening, T1 = the static match-up band); set per episode. A requested pair
    # keeps T0: there the viewer's comment card is the opening.
    TEXT_VARIANT = _os.environ.get("SIXDEG_TEXT") or spec.get("text_variant") or "T0"
    if TEXT_VARIANT == "T1" and (str(spec.get("requested_by", "")).startswith("@") or not narration):
        TEXT_VARIANT = "T0"
    T1 = TEXT_VARIANT == "T1"
    script = [hook] + [l["line"] for l in spec["links"]] + ([OUTRO] if use_outro else [])
    audio.prefetch(script, voice, log)                                       # Gemini: one request per episode
    vinfo = {"engine": audio.narrator(script, voice), "outro": use_outro, "rev": audio.VOICE_REV,
             "narration": narration, "links": len(spec["links"]), "style": STYLE,
             "lineage": sum(len(l["lineage"]["people"]) for l in spec["links"] if l.get("lineage")),
             "hook_variant": spec.get("hook_variant"), "text_variant": TEXT_VARIANT}   # A/B arms, copied to uploads
    vinfo["clipped"] = [t[:60] for t in script if audio.clipped_end(audio.say(t, voice)[0])]
    (out_dir / "voice.json").write_text(json.dumps(vinfo))
    a, words = audio.say(hook, voice)
    HOOK_VO = 0.0    # the question starts on the very first frame
    # owner: "the first word should always start the moment the video is scrolled to": trim any leading silence the
    # TTS clip carries (keep 10 ms so the attack isn't clipped) and shift the caption word times to match
    _a = np.abs(np.asarray(a, dtype=np.float32))
    _on = np.flatnonzero(_a > max(1e-4, 0.03 * float(_a.max() or 1.0)))
    _lead = max(0, int(_on[0]) - int(0.01 * audio.SR)) if len(_on) else 0
    if _lead:
        a = a[_lead:]
        _sh = _lead / audio.SR
        words = [(w, max(0.0, s0 - _sh), max(0.0, e0 - _sh)) for w, s0, e0 in words]
    # owner: "in the hook there cannot be any pauses, no dead space": link 1's voice follows the question at once
    # (the clip's own ~0.1 s tail + link 1's 0.08 s lead-in is the only breath; this used to hold +0.3 s of silence)
    d = max(len(a) / audio.SR - 0.06 + HOOK_VO, FF + RW + 1.3)
    if not narration:
        d = 3.0                     # the reference hits all hold the intro card for about 3 s
    segs.append({"kind": "hook", "t0": t, "t1": t + d, "cap": PixelCaptions(words, t + HOOK_VO),
                 "words": [(w, s0 + HOOK_VO, e0 + HOOK_VO) for w, s0, e0 in words]})
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
    _durs = []
    for i, link in enumerate(spec["links"]):
        a, words = audio.say(link["line"], voice)
        last = i == n_links - 1
        speech = len(a) / audio.SR + 0.08
        if link.get("lineage") and narration:   # a long line (45 presidents) gets a real beat, not just its sentence:
            speech = max(speech, min(7.0, 0.12 * len(link["lineage"]["people"]) + 0.6))   # ~0.12 s a portrait, cap 7 s
        if not narration:   # music-only: the reference hits' pace, ~3.3 s a link, a sprint at the end, fast lineages
            speech = 4.0 if link.get("lineage") else (2.2 if i >= n_links - 2 and n_links > 3 else mid_pace)
        # last photo: cut hard right after the final word / the CONNECTED! slam, for a seamless loop
        # the final destination stays hidden (camera tight on A) until the END of the last line, then lands and we
        # cut almost at once: "ends super fast after the last link is shown" (what the 40k-view 2025 hit did)
        rv = max(REVEAL, (speech - THROW - 0.3) / speech) if last and not link.get("lineage") else REVEAL
        if narration and not link.get("lineage"):
            # narrated: B lands as the voice says B's name (earlier spoils it on screen, later just lags)
            bt = {w.lower() for w in re.findall(r"[A-Za-z][\w'-]{2,}", f"{link['b']} {C.labels.get(link['b'], '')}")}
            at = {w.lower() for w in re.findall(r"[A-Za-z][\w'-]{2,}", f"{link['a']} {C.labels.get(link['a'], '')}")}
            hit = [a0 for w, a0, b0 in words if re.sub(r"[^\w'-]", "", w).lower() in (bt - at)]
            if hit:
                rv = max(REVEAL, (hit[0] + 0.08 - THROW + 0.15) / speech)
                rv = min(rv, (speech - THROW - 0.1) / speech) if not last else min(rv, max(REVEAL, (speech - THROW - 0.3) / speech))
        # last link: cut ~0.5 s after the voice ends (owner: "we should just cut as soon as the voice is done"), but
        # never before Buh has landed and CONNECTED! has stamped (~0.9 s after the landing)
        d = max(speech + 0.5, rv * speech + THROW + 0.9) if last else speech + PAD
        d = d if last else max(d, MIN_LINK)        # owner: the pace never speeds up; a short line still gets a full beat on screen
        # (the "payoff link as long as the average" hold was removed: it left ~3 s of music with no voice at the end)
        if not link.get("lineage"):
            _durs.append(d)
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
    track = (assets.pick_trending(slug) if not narration and spec.get("trending") is True else None) \
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

    def credit(base, i, xy=None):
        """Photo credit (CC licences require it), right-aligned at xy (default: under the cards photo)."""
        cr = C.fj["credits"][i]
        who = (cr.get("artist") or "").strip()
        if len(who) > 26 or "(" in who or "http" in who:
            who = "Wikimedia Commons" if cr.get("file") else ""
        txt = f"PHOTO: {who + ' / ' if who else ''}{cr.get('license', '')}".upper()
        ImageDraw.Draw(base).text(xy or (CARD[2], CARD[3] + 32), txt, font=sf(18), fill=(255, 255, 255), anchor="rm",
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
            _ci = pixel_text("CONNECTED!", 72, fill=RED)
            paste(base, _ci, CX, 1020, scale=min(2.0 - ease(st), (SAFE[2] - SAFE[0]) / _ci.width))   # never clipped
        upto = i + (1 if land >= 0 else 0)
        blocks(base, upto, {upto} if land >= 0 else {i}, tt, sharpen=min(1.0, land / 0.35) if land >= 0 else 1.0)
        seg["cap"].draw(base, tt)
        incut = tt - seg["t0"]
        if incut < 0.10:
            base.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(220 * (1 - incut / 0.10)))))
        return base

    def link_core(seg, tt):
        """THE SHOW'S CORE, shared by every presentation style (owner, 2026-11-12: Buh jumping with his pin and red
        string, name tags under the chins, and the zoom-out / move / zoom-in camera are crucial in every format).
        Returns the photo window (CARD-sized RGBA: photo + string + pin marker + tags, all riding the photo and
        clipped to its edge) and the pin's pose in card-local px. Styles place it with draw_core()."""
        i = seg["i"]
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
        ph, M = warp_photo(img, cam, center_y=(CARD[1] + CARD[3]) / 2)
        cw, ch = CARD[2] - CARD[0], CARD[3] - CARD[1]
        crop = ph.crop((CARD[0], CARD[1], CARD[2], CARD[3]))
        blk = int(46 * (1 - incut / 0.2)) if incut < 0.2 else 0
        if blk > 1:
            crop = mosaic(crop, blk)
        flat = card_bg[i].copy()          # blurred photo fills the card when zoomed out
        flat.alpha_composite(crop)
        M = M.copy(); M[0, 2] -= CARD[0]; M[1, 2] -= CARD[1]     # card-local coordinates
        pa = to_screen(M, cm["ba"][0] + cm["ba"][2] / 2, cm["ba"][1] - cm["ba"][3] * 0.08)
        pb = to_screen(M, cm["bb"][0] + cm["bb"][2] / 2, cm["bb"][1] - cm["bb"][3] * 0.08)
        # the red string + pinned marker live ON the photo (drawn into the card, so its edge clips them)
        if since >= 0:
            if since < THROW:
                x, y, _ = pixelpin_path(pa, pb, since / THROW)
                pe = (x, y)
            else:
                pe = pb
            sd = ImageDraw.Draw(flat)
            sd.line([pa, pe], fill=INK, width=14)
            sd.line([pa, pe], fill=RED, width=8)
            sd.rectangle([pa[0] - 10, pa[1] - 10, pa[0] + 10, pa[1] + 10], fill=RED, outline=INK, width=4)
        # name tags: stickers under the chins, riding the photo and sliding under its edge; never stacked
        names = (spec["links"][i]["a"], spec["links"][i]["b"])
        tags_ = []
        for role, nm, appear, col in (("ba", names[0], 0.0, CYAN), ("bb", names[1], R + THROW / seg["speech"], YELLOW)):
            bx = cm[role]
            if k >= appear:
                pop = ease_out_back((k - appear) * seg["speech"] / 0.12)
                tag = pixel_tag(C.labels[nm].upper(), col, 26 if len(C.labels[nm]) < 15 else 22)
                tx, ty = to_screen(M, bx[0] + bx[2] / 2, bx[1] + bx[3])   # under the chin
                tags_.append([tag, tx, ty + 34, pop])
        if len(tags_) == 2:
            (ta, ax, ay, _), (tb, bx_, by_, _) = tags_
            if abs(ax - bx_) < (ta.width + tb.width) / 2 + 8 and abs(ay - by_) < (ta.height + tb.height) / 2 + 6:
                tags_[1][2] = ay + (ta.height + tb.height) / 2 + 10 if by_ >= ay else ay - (ta.height + tb.height) / 2 - 10
        for tag, tx, ty, pop in tags_:
            sh_ = Image.new("RGBA", (tag.width + 8, tag.height + 8), (0, 0, 0, 0))
            ImageDraw.Draw(sh_).rectangle([8, 8, tag.width + 7, tag.height + 7], fill=(0, 0, 0, 140))
            paste(flat, sh_, tx + 4, ty + 4, scale=pop)                     # drop shadow: a printed sticker
            paste(flat, tag, tx, ty, scale=pop)
        return {"img": flat, "full": ph, "pa": pa, "pb": pb, "since": since, "land": land, "k": k, "R": R, "last": last,
                "shake": (sx, sy), "incut": incut, "i": i, "size": (cw, ch)}

    def draw_core(base, seg, tt, x0, y0, scale=1.0, frame=None):
        """Place the core photo window with its top-left at (x0, y0) (scaled), then Buh on top (never clipped).
        frame: optional callable(base, x0, y0, w, h) drawing the style's border/mount behind the photo.
        Returns the core dict plus screen-space pa/pb and the window rect."""
        c = link_core(seg, tt)
        sx, sy = c["shake"]
        cw, ch = c["size"]
        w, h = int(cw * scale), int(ch * scale)
        X0, Y0 = int(x0 + sx), int(y0 + sy)
        if frame:
            frame(base, X0, Y0, w, h)
        im = c["img"] if scale == 1.0 else c["img"].resize((w, h), Image.BILINEAR)
        base.alpha_composite(im, (X0, Y0))
        m = lambda p: (X0 + p[0] * scale, Y0 + p[1] * scale)   # noqa: E731
        pa, pb = m(c["pa"]), m(c["pb"])
        since, land, last = c["since"], c["land"], c["last"]
        if since < 0:
            bob = int(8 * abs(math.sin(tt * 7)))
            paste_pin(base, pin_sprite("happy", 0, 0, 1 if pb[0] > pa[0] else -1), pa[0], pa[1] - bob)
        elif since < THROW:
            x, y, ang = pixelpin_path(pa, pb, since / THROW)
            paste_pin(base, pin_sprite("shock", ang, -0.4), x, y)
        else:
            sq = 0.8 * math.exp(-land * 10) * math.cos(land * 30)
            ex = "shock" if land < 0.5 else ("happy" if last else "smug")
            hop = int(-abs(math.sin(land * 9)) * 30 * math.exp(-land * 5))
            if last and land > 0.6:
                hop = int(-abs(math.sin(land * 7)) * 40)  # celebrates on the final reveal
            paste_pin(base, pin_sprite(ex, 0, sq), pb[0], pb[1] + hop)
        c.update(spa=pa, spb=pb, rect=(X0, Y0, w, h))
        return c

    def core_finish(base, seg, tt, c, connected_y=1020, sub=True, chain=True, caption=True, credit_xy=None):
        """The shared payoff + chrome every style keeps: CONNECTED!, the Subscribe nudge, the title band and chain
        bar, the photo credit, the caption and the cut flashes. A style can move CONNECTED! or skip a piece it
        draws itself (chain=False if it has its own progress display)."""
        i, land, last, incut = c["i"], c["land"], c["last"], c["incut"]
        if last and land > 0.35:
            st = min(1.0, (land - 0.35) / 0.12)
            _ci = pixel_text("CONNECTED!", 72, fill=RED)
            paste(base, _ci, CX, connected_y, scale=min(2.0 - ease(st), (SAFE[2] - SAFE[0]) / _ci.width))
        if sub and last and land > 0.45:
            subscribe_button(base, land - 0.45)
        band(base, i)
        if chain:
            upto = i + (1 if land >= 0 else 0)
            blocks(base, upto, {upto} if land >= 0 else {i}, tt, sharpen=min(1.0, land / 0.35) if land >= 0 else 1.0)
        # the credit hugs the photo window the style actually drew (right edge, just under it), unless told otherwise
        X0, Y0, w_, h_ = c["rect"]
        cxy = credit_xy or (X0 + w_, Y0 + h_ + 32)
        if not credit_xy and cxy[1] > SAFE[3] - 4:          # no room under it: print it inside the photo's bottom edge
            cxy = (X0 + w_ - 12, Y0 + h_ - 18)
        credit(base, i, cxy)
        if caption:
            seg["cap"].draw(base, tt)
        if incut < 0.10:
            base.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(220 * (1 - incut / 0.10)))))
        if 0 <= land < 0.07:
            base.alpha_composite(Image.new("RGBA", (W, H), TONES[i % len(TONES)][0] + (int(120 * (1 - land / 0.07)),)))
        return base

    def frame_link(seg, tt):
        i = seg["i"]
        if C.fj["links"][i].get("lineage"):
            return frame_lineage(seg, tt)
        base = checker(tt, TONES[i % len(TONES)])

        def card_frame(b, X0, Y0, w, h):
            dr = ImageDraw.Draw(b)
            dr.rectangle([X0 - 14 + 16, Y0 - 14 + 16, X0 + w + 14 + 16, Y0 + h + 14 + 16], fill=INK)
            dr.rectangle([X0 - 14, Y0 - 14, X0 + w + 14, Y0 + h + 14], fill=WHITE)
        c = draw_core(base, seg, tt, CARD[0], CARD[1], 1.0, frame=card_frame)
        return core_finish(base, seg, tt, c)

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

    def hero(nm, h, fy, face=340, lift=False):
        """A full-width rectangular photo of nm's face (face ~`face` px tall, centred at (CX, fy)); gaps are a blur.
        lift=True applies the T1 grade."""
        key = (nm, h, fy, face, lift)
        if key not in _hero:
            img, box = C.hero_src[nm]
            ih, iw = img.shape[:2]
            if box is None:
                box = (iw * 0.3, ih * 0.15, iw * 0.4, iw * 0.4)
            x, y, bw, bh = box
            s_ = max(face / max(bh, bw * 1.1), W / iw, h / ih)      # cover: the photo always fills the band
            fx, fyy = x + bw / 2, y + bh * 0.45
            tx = min(0.0, max(W - iw * s_, CX - fx * s_))             # no gaps at the edges
            ty = min(0.0, max(h - ih * s_, fy - fyy * s_))
            M = np.float32([[s_, 0, tx], [0, s_, ty]])
            crop = cv2.warpAffine(img, M, (W, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            mask = cv2.warpAffine(np.full((ih, iw), 255, np.uint8), M, (W, h), flags=cv2.INTER_LINEAR)
            cover = max(W / iw, h / ih)                                   # blurred cover fill behind the photo
            bg = cv2.resize(img, (int(iw * cover) + 1, int(ih * cover) + 1))
            by0, bx0 = (bg.shape[0] - h) // 2, (bg.shape[1] - W) // 2
            bg = cv2.GaussianBlur(bg[by0:by0 + h, bx0:bx0 + W], (0, 0), 25) * 0.55
            m = (mask.astype(np.float32) / 255)[..., None]
            out = (crop * m + bg * (1 - m)).astype(np.uint8)
            if lift:
                out = t1_grade(out)
            _hero[key] = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).convert("RGBA")
            # where the head really is on screen (the cover/no-gap clamps can push the face off (CX, fy))
            _hero_head[key] = (fx * s_ + tx, y * s_ + ty - 0.12 * bh * s_)
        return _hero[key]

    _hero_head = {}

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
        av = _avatar(spec.get("request_avatar"))          # the viewer's real profile picture, when we have it
        if av is not None:
            mask = Image.new("L", (70, 70), 0)
            ImageDraw.Draw(mask).ellipse([0, 0, 69, 69], fill=255)
            im.paste(av, (22, 30), mask)
        else:
            d.ellipse([22, 30, 92, 100], fill=(120, 130, 150, 255))                         # default avatar
            d.ellipse([44, 44, 70, 70], fill=(225, 228, 235, 255))
            d.pieslice([32, 70, 82, 120], 180, 360, fill=(225, 228, 235, 255))
        d.text((112, 26), handle, font=fh, fill=(96, 96, 96))
        d.text((112, 64), text, font=ft, fill=(15, 15, 15))
        return im

    # owner: never "a viewer": the request card only appears with the real @handle (and their profile picture)
    comment_im = make_comment_card() if str(spec.get("requested_by", "")).startswith("@") else None

    # ---------- T1 opening: the static match-up band
    def _names(n, drop=()):
        stop = {"the", "and", "von", "van", "der", "mr.", "mrs."}
        toks = {w.lower() for w in re.findall(r"[A-Za-z][\w'.-]{2,}", f"{n} {C.labels.get(n, '')}")}
        return {w.rstrip(".") for w in toks - stop} - set(drop)

    def _spoken(words, names):
        for w, s0, _e in words:
            if re.sub(r"[^\w'-]", "", w).lower() in names:
                return s0
        return None

    _t1 = {}

    def t1_layout():
        """The band's text as parts: one line "HITLER → BIG BIRD?" if it fits at >= 92 px, else two lines
        ("HITLER →" / "COOKIE MONSTER?"), as big as fits (x 60..1020: the band sits above the like column)."""
        if "parts" in _t1:
            return _t1
        a_txt = (spec.get("people", {}).get(spec["from"], {}).get("short")
                 or short_name(spec["from"]).split()[-1]).upper()
        b_txt = C.labels[spec["to"]].upper() + "?"
        font = lambda s: ImageFont.truetype(str(FONTS / "Inter-Black.otf"), s)   # noqa: E731
        span = 960
        one = f"{a_txt} → {b_txt}"
        s1 = min(112, int(112 * span / font(112).getlength(one)))
        if s1 >= 92:
            lines, size = [[(a_txt, "a"), (" → ", None), (b_txt, "b")]], s1
        else:
            size = min(104, int(104 * span / max(font(104).getlength(f"{a_txt} →"), font(104).getlength(b_txt))))
            lines = [[(a_txt, "a"), (" →", None)], [(b_txt, "b")]]
        f = font(size)
        asc, desc = f.getmetrics()
        parts, n = [], len(lines)
        band_mid = (T1_BAND[0] + T1_BAND[1]) / 2
        for li, ln in enumerate(lines):
            widths = [f.getlength(s) for s, _ in ln]
            x = W / 2 - sum(widths) / 2
            cy = band_mid + (li - (n - 1) / 2) * size * 1.0 + size * 0.04
            for (s, role), w in zip(ln, widths):
                im = Image.new("RGBA", (int(w) + 8, asc + desc), (0, 0, 0, 0))
                ImageDraw.Draw(im).text((4, 0), s, font=f, fill=T1_INK)
                parts.append((im, x + w / 2, cy, role))     # Inter's cap centre ≈ the image centre
                x += w
        # band colour: yellow, or cyan when B's photo is mostly yellow (Big Bird); never red/black/white by Hitler
        bim = np.asarray(hero(spec["to"], T1_B[1] - T1_B[0], T1_B[2], T1_FACE[1], True).convert("RGB"))
        hsv = cv2.cvtColor(bim, cv2.COLOR_RGB2HSV)
        yellow = ((hsv[..., 0] >= 18) & (hsv[..., 0] <= 38) & (hsv[..., 1] > 110) & (hsv[..., 2] > 110)).mean()
        _t1.update(parts=parts, size=size, band=CYAN if yellow > 0.25 else YELLOW)
        return _t1

    def frame_hook_t1(seg, tt):
        kk = tt - seg["t0"]
        lay = t1_layout()
        base = Image.new("RGBA", (W, H), INK + (255,))
        # three elements only (A, the band, B): no Buh and no red string here. Every placement tried covered a face
        # or the band text, and the research says keep frame 0 to three elements and keep red away from Hitler.
        # Buh takes over from link 1 as usual.
        for nm, (y0, y1, fy), face, role in ((spec["from"], T1_A, T1_FACE[0], "a"),
                                             (spec["to"], T1_B, T1_FACE[1], "b")):
            im = hero(nm, y1 - y0, fy, face, True)          # frame 0 is complete: no slide-in, no shake
            if role == "b":     # one fast scale-in on B at 0.15-0.3 s, then still; a slow push-in after 0.7 s
                z = 1.0 + 0.06 * ease((kk - 0.15) / 0.15) + 0.04 * min(1.0, max(0.0, kk - 0.7) / 3.0)
            else:
                z = 1.0 + 0.04 * min(1.0, max(0.0, kk - 0.7) / 3.0)
            if z > 1.001:
                im = im.resize((int(W * z), int((y1 - y0) * z)), Image.BICUBIC)
                ox, oy = int(CX * (z - 1)), int(fy * (z - 1))
                im = im.crop((ox, oy, ox + W, oy + (y1 - y0)))
            base.alpha_composite(im, (0, y0))
        dr = ImageDraw.Draw(base)
        dr.rectangle([0, T1_BAND[0], W, T1_BAND[1]], fill=lay["band"])
        dr.rectangle([0, T1_BAND[0] - 5, W, T1_BAND[0]], fill=INK)
        dr.rectangle([0, T1_BAND[1], W, T1_BAND[1] + 5], fill=INK)
        words = seg.get("words") or []
        a_names = _names(spec["from"])
        t_a = _spoken(words, a_names)
        t_b = _spoken(words, _names(spec["to"], a_names))
        for im, cx, cy, role in lay["parts"]:        # the spoken name gets a small scale pop, never a colour change
            tw = t_a if role == "a" else t_b if role == "b" else None
            u = (kk - tw) / 0.22 if tw is not None else -1
            paste(base, im, cx, cy, scale=1.0 + 0.09 * math.sin(math.pi * u) if 0 <= u <= 1 else 1.0, nearest=False)
        return base

    def frame_hook(seg, tt):
        if T1:
            return frame_hook_t1(seg, tt)
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
                # both tags under their faces (B's used to sit above B's face, where Buh now lands)
                ty = y1 - tag.height // 2 - 24 if sgn < 0 else min(SAFE[3] + 60, y0 + fy + 210)
                paste(base, tag, CX + int(sx), ty + int(sy))
                # hero() puts the face box (brows..chin, ~340 px) centred at fy: its top is ~fy-150, the hair ~fy-300
                # Buh's pin tip goes into the hair at the top of the real face box (after the push-in zoom)
                hx, hy = _hero_head[(nm, y1 - y0, fy, 340, False)]
                hx, hy = hx * z - CX * (z - 1), hy * z - fy * (z - 1)
                spot = (int(hx) + slide + int(sx), int(max(y0 + (150 if sgn < 0 else 240), y0 + hy)) + int(sy))
                if sgn < 0:   # Buh starts on top of A's head...
                    a_spot = spot
                else:         # ...and jumps from head to head (owner)
                    b_spot = spot
            dr = ImageDraw.Draw(base)
            dr.rectangle([0, HERO_A[1], W, HERO_B[0]], fill=INK)               # the seam carries the question
            dr.rectangle([0, HERO_A[1] - 6, W, HERO_A[1]], fill=CYAN)
            dr.rectangle([0, HERO_B[0], W, HERO_B[0] + 6], fill=YELLOW)
            if comment_im is not None:   # the viewer's comment IS the hook (every reference hit opens on one)
                pop = ease_out_back(min(1.0, kk / 0.25))
                paste(base, comment_im, CX, (HERO_A[1] + HERO_B[0]) // 2 + 10, scale=0.6 + 0.4 * pop, nearest=False)
                dr = ImageDraw.Draw(base)
                ax = CX + comment_im.width // 2 - 70
                ay = (HERO_A[1] + HERO_B[0]) // 2 + comment_im.height // 2 + 10
                dr.line([(ax, ay), (ax + 30, ay + 70), (ax + 10, ay + 150)], fill=RED, width=14, joint="curve")
                dr.polygon([(ax - 18, ay + 132), (ax + 38, ay + 140), (ax + 6, ay + 190)], fill=RED)
                dr = ImageDraw.Draw(base)
            # owner: ~70 % stayed to watch, the opening sat still for 2.4 s. Now Buh jumps from A to B in the first
            # second and drags the red string across: the "how?" in motion, and the show's mechanic up front.
            J0, J1 = 0.55, 0.55 + THROW * 1.6
            if kk < J0:
                hop = abs(math.sin(kk * 9)) * 22
                paste_pin(base, pin_sprite("shock", 0, 0, 1), a_spot[0], a_spot[1] - hop)
            else:
                u = min(1.0, (kk - J0) / (J1 - J0))
                x, y, ang = pixelpin_path(a_spot, b_spot, u)
                pe = (x, y) if u < 1 else b_spot
                dr.line([a_spot, pe], fill=INK, width=14)
                dr.line([a_spot, pe], fill=RED, width=8)
                dr.rectangle([a_spot[0] - 10, a_spot[1] - 10, a_spot[0] + 10, a_spot[1] + 10], fill=RED, outline=INK, width=4)
                if u < 1:
                    paste_pin(base, pin_sprite("shock", ang, -0.4), x, y)
                else:
                    land = kk - J1
                    sq = 0.8 * math.exp(-land * 10) * math.cos(land * 30)
                    hop = abs(math.sin(land * 7)) * 18
                    paste_pin(base, pin_sprite("smug" if int(land * 2) % 2 else "happy", 0, sq), b_spot[0], b_spot[1] - hop)
            if comment_im is not None:
                return base               # Buh jumps OVER the comment card; the card is the question
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

    def framed(ph, tilt=0.0, border=16, tape=True):
        """A photo as a physical print: white border, thin dark keyline, tape on top, soft shadow, slight tilt."""
        w, h = ph.width + 2 * border, ph.height + 2 * border + 10
        card = Image.new("RGBA", (w + 40, h + 50), (0, 0, 0, 0))
        sh = Image.new("RGBA", card.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).rectangle([26, 34, 26 + w, 34 + h], fill=(0, 0, 0, 120))
        card.alpha_composite(sh.filter(ImageFilter.GaussianBlur(10)))
        d = ImageDraw.Draw(card)
        d.rectangle([14, 20, 14 + w, 20 + h], fill=(248, 246, 238, 255))
        card.alpha_composite(ph, (14 + border, 20 + border))
        d.rectangle([14 + border - 2, 20 + border - 2, 14 + border + ph.width + 1, 20 + border + ph.height + 1], outline=(40, 34, 30), width=2)
        if tape:
            tp = Image.new("RGBA", (150, 44), (236, 222, 170, 190))
            tp = tp.rotate(-6, expand=True, resample=Image.BICUBIC)
            card.alpha_composite(tp, (int(14 + w / 2 - tp.width / 2), 0))
        return card.rotate(tilt, expand=True, resample=Image.BICUBIC) if tilt else card

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

    def connected(base, seg, tt, y=1020):
        i = seg["i"]
        if i != n_links - 1:
            return
        land = tt - seg["t0"] - seg.get("rv", REVEAL) * seg["speech"] - THROW
        if land > 0.05:
            st = min(1.0, (land - 0.05) / 0.12)
            _ci = pixel_text("CONNECTED!", 72, fill=RED)
            paste(base, _ci, CX, y, scale=min(2.0 - ease(st), (SAFE[2] - SAFE[0]) / _ci.width))   # never clipped

    def year_of(i):
        l = spec["links"][i]
        m = re.search(r"\b(1\d{3}|20\d{2})\b", (l.get("fact") or "") + " " + l["line"])
        return int(m.group(1)) if m else None

    # --- zoom: the core, bigger. The photo window runs (almost) edge to edge at ~1.1x with only a thin keyline, over a
    # dark blurred copy of the same photo that slowly pans and breathes; everything else (camera, Buh, string, tags, payoff) is the shared core.
    Z_S = 1.08                                       # core window scale: 950 x 1026 (ends above the Subscribe button)
    Z_X0, Z_Y0 = (W - int((CARD[2] - CARD[0]) * Z_S)) // 2, 404
    _zbg = {}

    Z_OVER = 1.30                                    # the backdrop is this much bigger than the frame (room to pan)

    def z_bg(i):
        """A dark, heavily blurred, oversized copy of photo i (the backdrop around the window), kept at half
        resolution: it is blurred anyway, and the per-frame pan/zoom resamples it up to full frame."""
        if i not in _zbg:
            img = rgb(i)
            ih, iw = img.shape[:2]
            ow, oh = int(W * Z_OVER / 2), int(H * Z_OVER / 2)
            cs = max(ow / iw, oh / ih)
            b = cv2.resize(img, (int(iw * cs) + 2, int(ih * cs) + 2), interpolation=cv2.INTER_AREA)
            y0, x0 = (b.shape[0] - oh) // 2, (b.shape[1] - ow) // 2
            b = (cv2.GaussianBlur(b[y0:y0 + oh, x0:x0 + ow], (0, 0), 10) * 0.36).astype(np.uint8)
            _zbg[i] = Image.fromarray(b).convert("RGBA")
        return _zbg[i]

    def z_backdrop(i, tt):
        """The blurred copy never stands still: a slow Ken Burns drift across it plus a gentle breathing zoom
        (a few % over ~7 s). One bilinear resample of a half-res crop per frame."""
        bg = z_bg(i)
        bw, bh = bg.size
        s = 1.0 - 0.05 * (0.5 + 0.5 * math.sin(tt * 2 * math.pi / 7.0))        # breathe: up to 5 % closer
        vw, vh = W / 2 * s, H / 2 * s
        rx, ry = (bw - vw) / 2 * 0.92, (bh - vh) / 2 * 0.92                     # room left to pan
        cx = bw / 2 + rx * math.sin(tt * 0.31 + i * 1.7)
        cy = bh / 2 + ry * math.sin(tt * 0.23 + 0.8 + i * 2.3)
        return bg.resize((W, H), Image.BILINEAR, box=(cx - vw / 2, cy - vh / 2, cx + vw / 2, cy + vh / 2))

    def z_frame(b, X0, Y0, w, h):
        dr = ImageDraw.Draw(b)
        dr.rectangle([X0 - 6 + 10, Y0 - 6 + 10, X0 + w + 6 + 10, Y0 + h + 6 + 10], fill=INK)
        dr.rectangle([X0 - 6, Y0 - 6, X0 + w + 6, Y0 + h + 6], fill=WHITE)

    def z_follow(c):
        """Owner: "the background should move too when the image moves". The backdrop is the SAME camera frame as
        the window (link_core's full warp), mapped to where the window sits, then blurred and darkened, so it pans
        and zooms in lockstep with the photo, as if the window were a lens over one big picture."""
        full = c["full"]
        sw, sh = W // 4, H // 4                               # blur at quarter resolution (cheap, it's blurred anyway)
        # screen = (p - CARD.xy) * Z_S + (Z_X0, Z_Y0)  ->  sample full-frame point p for each backdrop pixel
        a = 1.0 / Z_S
        cx0 = CARD[0] - Z_X0 * a
        cy0 = CARD[1] - Z_Y0 * a
        src = full.convert("RGB").transform((sw, sh), Image.AFFINE, (a * 4, 0, cx0, 0, a * 4, cy0), Image.BILINEAR)
        src = src.filter(ImageFilter.GaussianBlur(6))
        src = Image.eval(src, lambda v: int(v * 0.42))
        return src.resize((W, H), Image.BILINEAR).convert("RGBA")

    def frame_zoom(seg, tt):
        i = seg["i"]
        if C.fj["links"][i].get("lineage"):
            return frame_lineage(seg, tt)
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        c = draw_core(layer, seg, tt, Z_X0, Z_Y0, Z_S, frame=z_frame)
        base = z_follow(c)
        base.alpha_composite(layer)
        return core_finish(base, seg, tt, c)

    # --- board: Buh pins each photo to a corkboard; red string from photo to photo; zoom out at the end
    BW, PW, PH = 1700, 600, 660
    bpos = [(BW / 2 + (-330 if j % 2 == 0 else 330), 560 + j * 620) for j in range(n_links)]
    BH = int(bpos[-1][1] + 760) if n_links else 1920
    _board = {}

    def polaroid(i, circled=True):
        card = Image.new("RGBA", (PW, PH), (250, 248, 240, 255))
        cam_ = both_cam(i, PW - 40, PH - 150, 0.8)
        card.alpha_composite(view(i, *cam_, PW - 40, PH - 150), (20, 20))
        if circled:                                      # red marker ring around the new person
            bx_, by_ = ctr(fbox(i, "b"))
            sx_, sy_ = (bx_ - cam_[0]) * cam_[2] + (PW - 40) / 2 + 20, (by_ - cam_[1]) * cam_[2] + (PH - 150) / 2 + 20
            rr = max(40, fbox(i, "b")[3] * cam_[2] * 0.75)
            dmk = ImageDraw.Draw(card)
            for o in range(3):
                dmk.ellipse([sx_ - rr * 1.1 - o, sy_ - rr - o, sx_ + rr * 1.1 + o, sy_ + rr + o], outline=(210, 20, 30, 255), width=5)
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
                im0 = Image.fromarray(np.clip(cork, 0, 255).astype(np.uint8)).convert("RGBA")
                dr0 = ImageDraw.Draw(im0)
                for _ in range(int(BH / 260)):                  # old clippings and scraps behind everything
                    cw, ch = int(rng.integers(160, 320)), int(rng.integers(120, 240))
                    clip = Image.new("RGBA", (cw, ch), (226, 220, 204, 255))
                    cd = ImageDraw.Draw(clip)
                    cd.rectangle([10, 10, cw - 10, 30], fill=(60, 56, 52))
                    for ly in range(44, ch - 10, 14):
                        cd.rectangle([10, ly, int(10 + (cw - 20) * rng.uniform(0.5, 1.0)), ly + 5], fill=(150, 144, 136))
                    clip = clip.rotate(float(rng.uniform(-12, 12)), expand=True, resample=Image.BICUBIC)
                    im0.alpha_composite(clip, (int(rng.integers(0, BW - 200)), int(rng.integers(0, BH - 200))))
                for _ in range(int(BH / 900) + 1):              # coffee rings
                    x, y, r = int(rng.integers(100, BW - 100)), int(rng.integers(100, BH - 100)), int(rng.integers(50, 80))
                    dr0.ellipse([x - r, y - r, x + r, y + r], outline=(110, 70, 36, 255), width=7)
                case = Image.new("RGBA", (760, 170), (250, 250, 244, 255))   # the case-file card at the top
                cd = ImageDraw.Draw(case)
                cd.rectangle([0, 0, 760, 30], fill=(200, 40, 40))
                fc = ImageFont.truetype(str(FONTS / "Inter-ExtraBold.otf"), 40)
                cd.text((380, 98), f"CASE: {C.labels[spec['from']].upper()} → {C.labels[spec['to']].upper()}"[:38], font=fc, fill=(30, 30, 30), anchor="mm")
                im0.alpha_composite(case.rotate(-2, expand=True, resample=Image.BICUBIC), (int(BW / 2 - 390), 40))
                yy, xx = np.mgrid[0:BH, 0:BW].astype(np.float32)   # warm edge lighting
                vig = 1 - 0.35 * (((xx - BW / 2) / (BW / 2)) ** 2)
                arr = np.array(im0).astype(np.float32)
                arr[..., :3] *= vig[..., None]
                _board[0] = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            else:
                im = board_layer(k - 1).copy()
                pc = polaroid(k - 1)
                x, y = bpos[k - 1]
                im.alpha_composite(pc, (int(x - pc.width / 2), int(y - pc.height / 2)))
                note = sticky(k - 1)                     # the sticky note with the year and fact
                nx = x + (PW / 2 + 40 if (k - 1) % 2 == 0 else -PW / 2 - 40)
                im.alpha_composite(note, (int(nx - note.width / 2), int(y + 40 - note.height / 2)))
                dr = ImageDraw.Draw(im)
                if k >= 2:
                    (x0, y0), (x1, y1) = bpos[k - 2], bpos[k - 1]
                    dr.line([(x0, y0 - PH / 2 + 20), (x1, y1 - PH / 2 + 20)], fill=(200, 20, 30), width=8)
                for j in range(k):
                    px, py = bpos[j]
                    dr.ellipse([px - 16, py - PH / 2 + 4, px + 16, py - PH / 2 + 36], fill=(220, 30, 40), outline=INK, width=4)
                _board[k] = im
        return _board[k]

    def sticky(i):
        sz = 300
        n = Image.new("RGBA", (sz, sz), (255, 226, 92, 255))
        d = ImageDraw.Draw(n)
        d.rectangle([0, 0, sz, 26], fill=(240, 206, 70, 255))
        fy = ImageFont.truetype(str(FONTS / "Inter-ExtraBold.otf"), 46)
        ft = ImageFont.truetype(str(FONTS / "Inter-Bold.otf"), 28)
        yr = year_of(i)
        d.text((sz / 2, 80), str(yr) if yr else "?", font=fy, fill=(40, 30, 20), anchor="mm")
        words, lines, cur_ = fact_of(spec["links"][i]).split(": ", 1)[-1].split(), [], ""
        for w in words:
            if ft.getlength(cur_ + " " + w) > sz - 40 and cur_:
                lines.append(cur_)
                cur_ = w
            else:
                cur_ = (cur_ + " " + w).strip()
        lines.append(cur_)
        for li, ln in enumerate(lines[:4]):
            d.text((sz / 2, 140 + li * 36), ln, font=ft, fill=(60, 50, 30), anchor="mm")
        return n.rotate(6 if i % 2 else -6, expand=True, resample=Image.BICUBIC)

    def board_view(layer, cx, cy, v):
        rw, rh = W / v, H / v
        if rw >= BW or rh >= BH:                         # fully zoomed out: show the whole board on cork
            out = Image.new("RGBA", (W, H), (120, 84, 50, 255))
            f = min(W / BW, H / BH)
            sm = layer.resize((int(BW * f), int(BH * f)), Image.BILINEAR)
            out.alpha_composite(sm, ((W - sm.width) // 2, (H - sm.height) // 2))
            return out
        cx = min(max(cx, rw / 2), BW - rw / 2)
        cy = min(max(cy, rh / 2), BH - rh / 2)
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
            vfit = min(W / BW, H / BH) * 0.999
            v = math.exp(math.log(1.25) + (math.log(vfit) - math.log(1.25)) * z)
            cx, cy = cx + (BW / 2 - cx) * z, cy + (BH / 2 - cy) * z
        layer = board_layer(i + 1 if (last and land > 0.25) else i)
        cx += 28 * math.sin(tt * 0.9)                    # the camera never sits still
        cy += 18 * math.cos(tt * 0.7)
        base = board_view(layer, cx, cy, v)
        to_s = lambda bx, by: ((bx - cx) * v + W / 2, (by - cy) * v + H / 2)
        if not (last and land > 0.25):
            drop = ease_out_back(min(1.0, local / 0.3))
            circled = local > R * sp + THROW
            pc = polaroid_cache.setdefault((i, circled), polaroid(i, circled))
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
        base = checker(tt * 6, ((28, 24, 52), (22, 18, 44)))
        bc_ = both_cam(i, 860, 720, 0.8)
        ph = view(i, bc_[0], bc_[1], bc_[2] * (1 + 0.06 * min(1.0, local / 3)), 860, 720)   # slow push-in
        card = framed(ph, tilt=1.5 * math.sin(local * 1.5) + (-1.5 if i % 2 else 1.5))
        slide_in = 1 - ease(min(1.0, local / 0.35))      # slides in from the right...
        left = seg["t1"] - tt
        slide_out = 1 - ease(min(1.0, left / 0.25)) if i < n_links - 1 else 0.0   # ...and out to the left
        paste(base, card, CX + W * 0.9 * slide_in - W * 0.9 * slide_out, 690 + 12 * math.sin(local * 2.3), nearest=False)
        an, bn = link_names(i)
        sp, R = seg["speech"], seg.get("rv", REVEAL)
        paste(base, pixel_tag(C.labels[an].upper(), CYAN, 28), CX - 220, 1030)
        if local > R * sp + THROW:
            paste(base, pixel_tag(C.labels[bn].upper(), YELLOW, 28), CX + 220, 1030)
        y_now = years[i] or ymin
        y_prev = (years[i - 1] or y_now) if i else y_now
        u = ease(min(1.0, local / 0.6))
        yv = y_prev + (y_now - y_prev) * u
        rolling = u < 1 and y_prev != y_now
        jx, jy = (shake(local, 10, 0.6) if rolling else (0, 0))
        landed = local - 0.6
        punch = 1.0 + (0.35 * math.exp(-landed * 9) if landed > 0 and y_prev != y_now else 0.0)
        paste(base, pixel_text(str(int(round(yv))), 110, fill=YELLOW), CX + jx, 1240 + jy, scale=punch)
        dr = ImageDraw.Draw(base)
        dr.rectangle([RX0, RY - 5, RX1, RY + 5], fill=WHITE)
        step = 10 if ymax - ymin <= 120 else (50 if ymax - ymin <= 600 else 250)
        for yy in range(int(ymin // step * step), int(ymax) + step, step):
            x = rx(yy)
            if RX0 <= x <= RX1:
                dr.rectangle([x - 2, RY - 16, x + 2, RY + 16], fill=WHITE)
        mx = rx(yv)                                     # the marker slides along the ruler
        dr.polygon([(mx - 18, RY + 30), (mx + 18, RY + 30), (mx, RY + 8)], fill=YELLOW, outline=INK)
        for j in range(i + 1):                        # faces collected on the ruler, bobbing
            if years[j] is None:
                continue
            xj = rx(years[j]) if j < i else mx
            th = sharp[C.people[j + 1]] if j + 1 < len(C.people) else None
            if th is not None:
                bob = 6 * math.sin(tt * 4 + j)
                paste(base, th.resize((64, 64), Image.BICUBIC), xj, RY - 52 - (j % 2) * 30 + bob, nearest=False)
        header(base, i)
        connected(base, seg, tt)
        seg["cap"].draw(base, tt, y=1150)
        return base

    # --- shared Earth imagery (NASA "Whole world - land and oceans", public domain)
    _map = {}

    def earth_tex():
        """NASA 'Whole world - land and oceans' (public domain), as an RGB array (equirectangular)."""
        if "tex" not in _map:
            _map["tex"] = np.array(Image.open(FONTS.parent / "earth_texture.jpg").convert("RGB"))
        return _map["tex"]

    # --- globe: the core photo window on top, over a starfield; below it a pixel-art Earth fills the lower screen.
    # At the start of each link the view flies from the last place to this one (a red dot hops along a dotted route,
    # the view frames both ends and zooms by distance), then the place is marked and named. The core stays the hero.
    G_Y0 = 1150                                         # map top (feathered); it runs to the bottom
    G_CY = 1610                                         # screen y of the view centre
    G_BOX = (330, 105)                                  # both ends of a hop stay within this many px of the centre
    G_R1 = 450                                          # globe radius on screen at zoom 1
    G_PX, G_M = 2, 24                                   # map pixel size; margin so the float never shows an edge
    G_FLY = 1.2                                         # seconds of flight at the start of a link
    G_S = 0.8                                           # core window scale (704 x 760)
    G_X0, G_WY0 = int(CX - (CARD[2] - CARD[0]) * G_S / 2), 425
    # zoom cap: one texel of the 3840 px Earth texture (3840 / 2pi px per radian) stays <= ~4 screen px (= 2 map pixels)
    G_ZMAX = 4.0 * (3840 / (2 * math.pi)) / G_R1
    _gxs = np.arange(-G_M, W + G_M, G_PX, dtype=np.float32) + G_PX / 2 - CX
    _gys = np.arange(G_Y0 - G_M, H + G_M, G_PX, dtype=np.float32) + G_PX / 2 - G_CY
    _ggx, _ggy = np.meshgrid(_gxs, _gys)
    _grho = np.sqrt(_ggx ** 2 + _ggy ** 2)
    _gfeather = np.clip((np.arange(_gys.size) * G_PX - G_M) / 70.0, 0, 1)[:, None]   # top edge fades in
    _gst = {}

    def ll(i):
        w = spec["links"][i].get("where") or {}
        return (w["lat"], w["lon"]) if "lat" in w else None

    def _unit(p):
        la, lo = math.radians(p[0]), math.radians(p[1])
        return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])

    def slerp_ll(a, b, u):
        p1, p2 = _unit(a), _unit(b)
        om = math.acos(max(-1.0, min(1.0, float(p1 @ p2))))
        if om < 1e-6:
            return a
        q = (math.sin((1 - u) * om) * p1 + math.sin(u * om) * p2) / math.sin(om)
        return math.degrees(math.asin(max(-1.0, min(1.0, q[2])))), math.degrees(math.atan2(q[1], q[0]))

    def ang_deg(a, b):
        return math.degrees(math.acos(max(-1.0, min(1.0, float(_unit(a) @ _unit(b))))))

    def prev_ll(i):
        return next((ll(j) for j in range(i - 1, -1, -1) if ll(j)), None)

    def g_fit(ctr_, pts):
        """The largest zoom (<= G_ZMAX) that keeps every point within G_BOX of the view centre, on the near side."""
        z = G_ZMAX
        for pt in pts:
            if ang_deg(ctr_, pt) > 80:
                return 1.0
            q = g_xy(pt, ctr_, 1.0)
            dx, dy = abs(q[0] - CX), abs(q[1] - G_CY)
            z = min(z, G_BOX[0] / max(dx, 1e-6), G_BOX[1] / max(dy, 1e-6))
        return max(1.0, z)

    def g_view(i):
        """Where the view settles for link i: (centre lat/lon, zoom), framed between the last place and this one."""
        if ("v", i) in _gst:
            return _gst[("v", i)]
        cur, p = ll(i), prev_ll(i)
        if p is None:
            v = (cur, 2.2)
        elif ang_deg(p, cur) < 0.05:
            v = (cur, G_ZMAX)                           # same place again: push in as far as the texture allows
        else:
            c_ = slerp_ll(p, cur, 0.5)
            v = (c_, g_fit(c_, [p, cur]))
        _gst[("v", i)] = v
        return v

    def g_state(i, local):
        """(view centre, zoom, Buh's position along the hop, flight progress) at `local` seconds into link i."""
        cur, p = ll(i), prev_ll(i)
        fly = ease(min(1.0, local / G_FLY))
        vn = g_view(i)
        if p is None:                                   # first place: settle in from a little further out
            return vn[0], vn[1] * (0.7 + 0.3 * fly), cur, 1.0
        vp = g_view(next(j for j in range(i - 1, -1, -1) if ll(j)))
        lzp, lzn = math.log(vp[1]), math.log(vn[1])
        mid_c = slerp_ll(vp[0], vn[0], 0.5)            # pull out mid-flight enough to see both ends of the hop
        bump = max(0.1, (lzp + lzn) / 2 - math.log(g_fit(mid_c, [p, cur]))) if ang_deg(p, cur) > 0.05 else 0.06
        z = math.exp(lzp + (lzn - lzp) * fly - bump * math.sin(math.pi * fly))
        return slerp_ll(vp[0], vn[0], fly), z, slerp_ll(p, cur, fly), fly

    def g_xy(pt, ctr_, z):
        """Screen position of pt on the globe facing ctr_ at zoom z, or None on the far side."""
        la, lo = math.radians(pt[0]), math.radians(pt[1])
        la0, lo0 = math.radians(ctr_[0]), math.radians(ctr_[1])
        cosc = math.sin(la0) * math.sin(la) + math.cos(la0) * math.cos(la) * math.cos(lo - lo0)
        if cosc < 0.02:
            return None
        R = G_R1 * z
        x = R * math.cos(la) * math.sin(lo - lo0)
        y = R * (math.cos(la0) * math.sin(la) - math.sin(la0) * math.cos(la) * math.cos(lo - lo0))
        return CX + x, G_CY - y

    def g_map(ctr_, z):
        """The map window (RGBA, W+2M wide, from G_Y0-M down): the globe facing ctr_ at zoom z, on transparent."""
        tex = earth_tex()
        lh, lw = tex.shape[:2]
        R = G_R1 * z
        la0, lo0 = math.radians(ctr_[0]), math.radians(ctr_[1])
        inside = _grho < R
        c = np.arcsin(np.clip(_grho / R, 0, 1))
        sc, cc = np.sin(c), np.cos(c)
        y = -_ggy
        rho = np.maximum(_grho, 1e-6)
        lat = np.arcsin(np.clip(cc * math.sin(la0) + y * sc * math.cos(la0) / rho, -1, 1))
        lon = lo0 + np.arctan2(_ggx * sc, rho * cc * math.cos(la0) - y * sc * math.sin(la0))
        px = ((np.degrees(lon) + 180) % 360 / 360 * lw).astype(np.int32).clip(0, lw - 1)
        py = ((90 - np.degrees(lat)) / 180 * lh).astype(np.int32).clip(0, lh - 1)
        shade = (0.45 + 0.55 * cc)[..., None]           # darker towards the rim
        img = np.zeros(_grho.shape + (4,), np.uint8)
        img[..., :3] = (tex[py, px].astype(np.float32) * shade).astype(np.uint8)
        img[..., 3] = (np.where(inside, 255, 0) * _gfeather).astype(np.uint8)
        rim = inside & (_grho > R - 2.2 * G_PX)
        img[rim, :3] = (16, 12, 30)
        return Image.fromarray(img).resize((img.shape[1] * G_PX, img.shape[0] * G_PX), Image.NEAREST)

    def g_stars():
        if "stars" not in _gst:
            rng = np.random.default_rng(3)
            st = np.zeros((H, W, 3), np.uint8)
            st[:] = (10, 8, 24)
            for _ in range(260):
                x, y = rng.integers(0, W), rng.integers(0, H)
                st[y:y + 3, x:x + 3] = (200, 200, 230) if rng.random() < 0.8 else (255, 214, 10)
            _gst["stars"] = Image.fromarray(st).convert("RGBA")
        return _gst["stars"]

    def g_frame(b, X0, Y0, w, h):
        """A slim white print border with a hard shadow, like the cards."""
        dr = ImageDraw.Draw(b)
        dr.rectangle([X0 - 12 + 14, Y0 - 12 + 14, X0 + w + 12 + 14, Y0 + h + 12 + 14], fill=INK)
        dr.rectangle([X0 - 12, Y0 - 12, X0 + w + 12, Y0 + h + 12], fill=WHITE)

    _G_SHOW = re.compile(r"(?i)\bshow\b|tonight|late night|\blive\b|\btv\b|network|records|studios?\b|\bcbs\b|\bnbc\b|\babc\b")
    _G_ABBR = {"NYC": "New York", "NY": "New York", "DC": "Washington", "LA": "Los Angeles", "UK": "London"}

    def g_place(place):
        """A map label is a place only: drop show/brand parts ("Late Show, NYC" -> "New York"), expand city codes."""
        parts = [q.strip() for q in (place or "").split(",") if q.strip()]
        keep = [q for q in parts if not _G_SHOW.search(q)]
        if len(keep) == 1:
            keep = [_G_ABBR.get(keep[0].upper().replace(".", ""), keep[0])]
        return ", ".join(keep)

    def g_drift(tt, z):
        """The globe never sits still: a slow turn (longitude) and a gentle nod (latitude), a fixed few screen px
        whatever the zoom, so it reads as a slowly turning planet when far out and a soft drift when close in."""
        k = 1.0 / (G_R1 * z)                                    # radians per screen px at this zoom
        return (math.degrees(18 * k * math.sin(tt * 0.31 + 0.7)),
                math.degrees(46 * k * math.sin(tt * 0.23)) + math.degrees(10 * k) * tt / 6)

    def g_stars_at(tt):
        """The starfield, panning slowly sideways (wraps), so space drifts too."""
        st = g_stars()
        off = int(tt * 7) % W
        if off == 0:
            return st.copy()
        out = Image.new("RGBA", (W, H))
        out.paste(st.crop((off, 0, W, H)), (0, 0))
        out.paste(st.crop((0, 0, off, H)), (W - off, 0))
        return out

    def g_route(dr, a, b, vc, z, fx, fy, upto, on_map, r=4, fill=WHITE, dim=None):
        """A dotted great-circle route a -> b: dots up to fraction `upto` in `fill`, the rest in `dim` (None: skip).
        Returns the screen points of the dots drawn."""
        nd = max(3, min(220, int(G_R1 * z * math.radians(ang_deg(a, b)) / 26)))
        pts = []
        for d_ in range(nd + 1):
            u = d_ / nd
            col = fill if u <= upto + 1e-6 else dim
            if col is None:
                break
            q = g_xy(slerp_ll(a, b, u), vc, z)
            if on_map(q):
                x, y = q[0] + fx, q[1] + fy
                dr.rectangle([x - r, y - r, x + r, y + r], fill=col, outline=INK, width=1)
                pts.append((x, y))
        return pts

    def g_label_spot(i, tw, th):
        """Where link i's place label goes relative to its dot: the side (right, left, below, above) that covers the
        fewest route dots and visited places in the settled view. Chosen once per link so it never flips."""
        key = ("lbl", i)
        if key in _gst:
            return _gst[key]
        cur, p = ll(i), prev_ll(i)
        vc, z, _, _ = g_state(i, G_FLY)
        dq = g_xy(cur, vc, z)
        obst = []
        if p is not None and ang_deg(p, cur) > 0.05:
            nd = max(3, min(220, int(G_R1 * z * math.radians(ang_deg(p, cur)) / 26)))
            obst += [g_xy(slerp_ll(p, cur, d_ / nd), vc, z) for d_ in range(nd)]
        obst += [g_xy(ll(j), vc, z) for j in range(i) if ll(j)]
        obst = [q for q in obst if q is not None]
        cands = [(24 + tw / 2, -8), (-24 - tw / 2, -8), (0, 34), (0, -44)]
        best, spot = None, cands[0]
        for k, (ox, oy) in enumerate(cands):
            x = max(SAFE[0] + tw / 2, min(SAFE[2] - tw / 2, dq[0] + ox)) if dq else ox
            y = (dq[1] if dq else 0) + oy
            hits = sum(1 for q in obst if abs(q[0] - x) < tw / 2 + 10 and abs(q[1] - y) < th / 2 + 10)
            pushed = abs(x - ((dq[0] if dq else 0) + ox)) > 40          # clamped far off its dot
            score = hits * 10 + (25 if pushed else 0) + k
            if best is None or score < best:
                best, spot = score, (ox, oy)
        _gst[key] = spot
        return spot

    def frame_globe(seg, tt):
        i = seg["i"]
        if C.fj["links"][i].get("lineage"):
            return frame_lineage(seg, tt)
        local = tt - seg["t0"]
        base = g_stars_at(tt)
        if ll(i) is not None:
            cur, p = ll(i), prev_ll(i)
            vc, z, travel, fly = g_state(i, local)
            dla, dlo = g_drift(tt, z)                    # slow turn + nod of the planet
            vc = (max(-85.0, min(85.0, vc[0] + dla)), vc[1] + dlo)
            fx, fy = int(10 * math.sin(tt * 0.8)), int(12 * math.sin(tt * 0.6 + 1))   # gentle float
            layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            layer.alpha_composite(g_map(vc, z), (-G_M + fx, G_Y0 - G_M + fy))
            dr = ImageDraw.Draw(layer)

            def on_map(q):
                return q is not None and -20 < q[0] < W + 20 and G_Y0 + 30 < q[1] < H + 20

            # the journey so far: earlier hops stay on the globe as faint small dots
            placed = [j for j in range(i) if ll(j)]
            for j0, j1 in itertools.pairwise(placed):
                if ang_deg(ll(j0), ll(j1)) > 0.05:
                    g_route(dr, ll(j0), ll(j1), vc, z, fx, fy, 1.0, on_map, r=2, fill=(150, 150, 190))
            # this hop's dotted route: the whole way is there from the first frame of the link (dim ahead of the
            # plane-dot, white behind it) and stays for the whole link
            moved = p is not None and ang_deg(p, cur) > 0.05
            if moved:
                g_route(dr, p, cur, vc, z, fx, fy, fly, on_map, fill=WHITE, dim=(110, 110, 150))
            for j in range(i + 1):                      # places visited so far (the current one only on arrival)
                if ll(j) and (j < i or fly > 0.96 or p is None):
                    q = g_xy(ll(j), vc, z)
                    if on_map(q):
                        r = 9 if j == i else 6
                        dr.rectangle([q[0] + fx - r, q[1] + fy - r, q[0] + fx + r, q[1] + fy + r], fill=RED, outline=INK, width=3)
            if moved and fly < 1:                    # the travelling marker: a little red plane-dot with a hop
                q = g_xy(travel, vc, z)
                if on_map(q):
                    hop = math.sin(math.pi * fly) * 40
                    x, y = q[0] + fx, q[1] + fy - hop
                    dr.ellipse([x - 12, y - 12, x + 12, y + 12], fill=RED, outline=INK, width=4)
            dest = g_xy(cur, vc, z)
            place = before_b(g_place((spec["links"][i].get("where") or {}).get("place", "")), spec["links"][i])
            if fly >= 1 and dest and place.strip("…"):
                tg = pixel_tag(place.upper(), WHITE, 20)
                dx_, dy_ = dest[0] + fx, dest[1] + fy
                ox, oy = g_label_spot(i, tg.width, tg.height)
                lx = max(SAFE[0] + tg.width / 2, min(SAFE[2] - tg.width / 2, dx_ + ox))
                ly = dy_ + oy
                paste(layer, tg, lx, ly)
            base.alpha_composite(layer)
        c = draw_core(base, seg, tt, G_X0, G_WY0, G_S, frame=g_frame)
        return core_finish(base, seg, tt, c)

    style_frames = {"globe": frame_globe, "cards": frame_link, "zoom": frame_zoom, "board": frame_board, "timeline": frame_timeline,
}
    if STYLE not in style_frames:                        # plug-in styles (sixdeg/styles/<name>.py)
        from . import styles as _styles
        style_frames.update(_styles.load(_styles.build_ctx(locals(), globals())))
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
        if not T1:                       # T1's band already asks the question
            q = pixel_text("?", 170, fill=YELLOW)                   # on the seam, between the two faces
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
