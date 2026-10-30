"""Render a connection chain as a vertical video.

Editing style follows the ioctl multi-image animator: each photo starts zoomed in
on the person shared with the previous photo, zooms out, pans to the other person,
zooms back in on them, then an ffmpeg-style "distance" transition cuts to the next
photo, which starts zoomed in on that same person. Because the zoom is normalised
to face size (and the eyes levelled), the shared face lines up across the cut.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import textwrap
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .faces import Face
from .util import load_image, log

ASSETS = Path(__file__).parent / "assets"
DEFAULT_FONT = ASSETS / "Anton-Regular.ttf"


@dataclass
class Shot:
    """One photo in the chain: `a` is the person carried over from the previous photo, `b` the next one."""

    image: Path
    a_name: str
    b_name: str
    a: Face
    b: Face


@dataclass
class Style:
    width: int = 1080
    height: int = 1920
    fps: int = 60
    # Per-photo timing (seconds), same defaults as the multi-animator.
    hold: float = 0.05
    zoom_out: float = 0.9
    pan: float = 1.0
    zoom_in: float = 0.9
    transition: float = 0.5
    transition_kind: str = "distance"  # "distance" (ffmpeg xfade=distance) or "fade"
    intro_hold: float = 0.6  # extra time on the first face so the title can be read
    outro_hold: float = 0.8  # extra time on the final face
    easing: str = "smooth"  # "smooth" or "linear"
    # Framing: in the close-up, the face box is this fraction of the frame width.
    face_width: float = 0.5
    zoom_ratio: float = 2.0  # close-up zoom / panning zoom (multi-animator: 6x -> 3x)
    min_fill: float = 0.6  # while panning, keep the photo at least this tall on screen when possible
    max_upscale: float = 10.0  # never magnify source pixels more than this
    level_eyes: bool = True
    labels: bool = True
    label_color: tuple = (255, 255, 255)
    title: Optional[str] = None
    title_color: tuple = (240, 255, 0)
    font: Path = DEFAULT_FONT
    background: str = "blur"  # "blur" or "black"
    speed: float = 1.0

    def scaled(self, seconds: float) -> float:
        return seconds / max(self.speed, 1e-3)


@dataclass
class Camera:
    """Image point `c` is shown at the frame centre, magnified `z` times and rotated by `theta`."""

    c: np.ndarray
    z: float
    theta: float = 0.0

    def matrix(self, w: int, h: int) -> np.ndarray:
        cos, sin = math.cos(self.theta) * self.z, math.sin(self.theta) * self.z
        a = np.array([[cos, -sin], [sin, cos]])
        t = np.array([w / 2, h / 2]) - a @ self.c
        return np.hstack([a, t[:, None]])

    def to_screen(self, p, w: int, h: int) -> np.ndarray:
        return self.matrix(w, h) @ np.array([p[0], p[1], 1.0])


def _ease(t: float, kind: str) -> float:
    t = min(1.0, max(0.0, t))
    return t if kind == "linear" else t * t * (3 - 2 * t)


def interpolate(c0: Camera, c1: Camera, t: float, focus=None, w: int = 0, h: int = 0) -> Camera:
    """Blend two cameras. Zoom is interpolated geometrically; with a focus point, that point
    travels in a straight line on screen (a natural "zoom into the face" move)."""
    z = math.exp((1 - t) * math.log(c0.z) + t * math.log(c1.z))
    theta = (1 - t) * c0.theta + t * c1.theta
    if focus is None:
        return Camera((1 - t) * c0.c + t * c1.c, z, theta)
    p = np.asarray(focus, float)
    s = (1 - t) * c0.to_screen(p, w, h) + t * c1.to_screen(p, w, h)
    cos, sin = math.cos(theta), math.sin(theta)
    rot_inv = np.array([[cos, sin], [-sin, cos]])
    c = p + rot_inv @ ((np.array([w / 2, h / 2]) - s) / z)
    return Camera(c, z, theta)


@dataclass
class _Phase:
    duration: float
    cam0: Camera
    cam1: Camera
    focus: Optional[tuple] = None


@dataclass
class _Clip:
    shot: Shot
    src: list  # image pyramid (RGB), level k is downscaled by 2**k
    bg: np.ndarray
    phases: list
    start: float = 0.0
    labels: list = field(default_factory=list)  # (name, face)

    @property
    def duration(self) -> float:
        return sum(p.duration for p in self.phases)

    @property
    def end(self) -> float:
        return self.start + self.duration


class Renderer:
    def __init__(self, style: Style):
        self.style = style
        self.W, self.H = style.width, style.height

    # ---------- framing ----------
    def _close_up(self, face: Face, img_w: int, img_h: int) -> Camera:
        z = self.style.face_width * self.W / max(face.w, 1.0)
        z = min(z, self.style.max_upscale)
        theta = 0.0
        if self.style.level_eyes and face.landmarks:
            theta = -max(-0.2, min(0.2, face.roll))
        return Camera(np.array(face.center, float), z, theta)

    def _wide(self, face: Face, z: float, img_w: int, img_h: int) -> Camera:
        """Pan framing: centred on the face, but kept inside the photo where the photo is big enough."""
        c = np.array(face.center, float)
        half = np.array([self.W, self.H]) / (2 * z)
        for k, size in enumerate((img_w, img_h)):
            if 2 * half[k] >= size:
                c[k] = size / 2
            else:
                c[k] = min(max(c[k], half[k]), size - half[k])
        return Camera(c, z, 0.0)

    def _build_clip(self, shot: Shot, first: bool, last: bool) -> _Clip:
        st = self.style
        bgr = load_image(shot.image)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        ih, iw = rgb.shape[:2]
        tight_a, tight_b = self._close_up(shot.a, iw, ih), self._close_up(shot.b, iw, ih)
        fit = min(self.W / iw, self.H / ih)
        z_mid = math.sqrt(tight_a.z * tight_b.z) / st.zoom_ratio
        z_mid = max(z_mid, fit, st.min_fill * self.H / ih)
        z_mid = min(z_mid, tight_a.z, tight_b.z)
        wide_a, wide_b = self._wide(shot.a, z_mid, iw, ih), self._wide(shot.b, z_mid, iw, ih)
        phases = [
            _Phase(st.scaled(st.hold + (st.intro_hold if first else 0.0)), tight_a, tight_a),
            _Phase(st.scaled(st.zoom_out), tight_a, wide_a, shot.a.center),
            _Phase(st.scaled(st.pan), wide_a, wide_b),
            _Phase(st.scaled(st.zoom_in), wide_b, tight_b, shot.b.center),
            _Phase(st.scaled(st.hold + (st.outro_hold if last else 0.0)), tight_b, tight_b),
        ]
        return _Clip(shot, _pyramid(rgb), self._background(rgb), phases,
                     labels=[(shot.a_name, shot.a), (shot.b_name, shot.b)])

    def _background(self, rgb: np.ndarray) -> np.ndarray:
        import cv2

        if self.style.background == "black":
            return np.zeros((self.H, self.W, 3), np.uint8)
        ih, iw = rgb.shape[:2]
        cover = max(self.W / iw, self.H / ih)
        small = cv2.resize(rgb, (max(1, int(iw * cover / 8)), max(1, int(ih * cover / 8))), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 6)
        big = cv2.resize(small, (int(math.ceil(iw * cover)), int(math.ceil(ih * cover))), interpolation=cv2.INTER_LINEAR)
        y0, x0 = (big.shape[0] - self.H) // 2, (big.shape[1] - self.W) // 2
        return (big[y0:y0 + self.H, x0:x0 + self.W].astype(np.float32) * 0.45).astype(np.uint8)

    # ---------- per-frame ----------
    def _camera_at(self, clip: _Clip, t: float) -> Camera:
        for ph in clip.phases:
            if t <= ph.duration or ph is clip.phases[-1]:
                u = _ease(t / ph.duration if ph.duration > 0 else 1.0, self.style.easing)
                return interpolate(ph.cam0, ph.cam1, u, ph.focus, self.W, self.H)
            t -= ph.duration
        raise AssertionError("unreachable")

    def _frame(self, clip: _Clip, t: float) -> np.ndarray:
        import cv2

        cam = self._camera_at(clip, t)
        level = min(len(clip.src) - 1, max(0, int(math.floor(math.log2(1.0 / cam.z))))) if cam.z < 1 else 0
        k = 2 ** level
        m = Camera(cam.c / k, cam.z * k, cam.theta).matrix(self.W, self.H)
        frame = clip.bg.copy()
        cv2.warpAffine(clip.src[level], m, (self.W, self.H), dst=frame, flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_TRANSPARENT)
        if self.style.labels:
            self._draw_labels(frame, clip, cam)
        return frame

    def _draw_labels(self, frame: np.ndarray, clip: _Clip, cam: Camera) -> None:
        placed = []
        m = cam.matrix(self.W, self.H)
        for name, face in sorted(clip.labels, key=lambda nf: nf[1].center[1]):
            cx, cy = m @ np.array([face.center[0], face.center[1], 1.0])
            if not (0 <= cx <= self.W and 0 <= cy <= self.H):
                continue  # that person is off screen right now
            face_px = face.w * cam.z
            size = int(max(0.032 * self.W, min(0.11 * self.W, 0.24 * face_px)) // 2 * 2)
            sprite = _text_sprite(name.upper(), size, self.style.label_color, str(self.style.font))
            sh, sw = sprite.shape[:2]
            below = m @ np.array([face.center[0], face.y + face.h * 1.12, 1.0])
            x, y = below[0] - sw / 2, below[1]
            rect = (x, y, x + sw, y + sh)
            if any(_overlaps(rect, r) for r in placed) or y + sh > self.H * 0.97:
                above = m @ np.array([face.center[0], face.y - face.h * 0.12, 1.0])
                x, y = above[0] - sw / 2, above[1] - sh
                rect = (x, y, x + sw, y + sh)
            x = min(max(x, self.W * 0.03), self.W * 0.97 - sw)
            if -sh < y < self.H:
                _blit(frame, sprite, int(round(x)), int(round(y)))
                placed.append((x, y, x + sw, y + sh))

    def _draw_title(self, frame: np.ndarray) -> None:
        if not self.style.title:
            return
        size = int(self.W * 0.075)
        chars = max(10, int(self.W * 0.9 / (size * 0.42)))
        y = int(self.H * 0.07)
        for line in textwrap.wrap(self.style.title.upper(), chars)[:3]:
            sprite = _text_sprite(line, size, self.style.title_color, str(self.style.font))
            _blit(frame, sprite, (self.W - sprite.shape[1]) // 2, y)
            y += int(sprite.shape[0] * 0.95)

    def _transition(self, f0: np.ndarray, f1: np.ndarray, q: float) -> np.ndarray:
        """q goes 0 -> 1. "distance" reproduces ffmpeg's xfade=transition=distance."""
        if self.style.transition_kind == "fade":
            return (f0 * (1 - q) + f1 * q).astype(np.uint8)
        import cv2

        progress = 1.0 - q  # ffmpeg counts down from 1 to 0
        y0 = cv2.cvtColor(f0, cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0
        y1 = cv2.cvtColor(f1, cv2.COLOR_RGB2YUV).astype(np.float32) / 255.0
        dist = np.sqrt(((y0 - y1) ** 2).sum(axis=2, keepdims=True))
        inner = np.where(dist <= progress, f0, f1).astype(np.float32)
        return (inner * progress + f1.astype(np.float32) * (1 - progress) + 0.5).astype(np.uint8)

    # ---------- driver ----------
    def render(self, shots: list[Shot], out_path: Path, music: Optional[Path] = None,
               progress: Optional[Callable[[int, int], None]] = None) -> Path:
        if not shots:
            raise ValueError("nothing to render")
        st = self.style
        clips = [self._build_clip(s, i == 0, i == len(shots) - 1) for i, s in enumerate(shots)]
        overlap = st.scaled(st.transition)
        t = 0.0
        for clip in clips:
            clip.start = t
            t = clip.end - min(overlap, clip.duration / 2)
        total = clips[-1].end
        n_frames = max(1, int(round(total * st.fps)))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = FFmpegWriter(out_path, self.W, self.H, st.fps, music=music, duration=total)
        try:
            for k in range(n_frames):
                now = min(k / st.fps, total - 1e-6)
                active = [c for c in clips if c.start <= now < c.end] or [clips[-1]]
                if len(active) >= 2:
                    c0, c1 = active[0], active[1]
                    q = (now - c1.start) / max(c0.end - c1.start, 1e-6)
                    frame = self._transition(self._frame(c0, now - c0.start), self._frame(c1, now - c1.start), q)
                else:
                    frame = self._frame(active[0], now - active[0].start)
                self._draw_title(frame)
                writer.write(frame)
                if progress:
                    progress(k + 1, n_frames)
        finally:
            writer.close()
        log.info("Wrote %s (%.1fs, %d photos)", out_path, total, len(shots))
        return out_path


def _pyramid(rgb: np.ndarray, levels: int = 4) -> list:
    import cv2

    out = [rgb]
    for _ in range(levels - 1):
        h, w = out[-1].shape[:2]
        if min(h, w) < 64:
            break
        out.append(cv2.resize(out[-1], (w // 2, h // 2), interpolation=cv2.INTER_AREA))
    return out


def _overlaps(r1, r2) -> bool:
    return not (r1[2] <= r2[0] or r2[2] <= r1[0] or r1[3] <= r2[1] or r2[3] <= r1[1])


@lru_cache(maxsize=512)
def _text_sprite(text: str, size: int, color: tuple, font_path: str) -> np.ndarray:
    """RGBA text with a black outline, as a float32 array with straight alpha in [0, 1]."""
    from PIL import Image, ImageDraw, ImageFont

    font = _font(font_path, size)
    stroke = max(2, size // 12)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    x0, y0, x1, y1 = probe.textbbox((0, 0), text, font=font, stroke_width=stroke)
    img = Image.new("RGBA", (x1 - x0 + 4, y1 - y0 + 4), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((2 - x0, 2 - y0), text, font=font, fill=tuple(color) + (255,),
                             stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
    return np.asarray(img).astype(np.float32) / 255.0


@lru_cache(maxsize=64)
def _font(font_path: str, size: int):
    from PIL import ImageFont

    for candidate in (font_path, str(DEFAULT_FONT), "DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _blit(frame: np.ndarray, sprite: np.ndarray, x: int, y: int, opacity: float = 1.0) -> None:
    h, w = sprite.shape[:2]
    H, W = frame.shape[:2]
    fx0, fy0, fx1, fy1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if fx0 >= fx1 or fy0 >= fy1:
        return
    sub = sprite[fy0 - y:fy1 - y, fx0 - x:fx1 - x]
    alpha = sub[:, :, 3:4] * opacity
    region = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    frame[fy0:fy1, fx0:fx1] = (region * (1 - alpha) + sub[:, :, :3] * 255.0 * alpha + 0.5).astype(np.uint8)


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError("ffmpeg not found. Install it (brew install ffmpeg) or `pip install imageio-ffmpeg`.") from e


class FFmpegWriter:
    """Pipe raw RGB frames into ffmpeg and encode an H.264 MP4 that plays everywhere."""

    def __init__(self, path: Path, width: int, height: int, fps: int, music: Optional[Path] = None,
                 duration: float = 0.0, crf: int = 18):
        exe = find_ffmpeg()
        encoders = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        codec = ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf)] if "libx264" in encoders \
            else ["-c:v", "mpeg4", "-q:v", "2"]
        cmd = [exe, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
        if music:
            fade_at = max(0.0, duration - 1.5)
            cmd += ["-stream_loop", "-1", "-i", str(music), "-map", "0:v", "-map", "1:a",
                    "-af", f"afade=t=out:st={fade_at:.2f}:d=1.5", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += codec + ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding the video")


def contact_sheet(shots: list[Shot], out_path: Path, thumb_h: int = 360) -> Path:
    """A quick-review image of every chosen photo with the matched faces boxed and named."""
    from PIL import Image, ImageDraw

    tiles = []
    for i, shot in enumerate(shots):
        im = Image.fromarray(load_image(shot.image)[:, :, ::-1])
        s = thumb_h / im.height
        im = im.resize((max(1, int(im.width * s)), thumb_h))
        d = ImageDraw.Draw(im)
        font = _font(str(DEFAULT_FONT), 22)
        for face, name, color in ((shot.a, shot.a_name, (255, 80, 80)), (shot.b, shot.b_name, (80, 160, 255))):
            box = [face.x * s, face.y * s, (face.x + face.w) * s, (face.y + face.h) * s]
            d.rectangle(box, outline=color, width=3)
            d.text((box[0], box[3] + 2), name, fill=color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        d.text((6, 4), str(i + 1), fill=(255, 255, 0), font=_font(str(DEFAULT_FONT), 30), stroke_width=2,
               stroke_fill=(0, 0, 0))
        tiles.append(im)
    per_row = 3
    rows = [tiles[i:i + per_row] for i in range(0, len(tiles), per_row)]
    width = max(sum(t.width for t in row) for row in rows)
    sheet = Image.new("RGB", (width, thumb_h * len(rows)), (20, 20, 20))
    for r, row in enumerate(rows):
        x = 0
        for t in row:
            sheet.paste(t, (x, r * thumb_h))
            x += t.width
    out_path = Path(out_path)
    sheet.save(out_path, quality=88)
    return out_path
