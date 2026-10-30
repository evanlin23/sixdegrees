import json
import math
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from sixdegrees.faces import Face
from sixdegrees.video import Camera, Renderer, Shot, Style, find_ffmpeg, interpolate


def test_camera_maps_centre_point_to_frame_centre():
    cam = Camera(np.array([100.0, 50.0]), 2.0, 0.3)
    assert np.allclose(cam.to_screen((100, 50), 200, 400), [100, 200])


def test_interpolate_moves_focus_point_in_a_straight_line():
    w, h = 400, 800
    c0, c1 = Camera(np.array([50.0, 60.0]), 4.0, 0.2), Camera(np.array([300.0, 200.0]), 1.0, 0.0)
    focus = (60.0, 70.0)
    s0, s1 = c0.to_screen(focus, w, h), c1.to_screen(focus, w, h)
    for t in (0.0, 0.25, 0.5, 1.0):
        cam = interpolate(c0, c1, t, focus, w, h)
        assert np.allclose(cam.to_screen(focus, w, h), (1 - t) * s0 + t * s1)
        assert math.isclose(cam.z, 4.0 ** (1 - t))  # geometric zoom


def test_distance_transition_endpoints():
    r = Renderer(Style(width=8, height=8))
    a = np.full((8, 8, 3), 100, np.uint8)
    b = np.full((8, 8, 3), 140, np.uint8)
    assert np.array_equal(r._transition(a, b, 0.0), a)
    assert np.array_equal(r._transition(a, b, 1.0), b)
    # similar pixels blend smoothly...
    assert abs(r._transition(a, b, 0.5).mean() - 120) < 1
    # ...very different pixels switch early, which gives xfade=distance its look
    black, white = np.zeros_like(a), np.full_like(a, 255)
    assert np.array_equal(r._transition(black, white, 0.5), white)


def _synthetic_photo(path, w, h, color):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = color
    img[h // 3:h // 3 + 40, w // 4:w // 4 + 40] = 255
    img[h // 3:h // 3 + 40, 3 * w // 4 - 40:3 * w // 4] = 128
    Image.fromarray(img).save(path)


@pytest.mark.skipif(not (shutil.which("ffmpeg") or shutil.which("ffprobe")), reason="ffmpeg not installed")
def test_render_small_video(tmp_path):
    p1, p2 = tmp_path / "1.jpg", tmp_path / "2.png"
    _synthetic_photo(p1, 400, 300, (30, 60, 90))
    _synthetic_photo(p2, 300, 450, (90, 30, 60))
    shots = [Shot(p1, "Ann Lee", "Bob Ray", Face(100, 100, 40, 40, 1.0), Face(260, 100, 40, 40, 1.0)),
             Shot(p2, "Bob Ray", "Cat Moe", Face(75, 150, 40, 40, 1.0), Face(185, 150, 40, 40, 1.0))]
    out = tmp_path / "v.mp4"
    style = Style(width=180, height=320, fps=10, title="Test")
    Renderer(style).render(shots, out)
    assert out.exists() and out.stat().st_size > 1000
    probe = shutil.which("ffprobe")
    if probe:
        info = json.loads(subprocess.run([probe, "-v", "error", "-show_entries", "format=duration:stream=width,height",
                                          "-of", "json", str(out)], capture_output=True, text=True).stdout)
        assert info["streams"][0]["width"] == 180
        clip = style.hold * 2 + style.zoom_out + style.pan + style.zoom_in
        expected = 2 * clip + style.intro_hold + style.outro_hold - style.transition
        assert abs(float(info["format"]["duration"]) - expected) < 0.25
