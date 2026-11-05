"""Pin, the channel mascot, as real pixel art (built on a 26x36 grid, scaled with nearest-neighbour).

draw(expr, scale, ...) -> RGBA sprite.  Expressions: happy, shock, smug, angry, wink.
Options: arms (little noodle arms with white gloves), hat (detective hat), palette name.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image

PALETTES = {
    "red":    {"R": (236, 44, 64), "D": (158, 22, 52), "L": (255, 132, 140)},
    "yellow": {"R": (255, 200, 30), "D": (200, 120, 10), "L": (255, 240, 150)},
    "cyan":   {"R": (40, 200, 255), "D": (20, 110, 200), "L": (170, 240, 255)},
    "purple": {"R": (160, 80, 255), "D": (90, 30, 170), "L": (215, 180, 255)},
}
FIXED = {"K": (24, 14, 32), "W": (255, 255, 255), "G": (196, 204, 218), "g": (116, 122, 140), "M": (90, 10, 34),
         "T": (255, 130, 150), "H": (120, 86, 60), "h": (80, 56, 38)}

GW, GH = 34, 44   # local art is drawn at offset (4, 6), leaving room for hat, arms, outline and border


def _grid(expr: str, arms: bool, hat: bool, look: int) -> list[list[str]]:
    big = [["." for _ in range(GW)] for _ in range(GH)]
    g = _Local(big, 4, 6)
    cx, cy, r = 13, 12, 10
    # head
    for y in range(36):
        for x in range(26):
            if (x - cx + 0.5) ** 2 + (y - cy + 0.5) ** 2 <= r * r:
                g[y][x] = "D" if (x - cx) * 0.6 + (y - cy) > 5 else "R"
    for (x, y) in [(8, 5), (9, 5), (7, 6), (8, 6), (7, 7)]:
        g[y][x] = "L"
    g[5][10] = "W"; g[6][9] = "W"
    # collar (flat disc under the head)
    for y, half, col in ((22, 8, "R"), (23, 9, "D"), (24, 8, "D")):
        for x in range(cx - half, cx + half):
            g[y][x] = col
    # needle
    for y in range(25, 35):
        g[y][12] = "G"; g[y][13] = "g"
    g[35][12] = "g"
    # eyes
    ey = 11
    for ex in (9, 16):
        if expr == "shock":
            for y in range(ey - 3, ey + 3):
                for x in range(ex - 2, ex + 3):
                    g[y][x] = "W"
            g[ey][ex + look] = "K"
        elif expr == "smug":
            for x in range(ex - 2, ex + 2):
                g[ey - 1][x] = "K"; g[ey][x] = "W"; g[ey + 1][x] = "W"
            g[ey + 1][ex + look] = "K"; g[ey + 1][ex - 1 + look] = "K"
        elif expr == "wink" and ex == 16:
            for x in range(ex - 2, ex + 2):
                g[ey + 1][x] = "K"
            g[ey][ex - 2] = "K"; g[ey][ex + 1] = "K"
        else:
            for y in range(ey - 2, ey + 2):
                for x in range(ex - 2, ex + 2):
                    g[y][x] = "W"
            for y in (ey - 1, ey):
                for x in (ex - 1 + look, ex + look):
                    g[y][x] = "K"
            g[ey - 2][ex - 2] = "R" if g[ey - 2][ex - 2] == "W" else g[ey - 2][ex - 2]
        if expr == "angry":
            g[ey - 3][ex - 2] = "K"; g[ey - 3][ex - 1] = "K"
            g[ey - 4 if ex == 9 else ey - 2][ex - 2 if ex == 9 else ex] = "K"
            g[ey - 2 if ex == 9 else ey - 4][ex + 1 if ex == 9 else ex + 1] = "K"
    # mouth
    my = 16
    if expr == "shock":
        for y in range(my - 1, my + 2):
            for x in range(12, 15):
                g[y][x] = "M"
        g[my - 1][12] = "K"; g[my - 1][14] = "K"; g[my + 1][12] = "K"; g[my + 1][14] = "K"
    elif expr == "smug":
        g[my][11] = "K"; g[my][12] = "K"; g[my][13] = "K"; g[my - 1][14] = "K"; g[my - 2][15] = "K"
    elif expr == "angry":
        for x in range(11, 16):
            g[my + 1][x] = "K"
        g[my][11] = "K"; g[my][15] = "K"
    else:
        g[my - 1][10] = "K"; g[my - 1][16] = "K"
        for x in range(11, 16):
            g[my][x] = "M"
        g[my + 1][12] = "M"; g[my + 1][13] = "T"; g[my + 1][14] = "M"
    if arms:  # noodle arms with white gloves, raised (2 px thick)
        for (x, y) in [(3, 15), (2, 14), (1, 13), (0, 12), (3, 14), (2, 13), (1, 12), (0, 11)]:
            g[y][x] = "R"
        for (x, y) in [(22, 15), (23, 14), (24, 13), (25, 12), (22, 14), (23, 13), (24, 12), (25, 11)]:
            g[y][x] = "R"
        for (x, y) in [(-2, 8), (-1, 8), (0, 8), (-2, 9), (-1, 9), (0, 9), (-1, 10), (0, 10),
                       (25, 8), (26, 8), (27, 8), (25, 9), (26, 9), (27, 9), (25, 10), (26, 10)]:
            g[y][x] = "W"
    if hat:  # detective hat sitting on top of the head
        for x in range(4, 22):
            g[1][x] = "h"
        for y in (-4, -3, -2, -1, 0):
            for x in range(8, 18):
                g[y][x] = "H"
        for x in range(8, 18):
            g[0][x] = "K"
    # 1-px outline around everything
    g = big
    filled = [[c != "." for c in row] for row in g]
    out = [row[:] for row in g]
    for y in range(GH):
        for x in range(GW):
            if not filled[y][x]:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < GW and 0 <= yy < GH and filled[yy][xx]:
                        out[y][x] = "K"
                        break
    return out


class _Local:
    """Write into the big grid with an offset, so local art coordinates can go slightly negative."""
    def __init__(self, big, ox, oy):
        self.big, self.ox, self.oy = big, ox, oy

    def __getitem__(self, y):
        big, ox, oy = self.big, self.ox, self.oy

        class Row:
            def __getitem__(_, x):
                return big[y + oy][x + ox]

            def __setitem__(_, x, v):
                big[y + oy][x + ox] = v
        return Row()


@lru_cache(maxsize=256)
def _sprite(expr: str, arms: bool, hat: bool, palette: str, look: int) -> Image.Image:
    pal = {**FIXED, **PALETTES[palette]}
    g = _grid(expr, arms, hat, look)
    a = np.zeros((GH, GW, 4), np.uint8)
    for y in range(GH):
        for x in range(GW):
            c = g[y][x]
            if c != ".":
                a[y, x, :3] = pal[c]
                a[y, x, 3] = 255
    return Image.fromarray(a)


def draw(expr: str = "happy", scale: int = 8, angle: float = 0.0, squash: float = 0.0, look: int = 0,
         arms: bool = True, hat: bool = False, palette: str = "red", white_border: int = 1) -> Image.Image:
    im = _sprite(expr, arms, hat, palette, max(-1, min(1, look)))
    if white_border:  # 1 sprite-pixel white sticker border, so it reads on any photo
        al = np.array(im)[..., 3] > 0
        b = np.zeros_like(al)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                b |= np.roll(np.roll(al, dy, 0), dx, 1)
        a = np.array(im)
        a[b & ~al] = (255, 255, 255, 255)
        im = Image.fromarray(a)
    w = max(1, round(im.width * scale * (1 + squash * 0.35)))
    h = max(1, round(im.height * scale * (1 - squash * 0.35)))
    im = im.resize((w, h), Image.NEAREST)
    if angle:
        q = round(angle / 15) * 15  # snap rotation: keeps the chunky pixel look
        im = im.rotate(q, resample=Image.NEAREST, expand=True)
    return im
