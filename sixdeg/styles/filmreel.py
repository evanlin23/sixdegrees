"""Film reel: the photos roll past on a strip of film; the current frame is in the gate, sprocket holes scroll."""
from __future__ import annotations

import math

from PIL import Image, ImageDraw

NAME = "filmreel"


def make(ctx):
    FW, FH = 820, 620                       # one film frame (photo window)
    PITCH = FH + 70                         # distance between frames on the strip
    _frames = {}

    def film_frame(i):
        if i not in _frames:
            _frames[i] = ctx.view(i, *ctx.both_cam(i, FW, FH, 0.8), FW, FH)
        return _frames[i]

    def frame(seg, tt):
        i = seg["i"]
        if ctx.C.fj["links"][i].get("lineage"):
            return ctx.frame_lineage(seg, tt)
        local = tt - seg["t0"]
        base = Image.new("RGBA", (ctx.W, ctx.H), (14, 12, 20, 255))
        # the strip advances one frame at the start of each link (a quick, eased pull-down)
        pos = i - 1 + ctx.ease(min(1.0, local / 0.45)) if i else 0.0
        cy0 = 760
        strip_x0, strip_x1 = ctx.CX - FW / 2 - 70, ctx.CX + FW / 2 + 70
        d = ImageDraw.Draw(base)
        d.rectangle([strip_x0, 0, strip_x1, ctx.H], fill=(24, 20, 18))
        off = (pos * PITCH) % 60                    # sprocket holes scroll with the strip
        for y in range(-60, ctx.H + 60, 60):
            yy = y - off
            for x in (strip_x0 + 18, strip_x1 - 52):
                d.rounded_rectangle([x, yy, x + 34, yy + 30], 6, fill=(230, 226, 214))
        for j in range(max(0, i - 2), min(ctx.n_links, i + 3)):
            if ctx.C.fj["links"][j].get("lineage"):
                continue
            y = cy0 + (j - pos) * PITCH
            if -FH < y < ctx.H + FH:
                im = film_frame(j)
                if j != i:
                    im = im.point(lambda v: int(v * 0.45))
                base.alpha_composite(im, (int(ctx.CX - FW / 2), int(y - FH / 2)))
                d.text((strip_x0 + 60, y + FH / 2 + 12), f"{j + 1:02d}A", fill=(240, 180, 60))
        flick = 0.92 + 0.08 * math.sin(tt * 37)       # projector flicker
        if flick < 0.97:
            base.alpha_composite(Image.new("RGBA", (ctx.W, ctx.H), (0, 0, 0, int(40 * (1 - flick) * 6))))
        an, bn = ctx.link_names(i)
        sp, R = seg["speech"], seg.get("rv", ctx.REVEAL)
        ctx.paste(base, ctx.pixel_tag(ctx.C.labels[an].upper(), ctx.CYAN, 30), ctx.CX - 200, cy0 + FH / 2 + 50)
        if local > R * sp + ctx.THROW:
            ctx.paste(base, ctx.pixel_tag(ctx.C.labels[bn].upper(), ctx.YELLOW, 30), ctx.CX + 200, cy0 + FH / 2 + 50)
        yr = ctx.year_of(i)
        if yr:
            ctx.paste(base, ctx.pixel_text(str(yr), 64, fill=ctx.YELLOW), ctx.CX, cy0 - FH / 2 - 60)
        ctx.header(base, i)
        ctx.connected(base, seg, tt)
        seg["cap"].draw(base, tt)
        return base

    return frame
