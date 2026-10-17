"""Text on top of the 3D frames: the title hook, the distance readout, subtitles and the ending.

Positions follow TikTok's safe area: nothing in the top ~220px,
the right ~150px or the bottom ~420px.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

from whatif.scenario import Scenario, Timeline, format_miles, ramp  # noqa: F401

FONTS = Path(__file__).resolve().parents[2] / "assets" / "fonts"
WHITE = (250, 248, 244)
SOFT = (226, 226, 232)

HOOK_TOP = 300
HOOK_SECONDS = 3.6
HUD_X = 92
HUD_LABEL_Y = 858
SUB_CENTER = (520, 1214)
SUB_MAX_W = 860
END_CENTER_Y = 905


@lru_cache(maxsize=None)
def serif(size: int, weight: int = 500) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONTS / "SourceSerif4-Variable.ttf"), size)
    axes = f.get_variation_axes()
    values = []
    for axis in axes:
        name = axis["name"].decode() if isinstance(axis["name"], bytes) else axis["name"]
        if name.lower().startswith("weight"):
            values.append(weight)
        else:  # optical size: the display cut at big sizes
            values.append(min(max(size * 0.75, axis["minimum"]), axis["maximum"]))
    f.set_variation_by_axes(values)
    return f


@lru_cache(maxsize=None)
def italic(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / "SourceSerif4-Italic.ttf"), size)


@lru_cache(maxsize=None)
def sans(size: int, weight: int = 700) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONTS / "Montserrat-Variable.ttf"), size)
    f.set_variation_by_axes([weight])
    return f


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


Run = tuple[str, tuple[int, int, int]]


def _run_width(text: str, font, tracking: float) -> float:
    return font.getlength(text) + tracking * max(len(text) - 1, 0)


def _line_width(runs: list[Run], font, tracking: float) -> float:
    return _run_width("".join(t for t, _ in runs), font, tracking)


def split_hook(hook: str, accent: tuple[int, int, int]) -> list[Run]:
    """'What if {Saturn} crashed' -> words with the braced one in the accent color."""
    runs: list[Run] = []
    for part in re.split(r"(\{[^}]*\})", hook):
        if not part:
            continue
        color = accent if part.startswith("{") else WHITE
        for word in part.strip("{}").split():
            runs.append((word, color))
    return runs


def balance(words: list[Run], font, max_width: float, tracking: float = 0.0) -> list[list[Run]]:
    """Fewest lines that fit, with line lengths as even as possible."""
    def lines_for(split: tuple[int, ...]) -> list[list[Run]]:
        bounds = (0, *split, len(words))
        return [words[a:b] for a, b in zip(bounds[:-1], bounds[1:])]

    def spaced(line: list[Run]) -> list[Run]:
        out: list[Run] = []
        for i, (w, c) in enumerate(line):
            tight = i == 0 or w[:1] in "?!.,:;"  # no space before punctuation after a highlighted phrase
            out.append((w if tight else " " + w, c))
        return out

    n = len(words)
    if _line_width(spaced(words), font, tracking) <= max_width or n < 2:
        return [spaced(words)]

    def keeps_phrase(i: int) -> bool:  # don't split a highlighted phrase across lines
        return not (words[i - 1][1] != WHITE and words[i][1] != WHITE)

    best = None
    for i in range(1, n):
        lines = [spaced(x) for x in lines_for((i,))]
        widest = max(_line_width(x, font, tracking) for x in lines)
        if best is None or widest < best[0]:
            best = (widest, lines)
    if best[0] <= max_width:
        return best[1]
    best = None
    for i in range(1, n - 1):  # three lines
        for j in range(i + 1, n):
            if best is not None and not (keeps_phrase(i) and keeps_phrase(j)):
                continue
            lines = [spaced(x) for x in lines_for((i, j))]
            widest = max(_line_width(x, font, tracking) for x in lines)
            fits = widest <= max_width
            score = (not fits, not (keeps_phrase(i) and keeps_phrase(j)), widest)
            if best is None or score < best[0]:
                best = (score, lines)
    return best[1]


def text_sprite(lines: list[list[Run]], font, line_height: int, align: str = "center", tracking: float = 0.0,
                shadow: float = 0.8, blur: int = 7, pad: int = 28) -> "Sprite":
    widths = [_line_width(line, font, tracking) for line in lines]
    w = int(max(widths)) + 2 * pad
    ascent, descent = font.getmetrics()
    h = line_height * (len(lines) - 1) + ascent + descent + 2 * pad
    color_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    mask = Image.new("L", (w, h), 0)
    draw_c = ImageDraw.Draw(color_layer)
    draw_m = ImageDraw.Draw(mask)
    for i, (line, width) in enumerate(zip(lines, widths)):
        x = pad + (max(widths) - width) / 2 if align == "center" else pad
        y = pad + i * line_height
        for text, color in line:
            for ch in (text if tracking else [text]):
                draw_c.text((x, y), ch, font=font, fill=color + (255,))
                draw_m.text((x, y), ch, font=font, fill=255)
                x += font.getlength(ch) + tracking
    # Blurring at half size is 4x cheaper and looks the same for a soft shadow.
    small = mask.resize((max(w // 2, 1), max(h // 2, 1)), Image.BILINEAR).filter(ImageFilter.GaussianBlur(blur / 2))
    shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow_layer.putalpha(small.resize((w, h), Image.BILINEAR).point(lambda a: int(min(a * shadow * 1.6, 255))))
    return Sprite(Image.alpha_composite(shadow_layer, color_layer))


class Sprite:
    """An RGBA text image split into the RGB and mask that Image.paste wants."""

    def __init__(self, rgba: Image.Image):
        self.rgb = rgba.convert("RGB")
        self.mask = rgba.getchannel("A")
        self.width, self.height = rgba.size


def paste(img: Image.Image, sprite: Sprite, x: float, y: float, alpha: float = 1.0) -> None:
    if alpha <= 0:
        return
    mask = sprite.mask if alpha >= 1 else sprite.mask.point(lambda a: int(a * alpha))
    img.paste(sprite.rgb, (int(round(x)), int(round(y))), mask)


class Overlay:
    PAD = 28

    def __init__(self, scenario: Scenario, timeline: Timeline):
        self.s = scenario
        self.tl = timeline
        self.accent = hex_rgb(scenario.accent)
        runs = split_hook(scenario.hook, self.accent)
        size = 88 if len(balance(runs, serif(88, 560), 880)) <= 2 else 80
        hook_font = serif(size, 560)
        self.hook = text_sprite(balance(runs, hook_font, 900), hook_font, int(size * 1.16))
        self._cache: OrderedDict[tuple, Sprite] = OrderedDict()

    def _sprite(self, key: tuple, make) -> Sprite:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        sprite = make()
        self._cache[key] = sprite
        if len(self._cache) > 96:
            self._cache.popitem(last=False)
        return sprite

    def _subtitle(self, text: str, size: int = 54) -> Sprite:
        font = italic(size)
        words = [(w, WHITE) for w in text.split()]
        return self._sprite(("sub", text, size), lambda: text_sprite(balance(words, font, SUB_MAX_W), font,
                                                                    int(size * 1.22)))

    @staticmethod
    def _backing_alpha(img: Image.Image, sprite: Sprite, x: float, y: float) -> float:
        """0 on a normal frame, rising to 1 when the area behind the subtitle is bright."""
        box = (int(x + sprite.width * 0.15), int(y + sprite.height * 0.3),
               int(x + sprite.width * 0.85), int(y + sprite.height * 0.7))
        lum = sum(ImageStat.Stat(img.crop(box).convert("L").resize((32, 8))).mean) / 255.0
        return min(max((lum - 0.5) / 0.25, 0.0), 1.0)

    def _backing(self, w: int, h: int) -> Sprite:
        def make():
            mask = Image.new("L", (w, h), 0)
            ImageDraw.Draw(mask).rounded_rectangle((w * 0.08, h * 0.22, w * 0.92, h * 0.78), radius=h // 4, fill=150)
            mask = mask.filter(ImageFilter.GaussianBlur(h / 7))
            rgba = Image.new("RGBA", (w, h), (12, 10, 20, 0))
            rgba.putalpha(mask)
            return Sprite(rgba)
        return self._sprite(("backing", w, h), make)

    def _sign(self, img: Image.Image, x: float, y: float, depth: float, text: str) -> None:
        """Amber LED lettering on the scene's temperature board."""
        px = 1662.0 * 1.5 / max(depth, 1.0)  # the board is 1.5 m tall; 1662 px = focal length at fov 60
        size = int(min(max(px * 0.5, 12), 120))
        sprite = self._sprite(("sign", text, size), lambda: text_sprite(
            [[(text, (255, 196, 92))]], sans(size, 800), size, shadow=0.0, blur=1, pad=4))
        paste(img, sprite, x - sprite.width / 2, y - sprite.height / 2)

    def _frost(self, img: Image.Image, amount: float) -> None:
        """Ice crystals growing in from the edges of the lens."""
        if amount <= 0.01:
            return
        sprite = self._sprite(("frost",), lambda: frost_sprite(img.width, img.height))
        paste(img, sprite, 0, 0, min(amount, 1.0))

    def apply(self, img: Image.Image, t: float, sign=None, frost: float = 0.0) -> None:
        if frost > 0:
            self._frost(img, frost)
        if sign is not None:
            self._sign(img, *sign)
        T = self.tl.T
        pad = self.PAD
        # Hook: on the first frame (it doubles as the cover), gone by the first subtitle.
        hook_alpha = 1.0 - ramp(t, HOOK_SECONDS, HOOK_SECONDS + 0.4)
        if hook_alpha > 0:
            paste(img, self.hook, 540 - self.hook.width / 2, HOOK_TOP - pad, hook_alpha)

        hud_alpha = 1.0 - ramp(t, self.tl.hud_end, self.tl.hud_end + 0.3)
        if hud_alpha > 0:
            label, value, details = self.tl.readout(t)
            sprite = self._sprite(("hudlabel", label), lambda: text_sprite([[(label, WHITE)]], sans(25, 700), 30,
                                                                         "left", tracking=3.2, blur=5))
            paste(img, sprite, HUD_X - pad, HUD_LABEL_Y - pad, hud_alpha)
            sprite = self._sprite(("value", value), lambda: text_sprite([[(value, WHITE)]], serif(88, 500), 100, "left"))
            paste(img, sprite, HUD_X - pad - 4, HUD_LABEL_Y + 30 - pad, hud_alpha)
            y = HUD_LABEL_Y + 140
            for detail in details:
                sprite = self._sprite(("detail", detail), lambda d=detail: text_sprite(
                    [[(d, SOFT)]], sans(22, 650), 28, "left", tracking=2.6, blur=5))
                paste(img, sprite, HUD_X - pad, y - pad, hud_alpha)
                y += 36

        text, alpha = self.tl.line_at(t)
        if text:
            after = t >= T
            sprite = self._subtitle(text, 60 if after else 54)
            cy = END_CENTER_Y if after else SUB_CENTER[1]
            cx = 540 if after else SUB_CENTER[0]
            x, y = cx - sprite.width / 2, cy - sprite.height / 2
            backing = self._backing_alpha(img, sprite, x, y)
            if backing > 0:  # a bright sky or flash behind the words: lay a soft shadow under them
                paste(img, self._backing(sprite.width, sprite.height), x, y, alpha * backing)
            paste(img, sprite, x, y, alpha)

        out_alpha = ramp(t, self.tl.outro_start, self.tl.outro_start + 0.4)
        if out_alpha > 0:
            big = self._sprite(("outro", self.s.outro), lambda: text_sprite(
                balance([(w, WHITE) for w in self.s.outro.split()], serif(76, 560), 860), serif(76, 560), 90))
            paste(img, big, 540 - big.width / 2, END_CENTER_Y - big.height / 2 - 30, out_alpha)
            if self.s.outro_small:
                small = self._sprite(("outro_small", self.s.outro_small), lambda: text_sprite(
                    [[(self.s.outro_small, self.accent)]], sans(28, 700), 32, tracking=3.0, blur=5))
                paste(img, small, 540 - small.width / 2, END_CENTER_Y + big.height / 2 - 10, out_alpha)


def frost_sprite(w: int, h: int, seed: int = 5) -> Sprite:
    """Six-armed crystals of many sizes, thick at the edges of the frame and thinning toward the middle."""
    import random
    rng = random.Random(seed)
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for _ in range(140):
        edge = rng.random()
        if edge < 0.25:
            x, y = rng.uniform(0, w), rng.uniform(0, h * 0.12)
        elif edge < 0.5:
            x, y = rng.uniform(0, w), rng.uniform(h * 0.88, h)
        elif edge < 0.75:
            x, y = rng.uniform(0, w * 0.14), rng.uniform(0, h)
        else:
            x, y = rng.uniform(w * 0.86, w), rng.uniform(0, h)
        r = rng.uniform(18, 70)
        a0 = rng.uniform(0, 1.05)
        alpha = int(rng.uniform(90, 200))
        width = 2 if r > 40 else 1
        for k in range(6):
            a = a0 + k * 3.14159 / 3
            ex, ey = x + r * math.cos(a), y + r * math.sin(a)
            draw.line((x, y, ex, ey), fill=(240, 246, 255, alpha), width=width)
            for f in (0.45, 0.7):  # side branches
                bx, by = x + (ex - x) * f, y + (ey - y) * f
                for side in (-1, 1):
                    b = a + side * 0.9
                    draw.line((bx, by, bx + r * 0.28 * math.cos(b), by + r * 0.28 * math.sin(b)),
                              fill=(240, 246, 255, alpha), width=1)
    glow = layer.filter(ImageFilter.GaussianBlur(2))
    return Sprite(Image.alpha_composite(glow, layer))
