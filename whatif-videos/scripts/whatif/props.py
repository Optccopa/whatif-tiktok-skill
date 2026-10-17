"""Reusable low-poly props for the street-level scenes: crowds, stalls, lights, rides, trees.

Static props go into an Instances once; animated ones (people, the Ferris wheel, balloons,
pigeons) are drawn every frame from small numpy state, so a crowd of 200 is a few calls.
Coordinates are meters, y up; ground level is y=0 unless a caller passes another.
"""

from __future__ import annotations

import math

import numpy as np

from whatif.meshes import FLAME, GLOW, STRIPES, Instances, beam, chain, rgb, rx, ry, rz, sc, tr

SKIN = [rgb(c) for c in ("#f1c6a5", "#e0ac86", "#c68e6a", "#a06a48", "#7a4f35", "#5a3a26")]
SHIRTS = [rgb(c) for c in ("#e8e4dc", "#d64545", "#3d7bd9", "#f2b134", "#4caf7a", "#9b59b6", "#f07f3c",
                           "#2c3e50", "#ff8fab", "#7fd1e8", "#ffffff", "#c0392b", "#e86fa0", "#3fb8af")]
PANTS = [rgb(c) for c in ("#2d3142", "#4f5d75", "#1f2a44", "#6b5b4b", "#3b3b3b", "#c9b79c", "#e8e4dc")]
HAIR = [rgb(c) for c in ("#2a1d14", "#4a2f1e", "#16110d", "#8a5a2c", "#d8b47a", "#6b6b6b")]
BULBS = [rgb(c) for c in ("#fff1c9", "#ffd36b", "#ff8a8a", "#8ae8c8", "#9cd2ff", "#ff9ad8", "#fff6e0")]
AWNINGS = [rgb(c) for c in ("#e8628f", "#9b6fd6", "#f2c14e", "#e2483d", "#4f8fe0", "#ff9a5a", "#c86fd0")]
BALLOONS = [rgb(c) for c in ("#e8483d", "#f2b134", "#3d9bd9", "#4caf50", "#e86fa0", "#9b59b6", "#ff8a3d", "#ffd23d")]
GONDOLAS = [rgb(c) for c in ("#e85d5d", "#f2b134", "#4fa3e0", "#6cc070", "#b36ad8", "#f07f3c", "#ff8ad8")]
WHITE = rgb("#f2efe8")
DARK = rgb("#26262c")


def block(x0, x1, y0, y1, z0, z1) -> np.ndarray:
    return chain(tr(((x0 + x1) / 2, y0, (z0 + z1) / 2)), sc((x1 - x0, y1 - y0, z1 - z0)))


def catenary(a, b, sag: float, n: int) -> np.ndarray:
    s = np.linspace(0, 1, n)[:, None]
    pts = np.asarray(a, dtype=float) * (1 - s) + np.asarray(b, dtype=float) * s
    pts[:, 1] -= sag * 4 * s[:, 0] * (1 - s[:, 0])
    return pts


# Static props -----------------------------------------------------------------


def string_lights(s: Instances, a, b, sag=0.7, n=16, rng=None, wire=True) -> None:
    """A strand of colored bulbs on a dark wire; bulbs follow the scene's power (p2=1)."""
    pts = catenary(a, b, sag, n)
    off = int(rng.integers(0, 7)) if rng is not None else 0
    colors = [BULBS[(i + off) % len(BULBS)] for i in range(n)]
    s.add("ico0", chain(tr(pts), sc(0.22)), colors, 9.0, GLOW, p2=1.0)
    if wire:
        s.add("box", beam(pts[:-1], pts[1:], 0.04), DARK)


def lamp_post(s: Instances, x, z, y0=0.0, h=4.6, globe=True) -> tuple:
    s.add("cyl5", chain(tr((x, y0, z)), sc((0.16, h, 0.16))), DARK)
    s.add("cyl", chain(tr((x, y0, z)), sc((0.42, 0.5, 0.42))), DARK)
    top = (x, y0 + h + 0.35, z)
    if globe:
        s.add("ico", chain(tr(top), sc(0.75)), rgb("#fff4dc"), 4.0, GLOW, p2=1.0)
    return top


def bench(s: Instances, x, z, yaw, color="#c4553c", y0=0.0) -> None:
    base = chain(tr((x, y0, z)), ry(yaw))
    s.add("box", chain(base, tr((0, 0.45, 0)), sc((2.0, 0.08, 0.55))), rgb(color))
    s.add("box", chain(base, tr((0, 0.55, -0.27)), sc((2.0, 0.5, 0.07))), rgb(color))
    for dx in (-0.85, 0.85):
        s.add("box", chain(base, tr((dx, 0, 0)), sc((0.08, 0.45, 0.5))), DARK)


def round_tree(s: Instances, x, z, h=7.0, leaf="#3f8a3a", rng=None, y0=0.0, trunk="#6b4a35") -> list:
    """A trunk with a cluster of faceted leaf balls. Returns the leaf balls (center, radius) for animation."""
    rng = rng or np.random.default_rng(int(abs(x * 7 + z * 13)))
    s.add("cyl5", chain(tr((x, y0, z)), sc((0.5, h * 0.55, 0.5))), rgb(trunk))
    balls = []
    for k in range(6):
        a = k * 2.1 + rng.uniform(0, 0.6)
        r = h * rng.uniform(0.2, 0.3)
        c = (x + math.cos(a) * h * 0.18, y0 + h * rng.uniform(0.62, 0.85), z + math.sin(a) * h * 0.18)
        balls.append((c, r))
    balls.append(((x, y0 + h * 0.9, z), h * 0.27))
    return balls


def draw_leaves(d: Instances, balls: list, color, scale=1.0) -> None:
    if scale <= 0.01 or not balls:
        return
    c = np.array([b[0] for b in balls])
    r = np.array([b[1] for b in balls]) * scale
    d.add("ico", chain(tr(c), ry(np.arange(len(r)) * 1.3), sc(np.stack([r * 2.1, r * 1.8, r * 2.1], 1))), color)


def stall(s: Instances, x, z, yaw, awning, rng, y0=0.0, width=4.6) -> dict:
    """A market booth facing +z (before yaw): counter, back wall, posts, striped awning, warm light."""
    base = chain(tr((x, y0, z)), ry(yaw))
    body = rgb(rng.choice(["#f4efe6", "#efe4d2", "#f6f1ea"]))
    s.add("box", chain(base, tr((0, 0, -0.9)), sc((width, 2.8, 1.2))), body)  # back room
    s.add("box", chain(base, tr((0, 0, 0.55)), sc((width, 1.05, 0.7))), body)  # counter
    s.add("box", chain(base, tr((0, 1.05, 0.55)), sc((width + 0.1, 0.08, 0.8))), rgb("#b88a5a"))
    for dx in (-width / 2 + 0.08, width / 2 - 0.08):
        s.add("box", chain(base, tr((dx, 0, 0.9)), sc((0.1, 2.6, 0.1))), WHITE)
    s.add("box", chain(base, tr((0, 2.5, 0.2)), rx(0.32), sc((width + 0.3, 0.1, 2.4))), awning, 0.0, STRIPES,
          p1=width * 1.4)
    s.add("box", chain(base, tr((0, 2.15, 1.32)), sc((width + 0.3, 0.32, 0.06))), awning, 0.0, STRIPES, p1=width * 1.4)
    s.add("box", chain(base, tr((0, 1.2, -0.28)), sc((width - 0.6, 1.1, 0.04))), rgb("#ffd9a0"), 1.6, GLOW, p2=1.0)
    goods = rng.integers(3, 7)
    gx = np.linspace(-width / 2 + 0.6, width / 2 - 0.6, goods)
    s.add("box", chain(base, tr(np.stack([gx, np.full(goods, 1.13), np.full(goods, 0.5)], 1)), sc(0.35)),
          [BALLOONS[i] for i in rng.integers(0, len(BALLOONS), goods)])
    return dict(pos=np.array([x, y0, z]), yaw=yaw, width=width, awning=awning)


def circus_tent(s: Instances, x, z, r=10.0, h=6.0, y0=0.0) -> None:
    s.add("cyl", chain(tr((x, y0, z)), sc((r * 2, h * 0.55, r * 2))), rgb("#e2483d"), 0.0, STRIPES, p1=16, p2=1)
    s.add("cone", chain(tr((x, y0 + h * 0.55, z)), sc((r * 2.15, h * 0.75, r * 2.15))), rgb("#e2483d"), 0.0,
          STRIPES, p1=16, p2=1)
    s.add("cyl5", chain(tr((x, y0 + h * 1.25, z)), sc((0.15, 2.4, 0.15))), DARK)
    s.add("box", chain(tr((x + 0.6, y0 + h * 1.25 + 1.8, z)), sc((1.2, 0.6, 0.05))), rgb("#f2c14e"))


def popcorn_cart(s: Instances, x, z, yaw=0.0, y0=0.0) -> np.ndarray:
    base = chain(tr((x, y0, z)), ry(yaw))
    s.add("box", chain(base, tr((0, 0.45, 0)), sc((1.6, 0.9, 0.9))), rgb("#d6393a"))
    s.add("box", chain(base, tr((0, 1.35, 0)), sc((1.3, 0.9, 0.75))), rgb("#cfe6f0"), 0.0)
    s.add("box", chain(base, tr((0, 1.35, 0)), sc((1.2, 0.55, 0.68))), rgb("#fff1c4"), 0.5)
    s.add("box", chain(base, tr((0, 2.25, 0)), sc((1.9, 0.12, 1.2))), rgb("#d6393a"), 0.0, STRIPES, p1=7)
    for dx in (-0.6, 0.6):
        s.add("cyl", chain(tr((x + dx * math.cos(yaw), y0 + 0.3, z - dx * math.sin(yaw))), rz(math.pi / 2),
                           sc((0.6, 0.12, 0.6))), DARK)
    return np.array([x, y0 + 1.7, z])


def sign_post(s: Instances, x, z, y0=0.0, h=4.2, w=3.0, panel_h=1.4, yaw=0.0) -> np.ndarray:
    """A black LED board on a pole; the text is drawn by the overlay at this point. Returns the panel center."""
    s.add("cyl5", chain(tr((x, y0, z)), sc((0.2, h, 0.2))), DARK)
    s.add("box", chain(tr((x, y0 + h - panel_h * 0.5, z)), ry(yaw), sc((w + 0.25, panel_h + 0.25, 0.25))), DARK)
    s.add("box", chain(tr((x, y0 + h - panel_h * 0.45, z + 0.13)), ry(yaw), sc((w, panel_h, 0.02))),
          rgb("#0c0c0e"))
    return np.array([x, y0 + h - panel_h * 0.45, z + 0.15])


def trash_bin(s: Instances, x, z, y0=0.0) -> None:
    s.add("cyl", chain(tr((x, y0, z)), sc((0.7, 1.0, 0.7))), rgb("#3d4a3f"))


# Animated props ---------------------------------------------------------------


class Crowd:
    """People strolling along lanes; each one leaves at its own time, walking off toward an exit.

    lanes: list of (n, x_range, z_range, axis) where axis 0 walks along x, 1 along z.
    """

    def __init__(self, rng, lanes, y_at=None, standing=0.15, exit_speed=3.4):
        xs, zs, axis, lo, hi = [], [], [], [], []
        for n, (x0, x1), (z0, z1), ax in lanes:
            xs.append(rng.uniform(x0, x1, n))
            zs.append(rng.uniform(z0, z1, n))
            axis.append(np.full(n, ax))
            lo.append(np.full(n, x0 if ax == 0 else z0))
            hi.append(np.full(n, x1 if ax == 0 else z1))
        p = dict(x=np.concatenate(xs), z=np.concatenate(zs), axis=np.concatenate(axis), lo=np.concatenate(lo),
                 hi=np.concatenate(hi))
        n = len(p["x"])
        p["speed"] = rng.uniform(0.8, 1.5, n) * (rng.random(n) > standing)  # some stand and chat
        p["dir"] = rng.choice([-1.0, 1.0], n)
        p["phase"] = rng.uniform(0, 2 * np.pi, n)
        p["height"] = rng.uniform(0.86, 1.06, n) * np.where(rng.random(n) < 0.12, 0.7, 1.0)  # a few kids
        p["face"] = rng.uniform(0, 2 * np.pi, n)
        p["skin"] = np.array([SKIN[i] for i in rng.integers(0, len(SKIN), n)])
        p["shirt"] = np.array([SHIRTS[i] for i in rng.integers(0, len(SHIRTS), n)])
        p["pants"] = np.array([PANTS[i] for i in rng.integers(0, len(PANTS), n)])
        p["hair"] = np.array([HAIR[i] for i in rng.integers(0, len(HAIR), n)])
        p["leave"] = np.full(n, 1e9)  # set by the scene
        p["exit"] = np.zeros((n, 3))
        p["sink"] = np.zeros(n)  # exits that go down (stairs): how deep, in meters
        self.p = p
        self.y_at = y_at or (lambda x, z: np.zeros_like(x))
        self.n = n
        self.exit_speed = exit_speed

    def positions(self, t: float):
        """Where everyone is at t: x, z, yaw, walking amount, phase, and how far they've sunk/vanished."""
        p = self.p
        span = np.maximum(p["hi"] - p["lo"], 1e-3)
        start = np.where(p["axis"] == 0, p["x"], p["z"])
        tau = np.minimum(t, p["leave"])
        s = (start - p["lo"]) + p["dir"] * p["speed"] * tau
        m = np.mod(s, 2 * span)
        forward = m < span
        coord = p["lo"] + np.where(forward, m, 2 * span - m)
        heading = p["dir"] * np.where(forward, 1.0, -1.0)
        x = np.where(p["axis"] == 0, coord, p["x"])
        z = np.where(p["axis"] == 1, coord, p["z"])
        walk_yaw = np.where(p["axis"] == 0, heading * np.pi / 2, np.where(heading > 0, 0.0, np.pi))
        yaw = np.where(p["speed"] > 0, walk_yaw, p["face"])
        amp = np.where(p["speed"] > 0, 0.5, 0.0)
        phase = p["phase"] + tau * p["speed"] * 2.6
        # Leaving: walk briskly to the exit.
        gone = t - p["leave"]
        leaving = gone > 0
        ex, ez = p["exit"][:, 0], p["exit"][:, 2]
        dx, dz = ex - x, ez - z
        dist = np.hypot(dx, dz) + 1e-6
        walked = np.minimum(np.maximum(gone, 0) * self.exit_speed, dist)
        x = np.where(leaving, x + dx / dist * walked, x)
        z = np.where(leaving, z + dz / dist * walked, z)
        yaw = np.where(leaving, np.arctan2(dx, dz), yaw)
        amp = np.where(leaving, 0.75, amp)
        phase = np.where(leaving, p["phase"] + np.maximum(gone, 0) * 8.0, phase)
        arrived = leaving & (walked >= dist - 1e-3)
        past = np.maximum(gone - dist / self.exit_speed, 0)  # seconds since reaching the exit
        sink = np.where(arrived, np.minimum(past * 3.0, p["sink"]), 0.0)
        hidden = arrived & ((p["sink"] <= 0) | (past * 3.0 >= p["sink"]))
        return x, z, yaw, amp, phase, sink, hidden

    def draw(self, d: Instances, t: float, tint=None) -> None:
        x, z, yaw, amp, phase, sink, hidden = self.positions(t)
        keep = ~hidden
        if not keep.any():
            return
        p = self.p
        x, z, yaw, amp, phase, sink = x[keep], z[keep], yaw[keep], amp[keep], phase[keep], sink[keep]
        hs = p["height"][keep]
        y = self.y_at(x, z) - sink
        base = chain(tr(np.stack([x, y, z], 1)), ry(yaw))
        n = len(x)
        swing = amp * np.sin(phase)
        zero = np.zeros(n)
        shirt, pants, skin, hair = p["shirt"][keep], p["pants"][keep], p["skin"][keep], p["hair"][keep]
        if tint is not None:  # a frosted or sooty cast over everyone
            shirt, pants = shirt * tint, pants * tint
        for side, sgn in ((-1, 1), (1, -1)):
            d.add("limb", chain(base, tr(np.stack([side * 0.1 * hs, 0.92 * hs, zero], 1)), rx(sgn * swing),
                                sc(np.stack([0.15 * hs, 0.9 * hs, 0.16 * hs], 1))), pants)
            d.add("limb", chain(base, tr(np.stack([side * 0.26 * hs, 1.46 * hs, zero], 1)), rx(-sgn * swing * 0.8),
                                sc(np.stack([0.1 * hs, 0.6 * hs, 0.11 * hs], 1))), shirt)
        d.add("box", chain(base, tr(np.stack([zero, 0.88 * hs, zero], 1)), sc(np.stack([0.42 * hs, 0.62 * hs, 0.25 * hs], 1))),
              shirt)
        d.add("ico", chain(base, tr(np.stack([zero, 1.67 * hs, zero], 1)), sc(0.28 * hs)), skin)
        d.add("ico0", chain(base, tr(np.stack([zero, 1.76 * hs, -0.03 * hs], 1)), sc(np.stack([0.3 * hs, 0.18 * hs, 0.3 * hs], 1))),
              hair)


def draw_ferris_wheel(d: Instances, center, radius: float, t: float, angle: float, power: float,
                      tilt: float = 0.0, char: float = 0.0, y_ground: float = 0.0) -> None:
    """A white wheel with bulbs on the rim and colored gondolas; legs included. Faces the camera (+z)."""
    cx, cy, cz = center
    white = np.array(WHITE) * (1 - char) + np.array(rgb("#1a1714")) * char
    for dx in (-radius * 0.45, radius * 0.45):
        for dz in (-1.6, 1.6):
            d.add("box", beam((cx + dx, y_ground, cz + dz), (cx, cy, cz + dz * 0.5), 0.55), white)
    pivot = chain(tr((cx, cy, cz)), rz(tilt))
    a = angle + np.linspace(0, 2 * np.pi, 40, endpoint=False)
    rim = np.stack([radius * np.cos(a), radius * np.sin(a), np.zeros_like(a)], 1)
    rim_next = np.roll(rim, -1, axis=0)
    inner = rim * 0.62
    inner_next = np.roll(inner, -1, axis=0)
    for dz in (-0.8, 0.8):
        off = np.array([0, 0, dz])
        d.add("box", np.matmul(pivot, beam(rim + off, rim_next + off, 0.3)), white)
        d.add("box", np.matmul(pivot, beam(inner + off, inner_next + off, 0.16)), white)
    spokes = rim[::2]
    d.add("box", np.matmul(pivot, beam(np.zeros_like(spokes), spokes, 0.12)), white)
    d.add("ico", np.matmul(pivot, sc(1.6)), rgb("#d6393a") if char < 0.5 else white)
    if power > 0.02:
        mid = (rim + rim_next) / 2
        colors = [BULBS[i % len(BULBS)] for i in range(len(mid))]
        d.add("ico0", np.matmul(pivot, chain(tr(mid + (0, 0, 0.9)), sc(0.32))), colors, 9.0 * power, GLOW)
        d.add("ico0", np.matmul(pivot, chain(tr(inner[::2] + (0, 0, 0.9)), sc(0.25))), BULBS[0], 7.0 * power, GLOW)
    hang = rim[1::2]
    cab = chain(tr(hang), rz(np.full(len(hang), -tilt)), tr((0, -2.1, 0)))
    colors = np.array([GONDOLAS[i % len(GONDOLAS)] for i in range(len(hang))])
    colors = colors * (1 - char) + np.array(rgb("#1a1714")) * char
    d.add("box", np.matmul(pivot, chain(cab, sc((1.6, 1.5, 1.4)))), colors)
    d.add("box", np.matmul(pivot, chain(cab, tr((0, 1.5, 0)), sc((1.8, 0.2, 1.6)))), colors)
    d.add("box", np.matmul(pivot, chain(tr(hang), sc((0.12, 0.7, 0.12)), tr((0, -1.0, 0)))), white)


def draw_balloons(d: Instances, anchor, count: int, t: float, popped=None, seed: int = 3, scale: float = 1.0) -> None:
    """A vendor's bunch of balloons on strings, swaying; popped ones are gone."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi, count)
    r = rng.uniform(0.2, 1.1, count)
    h = rng.uniform(2.2, 3.6, count) * (0.35 + 0.65 * scale)  # small balloons lose their lift and droop
    sway = 0.25 * np.sin(t * 1.3 + a)
    top = np.stack([anchor[0] + r * np.cos(a) + sway, anchor[1] + h, anchor[2] + r * np.sin(a)], 1)
    keep = np.ones(count, bool) if popped is None else ~popped
    if not keep.any():
        return
    top = top[keep]
    colors = np.array([BALLOONS[i % len(BALLOONS)] for i in range(count)])[keep]
    base = np.repeat(np.asarray(anchor, dtype=float)[None], len(top), 0)
    d.add("box", beam(base, top, 0.025), rgb("#e8e4dc"))
    d.add("ico", chain(tr(top + (0, 0.35 * scale, 0)), sc(np.stack([np.full(len(top), 0.62), np.full(len(top), 0.75),
                                                                      np.full(len(top), 0.62)], 1) * scale)), colors, 0.15)


def draw_pigeons(d: Instances, home: np.ndarray, t: float, fly_at: float, seed: int = 9) -> None:
    """Pigeons pecking around the square until fly_at, then flapping up and away."""
    rng = np.random.default_rng(seed)
    n = len(home)
    phase = rng.uniform(0, 6, n)
    yaw = rng.uniform(0, 2 * np.pi, n) + 0.4 * np.sin(t * 0.5 + phase)
    pos = home.copy()
    up = np.maximum(t - (fly_at + rng.uniform(0, 1.5, n)), 0)
    pos = pos + np.stack([np.sin(phase) * up * 6, up * up * 2.2 + up * 3, -up * 9], 1)
    peck = np.where(up > 0, 0.0, 0.18 * np.maximum(np.sin(t * 3.0 + phase * 3), 0))
    base = chain(tr(pos), ry(yaw))
    gray = rgb("#7c7f88")
    d.add("ico0", chain(base, tr((0, 0.18, 0)), sc((0.22, 0.2, 0.36))), gray)
    d.add("ico0", chain(base, tr(np.stack([np.zeros(n), 0.32 - peck, 0.17 + peck * 0.4], 1)), sc(0.13)), rgb("#5d6170"))
    if (up > 0).any():
        flap = 0.9 * np.sin(t * 22 + phase) * (up > 0)
        for side in (-1, 1):
            d.add("box", chain(base, tr((0, 0.24, 0)), rz(side * flap), tr((side * 0.22, 0, 0)), sc((0.42, 0.03, 0.2))),
                  gray)


def draw_flames(d: Instances, x, y, z, size, seed, t: float) -> None:
    """Licking flames (three cones each) plus drifting embers."""
    x, y, z, size, seed = (np.asarray(v, dtype=float) for v in (x, y, z, size, seed))
    if not len(x):
        return
    for k, (dx, dz, hk) in enumerate(((0, 0, 1.0), (0.35, 0.2, 0.7), (-0.3, -0.25, 0.6))):
        flick = 0.8 + 0.18 * np.sin(t * 9.1 + seed * (k + 1)) + 0.12 * np.sin(t * 15.3 + seed * 2.3 + k)
        h = size * hk * flick
        d.add("cone5", chain(tr(np.stack([x + dx * size * 0.5, y, z + dz * size * 0.5], 1)), ry(seed + t * 2.0 * (k + 1)),
                             sc(np.stack([h * 0.5, h, h * 0.5], 1))), rgb("#ff8a2a"), 5.5, FLAME)
    m = len(x)
    e = np.arange(min(m * 3, 900))
    src = e % m
    cycle = 2.6
    age = np.mod(t + e * 0.73, cycle)
    ex = x[src] + np.sin(e * 1.7 + t) * 0.4 * size[src] + age * 0.5
    ey = y[src] + size[src] * 0.5 + age * 2.0 * (0.6 + size[src] * 0.15)
    ez = z[src] + np.cos(e * 2.3) * 0.4 * size[src]
    d.add("ico0", chain(tr(np.stack([ex, ey, ez], 1)), sc(0.08 + 0.02 * size[src])), rgb("#ffb35a"),
          9.0 * (1 - age / cycle), GLOW)


def draw_puffs(d: Instances, x, z, y0, t: float, amount: float, color, rise=2.5, spread=1.0, seed=0,
               size=1.0, n_max=400) -> None:
    """Rising, swelling puffs (steam or smoke) over a set of source points."""
    n = int(n_max * min(max(amount, 0.0), 1.0))
    if n <= 0:
        return
    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    i = np.arange(n)
    src = i % len(x)
    cycle = 4.0
    age = np.mod(t * 0.9 + i * 0.37 + seed, cycle)
    k = age / cycle
    px = x[src] + np.sin(i * 2.1) * spread + k * 1.5
    pz = z[src] + np.cos(i * 1.3) * spread
    py = y0 + age * rise
    r = size * (0.5 + 1.6 * k) * (1 - k * 0.6)
    d.add("ico", chain(tr(np.stack([px, py, pz], 1)), ry(i * 0.7), sc(r)), color, 0.25)
