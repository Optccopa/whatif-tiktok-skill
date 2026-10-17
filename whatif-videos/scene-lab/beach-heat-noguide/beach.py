"""The beach scene: a tropical cove on a bright afternoon, for "what if it got hotter every second?".

Layout (meters): the camera stands on a hotel terrace at the back of the beach and looks out (-z)
over white sand, umbrellas and palms to a turquoise lagoon, a jetty with over-water bungalows, a
reef line and green islands. The sun is high behind us, so everything is front-lit under a deep
blue sky. Every change follows the temperature: people run off, the palms dry, the sea boils at
213°F, the power fails, beach balls swell and burst, plastic loungers slump, paperbacks catch at
451°F, polyester umbrellas melt near 480°F, the wood smokes, then everything wooden burns at 572°F.
"""

from __future__ import annotations

import math

import numpy as np

from whatif import props
from whatif.gl import Frame, look_at, ortho
from whatif.meshes import GLOW, GROUND, STRIPES, WATER, Instances, align_y, chain, rgb, rx, ry, sc, tr
from whatif.scenario import Effects, Timeline, ramp

SHORE_Z0 = -17.0
REEF_Z = -150.0
SUN_AZ_EL = (-150.0, 42.0)  # high behind the camera, a little left: front light, shadows away from us
CAMERA = np.array([0.0, 8.5, 26.0])
LOOK = np.array([0.0, -1.1, -74.0])
JETTY_X = -9.0
JETTY_END = -72.0
N_FLOATS = 8
BURST_F = np.random.default_rng(4).uniform(312.0, 348.0, N_FLOATS)  # also read by the audio (one pop each)
UMBRELLAS = [rgb(c) for c in ("#e8483d", "#f2b134", "#3fb8af", "#ff8fab", "#4f8fe0", "#ff8a3d", "#9b6fd6")]
TOWELS = [rgb(c) for c in ("#e8483d", "#3d9bd9", "#f2c14e", "#4caf7a", "#ff8fab", "#7fd1e8", "#f07f3c")]
BALLS = [rgb(c) for c in ("#e8483d", "#ff5a7a", "#3d9bd9", "#f2b134")]
WOOD = rgb("#a77b52")
THATCH = rgb("#c9a061")


def _dir(az, el):
    a, e = math.radians(az), math.radians(el)
    return np.array([math.sin(a) * math.cos(e), math.sin(e), -math.cos(a) * math.cos(e)])


def _mix(a, b, k):
    return tuple(x * (1 - k) + y * k for x, y in zip(a, b))


def shore_z(x):
    return SHORE_Z0 - 3.0 * np.sin(np.asarray(x) / 19.0 + 0.6) - 0.004 * np.asarray(x) ** 2


def sand_height(x, z):
    """A gentle beach rising from the waterline (h = 0) to a flat back beach, sloping on under the sea."""
    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    d = z - shore_z(x)
    ripple = 0.12 * np.sin(x * 0.21 + z * 0.13) * np.sin(z * 0.17 + 1.0) * np.clip(d / 8.0, 0.0, 1.0)
    return np.clip(0.05 * d, -2.5, 1.5) + ripple


def sand_mesh() -> np.ndarray:
    xs = np.arange(-160.0, 160.1, 3.0)
    zs = np.arange(70.0, -190.0, -2.5)
    X, Z = np.meshgrid(xs, zs)
    P = np.stack([X, sand_height(X, Z), Z], -1)
    a, b, c, d = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]
    return np.stack([np.stack([a, b, d], -2), np.stack([a, d, c], -2)], 2).reshape(-1, 3).astype(np.float32)


def _flat_strip(p: np.ndarray, width: float, thick: float = 0.04) -> np.ndarray:
    """Flat boxes joining consecutive points (foam lines on the water and the sand)."""
    a, b = p[:-1], p[1:]
    d = b - a
    length = np.hypot(d[:, 0], d[:, 2])
    n = len(length)
    return chain(tr((a + b) / 2), ry(np.arctan2(d[:, 0], d[:, 2])), sc(np.stack([np.full(n, width), np.full(n, thick), length], 1)))


def _cyl_between(a, b, w) -> np.ndarray:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = b - a
    length = np.linalg.norm(d, axis=-1)
    w = np.broadcast_to(np.asarray(w, dtype=float), length.shape)
    return chain(tr(a), align_y(d), sc(np.stack([w, length, w], -1)))


class Beach:
    body_az_el = SUN_AZ_EL
    moon_az_el = (-40.0, 25.0)

    def __init__(self, timeline: Timeline, seed: int = 0):
        self.timeline = timeline
        rng = np.random.default_rng(seed + 41)
        self.meshes = {"sand": sand_mesh()}
        self.static = Instances()
        self.static.add("sand", np.eye(4), rgb("#e4cfa2"), 0.0, GROUND, p2=1.0)  # p2: sand never chars
        self.fronds = []  # (crown, yaw, length) per frond
        self.umbrellas = []  # (x, y0, z, color)
        self.loungers = []  # (x, y0, z, yaw)
        self.floats = []  # (pos, kind, color)
        self.wood = []  # (x, y, z) points that smoke before they burn
        fires = []
        self._sea(rng)
        self._far(rng)
        self._jetty(fires)
        self._beach_gear(rng, fires)
        self._tower(fires)
        self._bar(fires)
        self._palms(rng, fires)
        self._crowd(rng)
        self.sign_point = props.sign_post(self.static, 4.6, -5.0, y0=float(sand_height(4.6, -5.0)), h=4.4, w=3.0,
                                          panel_h=1.5)
        a = np.array(fires)
        self.fires = dict(x=a[:, 0], y=a[:, 1], z=a[:, 2], size=a[:, 3], temp=a[:, 4], seed=rng.uniform(0, 100, len(a)))
        self.wood = np.array(self.wood)

    # Static -----------------------------------------------------------------

    def _sea(self, rng) -> None:
        s = self.static
        s.add("quad", chain(tr((0, 0, (10 + REEF_Z) / 2)), sc((900, 1, 10 - REEF_Z))), rgb("#27c9bd"), 0.45, WATER)  # lagoon: the bright sand bottom shows through
        s.add("quad", chain(tr((0, 0, (REEF_Z - 6000) / 2)), sc((12000, 1, 6000 + REEF_Z))), rgb("#0c4a8c"), 0.08, WATER)
        xs = np.arange(-400.0, 400.1, 6.0)  # surf breaking on the reef
        reef = np.stack([xs, np.full_like(xs, 0.03), REEF_Z + 2.5 * np.sin(xs / 23.0)], 1)
        s.add("box", _flat_strip(reef, 2.2), rgb("#f4f8fa"), 0.15)

    def _far(self, rng) -> None:
        s = self.static
        for x, z, r, h in ((-1500, -4200, 1800, 300), (900, -3600, 1100, 190), (2600, -5200, 1600, 260)):
            s.add("cone6", chain(tr((x, -5, z)), ry(rng.uniform(0, 1)), sc((r, h, r * 0.6))), rgb("#3f7a52"))
        # A small sandy islet with a few palms, past the reef.
        s.add("cyl", chain(tr((150, -1.2, -700)), sc((60, 2.0, 30))), rgb("#efdfbc"))
        for k in range(5):
            x = 132 + k * 9 + rng.uniform(-2, 2)
            s.add("cyl5", chain(tr((x, 0, -700)), sc((0.9, 11 + 3 * rng.random(), 0.9))), rgb("#7a6650"))
            s.add("ico0", chain(tr((x, 12.5, -700)), sc((7, 2.5, 7))), rgb("#3d7a3a"))
        for x, z in ((-60, -420), (90, -900), (-200, -1300)):  # sailboats
            s.add("hull", chain(tr((x, -1.0, z)), ry(rng.uniform(0, 6)), sc((2.4, 1.4, 7.0))), props.WHITE)
            s.add("cone", chain(tr((x, 0.0, z)), sc((3.5, 10.0, 0.2))), props.WHITE)
        # A catamaran in the lagoon.
        for dx in (-1.4, 1.4):
            s.add("hull", chain(tr((17 + dx, -0.5, -58)), sc((0.9, 0.8, 7.0))), props.WHITE)
        s.add("box", chain(tr((17, 0.3, -58)), sc((3.6, 0.15, 4.0))), rgb("#d9d2c4"))
        s.add("cone", chain(tr((17, 0.4, -58.5)), sc((4.0, 9.0, 0.15))), rgb("#fff4e6"))

    def _jetty(self, fires) -> None:
        s = self.static
        z0 = float(shore_z(JETTY_X)) + 4.0
        deck_y = 1.3
        s.add("box", props.block(JETTY_X - 1.2, JETTY_X + 1.2, deck_y - 0.15, deck_y, JETTY_END, z0), WOOD, 0.0, STRIPES,
              p1=(z0 - JETTY_END) / 0.6, p2=2)
        zs = np.arange(z0, JETTY_END, -4.0)
        for side in (-1.1, 1.1):
            s.add("box", chain(tr(np.stack([np.full_like(zs, JETTY_X + side), np.full_like(zs, -2.0), zs], 1)),
                               sc((0.22, deck_y + 2.0, 0.22))), rgb("#7a5a3a"))
        for z in zs[1::2]:
            fires.append((JETTY_X, deck_y, z, 2.0, np.random.default_rng(int(-z)).uniform(574, 600)))
            self.wood.append((JETTY_X, deck_y, z))
        # Over-water bungalows on a cross walk at the end.
        s.add("box", props.block(JETTY_X - 18, JETTY_X + 18, deck_y - 0.15, deck_y, JETTY_END - 1.2, JETTY_END + 1.2),
              WOOD, 0.0, STRIPES, p1=50, p2=2)
        for i, bx in enumerate((JETTY_X - 15, JETTY_X - 5, JETTY_X + 5, JETTY_X + 15)):
            bz = JETTY_END - 5.5
            s.add("box", chain(tr(np.array([[bx + dx, -2.0, bz + dz] for dx in (-1.8, 1.8) for dz in (-1.8, 1.8)])),
                               sc((0.25, deck_y + 2.0, 0.25))), rgb("#7a5a3a"))
            s.add("box", props.block(bx - 2.6, bx + 2.6, deck_y - 0.2, deck_y, bz - 2.6, bz + 2.6), WOOD)
            s.add("box", props.block(bx - 2.0, bx + 2.0, deck_y, deck_y + 2.6, bz - 2.0, bz + 2.0), rgb("#b88a5e"))
            s.add("box", chain(tr((bx, deck_y + 1.0, bz + 2.02)), sc((1.0, 1.0, 0.04))), rgb("#3a2a20"))  # door
            s.add("prism", chain(tr((bx, deck_y + 2.6, bz)), ry(math.pi / 2), sc((5.2, 2.0, 5.2))), THATCH)
            fires.append((bx, deck_y + 3.0, bz, 4.0, 575 + 6 * i))
            self.wood.append((bx, deck_y + 3.5, bz))

    def _beach_gear(self, rng, fires) -> None:
        s = self.static
        spots = []
        for z in (10.0, 4.0, -2.5, -9.0):
            for x in np.linspace(-9, 9, 4) if z > 8 else np.linspace(-11, 11, 5):
                if abs(x - JETTY_X) < 2.5 and z < -6:
                    continue
                spots.append((x + rng.uniform(-1.2, 1.2), z + rng.uniform(-1.0, 1.0)))
        palapa = set(rng.choice(len(spots), 4, replace=False).tolist())
        for i, (x, z) in enumerate(spots):
            y0 = float(sand_height(x, z))
            if i in palapa:  # a thatched umbrella on a wooden pole
                s.add("cyl5", chain(tr((x, y0, z)), sc((0.18, 2.6, 0.18))), WOOD)
                s.add("cone", chain(tr((x, y0 + 2.1, z)), sc((3.6, 1.0, 3.6))), THATCH)
                fires.append((x, y0 + 2.3, z, 2.4, rng.uniform(572, 590)))
                self.wood.append((x, y0 + 2.6, z))
            else:
                s.add("cyl5", chain(tr((x, y0, z)), sc((0.08, 2.3, 0.08))), props.WHITE)
                self.umbrellas.append((x, y0, z, UMBRELLAS[i % len(UMBRELLAS)]))
            if rng.random() < 0.55:  # two loungers
                for dx in (-0.6, 0.6):
                    self.loungers.append((x + dx, float(sand_height(x + dx, z + 0.3)), z + 0.3, rng.uniform(-0.15, 0.15)))
            else:  # towels, and a paperback someone was reading
                for dx in (-0.65, 0.65):
                    tx, tz = x + dx, z + 0.6
                    ty = float(sand_height(tx, tz))
                    yaw = rng.uniform(-0.25, 0.25)
                    s.add("box", chain(tr((tx, ty, tz)), ry(yaw), sc((0.9, 0.03, 1.8))),
                          TOWELS[rng.integers(0, len(TOWELS))], 0.0, STRIPES, p1=4)
                    if rng.random() < 0.8:
                        bx, bz = tx + rng.uniform(-0.2, 0.2), tz + rng.uniform(0.0, 0.6)
                        s.add("box", chain(tr((bx, ty + 0.03, bz)), ry(rng.uniform(0, 3)), sc((0.22, 0.04, 0.3))),
                              rgb(rng.choice(["#c0392b", "#2c3e50", "#f2efe8", "#3d7bd9"])))
                        fires.append((bx, ty + 0.05, bz, rng.uniform(1.0, 1.4), rng.uniform(451, 466)))
        for k in range(N_FLOATS):  # beach balls and swim rings left on the sand
            x, z = rng.uniform(-10, 10), rng.uniform(-13, 4)
            self.floats.append((np.array([x, float(sand_height(x, z)), z]), k % 2, BALLS[k % len(BALLS)]))
        for x, z, c in ((5.0, -13.5, "#ff8a3d"), (7.5, -14.2, "#f2c14e")):  # kayaks
            s.add("hull", chain(tr((x, float(sand_height(x, z)), z)), ry(0.3), sc((0.75, 0.35, 3.6))), rgb(c))

    def _tower(self, fires) -> None:
        s = self.static
        x, z = -5.0, -11.0
        y0 = float(sand_height(x, z))
        for dx in (-0.9, 0.9):
            for dz in (-0.9, 0.9):
                s.add("box", chain(tr((x + dx, y0, z + dz)), sc((0.16, 2.4, 0.16))), WOOD)
        s.add("box", props.block(x - 1.3, x + 1.3, y0 + 2.4, y0 + 2.55, z - 1.3, z + 1.3), WOOD)
        s.add("box", props.block(x - 1.0, x + 1.0, y0 + 2.55, y0 + 4.0, z - 1.0, z + 1.0), rgb("#f2efe8"))
        s.add("box", chain(tr((x, y0 + 4.0, z)), sc((2.6, 0.18, 2.6))), rgb("#d6393a"))
        s.add("box", props.beam((x, y0, z + 3.2), (x, y0 + 2.45, z + 1.3), 0.5), WOOD)  # ramp
        s.add("cyl5", chain(tr((x + 1.1, y0 + 4.0, z - 1.1)), sc((0.06, 2.0, 0.06))), props.WHITE)
        s.add("box", chain(tr((x + 1.6, y0 + 5.5, z - 1.1)), sc((1.0, 0.5, 0.03))), rgb("#e8483d"))
        fires.append((x, y0 + 2.6, z, 3.4, 575.0))
        self.wood.append((x, y0 + 3.5, z))

    def _bar(self, fires) -> None:
        s = self.static
        x, z = 10.5, 2.0
        y0 = float(sand_height(x, z))
        s.add("box", props.block(x - 2.5, x + 2.5, y0, y0 + 2.8, z - 3.0, z + 3.0), rgb("#b88a5e"))
        s.add("box", props.block(x - 3.4, x - 2.5, y0, y0 + 1.1, z - 3.0, z + 3.0), WOOD)  # counter
        s.add("box", props.block(x - 3.5, x - 2.4, y0 + 1.1, y0 + 1.2, z - 3.1, z + 3.1), rgb("#6b4a35"))
        s.add("prism", chain(tr((x - 0.4, y0 + 2.8, z)), sc((7.2, 2.0, 7.6))), THATCH)
        for dz in np.linspace(-2.4, 2.4, 4):  # stools
            s.add("cyl", chain(tr((x - 4.2, y0, z + dz)), sc((0.45, 0.75, 0.45))), rgb("#6b4a35"))
        s.add("box", chain(tr((x - 3.0, y0 + 1.9, z)), sc((0.05, 0.5, 2.2))), rgb("#ff5aa0"), 4.0, GLOW, p2=1.0)  # neon
        top = np.array([x - 3.9, y0 + 2.9, z])
        for end in ((x - 3.9, y0 + 2.9, z - 9.0), (-1.0, float(sand_height(-1.0, 12.0)) + 4.5, 12.0),
                    (x - 3.9, y0 + 2.9, z + 3.5)):
            props.string_lights(s, top, end, sag=0.5, n=12)
        for dz in (-2.5, 0.0, 2.5):
            fires.append((x - 0.8, y0 + 3.0, z + dz, 3.2, 573 + 5 * abs(dz)))
            self.wood.append((x - 0.8, y0 + 3.4, z + dz))

    def _palms(self, rng, fires) -> None:
        # (x, z, height, lean dx, lean dz): two tall ones frame the sides, the rest line the beach.
        layout = [(-5.5, 9.0, 12.5, -2.5, -0.5), (6.5, 6.0, 12.0, 2.2, -1.0), (-9.5, -6.0, 11.0, 1.5, -2.0),
                  (12.0, -11.0, 10.5, -2.5, -2.0), (-15.0, -15.0, 12.0, 2.5, -1.0),
                  (20.0, -16.0, 9.5, -2.0, -1.5), (-26.0, -24.0, 10.5, 2.0, -1.5), (28.0, -30.0, 11.0, -1.5, -2.0)]
        for x, z, h, lx, lz in layout:
            y0 = float(sand_height(x, z))
            k = np.linspace(0, 1, 8)[:, None]
            pts = np.hstack([x + lx * k ** 1.7, y0 + h * k, z + lz * k ** 1.7])
            w = np.linspace(0.6, 0.38, 7)
            self.static.add("cyl5", _cyl_between(pts[:-1], pts[1:], w), rgb("#8a7258"))
            crown = pts[-1]
            self.static.add("ico0", chain(tr(crown + rng.uniform(-0.3, 0.3, (4, 3)) - (0, 0.35, 0)), sc(0.45)),
                            rgb("#6b4a2a"))  # coconuts
            for a in np.linspace(0, 2 * np.pi, 9, endpoint=False) + rng.uniform(0, 1):
                self.fronds.append((crown, a + rng.uniform(-0.15, 0.15), h * rng.uniform(0.27, 0.33)))
            fires.append((*crown - (0, 1.0, 0), 3.0, rng.uniform(572, 592)))
            fires.append((*(pts[3] - (0, 0.5, 0)), 1.6, rng.uniform(574, 595)))
            self.wood.append(tuple(crown))
        n = len(self.fronds)
        self.frond_crown = np.array([f[0] for f in self.fronds])
        self.frond_yaw = np.array([f[1] for f in self.fronds])
        self.frond_len = np.array([f[2] for f in self.fronds])
        self.frond_col = rng.uniform(0.85, 1.1, n)

    def _crowd(self, rng) -> None:
        def ground(x, z):
            return np.maximum(sand_height(x, z), -1.15)  # swimmers stand chest-deep

        lanes = [(40, (-16.0, 16.0), (-21.0, -15.0), 0),  # strolling at the waterline
                 (22, (-16.0, 16.0), (-32.0, -23.0), 0),  # swimming
                 (80, (-11.0, 11.0), (-12.0, 12.0), 0),  # among the umbrellas
                 (14, (6.0, 8.0), (-2.0, 6.0), 1)]  # at the bar
        self.crowd = props.Crowd(rng, lanes, y_at=ground, standing=0.45, exit_speed=5.0)
        p, n = self.crowd.p, self.crowd.n
        leave_f = rng.uniform(95, 150, n)
        p["leave"] = np.array([self.timeline.time_at_temp(f) for f in leave_f])
        p["exit"] = np.stack([np.clip(p["x"] * 0.6, -8, 8), np.zeros(n), np.full(n, 26.0)], 1)  # up to the hotel

    # Dynamic ----------------------------------------------------------------

    def power(self, temp: float) -> float:
        return 1.0 - ramp(temp, 255.0, 290.0)

    def dynamic(self, t: float, fx: Effects) -> Instances:
        d = Instances()
        temp = fx.temp_f
        self._draw_wash(d, t)
        self.crowd.draw(d, t)
        self._draw_fronds(d, temp)
        self._draw_umbrellas(d, temp)
        self._draw_loungers(d, temp)
        self._draw_floats(d, t, temp)
        steam = ramp(temp, 208.0, 240.0)
        if steam > 0:
            for z, size, n in ((-24.0, 0.9, 70), (-38.0, 1.4, 60), (-58.0, 2.2, 45)):  # wisps over the whole lagoon
                xs = np.linspace(-50, 50, 25)
                props.draw_puffs(d, xs, np.full_like(xs, z), -0.6 * size, t, steam, rgb("#f2f2f0"), rise=1.2 * size,
                                 spread=3.0 * size, size=size, n_max=n, seed=-z)
        smoke = ramp(temp, 505.0, 555.0) * (1 - ramp(temp, 575.0, 600.0))  # wood smokes before it ignites
        if smoke > 0:
            w = self.wood
            props.draw_puffs(d, w[:, 0], w[:, 2], w[:, 1], t, smoke, rgb("#8a827a"), rise=1.1, spread=0.7, size=0.4,
                             n_max=360)
        f = self.fires
        on = temp >= f["temp"]
        if on.any():
            grow = np.clip((temp - f["temp"][on]) / 25.0, 0.15, 1.0)
            big = np.where(f["temp"][on] >= 570, 1.5, 1.0)  # the wood fires roar
            props.draw_flames(d, f["x"][on], f["y"][on], f["z"][on], f["size"][on] * grow * big, f["seed"][on], t)
        return d

    def _draw_wash(self, d: Instances, t: float) -> None:
        """Foam sliding up and down the sand with each little wave."""
        xs = np.arange(-60.0, 60.1, 2.5)
        for k, (amp, speed, off) in enumerate(((1.2, 0.8, 0.0), (0.8, 0.8, 2.2))):
            z = shore_z(xs) + 0.6 + amp * np.sin(t * speed + off + xs * 0.04)
            y = np.maximum(sand_height(xs, z), 0.0) + 0.02
            d.add("box", _flat_strip(np.stack([xs, y, z], 1), 0.45 - 0.15 * k), rgb("#f6f8f8"), 0.1)

    def _draw_fronds(self, d: Instances, temp: float) -> None:
        curl = ramp(temp, 300.0, 450.0)
        gone = ramp(temp, 572.0, 586.0)
        scale = (1 - 0.35 * curl) * (1 - gone)
        if scale < 0.02:
            return
        color = _mix(rgb("#4f9a3a"), rgb("#7a4a1e"), ramp(temp, 220.0, 380.0))
        cols = np.array(color)[None] * self.frond_col[:, None]
        L = self.frond_len * scale
        n = len(L)
        yaw = self.frond_yaw
        # Each frond arches: a rising inner blade, then a drooping outer one (drooping more as it curls).
        up, down = math.radians(-22 + 30 * curl), math.radians(35 + 30 * curl)
        dir_in = np.stack([np.sin(yaw) * math.cos(up), np.full(n, -math.sin(up)), np.cos(yaw) * math.cos(up)], 1)
        mid = self.frond_crown + dir_in * (L * 0.5)[:, None]
        d.add("ico0", chain(tr(self.frond_crown + dir_in * (L * 0.25)[:, None]), ry(yaw), rx(np.full(n, up)),
                            sc(np.stack([L * 0.3, L * 0.05, L * 0.55], 1))), cols)
        dir_out = np.stack([np.sin(yaw) * math.cos(down), np.full(n, -math.sin(down)), np.cos(yaw) * math.cos(down)], 1)
        d.add("ico0", chain(tr(mid + dir_out * (L * 0.3)[:, None]), ry(yaw), rx(np.full(n, down)),
                            sc(np.stack([L * 0.26, L * 0.04, L * 0.65], 1))), cols * 0.9)

    def _draw_umbrellas(self, d: Instances, temp: float) -> None:
        melt = ramp(temp, 478.0, 515.0)  # polyester melts around 250-260°C
        for x, y0, z, color in self.umbrellas:
            c = _mix(color, (0.05, 0.04, 0.03), 0.35 * melt)
            top = y0 + 1.95 - 0.7 * melt
            d.add("cone", chain(tr((x, top, z)), sc((3.0 * (1 + 0.12 * melt), 0.6 * (1 - 0.6 * melt), 3.0 * (1 + 0.12 * melt)))),
                  c, 0.0, STRIPES, p1=8, p2=1)
            if melt > 0.02:  # the canopy slumps down the pole like wax
                d.add("cone", chain(tr((x, top + 0.05, z)), rx(math.pi), sc((3.0, 0.45 * melt, 3.0))), c, 0.0, STRIPES,
                      p1=8, p2=1)

    def _draw_loungers(self, d: Instances, temp: float) -> None:
        sag = ramp(temp, 310.0, 400.0)  # polypropylene softens and slumps
        white = _mix(rgb("#f4f2ee"), rgb("#d8cfae"), sag)
        for x, y0, z, yaw in self.loungers:
            base = chain(tr((x, y0, z)), ry(yaw))
            d.add("box", chain(base, sc((0.7 * (1 + 0.25 * sag), 0.35 * (1 - 0.65 * sag), 1.4))), white)
            d.add("box", chain(base, tr((0, 0.33 * (1 - 0.65 * sag), -0.65)), rx(-1.0 + 0.85 * sag), tr((0, 0, -0.35)),
                               sc((0.7, 0.07, 0.75))), white)

    def _draw_floats(self, d: Instances, t: float, temp: float) -> None:
        swell = 1 + 0.25 * ramp(temp, 150.0, 310.0)  # the air inside expands with absolute temperature
        for (pos, kind, color), burst in zip(self.floats, BURST_F):
            if temp >= burst:
                continue
            if kind == 0:
                r = 0.4 * swell
                d.add("ico", chain(tr(pos + (0, r * 0.95, 0)), sc(r * 2)), color, 0.0, STRIPES, p1=6, p2=1)
            else:
                d.add("cyl", chain(tr(pos), sc((1.1 * swell, 0.28 * swell, 1.1 * swell))), color, 0.0, STRIPES, p1=8, p2=1)

    def lights(self, t: float, fx: Effects, cam: np.ndarray, power: float) -> list:
        return []

    def camera(self, t: float, fx: Effects) -> tuple[np.ndarray, np.ndarray, float]:
        return CAMERA.copy(), LOOK.copy(), 60.0

    def sign_text(self, fx: Effects) -> str:
        return f"{round(fx.temp_f):,}°F"

    # Look -------------------------------------------------------------------

    def look(self, t: float, fx: Effects, pos, look_pt, fov) -> Frame:
        temp = fx.temp_f
        haze = ramp(temp, 150.0, 450.0)  # heat haze bleaches the blue out of the sky
        smoke = ramp(temp, 540.0, 620.0)

        def sky(a, b, c, scale=1.0):
            return _mix(_mix(rgb(a, scale), rgb(b, scale), haze), rgb(c, scale), smoke)

        sun = _dir(*SUN_AZ_EL)
        horizon = sky("#cfe9f2", "#f2e6c6", "#a85a34")
        frame = Frame(
            cam_pos=pos, cam_look=look_pt, fov=fov, time=t,
            zenith=sky("#1d5fc0", "#7f95b8", "#3a2216"), mid=sky("#58a4e4", "#d6cdb2", "#7a4026"), horizon=horizon,
            stars=0.0, moon_dir=_dir(-40, 25), moon_r=0.0, moon_col=(0, 0, 0), sun_dir=sun, body_dir=sun, body_d=1e6,
            body_frame=np.eye(3), flattening=0.0, rings=(0.0, 0.0), bands=((1, 1, 1),) * 3, ring_col=(1, 1, 1),
            body_bright=0.0, body_light=(0.0, 0.0, 0.0), glow_col=(0, 0, 0), glow=0.0,
            amb_sky=sky("#8fc0ee", "#d8ccb0", "#6a3a24", 0.55), amb_ground=sky("#e6d2aa", "#e8c890", "#5a3420", 0.4),
            fog_col=horizon, fog_density=0.0004 + 0.0008 * haze + 0.003 * ramp(temp, 213.0, 260.0) + 0.004 * smoke, power=self.power(temp), lava=0.0,
            exposure=1.0, flash=0.0, flash_col=(1, 1, 1), fade=fx.fade,
        )
        frame.body_type = -1
        frame.moon_vis = 0.0
        frame.lights = []
        frame.key_dir = sun
        frame.key_col = rgb("#fff0d8", 1.9 * (1 - 0.7 * smoke))
        frame.cloud = sky("#ffffff", "#f4ecd8", "#5a3a2a", 0.95)
        frame.cloud_amt = 1.0 - 0.8 * haze
        frame.dry = ramp(temp, 110.0, 260.0)
        frame.scorch = 0.92 * ramp(temp, 574.0, 625.0)
        frame.boil = ramp(temp, 213.0, 228.0)
        center = np.array([0.0, 0.0, -35.0])
        frame.shadow_vp = ortho(170.0, 10.0, 9000.0) @ look_at(center + sun * 4000.0, center)
        return frame

