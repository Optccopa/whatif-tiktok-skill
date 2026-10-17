"""The harbor scene: a seaside town at night, looking out to sea where the body rises.

Layout (meters): the camera mostly looks down -z, out to sea. The shore runs along z=0,
the promenade along z 14..26, the pier runs out along x=0 to a platform with a Ferris
wheel, the marina and the city sit to the right (x > 30), hills to the left, and a
mountain range lines the far side of the bay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from whatif.meshes import (
    BUILDING, FLAME, GLOW, GROUND, MOUNTAIN, SOLID, STRIPES, WATER, Instances, beam, chain, rgb, rx, ry, rz, sc, tr,
)
from whatif.scenario import Effects, Timeline, ramp


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


BODY_DIR = _unit((0.0, 0.1736, -0.985))  # straight out to sea, 10 degrees up
MOON_DIR = _unit((-0.208, 0.191, -0.959))  # up and to the left, beside it, so you can compare sizes
SUN_DIR = _unit((-0.45, -0.22, 1.0))  # below the horizon behind us: it's night, and the body is lit

LAND_Y = 1.5
PIER_Y = 3.5
PIER_HALF = 4.5
PIER_END = -112.0
PLATFORM = (-24.0, 24.0, -156.0, PIER_END)  # x0, x1, z0, z1
WHEEL = np.array([0.0, 22.5, -140.0])
WHEEL_R = 16.0
QUAY_X = 30.0
HEADLAND_Y = 3.0
HILL_FOOT = 30.0
HILL_SLOPE = 0.29
HILLS = ((-150, -40, 55, 120), (-230, -170, 90, 170), (-175, -290, 70, 140), (-320, 60, 110, 220), (-120, 120, 45, 90))
CAMPFIRE = np.array([-14.0, 0.0, 9.0])

SKIN = [rgb(c) for c in ("#f1c6a5", "#d9a27e", "#b07a55", "#8a5a3c", "#5e3b27")]
SHIRTS = [rgb(c) for c in ("#e8e4dc", "#d64545", "#3d7bd9", "#f2b134", "#4caf7a", "#9b59b6", "#f07f3c",
                           "#2c3e50", "#ff8fab", "#7fd1e8", "#ffffff", "#c0392b")]
PANTS = [rgb(c) for c in ("#2d3142", "#4f5d75", "#1f2a44", "#6b5b4b", "#3b3b3b", "#c9b79c")]
BULBS = [rgb(c) for c in ("#ffd59a", "#ff7b7b", "#ffe17a", "#7af0c8", "#8fd4ff", "#ff8ad8", "#fff3d6")]
WARM_BULB = rgb("#ffd08a")
BUILDINGS = [rgb(c) for c in ("#d9cbb3", "#c7a98a", "#a9b4c2", "#e3d7c3", "#b88f72", "#8e9aa8", "#cdb59a",
                              "#9c8a7a")]
GONDOLAS = [rgb(c) for c in ("#e85d5d", "#f2b134", "#4fa3e0", "#6cc070", "#b36ad8", "#f07f3c")]
AWNINGS = [rgb(c) for c in ("#d64545", "#3d7bd9", "#f2b134", "#4caf7a", "#e86fa0")]


def beach_y(z):
    return -1.2 + (np.asarray(z, dtype=float) + 6.0) * (2.7 / 20.0)


def seabed_y(z):
    z = np.asarray(z, dtype=float)
    return np.where(z < -170.0, -14.0, -1.2 + (z + 6.0) * (12.8 / 164.0))


def hill_y(z):
    """The hillside behind the promenade, climbing to the balcony we watch from."""
    z = np.asarray(z, dtype=float)
    return LAND_Y + np.clip(z - HILL_FOOT, 0.0, None) * HILL_SLOPE


def ground_y(x, z):
    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    on_pier = (np.abs(x) <= PIER_HALF) & (z >= PIER_END) & (z <= 14.0)
    on_platform = (x >= PLATFORM[0]) & (x <= PLATFORM[1]) & (z >= PLATFORM[2]) & (z <= PLATFORM[3])
    y = np.where(z >= 14.0, hill_y(z), np.where(z >= -6.0, beach_y(z), seabed_y(z)))
    y = np.where(x >= QUAY_X, LAND_Y, y)
    return np.where(on_pier | on_platform, PIER_Y, y)


def _slab(z0, y0, z1, y1, x0, x1, thickness=2.0) -> np.ndarray:
    """A box whose top surface runs from (z0, y0) to (z1, y1)."""
    length = math.hypot(z1 - z0, y1 - y0)
    angle = math.atan2(y1 - y0, z1 - z0)
    mid = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    return chain(tr(mid), rx(-angle), sc((x1 - x0, thickness, length)), tr((0, -1.0, 0)))


def _block(x0, x1, y0, y1, z0, z1) -> np.ndarray:
    return chain(tr(((x0 + x1) / 2, y0, (z0 + z1) / 2)), sc((x1 - x0, y1 - y0, z1 - z0)))


def _catenary(a, b, sag: float, n: int) -> np.ndarray:
    s = np.linspace(0, 1, n)[:, None]
    pts = np.asarray(a) * (1 - s) + np.asarray(b) * s
    pts[:, 1] -= sag * 4 * s[:, 0] * (1 - s[:, 0])
    return pts


@dataclass
class Shot:
    start: float
    end: float
    pos0: tuple
    pos1: tuple
    look0: tuple
    look1: tuple
    fov: float = 58.0


CAMERA = (-6.0, 40.0, 150.0)  # a balcony high above the town, looking out over the bay
LOOK = (0.0, 24.0, -300.0)

# One locked-off shot: the frame never changes, so every change in it reads at a glance.
SHOTS = [Shot(0.0, 1e9, CAMERA, CAMERA, LOOK, LOOK, 60.0)]


class Harbor:
    def __init__(self, timeline: Timeline, seed: int = 0):
        self.timeline = timeline
        rng = np.random.default_rng(seed)
        self.static = Instances()
        self.lamps: list[tuple] = []
        self.roofs: list[tuple] = []
        self._ground()
        self._pier()
        self._promenade(rng)
        self._beach(rng)
        self._city(rng)
        self._hillside(rng)
        self._landscape(rng)
        self._marina(rng)
        self._people(rng)
        self._boats(rng)
        self._fire_spots(rng)
        self._particles(rng)
        self._balcony()

    # Static -----------------------------------------------------------------

    def _ground(self) -> None:
        s = self.static
        s.add("box", _block(-700, QUAY_X, -20, LAND_Y, 14, HILL_FOOT + 0.5), rgb("#4b5446"), 0.0, GROUND)
        top = float(hill_y(400))
        s.add("box", _slab(HILL_FOOT, LAND_Y, 400, top, -700, QUAY_X, 60), rgb("#5d7146"), 0.0, GROUND)  # hillside
        s.add("box", _block(-700, QUAY_X, LAND_Y, LAND_Y + 0.05, 14, 26), rgb("#9a8f86"))  # promenade paving
        s.add("box", _block(QUAY_X, 900, -20, LAND_Y, -900, 800), rgb("#4f5552"), 0.0, GROUND)  # city ground
        s.add("box", _block(-900, -75, -20, HEADLAND_Y, -420, 14), rgb("#33463a"), 0.0, GROUND)  # left headland
        s.add("box", _block(QUAY_X, QUAY_X + 7, LAND_Y, LAND_Y + 0.05, -900, 14), rgb("#8d867e"))  # quay walk
        s.add("box", _slab(-6, -1.2, 14, 1.5, -700, QUAY_X), rgb("#d9c094"), 0.0, GROUND)  # beach
        s.add("box", _slab(-170, -14, -6, -1.2, -700, QUAY_X, 3.0), rgb("#6e604c"), 0.0, GROUND)  # sloping seabed
        s.add("box", _block(-9000, 9000, -40, -14, -9000, -170), rgb("#3f3a33"), 0.0, GROUND)  # deep seabed
        # Rocks scattered on the seabed, seen when the sea drains away.
        rng = np.random.default_rng(7)
        n = 90
        x = rng.uniform(-200, QUAY_X - 2, n)
        z = rng.uniform(-150, -10, n)
        size = rng.uniform(0.6, 2.4, n)
        mats = chain(tr(np.stack([x, seabed_y(z), z], 1)), ry(rng.uniform(0, 6, n)),
                     sc(np.stack([size * 1.4, size * 0.8, size], 1)))
        self.static.add("ico0", mats, rgb("#5b5348"))

    def _balcony(self) -> None:
        """The building we watch from, and its wooden railing across the bottom of the frame."""
        s = self.static
        cx, cy, cz = CAMERA
        s.add("box", _block(cx - 9, cx + 9, float(hill_y(cz)) - 2, cy - 2.1, cz + 0.5, cz + 18), rgb("#b8a590"), 0.0,
              BUILDING, seed=3.0, p1=0.3)
        s.add("box", _block(cx - 9, cx + 9, cy - 2.6, cy - 2.1, cz - 3.6, cz + 0.5), rgb("#6e655c"))
        wood, rail_z = rgb("#6b4a35"), cz - 3.2
        for y in (cy - 0.9, cy - 1.55):
            s.add("box", _block(cx - 4, cx + 4, y - 0.07, y + 0.07, rail_z - 0.07, rail_z + 0.07), wood)
        s.add("box", _block(cx - 1.05, cx - 0.89, cy - 2.1, cy - 0.8, rail_z - 0.08, rail_z + 0.08), wood)

    def _lamp(self, x, y, z, height=4.2, color=WARM_BULB, glow=9.0) -> None:
        self.lamps.append((x, y + height, z))
        self.static.add("cyl5", chain(tr((x, y, z)), sc((0.14, height, 0.14))), rgb("#2b2b33"))
        self.static.add("ico", chain(tr((x, y + height + 0.15, z)), sc(0.55)), color, glow, GLOW, p2=1.0)

    def _string_lights(self, a, b, sag=0.8, n=18, rng=None) -> None:
        pts = _catenary(a, b, sag, n)
        colors = [BULBS[(i * 3 + (rng.integers(0, 7) if rng is not None else 0)) % len(BULBS)] for i in range(n)]
        self.static.add("ico0", chain(tr(pts), sc(0.26)), colors, 10.0, GLOW, p2=1.0)

    def _pier(self) -> None:
        s = self.static
        wood, post = rgb("#7d6450"), rgb("#4e4038")
        s.add("box", _block(-PIER_HALF, PIER_HALF, PIER_Y - 0.6, PIER_Y, PIER_END, 14), wood, 0.0, STRIPES,
              p1=(14 - PIER_END) / 0.8, p2=2)
        x0, x1, z0, z1 = PLATFORM
        s.add("box", _block(x0, x1, PIER_Y - 0.6, PIER_Y, z0, z1), wood, 0.0, STRIPES, p1=(z1 - z0) / 0.8, p2=2)
        zs = np.arange(PIER_END, 14, 9.0)
        for px in (-PIER_HALF + 0.4, PIER_HALF - 0.4):
            s.add("cyl", chain(tr(np.stack([np.full_like(zs, px), np.full_like(zs, -16.0), zs], 1)),
                               sc((0.7, PIER_Y + 15.5, 0.7))), post)
        for px in np.arange(x0 + 2, x1, 8.0):
            for pz in (z0 + 1.5, z1 - 1.5, (z0 + z1) / 2):
                s.add("cyl", chain(tr((px, -16.0, pz)), sc((0.8, PIER_Y + 15.5, 0.8))), post)
        # Railings with lamps, and string lights zigzagging across the pier.
        for side in (-1, 1):
            rx_ = side * (PIER_HALF - 0.15)
            s.add("box", _block(rx_ - 0.08, rx_ + 0.08, PIER_Y + 1.0, PIER_Y + 1.12, PIER_END, 14), rgb("#d8d2c8"))
            posts = np.arange(PIER_END, 14, 3.0)
            s.add("box", chain(tr(np.stack([np.full_like(posts, rx_), np.full_like(posts, PIER_Y), posts], 1)),
                               sc((0.12, 1.1, 0.12))), rgb("#d8d2c8"))
        lamp_z = np.arange(PIER_END + 4, 14, 14.0)
        for i, z in enumerate(lamp_z):
            for side in (-1, 1):
                self._lamp(side * (PIER_HALF - 0.3), PIER_Y, z)
            if i + 1 < len(lamp_z):
                self._string_lights((-PIER_HALF + 0.3, PIER_Y + 4.3, z), (PIER_HALF - 0.3, PIER_Y + 4.3, lamp_z[i + 1]),
                                    sag=0.9, n=14)
        # Platform: carousel, stalls, lamps.
        cx, cz = -13.0, -124.0
        s.add("cyl", chain(tr((cx, PIER_Y, cz)), sc((10, 0.8, 10))), rgb("#c94f4f"))
        s.add("cyl", chain(tr((cx, PIER_Y + 0.8, cz)), sc((1.0, 4.2, 1.0))), rgb("#f2e6c9"))
        s.add("cone", chain(tr((cx, PIER_Y + 5.0, cz)), sc((11, 3.2, 11))), rgb("#d64545"), 0.0, STRIPES, p1=16, p2=1)
        ring = np.linspace(0, 2 * np.pi, 28, endpoint=False)
        s.add("ico0", chain(tr(np.stack([cx + 5.4 * np.cos(ring), np.full(28, PIER_Y + 5.0), cz + 5.4 * np.sin(ring)], 1)),
                            sc(0.3)), [BULBS[i % len(BULBS)] for i in range(28)], 10.0, GLOW, p2=1.0)
        for i, sx in enumerate((8.0, 14.0, 20.0)):
            self._stall(sx, PIER_Y, -116.5, np.pi, AWNINGS[i % len(AWNINGS)])
        for lx in (x0 + 1, x1 - 1):
            for lz in (z0 + 1, z1 - 1):
                self._lamp(lx, PIER_Y, lz, 5.0)
        # Ferris wheel legs (the wheel itself turns, so it's drawn each frame).
        for dx in (-7.5, 7.5):
            for dz in (-2.2, 2.2):
                s.add("box", beam((WHEEL[0] + dx, PIER_Y, WHEEL[2] + dz), WHEEL + (0, 0, dz * 0.6), 0.6),
                      rgb("#e8e6e1"))

    def _stall(self, x, y, z, yaw, awning) -> None:
        s = self.static
        base = chain(tr((x, y, z)), ry(yaw))
        s.add("box", chain(base, sc((4.2, 2.6, 3.0))), rgb("#efe6d8"))
        s.add("box", chain(base, tr((0, 2.6, 0)), sc((4.4, 0.25, 3.2))), rgb("#efe6d8"))
        s.add("box", chain(base, tr((0, 2.55, 2.0)), rx(0.45), sc((4.4, 0.12, 2.2))), awning, 0.0, STRIPES, p1=7)
        s.add("box", chain(base, tr((0, 1.1, 1.52)), sc((3.6, 1.2, 0.05))), rgb("#ffcf8a"), 2.2, GLOW, p2=1.0)

    def _promenade(self, rng) -> None:
        s = self.static
        # Sea wall and railing along the beach.
        s.add("box", _block(-700, QUAY_X, LAND_Y - 1.2, LAND_Y + 0.5, 13.6, 14.2), rgb("#b9ab98"))
        lamp_x = [x for x in np.arange(-110, QUAY_X, 12.0) if abs(x) > 6]
        for x in lamp_x:
            self._lamp(x, LAND_Y, 14.6)
        for a, b in zip(lamp_x[:-1], lamp_x[1:]):
            if b - a < 13:
                self._string_lights((a, LAND_Y + 4.3, 14.6), (b, LAND_Y + 4.3, 14.6), 0.9, 12, rng)
        # String lights across the promenade.
        for x in np.arange(-100, QUAY_X - 4, 9.0):
            if abs(x) > 8:
                self._string_lights((x, LAND_Y + 4.3, 14.6), (x + 5, LAND_Y + 4.6, 25.5), 0.7, 11, rng)
        for x in np.arange(-96, QUAY_X - 6, 7.0):
            if abs(x) > 8 and rng.random() < 0.75:
                self._stall(x, LAND_Y, 24.0, np.pi, AWNINGS[rng.integers(0, len(AWNINGS))])
        for x in np.arange(-100, QUAY_X, 10.0):
            if abs(x) > 7:
                s.add("box", chain(tr((x + 3, LAND_Y, 16.0)), sc((2.0, 0.5, 0.6))), rgb("#6d5544"))  # benches
        # Palm trees between the stalls.
        for x in np.arange(-104, QUAY_X - 2, 14.0) + 5:
            if abs(x) < 9:
                continue
            h = rng.uniform(7, 10)
            lean = rng.uniform(-0.15, 0.15)
            top = (x + h * math.sin(lean), LAND_Y + h, 21.0)
            s.add("cyl5", beam((x, LAND_Y, 21.0), top, 0.45), rgb("#6b5040"))
            for k in range(6):
                a = k * math.pi / 3 + rng.uniform(0, 0.4)
                tip = (top[0] + 3.2 * math.cos(a), top[1] - 1.4, top[2] + 3.2 * math.sin(a))
                s.add("box", beam(top, tip, 0.9), rgb("#2f6b45"))

    def _beach(self, rng) -> None:
        s = self.static
        for _ in range(12):
            x = rng.choice([rng.uniform(-70, -20), rng.uniform(8, 24)])
            z = rng.uniform(5, 12)
            y = float(beach_y(z))
            s.add("cyl5", chain(tr((x, y, z)), sc((0.1, 2.5, 0.1))), rgb("#dddddd"))
            s.add("cone", chain(tr((x, y + 2.1, z)), sc((3.2, 0.8, 3.2))), AWNINGS[rng.integers(0, 5)], 0.0,
                  STRIPES, p1=8, p2=1)
            s.add("box", chain(tr((x + 1.2, y + 0.02, z + 1.0)), ry(rng.uniform(-0.4, 0.4)), sc((1.0, 0.05, 1.9))),
                  SHIRTS[rng.integers(0, len(SHIRTS))])
        cx, cz = CAMPFIRE[0], CAMPFIRE[2]
        cy = float(beach_y(cz))
        for a in (0.4, 2.0):
            s.add("box", chain(tr((cx, cy, cz)), ry(a), sc((0.25, 0.25, 1.6)), tr((0, 0, -0.5))), rgb("#4a3426"))
        ring = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        s.add("ico0", chain(tr(np.stack([cx + 0.95 * np.cos(ring), np.full(9, cy), cz + 0.95 * np.sin(ring)], 1)),
                            sc(0.35)), rgb("#6d675f"))

    def _city(self, rng) -> None:
        """Waterfront city on the right, receding along the coast."""
        s = self.static
        for z in np.arange(-880, -150, 16.0):
            for col, x in enumerate(np.arange(QUAY_X + 12, 260, 18.0)):
                if rng.random() < 0.35:
                    continue
                w, d = rng.uniform(10, 16), rng.uniform(10, 15)
                h = rng.uniform(6, 14) if col == 0 else rng.uniform(8, 22)
                if z < -500 and rng.random() < 0.06:
                    h = rng.uniform(30, 45)  # a few towers far off
                bx, bz = x + rng.uniform(-1.5, 1.5), z + rng.uniform(-1.5, 1.5)
                color = BUILDINGS[rng.integers(0, len(BUILDINGS))]
                s.add("box", _block(bx - w / 2, bx + w / 2, LAND_Y, LAND_Y + h, bz - d / 2, bz + d / 2), color, 0.0,
                      BUILDING, seed=rng.uniform(0, 100), p1=rng.uniform(0.18, 0.5))
                s.add("box", _block(bx - w / 2 - 0.3, bx + w / 2 + 0.3, LAND_Y + h, LAND_Y + h + 0.6,
                                    bz - d / 2 - 0.3, bz + d / 2 + 0.3), tuple(c * 0.6 for c in color))
                if h > 30:
                    self.roofs.append((bx, LAND_Y + h + 0.6, bz, min(w, d)))
                if h > 60:
                    s.add("ico0", chain(tr((bx, LAND_Y + h + 1.5, bz)), sc(0.9)), rgb("#ff3b3b"), 14.0, GLOW, p2=1.0)
        for z in np.arange(-880, 14, 15.0):
            self._lamp(QUAY_X + 1.0, LAND_Y, z, 4.5)

    def _hillside(self, rng) -> None:
        """A grassy slope with a few cottages and trees, running down to the promenade."""
        s = self.static
        self.chimneys = []
        placed = []
        while len(placed) < 16:
            x, z = rng.uniform(-80, 24), rng.uniform(44, 100)
            if abs(x - CAMERA[0]) < 12 and z > 120 or any(math.hypot(x - px, z - pz) < 16 for px, pz in placed):
                continue
            placed.append((x, z))
            y = float(hill_y(z)) - 0.4
            w, d, h = rng.uniform(5.5, 8), rng.uniform(7, 10), rng.uniform(3.2, 4.2)
            yaw = rng.choice([0.0, np.pi / 2]) + rng.uniform(-0.15, 0.15)
            base = chain(tr((x, y, z)), ry(yaw))
            wall = rgb(rng.choice(["#9b3b2e", "#a8442f", "#e6dccb", "#8c3a2c", "#d8cdb8"]))
            s.add("box", chain(base, sc((w, h, d))), wall, 0.0, BUILDING, seed=rng.uniform(0, 100), p1=0.45)
            s.add("prism", chain(base, tr((0, h, 0)), sc((w + 0.6, 2.4, d + 0.6))), rgb("#3b2f2c"))
            cx_, cz_ = w * 0.25, d * 0.2
            chimney = np.array([x + cx_ * math.cos(yaw) + cz_ * math.sin(yaw), y + h + 2.6,
                                z - cx_ * math.sin(yaw) + cz_ * math.cos(yaw)])
            s.add("box", chain(tr(chimney - (0, 1.4, 0)), sc((0.6, 1.6, 0.6))), rgb("#5a4a42"))
            self.chimneys.append(chimney + (0, 0.2, 0))
        trees = 0
        while trees < 70:
            x, z = rng.uniform(-110, 28), rng.uniform(34, 150)
            in_view = z > 70 and abs(x - CAMERA[0]) < 0.36 * (CAMERA[2] - z) + 10
            if any(math.hypot(x - px, z - pz) < 7 for px, pz in placed) or in_view:
                continue
            trees += 1
            h = rng.uniform(5, 10)
            y = float(hill_y(z))
            s.add("cyl5", chain(tr((x, y - 0.3, z)), sc((0.35, h * 0.3, 0.35))), rgb("#4a3a2e"))
            s.add("cone6", chain(tr((x, y + h * 0.22, z)), ry(rng.uniform(0, 1)), sc((h * 0.5, h * 0.85, h * 0.5))),
                  rgb(rng.choice(["#1f3d2b", "#26472f", "#2d5233"])))
        # Low sheds along the quay, where the city used to crowd the frame.
        for z in np.arange(-140, 10, 22.0):
            if rng.random() < 0.7:
                w, d, h = rng.uniform(10, 16), rng.uniform(12, 18), rng.uniform(5, 9)
                x = QUAY_X + 14 + rng.uniform(-2, 6)
                s.add("box", _block(x - w / 2, x + w / 2, LAND_Y, LAND_Y + h, z - d / 2, z + d / 2),
                      rgb(rng.choice(["#8e9aa8", "#a9b4c2", "#b88f72"])), 0.0, BUILDING, seed=rng.uniform(0, 100), p1=0.2)

    def _landscape(self, rng) -> None:
        s = self.static
        # Hills on the left headland.
        for x, z, h, r in HILLS:
            s.add("cone6", chain(tr((x, -8, z)), ry(rng.uniform(0, 1)), sc((r, h + 8, r))), rgb("#2c3b35"))
        for _ in range(70):
            x, z = rng.uniform(-260, -80), rng.uniform(-260, 10)
            h = rng.uniform(6, 12)
            s.add("cone6", chain(tr((x, HEADLAND_Y, z)), sc((h * 0.55, h, h * 0.55))), rgb("#1f3a2c"))
        # Lighthouse on the point.
        lx, lz = -96.0, -236.0
        s.add("cone6", chain(tr((lx, -14, lz)), sc((40, 22, 40))), rgb("#4a4640"))
        for i in range(6):
            s.add("cyl", chain(tr((lx, 8 + i * 3.4, lz)), sc((4.0 - i * 0.25, 3.4, 4.0 - i * 0.25))),
                  rgb("#f2efe8") if i % 2 == 0 else rgb("#c0392b"))
        s.add("ico", chain(tr((lx, 30.5, lz)), sc(2.2)), rgb("#fff1c4"), 16.0, GLOW, p2=1.0)
        # Mountains across the bay, kept low straight ahead so the body clears them at the start.
        for _ in range(26):
            x = rng.uniform(-4500, 4500)
            z = rng.uniform(-4200, -2600)
            h = rng.uniform(200, 340) if abs(x) < 900 else rng.uniform(350, 800)
            r = rng.uniform(700, 1500)
            s.add("cone6", chain(tr((x, -20, z)), ry(rng.uniform(0, 1)), sc((r, h, r * 0.8))), rgb("#211d3a"), 0.0,
                  MOUNTAIN, seed=rng.uniform(0, 100))
        for _ in range(16):
            x = rng.uniform(-7000, 7000)
            h = rng.uniform(600, 1300)
            s.add("cone5", chain(tr((x, -20, rng.uniform(-7500, -5500))), sc((rng.uniform(1600, 2600), h, 1500))),
                  rgb("#2c2850"), 0.0, MOUNTAIN, seed=rng.uniform(0, 100))
        # Far shore lights.
        n = 160
        x = rng.uniform(-4000, 4000, n)
        z = rng.uniform(-2650, -2450, n)
        self.static.add("ico0", chain(tr(np.stack([x, rng.uniform(2, 14, n), z], 1)), sc(rng.uniform(3, 6, n))),
                        WARM_BULB, 6.0, GLOW, p2=1.0)

    def _marina(self, rng) -> None:
        s = self.static
        for z in (-30.0, -55.0, -80.0):
            s.add("box", _block(12, QUAY_X, 0.6, 1.1, z - 1.2, z + 1.2), rgb("#7a6a58"))
            xs = np.arange(14, QUAY_X, 5.0)
            s.add("cyl5", chain(tr(np.stack([xs, np.full_like(xs, -10.0), np.full_like(xs, z + 1.3)], 1)),
                                sc((0.35, 11.6, 0.35))), rgb("#4e4038"))
        # Breakwater.
        n = 70
        x = np.linspace(40, 150, n) + rng.uniform(-2, 2, n)
        z = -140 + rng.uniform(-3, 3, n)
        size = rng.uniform(3, 6, n)
        s.add("ico0", chain(tr(np.stack([x, rng.uniform(-3, -1, n), z], 1)), ry(rng.uniform(0, 6, n)),
                            sc(np.stack([size, size * 0.8, size], 1))), rgb("#5c5752"))
        s.add("cyl", chain(tr((150, 0, -140)), sc((2.4, 9, 2.4))), rgb("#e8e2d6"))
        s.add("ico", chain(tr((150, 9.4, -140)), sc(1.2)), rgb("#7dff9a"), 12.0, GLOW, p2=1.0)

    # Dynamic ----------------------------------------------------------------

    def _people(self, rng) -> None:
        groups = []

        def group(n, x, z, axis, lo, hi, speed, sitting=False):
            groups.append(dict(x=x, z=z, axis=np.full(n, axis), lo=np.full(n, lo), hi=np.full(n, hi),
                               speed=speed, sit=np.full(n, sitting)))

        n = 80
        group(n, rng.uniform(-90, 26, n), rng.uniform(15.5, 23, n), 0, -95, 27, rng.uniform(0.9, 1.5, n))
        n = 46
        group(n, rng.uniform(-3.4, 3.4, n), rng.uniform(-108, 12, n), 1, -108, 12, rng.uniform(0.9, 1.4, n))
        n = 22
        group(n, rng.uniform(-22, 20, n), rng.uniform(-150, -116, n), 0, -22, 22, rng.uniform(0.5, 1.1, n))
        n = 18
        group(n, rng.uniform(-80, 24, n), rng.uniform(3.5, 12, n), 0, -85, 26, rng.uniform(0.4, 0.9, n))
        n = 26
        group(n, rng.uniform(QUAY_X + 1.5, QUAY_X + 6, n), rng.uniform(-200, 12, n), 1, -200, 12, rng.uniform(0.9, 1.4, n))
        a = np.linspace(0, 2 * np.pi, 7, endpoint=False) + 0.3
        group(7, CAMPFIRE[0] + 2.2 * np.cos(a), CAMPFIRE[2] + 2.2 * np.sin(a), 0, -1e9, 1e9, np.zeros(7), True)
        n = 16
        bx, bz = rng.uniform(-60, 22, n), rng.uniform(5, 12, n)
        group(n, bx, bz, 0, -1e9, 1e9, np.zeros(n), True)

        p = {k: np.concatenate([g[k] for g in groups]) for k in groups[0]}
        count = len(p["x"])
        p["dir"] = rng.choice([-1.0, 1.0], count)
        p["phase"] = rng.uniform(0, 2 * np.pi, count)
        p["height"] = rng.uniform(0.88, 1.08, count)
        p["stop"] = 20.0 + rng.uniform(0, 4.5, count)
        p["flee_delay"] = rng.uniform(0, 2.5, count)
        p["run"] = rng.uniform(3.5, 5.5, count)
        sit_yaw = np.arctan2(CAMPFIRE[0] - p["x"], CAMPFIRE[2] - p["z"])
        p["sit_yaw"] = np.where(np.hypot(p["x"] - CAMPFIRE[0], p["z"] - CAMPFIRE[2]) < 3, sit_yaw, np.pi)
        p["skin"] = np.array([SKIN[i] for i in rng.integers(0, len(SKIN), count)])
        p["shirt"] = np.array([SHIRTS[i] for i in rng.integers(0, len(SHIRTS), count)])
        p["pants"] = np.array([PANTS[i] for i in rng.integers(0, len(PANTS), count)])
        self.people = p

    def _boats(self, rng) -> None:
        xs, zs, yaws, lengths, sails = [], [], [], [], []
        for dock_z in (-30.0, -55.0, -80.0):
            for side in (-1, 1):
                for x in np.arange(16, QUAY_X - 3, 6.5):
                    if rng.random() < 0.8:
                        xs.append(x + rng.uniform(-0.5, 0.5))
                        zs.append(dock_z + side * 4.2)
                        yaws.append(np.pi / 2 + side * 0.04 + np.pi * (rng.random() < 0.5))
                        lengths.append(rng.uniform(7, 10))
                        sails.append(rng.random() < 0.6)
        for _ in range(9):
            xs.append(rng.uniform(-260, 220))
            zs.append(rng.uniform(-620, -200))
            yaws.append(rng.uniform(0, 2 * np.pi))
            lengths.append(rng.uniform(10, 16))
            sails.append(rng.random() < 0.7)
        n = len(xs)
        self.boats = dict(x=np.array(xs), z=np.array(zs), yaw=np.array(yaws), length=np.array(lengths),
                          sail=np.array(sails), phase=rng.uniform(0, 6, n), drift=rng.uniform(0.6, 1.4, n),
                          color=np.array([rgb(c) for c in rng.choice(["#f4f1ea", "#e8e4dc", "#2c3e50", "#c0392b",
                                                                      "#2f6db5"], n)]))

    def _fire_spots(self, rng) -> None:
        """Fires break out where the flood can't reach: rooftops and the hills."""
        pts = []
        roofs = [self.roofs[i] for i in rng.permutation(len(self.roofs))[:60]]
        for x, y, z, w in roofs:
            pts.append((x + rng.uniform(-w, w) * 0.3, y, z + rng.uniform(-w, w) * 0.3, rng.uniform(4, 9)))
        for hx, hz, h, r in HILLS:
            for _ in range(10):
                f = rng.uniform(0.15, 0.6)
                a = rng.uniform(0, 2 * np.pi)
                pts.append((hx + r * f * 0.5 * math.cos(a), -8 + (h + 8) * (1 - f), hz + r * f * 0.5 * math.sin(a),
                            rng.uniform(5, 12)))
        a = np.array(pts)
        self.fires = dict(x=a[:, 0], y=a[:, 1], z=a[:, 2], size=a[:, 3], threshold=rng.uniform(0.02, 1.0, len(a)),
                          seed=rng.uniform(0, 100, len(a)))

    def lights(self, t: float, fx: Effects, cam: np.ndarray, power: float) -> list[tuple]:
        """Up to 8 point lights near the camera: (x, y, z, radius, r, g, b)."""
        out = []
        cy = float(beach_y(CAMPFIRE[2]))
        if fx.water < cy:
            k = 3.2 * (0.85 + 0.15 * math.sin(t * 11.0) * math.sin(t * 7.3))
            out.append((CAMPFIRE[0], cy + 0.9, CAMPFIRE[2], 11.0, 1.0 * k, 0.5 * k, 0.18 * k))
        f = self.fires
        on = np.nonzero(f["threshold"] < fx.fire)[0]
        if len(on):
            d = np.hypot(f["x"][on] - cam[0], f["z"][on] - cam[2])
            for i in on[np.argsort(d)[:4]]:
                k = 2.5 * (0.85 + 0.15 * math.sin(t * 9.0 + f["seed"][i]))
                out.append((f["x"][i], f["y"][i] + f["size"][i] * 0.5, f["z"][i], f["size"][i] * 5,
                            1.0 * k, 0.45 * k, 0.12 * k))
        if power > 0.05:
            lamps = np.array(self.lamps)
            lit = lamps[:, 1] > fx.water
            lamps = lamps[lit]
            d = np.linalg.norm(lamps - cam, axis=1)
            for x, y, z in lamps[np.argsort(d)[: 8 - len(out)]]:
                out.append((x, y - 0.3, z, 11.0, 1.0 * power, 0.72 * power, 0.42 * power))
        return out[:8]

    def _particles(self, rng) -> None:
        T = self.timeline.T
        start = self.timeline.t_rings
        n = 220
        az = rng.uniform(-0.75, 0.75, n)
        el = rng.uniform(0.25, 1.15, n)
        dist = rng.uniform(1300, 3600, n)
        direction = np.stack([np.sin(az) * np.cos(el), np.sin(el), -np.cos(az) * np.cos(el)], 1)
        self.meteors = dict(
            birth=np.sort(rng.uniform(start - 0.3, max(T, start) + 0.3, n)), p0=direction * dist[:, None],
            v=np.stack([rng.uniform(-380, 380, n), rng.uniform(-1150, -650, n), rng.uniform(100, 650, n)], 1),
            width=rng.uniform(1.6, 5.0, n) * np.where(rng.random(n) < 0.12, 2.5, 1.0), life=rng.uniform(1.0, 1.8, n),
        )
        n = 320
        x = rng.uniform(-170, 170, n)
        z = rng.uniform(-260, 40, n)  # never right in front of the balcony
        self.debris = dict(x=x, z=z, y=ground_y(x, z), start=T - 3.6 + rng.uniform(0, 3.2, n), accel=rng.uniform(5, 16, n),
                           size=rng.uniform(0.4, 2.8, n), spin=rng.uniform(-3, 3, (n, 3)),
                           color=np.array([rgb(c) for c in rng.choice(["#6b6259", "#4f4842", "#8a7a68", "#3e3a36"], n)]))

    def dynamic(self, t: float, fx: Effects) -> Instances:
        d = Instances()
        self._draw_water(d, t, fx)
        self._draw_wheel(d, t, fx)
        self._draw_people(d, t, fx)
        self._draw_boats(d, t, fx)
        self._draw_fires(d, t, fx)
        self._draw_meteors(d, t)
        self._draw_debris(d, t, fx)
        return d

    def _draw_water(self, d: Instances, t: float, fx: Effects) -> None:
        d.add("quad", chain(tr((0, fx.water, -4100)), sc((18000, 1, 9800))), rgb("#1b3048"), 0.0, WATER)

    def _draw_wheel(self, d: Instances, t: float, fx: Effects) -> None:
        tilt = 0.32 * ramp(t, self.timeline.t_flood + 2.5, self.timeline.t_flood + 9)
        pivot = chain(tr((WHEEL[0], PIER_Y, WHEEL[2])), rz(-tilt), tr((0, WHEEL[1] - PIER_Y, 0)))
        spin = 0.09 * t
        a = spin + np.linspace(0, 2 * np.pi, 36, endpoint=False)
        rim = np.stack([WHEEL_R * np.cos(a), WHEEL_R * np.sin(a), np.zeros_like(a)], 1)
        rim_next = np.roll(rim, -1, axis=0)
        white = rgb("#eceae4")
        for dz in (-0.9, 0.9):
            off = np.array([0, 0, dz])
            d.add("box", np.matmul(pivot, beam(rim + off, rim_next + off, 0.32)), white)
        spokes = rim[::2]
        d.add("box", np.matmul(pivot, beam(np.zeros_like(spokes), spokes, 0.14)), white)
        d.add("ico", np.matmul(pivot, sc(1.8)), white)
        mid = (rim + rim_next) / 2
        colors = [BULBS[i % len(BULBS)] for i in range(len(mid))]
        d.add("ico0", np.matmul(pivot, chain(tr(mid + (0, 0, 1.0)), sc(0.45))), colors, 9.0, GLOW, p2=1.0)
        d.add("ico0", np.matmul(pivot, chain(tr(rim[::2] + (0, 0, 1.0)), sc(0.5))), WARM_BULB, 9.0, GLOW, p2=1.0)
        hang = rim[1::2]
        cabins = chain(tr(hang), rz(np.full(len(hang), tilt)), tr((0, -2.6, 0)), sc((1.7, 1.9, 1.7)))
        d.add("box", np.matmul(pivot, cabins), [GONDOLAS[i % len(GONDOLAS)] for i in range(len(hang))])
        lit = chain(tr(hang), rz(np.full(len(hang), tilt)), tr((0, -1.9, 0.86)), sc((1.2, 0.7, 0.05)))
        d.add("box", np.matmul(pivot, lit), rgb("#ffe0a8"), 3.0, GLOW, p2=1.0)

    def _draw_people(self, d: Instances, t: float, fx: Effects) -> None:
        p = self.people
        walking = ~p["sit"]
        stop = p["stop"]
        tau = np.where(t < stop, t, stop + 0.6 * (1 - np.exp(-(np.maximum(t - stop, 0)) / 0.6)))
        span = p["hi"] - p["lo"]
        start = np.where(p["axis"] == 0, p["x"], p["z"])
        s = (start - p["lo"]) + p["dir"] * p["speed"] * tau
        m = np.mod(s, 2 * span)
        forward = m < span
        coord = p["lo"] + np.where(forward, m, 2 * span - m)
        heading = p["dir"] * np.where(forward, 1.0, -1.0)
        x = np.where(p["axis"] == 0, coord, p["x"])
        z = np.where(p["axis"] == 1, coord, p["z"])
        walk_yaw = np.where(p["axis"] == 0, heading * np.pi / 2, np.where(heading > 0, 0.0, np.pi))
        turn = np.clip((t - stop) / 1.2, 0, 1)
        yaw = walk_yaw + _angle_diff(np.pi, walk_yaw) * (turn * turn * (3 - 2 * turn))
        moving = np.where(t < stop, 1.0, np.exp(-np.maximum(t - stop, 0) / 0.4))
        amp = 0.5 * moving * np.minimum(p["speed"] / 1.2, 1.0)
        phase = p["phase"] + tau * p["speed"] * 2.4
        sit = p["sit"].copy()
        yaw = np.where(sit, p["sit_yaw"], yaw)

        flee_t = fx.flee + p["flee_delay"]
        fleeing = t >= flee_t
        run_time = np.maximum(t - flee_t, 0)
        safe_z = 96.0  # high enough up the hill; they stop and watch from there
        reached = np.where(fleeing, z + p["run"] * run_time >= safe_z, False)
        z = np.where(fleeing, np.minimum(z + p["run"] * run_time, safe_z), z)
        x = np.where(fleeing & ~walking, x + 0.3 * run_time, x)
        yaw = np.where(fleeing, 0.0, yaw)
        amp = np.where(fleeing, np.where(reached, 0.0, 0.85), amp)
        yaw = np.where(reached, np.pi, yaw)
        phase = np.where(fleeing, p["phase"] + run_time * 9.0, phase)
        sit = sit & ~fleeing

        hs = p["height"]
        y = ground_y(x, z) - np.where(sit, 0.5 * hs, 0.0)
        base = chain(tr(np.stack([x, y, z], 1)), ry(yaw))
        n = len(x)
        swing = amp * np.sin(phase)
        leg = np.where(sit, -np.pi / 2, 0.0)
        zero = np.zeros(n)
        for side, sgn in ((-1, 1), (1, -1)):
            d.add("limb", chain(base, tr(np.stack([side * 0.1 * hs, 0.92 * hs, zero], 1)), rx(leg + sgn * swing),
                                sc(np.stack([0.15 * hs, 0.9 * hs, 0.16 * hs], 1))), p["pants"])
            d.add("limb", chain(base, tr(np.stack([side * 0.26 * hs, 1.46 * hs, zero], 1)),
                                rx(-sgn * swing * 0.8 + np.where(sit, -0.5, 0.0)),
                                sc(np.stack([0.1 * hs, 0.6 * hs, 0.11 * hs], 1))), p["shirt"])
        d.add("box", chain(base, tr(np.stack([zero, 0.88 * hs, zero], 1)), sc(np.stack([0.4 * hs, 0.62 * hs, 0.24 * hs], 1))),
              p["shirt"])
        d.add("ico", chain(base, tr(np.stack([zero, 1.66 * hs, zero], 1)), sc(0.27 * hs)), p["skin"])

    def _draw_boats(self, d: Instances, t: float, fx: Effects) -> None:
        b = self.boats
        n = len(b["x"])
        surge = max(fx.water, 0.0) if t > self.timeline.t_tide_surge else 0.0
        z = b["z"] + np.minimum(surge * 4.5 * b["drift"], 120.0)
        x = b["x"] + np.sin(b["phase"]) * surge * 0.8
        floor = seabed_y(z) + 0.9
        stranded = fx.water + 0.0 < floor
        y = np.maximum(fx.water, floor) + np.where(stranded, 0.0, 0.18 * np.sin(t * 1.3 + b["phase"]))
        roll = np.where(stranded, 0.28, 0.05 * np.sin(t * 0.9 + b["phase"]) + 0.12 * min(surge / 10, 1) * np.sin(t * 2.1 + b["phase"]))
        pitch = 0.03 * np.sin(t * 0.7 + b["phase"] * 2)
        base = chain(tr(np.stack([x, y - 0.9, z], 1)), ry(b["yaw"]), rz(roll), rx(pitch))
        L = b["length"]
        W = L * 0.34
        d.add("hull", chain(base, sc(np.stack([W, np.full(n, 1.5), L], 1))), b["color"])
        d.add("box", chain(base, tr(np.stack([np.zeros(n), np.full(n, 1.5), 0.08 * L], 1)),
                           sc(np.stack([W * 0.62, np.full(n, 1.3), L * 0.36], 1))), rgb("#f2efe8"))
        d.add("box", chain(base, tr(np.stack([np.zeros(n), np.full(n, 2.1), 0.08 * L + L * 0.18 + 0.01], 1)),
                           sc(np.stack([W * 0.5, np.full(n, 0.4), np.full(n, 0.05)], 1))), rgb("#ffd59a"), 3.0, GLOW, p2=1.0)
        sail = b["sail"]
        if sail.any():
            mast = base[sail]
            k = int(sail.sum())
            d.add("cyl5", chain(mast, tr((0, 1.5, -0.1 * 9)), sc(np.stack([np.full(k, 0.14), L[sail] * 1.1,
                                                                          np.full(k, 0.14)], 1))), rgb("#d8d2c8"))
            d.add("ico0", chain(mast, tr(np.stack([np.zeros(k), 1.5 + L[sail] * 1.1, np.full(k, -0.9)], 1)), sc(0.35)),
                  rgb("#ff5a5a"), 10.0, GLOW, p2=1.0)

    def _draw_fires(self, d: Instances, t: float, fx: Effects) -> None:
        f = self.fires
        on = f["threshold"] < fx.fire
        cx, cz = CAMPFIRE[0], CAMPFIRE[2]
        cy = float(beach_y(cz))
        xs, ys, zs, sizes, seeds = [cx], [cy + 0.1], [cz], [1.0], [3.0]
        if fx.water < cy:
            xs, ys, zs, sizes, seeds = list(xs), list(ys), list(zs), list(sizes), list(seeds)
        else:
            xs, ys, zs, sizes, seeds = [], [], [], [], []
        grow = np.clip((fx.fire - f["threshold"]) * 5, 0, 1)
        xs += list(f["x"][on]); ys += list(f["y"][on]); zs += list(f["z"][on])
        sizes += list(f["size"][on] * (0.3 + 0.7 * grow[on])); seeds += list(f["seed"][on])
        if not xs:
            return
        x, y, z, size, seed = (np.array(v, dtype=float) for v in (xs, ys, zs, sizes, seeds))
        for k, (dx, dz, hk) in enumerate(((0, 0, 1.0), (0.35, 0.2, 0.7), (-0.3, -0.25, 0.6))):
            flick = 0.8 + 0.18 * np.sin(t * 9.1 + seed * (k + 1)) + 0.12 * np.sin(t * 15.3 + seed * 2.3 + k)
            h = size * hk * flick
            mats = chain(tr(np.stack([x + dx * size, y, z + dz * size], 1)), ry(seed + t * 2.0 * (k + 1)),
                         sc(np.stack([h * 0.5, h, h * 0.5], 1)))
            d.add("cone5", mats, rgb("#ff8a2a"), 7.0, FLAME)
        # Embers drifting up.
        m = len(x)
        e = np.arange(m * 4)
        src = e % m
        cycle = 2.8
        age = np.mod(t + e * 0.73, cycle)
        ex = x[src] + np.sin(e * 1.7 + t) * 0.6 * size[src] * 0.3 + age * 0.4
        ey = y[src] + size[src] * 0.4 + age * 2.2 * (0.6 + size[src] * 0.1)
        ez = z[src] + np.cos(e * 2.3) * 0.5
        d.add("ico0", chain(tr(np.stack([ex, ey, ez], 1)), sc(0.12 + 0.02 * size[src])), rgb("#ffb35a"),
              10.0 * (1 - age / cycle), GLOW)

    def _draw_meteors(self, d: Instances, t: float) -> None:
        m = self.meteors
        age = t - m["birth"]
        alive = (age >= 0) & (age < m["life"])
        if not alive.any():
            return
        p = m["p0"][alive] + m["v"][alive] * age[alive, None]
        above = p[:, 1] > 0
        p, v = p[above], m["v"][alive][above]
        if not len(p):
            return
        fade = 1 - age[alive][above] / m["life"][alive][above]
        tail = p - v * 0.09
        d.add("box", beam(tail, p, m["width"][alive][above]), rgb("#dfeaff"), 14.0 * fade, GLOW)
        d.add("ico0", chain(tr(p), sc(m["width"][alive][above] * 1.6)), rgb("#fff4e0"), 18.0 * fade, GLOW)

    def _draw_debris(self, d: Instances, t: float, fx: Effects) -> None:
        if fx.debris <= 0:
            return
        r = self.debris
        dt = t - r["start"]
        up = dt > 0
        if not up.any():
            return
        dt = dt[up]
        rise = 0.5 * r["accel"][up] * dt * dt
        pos = np.stack([r["x"][up] + rise * BODY_DIR[0] * 0.5, r["y"][up] + rise, r["z"][up] + rise * BODY_DIR[2] * 0.5], 1)
        spin = r["spin"][up] * dt[:, None]
        size = r["size"][up]
        d.add("ico0", chain(tr(pos), rx(spin[:, 0]), ry(spin[:, 1]), rz(spin[:, 2]),
                            sc(np.stack([size, size * 0.7, size * 0.9], 1))), r["color"][up])

    # Camera -----------------------------------------------------------------

    def camera(self, t: float, fx: Effects) -> tuple[np.ndarray, np.ndarray, float]:
        shot = next((s for s in SHOTS if s.start <= t < s.end), SHOTS[-1])
        u = min(max((t - shot.start) / (shot.end - shot.start), 0.0), 1.0)
        pos = np.array(shot.pos0, dtype=float) * (1 - u) + np.array(shot.pos1, dtype=float) * u
        look = np.array(shot.look0, dtype=float) * (1 - u) + np.array(shot.look1, dtype=float) * u
        if fx.shake > 0:  # the ground shakes, but we never leave the balcony: only the view trembles
            k = fx.shake
            look = look + k * 4.5 * np.array([math.sin(t * 19.7 + 2), math.sin(t * 27.1) + math.sin(t * 41.3),
                                              0.0])
        return pos, look, shot.fov


def _angle_diff(a, b):
    return np.mod(a - b + np.pi, 2 * np.pi) - np.pi
