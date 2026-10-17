"""The village scene: a snowy ski village at dusk, seen from a chalet balcony above the valley.

The camera looks up the valley (-z). The village sits on the valley floor below; a ski mountain
with lit pistes and a gondola rises on the right, a forested peak on the left, and a snowy range
closes the valley far off, low enough for the sky above the saddle to carry the story.
Terrain is a heightfield (snow everywhere but the steepest rock); trees are instances.
"""

from __future__ import annotations

import math

import numpy as np

from whatif.forest import _smooth, fbm
from whatif.meshes import BUILDING, GLOW, Instances, beam, chain, rgb, ry, sc, tr
from whatif.scenario import Effects, Timeline, ramp

TERRAIN = 8
BODY_AZ_EL = (0.0, 15.0)
MOON_AZ_EL = (-9.0, 11.0)
SKI = (330.0, -1600.0)  # the ski mountain's summit (x, z), kept left of TikTok's button column
PISTES = (((280.0, -1540.0), (200.0, -1200.0), (110.0, -820.0)),  # (x, z) waypoints, top to bottom
          ((345.0, -1520.0), (330.0, -1200.0), (245.0, -860.0)))
PISTE_HALF = 22.0
VILLAGE = (-240.0, 240.0, -760.0, -190.0)  # x0, x1, z0, z1
LIFT = ((170.0, -760.0), (305.0, -1490.0))  # gondola: base and top station (x, z)
WINDOW_WARM = rgb("#ffcf8a")


def terrain_height(x, z):
    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    d = np.maximum(-z, 0.0)
    h = 5.0 + 42.0 * _smooth((z + 150.0) / 230.0)  # our hillside, down to the valley floor
    h = h + 300.0 * _smooth((z - 60.0) / 900.0)  # behind us it keeps climbing (and shades the valley at dusk)
    wall = np.maximum(np.abs(x) - 130.0, 0.0)
    h = h + np.minimum(0.0016 * wall * wall, 320.0) * (0.55 + 0.45 * _smooth(d / 1500.0))  # a U-shaped valley
    h = h + 400.0 * np.exp(-(((x - SKI[0]) / 400.0) ** 2 + ((z - SKI[1]) / 520.0) ** 2))  # the ski mountain
    h = h + 470.0 * np.exp(-(((x + 430.0) / 380.0) ** 2 + ((z + 1850.0) / 540.0) ** 2))  # the forested peak
    ridges = 1.0 - np.abs(2.0 * fbm(x / 900.0, z / 900.0, 5.0) - 1.0)  # sharp crests
    h = h + _smooth((d - 3000.0) / 2600.0) * (330.0 + 240.0 * fbm(x / 1300.0, z / 1300.0, 3.0) + 430.0 * ridges ** 2)
    floor = _smooth((np.abs(x) - 60.0) / 300.0) + _smooth((d - 900.0) / 600.0)  # keep the village floor flat
    h = h + 16.0 * (fbm(x / 260.0, z / 260.0, 1.0) - 0.5) * np.minimum(floor + 0.15, 1.0)
    h = h + 80.0 * (fbm(x / 1100.0, z / 1100.0, 7.0) - 0.5) * _smooth((d - 700.0) / 900.0)
    return h


def terrain_mesh() -> np.ndarray:
    """Flat-shaded triangles: fine over the village and the ski mountain, coarser toward the range."""
    zs = np.concatenate([np.arange(1500.0, 300.0, -60.0), np.arange(300.0, -2400.0, -12.0),
                         np.arange(-2400.0, -9500.0, -55.0)])
    xs = np.concatenate([np.arange(-4000.0, -1400.0, 50.0), np.arange(-1400.0, 1400.0, 14.0),
                         np.arange(1400.0, 4001.0, 50.0)])
    X, Z = np.meshgrid(xs, zs)
    Y = terrain_height(X, Z)
    P = np.stack([X, Y, Z], -1)
    a, b, c, d = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]
    tris = np.stack([np.stack([a, b, d], -2), np.stack([a, d, c], -2)], 2)
    return tris.reshape(-1, 3).astype(np.float32)


def _piste_distance(x, z):
    """Horizontal distance to the nearest piste centerline."""
    best = np.full(np.shape(x), 1e9)
    for pts in PISTES:
        for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:]):
            dx, dz = x1 - x0, z1 - z0
            k = np.clip(((x - x0) * dx + (z - z0) * dz) / (dx * dx + dz * dz), 0.0, 1.0)
            best = np.minimum(best, np.hypot(x - x0 - k * dx, z - z0 - k * dz))
    return best


def avalanche(timeline: Timeline, t: float) -> float:
    """0..1: powder pouring off the slopes once the tides shake the mountains."""
    return ramp(t, timeline.t_flood - 1.5, timeline.t_flood + 9.0)


class Village:
    body_az_el = BODY_AZ_EL
    moon_az_el = MOON_AZ_EL

    def __init__(self, timeline: Timeline, seed: int = 0):
        self.timeline = timeline
        rng = np.random.default_rng(seed + 23)
        self.camera_pos = np.array([0.0, float(terrain_height(0.0, 0.0)) + 7.0, 0.0])
        self.look_at = self.camera_pos + np.array([0.0, -math.tan(math.radians(3.5)) * 1000.0, -1000.0])
        self.meshes = {"terrain": terrain_mesh()}
        self.static = Instances()
        self.static.add("terrain", np.eye(4), rgb("#e8ecf2"), 0.0, TERRAIN)
        self.lamps: list[tuple] = []
        self._village(rng)
        self._trees(rng)
        self._pistes()
        self._lift()
        self._railing()

    # Static -----------------------------------------------------------------

    def _chalet(self, x, z, yaw, w, d, h, wall, lit) -> None:
        y = float(terrain_height(x, z)) - 0.6
        base = chain(tr((x, y, z)), ry(yaw))
        s = self.static
        s.add("box", chain(base, sc((w, h, d))), wall, 0.0, BUILDING, seed=float(x * 0.37 + z * 0.11) % 100, p1=lit)
        s.add("prism", chain(base, tr((0, h, 0)), sc((w + 1.6, 0.42 * w, d + 1.6))), rgb("#3a2a22"))
        cx = w * 0.22
        chimney = (x + cx * math.cos(yaw), y + h + 0.42 * w * 0.6, z - cx * math.sin(yaw))
        s.add("box", chain(tr(chimney), sc((0.9, 1.8, 0.9))), rgb("#4d4038"))

    def _village(self, rng) -> None:
        x0, x1, z0, z1 = VILLAGE
        placed = []
        walls = [rgb(c) for c in ("#5a3a26", "#6b4530", "#4e3322", "#7a5236", "#d8cfc2", "#c9b8a2")]
        tries = 0
        while len(placed) < 46 and tries < 4000:
            tries += 1
            x, z = rng.uniform(x0, x1), rng.uniform(z0, z1)
            if abs(z + 430) < 12 or abs(x + 60) < 22 and abs(z + 470) < 26:  # the main street and the church
                continue
            if any(math.hypot(x - px, z - pz) < 26 for px, pz in placed):
                continue
            placed.append((x, z))
            w, d = rng.uniform(10, 15), rng.uniform(8, 12)
            yaw = rng.choice([0.0, np.pi / 2]) + rng.uniform(-0.12, 0.12)
            self._chalet(x, z, yaw, w, d, rng.uniform(6.5, 9.5), walls[rng.integers(0, len(walls))],
                         rng.uniform(0.45, 0.8))
        # The church: white nave and a tall steeple.
        cx, cz = -60.0, -470.0
        y = float(terrain_height(cx, cz)) - 0.6
        s = self.static
        s.add("box", chain(tr((cx, y, cz)), sc((11, 9, 24))), rgb("#e6e0d4"), 0.0, BUILDING, seed=7.0, p1=0.35)
        s.add("prism", chain(tr((cx, y + 9, cz)), sc((12.5, 5.5, 25))), rgb("#3a2a22"))
        s.add("box", chain(tr((cx, y, cz + 13)), sc((6, 24, 6))), rgb("#ece6da"))
        s.add("cone", chain(tr((cx, y + 24, cz + 13)), ry(math.pi / 8), sc((7.5, 14, 7.5))), rgb("#2e3a44"))
        s.add("ico0", chain(tr((cx, y + 20, cz + 16.1)), sc(1.6)), rgb("#ffe4b0"), 5.0, GLOW, p2=1.0)  # the clock
        # Street lamps along the main street.
        for x in np.arange(x0 + 10, x1, 28.0):
            for side in (-1, 1):
                z = -430.0 + side * 8.0
                gy = float(terrain_height(x, z))
                s.add("cyl5", chain(tr((x, gy, z)), sc((0.25, 5.5, 0.25))), rgb("#2a2a30"))
                s.add("ico", chain(tr((x, gy + 5.8, z)), sc(1.3)), rgb("#ffd59a"), 7.0, GLOW, p2=1.0)
                self.lamps.append((x, gy + 5.8, z))

    def _trees(self, rng) -> None:
        n = 22000
        x = rng.uniform(-1500, 1500, n)
        z = rng.uniform(-2600, -40, n)
        cam_x, cam_z = self.camera_pos[0], self.camera_pos[2]
        in_view = np.abs(x - cam_x) < 0.36 * (cam_z - z) + 60
        h = terrain_height(x, z)
        slope = np.hypot(terrain_height(x + 3, z) - terrain_height(x - 3, z),
                         terrain_height(x, z + 3) - terrain_height(x, z - 3)) / 6
        x0, x1, z0, z1 = VILLAGE
        in_village = (x > x0 - 15) & (x < x1 + 15) & (z > z0 - 15) & (z < z1 + 25)
        near = (z > -170) & (np.abs(x) < 70)  # keep the view down the slope clear
        lx, lz = LIFT[0], LIFT[1]
        lift = np.abs((x - lx[0]) * (lz[1] - lx[1]) - (z - lx[1]) * (lz[0] - lx[0])) / math.hypot(
            lz[0] - lx[0], lz[1] - lx[1]) < 14
        open_floor = (h < 30) & (rng.random(n) < 0.9)  # meadows under snow around the village
        clearing = fbm(x / 260.0, z / 260.0, 11.0) < 0.47  # open snowfields between the stands of trees
        keep = (in_view & (h < 560) & (slope < 0.95) & ~in_village & ~near & ~lift & ~open_floor & ~clearing
                & (_piste_distance(x, z) > PISTE_HALF + 4))
        x, z, h = x[keep], z[keep], h[keep]
        far = z < -1700
        keep = ~far | (rng.random(len(x)) < 0.6)
        x, z, h = x[keep], z[keep], h[keep]
        m = len(x)
        height = rng.uniform(10, 20, m) * np.where(h > 420, 0.75, 1.0)
        yaw = rng.uniform(0, 2 * np.pi, m)
        greens = np.array([rgb(c) for c in ("#16301f", "#1b3a26", "#203f2a", "#132a1b", "#24462f")])
        color = greens[rng.integers(0, len(greens), m)] * rng.uniform(0.85, 1.1, (m, 1))
        base = chain(tr(np.stack([x, h - 0.5, z], 1)), ry(yaw))
        s = self.static
        s.add("cyl5", chain(base, sc(np.stack([0.5 * np.ones(m), height * 0.25, 0.5 * np.ones(m)], 1))), rgb("#3a2e26"))
        s.add("cone6", chain(base, tr(np.stack([np.zeros(m), height * 0.18, np.zeros(m)], 1)),
                             sc(np.stack([height * 0.4, height * 0.62, height * 0.4], 1))), color)
        s.add("cone6", chain(base, tr(np.stack([np.zeros(m), height * 0.5, np.zeros(m)], 1)),
                             sc(np.stack([height * 0.26, height * 0.5, height * 0.26], 1))), color * 1.08)
        self.tree_count = m

    def _pistes(self) -> None:
        """Floodlights down both edges of each piste (night skiing); they follow the power."""
        pts = []
        for line in PISTES:
            for (x0, z0), (x1, z1) in zip(line[:-1], line[1:]):
                length = math.hypot(x1 - x0, z1 - z0)
                nx, nz = -(z1 - z0) / length, (x1 - x0) / length
                for k in np.arange(0.0, 1.0, 55.0 / length):
                    for side in (-1, 1):
                        px = x0 + (x1 - x0) * k + side * nx * (PISTE_HALF + 2)
                        pz = z0 + (z1 - z0) * k + side * nz * (PISTE_HALF + 2)
                        pts.append((px, float(terrain_height(px, pz)) + 9.0, pz))
        p = np.array(pts)
        self.static.add("ico0", chain(tr(p), sc(2.2)), rgb("#fff0d0"), 6.0, GLOW, p2=1.0)

    def _lift(self) -> None:
        (bx, bz), (tx, tz) = LIFT
        a = np.array([bx, float(terrain_height(bx, bz)) + 10.0, bz])
        b = np.array([tx, float(terrain_height(tx, tz)) + 10.0, tz])
        s = self.static
        for p, size in ((a, 12.0), (b, 10.0)):  # the stations
            s.add("box", chain(tr(p - (0, 11, 0)), sc((size, 11, size * 1.4))), rgb("#7a7f88"), 0.0, BUILDING, seed=3.0, p1=0.9)
            s.add("box", chain(tr(p), sc((size + 2, 1.2, size * 1.4 + 2))), rgb("#3a3c42"))
        span = b - a
        self.lift = (a, b)
        for k in np.linspace(0.12, 0.88, 6):
            p = a + span * k
            ground = float(terrain_height(p[0], p[2]))
            s.add("box", chain(tr((p[0], ground, p[2])), sc((1.4, p[1] - ground + 1.5, 1.4))), rgb("#5a5e66"))
        for dx in (-3.0, 3.0):
            s.add("box", beam(a + (dx, 0, 0), b + (dx, 0, 0), 0.35), rgb("#2a2a2e"))

    def _railing(self) -> None:
        cx, cy, cz = self.camera_pos
        wood, z = rgb("#4a3426"), cz - 3.2
        s = self.static
        for y in (cy - 1.25, cy - 1.9):
            s.add("box", chain(tr((cx, y - 0.07, z)), sc((8.0, 0.14, 0.14))), wood)
        for x in (cx - 0.97, cx + 1.6):
            s.add("box", chain(tr((x, cy - 2.1, z)), sc((0.16, 1.3, 0.16))), wood)
        s.add("box", chain(tr((cx, cy - 2.25, cz - 1.5)), sc((9.0, 0.3, 4.4))), rgb("#3e3028"))
        self.lantern = (cx + 1.6, cy - 0.55, z)
        s.add("ico", chain(tr(self.lantern), sc(0.22)), rgb("#ffcf8a"), 6.0, GLOW, p2=1.0)

    # Dynamic ----------------------------------------------------------------

    def dynamic(self, t: float, fx: Effects) -> Instances:
        d = Instances()
        a, b = self.lift
        span = b - a
        length = float(np.linalg.norm(span))
        # The gondola runs while there's power; when it fails, the cabins stop where they are.
        stop = self.timeline.t_flood + 1.0
        run = t if t < stop else stop + 2.0 * (1 - math.exp(-(t - stop) / 2.0))
        travel = run * 5.0  # m/s
        k = np.mod(np.arange(0.0, 1.0, 45.0 / length) + travel / length, 1.0)
        for dx, kk in ((-3.0, k), (3.0, 1.0 - k)):  # up on one cable, down on the other
            p = a + np.outer(kk, span) + (dx, -2.2, 0)
            sway = 0.15 * fx.shake * np.sin(t * 7.0 + kk * 40.0)
            d.add("box", chain(tr(p), ry(np.full(len(kk), math.atan2(span[0], span[2]))), sc((2.4, 2.4, 2.8))),
                  rgb("#c0392b"))
            d.add("box", chain(tr(p + np.stack([sway, np.full(len(kk), 0.9), np.zeros(len(kk))], 1)),
                               sc((2.5, 0.9, 2.9))), WINDOW_WARM, 3.5, GLOW, p2=1.0)
        return d

    def lights(self, t: float, fx: Effects, cam: np.ndarray, power: float) -> list:
        k = 1.6 * power * (0.92 + 0.08 * math.sin(t * 9.0) * math.sin(t * 5.3))
        return [(*self.lantern, 7.0, 1.0 * k, 0.62 * k, 0.3 * k)] if power > 0.05 else []

    def camera(self, t: float, fx: Effects) -> tuple[np.ndarray, np.ndarray, float]:
        look = self.look_at.copy()
        if fx.shake > 0:  # only the view trembles; the balcony never moves
            k = fx.shake
            look = look + k * 8.0 * np.array([math.sin(t * 19.7 + 2), math.sin(t * 27.1) + math.sin(t * 41.3), 0.0])
        return self.camera_pos.copy(), look, 60.0
