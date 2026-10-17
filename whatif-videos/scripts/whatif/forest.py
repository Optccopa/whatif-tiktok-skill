"""The forest scene: a fire lookout tower above a pine valley with a lake, peaks beyond.

The camera looks east (-z) over the treetops at golden hour, with the sun low behind us.
The valley floor holds a lake with a few cabins and a dock; forested hills rise beyond it,
then a snowy range. Terrain is a heightfield mesh; trees are thousands of instances.
"""

from __future__ import annotations

import math

import numpy as np

from whatif.meshes import BUILDING, Instances, chain, rgb, ry, rz, sc, tr
from whatif.scenario import Effects, Timeline, ramp

TERRAIN = 8
LAKE = (0.0, -760.0, 560.0, 280.0)  # center x, center z, half-width, half-depth
SNOW_LINE = 560.0
BODY_AZ_EL = (3.0, 8.5)
MOON_AZ_EL = (-11.0, 7.0)


def _hash(ix, iz, seed):
    return np.mod(np.sin(ix * 127.1 + iz * 311.7 + seed * 74.7) * 43758.5453, 1.0)


def value_noise(x, z, seed=0.0):
    ix, iz = np.floor(x), np.floor(z)
    fx, fz = x - ix, z - iz
    ux, uz = fx * fx * (3 - 2 * fx), fz * fz * (3 - 2 * fz)
    a, b = _hash(ix, iz, seed), _hash(ix + 1, iz, seed)
    c, d = _hash(ix, iz + 1, seed), _hash(ix + 1, iz + 1, seed)
    return (a + (b - a) * ux) * (1 - uz) + (c + (d - c) * ux) * uz


def fbm(x, z, seed=0.0, octaves=4):
    total, amp, norm = 0.0, 0.5, 0.0
    for i in range(octaves):
        total = total + amp * value_noise(x, z, seed + i * 13.1)
        norm += amp
        x, z, amp = x * 2.03 + 5.2, z * 2.03 + 1.7, amp * 0.5
    return total / norm


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def lake_shape(x, z):
    cx, cz, hw, hd = LAKE
    return ((x - cx) / hw) ** 2 + ((z - cz) / hd) ** 2


def terrain_height(x, z):
    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    h = 6.0 + np.minimum(0.00008 * x * x, 240.0)  # valley walls
    h = h + 150.0 * _smooth((z + 200.0) / 550.0)  # our ridge
    h = h + 230.0 * _smooth((-z - 1150.0) / 900.0)  # hills beyond the lake
    ridges = 1.0 - np.abs(2.0 * fbm(x / 900.0, z / 900.0, 5.0) - 1.0)  # sharp crests
    # The range sits low enough (about 6 degrees from the tower) for the sky above it to carry the story.
    h = h + _smooth((-z - 3800.0) / 2400.0) * (190.0 + 200.0 * fbm(x / 1300.0, z / 1300.0, 3.0) + 300.0 * ridges ** 2)
    h = h + 26.0 * (fbm(x / 380.0, z / 380.0, 1.0) - 0.5)
    h = h + 90.0 * (fbm(x / 1500.0, z / 1500.0, 7.0) - 0.5) * _smooth((-z - 500.0) / 1000.0)
    bowl = _smooth((1.15 - lake_shape(x, z)) * 2.5)  # the lake bed
    return h * (1 - bowl) + (-6.0) * bowl


def terrain_mesh() -> np.ndarray:
    """Flat-shaded triangles: fine near the lookout, coarser toward the mountains."""
    zs = np.concatenate([np.arange(420.0, -1600.0, -12.0), np.arange(-1600.0, -9000.0, -55.0)])
    xs = np.concatenate([np.arange(-3600.0, -1500.0, 45.0), np.arange(-1500.0, 1500.0, 15.0),
                         np.arange(1500.0, 3601.0, 45.0)])
    X, Z = np.meshgrid(xs, zs)
    Y = terrain_height(X, Z)
    P = np.stack([X, Y, Z], -1)
    a, b, c, d = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]
    tris = np.stack([np.stack([a, b, d], -2), np.stack([a, d, c], -2)], 2)
    return tris.reshape(-1, 3).astype(np.float32)


class Forest:
    body_az_el = BODY_AZ_EL
    moon_az_el = MOON_AZ_EL

    def __init__(self, timeline: Timeline, seed: int = 0):
        self.timeline = timeline
        rng = np.random.default_rng(seed + 11)
        ground = terrain_height(0.0, 320.0)
        self.camera_pos = np.array([0.0, float(ground) + 26.0, 320.0])
        self.look_at = np.array([0.0, self.camera_pos[1] - 0.07 * 1000.0, 320.0 - 1000.0])
        self.meshes = {"terrain": terrain_mesh()}
        self.static = Instances()
        self.static.add("terrain", np.eye(4), rgb("#3f5a2c"), 0.0, TERRAIN)
        self._trees(rng)
        self._cabins(rng)
        self._railing()
        self._birds(rng)
        self._aftermath(rng)

    # Static -----------------------------------------------------------------

    def _trees(self, rng) -> None:
        n = 26000
        x = rng.uniform(-1700, 1700, n)
        z = rng.uniform(-2700, 285, n)
        cam_x, cam_z = self.camera_pos[0], self.camera_pos[2]
        in_view = np.abs(x - cam_x) < 0.36 * (cam_z - z) + 80
        h = terrain_height(x, z)
        slope = np.hypot(terrain_height(x + 3, z) - terrain_height(x - 3, z),
                         terrain_height(x, z + 3) - terrain_height(x, z - 3)) / 6
        lake = lake_shape(x, z)
        meadow = (lake > 1.0) & (lake < 1.9) & (rng.random(n) < 0.85)
        keep = in_view & (h > 9) & (h < 520) & (slope < 0.85) & (lake > 1.25) & ~meadow
        x, z, h = x[keep], z[keep], h[keep]
        # Thin out the far forest a little; it reads as texture there anyway.
        far = (cam_z - z) > 1600
        keep = ~far | (rng.random(len(x)) < 0.6)
        x, z, h = x[keep], z[keep], h[keep]
        m = len(x)
        height = rng.uniform(11, 22, m) * np.where(h > 380, 0.75, 1.0)
        yaw = rng.uniform(0, 2 * np.pi, m)
        greens = np.array([rgb(c) for c in ("#1d3a24", "#22432a", "#2a4d2c", "#1a3320", "#2f5232")])
        color = greens[rng.integers(0, len(greens), m)] * rng.uniform(0.85, 1.1, (m, 1))
        base = chain(tr(np.stack([x, h - 0.5, z], 1)), ry(yaw))
        s = self.static
        s.add("cyl5", chain(base, sc(np.stack([0.5 * np.ones(m), height * 0.25, 0.5 * np.ones(m)], 1))),
              rgb("#4a3a2c"))
        s.add("cone6", chain(base, tr(np.stack([np.zeros(m), height * 0.18, np.zeros(m)], 1)),
                             sc(np.stack([height * 0.42, height * 0.62, height * 0.42], 1))), color)
        s.add("cone6", chain(base, tr(np.stack([np.zeros(m), height * 0.5, np.zeros(m)], 1)),
                             sc(np.stack([height * 0.28, height * 0.5, height * 0.28], 1))), color * 1.08)
        self.tree_count = m

    def _cabins(self, rng) -> None:
        cx, cz, hw, hd = LAKE
        s = self.static
        for a in np.linspace(0.35, np.pi - 0.35, 7) + rng.uniform(-0.12, 0.12, 7):
            r = rng.uniform(1.3, 1.6)
            x, z = cx + hw * r * math.cos(a), cz + hd * r * math.sin(a)
            y = float(terrain_height(x, z)) - 0.5
            w, d, hh = rng.uniform(7, 10), rng.uniform(6, 8), rng.uniform(3.5, 4.5)
            base = chain(tr((x, y, z)), ry(math.atan2(cx - x, cz - z) + rng.uniform(-0.3, 0.3)))
            wall = rgb(rng.choice(["#7a4a30", "#8c3a2c", "#6b4a35", "#9b3b2e"]))
            s.add("box", chain(base, sc((w, hh, d))), wall, 0.0, BUILDING, seed=rng.uniform(0, 100), p1=0.6)
            s.add("prism", chain(base, tr((0, hh, 0)), ry(math.pi / 2), sc((d + 0.8, 2.6, w + 0.8))), rgb("#2e2622"))
            s.add("box", chain(base, tr((w * 0.25, hh + 1.0, 0)), sc((0.7, 2.4, 0.7))), rgb("#4d4038"))
        # A dock and a rowing boat on the near shore.
        x, z = cx + 60, cz + hd * 0.97
        s.add("box", chain(tr((x, 0.4, z - 12)), sc((2.4, 0.3, 26))), rgb("#7a6450"))
        s.add("hull", chain(tr((x + 4, -0.3, z - 20)), ry(0.4), sc((1.6, 0.9, 4.5))), rgb("#c0392b"))

    def _railing(self) -> None:
        cx, cy, cz = self.camera_pos
        wood, z = rgb("#3b3029"), cz - 3.2
        s = self.static
        for y in (cy - 0.9, cy - 1.55):
            s.add("box", chain(tr((cx, y - 0.07, z)), sc((8.0, 0.14, 0.14))), wood)
        s.add("box", chain(tr((cx - 0.97, cy - 2.1, z)), sc((0.16, 1.3, 0.16))), wood)
        s.add("box", chain(tr((cx, cy - 2.25, cz - 1.5)), sc((9.0, 0.3, 4.4))), rgb("#4a3a2e"))

    def _birds(self, rng) -> None:
        n = 26
        self.birds = dict(
            start=rng.uniform(15.0, 22.0, n), speed=rng.uniform(13, 18, n), x0=rng.uniform(260, 420, n),
            z=self.camera_pos[2] - rng.uniform(250, 520, n), y=self.camera_pos[1] + rng.uniform(20, 120, n),
            phase=rng.uniform(0, 6, n), drift=rng.uniform(-3, 3, n),
        )

    def _aftermath(self, rng) -> None:
        """For strikes: the dust wall of the air blast, and rock falling back from space."""
        n = 420
        self.dust = dict(x=rng.uniform(-3800, 3800, n), up=rng.uniform(0, 1, n) ** 1.5, dz=rng.uniform(0, 700, n),
                         r=rng.uniform(70, 220, n), shade=rng.uniform(0.75, 1.15, n), yaw=rng.uniform(0, 6, n))
        n = 180
        az = rng.uniform(-0.8, 0.8, n)
        el = rng.uniform(0.3, 1.2, n)
        dist = rng.uniform(2500, 7000, n)
        d = np.stack([np.sin(az) * np.cos(el), np.sin(el), -np.cos(az) * np.cos(el)], 1)
        self.ejecta = dict(p0=self.camera_pos + d * dist[:, None], offset=rng.uniform(0, 1, n),
                           v=np.stack([rng.uniform(-300, 300, n), rng.uniform(-1300, -700, n), rng.uniform(-100, 400, n)], 1),
                           width=rng.uniform(6, 16, n), life=rng.uniform(1.2, 2.2, n))

    # Dynamic ----------------------------------------------------------------

    def dynamic(self, t: float, fx: Effects) -> Instances:
        d = Instances()
        cx, cz, hw, hd = LAKE
        slosh = min(math.log10(1 + fx.tide) / 3.0, 1.0) * 4.5
        level = slosh * math.sin(t * 0.9) + (fx.water if self.timeline.strike else 0.0)
        d.add("quad", chain(tr((cx, level, cz)), sc((hw * 2.6, 1, hd * 2.8))), rgb("#24384a"), 0.0, 1)
        b = self.birds
        age = t - b["start"]
        live = (age > 0) & (age < 40)
        if live.any():
            k = int(live.sum())
            x = b["x0"][live] - b["speed"][live] * age[live]
            y = b["y"][live] + b["drift"][live] * age[live] * 0.3
            pos = np.stack([x, y, b["z"][live]], 1)
            flap = 0.6 * np.sin(t * 9.0 + b["phase"][live])
            yaw = np.full(k, -np.pi / 2)
            for side in (-1, 1):
                wing = chain(tr(pos), ry(yaw), rz(side * flap), tr((side * 0.9, 0, 0)), sc((1.8, 0.08, 0.5)),
                             tr((0, -0.5, 0)))
                d.add("box", wing, rgb("#16130f"))
        if self.timeline.strike:
            self._draw_aftermath(d, t, fx)
        if fx.air_snow > 0:
            self._draw_air_snow(d, t, fx)
        return d

    def _draw_air_snow(self, d: Instances, t: float, fx: Effects) -> None:
        """The air itself freezing out: flakes drifting down past the tower."""
        n = int(500 * fx.air_snow)
        if n == 0:
            return
        i = np.arange(n)
        rng = np.random.default_rng(5)
        x0, z0 = rng.uniform(-40, 40, 500)[:n], rng.uniform(-160, -6, 500)[:n]
        speed, off = rng.uniform(1.2, 2.4, 500)[:n], rng.uniform(0, 60, 500)[:n]
        cam = self.camera_pos
        y = cam[1] + 30 - np.mod(t * speed + off, 60.0)
        x = cam[0] + x0 + np.sin(t * 0.7 + i) * 1.5
        z = cam[2] + z0
        size = 0.14 + 0.0025 * np.abs(z0)
        d.add("ico0", chain(tr(np.stack([x, y, z], 1)), sc(size)), rgb("#dfe6f2"), 1.8, 2)

    def _draw_aftermath(self, d: Instances, t: float, fx: Effects) -> None:
        tl = self.timeline
        start, end = fx.meteors_from, tl.t_blast + 1.0
        if start < t < end:  # rock blasted into space, falling back red-hot
            e = self.ejecta
            n = len(e["offset"])
            span = end - start
            birth = start + e["offset"] * span
            age = t - birth
            alive = (age > 0) & (age < e["life"])
            if alive.any():
                p = e["p0"][alive] + e["v"][alive] * age[alive, None]
                v = e["v"][alive]
                fade = 1 - age[alive] / e["life"][alive]
                tail = p - v * 0.12
                from whatif.meshes import GLOW, beam
                d.add("box", beam(tail, p, e["width"][alive]), rgb("#ff7a30"), 9.0 * fade, GLOW)
                d.add("ico0", chain(tr(p), sc(e["width"][alive] * 2.2)), rgb("#ffd08a"), 14.0 * fade, GLOW)
        arrive = tl.t_blast
        if False and t > arrive - 4.2:  # (the dust wall is drawn by the shaders; see dust_front)
            k = (t - (arrive - 4.2)) / 4.7
            front = -7000.0 + (self.camera_pos[2] + 7300.0) * min(k, 1.3) ** 1.15
            w = self.dust
            z = front - w["dz"]
            near = (z > -7600) & (z < self.camera_pos[2] - 450)  # up close, the dust is just fog
            if near.any():
                x, zz = w["x"][near], z[near]
                ground = terrain_height(x, zz)
                age = np.clip(k - w["dz"][near] / 7300.0, 0, None)
                r = w["r"][near] * (1.0 + 0.6 * age)
                y = ground + w["up"][near] * 520.0 + r * 0.3
                color = np.outer(w["shade"][near], rgb("#b0703e"))
                d.add("ico", chain(tr(np.stack([x, y, zz], 1)), ry(w["yaw"][near] + t * 0.2),
                                   sc(np.stack([r * 1.3, r, r * 1.2], 1))), color, 0.0)

    def dust_front(self, t: float) -> float | None:
        """z of the air blast's dust wall rolling across the valley toward us, or None."""
        if not self.timeline.strike:
            return None
        k = (t - (self.timeline.t_blast - 4.2)) / 4.7
        if k <= 0:
            return None
        return -7000.0 + (self.camera_pos[2] + 7300.0) * min(k, 1.3) ** 1.15

    def lights(self, t: float, fx: Effects, cam: np.ndarray, power: float) -> list:
        return []

    def camera(self, t: float, fx: Effects) -> tuple[np.ndarray, np.ndarray, float]:
        look = self.look_at.copy()
        if fx.shake > 0:  # only the view trembles; we never leave the tower
            k = fx.shake
            look = look + k * 9.0 * np.array([math.sin(t * 19.7 + 2), math.sin(t * 27.1) + math.sin(t * 41.3), 0.0])
        return self.camera_pos.copy(), look, 60.0
