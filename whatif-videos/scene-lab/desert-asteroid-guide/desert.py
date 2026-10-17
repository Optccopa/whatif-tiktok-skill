"""The desert scene: a canyon-rim overlook above a highway, buttes and mesas beyond.

The camera stands at a stone overlook wall on the rim (8 m above the plateau, y=0) and looks
north (-z) down a red-rock canyon. The canyon heads right below the wall; its floor, 150 m down,
carries a river and a two-lane highway with traffic, winding away for 2 km until the canyon opens
into a valley. There three buttes stand ~300 m tall around the road, and a broken line of mesas
closes the horizon 5-11 km out. Golden hour: the sun 7 degrees up, behind us to the right.
"""

from __future__ import annotations

import math

import numpy as np

from whatif.forest import fbm, _smooth
from whatif.meshes import GLOW, GROUND, WATER, Instances, chain, rgb, ry, rz, sc, tr
from whatif.scenario import Effects, Timeline

ROCK = 9  # the shader's desert material: strata on cliffs, sand on flats
FLOOR = -150.0
CAM = np.array([0.0, 8.0, -24.0])  # out on a platform past the rim, so the canyon (not bare ground) fills the bottom
W0, W1 = 55.0, 50.0  # canyon floor half-width, bench width (they grow downstream)
HEAD_Z = -3.0 - (W0 + 12 + W1 + 16)  # the canyon heads right under the overlook wall
CENTER = [(0, HEAD_Z), (-10, -330), (30, -580), (95, -840), (70, -1130), (-20, -1430), (-60, -1760),
          (-40, -2100), (-10, -2450), (0, -2800)]
BUTTES = [  # x, z, radius x, radius z, yaw, top height
    (-470, -3150, 170, 85, 0.3, 175), (-270, -2950, 34, 34, 0.0, 125), (430, -3350, 150, 95, -0.25, 160),
    (1050, -4100, 320, 170, 0.1, 110), (-1300, -3800, 260, 140, -0.4, 95),
]


def catmull(points, step=15.0) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    p = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(p) - 2):
        a, b, c, d = p[i - 1], p[i], p[i + 1], p[i + 2]
        n = max(int(np.linalg.norm(c - b) / step), 1)
        s = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(0.5 * (2 * b + (-a + c) * s + (2 * a - 5 * b + 4 * c - d) * s ** 2 + (-a + 3 * b - 3 * c + d) * s ** 3))
    out.append(p[-2][None])
    return np.vstack(out)


CENTERLINE = catmull(CENTER, 30.0)


def offset_line(line: np.ndarray, off: float) -> np.ndarray:
    tang = np.gradient(line, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    right = np.stack([-tang[:, 1], tang[:, 0]], 1)  # heading -z, right is +x
    return line + right * off


def dist_to_line(x, z, line):
    a, b = line[:-1], line[1:]
    best = np.full(np.shape(x), 1e9)
    for (ax, az), (bx, bz) in zip(a, b):
        dx, dz = bx - ax, bz - az
        k = np.clip(((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz), 0, 1)
        best = np.minimum(best, np.hypot(x - ax - k * dx, z - az - k * dz))
    return best


def terrain_height(x, z):
    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    # The plateau dips gently away from us, with rolling slickrock.
    h = -0.035 * np.clip(-z - 300.0, 0, 3000) + 6.0 * (fbm(x / 140.0, z / 140.0, 2.0) - 0.5)
    h = h + 2.0 * np.clip(1 - np.hypot(x, z) / 40.0, 0, 1)  # the overlook knob
    # 2 km out the plateau breaks off into the valley.
    edge = -2050.0 + 300.0 * (fbm(x / 700.0, 0.3, 4.0) - 0.5)
    h = h + (FLOOR - h) * _smooth((edge - z) / 60.0)
    # The canyon: floor, a cliff, a bench, a second cliff up to the rim (stair-stepped like the Southwest).
    g = 1.0 + 0.45 * np.clip((HEAD_Z - z) / 2000.0, 0, 1)
    d = dist_to_line(x, z, CENTERLINE)
    w0, w1 = W0 * g, (W0 + 12 + W1) * g
    canyon = (FLOOR + 6.0 * _smooth((d - w0 + 20) / 20) + 64.0 * _smooth((d - w0) / 18.0)
              + 10.0 * np.clip((d - w0 - 18) / (w1 - w0 - 18), 0, 1) + 400.0 * _smooth((d - w1) / 16.0))
    h = np.minimum(h, canyon)
    # Buttes: a talus apron, then sheer walls to a flat top.
    for bx, bz, rx, rz_, yaw, top in BUTTES:
        u = ((x - bx) * math.cos(yaw) - (z - bz) * math.sin(yaw)) / rx
        v = ((x - bx) * math.sin(yaw) + (z - bz) * math.cos(yaw)) / rz_
        r = np.hypot(u, v) + 0.12 * (fbm(x / 60.0, z / 60.0, 9.0) - 0.5)
        tall = top - FLOOR
        butte = FLOOR + 0.3 * tall * (1 - _smooth((r - 1.0) / 1.3)) + 0.7 * tall * (1 - _smooth((r - 0.93) / 0.12))
        h = np.maximum(h, butte)
    # Far mesas: flat tops with cliff edges, then a long ridge closing the horizon.
    far = _smooth((-z - 4800.0) / 1200.0)
    mesa = _smooth((fbm(x / 1800.0, z / 1800.0, 6.0) - 0.5) / 0.03) * 420.0 * far
    ridge = (260.0 + 160.0 * fbm(x / 1500.0, 0.0, 8.0)) * _smooth((-z - 9500.0) / 1500.0)
    return np.maximum(h, FLOOR + np.maximum(mesa, ridge))


def terrain_mesh() -> np.ndarray:
    # 2.5 m where the cliffs are near (a 10 m grid turned them into sawtooth), coarser with distance.
    zs = np.concatenate([np.arange(40.0, -1400.0, -2.5), np.arange(-1400.0, -2200.0, -5.0),
                         np.arange(-2200.0, -4600.0, -20.0), np.arange(-4600.0, -12500.0, -70.0)])
    xs = np.concatenate([np.arange(-6000.0, -1200.0, 60.0), np.arange(-1200.0, -450.0, 10.0),
                         np.arange(-450.0, 450.0, 2.5), np.arange(450.0, 1200.0, 10.0),
                         np.arange(1200.0, 6001.0, 60.0)])
    X, Z = np.meshgrid(xs, zs)
    P = np.stack([X, terrain_height(X, Z), Z], -1)
    a, b, c, d = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]
    tris = np.stack([np.stack([a, b, d], -2), np.stack([a, d, c], -2)], 2)
    return tris.reshape(-1, 3).astype(np.float32)


def _strip(s: Instances, line: np.ndarray, width: float, y_of, color, emissive=0.0, material=0) -> None:
    """Flat quads along a polyline, lifted a little more with distance so they never z-fight the ground."""
    a, b = line[:-1], line[1:]
    mid = (a + b) / 2
    dist = np.hypot(mid[:, 0] - CAM[0], mid[:, 1] - CAM[2])
    y = y_of + 0.25 + (dist / 1000.0) ** 2 * 0.9
    yaw = np.arctan2(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])
    length = np.hypot(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]) * 1.04
    m = chain(tr(np.stack([mid[:, 0], y, mid[:, 1]], 1)), ry(yaw), sc(np.stack([np.full(len(a), width), np.ones(len(a)), length], 1)))
    s.add("quad", m, color, emissive, material)


class Desert:
    body_az_el = (3.0, 8.5)
    moon_az_el = (-11.0, 7.0)
    strike_look = dict(
        zenith=("#2c5aa8", "#3a1a14", "#2a1a12"), mid=("#86aedb", "#9a4420", "#4a3020"),
        horizon=("#f2c49a", "#e8702a", "#6a4430"), cloud=("#f6cfa8", "#c05a2a", "#4a3a2c"),
        amb_sky=("#9ab0d8", "#b0603a", "#5a3c2a"), amb_ground=("#7a4a30", "#7a3a20", "#2e1e14"),
        plume=("#f0b88a", "#d07040", "#6a4430"), dust="#b0683e", sun=(0.72, 0.69), fog=0.00009, clouds=0.45,
        shadow=((0.0, -110.0, -650.0), 700.0),
    )

    def __init__(self, timeline: Timeline, seed: int = 0):
        self.timeline = timeline
        rng = np.random.default_rng(seed + 23)
        self.camera_pos = CAM.copy()
        self.look_at = CAM + np.array([0.0, -0.07 * 1000.0, -1000.0])
        self.meshes = {"terrain": terrain_mesh()}
        self.static = Instances()
        self.static.add("terrain", np.eye(4), (1.0, 1.0, 1.0), 0.0, ROCK)
        self._road_and_river()
        self._scrub(rng)
        self._overlook()
        self._traffic(rng)
        self._birds(rng)
        self._ejecta(rng)

    # Static -----------------------------------------------------------------

    def _road_and_river(self) -> None:
        canyon = CENTERLINE[CENTERLINE[:, 1] > -2300]
        valley = catmull([(canyon[-1, 0] + 28, canyon[-1, 1]), (-20, -2700), (10, -3200), (-60, -5000), (-200, -12000)], 40.0)
        self.road = np.vstack([offset_line(canyon, 28.0), valley[1:]])
        s = self.static
        _strip(s, self.road, 10.0, FLOOR, rgb("#2f2b2c"))
        near = self.road[np.hypot(self.road[:, 0], self.road[:, 1]) < 1400]
        for off in (-4.6, 4.6):  # white edge lines
            _strip(s, offset_line(near, off), 0.25, FLOOR + 0.04, rgb("#e8e2d4"))
        # the dashed yellow center line
        dense = catmull(near[::2], 3.0)
        dashes = dense[::4]
        for a, b in zip(dashes, dashes[1:]):
            if np.hypot(*(b - a)) < 8:
                _strip(s, np.array([a, a + (b - a) * 0.3]), 0.25, FLOOR + 0.06, rgb("#f0c040"))
        river = np.vstack([offset_line(canyon, -22.0),
                           catmull([(canyon[-1, 0] - 22, canyon[-1, 1]), (-300, -2420), (-900, -2620), (-2400, -2750)], 40.0)[1:]])
        _strip(s, river, 20.0, FLOOR - 0.1, rgb("#3c4a3e"), 0.0, WATER)
        # cumulative length along the road, for the traffic
        self.road_s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(self.road, axis=0).T))])

    def _scrub(self, rng) -> None:
        n = 42000
        z = -(30.0 + 4600.0 * rng.random(n) ** 1.5)
        x = rng.uniform(-1, 1, n) * (0.4 * -z + 40)
        h = terrain_height(x, z)
        slope = np.hypot(terrain_height(x + 2, z) - terrain_height(x - 2, z),
                         terrain_height(x, z + 2) - terrain_height(x, z - 2)) / 4
        keep = slope < 0.3
        x, z, h = x[keep], z[keep], h[keep]
        on_floor = h < FLOOR + 3
        road_d = dist_to_line(x[on_floor], z[on_floor], self.road[::2])
        ok = np.ones(len(x), bool)
        ok[np.where(on_floor)[0]] = road_d > 10
        x, z, h = x[ok], z[ok], h[ok]
        m = len(x)
        juniper = rng.random(m) < 0.45
        size = np.where(juniper, rng.uniform(2.2, 4.5, m), rng.uniform(0.9, 1.6, m))
        greens = np.array([rgb(c) for c in ("#4f5a32", "#5e6a3c", "#3f4a2c", "#6b7048")])
        sage = np.array([rgb(c) for c in ("#8d8f68", "#9a9a74", "#7d8460")])
        color = np.where(juniper[:, None], greens[rng.integers(0, 4, m)], sage[rng.integers(0, 3, m)])
        color = color * rng.uniform(0.85, 1.1, (m, 1))
        self.static.add("ico", chain(tr(np.stack([x, h + size * 0.3, z], 1)), ry(rng.uniform(0, 6, m)),
                                     sc(np.stack([size, size * 0.8, size], 1))), color)
        # boulders fallen from the cliffs
        k = 2500
        bx = rng.uniform(-1, 1, k) * 220
        bz = rng.uniform(-2100, -200, k)
        bh = terrain_height(bx, bz)
        sel = (bh < FLOOR + 30) & (bh > FLOOR + 1)
        r = rng.uniform(1.5, 5.0, sel.sum())
        self.static.add("ico0", chain(tr(np.stack([bx[sel], bh[sel] + r * 0.2, bz[sel]], 1)), ry(rng.uniform(0, 6, sel.sum())),
                                      sc(np.stack([r * 1.3, r, r * 1.1], 1))), rgb("#8a4426"))
        self.scrub_count = m

    def _overlook(self) -> None:
        """A thin steel pipe railing 2.6 m ahead: just a line across the bottom of the frame, no wall."""
        cx, cy, cz = self.camera_pos
        s = self.static
        z, top = cz - 2.6, cy - 1.5
        steel = rgb("#2a2c30")
        for y in (top + 0.35, top - 0.15):  # two rails
            s.add("cyl", chain(tr((cx - 9, y, z)), rz(-math.pi / 2), sc((0.06, 18.0, 0.06))), steel)
        for px in np.arange(-8.0, 8.1, 2.0):
            s.add("cyl5", chain(tr((cx + px, top - 2.0, z)), sc((0.06, 2.4, 0.06))), steel)
    def _traffic(self, rng) -> None:
        n = 20
        self.cars = dict(
            s0=rng.uniform(0, 3200, n), v=rng.uniform(24, 31, n), lane=rng.choice([-1.0, 1.0], n),
            truck=rng.random(n) < 0.25, color=np.array([rgb(c) for c in rng.choice(
                ["#f2efe8", "#c0392b", "#b8bcc4", "#2c4a7a", "#26262c", "#e8b04a", "#6a7a80"], n)]),
            stop=self.timeline.tc + 1.0 + rng.uniform(0.0, 2.5, n) if self.timeline.strike else np.full(n, 1e9),
        )
        self.road_len = 3200.0  # ponytail: cars wrap 3.2 km out, where they're specks in the haze

    def _birds(self, rng) -> None:
        n = 3  # ravens circling over the canyon
        self.birds = dict(c=np.stack([rng.uniform(-30, 40, n), CAM[1] + rng.uniform(14, 34, n), -rng.uniform(60, 130, n)], 1),
                          r=rng.uniform(12, 22, n), w=rng.uniform(0.18, 0.3, n) * rng.choice([-1, 1], n),
                          ph=rng.uniform(0, 6, n))

    def _ejecta(self, rng) -> None:
        n = 180
        az = rng.uniform(-0.8, 0.8, n)
        el = rng.uniform(0.3, 1.2, n)
        dist = rng.uniform(2500, 7000, n)
        d = np.stack([np.sin(az) * np.cos(el), np.sin(el), -np.cos(az) * np.cos(el)], 1)
        self.ejecta = dict(p0=self.camera_pos + d * dist[:, None], offset=rng.uniform(0, 1, n),
                           v=np.stack([rng.uniform(-300, 300, n), rng.uniform(-1300, -700, n), rng.uniform(-100, 400, n)], 1),
                           width=rng.uniform(6, 16, n), life=rng.uniform(1.2, 2.2, n))

    # Dynamic ----------------------------------------------------------------

    def _road_at(self, s):
        x = np.interp(s, self.road_s, self.road[:, 0])
        z = np.interp(s, self.road_s, self.road[:, 1])
        dx = np.interp(s + 3, self.road_s, self.road[:, 0]) - np.interp(s - 3, self.road_s, self.road[:, 0])
        dz = np.interp(s + 3, self.road_s, self.road[:, 1]) - np.interp(s - 3, self.road_s, self.road[:, 1])
        return x, z, np.arctan2(dx, dz)

    def _draw_traffic(self, d: Instances, t: float) -> None:
        c = self.cars
        tau = 1.4  # braking time constant once the ground starts shaking
        run = np.minimum(t, c["stop"]) + tau * (1 - np.exp(-np.maximum(t - c["stop"], 0) / tau))
        s = np.mod(c["s0"] + c["lane"] * c["v"] * run, self.road_len)
        x, z, yaw = self._road_at(s)
        yaw = np.where(c["lane"] > 0, yaw, yaw + np.pi)  # oncoming cars face us
        # drive on the right: away-lane on +right of the road's direction
        rx_, rz_ = np.cos(yaw), -np.sin(yaw)
        x, z = x + rx_ * 2.2, z + rz_ * 2.2
        dist = np.hypot(x, z)
        y = FLOOR + 0.25 + (dist / 1000.0) ** 2 * 0.9
        truck = c["truck"]
        length = np.where(truck, 17.0, 4.6)
        height = np.where(truck, 3.9, 1.0)
        width = np.where(truck, 2.6, 1.9)
        base = chain(tr(np.stack([x, y, z], 1)), ry(yaw))
        body = np.where(truck[:, None], np.array(rgb("#ecebe6")), c["color"])
        d.add("box", chain(base, sc(np.stack([width, height, length], 1))), body)
        cab = ~truck
        if cab.any():
            d.add("box", chain(base[cab], tr((0, 1.0, -0.3)), sc((1.7, 0.6, 2.4))), c["color"][cab] * 0.8)
        if truck.any():
            d.add("box", chain(base[truck], tr((0, 0, 10.2)), sc((2.5, 3.2, 3.2))), c["color"][truck])
        braking = np.where(t > c["stop"], 3.0, 1.0)
        front = np.where(truck, 11.8, 2.32)
        glow = (0.45 + dist * 0.0012)[:, None] * np.ones(3)  # so a car's lights still read 1 km away
        for side in (-0.7, 0.7):
            d.add("ico0", chain(base, tr(np.stack([np.full(len(x), side), np.full(len(x), 0.6), front], 1)), sc(glow)),
                  rgb("#fff1d0"), 9.0, GLOW)
            d.add("ico0", chain(base, tr(np.stack([np.full(len(x), side), np.full(len(x), 0.7), -length / 2 - 0.02], 1)),
                                sc(glow * 0.9)), rgb("#ff2a1a"), 3.0 * braking, GLOW)

    def dynamic(self, t: float, fx: Effects) -> Instances:
        d = Instances()
        self._draw_traffic(d, t)
        b = self.birds
        a = b["ph"] + b["w"] * t
        pos = b["c"] + np.stack([np.cos(a) * b["r"], 0.6 * np.sin(t * 0.4 + b["ph"]), np.sin(a) * b["r"]], 1)
        yaw = a + np.sign(b["w"]) * np.pi  # tangent to the circle
        for side in (-1, 1):
            wing = chain(tr(pos), ry(-yaw), rz(side * 0.15), tr((side * 0.55, 0, 0)), sc((1.1, 0.05, 0.38)))
            d.add("box", wing, rgb("#141210"))
        d.add("box", chain(tr(pos), ry(-yaw), sc((0.22, 0.18, 0.7))), rgb("#141210"))
        if self.timeline.strike:
            self._draw_ejecta(d, t, fx)
        return d

    def _draw_ejecta(self, d: Instances, t: float, fx: Effects) -> None:
        start, end = fx.meteors_from, self.timeline.t_blast + 1.0
        if not start < t < end:
            return
        from whatif.meshes import beam
        e = self.ejecta
        birth = start + e["offset"] * (end - start)
        age = t - birth
        alive = (age > 0) & (age < e["life"])
        if alive.any():
            p = e["p0"][alive] + e["v"][alive] * age[alive, None]
            fade = 1 - age[alive] / e["life"][alive]
            d.add("box", beam(p - e["v"][alive] * 0.12, p, e["width"][alive]), rgb("#ff7a30"), 9.0 * fade, GLOW)
            d.add("ico0", chain(tr(p), sc(e["width"][alive] * 2.2)), rgb("#ffd08a"), 14.0 * fade, GLOW)

    def dust_front(self, t: float) -> float | None:
        """z of the air blast's dust wall rolling up the canyon; it stops just past the wall so the railing shows."""
        if not self.timeline.strike:
            return None
        k = (t - (self.timeline.t_blast - 4.2)) / 4.7
        if k <= 0:
            return None
        return min(-7000.0 + (self.camera_pos[2] + 7300.0) * min(k, 1.3) ** 1.15, self.camera_pos[2] - 6.0)

    def lights(self, t: float, fx: Effects, cam: np.ndarray, power: float) -> list:
        return []

    def camera(self, t: float, fx: Effects) -> tuple[np.ndarray, np.ndarray, float]:
        look = self.look_at.copy()
        if fx.shake > 0:  # only the view trembles; we never leave the overlook
            k = fx.shake
            look = look + k * 9.0 * np.array([math.sin(t * 19.7 + 2), math.sin(t * 27.1) + math.sin(t * 41.3), 0.0])
        return self.camera_pos.copy(), look, 60.0
