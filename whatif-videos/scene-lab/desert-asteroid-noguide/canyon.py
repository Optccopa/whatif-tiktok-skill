"""The canyon scene: a desert overlook at the head of a red-rock canyon, a highway running down its floor.

The camera stands at a stone parapet on the rim, 200 m above the floor, looking east (-z) down the
canyon at golden hour with the sun low behind us. Layered cliffs wall the floor on both sides,
buttes stand on it, a muddy river meanders beside a straight two-lane highway that runs to the
vanishing point, and cars drive both ways on it. A far mesa wall closes the skyline.
Terrain is a heightfield mesh; the shader layers it into sandstone strata (material DESERT).
"""

from __future__ import annotations

import math

import numpy as np

from whatif.forest import Forest, fbm
from whatif.meshes import GLOW, Instances, WATER, chain, rgb, rz, ry, sc, tr
from whatif.scenario import Effects, Timeline

DESERT = 9
RIM_Y = 200.0  # the plateau; the floor sits near 0
CAM_Z = 300.0
WALL_W = 240.0  # horizontal run of a canyon wall, floor to rim
ROAD_Z0, ROAD_Z1 = 60.0, -11500.0
ROAD_X0, ROAD_SLOPE = 70.0, 0.048  # x = ROAD_X0 + ROAD_SLOPE * (z - 250): vanishes a little left of center
FAR_Z = -11800.0  # the far mesa wall
BODY_AZ_EL = (3.0, 8.5)
MOON_AZ_EL = (-11.0, 7.0)


def road_x(z):
    return ROAD_X0 + ROAD_SLOPE * (np.asarray(z, dtype=float) - 250.0)


def river_x(z):
    z = np.asarray(z, dtype=float)
    return road_x(z) - 130.0 + 50.0 * np.sin(z / 900.0 + 0.4) + 25.0 * np.sin(z / 350.0)


def canyon_center(z):
    z = np.asarray(z, dtype=float)
    return 90.0 * np.sin(z / 1900.0 + 0.6) + 60.0 * np.sin(z / 800.0)


def half_width(z):
    return 300.0 + 0.05 * np.maximum(220.0 - np.asarray(z, dtype=float), 0.0)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# A Colorado Plateau wall: talus, a cliff, a bench, then the rim cliff.
WALL_U = [0.0, 0.26, 0.36, 0.6, 0.68, 0.8, 0.92, 1.0]
WALL_H = [0.0, 28.0, 105.0, 116.0, 124.0, 150.0, 196.0, 200.0]


def floor_grade(x, z):
    """The floor's gentle, low-frequency shape (the road follows it)."""
    return 6.0 + 6.0 * fbm(np.asarray(x) / 900.0, np.asarray(z) / 900.0, 4.0, octaves=2)


def terrain_height(x, z):
    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    floor = floor_grade(x, z) + 2.5 * (fbm(x / 140.0, z / 140.0, 9.0) - 0.5)
    # The road bed is graded flat; the river cuts a channel.
    near_road = 1.0 - _smooth((np.abs(x - road_x(z)) - 18.0) / 30.0)
    floor = floor * (1 - near_road) + (floor_grade(road_x(z), z) - 0.2) * near_road
    dr = np.abs(x - river_x(z))
    floor = floor - (floor + 3.0) * (1.0 - _smooth((dr - 8.0) / 48.0))
    # Distance into the walls: the sides, the head behind the overlook, and the far end.
    side = (np.abs(x - canyon_center(z)) - half_width(z)) / WALL_W
    wobble = 0.35 * (fbm(x / 700.0, z / 700.0, 2.0) - 0.5) + 0.12 * (fbm(x / 160.0, z / 160.0, 6.0) - 0.5)
    u = side + wobble
    tall = 1.0 + 0.6 * _smooth((CAM_Z - 400.0 - z) / 800.0)  # the side rims rise above ours
    head = 1.0 + (z - (CAM_Z - 3.0)) / 22.0  # the overlook: a sheer drop just past the parapet
    u = np.maximum(u, head)
    h = floor * (1 - np.clip(u, 0.0, 1.0)) + np.maximum(np.interp(u - (head >= u) * 9, WALL_U, WALL_H) * tall,
                                                        np.interp(head, [0.0, 0.9, 1.0], [0.0, 190.0, 200.0]))
    h = h + 5.0 * fbm(x / 500.0, z / 500.0, 8.0) * np.clip(u - 1.0, 0.0, 1.0) * (z < CAM_Z - 60)
    # The far mesa wall closing the view: another step up, higher than our rim.
    far = (FAR_Z - z) / 900.0 + 0.4 * (fbm(x / 1500.0, z / 1500.0, 3.0) - 0.5)
    h = np.where(far > 0, np.maximum(h, RIM_Y * 0.15 + 1.35 * np.interp(far, WALL_U, WALL_H)), h)
    return h


CAM = np.array([0.0, float(terrain_height(0.0, CAM_Z)) + 1.7, CAM_Z])


def terrain_mesh() -> np.ndarray:
    """Flat-shaded triangles: fine in the near canyon, coarser toward the far wall."""
    zs = np.concatenate([np.arange(420.0, -2000.0, -12.0), np.arange(-2000.0, -6000.0, -40.0),
                         np.arange(-6000.0, -16500.0, -110.0)])
    xs = np.concatenate([np.arange(-6000.0, -1500.0, 90.0), np.arange(-1500.0, 1500.0, 14.0),
                         np.arange(1500.0, 6001.0, 90.0)])
    X, Z = np.meshgrid(xs, zs)
    Y = terrain_height(X, Z)
    P = np.stack([X, Y, Z], -1)
    a, b, c, d = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]
    tris = np.stack([np.stack([a, b, d], -2), np.stack([a, d, c], -2)], 2)
    return tris.reshape(-1, 3).astype(np.float32)


def road_strip(z0: float, z1: float, offset: float, width: float, step: float = 8.0, lift: float = 0.15) -> np.ndarray:
    """Triangles for a band along the road (the asphalt, or a painted line), following the floor's grade."""
    z = np.arange(z0, z1, -step)
    x = road_x(z) + offset
    y = floor_grade(road_x(z), z) + lift
    L = np.stack([x - width / 2, y, z], 1)
    R = np.stack([x + width / 2, y, z], 1)
    a, b, c, d = L[:-1], R[:-1], L[1:], R[1:]
    return np.stack([a, b, d, a, d, c], 1).reshape(-1, 3).astype(np.float32)


class Canyon:
    body_az_el = BODY_AZ_EL
    moon_az_el = MOON_AZ_EL

    # The strike aftermath and the locked camera are the forest's: same sky, same physics.
    _aftermath = Forest._aftermath
    _draw_aftermath = Forest._draw_aftermath
    dust_front = Forest.dust_front
    camera = Forest.camera

    def __init__(self, timeline: Timeline, seed: int = 0):
        self.timeline = timeline
        rng = np.random.default_rng(seed + 17)
        self.camera_pos = CAM.copy()
        self.look_at = np.array([0.0, CAM[1] - 0.075 * 1000.0, CAM[2] - 1000.0])
        self.meshes = {
            "terrain": terrain_mesh(),
            "road": road_strip(ROAD_Z0, ROAD_Z1, 0.0, 9.0),
            "lines": np.concatenate([road_strip(ROAD_Z0, -3500.0, o, 0.3, lift=0.2) for o in (-0.25, 0.25)]),
            "edges": np.concatenate([road_strip(ROAD_Z0, -3500.0, o, 0.3, lift=0.2) for o in (-4.2, 4.2)]),
        }
        s = self.static = Instances()
        s.add("terrain", np.eye(4), rgb("#b0643a"), 0.0, DESERT)
        s.add("road", np.eye(4), rgb("#3a3532"))
        s.add("lines", np.eye(4), rgb("#e0b030"), 0.15)
        s.add("edges", np.eye(4), rgb("#d8d4cc"), 0.1)
        self._buttes(rng)
        self._shrubs(rng)
        self._parapet()
        self._cars(rng)
        self._hawks(rng)
        self._aftermath(rng)

    # Static -----------------------------------------------------------------

    def _buttes(self, rng) -> None:
        """Sandstone buttes and spires standing on the floor (layered by the same shader as the walls)."""
        s = self.static
        # (distance ahead, offset from the canyon's center, radius, height)
        for dz, off, r, h in ((1500, -200, 40, 110), (2600, 190, 60, 140), (3900, -260, 90, 150),
                              (5200, 220, 35, 170), (6400, -120, 120, 120), (7600, 380, 60, 150)):
            z = CAM[2] - dz
            x = float(canyon_center(z)) + off
            y = float(terrain_height(x, z)) - 2.0
            yaw = rng.uniform(0, 2 * np.pi)
            s.add("cone6", chain(tr((x, y, z)), ry(yaw), sc((r * 3.4, h * 0.42, r * 3.0))), rgb("#b0643a"), 0.0, DESERT)
            s.add("cyl5", chain(tr((x, y, z)), ry(yaw + 0.5), sc((r * 2.0, h, r * 1.6))), rgb("#b0643a"), 0.0, DESERT)

    def _shrubs(self, rng) -> None:
        n = 30000
        x = rng.uniform(-1600, 1600, n)
        z = rng.uniform(-4500, 150, n)
        in_view = np.abs(x - CAM[0]) < 0.36 * (CAM[2] - z) + 40
        h = terrain_height(x, z)
        slope = np.hypot(terrain_height(x + 3, z) - terrain_height(x - 3, z),
                         terrain_height(x, z + 3) - terrain_height(x, z - 3)) / 6
        clear = (np.abs(x - road_x(z)) > 16) & (np.abs(x - river_x(z)) > 30)
        keep = in_view & (slope < 0.35) & clear & (rng.random(n) < np.where(np.abs(x - river_x(z)) < 70, 1.0, 0.25))
        x, z, h = x[keep], z[keep], h[keep]
        m = len(x)
        size = rng.uniform(2.5, 5.5, m)
        greens = np.array([rgb(c) for c in ("#4a5230", "#5a5a36", "#3d4a2c", "#6b6a44")])
        color = greens[rng.integers(0, len(greens), m)] * rng.uniform(0.85, 1.1, (m, 1))
        self.static.add("ico0", chain(tr(np.stack([x, h + size * 0.25, z], 1)),
                                      sc(np.stack([size, size * 0.7, size], 1))), color)
        self.shrub_count = m

    def _parapet(self) -> None:
        """The overlook's see-through pipe railing on a low stone curb, framing the bottom of the shot."""
        cx, cy, cz = self.camera_pos
        s = self.static
        z = cz - 1.8
        metal = rgb("#4a4642")
        for y in (cy - 1.0, cy - 1.45):
            s.add("box", chain(tr((cx, y - 0.04, z)), sc((14.0, 0.08, 0.08))), metal)
        for x in (-1.1, 1.1):
            s.add("box", chain(tr((cx + x, RIM_Y, z)), sc((0.09, cy - 1.0 - RIM_Y, 0.09))), metal)
        s.add("box", chain(tr((cx, RIM_Y - 0.5, z)), sc((14.0, 0.6, 0.4))), rgb("#8a6248"))
        s.add("box", chain(tr((cx, RIM_Y - 0.5, cz + 0.6)), sc((16.0, 0.5, 5.0))), rgb("#8a6a50"))

    def _cars(self, rng) -> None:
        """Two lanes of traffic: away from us on the right lane, toward us on the left."""
        n = 140
        self.road_len = -(ROAD_Z1 - ROAD_Z0) - 300.0
        kind = rng.choice([0, 0, 0, 1, 1, 2], n)  # car, pickup/SUV, semi
        palette = ["#c8c8c4", "#2b2b30", "#8a1f1f", "#f0f0ec", "#2a4a7a", "#6a6e72", "#b88a3a"]
        self.cars = dict(
            s0=rng.uniform(0, self.road_len, n), v=rng.uniform(25, 32, n), lane=np.where(np.arange(n) % 2 == 0, 1, -1),
            kind=kind, color=np.array([rgb(palette[i]) for i in rng.integers(0, len(palette), n)]),
            brake=self.timeline.tc + rng.uniform(0.4, 2.0, n) if self.timeline.strike else np.full(n, 1e9),
        )

    def _hawks(self, rng) -> None:
        n = 2
        self.hawks = dict(center=np.array([[-60.0, 150.0, 60.0], [70.0, 135.0, -40.0]]), r=rng.uniform(28, 40, n),
                          speed=rng.uniform(0.22, 0.3, n) * np.array([1, -1]), phase=rng.uniform(0, 6, n))

    # Dynamic ----------------------------------------------------------------

    def dynamic(self, t: float, fx: Effects) -> Instances:
        d = Instances()
        level = 0.3 * fx.water
        d.add("quad", chain(tr((0.0, level, -6000.0)), sc((5000.0, 1, 13000.0))), rgb("#5a4a32"), 0.0, WATER)
        self._draw_cars(d, t)
        self._draw_hawks(d, t)
        if self.timeline.strike:
            self._draw_aftermath(d, t, fx)
        return d

    def _draw_cars(self, d: Instances, t: float) -> None:
        c = self.cars
        v, ts = c["v"], c["brake"]
        tau = 1.2  # hard braking: stopped within about four seconds
        dist = v * np.minimum(t, ts) + v * tau * (1 - np.exp(-np.maximum(t - ts, 0.0) / tau))
        u = np.mod(c["s0"] + dist, self.road_len)
        away = c["lane"] > 0
        along = np.where(away, u, self.road_len - u)
        z = ROAD_Z0 - along
        x = road_x(z) + c["lane"] * 2.1
        y = floor_grade(road_x(z), z) + 0.15
        heading = math.atan2(-ROAD_SLOPE, -1.0)
        yaw = np.where(away, heading, heading + math.pi)
        base = chain(tr(np.stack([x, y, z], 1)), ry(yaw))
        braking = np.clip((t - ts) * 3.0, 0.0, 1.0)
        for k, (length, width, height, cab) in enumerate(((4.6, 1.85, 0.85, 0.6), (5.4, 2.0, 1.1, 0.75),
                                                          (2.6, 2.5, 3.0, 0.0))):
            sel = c["kind"] == k
            if not sel.any():
                continue
            b = base[sel]
            d.add("box", chain(b, tr((0, 0.3, 0)), sc((width, height, length))), c["color"][sel])
            if cab:
                d.add("box", chain(b, tr((0, 0.3 + height, -0.3)), sc((width * 0.86, cab, length * 0.5))),
                      c["color"][sel] * 0.6)
            if k == 2:  # a semi: cab in front, a long white trailer behind
                d.add("box", chain(b, tr((0, 0.6, -9.6)), sc((2.6, 3.6, 15.5))), rgb("#e4e2dc"))
            front, back = length / 2 + 0.02, (-17.4 if k == 2 else -length / 2 - 0.02)
            for side in (-1, 1):
                d.add("box", chain(b, tr((side * width * 0.32, 0.55, front)), sc((0.6, 0.4, 0.1))), rgb("#fff1d0"),
                      14.0, GLOW)
                d.add("box", chain(b, tr((side * width * 0.36, 0.7, back)), sc((0.55, 0.4, 0.1))), rgb("#ff2a14"),
                      3.0 + 14.0 * braking[sel], GLOW)

    def _draw_hawks(self, d: Instances, t: float) -> None:
        h = self.hawks
        ang = h["phase"] + h["speed"] * t
        pos = h["center"] + np.stack([h["r"] * np.cos(ang), 2.0 * np.sin(ang * 0.5), h["r"] * np.sin(ang)], 1)
        yaw = ang + np.where(h["speed"] > 0, 0.0, math.pi)  # flying along the circle
        bank = np.where(h["speed"] > 0, -0.35, 0.35)
        body = chain(tr(pos), ry(-yaw), rz(bank))
        d.add("box", chain(body, sc((0.22, 0.18, 0.55))), rgb("#3a2a20"))
        for side in (-1, 1):
            tip = 0.12 * math.sin(t * 1.3)
            d.add("box", chain(body, rz(side * tip), tr((side * 0.6, 0, 0)), sc((1.2, 0.05, 0.35))), rgb("#4a3626"))

    def lights(self, t: float, fx: Effects, cam: np.ndarray, power: float) -> list:
        return []
