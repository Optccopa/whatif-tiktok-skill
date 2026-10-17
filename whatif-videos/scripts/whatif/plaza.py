"""The plaza scene: a riverside city square on a summer evening, for "what if it got colder every second?".

Layout (meters): the camera stands above the square and looks north (-z) across it to a
railing, the river, a blue truss bridge on the left and brick buildings on the far bank.
A fountain plays in the foreground, a big tree shades the right, pigeons peck by a balloon
cart, and people stroll until the cold drives them down the subway stairs. Every change
follows the temperature: the fountain and river freeze, the river steams, leaves fall,
carbon dioxide snows out, oxygen and then nitrogen rain out of the air, and the sky goes
black and silent on the way to absolute zero.
"""

from __future__ import annotations

import math

import numpy as np

from whatif import props
from whatif.gl import Frame, look_at, ortho
from whatif.meshes import BUILDING, GROUND, Instances, beam, chain, rgb, rx, ry, sc, tr
from whatif.scenario import Effects, Timeline, format_temp, ramp

RAIL_Z = -30.0
RIVER = (-32.0, -62.0)  # near and far bank (z)
WATER_Y = -1.8
FOUNTAIN = np.array([-9.0, 0.0, 12.0])
TREE = (15.0, -21.0)
SUBWAY = np.array([11.0, 0.0, -6.0])
SUN_AZ_EL = (14.0, 13.0)
CAMERA = np.array([3.0, 15.0, 42.0])
LOOK = np.array([0.0, 6.0, -50.0])


def _dir(az, el):
    a, e = math.radians(az), math.radians(el)
    return np.array([math.sin(a) * math.cos(e), math.sin(e), -math.cos(a) * math.cos(e)])


def _mix(a, b, k):
    return tuple(x * (1 - k) + y * k for x, y in zip(a, b))


class Plaza:
    body_az_el = SUN_AZ_EL
    moon_az_el = (-40.0, 25.0)

    def __init__(self, timeline: Timeline, seed: int = 0):
        self.timeline = timeline
        rng = np.random.default_rng(seed + 31)
        self.static = Instances()
        self._ground(rng)
        self._river_and_bridge(rng)
        self._far_bank(rng)
        self._square(rng)
        self._crowd(rng)
        self.sign_point = props.sign_post(self.static, 16.0, -15.0, h=4.6, w=3.0, panel_h=1.5)
        n = 34
        self.pigeons = np.stack([rng.uniform(0, 12, n), np.zeros(n), rng.uniform(4, 26, n)], 1)

    # Static -----------------------------------------------------------------

    def _ground(self, rng) -> None:
        s = self.static
        s.add("box", props.block(-400, 400, -6, 0, RIVER[0] + 1.5, 300), rgb("#d6d0c6"), 0.0, GROUND)  # paving
        line = rgb("#b9b2a6")
        for z in np.arange(RAIL_Z, 60, 2.5):  # tile joints
            s.add("box", props.block(-60, 60, 0, 0.015, z - 0.04, z + 0.04), line)
        for x in np.arange(-60, 60.1, 2.5):
            s.add("box", props.block(x - 0.04, x + 0.04, 0, 0.015, RAIL_Z, 60), line)
        # Railing along the river.
        s.add("box", props.block(-120, 120, 1.05, 1.15, RAIL_Z - 0.05, RAIL_Z + 0.05), props.DARK)
        s.add("box", props.block(-120, 120, 0.5, 0.56, RAIL_Z - 0.03, RAIL_Z + 0.03), props.DARK)
        xs = np.arange(-120, 120.1, 1.6)
        s.add("box", chain(tr(np.stack([xs, np.zeros_like(xs), np.full_like(xs, RAIL_Z)], 1)), sc((0.07, 1.1, 0.07))),
              props.DARK)

    def _river_and_bridge(self, rng) -> None:
        s = self.static
        near, far = RIVER
        s.add("box", props.block(-400, 400, -8, 0.0, near, near + 1.5), rgb("#a8a196"))  # embankment walls
        s.add("box", props.block(-400, 400, -8, 0.0, far - 1.5, far), rgb("#a8a196"))
        s.add("box", props.block(-400, 400, -9, -6, far, near), rgb("#4a4238"))  # river bed
        # A blue steel truss bridge crossing on the left, angled toward us.
        base = chain(tr((-30.0, 0.0, (near + far) / 2)), ry(0.42))
        length, half_w, deck_y, top_y = 46.0, 3.4, 2.2, 8.2
        steel = rgb("#3b5fae")
        s.add("box", chain(base, tr((0, deck_y - 0.6, 0)), sc((half_w * 2, 0.6, length))), rgb("#7a7670"))
        zs = np.linspace(-length / 2, length / 2, 9)
        for side in (-1, 1):
            x = side * half_w
            bot = np.stack([np.full_like(zs, x), np.full_like(zs, deck_y), zs], 1)
            top = np.stack([np.full_like(zs, x), np.full_like(zs, top_y), zs], 1)
            pb = np.einsum("ij,nj->ni", base[:3, :3], bot) + base[:3, 3]
            pt = np.einsum("ij,nj->ni", base[:3, :3], top) + base[:3, 3]
            s.add("box", beam(pb[:-1], pb[1:], 0.35), steel)
            s.add("box", beam(pt[1:-2], pt[2:-1], 0.35), steel)
            s.add("box", beam(pb[1:-1], pt[1:-1], 0.25), steel)
            s.add("box", beam(pb[:-1:2], pt[1::2], 0.22), steel)
            s.add("box", beam(pb[2::2], pt[1::2], 0.22), steel)
            s.add("box", beam(pb[:1], pt[1:2], 0.3), steel)
            s.add("box", beam(pb[-1:], pt[-2:-1], 0.3), steel)
        for k in range(1, 8, 2):  # cross braces over the deck
            a = np.einsum("ij,j->i", base[:3, :3], (-half_w, top_y, zs[k])) + base[:3, 3]
            b = np.einsum("ij,j->i", base[:3, :3], (half_w, top_y, zs[k])) + base[:3, 3]
            s.add("box", beam(a, b, 0.2), steel)
        # A tour boat moored on the river.
        s.add("hull", chain(tr((4.0, WATER_Y - 0.6, -47.0)), ry(math.pi / 2 + 0.05), sc((3.2, 1.6, 11.0))), props.WHITE)
        s.add("box", chain(tr((4.0, WATER_Y + 1.0, -47.0)), sc((7.0, 1.4, 2.4))), rgb("#f4f1ea"))
        s.add("box", chain(tr((4.0, WATER_Y + 2.4, -47.0)), sc((7.4, 0.15, 2.8))), rgb("#e2553d"))

    def _far_bank(self, rng) -> None:
        s = self.static
        far = RIVER[1]
        s.add("box", props.block(-600, 600, -6, 0, -900, far - 1.5), rgb("#bdb6aa"), 0.0, GROUND)
        x = -70.0
        while x < 80:
            w = rng.uniform(14, 22)
            h = rng.uniform(16, 27)
            d = rng.uniform(12, 18)
            z = far - 6 - d / 2 + rng.uniform(-2, 2)
            color = rgb(rng.choice(["#b5503a", "#a8452f", "#8fa0bf", "#c86a4a", "#d8c4a6", "#9aa8c4"]))
            s.add("box", props.block(x, x + w, 0, h, z - d / 2, z + d / 2), color, 0.0, BUILDING,
                  seed=rng.uniform(0, 100), p1=0.08)
            s.add("box", props.block(x - 0.2, x + w + 0.2, h, h + 0.5, z - d / 2 - 0.2, z + d / 2 + 0.2),
                  tuple(c * 0.7 for c in color))
            s.add("box", props.block(x + 0.5, x + w - 0.5, 0, 3.0, z + d / 2, z + d / 2 + 0.1), rgb("#2a2a30"))  # shopfronts
            x += w + rng.uniform(0.0, 1.5)
        for _ in range(9):  # pale towers behind
            tx = rng.uniform(-90, 100)
            w = rng.uniform(16, 26)
            h = rng.uniform(38, 62)
            tz = rng.uniform(-150, -115)
            s.add("box", props.block(tx - w / 2, tx + w / 2, 0, h, tz - 10, tz + 10),
                  rgb(rng.choice(["#cfcad2", "#c4c2cc", "#d8d2cc"])), 0.0, BUILDING, seed=rng.uniform(0, 100), p1=0.05)

    def _square(self, rng) -> None:
        s = self.static
        fx, _, fz = FOUNTAIN
        stone = rgb("#d9d1c3")
        s.add("cyl", chain(tr((fx, 0, fz)), sc((13.0, 0.9, 13.0))), stone)
        s.add("cyl", chain(tr((fx, 0, fz)), sc((11.8, 0.92, 11.8))), rgb("#c7bfb2"))
        s.add("cyl", chain(tr((fx, 0, fz)), sc((1.4, 2.6, 1.4))), stone)
        s.add("cyl", chain(tr((fx, 2.6, fz)), sc((4.2, 0.5, 4.2))), stone)
        s.add("cyl", chain(tr((fx, 3.1, fz)), sc((0.6, 1.0, 0.6))), stone)
        self.tree_balls = props.round_tree(s, TREE[0], TREE[1], 12.0, leaf="#3f8a3a", rng=rng)
        self.tree_balls += props.round_tree(s, -34.0, 30.0, 9.0, rng=rng)
        self.tree_balls += props.round_tree(s, 34.0, 24.0, 8.0, rng=rng)
        self.lamps = []
        for x, z in ((-4.0, 20.0), (7.0, 0.0), (-22.0, -24.0), (4.0, -26.0), (22.0, 16.0), (-26.0, 12.0), (16.0, -28.0)):
            self.lamps.append(props.lamp_post(s, x, z, h=4.8))
        for x in (-16.0, -8.0, 8.0):
            props.bench(s, x, RAIL_Z + 2.0, math.pi, color="#d0623e")
        # Flagpole at the railing.
        s.add("cyl5", chain(tr((1.5, 0, RAIL_Z + 1.0)), sc((0.14, 11.0, 0.14))), rgb("#e8e6e1"))
        s.add("box", chain(tr((2.9, 9.0, RAIL_Z + 1.0)), sc((2.8, 1.7, 0.05))), rgb("#d9312c"))
        # Subway entrance: low walls around a dark stairwell, a rail, and a blue sign.
        sx, _, sz = SUBWAY
        wall = rgb("#2b2b30")
        s.add("box", props.block(sx - 2.2, sx + 2.2, 0, 1.1, sz - 3.4, sz - 3.0), wall)
        for dx in (-2.2, 2.2):
            s.add("box", props.block(sx + dx - 0.2, sx + dx + 0.2, 0, 1.1, sz - 3.4, sz + 3.4), wall)
        s.add("box", props.block(sx - 2.0, sx + 2.0, 0, 0.02, sz - 3.0, sz + 3.4), rgb("#050506"))
        s.add("cyl5", chain(tr((sx + 3.2, 0, sz + 3.0)), sc((0.12, 3.2, 0.12))), props.DARK)
        s.add("box", chain(tr((sx + 3.2, 2.6, sz + 3.0)), sc((1.3, 1.1, 0.12))), rgb("#2f6fd6"))
        self.balloon_anchor = np.array([7.0, 1.4, 16.0])
        s.add("box", chain(tr((7.0, 0.3, 16.0)), sc((2.2, 1.4, 1.4))), rgb("#f2c14e"))
        for dx in (-0.8, 0.8):
            s.add("cyl", chain(tr((7.0 + dx, 0.3, 16.0)), rx(math.pi / 2), sc((0.6, 1.5, 0.6)), tr((0, -0.5, 0))), props.DARK)
        props.trash_bin(s, -2.0, -4.0)
        props.trash_bin(s, 17.0, 6.0)

    def _crowd(self, rng) -> None:
        self.crowd = props.Crowd(rng, [(80, (-30.0, 30.0), (-26.0, 34.0), 0), (40, (-30.0, 30.0), (-26.0, 34.0), 1)], exit_speed=4.5)
        p = self.crowd.p
        n = self.crowd.n
        leave_f = rng.uniform(30, 64, n)  # everyone has their own limit
        p["leave"] = np.array([self.timeline.time_at_temp(f) for f in leave_f])
        if self.timeline.t_dark < 1e8:  # the Sun goes out: people head home in the dark
            p["leave"] = self.timeline.t_dark + rng.uniform(0.2, 3.5, n)
        p["exit"] = np.tile(SUBWAY + (0, 0, 2.6), (n, 1)) + np.stack([rng.uniform(-1.2, 1.2, n), np.zeros(n),
                                                                     np.zeros(n)], 1)
        p["sink"] = np.full(n, 3.5)  # down the stairs

    # Dynamic ----------------------------------------------------------------

    def dynamic(self, t: float, fx: Effects) -> Instances:
        d = Instances()
        temp = fx.temp_f
        tl = self.timeline
        # Water: the river and the fountain basin (both freeze through the shader's ice).
        d.add("quad", chain(tr((0, WATER_Y, (RIVER[0] + RIVER[1]) / 2)), sc((900, 1, RIVER[0] - RIVER[1]))),
              rgb("#3c6a96"), 0.0, 1)
        d.add("cyl", chain(tr((FOUNTAIN[0], 0.55, FOUNTAIN[2])), sc((11.6, 0.3, 11.6))), rgb("#4f86b8"), 0.0, 1)
        self._draw_fountain(d, t, temp)
        self.crowd.draw(d, t)
        props.draw_pigeons(d, self.pigeons, t, min(tl.time_at_temp(48.0), tl.t_dark + 0.3))
        # Balloons shrink as the helium inside contracts (volume goes with absolute temperature).
        k = max((temp + 459.67) / (75 + 459.67), 0.02) ** (1 / 3)
        props.draw_balloons(d, self.balloon_anchor, 12, t, scale=k)
        leaf = _mix(rgb("#3f8a3a"), rgb("#c8782c"), ramp(temp, 55.0, 30.0))
        props.draw_leaves(d, self.tree_balls, leaf, scale=1 - ramp(temp, 30.0, 8.0))
        self._draw_falling_leaves(d, t, temp)
        steam = ramp(temp, 20.0, -10.0) * (1 - ramp(temp, -50.0, -90.0))
        if steam > 0:  # sea smoke: the river, warmer than the air, steams
            xs = np.linspace(-60, 60, 30)
            props.draw_puffs(d, xs, np.full_like(xs, -47.0), WATER_Y + 0.3, t, steam, rgb("#eef2f6"), rise=1.6,
                             spread=7.0, size=3.2, n_max=240)
        self._draw_falling(d, t, temp)
        return d

    def _draw_fountain(self, d: Instances, t: float, temp: float) -> None:
        t_freeze = self.timeline.time_at_temp(32.0)
        frozen = t >= t_freeze
        tt = min(t, t_freeze)
        n_jets, per = 10, 14
        j = np.arange(n_jets * per)
        jet = j // per
        k = (j % per) / per
        a = jet / n_jets * 2 * np.pi
        cycle = 1.2
        u = np.mod(k + tt / cycle, 1.0)  # each drop's progress along its arc
        r = 0.7 + 4.2 * u
        y = 3.4 + 4.2 * u - 6.5 * u * u
        pos = np.stack([FOUNTAIN[0] + r * np.cos(a), np.maximum(y, 0.7), FOUNTAIN[2] + r * np.sin(a)], 1)
        color = rgb("#e6f2fb") if frozen else rgb("#f4fbff")
        size = 0.32 if frozen else 0.26
        d.add("ico0", chain(tr(pos), sc(size)), color, 0.6 if not frozen else 0.15)
        top = FOUNTAIN + (0, 4.1, 0)
        up = 0.6 * (1 + 0.15 * math.sin(tt * 9))
        d.add("ico0", chain(tr(top + (0, up, 0)), sc((0.5, 1.4, 0.5))), color, 0.4)

    def _draw_falling_leaves(self, d: Instances, t: float, temp: float) -> None:
        t0 = self.timeline.time_at_temp(32.0)
        t1 = self.timeline.time_at_temp(5.0)
        if not t0 < t < t1 + 4:
            return
        rng = np.random.default_rng(8)
        n = 160
        c = np.array([b[0] for b in self.tree_balls[:7]])
        src = c[rng.integers(0, len(c), n)] + rng.uniform(-2.5, 2.5, (n, 3))
        start = rng.uniform(t0, t1, n)
        age = t - start
        live = (age > 0) & (age < 4.0)
        if not live.any():
            return
        p = src[live] + np.stack([np.sin(age[live] * 2 + start[live]) * 1.2, -age[live] * 2.2, age[live] * 0.6], 1)
        p[:, 1] = np.maximum(p[:, 1], 0.05)
        d.add("box", chain(tr(p), ry(age[live] * 3), sc((0.35, 0.04, 0.25))), rgb("#c06a2a"))

    def _draw_falling(self, d: Instances, t: float, temp: float) -> None:
        """Snow, then carbon dioxide frost, then rain of liquid oxygen and nitrogen, all near the camera."""
        cam = CAMERA
        rng = np.random.default_rng(15)
        n = 700
        x0, z0 = rng.uniform(-30, 30, n), rng.uniform(-40, 30, n)
        off = rng.uniform(0, 40, n)
        snow = ramp(temp, 30.0, 10.0) * (1 - ramp(temp, -20.0, -45.0))
        dry_ice = ramp(temp, -105.0, -125.0) * (1 - ramp(temp, -230.0, -260.0))
        o2 = ramp(temp, -297.0, -303.0) * (1 - ramp(temp, -340.0, -350.0))
        n2 = ramp(temp, -320.0, -326.0) * (1 - ramp(temp, -345.0, -352.0))
        if snow + dry_ice > 0.01:
            m = int(n * min(snow + dry_ice, 1.0))
            y = 26 - np.mod(t * 2.2 + off[:m], 26.0)
            x = x0[:m] + np.sin(t * 0.8 + off[:m]) * 1.2
            col = rgb("#ffffff") if snow > dry_ice else rgb("#f4f8ff")
            d.add("ico0", chain(tr(np.stack([x, y, z0[:m]], 1)), sc(0.13 if snow > dry_ice else 0.09)), col,
                  0.5 if snow > dry_ice else 2.5)
        rain = max(o2 * 0.6, n2)
        if rain > 0.01:  # liquid air falling in streaks
            m = int(n * rain)
            y = 30 - np.mod(t * 18.0 + off[:m] * 3, 30.0)
            p = np.stack([x0[:m], y, z0[:m]], 1)
            col = rgb("#9cc8ff") if o2 > n2 else rgb("#d8e6f4")
            d.add("box", beam(p, p + (0, 1.4, 0), 0.05), col, 1.2)

    def power(self, fx: Effects) -> float:
        if fx.daylight >= 1.0:
            return 1.0
        return 1.0 - ramp(fx.temp_f, -40.0, -110.0)  # with no Sun, the grid holds for a while, then fails

    def lights(self, t: float, fx: Effects, cam: np.ndarray, power: float) -> list:
        """At night: the street lamps light the square (up to 8, nearest the camera)."""
        if power < 0.02 or fx.daylight >= 1.0:
            return []
        lamps = np.array(self.lamps)
        k = 1.4 * power * (1 - fx.daylight)
        d = np.linalg.norm(lamps - cam, axis=1)
        return [(x, y - 0.3, z, 16.0, 1.0 * k, 0.78 * k, 0.5 * k) for x, y, z in lamps[np.argsort(d)[:8]]]

    def camera(self, t: float, fx: Effects) -> tuple[np.ndarray, np.ndarray, float]:
        return CAMERA.copy(), LOOK.copy(), 60.0

    def sign_text(self, fx: Effects) -> str:
        v = round(fx.temp_f)
        return f"{'−' if v < 0 else ''}{abs(v)}°F" if fx.temp_f > -459.6 else "−459.67°F"

    def lens_frost(self, fx: Effects) -> float:
        return 0.75 * ramp(fx.temp_f, 0.0, -160.0)

    # Look -------------------------------------------------------------------

    def look(self, t: float, fx: Effects, pos, look_pt, fov) -> Frame:
        temp = fx.temp_f
        icy = ramp(temp, 40.0, -30.0)  # summer evening to frozen haze
        clear = ramp(temp, -60.0, -140.0)  # the air dries out: clear, deep blue
        vacuum = ramp(temp, -318.0, -350.0)  # the air rains out: black sky, no haze
        dark = 1.0 - min(fx.daylight, 1.0)  # the Sun has gone out
        night = max(vacuum, dark)
        night_sky = (0.003, 0.004, 0.01) if vacuum < dark else (0.0, 0.0, 0.0)

        def sky(summer, frozen, dry, scale=1.0):
            c = _mix(_mix(rgb(summer, scale), rgb(frozen, scale), icy), rgb(dry, scale), clear)
            return _mix(c, night_sky, night)

        sun = _dir(*SUN_AZ_EL)
        horizon = sky("#ffcf98", "#d6dee8", "#b8cde6")
        fog = 0.0006 + 0.0022 * icy * (1 - clear) + 0.0004 * clear
        frame = Frame(
            cam_pos=pos, cam_look=look_pt, fov=fov, time=t,
            zenith=sky("#6f8cc8", "#b4c4d6", "#2e4a8a"), mid=sky("#d8b8b8", "#d0dae4", "#6f8fc0"), horizon=horizon,
            stars=night, moon_dir=_dir(-40, 25), moon_r=0.0, moon_col=(0, 0, 0), sun_dir=sun, body_dir=sun, body_d=1e6,
            body_frame=np.eye(3), flattening=0.0, rings=(0.0, 0.0), bands=((1, 1, 1),) * 3, ring_col=(1, 1, 1),
            body_bright=0.0, body_light=(0.0, 0.0, 0.0), glow_col=(0, 0, 0), glow=0.0,
            amb_sky=_mix(sky("#aab4d0", "#c8d2de", "#8aa4cc", 0.62), (0.02, 0.022, 0.035), night),
            amb_ground=_mix(sky("#8a7a68", "#a8b0b8", "#7a8aa0", 0.5), (0.01, 0.01, 0.015), night),
            fog_col=horizon, fog_density=fog * (1 - vacuum), power=self.power(fx),
            lava=0.0, exposure=1.1, flash=0.0, flash_col=(1, 1, 1), fade=fx.fade,
        )
        frame.body_type = -1
        frame.moon_vis = 0.0
        frame.lights = []
        frame.key_dir = sun
        warm = _mix(rgb("#ffcf98"), rgb("#fff4e6"), max(icy, clear))
        frame.key_col = tuple(c * (2.2 - 0.9 * icy * (1 - clear) + 0.6 * vacuum) * (1 - dark) for c in warm)
        frame.lights = self.lights(t, fx, pos, self.power(fx)) if dark > 0 else []
        frame.cloud = sky("#f2d8d0", "#e8ecf0", "#c8d8ec", 0.95)
        frame.cloud_amt = (1.0 - clear) * (1 - vacuum)
        frame.sun_show, frame.sun_r = 1.0 - dark, 0.0095
        frame.sun_b, frame.sun_glow = 6.0 + 6.0 * vacuum, 0.8 * (1 - vacuum)
        frame.frost = ramp(temp, 32.0, -60.0)
        frame.ice = ramp(temp, 32.0, -40.0)
        center = np.array([0.0, 0.0, -10.0])
        frame.shadow_vp = ortho(110.0, 10.0, 9000.0) @ look_at(center + sun * 4000.0, center)
        return frame

