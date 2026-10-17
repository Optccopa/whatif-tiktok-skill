"""The boardwalk scene: a carnival pier at sunset, for "what if it got hotter every second?".

Layout (meters): the camera stands high over the start of the boardwalk and looks down it
(-z) toward a Ferris wheel on the lakeshore. Striped stalls line both sides under zigzags of
string lights; a circus tent sits on the left, a swing ride on the right; the lake, a far
city and purple hills sit under a low sun. Every change follows the temperature: people leave,
grass turns to straw, the lake boils, balloons pop, the power fails, popcorn pops at 356°F,
paper lights at 451°F and wood at 572°F.
"""

from __future__ import annotations

import math

import numpy as np

from whatif import props
from whatif.gl import Frame, look_at, ortho
from whatif.meshes import BUILDING, GLOW, GROUND, STRIPES, Instances, chain, rgb, ry, rz, sc, tr
from whatif.scenario import Effects, Timeline, ramp

HALF = 8.0  # boardwalk half-width
END_Z = -112.0  # the boardwalk ends at a small plaza on the shore
SHORE_Z = -138.0
WHEEL = (-3.0, 21.0, -127.0)
WHEEL_R = 17.0
SUN_AZ_EL = (16.0, 6.5)
SUN_HIGH_AZ_EL = (6.0, 15.5)  # when the Sun itself is the story, keep it clear of the wheel and the edges
CAMERA = np.array([0.0, 15.5, 44.0])
LOOK = np.array([0.0, 8.5, -60.0])


def _dir(az, el):
    a, e = math.radians(az), math.radians(el)
    return np.array([math.sin(a) * math.cos(e), math.sin(e), -math.cos(a) * math.cos(e)])


def _mix(a, b, k):
    return tuple(x * (1 - k) + y * k for x, y in zip(a, b))


class Boardwalk:
    body_az_el = SUN_AZ_EL
    moon_az_el = (-40.0, 25.0)

    def __init__(self, timeline: Timeline, seed: int = 0):
        self.timeline = timeline
        rng = np.random.default_rng(seed + 21)
        self.static = Instances()
        self.trees = []
        self.bins = []
        self.stalls = []
        self._ground(rng)
        self._far_side(rng)
        self._stalls(rng)
        self._lights(rng)
        self._attractions(rng)
        self._greenery(rng)
        self._crowd(rng)
        self._fire_spots(rng)
        self.sign_point = props.sign_post(self.static, 6.4, 14.0, h=4.4, w=3.0, panel_h=1.5)

    # Static -----------------------------------------------------------------

    def _ground(self, rng) -> None:
        s = self.static
        s.add("box", props.block(-900, 900, -8, 0, SHORE_Z, 400), rgb("#6e9a48"), 0.0, GROUND)  # grass
        s.add("box", props.block(-HALF, HALF, 0, 0.12, END_Z, 120), rgb("#9b8670"), 0.0, STRIPES,
              p1=(120 - END_Z) / 0.9, p2=2)  # boardwalk planks
        s.add("box", props.block(-34, 34, 0, 0.12, SHORE_Z, END_Z), rgb("#b7a48c"), 0.0, STRIPES, p1=30, p2=2)
        for side in (-1, 1):  # sandy strips along the stall backs
            s.add("box", props.block(side * HALF if side > 0 else -HALF - 9, HALF + 9 if side > 0 else -HALF,
                                     0, 0.06, END_Z, 120), rgb("#c8b28a"), 0.0, GROUND)
        s.add("box", props.block(-900, 900, -9, -6, -3000, SHORE_Z), rgb("#5a4a3a"), 0.0, GROUND)  # lake bed
        s.add("box", props.block(-900, 900, -6, 0.1, SHORE_Z - 3, SHORE_Z), rgb("#c9b38c"))  # sea wall top
        rail_z = SHORE_Z + 0.4
        s.add("box", props.block(-34, 34, 1.0, 1.1, rail_z - 0.05, rail_z + 0.05), props.WHITE)
        xs = np.arange(-34, 34.1, 2.0)
        s.add("box", chain(tr(np.stack([xs, np.zeros_like(xs), np.full_like(xs, rail_z)], 1)), sc((0.08, 1.1, 0.08))),
              props.WHITE)

    def _far_side(self, rng) -> None:
        s = self.static
        for _ in range(120):  # a low city along the far shore
            x = rng.uniform(-1400, 1400)
            z = rng.uniform(-1250, -1050)
            w, h = rng.uniform(20, 60), rng.uniform(10, 45) * (1.6 if abs(x) > 300 and rng.random() < 0.2 else 1)
            s.add("box", props.block(x - w / 2, x + w / 2, -2, h, z - 15, z + 15),
                  rgb(rng.choice(["#b9a8b8", "#a798b0", "#c4b0b4", "#9c90a8"])), 0.0, BUILDING,
                  seed=rng.uniform(0, 100), p1=0.25)
        for _ in range(22):  # soft purple hills
            x = rng.uniform(-3500, 3500)
            h = rng.uniform(120, 260) if abs(x) < 900 else rng.uniform(200, 420)
            r = rng.uniform(700, 1400)
            s.add("cone6", chain(tr((x, -10, rng.uniform(-2600, -1800))), ry(rng.uniform(0, 1)), sc((r, h, r * 0.7))),
                  rgb(rng.choice(["#8a76a6", "#7f6c9c", "#9580ad"])))
        for _ in range(9):  # sailboats on the lake
            x, z = rng.uniform(-300, 300), rng.uniform(-700, -260)
            s.add("hull", chain(tr((x, -1.2, z)), ry(rng.uniform(0, 6)), sc((2.2, 1.2, 6.0))), props.WHITE)
            s.add("cone", chain(tr((x, -0.2, z)), sc((3.0, 8.0, 0.2))), props.WHITE)

    def _stalls(self, rng) -> None:
        for side in (-1, 1):
            z = 40.0
            while z > END_Z + 6:
                if rng.random() < 0.88:
                    w = rng.uniform(4.2, 5.2)
                    st = props.stall(self.static, side * (HALF + 2.6), z, -side * math.pi / 2,
                                     props.AWNINGS[rng.integers(0, len(props.AWNINGS))], rng, width=w)
                    self.stalls.append(st)
                z -= rng.uniform(5.4, 6.4)

    def _lights(self, rng) -> None:
        s = self.static
        zs = np.arange(40.0, END_Z, -7.5)
        for z in zs[::2]:
            for side in (-1, 1):
                s.add("cyl5", chain(tr((side * (HALF - 0.3), 0, z)), sc((0.12, 5.2, 0.12))), props.WHITE)
        for i, z in enumerate(zs[:-1]):
            a = (-(HALF - 0.3) if i % 2 == 0 else HALF - 0.3, 5.0, z)
            b = (HALF - 0.3 if i % 2 == 0 else -(HALF - 0.3), 5.0, zs[i + 1])
            props.string_lights(s, a, b, sag=0.9, n=18, rng=rng)
        for z in np.arange(30.0, END_Z, -16.0):
            props.lamp_post(s, -HALF + 0.6, z)
            props.lamp_post(s, HALF - 0.6, z - 8)
        for x in np.arange(-30, 31, 10.0):  # lights around the end plaza
            props.lamp_post(s, x, SHORE_Z + 2.0)

    def _attractions(self, rng) -> None:
        s = self.static
        props.circus_tent(s, -27.0, 6.0, r=9.5, h=7.0)
        props.circus_tent(s, -34.0, -48.0, r=7.0, h=5.5)
        # The swing ride (its seats spin while the power is on) and its striped canopy.
        self.swing = np.array([24.0, 0.0, -84.0])
        s.add("cyl", chain(tr(self.swing), sc((1.0, 11.0, 1.0))), props.WHITE)
        s.add("cyl", chain(tr(self.swing), sc((7.0, 0.6, 7.0))), rgb("#e8628f"))
        # Red game booths and a ticket hut.
        for x, z in ((19.0, -40.0), (-19.0, -86.0), (19.0, 2.0)):
            s.add("box", props.block(x - 3.5, x + 3.5, 0, 3.4, z - 2.5, z + 2.5), rgb("#c0392b"))
            s.add("box", props.block(x - 3.8, x + 3.8, 3.4, 3.8, z - 2.8, z + 2.8), rgb("#f2efe8"))
        self.popcorn = props.popcorn_cart(s, 3.4, 4.0, yaw=0.3)
        self.balloon_anchor = np.array([-3.2, 1.3, 10.0])
        s.add("box", chain(tr((-3.2, 0, 10.0)), sc((1.0, 1.1, 0.8))), rgb("#3d7bd9"))
        for z in np.arange(30.0, END_Z, -24.0):  # benches and bins in the middle
            props.bench(s, 0.0, z, math.pi / 2, color=rng.choice(["#c4553c", "#3d7bd9", "#e8a23a"]))
            self.bins.append((1.6, z + 2.0))
        # Ferris wheel platform.
        s.add("box", props.block(WHEEL[0] - 12, WHEEL[0] + 12, 0, 0.5, WHEEL[2] - 5, WHEEL[2] + 5), rgb("#d9d2c4"))

    def _greenery(self, rng) -> None:
        placed = 0
        while placed < 34:
            side = rng.choice([-1, 1])
            x = side * rng.uniform(HALF + 9, 70)
            z = rng.uniform(-125, 30)
            if (abs(x + 27) < 12 and abs(z - 6) < 12) or (abs(x - 24) < 10 and abs(z + 84) < 10):
                continue
            h = rng.uniform(6, 10)
            self.trees += props.round_tree(self.static, x, z, h, rng=rng)
            placed += 1
        n = 26  # bushes along the stall backs
        bx = rng.choice([-1, 1], n) * rng.uniform(HALF + 6, HALF + 8.5, n)
        bz = rng.uniform(END_Z, 36, n)
        self.static.add("ico", chain(tr(np.stack([bx, np.full(n, 0.4), bz], 1)), sc(rng.uniform(1.4, 2.2, n))),
                        rgb("#3d7a3a"))

    def _crowd(self, rng) -> None:
        self.crowd = props.Crowd(rng, [(230, (-6.5, 6.5), (END_Z + 6, 34.0), 1), (24, (-26.0, 24.0), (SHORE_Z + 4, END_Z), 0)])
        p = self.crowd.p
        n = self.crowd.n
        heat = self.timeline
        leave_f = rng.uniform(92, 150, n)  # everyone has their own limit
        p["leave"] = np.array([heat.time_at_temp(f) for f in leave_f])
        side = np.where(p["x"] >= 0, 1.0, -1.0)
        p["exit"] = np.stack([side * (HALF + 3.0), np.zeros(n), p["z"] - 1.5], 1)  # into the shade of the stalls

    def _fire_spots(self, rng) -> None:
        """What burns, and at what temperature (°F): litter (451), awnings (~480), then all the wood (572)."""
        pts = []
        for _ in range(70):  # dropped napkins, tickets, flyers
            pts.append((rng.uniform(-7, 7), 0.12, rng.uniform(END_Z, 38), rng.uniform(0.4, 0.9), rng.uniform(451, 468)))
        for st in self.stalls:
            x, _, z = st["pos"]
            k = 1 if x > 0 else -1
            for dz in np.linspace(-st["width"] / 2 + 0.6, st["width"] / 2 - 0.6, 3):
                pts.append((x - k * 1.0, 2.7, z + dz, rng.uniform(1.0, 1.8), rng.uniform(470, 530)))  # awning
            pts.append((x, 1.2, z, rng.uniform(2.2, 3.4), rng.uniform(568, 590)))  # the booth itself
        for z in np.arange(40, END_Z, -5.0):  # boards
            pts.append((rng.uniform(-6, 6), 0.12, z, rng.uniform(1.6, 2.6), rng.uniform(572, 600)))
        for (c, r) in self.trees[::3]:
            pts.append((c[0], c[1] - r, c[2], r * 2.2, rng.uniform(572, 595)))
        for x, z, r in ((-27.0, 6.0, 9.5), (-34.0, -48.0, 7.0)):  # the tents' canvas
            for a in np.linspace(0, 2 * np.pi, 6, endpoint=False):
                pts.append((x + r * 0.8 * math.cos(a), 3.0, z + r * 0.8 * math.sin(a), rng.uniform(3, 4.5),
                            rng.uniform(480, 540)))
        a = np.array(pts)
        self.fires = dict(x=a[:, 0], y=a[:, 1], z=a[:, 2], size=a[:, 3], temp=a[:, 4],
                          seed=rng.uniform(0, 100, len(a)))

    # Dynamic ----------------------------------------------------------------

    def power(self, temp: float) -> float:
        return 1.0 - ramp(temp, 255.0, 290.0)  # wiring and transformers give out

    def dynamic(self, t: float, fx: Effects) -> Instances:
        d = Instances()
        temp = fx.temp_f
        tl = self.timeline
        level = -1.0 - 5.0 * ramp(temp, 212.0, 640.0)  # the lake boils away
        d.add("quad", chain(tr((0, level, -1600)), sc((3600, 1, 2800))), rgb("#3f6c96"), 0.0, 1)
        # The wheel slows to a stop when the power goes.
        t_stop = tl.time_at_temp(270.0)
        turn = 0.075 * (min(t, t_stop) + 2.0 * (1 - math.exp(-max(t - t_stop, 0) / 2.0)))
        char = ramp(temp, 575.0, 620.0)
        props.draw_ferris_wheel(d, WHEEL, WHEEL_R, t, turn, self.power(temp), char=char)
        self._draw_swing(d, t, temp)
        self.crowd.draw(d, t)
        pop_f = np.random.default_rng(4).uniform(285, 330, 14)
        props.draw_balloons(d, self.balloon_anchor, 14, t, popped=temp > pop_f)
        self._draw_popcorn(d, t, temp)
        self._draw_bins(d, temp)
        brown = _mix(rgb("#3f8a3a"), rgb("#8a5a24"), ramp(temp, 220.0, 380.0))
        props.draw_leaves(d, self.trees, brown, scale=(1 - 0.3 * ramp(temp, 300.0, 450.0)) * (1 - ramp(temp, 572.0, 586.0)))
        steam = ramp(temp, 205.0, 235.0) * (1 - ramp(temp, 470.0, 600.0))
        if steam > 0:
            xs = np.linspace(-120, 120, 40)
            props.draw_puffs(d, xs, np.full_like(xs, SHORE_Z - 30.0), level + 0.5, t, steam, rgb("#f2eef0"), rise=4.0,
                             spread=8.0, size=4.5, n_max=260)
        f = self.fires
        on = temp >= f["temp"]
        if on.any():
            grow = np.clip((temp - f["temp"][on]) / 25.0, 0.15, 1.0)
            props.draw_flames(d, f["x"][on], f["y"][on], f["z"][on], f["size"][on] * grow, f["seed"][on], t)
        return d

    def _draw_swing(self, d: Instances, t: float, temp: float) -> None:
        p = self.power(temp)
        t_stop = self.timeline.time_at_temp(270.0)
        angle = 0.9 * (min(t, t_stop) + 1.5 * (1 - math.exp(-max(t - t_stop, 0) / 1.5)))
        out = 0.55 * p + 0.05
        top = self.swing + (0, 10.5, 0)
        d.add("cone", chain(tr(top), sc((9.0, 2.4, 9.0))), rgb("#e8628f"), 0.0, STRIPES, p1=12, p2=1)
        a = angle + np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ring = np.stack([np.cos(a), np.zeros_like(a), np.sin(a)], 1)
        hang = top + ring * 4.0
        seat = hang + ring * 5.5 * math.sin(out) + np.array([0, -5.5 * math.cos(out), 0])
        d.add("box", props.beam(hang, seat, 0.05), props.DARK)
        d.add("box", chain(tr(seat), sc((0.7, 0.5, 0.7))), [props.GONDOLAS[i % 7] for i in range(12)])
        if p > 0.02:
            d.add("ico0", chain(tr(hang), sc(0.28)), props.BULBS[1], 9.0 * p, GLOW)

    def _draw_popcorn(self, d: Instances, t: float, temp: float) -> None:
        t_pop = self.timeline.time_at_temp(356.0)
        if t < t_pop:
            return
        rng = np.random.default_rng(12)
        n = 260
        delay = rng.uniform(0, 2.0, n)
        v = np.stack([rng.uniform(-2.5, 2.5, n), rng.uniform(3, 7, n), rng.uniform(-2.5, 2.5, n)], 1)
        age = np.maximum(t - t_pop - delay, 0)
        land = 2 * v[:, 1] / 9.8 + 0.25
        a = np.minimum(age, land)
        pos = self.popcorn + v * a[:, None] + np.array([0, -4.9, 0]) * (a * a)[:, None]
        pos[:, 1] = np.maximum(pos[:, 1], 0.08)
        live = age > 0
        burnt = ramp(temp, 451.0, 500.0)
        color = _mix(rgb("#fff3cc"), rgb("#3a2a1c"), burnt)
        d.add("ico0", chain(tr(pos[live]), sc(0.11)), color, 0.1)

    def _draw_bins(self, d: Instances, temp: float) -> None:
        sag = ramp(temp, 280.0, 380.0)  # plastic softens and slumps
        for x, z in self.bins:
            d.add("cyl", chain(tr((x, 0, z)), sc((0.7 + 0.5 * sag, 1.0 - 0.7 * sag, 0.7 + 0.5 * sag))), rgb("#3d6a4a"))

    def lights(self, t: float, fx: Effects, cam: np.ndarray, power: float) -> list:
        return []

    def camera(self, t: float, fx: Effects) -> tuple[np.ndarray, np.ndarray, float]:
        return CAMERA.copy(), LOOK.copy(), 60.0

    def sign_text(self, fx: Effects) -> str:
        return f"{round(fx.temp_f):,}°F"

    # Look -------------------------------------------------------------------

    def look(self, t: float, fx: Effects, pos, look_pt, fov) -> Frame:
        temp = fx.temp_f
        warm = ramp(temp, 150.0, 420.0)
        smoke = ramp(temp, 540.0, 640.0)
        glare = min(max((fx.daylight - 1.0) / 2.2, 0.0), 1.0)  # a closer Sun bleaches the sky

        def sky(a, b, c, scale=1.0):
            col = _mix(_mix(rgb(a, scale), rgb(b, scale), warm), rgb(c, scale), smoke)
            return _mix(col, rgb("#fff4dc", scale * 1.1), 0.35 * glare)

        sun = _dir(*(SUN_HIGH_AZ_EL if fx.daylight != 1.0 else SUN_AZ_EL))  # sun mode: up in open sky
        horizon = sky("#f6c8a0", "#ffb06a", "#9a4424")
        frame = Frame(
            cam_pos=pos, cam_look=look_pt, fov=fov, time=t,
            zenith=sky("#5a5cae", "#a0583e", "#3a1a12"), mid=sky("#b08cc0", "#e08850", "#6e2e1a"), horizon=horizon,
            stars=0.0, moon_dir=_dir(-40, 25), moon_r=0.0, moon_col=(0, 0, 0), sun_dir=sun, body_dir=sun, body_d=1e6,
            body_frame=np.eye(3), flattening=0.0, rings=(0.0, 0.0), bands=((1, 1, 1),) * 3, ring_col=(1, 1, 1),
            body_bright=0.0, body_light=(0.0, 0.0, 0.0), glow_col=(0, 0, 0), glow=0.0,
            amb_sky=sky("#a29ac8", "#c09070", "#5a3020", 0.62), amb_ground=sky("#8a6a50", "#a0603a", "#40200f", 0.5),
            fog_col=horizon, fog_density=0.0007 + 0.0009 * warm + 0.004 * smoke, power=self.power(temp), lava=0.0,
            exposure=1.05, flash=0.0, flash_col=(1, 1, 1), fade=fx.fade,
        )
        frame.body_type = -1
        frame.moon_vis = 0.0
        frame.lights = []
        frame.key_dir = sun
        frame.key_col = rgb("#ffc890", 2.3 * (1 - 0.65 * smoke) * fx.daylight ** 0.45)
        frame.cloud = sky("#efb0bc", "#f0a070", "#4a2a20", 0.95)
        frame.cloud_amt = 1.0 - 0.6 * warm
        frame.sun_show, frame.sun_r = 1.0, (0.012 if fx.daylight == 1.0 else 0.016) * fx.daylight ** 0.5  # angular size goes as 1/distance
        frame.sun_b = 7.0 * (1 - 0.6 * smoke) * fx.daylight ** 0.5
        frame.sun_glow = 0.9 * (1 - 0.5 * smoke) * fx.daylight ** 0.5
        frame.dry = ramp(temp, 110.0, 260.0)
        frame.scorch = 0.92 * ramp(temp, 575.0, 645.0)
        frame.boil = ramp(temp, 205.0, 225.0) * (1 - ramp(temp, 520.0, 620.0))
        center = np.array([0.0, 0.0, -60.0])
        frame.shadow_vp = ortho(150.0, 10.0, 9000.0) @ look_at(center + sun * 4000.0, center)
        return frame
