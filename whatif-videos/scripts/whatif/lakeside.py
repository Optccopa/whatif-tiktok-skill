"""Lakeside park at sunset: the starter template for new scenes (`whatif_cli.py new-scene <name>` copies it).

Layout (meters): the camera stands 16 m up and looks down a lamp-lit promenade (-z) to a pier with a
lighthouse on the lake. Trees, benches and strolling people line the path; across the water sit a far
town and purple hills. Heat and cold both read: people leave, grass dries, the lake boils or freezes,
the power fails, litter, benches and trees burn. Rename things, keep the structure.
"""

from __future__ import annotations

import numpy as np

from whatif import kit, props
from whatif.meshes import GLOW, STRIPES, Instances, chain, rgb, sc, tr
from whatif.scenario import ramp

SHORE_Z = -100.0
LIGHTHOUSE = (6.0, 0.0, -140.0)
# Palettes the sky blends toward as the story turns (any keys from kit.PALETTES).
HOT = dict(zenith="#a0583e", mid="#e08850", horizon="#ffb06a", cloud="#f0a070", amb_sky="#c09070",
           amb_ground="#a0603a")
SMOKE = dict(zenith="#3a1a12", mid="#6e2e1a", horizon="#9a4424", cloud="#4a2a20", amb_sky="#5a3020",
             amb_ground="#40200f")
COLD = dict(zenith="#b4c4d6", mid="#d0dae4", horizon="#d6dee8", cloud="#e8ecf0", amb_sky="#c8d2de",
            amb_ground="#a8b0b8", key="#fff4e6")


class Lakeside(kit.Scene):
    palette = "sunset"
    camera_at = kit.framing(height=16.0, target=(0.0, 5.0, -150.0))
    sun_az = 14.0
    sound_bed = "boardwalk"
    shadow_center = (0.0, 0.0, -60.0)
    shadow_size = 130.0
    landmarks = [(LIGHTHOUSE[0], 28.0, LIGHTHOUSE[2])]

    def build(self, rng) -> None:
        s = self.static
        pal = kit.palette(self.palette)
        # Ground and the promenade (the leading line) down to the pier.
        kit.ground(s, "#6e9a48", z_far=SHORE_Z)
        s.add("box", props.block(-4, 4, 0, 0.1, SHORE_Z, 40), rgb("#b7a48c"), 0.0, STRIPES, p1=70, p2=2)
        s.add("box", props.block(-900, 900, -9, -6, -3000, SHORE_Z), rgb("#5a4a3a"))  # lake bed
        kit.railing(s, SHORE_Z + 0.5, half=40.0)
        s.add("box", props.block(-2, 2, 0.6, 0.9, LIGHTHOUSE[2] - 4, SHORE_Z), rgb("#9b8670"), 0.0, STRIPES, p1=40, p2=2)
        # The landmark: a striped lighthouse at the end of the pier.
        x, _, z = LIGHTHOUSE
        s.add("cyl", chain(tr((x, 0.9, z)), sc((4.4, 24.0, 4.4))), rgb("#d8463c"), 0.0, STRIPES, p1=6, p2=1)
        s.add("cyl", chain(tr((x, 24.9, z)), sc((5.4, 0.8, 5.4))), props.DARK)
        s.add("ico", chain(tr((x, 27.2, z)), sc(2.0)), rgb("#fff1c9"), 9.0, GLOW, p2=1.0)
        s.add("cone", chain(tr((x, 28.8, z)), sc((5.0, 3.2, 5.0))), props.DARK)
        # Far layers: hills and a town across the water, already hazed toward the horizon.
        kit.far_hills(s, rng, pal["hills"])
        kit.far_city(s, rng, pal["city"])
        # Props along the path: lamps, benches, string lights, trees, the sign.
        self.lamps = []
        for z in np.arange(30.0, SHORE_Z, -14.0):
            for side in (-1, 1):
                self.lamps.append(props.lamp_post(s, side * 4.6, z + (7 if side > 0 else 0)))
            props.bench(s, -5.6, z - 3, np.pi / 2, color=rng.choice(["#c4553c", "#3d7bd9", "#e8a23a"]))
        for z in np.arange(30.0, SHORE_Z + 10, -10.0):
            props.string_lights(s, (-4.4, 5.0, z), (4.4, 5.0, z - 10), sag=0.8, n=16, rng=rng)
        self.trees = []
        while len(self.trees) < 40 * 7:  # 7 leaf balls per tree
            side = rng.choice([-1, 1])
            self.trees += props.round_tree(s, side * rng.uniform(9, 60), rng.uniform(SHORE_Z + 6, 30),
                                           rng.uniform(6, 10), rng=rng)
        self.sign_point = props.sign_post(s, 7.0, 12.0, h=4.4, w=3.0, panel_h=1.5)
        # People: strolling the promenade and the lawns; each leaves at their own temperature.
        self.crowd = props.Crowd(rng, [(120, (-3.2, 3.2), (SHORE_Z + 4, 34.0), 1), (40, (-30.0, 30.0), (-60.0, 20.0), 0)])
        heat = self.timeline.scenario.heat
        if heat is not None and heat.rate_f_per_s:
            limit = rng.uniform(92, 150, self.crowd.n) if heat.rate_f_per_s > 0 else rng.uniform(10, -30, self.crowd.n)
            self.crowd.p["leave"] = np.array([self.timeline.time_at_temp(f) for f in limit])
            self.crowd.p["exit"] = np.stack([np.sign(self.crowd.p["x"] + 1e-3) * 70, np.zeros(self.crowd.n),
                                             self.crowd.p["z"]], 1)
        # What burns, and at what °F: litter (451), benches (572), trees (572+).
        pts = [(rng.uniform(-3.5, 3.5), 0.1, rng.uniform(SHORE_Z, 36), rng.uniform(0.4, 0.8), rng.uniform(451, 470))
               for _ in range(50)]
        pts += [(c[0], c[1] - r, c[2], r * 2.0, rng.uniform(572, 595)) for c, r in self.trees[::7]]
        a = np.array(pts)
        self.fires = dict(x=a[:, 0], y=a[:, 1], z=a[:, 2], size=a[:, 3], temp=a[:, 4], seed=rng.uniform(0, 100, len(a)))

    def power(self, fx) -> float:
        return 1.0 - ramp(fx.temp_f, 255.0, 290.0) - ramp(fx.temp_f, -60.0, -110.0)

    def sky(self, t, fx) -> dict:
        temp = fx.temp_f
        return kit.blend(kit.palette(self.palette), (HOT, ramp(temp, 150.0, 420.0)), (SMOKE, ramp(temp, 540.0, 640.0)),
                         (COLD, ramp(temp, 40.0, -30.0)))

    def effects(self, t, fx) -> dict:
        temp = fx.temp_f
        return dict(dry=ramp(temp, 110.0, 260.0), boil=ramp(temp, 205.0, 225.0) * (1 - ramp(temp, 520.0, 620.0)),
                    scorch=0.92 * ramp(temp, 575.0, 645.0), frost=ramp(temp, 32.0, -60.0), ice=ramp(temp, 32.0, -40.0))

    def dynamic(self, t, fx) -> Instances:
        d = Instances()
        temp = fx.temp_f
        level = -1.0 - 5.0 * ramp(temp, 212.0, 640.0)  # the lake boils away
        kit.water(d, level)
        self.crowd.draw(d, t)
        green = kit.mix(rgb("#3f8a3a"), rgb("#8a5a24"), ramp(temp, 220.0, 380.0))
        props.draw_leaves(d, self.trees, kit.mix(green, rgb("#e8eef2"), ramp(temp, 32.0, -20.0)),
                          scale=1 - ramp(temp, 572.0, 586.0))
        steam = ramp(temp, 205.0, 235.0) * (1 - ramp(temp, 470.0, 600.0))
        if steam > 0:
            xs = np.linspace(-120, 120, 40)
            props.draw_puffs(d, xs, np.full_like(xs, SHORE_Z - 30.0), level + 0.5, t, steam, rgb("#f2eef0"),
                             rise=4.0, spread=8.0, size=4.5, n_max=260)
        f = self.fires
        on = temp >= f["temp"]
        if on.any():
            grow = np.clip((temp - f["temp"][on]) / 25.0, 0.15, 1.0)
            props.draw_flames(d, f["x"][on], f["y"][on], f["z"][on], f["size"][on] * grow, f["seed"][on], t)
        return d


Scene = Lakeside
