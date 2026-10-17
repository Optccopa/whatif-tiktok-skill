"""Scene kit: the shared pieces every scene needs, so a new scene is mostly its layout.

    from whatif import kit, props
    class Beach(kit.Scene):
        palette = "sunset"            # sky, light, haze and far-layer colors (see PALETTES)
        camera_at = kit.framing(height=16, target=(0, 4, -150))
        def build(self, rng): ...     # add static instances to self.static
        def dynamic(self, t, fx): ... # per-frame instances
    Scene = Beach                     # video.py finds the class through this name

Registration is automatic: `scene: beach` loads `whatif/beach.py`. Street scenes (temperature) get their sky
and light from `look()`, which this base implements from the palette. Space scenes set `space_look = "forest"`
or `"harbor"` to use those looks (every body kind, strikes, escapes) instead.
"""

from __future__ import annotations

import math

import numpy as np

from whatif import props
from whatif.gl import Frame, look_at, ortho
from whatif.meshes import BUILDING, GROUND, Instances, chain, rgb, ry, sc, tr

W, H = 1080, 1920
# Overlay text zones in output pixels (x0, y0, x1, y1): keep landmarks, the body and the sign out of them.
TEXT_ZONES = {"hook": (90, 220, 990, 500), "readout": (80, 850, 620, 1060), "subtitles": (110, 1140, 930, 1300),
              "buttons": (930, 700, 1080, 1700)}


def direction(az: float, el: float) -> np.ndarray:
    """Unit vector for azimuth/elevation in degrees; az 0 is straight ahead (-z), positive to the right."""
    a, e = math.radians(az), math.radians(el)
    return np.array([math.sin(a) * math.cos(e), math.sin(e), -math.cos(a) * math.cos(e)])


def mix(a, b, k: float) -> tuple:
    return tuple(x * (1 - k) + y * k for x, y in zip(a, b))


def framing(height: float = 16.0, target=(0.0, 4.0, -150.0), back: float = 44.0) -> tuple:
    """The house camera: `back` m behind the origin, `height` m up, looking at `target`. Returns (pos, look)."""
    return np.array([0.0, height, back]), np.array(target, dtype=float)


# Time-of-day palettes (hex). zenith/mid/horizon: the sky; cloud; amb_sky/amb_ground: fill light from above and
# below; key: the sun's color; sun_el: its height (deg); hills/city: far-layer colors already hazed toward the sky.
PALETTES = {
    "golden": dict(zenith="#6f8cc8", mid="#d8b8b8", horizon="#ffcf98", cloud="#f2d8d0", amb_sky="#aab4d0",
                   amb_ground="#8a7a68", key="#ffcf98", sun_el=7.0,
                   hills=["#8a90b0", "#7f88a8", "#949ab8"], city=["#c8b8b0", "#b8aca8", "#d0c0b4"]),
    "sunset": dict(zenith="#5a5cae", mid="#b08cc0", horizon="#f6c8a0", cloud="#efb0bc", amb_sky="#a29ac8",
                   amb_ground="#8a6a50", key="#ffc890", sun_el=6.5,
                   hills=["#8a76a6", "#7f6c9c", "#9580ad"], city=["#b9a8b8", "#a798b0", "#c4b0b4"]),
    "dusk": dict(zenith="#1c2a5a", mid="#5a4a8a", horizon="#e89a8a", cloud="#a07090", amb_sky="#5a5a8a",
                 amb_ground="#3a3040", key="#ff9a70", sun_el=1.5,
                 hills=["#4a3e66", "#40385c", "#54476e"], city=["#6a5a78", "#5e5270", "#746480"]),
    "day": dict(zenith="#4a7ac8", mid="#9cc0e8", horizon="#dfe9f2", cloud="#ffffff", amb_sky="#b0c4dc",
                amb_ground="#8a8478", key="#fff2dc", sun_el=35.0,
                hills=["#8ea4c0", "#98acc6", "#86a0bc"], city=["#c4ccd6", "#b8c2ce", "#d0d6de"]),
}


def palette(name_or_dict, **overrides) -> dict:
    p = dict(PALETTES[name_or_dict]) if isinstance(name_or_dict, str) else dict(name_or_dict)
    p.update(overrides)
    return p


def blend(base: dict, *shifts) -> dict:
    """Blend palette colors toward others: blend(p, (hot, 0.4), (smoke, 0.8)). Non-color keys come from base."""
    out = dict(base)
    for other, k in shifts:
        if k <= 0:
            continue
        for key, v in other.items():
            if isinstance(v, str) and v.startswith("#") and key in out:
                out[key] = mix(_lin(out[key]), _lin(v), min(k, 1.0))
    return out


def _lin(c, scale=1.0):
    return rgb(c, scale) if isinstance(c, str) else tuple(x * scale for x in c)


# Layers ------------------------------------------------------------------------------


def ground(s: Instances, color: str, z_far: float = -400.0, z_near: float = 400.0, half: float = 900.0,
           material=GROUND) -> None:
    """A huge ground slab (top at y=0): big enough that no edge ever shows."""
    s.add("box", props.block(-half, half, -8, 0, z_far, z_near), rgb(color), 0.0, material)


def far_hills(s: Instances, rng, colors, n: int = 22, z=(-2600.0, -1800.0), height=(120.0, 260.0)) -> None:
    """Soft low-poly hills that close the horizon; taller toward the sides."""
    for _ in range(n):
        x = rng.uniform(-3500, 3500)
        h = rng.uniform(*height) * (1.0 if abs(x) < 900 else 1.6)
        r = rng.uniform(700, 1400)
        s.add("cone6", chain(tr((x, -10, rng.uniform(*z))), ry(rng.uniform(0, 1)), sc((r, h, r * 0.7))),
              rgb(rng.choice(colors)))


def far_city(s: Instances, rng, colors, n: int = 120, z=(-1250.0, -1050.0), x=(-1400.0, 1400.0),
             lit: float = 0.25) -> None:
    """A low skyline of boxes with windows (`lit` = fraction of windows lit)."""
    for _ in range(n):
        cx = rng.uniform(*x)
        cz = rng.uniform(*z)
        w, h = rng.uniform(20, 60), rng.uniform(10, 45) * (1.6 if rng.random() < 0.12 else 1)
        s.add("box", props.block(cx - w / 2, cx + w / 2, -2, h, cz - 15, cz + 15), rgb(rng.choice(colors)), 0.0,
              BUILDING, seed=rng.uniform(0, 100), p1=lit)


def water(d: Instances, level: float = -1.0, color: str = "#3f6c96", z_center: float = -1600.0) -> None:
    """A huge water plane (WATER material) at `level`; draw it in dynamic() if the level changes."""
    d.add("quad", chain(tr((0, level, z_center)), sc((3600, 1, 2800))), rgb(color), 0.0, 1)


def railing(s: Instances, z: float, half: float = 30.0, y: float = 0.0, color=props.WHITE) -> None:
    """A thin rail with posts every 2 m, across x at depth z."""
    s.add("box", props.block(-half, half, y + 1.0, y + 1.1, z - 0.05, z + 0.05), color)
    xs = np.arange(-half, half + 0.1, 2.0)
    s.add("box", chain(tr(np.stack([xs, np.full_like(xs, y), np.full_like(xs, z)], 1)), sc((0.08, 1.1, 0.08))), color)


# The base scene ------------------------------------------------------------------------


class Scene:
    palette = "sunset"
    camera_at = framing()
    fov = 60.0
    sun_az = 16.0  # where the sun sits (deg); keep it in open sky over the landmark, out of the text zones
    body_az_el = (0.0, 10.0)  # space scenes: where an approaching body sits
    moon_az_el = (-40.0, 25.0)
    space_look = None  # "forest" or "harbor" for space stories; None = this class's look() (street scenes)
    sound_bed = "boardwalk"  # an existing ambience to reuse: harbor, forest, boardwalk, plaza
    shadow_center = (0.0, 0.0, -60.0)
    shadow_size = 150.0  # half-width (m) of the shadow box: just cover the playfield
    landmarks: list = []  # world points that must stay visible and clear of the text (checked by `stills`)

    def __init__(self, timeline, seed: int = 0):
        self.timeline = timeline
        self.static = Instances()
        self.build(np.random.default_rng(seed + 21))

    # Override these -------------------------------------------------------------------
    def build(self, rng) -> None:
        """Add the static scene to self.static."""

    def dynamic(self, t: float, fx) -> Instances:
        return Instances()

    def power(self, fx) -> float:
        """1 = lights on. Key it to the story, e.g. 1 - ramp(fx.temp_f, 255, 290)."""
        return fx.power if self.space_look else 1.0

    def sky(self, t: float, fx) -> dict:
        """The palette for this moment: blend the base toward others as the story turns (see boardwalk.py)."""
        return palette(self.palette)

    def effects(self, t: float, fx) -> dict:
        """Global shader effects: dry, scorch, boil, frost, ice (0..1), keyed to the story's numbers."""
        return {}

    # Provided --------------------------------------------------------------------------
    def camera(self, t: float, fx):
        pos, look = self.camera_at
        return pos.copy(), look.copy(), self.fov

    def lights(self, t: float, fx, cam, power: float) -> list:
        return []

    def dust_front(self, t: float):
        """Strike stories: z of the advancing dust wall, or None (see forest.py for a real one)."""
        return None

    def sign_text(self, fx) -> str:
        v = round(fx.temp_f)
        return f"{'−' if v < 0 else ''}{abs(v):,}°F"

    def look(self, t: float, fx, pos, look_pt, fov) -> Frame:
        p = self.sky(t, fx)
        sun = direction(self.sun_az, p["sun_el"])
        c = {k: _lin(v) for k, v in p.items() if k in ("zenith", "mid", "horizon", "cloud", "key")}
        frame = Frame(
            cam_pos=pos, cam_look=look_pt, fov=fov, time=t,
            zenith=c["zenith"], mid=c["mid"], horizon=c["horizon"],
            stars=p.get("stars", 0.0), moon_dir=direction(*self.moon_az_el), moon_r=0.0, moon_col=(0, 0, 0),
            sun_dir=sun, body_dir=sun, body_d=1e6, body_frame=np.eye(3), flattening=0.0, rings=(0.0, 0.0),
            bands=((1, 1, 1),) * 3, ring_col=(1, 1, 1), body_bright=0.0, body_light=(0.0, 0.0, 0.0),
            glow_col=(0, 0, 0), glow=0.0, amb_sky=_lin(p["amb_sky"], 0.62), amb_ground=_lin(p["amb_ground"], 0.5),
            fog_col=c["horizon"], fog_density=p.get("fog", 0.0007), power=self.power(fx), lava=0.0,
            exposure=p.get("exposure", 1.05), flash=0.0, flash_col=(1, 1, 1), fade=fx.fade,
        )
        frame.body_type = -1
        frame.moon_vis = 0.0
        frame.key_dir = sun
        frame.key_col = tuple(x * p.get("key_power", 2.3) for x in c["key"])
        frame.lights = self.lights(t, fx, pos, frame.power)
        frame.cloud = tuple(x * 0.95 for x in c["cloud"])
        frame.cloud_amt = p.get("cloud_amt", 1.0)
        frame.sun_show, frame.sun_r, frame.sun_b, frame.sun_glow = 1.0, 0.012, 7.0, 0.9
        for k, v in self.effects(t, fx).items():
            setattr(frame, k, v)
        center = np.array(self.shadow_center)
        frame.shadow_vp = ortho(self.shadow_size, 10.0, 9000.0) @ look_at(center + sun * 4000.0, center)
        return frame


def framing_warnings(scene, frame, w: int = W, h: int = H) -> list[str]:
    """Where the landmarks, sign and sun/body land on screen, and which text zone or edge they collide with."""
    from whatif.video import project

    pos = np.asarray(frame.cam_pos, float)
    points = [(f"landmark {i}", np.asarray(p, float)) for i, p in enumerate(getattr(scene, "landmarks", []))]
    if getattr(scene, "sign_point", None) is not None:
        points.append(("sign", np.asarray(scene.sign_point, float)))
    if frame.body_type != -1 or getattr(scene, "space_look", None):
        points.append(("body", pos + np.asarray(frame.body_dir, float) * 5000.0))
    else:
        points.append(("sun", pos + np.asarray(frame.sun_dir, float) * 5000.0))
    out = []
    for name, p in points:
        at = project(frame, p, w, h)
        if at is None or not (0 <= at[0] <= w and 0 <= at[1] <= h):
            out.append(f"{name} is off screen")
            continue
        for zone, (x0, y0, x1, y1) in TEXT_ZONES.items():
            if x0 <= at[0] <= x1 and y0 <= at[1] <= y1:
                out.append(f"{name} at ({at[0]:.0f}, {at[1]:.0f}) px sits under the {zone}")
    return out
