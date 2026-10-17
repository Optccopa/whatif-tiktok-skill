"""Renders a what-if scenario to MP4: 3D frames from the GPU (or Mesa), text on top, into ffmpeg."""

from __future__ import annotations

import logging
import math
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from whatif.encode import H, W, bitrate_cap_kbps, ffmpeg_command, pick_music, shrink_to_fit
from whatif.gl import Frame, Renderer, look_at, ortho
from whatif.harbor import MOON_DIR, SUN_DIR, Harbor
from whatif.meshes import rgb
from whatif.audio import soundtrack, write_wav
from whatif.overlay import Overlay
from whatif.scenario import PULSE_T0, Effects, Scenario, Timeline, ramp

log = logging.getLogger(__name__)

MOON_RADIUS = math.radians(0.26)
X264_PARAMS = "aq-mode=3:aq-strength=0.9:deblock=1,1"  # keeps dark gradients from banding


def _mix(a, b, k: float):
    return tuple(x * (1 - k) + y * k for x, y in zip(a, b))


def _slerp(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    dot = float(np.clip(a @ b, -1, 1))
    omega = math.acos(dot)
    if omega < 1e-6:
        return a
    out = (math.sin((1 - k) * omega) * a + math.sin(k * omega) * b) / math.sin(omega)
    return out / np.linalg.norm(out)


def body_frame(direction: np.ndarray, opening_deg: float, roll_deg: float = -16.0) -> np.ndarray:
    """Columns: ring-plane x, pole, ring-plane z. Negative opening = rings seen from below."""
    s = direction
    right = np.cross(s, (0.0, 1.0, 0.0))
    right /= np.linalg.norm(right)
    up = np.cross(right, s)
    o = math.radians(opening_deg)
    pole = up * math.cos(o) - s * math.sin(o)
    roll = math.radians(roll_deg)  # turn the pole within the sky plane (Rodrigues about s)
    pole = pole * math.cos(roll) + np.cross(s, pole) * math.sin(roll) + s * (s @ pole) * (1 - math.cos(roll))
    pole /= np.linalg.norm(pole)
    x = np.cross(pole, s)
    x /= np.linalg.norm(x)
    z = np.cross(x, pole)
    return np.stack([x, pole, z], 1)


def _direction(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.sin(az) * math.cos(el), math.sin(el), -math.cos(az) * math.cos(el)])


PULSAR_AZ_EL = (2.0, 15.0)

RING_STYLES = {"saturn": 0, "uranus": 1, "neptune": 2}


def _body_frame(body, body_dir: np.ndarray, animated_open: float) -> np.ndarray:
    if body.ring_open_deg is not None:
        return body_frame(body_dir, body.ring_open_deg, body.ring_roll_deg)
    return body_frame(body_dir, animated_open, body.ring_roll_deg)


def star_light(body, distance_mi: float) -> float:
    """0 (a point among the stars) .. 1 (as bright as our Sun), on a log scale."""
    flux = body.luminosity_suns / max(distance_mi / 93_000_000.0, 1e-9) ** 2
    return min(max(math.log10(max(flux, 1e-12) * 1e6) / 6.0, 0.0), 1.0)


def pulsar_beam(body_dir: np.ndarray, t: float, spin_hz: float, tilt_deg: float = 25.0,
                roll_deg: float = -58.0) -> tuple[np.ndarray, float]:
    """One beam's direction at t, and how squarely it points at us (1 = a full lighthouse flash).

    The spin axis leans `tilt_deg` toward us from the sky's "up" (turned by `roll_deg`), and the beam
    sits at 90° - tilt from the axis, so once per turn it sweeps straight across Earth.
    """
    s = body_dir / np.linalg.norm(body_dir)
    us = -s
    up = np.array([0.0, 1.0, 0.0]) - s * s[1]
    up /= np.linalg.norm(up)
    r = math.radians(roll_deg)
    up = up * math.cos(r) + np.cross(s, up) * math.sin(r)
    g = math.radians(tilt_deg)
    axis = up * math.cos(g) + us * math.sin(g)
    alpha = math.pi / 2 - g
    e1 = us - axis * (axis @ us)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    phi = 2 * math.pi * spin_hz * (t - PULSE_T0)
    beam = axis * math.cos(alpha) + math.sin(alpha) * (e1 * math.cos(phi) + e2 * math.sin(phi))
    off = math.acos(min(max(float(beam @ us), -1.0), 1.0))
    return beam, math.exp(-(off / 0.2) ** 2)


def harbor_look(scene: "WhatIfScene", t: float, fx: Effects, body_dir: np.ndarray, pos, look, fov) -> Frame:
    """Dusk over the bay, warming as the body nears, burning at the end."""
    body = scene.scenario.body
    star = body.kind == "star"
    pulsar = body.kind == "pulsar"
    warm = star_light(body, scene.timeline.center(t)) ** 2 if star else 0.0 if pulsar else fx.body_light
    hot = max(fx.fire, fx.lava * 0.7)
    # Dusk: deep blue overhead, the pink band of twilight low over the sea.
    zenith = _mix(_mix(rgb("#111842"), rgb("#2a2238"), warm * 0.5), rgb("#2a0d08"), hot * 0.6)
    mid = _mix(_mix(rgb("#3a4382"), rgb("#6b5066"), warm * 0.55), rgb("#6e2410"), hot * 0.6)
    horizon = _mix(_mix(rgb("#b88aa0"), rgb("#c99670"), warm * 0.6), rgb("#d0582a"), hot * 0.65)
    amb_sky = _mix(_mix(rgb("#5a69a8", 0.8), rgb("#b89070", 0.7), warm * 0.7), rgb("#c0603a", 0.6), hot * 0.5)
    amb_ground = _mix(rgb("#3a3440", 0.7), rgb("#a04a20", 0.8), hot * 0.6)
    moon_dir = _slerp(MOON_DIR, body_dir, fx.moon_pull)
    power = fx.power
    if power < 1.0:
        power *= 0.75 + 0.25 * math.sin(t * 37.0) * math.sin(t * 11.3)
    frame = Frame(
        cam_pos=pos, cam_look=look, fov=fov, time=t,
        zenith=zenith, mid=mid, horizon=horizon, stars=1.0 - 0.95 * min(warm * 1.4 + hot, 1.0),
        moon_dir=moon_dir, moon_r=MOON_RADIUS, moon_col=_mix(rgb("#9fb0ff", 0.45), (0, 0, 0), warm * 0.5),
        sun_dir=SUN_DIR, body_dir=body_dir, body_d=fx.distance_radii,
        body_frame=_body_frame(body, body_dir, fx.ring_open), flattening=body.flattening,
        rings=body.rings or (0.0, 0.0), bands=scene.bands, ring_col=scene.ring_col,
        body_bright=0.95, body_light=rgb(body.light_color, (2.2 if star else 0.9) * warm), glow_col=rgb("#ffcf95"),
        glow=fx.glow, amb_sky=amb_sky, amb_ground=amb_ground, fog_col=horizon,
        fog_density=0.00013, power=power, lava=fx.lava, exposure=1.35,
        flash=fx.flash, flash_col=(1.0, 0.95, 0.86), fade=fx.fade,
    )
    frame.lights = scene.scene.lights(t, fx, pos, power)
    frame.cloud = _mix(_mix(rgb("#8a7fa8", 0.8), rgb("#d6a77c", 0.9), warm * 0.8), rgb("#7a3a22"), hot * 0.6)
    frame.ring_style = RING_STYLES.get(body.ring_style, 0)
    if body.is_moon:  # it's our own Moon falling: there is no other one in the sky
        frame.moon_vis = 0.0
    if star:  # a red second sun: the night turns into a red day
        tf = fx.temp_f
        frame.dry = ramp(tf, 110.0, 260.0)  # its heat: plants dry, the bay boils, everything chars
        frame.boil = ramp(tf, 205.0, 230.0) * (1 - ramp(tf, 900.0, 1400.0))
        frame.scorch = 0.9 * ramp(tf, 575.0, 760.0)
        frame.body_type = 3
        frame.star_col = rgb(body.star_color)
        frame.star_bright = 1.5 + 2.5 * warm
        frame.glow = warm
        red = warm
        frame.zenith = _mix(frame.zenith, rgb("#5a2430"), red * 0.8)
        frame.mid = _mix(frame.mid, rgb("#b04a3a"), red * 0.8)
        frame.horizon = _mix(frame.horizon, rgb("#f08a5a"), red * 0.8)
        frame.fog_col = frame.horizon
        frame.amb_sky = _mix(frame.amb_sky, rgb("#e07a5a", 0.7), red * 0.8)
        frame.cloud = _mix(frame.cloud, rgb("#e08a6a", 0.9), red * 0.8)
        frame.stars = 1.0 - red
    if pulsar:  # a blue-white point and its beams; each sweep across us flashes over the bay
        b = fx.body_light
        beam, pulse = pulsar_beam(body_dir, t, body.spin_hz)
        cool = 0.45 * b * b
        frame.body_type = 4
        frame.star_col = rgb(body.star_color)
        frame.star_bright = 1.6 + 4.0 * b
        frame.glow = max(fx.glow, 0.6 * b * b)
        frame.zenith = _mix(frame.zenith, rgb("#1c2456"), cool)
        frame.mid = _mix(frame.mid, rgb("#4a5aa0"), cool)
        frame.horizon = _mix(frame.horizon, rgb("#7884bc"), cool * (1 - hot))
        frame.fog_col = frame.horizon
        frame.amb_sky = _mix(frame.amb_sky, rgb("#8aa0e0", 0.8), cool)
        frame.body_light = rgb(body.light_color, (0.3 + 1.5 * pulse) * b * b)
        frame.flash = max(frame.flash, 0.04 * pulse * b * b)
        frame.flash_col = (0.82, 0.9, 1.0) if fx.flash < 0.01 else frame.flash_col
        frame.sky_extra = dict(
            u_beam_dir=beam, u_beam_col=rgb(body.beam_color), u_beam_len=0.025 + 0.75 * b,
            u_beam_bright=0.35 + 1.4 * b, u_pulse=pulse * (0.6 + 1.4 * b),
        )
    return frame


def comet_look(scene: "WhatIfScene", t: float, fx: Effects, body_dir: np.ndarray, pos, look, fov) -> Frame:
    """Golden hour holds until it hits; then a fireball beyond the peaks, a burning sky, and the dust."""
    from whatif.forest import SNOW_LINE

    tl = scene.timeline
    st = scene.scenario.strike
    tc, tb = tl.tc, tl.t_blast
    heat = ramp(t, tc + 7.0, tb)
    dust = ramp(t, tb - 0.2, tb + 2.2)
    sun_el = 7.0
    sun = np.array([0.34 * math.cos(math.radians(sun_el)), math.sin(math.radians(sun_el)),
                    0.94 * math.cos(math.radians(sun_el))])
    sun /= np.linalg.norm(sun)

    def phase(gold: str, hot: str, dusty: str, scale: float = 1.0):
        return _mix(_mix(rgb(gold, scale), rgb(hot, scale), heat), rgb(dusty, scale), dust)

    site = _direction(0.0, 3.0)  # the fireball's center, beyond the peaks
    after = t >= tc
    horizon = phase("#e9b89c", "#e0702e", "#5a4836")
    altitude = max(fx.gap_mi, 1e-3)
    frame = Frame(
        cam_pos=pos, cam_look=look, fov=fov, time=t,
        zenith=phase("#3d5aa0", "#3a1a14", "#241c16"), mid=phase("#8a9ccc", "#93421e", "#3e3226"),
        horizon=horizon, stars=0.0,
        moon_dir=_direction(*scene.scene.moon_az_el), moon_r=MOON_RADIUS, moon_col=(0, 0, 0),
        sun_dir=sun, body_dir=site if after else body_dir, body_d=fx.distance_radii,
        body_frame=body_frame(body_dir, 0.0), flattening=0.0, rings=(0.0, 0.0),
        bands=scene.bands, ring_col=scene.ring_col, body_bright=1.0,
        body_light=rgb("#ff9a50", 2.4 * fx.glow) if after else rgb("#dfe8ff", 1.6 * fx.body_light),
        glow_col=rgb("#ffb070"), glow=0.0,
        amb_sky=phase("#8a9cc8", "#b0603a", "#4a3c30", 0.6), amb_ground=phase("#5a4632", "#7a3a20", "#2a2018", 0.45),
        fog_col=horizon, fog_density=0.00006 + 0.006 * dust, power=fx.power, lava=0.0, exposure=1.25,
        flash=fx.flash, flash_col=(1.0, 0.97, 0.9), fade=fx.fade,
    )
    sun_k = 1.0 - 0.75 * ramp(t, tc + 6.0, tb) - 0.25 * dust
    frame.cloud = phase("#f2c4a2", "#c05a2a", "#4a3a2c", 0.9)
    frame.key_dir = sun
    frame.key_col = rgb("#ffb06a", 2.1 * sun_k)
    frame.lights = []
    frame.dust_front = scene.scene.dust_front(t)
    frame.dust_h = 650.0
    frame.dust_col = rgb("#a7714a", 0.9)
    frame.body_type = 2
    frame.moon_vis = 0.0
    frame.snow_line = SNOW_LINE
    entry = ramp(t, tl.t_entry, tl.t_entry + 0.4) * (1 - ramp(t, tc - 0.05, tc + 0.05))
    el_now = tl.direction(t)[1]
    frame.sky_extra = dict(
        u_coma_r=min(math.atan(st.coma_mi / altitude), 0.5), u_comet_bright=0.0 if after else 0.55 + 0.6 * ramp(t, 0, tl.t_entry),
        u_tail_dir=_direction(-40.0, 55.0), u_tail_len=0.0 if scene.scenario.body.kind == "asteroid" else 0.4,
        u_tail_col=rgb("#fff0d6"),
        u_entry=entry, u_trail_dir=_direction(25.0, 60.0), u_trail_len=math.radians(18.0 - el_now) + 0.05,
        u_site_dir=site, u_fireball=fx.glow, u_fireball_r=0.03 + 0.2 * ramp(t, tc, tc + 5.0),
        u_plume=ramp(t, tc + 0.3, tc + 11.0), u_plume_lit=phase("#e8b08a", "#d07040", "#5a4836"),
    )
    if sun_k > 0.05:
        center = np.array([0.0, 80.0, -700.0])
        frame.shadow_vp = ortho(1500.0, 10.0, 9000.0) @ look_at(center + sun * 4000.0, center)
    return frame


def forest_look(scene: "WhatIfScene", t: float, fx: Effects, body_dir: np.ndarray, pos, look, fov) -> Frame:
    """Golden hour over the valley, sliding into night as the Sun is left behind."""
    from whatif.forest import SNOW_LINE

    if scene.timeline.strike:
        return comet_look(scene, t, fx, body_dir, pos, look, fov)
    if scene.timeline.escape:
        return escape_look(scene, t, fx, body_dir, pos, look, fov)

    dusk = ramp(t, 28.0, 42.0)
    night = ramp(t, 38.0, 50.0)
    sun_el = round((7.0 - 16.0 * ramp(t, 28.0, 46.0)) * 4) / 4  # quarter-degree steps let the shadow map be reused
    sun = np.array([0.34 * math.cos(math.radians(sun_el)), math.sin(math.radians(sun_el)),
                    0.94 * math.cos(math.radians(sun_el))])
    sun /= np.linalg.norm(sun)
    sun_k = min(max((sun_el + 1.0) / 5.0, 0.0), 1.0)

    def phase(gold: str, dim: str, dark: str, scale: float = 1.0):
        return _mix(_mix(rgb(gold, scale), rgb(dim, scale), dusk), rgb(dark, scale), night)

    planet = scene.scenario.body
    hole = planet.kind == "black_hole"
    # What the disk of hot gas adds: an orange light from the black hole once it's big.
    D = max(fx.distance_radii, 1.0)
    disk_size = 9.0 / D
    disk_light = min((disk_size / 0.6) ** 2, 1.0)
    moon_dir = _slerp(_direction(*scene.scene.moon_az_el), body_dir, fx.moon_pull)
    horizon = phase("#e9b89c", "#b9839a", "#1c1a30")
    frame = Frame(
        cam_pos=pos, cam_look=look, fov=fov, time=t,
        zenith=phase("#3d5aa0", "#1e2756", "#04060e"), mid=phase("#8a9ccc", "#4c5288", "#0b1024"),
        horizon=horizon, stars=night,
        moon_dir=moon_dir, moon_r=MOON_RADIUS, moon_col=(0, 0, 0),
        sun_dir=sun, body_dir=body_dir, body_d=fx.distance_radii,
        body_frame=_body_frame(planet, body_dir, 11.0), flattening=planet.flattening,
        rings=planet.rings or (0.0, 0.0), bands=scene.bands, ring_col=scene.ring_col, body_bright=1.0,
        body_light=rgb("#ffa860", 1.1 * disk_light) if hole else rgb(planet.light_color, 0.9 * fx.body_light),
        glow_col=rgb("#ff9a50") if hole else rgb("#cfefff"),
        glow=0.5 * disk_light if hole else fx.glow, amb_sky=phase("#8a9cc8", "#4a5080", "#121a33", 0.6),
        amb_ground=phase("#5a4632", "#3a3040", "#0a0a12", 0.45), fog_col=horizon,
        fog_density=0.00006, power=ramp(t, 32.0, 44.0), lava=0.0, exposure=1.25,
        flash=fx.flash, flash_col=(1.0, 0.95, 0.86), fade=fx.fade,
    )
    frame.cloud = phase("#f2c4a2", "#9a7d9a", "#1d1e30", 0.9)
    frame.key_dir = sun
    frame.key_col = rgb("#ffb06a", 2.1 * sun_k)
    frame.lights = []
    frame.body_type = 1 if hole else 0
    frame.ring_style = RING_STYLES.get(planet.ring_style, 0)
    frame.disk_bright = 1.5
    frame.bh_veil = 0.1 * (1 - night) + 0.02
    frame.moon_vis = 1.0 - ramp(fx.moon_pull, 0.7, 1.0)
    frame.snow_line = SNOW_LINE
    if sun_k > 0:
        center = np.array([0.0, 80.0, -700.0])
        view = look_at(center + sun * 4000.0, center)
        frame.shadow_vp = ortho(1500.0, 10.0, 9000.0) @ view
    return frame


def escape_look(scene: "WhatIfScene", t: float, fx: Effects, body_dir: np.ndarray, pos, look, fov) -> Frame:
    """A clear day that never comes back: the Sun shrinks to a star, frost, then the galaxy from outside."""
    from whatif.forest import SNOW_LINE

    tl = scene.timeline
    day = smooth_(fx.daylight)
    sun = _direction(0.0, 9.0)

    def sky(d: str, dim: str, dark: str, scale: float = 1.0):
        return _mix(rgb(dark, scale), _mix(rgb(dim, scale), rgb(d, scale), smooth_(max(day * 1.6 - 0.6, 0))),
                    smooth_(min(day * 1.6, 1.0)))

    horizon = sky("#d6e2ec", "#7d7f9e", "#06080f")
    galaxy = ramp(t, tl.time_at_years(150_000.0), tl.time_at_years(2_000_000.0))
    frame = Frame(
        cam_pos=pos, cam_look=look, fov=fov, time=t,
        zenith=sky("#3a69c2", "#1a2550", "#010205"), mid=sky("#84ace0", "#45558a", "#03050b"), horizon=horizon,
        stars=1.0 - day, moon_dir=_direction(*scene.scene.moon_az_el), moon_r=MOON_RADIUS, moon_col=(0, 0, 0),
        sun_dir=sun, body_dir=sun, body_d=1e6, body_frame=body_frame(sun, 0.0), flattening=0.0, rings=(0.0, 0.0),
        bands=scene.bands, ring_col=scene.ring_col, body_bright=0.0, body_light=(0.0, 0.0, 0.0),
        glow_col=(0.0, 0.0, 0.0), glow=0.0,
        amb_sky=_mix(rgb("#2b3552", 0.55), rgb("#8aa4d0", 0.7), day), amb_ground=_mix(rgb("#0e1018", 0.4), rgb("#4a4636", 0.45), day),
        fog_col=horizon, fog_density=0.00005, power=fx.power, lava=0.0, exposure=1.25 + 0.6 * (1 - day),
        flash=0.0, flash_col=(1, 1, 1), fade=fx.fade,
    )
    frame.cloud = sky("#f4f6f8", "#6a6c86", "#05070c", 0.95)
    frame.key_dir = sun
    frame.key_col = rgb("#fff1dc", 2.0 * day)
    frame.lights = []
    frame.body_type = -1
    frame.moon_vis = day ** 0.5
    frame.snow_line = SNOW_LINE - 620.0 * fx.frost
    frame.frost, frame.ice = fx.frost, fx.ice
    frame.sun_show, frame.sun_r = 1.0, fx.angular_radius
    frame.sun_b, frame.sun_glow = 3.0 + 40.0 * day, 0.6 * day
    frame.milky = (1.0 - day) * (1.0 - galaxy)
    frame.galaxy = galaxy
    frame.galaxy_r = 0.55 - 0.23 * ramp(t, tl.time_at_years(300_000.0), tl.T)
    frame.galaxy_dir = tuple(_direction(0.0, 15.0))
    frame.cloud_amt = day  # no sunlight, no weather: the clouds freeze out
    if day > 0.05:
        center = np.array([0.0, 80.0, -700.0])
        frame.shadow_vp = ortho(1500.0, 10.0, 9000.0) @ look_at(center + sun * 4000.0, center)
    return frame


def smooth_(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


LOOKS = {"harbor": harbor_look, "forest": forest_look}


def _scene_class(name: str):
    if name == "forest":
        from whatif.forest import Forest
        return Forest
    if name == "boardwalk":
        from whatif.boardwalk import Boardwalk
        return Boardwalk
    if name == "plaza":
        from whatif.plaza import Plaza
        return Plaza
    if name == "harbor":
        return Harbor
    import importlib  # any other scene: whatif/<name>.py, exporting its class as `Scene`
    return importlib.import_module(f"whatif.{name}").Scene


def sound_bed(scenario: Scenario) -> str:
    """The ambience: the built-in scenes have their own; a kit scene names the one it reuses."""
    if scenario.scene in ("harbor", "forest", "boardwalk", "plaza"):
        return scenario.scene
    return getattr(_scene_class(scenario.scene), "sound_bed", "forest")


def project(frame: Frame, point, w: int = W, h: int = H) -> tuple[float, float, float] | None:
    """Screen position (px) and depth of a world point for this frame's camera, or None if behind it."""
    from whatif.gl import perspective
    view = look_at(np.asarray(frame.cam_pos, float), np.asarray(frame.cam_look, float))
    clip = perspective(frame.fov, w / h, 0.3, 30000.0) @ view @ np.array([*point, 1.0])
    if clip[3] <= 0:
        return None
    ndc = clip[:3] / clip[3]
    return (ndc[0] * 0.5 + 0.5) * w, (0.5 - ndc[1] * 0.5) * h, clip[3]


class WhatIfScene:
    """Ties a scenario's timeline to its scene and the look of each moment."""

    def __init__(self, scenario: Scenario, seconds: float, seed: int = 0):
        self.scenario = scenario
        self.timeline = Timeline(scenario, seconds)
        self.scene = _scene_class(scenario.scene)(self.timeline, seed=seed)
        b = scenario.body
        self.bands = tuple(rgb(c) for c in b.bands)
        self.ring_col = rgb(b.ring_color)

    @property
    def meshes(self) -> dict:
        return getattr(self.scene, "meshes", {})

    def frame(self, t: float) -> tuple[Frame, dict, Effects]:
        fx = self.timeline.effects(t)
        pos, look, fov = self.scene.camera(t, fx)
        if self.timeline.flyby or self.timeline.strike:
            body_dir = _direction(*self.timeline.direction(t))
        elif self.scenario.body.kind == "pulsar":  # a point, not a disk: lift it clear of the peaks
            body_dir = _direction(*PULSAR_AZ_EL)
        else:
            az, el = getattr(self.scene, "body_az_el", (0.0, 10.0))
            if self.scenario.body.el_deg is not None:
                el = self.scenario.body.el_deg
            body_dir = _direction(az, el)
        space = getattr(self.scene, "space_look", None)
        if hasattr(self.scene, "look") and not space:  # street scenes carry their own sky and light
            frame = self.scene.look(t, fx, pos, look, fov)
            return frame, self.scene.dynamic(t, fx).arrays(), fx
        frame = LOOKS.get(space or self.scenario.scene, harbor_look)(self, t, fx, body_dir, pos, look, fov)
        if self.scenario.body.surface == "rocky":
            frame.surface = 1.0
        spot = self.scenario.body.spot
        if spot:  # facing us is longitude -90 in the body's frame
            frame.spot = (math.sin(math.radians(spot[0])), math.radians(spot[1] - 90.0), spot[2])
            frame.spot_col = rgb(self.scenario.body.spot_color)
        return frame, self.scene.dynamic(t, fx).arrays(), fx


# Render quality presets. "standard" was tuned on a CPU (software OpenGL); with a GPU, "high" costs little.
QUALITY = {
    "draft": dict(samples=0, scale=1, shadow=1024, grade=0.0),  # fast previews: check timing, not looks
    "standard": dict(samples=4, scale=1, shadow=2048, grade=0.0),
    "high": dict(samples=4, scale=2, shadow=4096, grade=1.0),  # 2x supersampling, sharper shadows, a light grade
}


def render_frames(scenario: Scenario, seconds: float, fps: int, seed: int = 0, only: list[int] | None = None,
                  quality: str = "standard"):
    """Yields (frame index, PIL image) for every frame (or just the ones in `only`)."""
    scene = WhatIfScene(scenario, seconds, seed)
    renderer = Renderer(W, H, **QUALITY[quality])
    overlay = Overlay(scenario, scene.timeline)
    try:
        for name, mesh in scene.meshes.items():
            renderer.add_mesh(name, mesh)
        renderer.set_static(scene.scene.static.arrays())
        total = int(round(seconds * fps))
        for i in (only if only is not None else range(total)):
            t = i / fps
            frame, dynamic, fx = scene.frame(t)
            img = Image.fromarray(renderer.render(frame, dynamic))
            extras = {}
            sign = getattr(scene.scene, "sign_point", None)
            if sign is not None:  # the temperature board in the scene, lettered in the overlay
                at = project(frame, sign)
                if at:
                    extras["sign"] = (at[0], at[1], at[2], scene.scene.sign_text(fx))
            if hasattr(scene.scene, "lens_frost"):
                extras["frost"] = scene.scene.lens_frost(fx)
            overlay.apply(img, t, **extras)
            yield i, img
    finally:
        renderer.release()


def render_whatif(
    scenario: Scenario,
    out_path: Path,
    fps: int = 30,
    seconds: float = 65.0,
    preset: str = "medium",
    crf: int = 18,
    music_dir: str = "",
    max_bytes: int | None = None,
    seed: int = 0,
    quality: str = "standard",
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".part.mp4")
    total = int(round(seconds * fps))
    duration = total / fps
    music = pick_music(music_dir, scenario.id)
    sound = None
    if not music:  # the synthesized ambience, unless real music was supplied
        sound = write_wav(tmp.with_suffix(".wav"), soundtrack(Timeline(scenario, seconds), sound_bed(scenario), duration, seed))
    audio_kbps = 128 if (music or sound) else 32
    max_kbps = bitrate_cap_kbps(max_bytes, duration, audio_kbps)
    cmd = ffmpeg_command(tmp, fps, duration, preset, crf, music, max_kbps, audio_kbps, x264_params=X264_PARAMS,
                         soundtrack=sound)

    began = time.monotonic()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i, img in render_frames(scenario, seconds, fps, seed, quality=quality):
            proc.stdin.write(img.tobytes())
            if i % (fps * 10) == 0:
                log.info("%s: %ds rendered", scenario.id, i // fps)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    stderr = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed: {stderr.strip()[-2000:]}")
    if sound:
        sound.unlink(missing_ok=True)
    if max_bytes and max_kbps:
        shrink_to_fit(tmp, max_bytes, max_kbps, preset, crf)
    tmp.replace(out_path)
    log.info("rendered %s (%d frames, %.1f MB) in %.0fs", out_path.name, total, out_path.stat().st_size / 1e6,
             time.monotonic() - began)
    return out_path


def add_soundtrack(video: Path, scenario: Scenario, seconds: float = 65.0, seed: int = 0) -> Path:
    """Replace the audio of an already-rendered video with the scenario's ambience (video copied as is)."""
    wav = write_wav(video.with_suffix(".ambience.wav"), soundtrack(Timeline(scenario, seconds), sound_bed(scenario),
                                                                   seconds, seed))
    out = video.with_name(video.stem + ".sound.mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-i", str(wav),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest",
                    "-movflags", "+faststart", str(out)], check=True)
    wav.unlink(missing_ok=True)
    out.replace(video)
    return video


def render_whatif_stills(scenario: Scenario, times: list[float], out_dir: Path, seconds: float = 65.0,
                         fps: int = 30, seed: int = 0, quality: str = "standard") -> list[Path]:
    """PNG frames at the given seconds, for checking the look without encoding a video."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {int(round(t * fps)): t for t in times}
    paths = []
    for i, img in render_frames(scenario, seconds, fps, seed, only=sorted(wanted), quality=quality):
        path = out_dir / f"{scenario.id}-{wanted[i]:05.1f}s.png"
        img.save(path)
        paths.append(path)
    return paths
