"""What-if scenarios: a body falls toward Earth, and everything on screen follows from physics.

The body's distance shrinks exponentially from its real distance to contact, so each
halving of the distance takes the same time. Story beats are pinned either to a time or
to a distance, and the scene's big moments (tides, losing the Moon, the rings, impact)
come from the body's mass and size, so adding another planet is a catalog entry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CATALOG_PATH = Path(__file__).with_name("scenarios.yaml")

EARTH_RADIUS_MI = 3_959.0
MOON_DISTANCE_MI = 238_855.0
MOON_MASS_EARTHS = 0.0123
SUN_DISTANCE_MI = 93_000_000.0
SUN_MASS_EARTHS = 333_000.0

LINE_MAX_SECONDS = 4.2
LINE_FADE_IN = 0.25
LINE_FADE_OUT = 0.3


def smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ramp(t: float, start: float, end: float) -> float:
    return smooth((t - start) / (end - start)) if end != start else float(t >= start)  # end < start: a falling ramp


@dataclass(frozen=True)
class Body:
    name: str
    radius_mi: float
    mass_earths: float
    start_mi: float  # center to center, at the start of the video
    flattening: float = 0.0
    rings: tuple[float, float] | None = None  # inner and outer ring radius, in body radii
    bands: tuple[str, ...] = ("#e8d5a9", "#c9a46c", "#9fa6ad")  # light band, dark band, poles
    ring_color: str = "#d9c7a3"
    kind: str = "planet"  # planet, star, pulsar, comet, or black_hole (radius_mi is then the event horizon)
    ring_style: str = "saturn"  # saturn (broad bright rings), uranus or neptune (thin dark ones)
    ring_open_deg: float | None = None  # fixed ring tilt toward us; None lets the scene animate it
    ring_roll_deg: float = -16.0
    light_color: str = "#ffd9a8"  # the light it throws on the scene
    luminosity_suns: float = 0.0  # stars: how bright, in Suns
    star_color: str = "#ff7a3c"
    spot: tuple[float, float, float] | None = None  # great storm: latitude (deg), longitude (deg), size
    spot_color: str = "#c2603f"
    is_moon: bool = False  # the body IS our Moon: hide the usual one, and tides read "× NORMAL"
    closest_mi: float = 0.0  # flybys: center distance at the closest point
    end_mi: float = 0.0  # flybys: center distance when the story ends
    surface: str = "bands"  # planets: bands (gas/ice giant) or rocky (mottled terrain, small polar caps)
    el_deg: float | None = None  # impact mode: raise or lower the body in the sky (scene default if None)
    spin_hz: float = 0.5  # pulsars: turns per second on screen (slowed down so the beams can be seen)
    spin_real_hz: float = 0.0  # pulsars: the real spin, for the "SPIN SHOWN N× SLOWER" note (0 = no note)
    beam_color: str = "#9cc4ff"  # pulsars: the color of the two beams


@dataclass(frozen=True)
class Strike:
    """A small body (comet, asteroid) that hits somewhere we can see from here."""

    at_s: float  # moment of impact
    entry_alt_mi: float = 62.0  # where the atmosphere starts (the Karman line)
    angle_deg: float = 45.0  # entry angle
    speed_mph: float = 100_000.0
    site_mi: float = 300.0  # how far away it hits
    blast_after_s: float = 15.0  # on screen, from impact to the air blast reaching us
    blast_real_s: float = 1100.0  # the same, in real seconds
    coma_mi: float = 10_000.0  # comets: radius of the glowing head


@dataclass(frozen=True)
class Escape:
    """Earth thrown out of the solar system and the galaxy: deep time on a log clock."""

    speed_mph: float = 2_200_000.0  # ~1,000 km/s: about twice the Milky Way's escape speed out here
    start_hours: float = 1.0  # elapsed time at the first frame
    end_years: float = 30_000_000.0  # elapsed time when the story ends
    # Surface temperature (°F) at elapsed years: interpolated in log time.
    temperatures: tuple = ((0.0, 59.0), (1 / 52, 0.0), (1.0, -100.0), (1_000.0, -250.0), (100_000.0, -350.0),
                           (10_000_000.0, -400.0))


LIGHT_YEAR_MI = 5.879e12

PULSE_T0 = 0.3  # pulsars: the first moment a beam sweeps across us; then once per turn


@dataclass(frozen=True)
class Heat:
    """The temperature itself is the story: it climbs (or falls) by a fixed amount every second."""

    start_f: float = 75.0  # a summer evening
    rate_f_per_s: float = 10.0  # negative for colder
    floor_f: float = -459.67  # absolute zero: it can't go lower
    sun_factor_per_s: float = 0.0  # if set (e.g. 0.99): Earth's distance to the Sun shrinks by this factor
    # every second, and the temperature follows (average surface temperature goes as distance^-1/2)
    # A log clock instead (the Sun goes out): elapsed time runs from start_hours to end_years, and the
    # temperature follows the anchors, as in escape mode. light_delay_min: when the sunlight stops arriving.
    start_hours: float = 0.0
    end_years: float = 0.0
    temperatures: tuple = ((0.0, 59.0), (1 / 52, 0.0), (1.0, -100.0), (1_000.0, -250.0), (100_000.0, -350.0),
                           (10_000_000.0, -400.0))
    light_delay_min: float = 0.0


ABSOLUTE_ZERO_F = -459.67


def format_temp(f: float) -> str:
    """'75°F', '−258°F' (a real minus sign, like a thermometer display)."""
    if f <= ABSOLUTE_ZERO_F + 0.005:
        return "\u2212459.67°F"
    v = round(f)
    sign = "\u2212" if v < 0 else ""
    return f"{sign}{abs(v):,}°F"


def pulsar_light(distance_mi: float) -> float:
    """0 (a faint dot, ~1.7 light-years out) .. 1 (a million miles away), smooth on a log scale."""
    return smooth((13.0 - math.log10(max(distance_mi, 1.0))) / 7.0)


def pulse_times(spin_hz: float, seconds: float) -> list[float]:
    """When a pulsar's beam points straight at us (the lighthouse flash)."""
    out, t = [], PULSE_T0
    while t < seconds:
        out.append(t)
        t += 1.0 / spin_hz
    return out
HOURS_PER_YEAR = 8_766.0


def format_miles(miles: float, light_years: bool = False) -> str:
    if light_years and miles >= 0.05 * LIGHT_YEAR_MI:
        ly = miles / LIGHT_YEAR_MI
        return f"{ly:,.2f} light-years" if ly < 10 else f"{ly:,.0f} light-years"
    if miles >= 1e12:
        return f"{miles / 1e12:.2f} trillion mi"
    if miles >= 1e9:
        return f"{miles / 1e9:.2f} billion mi"
    return f"{miles:,.0f} mi"


@dataclass(frozen=True)
class Beat:
    at: float  # seconds into the video
    line: str = ""  # subtitle
    label: str = ""  # HUD milestone under the distance


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    hook: str  # "What if {Saturn} crashed into Earth?" (braces mark the highlighted word)
    category: str
    scene: str
    body: Body
    hud_label: str
    impact_s: float
    beats: tuple[Beat, ...]
    final_line: str
    outro: str
    outro_small: str
    teaser: str
    question: str
    accent: str = "#f0c674"
    hashtags: tuple[str, ...] = field(default_factory=tuple)
    parts: int = 1
    mode: str = "impact"  # impact: it hits at impact_s. flyby: closest at closest_s, gone by impact_s.
    # strike: a small body hits at strike.at_s within sight; impact_s is when the story ends.
    closest_s: float = 0.0
    strike: Strike | None = None
    escape: Escape | None = None
    heat: Heat | None = None  # mode: temperature

    def parts_for_variant(self, variant: int) -> int:
        return 1


@dataclass
class Effects:
    """Everything the scene needs to know at one moment."""

    t: float
    gap_mi: float
    angular_radius: float  # radians
    distance_radii: float  # body center distance, in body radii
    tide: float  # tidal pull relative to the Moon's
    water: float  # sea level, meters
    look: float  # 0..1 people stop and stare
    flee: float  # time people start running (seconds), for per-person offsets
    shake: float
    lava: float
    fire: float
    power: float
    moon_pull: float  # 0 = Moon in place, 1 = swallowed
    ring_open: float  # degrees; negative = we see the rings from below
    meteors_from: float
    debris: float
    glow: float
    flash: float
    fade: float
    body_light: float  # how much the body lights the scene
    daylight: float = 1.0  # escapes: how much sunlight is left (1 = a normal day)
    frost: float = 0.0
    ice: float = 0.0
    air_snow: float = 0.0
    temp_f: float = 59.0  # temperature mode: the air temperature right now


class Timeline:
    def __init__(self, scenario: Scenario, seconds: float):
        self.scenario = scenario
        self.seconds = seconds
        b = scenario.body
        self.T = scenario.impact_s
        self.flyby = scenario.mode == "flyby"
        self.strike = scenario.mode == "strike"
        self.escape = scenario.mode == "escape"
        self.heat = scenario.mode == "temperature"
        if self.heat:
            h = scenario.heat
            self.k = math.log(h.end_years * HOURS_PER_YEAR / h.start_hours) / self.T if h.end_years else 0.0
            self.tc = self.T
            self.contact = 0.0
            self.t_tide_low = self.t_tide_surge = self.t_flood = self.t_steal = self.t_sun = self.t_rings = 1e9
            return
        if self.escape:
            e = scenario.escape
            self.k = math.log(e.end_years * HOURS_PER_YEAR / e.start_hours) / self.T
            self.tc = self.T
            self.contact = 0.0
            self.t_tide_low = self.t_tide_surge = self.t_flood = self.t_steal = self.t_sun = self.t_rings = 1e9
            return
        self.contact = b.radius_mi + EARTH_RADIUS_MI
        self.k = math.log(b.start_mi / self.contact) / self.T
        self.tc = scenario.closest_s if self.flyby else self.T
        if self.strike:
            st = scenario.strike
            self.contact = 0.0  # distances are to the ground
            self.tc = st.at_s
            path_mi = st.entry_alt_mi / math.sin(math.radians(st.angle_deg))
            self.t_entry = self.tc - path_mi / (st.speed_mph / 3600.0)
            self.k = math.log(b.start_mi / st.entry_alt_mi) / self.t_entry
            self.t_blast = self.tc + st.blast_after_s
            self.t_tide_low = self.t_tide_surge = self.t_flood = self.t_steal = self.t_sun = self.t_rings = 1e9
            return
        self.t_tide_low = self.time_at_tide(10)
        self.t_tide_surge = self.time_at_tide(300)
        self.t_flood = self.time_at_tide(3000)
        self.t_steal = self.time_at_center(MOON_DISTANCE_MI * (3 * b.mass_earths) ** (1 / 3))
        # When its tides pull Earth away from the Sun (the Sun's Hill sphere shrinks inside Earth's orbit).
        self.t_sun = self.time_at_center(SUN_DISTANCE_MI * (b.mass_earths / (3 * SUN_MASS_EARTHS)) ** (1 / 3))
        self.t_rings = (self.time_at_center(b.rings[1] * b.radius_mi + EARTH_RADIUS_MI)
                        if b.rings and not self.flyby else 1e9)

    # Distance

    # Flybys: log-distance moves at an even pace, slowing smoothly into the closest point.
    FLYBY_EASE = 0.25  # fraction of each leg spent easing

    @classmethod
    def _ease(cls, u: float) -> float:
        a = cls.FLYBY_EASE
        g = u * u / (2 * a) if u < a else u - a / 2
        return g / (1 - a / 2)

    @classmethod
    def _unease(cls, frac: float) -> float:
        a = cls.FLYBY_EASE
        g = frac * (1 - a / 2)
        return math.sqrt(2 * a * g) if g < a / 2 else g + a / 2

    # Escapes: a log clock, and the distance from the Sun it implies.

    def _clock(self):
        return self.scenario.heat if self.heat else self.scenario.escape

    def elapsed_hours(self, t: float) -> float:
        return self._clock().start_hours * math.exp(self.k * min(max(t, 0.0), self.T))

    def time_at_years(self, years: float) -> float:
        return min(max(math.log(years * HOURS_PER_YEAR / self._clock().start_hours) / self.k, 0.0), self.T)

    def temperature(self, t: float) -> float:
        years = self.elapsed_hours(t) / HOURS_PER_YEAR
        pts = self._clock().temperatures
        if years <= pts[1][0]:  # the first week: straight from the starting temperature
            return pts[0][1] + (pts[1][1] - pts[0][1]) * min(years / pts[1][0], 1.0)
        for (y0, f0), (y1, f1) in zip(pts[1:], pts[2:]):
            if years <= y1:
                k = math.log(years / y0) / math.log(y1 / y0)
                return f0 + (f1 - f0) * k
        return pts[-1][1]

    def temp_f(self, t: float) -> float:
        """Temperature mode: the air temperature at t (held once the story ends, never below absolute zero)."""
        h = self.scenario.heat
        if h.end_years:
            return self.temperature(t)
        if h.sun_factor_per_s:
            kelvin = (h.start_f - 32) / 1.8 + 273.15
            return (kelvin * self.sun_au(t) ** -0.5 - 273.15) * 1.8 + 32
        return max(h.start_f + h.rate_f_per_s * min(max(t, 0.0), self.T), h.floor_f)

    def time_at_temp(self, f: float) -> float:
        h = self.scenario.heat
        if h.end_years:  # falling, so bisect
            lo, hi = 0.0, self.T
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if self.temp_f(mid) > f else (lo, mid)
            return hi
        if h.sun_factor_per_s:
            k0, k = (h.start_f - 32) / 1.8 + 273.15, (f - 32) / 1.8 + 273.15
            return self.time_at_au((k0 / max(k, 1e-3)) ** 2)
        return min(max((f - h.start_f) / h.rate_f_per_s, 0.0), self.T)

    def sun_au(self, t: float) -> float:
        """Sun mode: Earth's distance from the Sun, in units of today's distance."""
        return self.scenario.heat.sun_factor_per_s ** min(max(t, 0.0), self.T)

    def time_at_au(self, au: float) -> float:
        return min(max(math.log(au) / math.log(self.scenario.heat.sun_factor_per_s), 0.0), self.T)

    def center(self, t: float) -> float:
        if self.heat:
            return 1.0
        b = self.scenario.body
        if self.escape:  # distance from the Sun, flying straight away from it
            return SUN_DISTANCE_MI + self.scenario.escape.speed_mph * self.elapsed_hours(t)
        if self.strike:  # distance to the ground: shrinking evenly in log, then straight down through the air
            st = self.scenario.strike
            if t < self.t_entry:
                return b.start_mi * math.exp(-self.k * max(t, 0.0))
            return st.entry_alt_mi * max(1.0 - (t - self.t_entry) / (self.tc - self.t_entry), 0.0)
        if not self.flyby:
            return b.start_mi * math.exp(-self.k * min(max(t, 0.0), self.T))
        lc = math.log(b.closest_mi)
        if t <= self.tc:
            u = 1 - max(t, 0.0) / self.tc
            return math.exp(lc + (math.log(b.start_mi) - lc) * self._ease(u))
        u = min((t - self.tc) / (self.T - self.tc), 1.0)
        return math.exp(lc + (math.log(b.end_mi) - lc) * self._ease(u))

    def direction(self, t: float) -> tuple[float, float]:
        """(azimuth, elevation) in degrees. A flyby drifts in from the left, passes ahead, and leaves up and right."""
        if self.strike:  # hangs over the valley, climbing as it nears, then streaks down behind the peaks
            if t < self.t_entry:
                return 9.0, 13.0 + 5.0 * smooth(t / self.t_entry)
            k = min((t - self.tc + (self.tc - self.t_entry)) / (self.tc - self.t_entry), 1.0)
            return 9.0 - 9.0 * k, 18.0 - 19.5 * k
        if not self.flyby:
            return 0.0, 10.0
        if t <= self.tc:
            k = smooth(t / self.tc)
            return -9.0 + 9.0 * k, 9.0 + 2.0 * k
        k = smooth(min((t - self.tc) / (self.T - self.tc), 1.0))
        return 24.0 * k, 11.0 + 9.0 * k

    def gap(self, t: float) -> float:
        return max(self.center(t) - self.contact, 0.0)

    def time_at_center(self, center_mi: float) -> float:
        b = self.scenario.body
        if self.escape:
            hours = max(center_mi - SUN_DISTANCE_MI, 1.0) / self.scenario.escape.speed_mph
            return self.time_at_years(hours / HOURS_PER_YEAR)
        if self.strike:
            st = self.scenario.strike
            if center_mi >= st.entry_alt_mi:
                return min(max(math.log(b.start_mi / center_mi) / self.k, 0.0), self.t_entry)
            return self.t_entry + (1 - max(center_mi, 0.0) / st.entry_alt_mi) * (self.tc - self.t_entry)
        if self.flyby:  # on the way in
            center_mi = min(max(center_mi, b.closest_mi), b.start_mi)
            lc = math.log(b.closest_mi)
            frac = (math.log(center_mi) - lc) / (math.log(b.start_mi) - lc)
            return self.tc * (1 - self._unease(frac))
        center_mi = max(center_mi, self.contact)
        t = math.log(b.start_mi / center_mi) / self.k
        return min(max(t, 0.0), self.T)

    def star_heats(self) -> bool:
        b = self.scenario.body
        return b.kind == "star" and b.luminosity_suns >= 1.0

    def star_temp_f(self, t: float) -> float:
        """Earth's average temperature with a bright star nearby: sunlight plus starlight, T ~ flux^(1/4)."""
        flux = self.scenario.body.luminosity_suns / (self.center(t) / SUN_DISTANCE_MI) ** 2
        return (288.15 * (1.0 + flux) ** 0.25 - 273.15) * 1.8 + 32

    def time_at_star_temp(self, f: float) -> float:
        kelvin = (f - 32) / 1.8 + 273.15
        flux = max((kelvin / 288.15) ** 4 - 1.0, 1e-12)
        return self.time_at_center(SUN_DISTANCE_MI * math.sqrt(self.scenario.body.luminosity_suns / flux))

    def time_at_gap(self, gap_mi: float) -> float:
        return self.time_at_center(gap_mi + self.contact)

    def tide(self, t: float) -> float:
        b = self.scenario.body
        if self.strike or self.escape or self.heat:
            return 0.0
        return b.mass_earths / MOON_MASS_EARTHS * (MOON_DISTANCE_MI / self.center(t)) ** 3

    def time_at_tide(self, factor: float) -> float:
        b = self.scenario.body
        return self.time_at_center(MOON_DISTANCE_MI * (b.mass_earths / MOON_MASS_EARTHS / factor) ** (1 / 3))

    # Text

    def label_at(self, t: float) -> str:
        label = ""
        for beat in self.scenario.beats:
            if beat.at <= t and beat.label:
                label = beat.label
        return label

    def lines(self) -> list[tuple[float, float, str]]:
        """(start, end, text) for every subtitle, including the line after impact."""
        beats = [b for b in self.scenario.beats if b.line]
        out = []
        for i, beat in enumerate(beats):
            nxt = beats[i + 1].at if i + 1 < len(beats) else self.T
            out.append((beat.at, min(beat.at + LINE_MAX_SECONDS, nxt - 0.05), beat.line))
        if self.scenario.final_line:
            out.append((self.T + 1.2, self.T + 4.3, self.scenario.final_line))
        return out

    def line_at(self, t: float) -> tuple[str, float]:
        for start, end, text in self.lines():
            if start <= t < end:
                alpha = min((t - start) / LINE_FADE_IN, (end - t) / LINE_FADE_OUT, 1.0)
                return text, max(alpha, 0.0)
        return "", 0.0

    @property
    def outro_start(self) -> float:
        return self.T + 4.5

    @property
    def hud_end(self) -> float:
        return self.t_blast + 0.4 if self.strike else self.T + 0.1

    def readout(self, t: float) -> tuple[str, str, list[str]]:
        """The HUD: a label, a big value, and small lines under it."""
        s = self.scenario
        details: list[str] = []
        if self.heat:
            milestone = self.label_at(t)
            if s.heat.end_years:  # temperature up top, elapsed time underneath
                details = [format_elapsed(self.elapsed_hours(t))] + ([milestone] if milestone else [])
                return s.hud_label, format_temp(self.temp_f(t)), details
            if s.heat.sun_factor_per_s:  # distance up top, temperature underneath
                details = [f"AVERAGE {format_temp(self.temp_f(t))}"] + ([milestone] if milestone else [])
                return s.hud_label, format_miles(SUN_DISTANCE_MI * self.sun_au(t)), details
            label, value = s.hud_label, format_temp(self.temp_f(t))
            return label, value, [milestone] if milestone else []
        if self.escape:
            label, value = s.hud_label, format_miles(self.center(t), light_years=True)
            details += [format_elapsed(self.elapsed_hours(t)), f"SURFACE {self.temperature(t):,.0f}°F"]
        elif self.strike:
            st = s.strike
            if t < self.t_entry:
                label, value = s.hud_label, format_miles(self.gap(t))
                details.append(f"{st.speed_mph:,.0f} MPH")
            elif t < self.tc:
                label, value = "ALTITUDE", f"{self.gap(t):,.0f} mi"
                details.append(f"{st.speed_mph:,.0f} MPH")
            else:
                # Real seconds until the blast: sped up at first, real time for the last few.
                real = 4.0
                if t < self.t_blast - real:
                    k = (t - self.tc) / (self.t_blast - real - self.tc)
                    remaining = st.blast_real_s - k * (st.blast_real_s - real)
                    speed = (st.blast_real_s - real) / (self.t_blast - real - self.tc)
                    pace = f"TIME SPED UP {round(speed, -1):,.0f}×"
                else:
                    remaining, pace = max(self.t_blast - t, 0.0), "REAL TIME"
                m, sec = divmod(int(math.ceil(remaining)), 60)
                label, value = "AIR BLAST ARRIVES IN", f"{m}:{sec:02d}"
                details += [f"{st.site_mi:,.0f} MILES AWAY", pace]
                return label, value, details
        else:
            far = s.body.kind in ("pulsar", "star")
            label, value = s.hud_label, format_miles(self.gap(t), light_years=far)
            tide = self.tide(t)
            if self.star_heats():
                details.append(f"AVERAGE {format_temp(self.star_temp_f(t))}")
            elif tide >= 1.0:
                than = "NORMAL" if s.body.is_moon else "THE MOON'S"
                details.append(f"TIDES {format_factor(tide)} {than}")
        milestone = self.label_at(t)
        if milestone:
            details.append(milestone)
        if s.body.kind == "pulsar" and s.body.spin_real_hz > 0 and len(details) < 2:
            details.append(f"SPIN SHOWN {s.body.spin_real_hz / s.body.spin_hz:,.0f}× SLOWER")
        return label, value, details

    # Scene

    def effects(self, t: float) -> Effects:
        T = self.T
        b = self.scenario.body
        if self.heat:
            return self._temperature_effects(t)
        center = self.center(t)
        if self.strike:
            return self._strike_effects(t, center)
        if self.escape:
            return self._escape_effects(t, center)
        radii = center / b.radius_mi
        ang = math.asin(min(1.0 / radii, 1.0))

        if t < self.t_tide_low:
            water = 0.0
        elif t < self.t_tide_surge:
            water = -8.0 * ramp(t, self.t_tide_low, self.t_tide_surge)
        elif t < self.t_flood:
            water = -8.0 + 22.0 * ramp(t, self.t_tide_surge, self.t_flood)
        else:
            settle = min(t, T if not self.flyby else self.tc + 6) - self.t_flood
            water = 14.0 + 8.0 * ramp(t, self.t_flood, self.tc if self.flyby else T) + 2.0 * math.sin(settle * 1.7) * min(settle, 1.0)

        tc = self.tc
        if self.flyby:
            closeness = min(max((ang - 0.08) / 0.14, 0.0), 1.0)
            shake = ramp(t, self.t_flood - 3, tc) * (1 - 0.7 * ramp(t, tc + 2, tc + 10))
            flash, fade = 0.0, 0.6 * ramp(t, T, T + 1.5) + 0.4 * ramp(t, T + 4.0, T + 4.5)
            debris, glow = 0.0, 0.6 * closeness
        elif b.kind == "black_hole":  # no flash, no debris: the light just goes out
            shake = 0.35 * ramp(t, T - 8, T)
            flash, fade = 0.0, ramp(t, T - 0.8, T + 0.4)
            debris, glow = 0.0, 0.0
        else:
            shake = ramp(t, self.t_flood - 3, T)
            flash = ramp(t, T, T + 0.3) * (1 - ramp(t, T + 0.5, T + 1.4))
            fade = ramp(t, T + 0.5, T + 1.4)
            debris, glow = ramp(t, T - 3.5, T), 0.7 * ramp(t, T - 6, T)
        fx = Effects(
            t=t,
            gap_mi=self.gap(t),
            angular_radius=ang,
            distance_radii=radii,
            tide=self.tide(t),
            water=water,
            look=ramp(t, 20.0, 24.0),
            flee=self.t_tide_surge - 0.5,
            shake=shake,
            lava=ramp(t, self.t_flood + 1, self.t_flood + 5),
            fire=ramp(t, self.t_flood + 2, (tc + 8) if self.flyby else T - 1),
            power=1.0 - 0.85 * ramp(t, self.t_flood, self.t_flood + 6),
            moon_pull=ramp(t, self.t_steal, self.t_steal + 8),
            ring_open=-22.0 + 19.0 * ramp(t, self.t_rings - 9, self.t_rings),
            meteors_from=self.t_rings,
            debris=debris,
            glow=glow,
            flash=flash,
            fade=fade,
            body_light=pulsar_light(center) if b.kind == "pulsar" else min((ang / 0.3) ** 2, 1.0),
        )
        if self.star_heats():  # its light heats Earth long before it touches us
            tf = self.star_temp_f(t)
            fx.temp_f = tf
            fx.fire = max(fx.fire, ramp(tf, 572.0, 700.0))  # wood ignites at about 572°F
            fx.lava = max(fx.lava, ramp(tf, 1300.0, 2200.0))  # rock melts
            fx.power = min(fx.power, 1.0 - 0.85 * ramp(tf, 255.0, 300.0))
            fx.water = fx.water - 6.0 * ramp(tf, 212.0, 700.0)  # the bay boils away
            fx.flee = self.time_at_star_temp(115.0)
            fx.debris = 0.0  # Earth is vaporized, not torn apart
        return fx


    def _temperature_effects(self, t: float) -> Effects:
        T = self.T
        return Effects(
            t=t, gap_mi=0.0, angular_radius=0.0, distance_radii=1.0, tide=0.0, water=0.0, look=0.0, flee=1e9,
            shake=0.0, lava=0.0, fire=0.0, power=1.0, moon_pull=0.0, ring_open=0.0, meteors_from=1e9, debris=0.0,
            glow=0.0, flash=0.0, fade=0.6 * ramp(t, T, T + 1.5) + 0.4 * ramp(t, T + 4.0, T + 4.5), body_light=0.0,
            temp_f=self.temp_f(t), daylight=self._daylight(t),
        )

    @property
    def t_dark(self) -> float:
        """Sun-out stories: when the last sunlight arrives (8 min 20 s after the Sun goes)."""
        h = self.scenario.heat
        return self.time_at_years(h.light_delay_min / 60 / HOURS_PER_YEAR) if h and h.light_delay_min else 1e9

    def _daylight(self, t: float) -> float:
        h = self.scenario.heat
        if h.sun_factor_per_s:
            return 1.0 / self.sun_au(t) ** 2
        if h.light_delay_min:
            return 1.0 - ramp(t, self.t_dark, self.t_dark + 0.5)
        return 1.0

    def _escape_effects(self, t: float, distance: float) -> Effects:
        T = self.T
        flux = (SUN_DISTANCE_MI / distance) ** 2
        daylight = min(max(math.log10(flux * 1e4) / 4.0, 0.0), 1.0)
        temp = self.temperature(t)
        return Effects(
            t=t, gap_mi=distance, angular_radius=0.00465 * SUN_DISTANCE_MI / distance, distance_radii=1.0,
            tide=0.0, water=0.0, look=0.0, flee=1e9, shake=0.0, lava=0.0, fire=0.0,
            power=ramp(t, 8.0, 12.0) * (1 - ramp(t, 30.0, 36.0)), moon_pull=0.0, ring_open=0.0,
            meteors_from=1e9, debris=0.0, glow=0.0, flash=0.0,
            fade=0.6 * ramp(t, T, T + 1.5) + 0.4 * ramp(t, T + 4.0, T + 4.5), body_light=0.0,
            daylight=daylight, frost=min(max((40.0 - temp) / 60.0, 0.0), 1.0),
            ice=min(max((32.0 - temp) / 25.0, 0.0), 1.0), air_snow=min(max((-300.0 - temp) / 30.0, 0.0), 1.0),
        )

    def _strike_effects(self, t: float, altitude: float) -> Effects:
        tc, tb, T = self.tc, self.t_blast, self.T
        quake = ramp(t, tc + 1.0, tc + 2.5) * (0.75 - 0.35 * ramp(t, tc + 5, tc + 12))
        blast = ramp(t, tb - 0.3, tb + 0.6)
        return Effects(
            t=t, gap_mi=altitude, angular_radius=0.0, distance_radii=max(altitude, 1e-3) / self.scenario.body.radius_mi,
            tide=0.0, water=1.5 * quake * math.sin(t * 2.3) + 2.5 * blast * math.sin(t * 3.1),
            look=0.0, flee=1e9, shake=max(quake, blast), lava=0.0,
            fire=ramp(t, tc + 9, tb), power=ramp(t, tc + 6, tc + 12),
            moon_pull=0.0, ring_open=0.0, meteors_from=tc + 9.0, debris=0.0,
            glow=ramp(t, tc, tc + 0.4) * (1 - 0.6 * ramp(t, tc + 2, tc + 12)),
            flash=ramp(t, tc, tc + 0.12) * (1 - ramp(t, tc + 0.35, tc + 1.6)),
            fade=0.8 * ramp(t, tb + 2.5, T) + 0.2 * ramp(t, T, T + 0.8),
            body_light=ramp(t, self.t_entry - 4, self.t_entry) * (1 - ramp(t, tc, tc + 0.5)),
        )


def format_factor(x: float) -> str:
    """1.5×, 27×, 84,624×, then 1.2 MILLION× / 9.7 BILLION× so huge tides stay readable."""
    if x < 10:
        return f"{x:,.1f}×"
    if x < 1e6:
        return f"{x:,.0f}×"
    for scale, word in ((1e12, "TRILLION"), (1e9, "BILLION"), (1e6, "MILLION")):
        if x >= scale:
            v = x / scale
            return f"{v:,.1f} {word}×" if v < 100 else f"{v:,.0f} {word}×"
    return f"{x:,.0f}×"


def format_elapsed(hours: float) -> str:
    if hours < 1:
        minutes = max(round(hours * 60), 1)
        return "1 MINUTE LATER" if minutes == 1 else f"{minutes} MINUTES LATER"
    if hours < 48:
        return f"{hours:,.0f} HOURS LATER" if hours >= 2 else "1 HOUR LATER"
    days = hours / 24
    if days < 60:
        return f"{days:,.0f} DAYS LATER"
    years = hours / HOURS_PER_YEAR
    if years < 2:
        return f"{days / 30.44:,.0f} MONTHS LATER"
    if years < 100:
        return f"{years:,.0f} YEARS LATER"
    digits = int(math.floor(math.log10(years))) - 1  # two significant figures
    return f"{round(years, -digits):,.0f} YEARS LATER"


def _beat_time(item: dict, timeline: Timeline) -> float:
    if "at_s" in item:
        return float(item["at_s"])
    if "years" in item:
        return timeline.time_at_years(float(item["years"]))
    if "temp_f" in item:
        f = float(item["temp_f"])
        return timeline.time_at_temp(f) if timeline.heat else timeline.time_at_star_temp(f)
    if "au" in item:
        return timeline.time_at_au(float(item["au"]))
    if "gap_mi" in item:
        return timeline.time_at_gap(float(item["gap_mi"]))
    if "tide" in item:
        return timeline.time_at_tide(float(item["tide"]))
    events = {"steal": timeline.t_steal, "recede": timeline.t_tide_low, "surge": timeline.t_tide_surge,
              "flood": timeline.t_flood, "rings": timeline.t_rings, "impact": timeline.T,
              "closest": timeline.tc, "sun": timeline.t_sun}
    if timeline.strike:
        events.update({"entry": timeline.t_entry, "strike": timeline.tc, "blast": timeline.t_blast})
    if item.get("event") in events:
        return events[item["event"]]
    raise ValueError(f"beat needs at_s, gap_mi, tide, temp_f, years or one of the events {sorted(events)}: {item}")


def load_scenarios(path: Path = CATALOG_PATH, seconds: float = 65.0) -> dict[str, Scenario]:
    raw = yaml.safe_load(path.read_text())
    out: dict[str, Scenario] = {}
    for item in raw:
        body = item.get("body") or {"name": "Air", "radius_mi": 1, "mass_earths": 0, "start_mi": 1}
        rings = body.get("rings")
        scenario = Scenario(
            id=item["id"], title=item["title"], hook=item["hook"], category=item["category"],
            scene=item.get("scene", "harbor"),
            body=Body(
                name=body["name"], radius_mi=float(body["radius_mi"]), mass_earths=float(body["mass_earths"]),
                start_mi=float(body["start_mi"]), flattening=float(body.get("flattening", 0.0)),
                rings=(float(rings[0]), float(rings[1])) if rings else None,
                bands=tuple(body.get("bands", Body.bands)), ring_color=body.get("ring_color", Body.ring_color),
                closest_mi=float(body.get("closest_mi", 0)), end_mi=float(body.get("end_mi", 0)),
                kind=body.get("kind", "planet"), ring_style=body.get("ring_style", "saturn"),
                ring_open_deg=body.get("ring_open_deg"), ring_roll_deg=float(body.get("ring_roll_deg", -16.0)),
                light_color=body.get("light_color", "#ffd9a8"),
                luminosity_suns=float(body.get("luminosity_suns", 0.0)), star_color=body.get("star_color", "#ff7a3c"),
                spot=tuple(body["spot"]) if body.get("spot") else None, spot_color=body.get("spot_color", "#c2603f"),
                surface=body.get("surface", "bands"), is_moon=bool(body.get("is_moon", False)),
                el_deg=float(body["el_deg"]) if body.get("el_deg") is not None else None,
                spin_hz=float(body.get("spin_hz", 0.5)), spin_real_hz=float(body.get("spin_real_hz", 0.0)),
                beam_color=body.get("beam_color", "#9cc4ff"),
            ),
            mode=item.get("mode", "impact"), closest_s=float(item.get("closest_s", 0)),
            strike=Strike(**item["strike"]) if "strike" in item else None,
            escape=Escape(**item["escape"]) if "escape" in item else (Escape() if item.get("mode") == "escape" else None),
            heat=Heat(**item.get("temperature", {})) if item.get("mode") == "temperature" else None,
            hud_label=item["hud_label"], impact_s=float(item["impact_s"]), beats=(),
            final_line=item.get("final_line", ""), outro=item["outro"], outro_small=item.get("outro_small", ""),
            teaser=item.get("teaser", ""), question=item["question"], accent=item.get("accent", "#f0c674"),
            hashtags=tuple(item.get("hashtags", ())),
        )
        if scenario.impact_s > seconds - 7:
            raise ValueError(f"{scenario.id}: impact_s must leave 7s for the ending")
        if scenario.mode == "flyby" and not (0 < scenario.closest_s < scenario.impact_s and
                                             scenario.body.closest_mi > 0 and scenario.body.end_mi > 0):
            raise ValueError(f"{scenario.id}: a flyby needs closest_s < impact_s, body.closest_mi and body.end_mi")
        if scenario.mode == "strike" and not (scenario.strike and scenario.strike.at_s + scenario.strike.blast_after_s
                                              < scenario.impact_s):
            raise ValueError(f"{scenario.id}: a strike needs strike.at_s + blast_after_s < impact_s")
        timeline = Timeline(scenario, seconds)
        beats = sorted((Beat(at=_beat_time(b, timeline), line=b.get("line", ""), label=b.get("label", ""))
                        for b in item["beats"]), key=lambda b: b.at)
        scenario = Scenario(**{**scenario.__dict__, "beats": tuple(beats)})
        if scenario.id in out:
            raise ValueError(f"duplicate scenario id {scenario.id}")
        out[scenario.id] = scenario
    return out
