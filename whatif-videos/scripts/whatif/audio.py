"""Ambient soundtrack, synthesized to follow each scenario's timeline: no samples, nothing to license.

Every layer is built from noise and sine waves shaped in the frequency domain (numpy FFT),
then gated by curves taken from the same timeline the picture uses: birds by daylight,
crickets at dusk, waves in the harbor, wind in the pines, rumbles while the ground shakes,
and the big moments (a flood, an impact, an air blast, the silence of an event horizon).
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from whatif.scenario import Timeline, pulse_times, ramp

SR = 44_100
CONTROL_HZ = 100


class Mix:
    def __init__(self, seconds: float, seed: int):
        self.n = int(seconds * SR)
        self.t = np.arange(self.n) / SR
        self.left = np.zeros(self.n)
        self.right = np.zeros(self.n)
        self.rng = np.random.default_rng(seed)

    def add(self, mono: np.ndarray, start: float = 0.0, gain=1.0, pan: float = 0.0) -> None:
        """Add a mono signal (equal-power pan: -1 left .. 1 right). gain may be a full-length curve."""
        i = int(start * SR)
        if i >= self.n or i + len(mono) <= 0:
            return
        a, b = max(i, 0), min(i + len(mono), self.n)
        seg = mono[a - i:b - i]
        g = gain[a:b] if isinstance(gain, np.ndarray) else gain
        angle = (pan + 1) * math.pi / 4
        self.left[a:b] += seg * g * math.cos(angle)
        self.right[a:b] += seg * g * math.sin(angle)

    def noise(self, lo: float, hi: float, slope: float = 1.0, n: int | None = None) -> np.ndarray:
        """Noise band-limited to [lo, hi] Hz with a 1/f^slope power tilt, unit RMS."""
        n = n or self.n
        spec = np.fft.rfft(self.rng.standard_normal(n))
        f = np.fft.rfftfreq(n, 1 / SR)
        f[0] = 1e-3
        mask = 1 / (1 + (lo / f) ** 4) / (1 + (f / hi) ** 4) * (f / 1000.0) ** (-slope / 2)
        y = np.fft.irfft(spec * mask, n)
        return y / (np.sqrt(np.mean(y * y)) + 1e-12)

    def stereo(self, calm: tuple[float, float] = (2.0, 14.0), calm_db: float = -26.0) -> np.ndarray:
        """Set the calm opening to calm_db RMS, then soft-limit the big moments so nothing clips."""
        out = np.stack([self.left, self.right], 1)
        a, b = int(calm[0] * SR), int(calm[1] * SR)
        rms = np.sqrt(np.mean(out[a:b] ** 2)) + 1e-9
        out *= 10 ** (calm_db / 20) / rms
        return np.tanh(out * 1.2) / 1.2 * 0.97


def tone(f0: float, f1: float, dur: float, harmonics=(1.0, 0.2), curve: float = 1.0) -> np.ndarray:
    """A pitch glide from f0 to f1 with a soft envelope."""
    n = max(int(dur * SR), 2)
    t = np.arange(n) / SR
    f = f0 + (f1 - f0) * (t / dur) ** curve
    phase = 2 * np.pi * np.cumsum(f) / SR
    y = sum(a * np.sin(phase * (k + 1)) for k, a in enumerate(harmonics))
    return y * np.sin(np.pi * t / dur) ** 2


def bird_song(rng) -> np.ndarray:
    kind = rng.choice(["whistle", "trill", "chirps", "warble"])
    gap = lambda: np.zeros(int(rng.uniform(0.04, 0.12) * SR))  # noqa: E731
    parts = []
    if kind == "whistle":
        for _ in range(rng.integers(2, 5)):
            f0 = rng.uniform(2200, 4200)
            parts += [tone(f0, f0 * rng.uniform(0.75, 1.3), rng.uniform(0.1, 0.28)), gap()]
    elif kind == "trill":
        f0 = rng.uniform(4000, 6000)
        for i in range(rng.integers(8, 18)):
            parts += [tone(f0 - i * 25, f0 - i * 25 - 400, 0.025, (1.0,)), np.zeros(int(0.03 * SR))]
    elif kind == "chirps":
        for _ in range(rng.integers(2, 5)):
            parts += [tone(rng.uniform(5500, 7000), rng.uniform(2800, 3500), rng.uniform(0.04, 0.07)), gap()]
    else:
        n = int(rng.uniform(0.3, 0.6) * SR)
        t = np.arange(n) / SR
        f = rng.uniform(2800, 3800) + 500 * np.sin(2 * np.pi * rng.uniform(18, 30) * t)
        parts.append(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / t[-1]) ** 2)
    return np.concatenate(parts)


def gull_call(rng) -> np.ndarray:
    calls = [tone(rng.uniform(1300, 1600), rng.uniform(800, 1000), rng.uniform(0.3, 0.45), (1.0, 0.5, 0.35, 0.2), 0.6)]
    for _ in range(rng.integers(1, 4)):
        calls += [np.zeros(int(0.08 * SR)), tone(1250, 1100, 0.09, (1.0, 0.5, 0.3))]
    return np.concatenate(calls)


def crickets(mix: Mix, gain: np.ndarray, count: int = 9) -> None:
    rng = mix.rng
    t = mix.t
    for _ in range(count):
        f = rng.uniform(3900, 5100)
        period = rng.uniform(0.45, 0.8)
        pulses, plen, pgap = rng.integers(3, 5), 0.013, 0.03
        local = np.mod(t + rng.uniform(0, period), period)
        idx = np.floor(local / pgap)
        within = local - idx * pgap
        env = ((idx < pulses) & (within < plen)) * np.sin(np.pi * np.clip(within / plen, 0, 1))
        mix.add(np.sin(2 * np.pi * f * t) * env, 0, gain * rng.uniform(0.004, 0.011), rng.uniform(-0.8, 0.8))


def thump(mix: Mix, dur: float = 1.6) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 30 + 40 * np.exp(-t / 0.25)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.45)
    crack = mix.noise(150, 2500, 0.5, n) * np.exp(-t / 0.07) * 0.35
    return body + crack


def boom(mix: Mix, dur: float = 7.0) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 22 + 48 * np.exp(-t / 1.4)
    low = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 2.4)
    rumble = mix.noise(20, 400, 1.5, n) * np.exp(-t / 2.0) * 0.8
    crack = mix.noise(200, 6000, 0.3, n) * np.exp(-t / 0.18) * 0.6
    return (low + rumble + crack) * np.minimum(t / 0.01, 1.0)


def soundtrack(timeline: Timeline, scene: str, seconds: float, seed: int = 0) -> np.ndarray:
    """Stereo float array (n, 2) at SR, following the scenario's timeline."""
    mix = Mix(seconds, seed)
    ct = np.arange(0, seconds + 0.01, 1 / CONTROL_HZ)
    fx = [timeline.effects(float(x)) for x in ct]

    def curve(values) -> np.ndarray:
        return np.interp(mix.t, ct, np.asarray(values, dtype=float))

    def over(fn) -> np.ndarray:
        return curve([fn(float(x)) for x in ct])

    shake = curve([f.shake for f in fx])
    T = timeline.T
    s = timeline.scenario
    rng = mix.rng

    # When everything stops: impacts cut to the boom, black holes to silence, strikes fade into the dust.
    if timeline.heat:
        cold = s.heat.rate_f_per_s < 0 or s.heat.end_years > 0
        # No air, no sound: once the air rains out, everything goes silent.
        alive = over(lambda x: (1 - ramp(timeline.temp_f(x), -318.0, -345.0)) if cold else 1 - 0.85 * ramp(x, T + 3.5, T + 6.0))
    elif timeline.strike:
        alive = over(lambda x: 1 - ramp(x, T - 1.0, T + 1.2))
    elif timeline.flyby:
        alive = over(lambda x: 1 - 0.85 * ramp(x, T + 3.5, T + 6.0))
    elif s.body.kind == "black_hole":
        alive = over(lambda x: 1 - ramp(x, T - 0.15, T + 0.05))
    else:
        alive = over(lambda x: 1 - ramp(x, T + 0.05, T + 0.4))

    # Wind, always: slow gusts.
    gust = 0.6 + 0.4 * mix.noise(0.04, 0.4, 0.0)
    gust = np.clip(gust, 0.2, 1.4)
    wind_gain = 0.05 * gust * (1 + 2.5 * shake) * alive
    mix.add(mix.noise(60, 900, 1.0), 0, wind_gain, -0.2)
    mix.add(mix.noise(60, 900, 1.0), 0, wind_gain, 0.2)

    # A low drone that swells as the body fills the sky.
    closeness = curve([f.body_light for f in fx])
    if s.body.kind == "black_hole":
        pitch = 33 + 22 * closeness
    else:
        pitch = 41 + 6 * closeness
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    drone = np.sin(phase) + 0.6 * np.sin(phase * 1.5 + 0.3 * np.sin(mix.t * 0.7)) + 0.3 * np.sin(phase * 2)
    mix.add(drone + 0.6 * mix.noise(25, 140, 1.5), 0, 0.09 * closeness ** 1.3 * alive)

    # Earthquakes: rumble and thumps while the ground shakes.
    mix.add(mix.noise(18, 160, 1.6), 0, 0.32 * shake * alive)
    x = 0.5
    while x < seconds - 1:
        k = float(np.interp(x, ct, [f.shake for f in fx]))
        if k > 0.05 and rng.random() < k:
            mix.add(thump(mix), x, 0.25 * k * float(np.interp(x, mix.t, alive)), rng.uniform(-0.6, 0.6))
        x += rng.uniform(0.25, 0.7)

    if scene == "harbor":
        _harbor(mix, timeline, curve, over, shake, alive)
    elif scene == "boardwalk":
        _boardwalk(mix, timeline, over, alive)
    elif scene == "plaza":
        _plaza(mix, timeline, over, alive)
    else:
        _forest(mix, timeline, curve, over, shake, alive)

    if s.body.kind == "pulsar":
        _pulses(mix, timeline, alive)

    if not timeline.strike and not timeline.flyby and not timeline.heat and s.body.kind != "black_hole":
        mix.add(boom(mix), T, 0.9)
    return mix.stereo()


def _harbor(mix: Mix, tl: Timeline, curve, over, shake, alive) -> None:
    rng, T = mix.rng, tl.T
    calm = np.clip(1 - 3 * shake, 0, 1)
    after = over(lambda x: ramp(x, T + 6.5, T + 8.0)) if tl.flyby else 0.0
    # Waves washing in, swelling with the tides, then the roar of the flood.
    swell = np.zeros(mix.n)
    x = rng.uniform(0, 3)
    while x < tl.seconds:
        i = int(x * SR)
        n = min(int(6 * SR), mix.n - i)
        tt = np.arange(n) / SR
        swell[i:i + n] += np.minimum(tt / 1.3, 1.0) ** 2 * np.exp(-np.maximum(tt - 1.3, 0) / 2.2)
        x += rng.uniform(4.5, 8.0)
    surge = over(lambda x: ramp(x, tl.t_tide_surge - 1, tl.t_tide_surge + 1.5) * (1 - 0.5 * ramp(x, tl.t_flood + 3, T)))
    mix.add(mix.noise(100, 2200, 1.0), 0, (0.045 + 0.06 * swell) * alive, -0.3)
    mix.add(mix.noise(100, 2200, 1.0), 0, (0.045 + 0.06 * np.roll(swell, int(1.7 * SR))) * alive, 0.3)
    mix.add(mix.noise(120, 4000, 0.7), 0, 0.22 * surge * alive)
    # Crickets on shore and a few gulls, until the shaking starts (and again once it's over, for a flyby).
    crickets(mix, (calm * over(lambda x: 1 - ramp(x, tl.t_tide_surge, tl.t_tide_surge + 2)) + 0.7 * after) * alive)
    if tl.star_heats():  # people on the promenade, as the heat becomes unbearable
        _screams(mix, tl.time_at_star_temp(115.0), tl.time_at_star_temp(170.0), 7, 0.016, alive)
    elif tl.t_tide_surge < T - 1:  # people on the promenade, as the sea comes back
        _screams(mix, tl.t_tide_surge - 0.5, min(tl.t_flood + 4.0, T - 0.5), 7, 0.016, alive)
    for _ in range(4):
        start = rng.uniform(1, 16)
        mix.add(gull_call(rng), start, rng.uniform(0.02, 0.04), rng.uniform(-0.7, 0.7))
    # Ice from the rings tearing through the sky.
    if tl.t_rings < T:
        hiss = over(lambda x: ramp(x, tl.t_rings - 0.5, tl.t_rings + 1.5))
        mix.add(mix.noise(1500, 9000, 0.4), 0, 0.05 * hiss * alive, 0.2)


def _forest(mix: Mix, tl: Timeline, curve, over, shake, alive) -> None:
    rng, T = mix.rng, tl.T
    gust = np.clip(0.5 + 0.5 * mix.noise(0.05, 0.5, 0.0), 0.1, 1.3)
    mix.add(mix.noise(700, 6000, 0.5), 0, 0.012 * gust * alive, 0.4)  # needles rustling
    if tl.strike:
        day = over(lambda x: 1 - ramp(x, tl.tc - 0.05, tl.tc + 0.05))
        dusk = 0.3 * day
    elif tl.scenario.body.kind == "black_hole":
        day = over(lambda x: 1 - ramp(x, 30.0, 42.0))
        dusk = over(lambda x: ramp(x, 34.0, 44.0))
    else:  # the same sunset as the black hole's
        day = over(lambda x: 1 - ramp(x, 30.0, 42.0))
        dusk = over(lambda x: ramp(x, 34.0, 44.0))
    # Songbirds by day.
    x = 0.2
    while x < tl.seconds - 1:
        d = float(np.interp(x, mix.t[::SR // 10], day[::SR // 10]))
        if rng.random() < d:
            mix.add(bird_song(rng), x, rng.uniform(0.012, 0.045), rng.uniform(-0.9, 0.9))
        x += rng.exponential(0.7)
    # The flock taking off (they cross the sky around 15-22s): wingbeats and alarm calls.
    for _ in range(60):
        at = rng.uniform(14.5, 21.0)
        n = int(0.06 * SR)
        flap = mix.noise(300, 3000, 0.6, n) * np.exp(-np.arange(n) / SR / 0.02)
        mix.add(flap, at, 0.03, rng.uniform(-0.6, 0.6))
    for _ in range(8):
        mix.add(tone(rng.uniform(3000, 4200), rng.uniform(2200, 2800), 0.09), rng.uniform(14.5, 19.0), 0.03,
                rng.uniform(-0.7, 0.7))
    crickets(mix, np.asarray(dusk, dtype=float) * alive if not np.isscalar(dusk) else dusk * alive)
    if tl.strike:
        _strike(mix, tl, over, alive)
    elif tl.scenario.body.kind == "black_hole":
        rise = over(lambda x: ramp(x, T - 6, T - 0.2))
        mix.add(mix.noise(40, 3000, 1.2), 0, 0.12 * rise ** 2 * alive)  # falling: a rushing swell, then nothing


def _strike(mix: Mix, tl: Timeline, over, alive) -> None:
    rng, tc, tb, T = mix.rng, tl.tc, tl.t_blast, tl.T
    # The flash makes no sound: it's 300 miles away. The ground shakes first...
    # ...rock falling back from space crackles across the sky...
    falling = over(lambda x: ramp(x, tc + 9, tc + 11) * (1 - ramp(x, tb, tb + 0.5)))
    mix.add(mix.noise(800, 7000, 0.5), 0, 0.04 * falling * alive)
    x = tc + 9.2
    while x < tb:
        n = int(1.2 * SR)
        tt = np.arange(n) / SR
        whistle = tone(rng.uniform(1800, 2600), rng.uniform(500, 800), 1.2, (1.0, 0.1)) * 0.3
        whoosh = mix.noise(300, 5000, 0.8, n) * np.sin(np.pi * tt / 1.2) ** 2
        mix.add(whistle + whoosh, x, 0.05, rng.uniform(-0.8, 0.8))
        x += rng.uniform(0.3, 0.8)
    # ...then the air blast arrives with everything at once.
    roar = over(lambda x: ramp(x, tb - 0.25, tb + 0.4))
    muffle = over(lambda x: ramp(x, tb + 2.0, T))
    mix.add(mix.noise(25, 5000, 1.1), 0, 0.5 * roar * (1 - 0.7 * muffle) * alive)
    mix.add(mix.noise(20, 300, 1.6), 0, 0.45 * roar * alive)
    mix.add(boom(mix, 5.0), tb - 0.05, 0.8)


def _pulses(mix: Mix, tl: Timeline, alive: np.ndarray) -> None:
    """A pulsar's beat: a soft tick, then a deep whoomp, each time a beam sweeps across us."""
    body = tl.scenario.body
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    f = 55 + 70 * np.exp(-t / 0.06)
    whoomp = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.22) * np.minimum(t / 0.004, 1.0)
    for at in pulse_times(body.spin_hz, tl.seconds):
        if at > tl.T:
            break
        b = tl.effects(at).body_light
        tick = mix.noise(1500, 7000, 0.4, int(0.05 * SR)) * np.exp(-np.arange(int(0.05 * SR)) / SR / 0.008)
        a = float(alive[min(int(at * SR), mix.n - 1)])
        mix.add(tick, at - 0.01, (0.035 + 0.04 * b) * a)
        mix.add(whoomp + 0.25 * mix.noise(30, 400, 1.2, n) * np.exp(-t / 0.15), at, (0.05 + 0.3 * b * b) * a)


def scream(mix: Mix, rng, dur: float | None = None, pitch: float | None = None) -> np.ndarray:
    """A distant human scream: a breathy 'ah' (formants around 800/1200/2500 Hz) gliding up then down,
    with vibrato, softened by distance (high frequencies rolled off) and a short echo off the buildings."""
    dur = dur or rng.uniform(0.55, 1.3)
    f0 = pitch or rng.uniform(380, 700)
    n = int(dur * SR)
    t = np.arange(n) / SR
    glide = f0 * (1 + 0.35 * np.sin(np.pi * np.minimum(t / dur, 1.0)) ** 0.7 - 0.25 * (t / dur) ** 2)
    glide *= 1 + 0.025 * np.sin(2 * np.pi * rng.uniform(5, 7) * t)
    phase = 2 * np.pi * np.cumsum(glide) / SR
    voice = sum(np.sin(phase * k) / k ** 0.8 for k in range(1, 12))
    voice += 0.5 * mix.noise(300, 4000, 0.6, n)  # breath
    spec = np.fft.rfft(voice)
    f = np.fft.rfftfreq(n, 1 / SR)
    formants = sum(a * np.exp(-((f - fc) / bw) ** 2) for fc, bw, a in ((850, 220, 1.0), (1250, 260, 0.7), (2600, 400, 0.35)))
    distance = 1 / (1 + (f / 2200.0) ** 2)
    y = np.fft.irfft(spec * (0.15 + formants) * distance, n)
    env = np.minimum(t / 0.06, 1.0) * np.minimum((dur - t) / 0.25, 1.0).clip(0, 1)
    y = y * env / (np.sqrt(np.mean(y * y)) + 1e-9)
    echo = np.zeros(n + int(0.23 * SR))
    echo[:n] += y
    echo[int(0.23 * SR):] += 0.35 * y
    return echo


def _screams(mix: Mix, start: float, end: float, count: int, gain: float, alive=None) -> None:
    """A few far-off screams scattered between start and end, kept quiet: felt more than heard."""
    rng = np.random.default_rng(mix.rng.integers(0, 1 << 30))
    for _ in range(count):
        at = rng.uniform(start, max(end, start + 0.1))
        a = 1.0 if alive is None else float(alive[min(int(at * SR), mix.n - 1)])
        mix.add(scream(mix, rng), at, gain * rng.uniform(0.5, 1.0) * a, rng.uniform(-0.85, 0.85))


def _crowd(mix: Mix, gain: np.ndarray) -> None:
    """A crowd's murmur: voice-band noise with a slow, uneven swell, plus scattered louder voices."""
    rng = mix.rng
    swell = np.clip(0.75 + 0.35 * mix.noise(0.2, 1.5, 0.0), 0.3, 1.4)
    mix.add(mix.noise(250, 2600, 1.2), 0, 0.05 * swell * gain, -0.25)
    mix.add(mix.noise(250, 2600, 1.2), 0, 0.05 * swell * gain, 0.25)
    x = 0.3
    while x < len(gain) / SR - 1:
        k = float(gain[int(x * SR)])
        if k > 0.05:
            f0 = rng.uniform(170, 320)
            mix.add(tone(f0, f0 * rng.uniform(0.8, 1.3), rng.uniform(0.15, 0.4), (1.0, 0.5, 0.3, 0.2)), x,
                    0.012 * k, rng.uniform(-0.8, 0.8))
        x += rng.exponential(0.25)


def _calliope(mix: Mix, power: np.ndarray) -> None:
    """A fairground organ tune that sags in pitch and dies when the power goes."""
    notes = np.array([0, 4, 7, 12, 7, 4, 0, 4, 5, 9, 12, 9, 5, 2, 7, 11, 14, 11, 7, 2, 0, 4, 7, 4])
    beat = 0.32
    t = mix.t
    pitch = np.maximum(power ** 0.35, 0.3)
    f = 392.0 * 2 ** (notes[(t / beat).astype(int) % len(notes)] / 12.0) * pitch
    within = np.mod(t, beat) / beat
    env = np.minimum(within / 0.05, 1.0) * np.exp(-within * 1.5)
    phase = 2 * np.pi * np.cumsum(f) / SR
    melody = (np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)) * env
    fb = 98.0 * 2 ** (np.array([0, 7])[(t / (beat * 2)).astype(int) % 2] / 12.0) * pitch
    bass = np.sin(2 * np.pi * np.cumsum(fb) / SR) * env
    mix.add(melody + 0.6 * bass, 0, 0.022 * np.clip(power * 1.3, 0, 1), 0.35)


def _clicks(mix: Mix, start: float, end: float, rate, gain: float, lo=1500, hi=9000, dur=0.012) -> None:
    """Short noise bursts at a (possibly time-varying) rate per second: crackles, pops."""
    rng = mix.rng
    n = int(dur * SR)
    burst = np.exp(-np.arange(n) / SR / (dur / 4))
    x = start
    while x < end:
        r = rate(x) if callable(rate) else rate
        if r > 0.01:
            mix.add(mix.noise(lo, hi, 0.3, n) * burst, x, gain * rng.uniform(0.4, 1.0), rng.uniform(-0.9, 0.9))
            x += rng.exponential(1.0 / r)
        else:
            x += 0.1


def _boardwalk(mix: Mix, tl: Timeline, over, alive) -> None:
    rng = mix.rng
    temp = tl.temp_f
    crowd = over(lambda x: 1 - ramp(temp(x), 92.0, 150.0))
    power = over(lambda x: 1 - ramp(temp(x), 255.0, 290.0))
    _crowd(mix, crowd * alive)
    _screams(mix, tl.time_at_temp(105.0), tl.time_at_temp(150.0), 6, 0.016, alive)
    _calliope(mix, power * alive)
    for _ in range(3):
        mix.add(gull_call(rng), rng.uniform(0.5, 8), rng.uniform(0.015, 0.03), rng.uniform(-0.7, 0.7))
    # The lake boils: a hiss and a stew of low bubbles.
    boil = over(lambda x: ramp(temp(x), 205.0, 240.0) * (1 - ramp(temp(x), 480.0, 600.0)))
    mix.add(mix.noise(1500, 8000, 0.6), 0, 0.03 * boil * alive, -0.3)
    x = tl.time_at_temp(205.0)
    while x < tl.time_at_temp(600.0):
        k = float(boil[min(int(x * SR), mix.n - 1)])
        if k > 0.05:
            f0 = rng.uniform(70, 220)
            mix.add(tone(f0, f0 * 1.8, rng.uniform(0.04, 0.09), (1.0, 0.3)), x, 0.05 * k, rng.uniform(-0.6, 0.6))
        x += rng.exponential(0.03)
    # Every kernel pops at once.
    t_pop = tl.time_at_temp(356.0)
    _clicks(mix, t_pop, t_pop + 2.2, lambda x: 120 * math.exp(-((x - t_pop - 0.8) / 0.6) ** 2), 0.07, 800, 6000, 0.02)
    # Fire: crackles from 451, a roar once the wood goes.
    _clicks(mix, tl.time_at_temp(451.0), tl.seconds - 0.5, lambda x: 4 + 60 * ramp(temp(x), 451.0, 600.0), 0.06)
    roar = over(lambda x: ramp(temp(x), 470.0, 610.0))
    mix.add(mix.noise(30, 1200, 1.3), 0, 0.28 * roar * alive)


def _plaza(mix: Mix, tl: Timeline, over, alive) -> None:
    rng = mix.rng
    temp = tl.temp_f
    dark = tl.t_dark
    if dark < 1e8:  # the lights go out: gasps, then everyone heads home
        crowd = over(lambda x: 1 - ramp(x, dark + 0.5, dark + 6.5))
        yelp_from, yelp_to = dark, dark + 2.0
    else:
        crowd = over(lambda x: 1 - ramp(temp(x), 64.0, 30.0))
        yelp_from, yelp_to = tl.time_at_temp(55.0), tl.time_at_temp(32.0)
    _crowd(mix, crowd * alive)
    rng_y = np.random.default_rng(mix.rng.integers(0, 1 << 30))
    for _ in range(3):
        mix.add(scream(mix, rng_y, dur=rng_y.uniform(0.3, 0.5)), rng_y.uniform(yelp_from, yelp_to),
                0.012, rng_y.uniform(-0.8, 0.8))
    # Pigeons cooing, then taking off all at once.
    t_fly = min(tl.time_at_temp(48.0), tl.t_dark + 0.3)
    x = 0.2
    while x < t_fly:
        f0 = rng.uniform(380, 520)
        n = int(0.35 * SR)
        coo = tone(f0, f0 * 0.8, 0.35, (1.0, 0.4))[:n] * (0.6 + 0.4 * np.sin(np.linspace(0, 18, n)))
        mix.add(coo, x, 0.015, rng.uniform(-0.6, 0.6))
        x += rng.exponential(0.5)
    for _ in range(40):
        n = int(0.05 * SR)
        mix.add(mix.noise(300, 3000, 0.6, n) * np.exp(-np.arange(n) / SR / 0.015), t_fly + rng.uniform(0, 1.6), 0.035,
                rng.uniform(-0.7, 0.7))
    # The fountain splashes until it freezes.
    splash = over(lambda x: 1 - ramp(temp(x), 34.0, 31.0))
    mix.add(mix.noise(600, 7000, 0.4), 0, 0.03 * splash * alive, -0.4)
    # Ice cracking on the river and the fountain.
    x = tl.time_at_temp(32.0)
    while x < tl.time_at_temp(-90.0):
        n = int(0.25 * SR)
        tt = np.arange(n) / SR
        crack = mix.noise(200, 5000, 0.5, n) * np.exp(-tt / 0.03) + 0.6 * np.sin(2 * np.pi * 60 * tt) * np.exp(-tt / 0.1)
        mix.add(crack, x, rng.uniform(0.03, 0.08) * float(alive[min(int(x * SR), mix.n - 1)]), rng.uniform(-0.8, 0.8))
        x += rng.exponential(0.6)
    # Wind picks up as the cold deepens.
    howl = over(lambda x: ramp(temp(x), 20.0, -150.0))
    mix.add(mix.noise(150, 1500, 1.0) * (0.7 + 0.3 * np.sin(mix.t * 0.6)), 0, 0.09 * howl * alive, 0.2)


def write_wav(path: Path, stereo: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(stereo, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path
