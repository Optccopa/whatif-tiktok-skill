import numpy as np
import pytest

from whatif.scenario import EARTH_RADIUS_MI, Timeline, load_scenarios


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios()


def test_catalog_loads_with_beats_in_order(scenarios):
    assert {"saturn-crash", "rogue-gas-giant"} <= set(scenarios)
    for s in scenarios.values():
        times = [b.at for b in s.beats]
        assert times == sorted(times)
        assert all(0 <= t <= s.impact_s for t in times)


def test_impact_distance_reaches_contact(scenarios):
    s = scenarios["saturn-crash"]
    tl = Timeline(s, 65)
    assert tl.gap(0) == pytest.approx(s.body.start_mi - s.body.radius_mi - EARTH_RADIUS_MI)
    assert tl.gap(s.impact_s) == 0
    gaps = [tl.gap(t / 2) for t in range(0, 2 * int(s.impact_s))]
    assert gaps == sorted(gaps, reverse=True)
    # Saturn's tides match the Moon's around 4.7 million miles out.
    assert tl.center(tl.time_at_tide(1)) == pytest.approx(4.72e6, rel=0.02)


def test_flyby_comes_in_and_goes_out(scenarios):
    s = scenarios["rogue-gas-giant"]
    tl = Timeline(s, 65)
    assert tl.center(s.closest_s) == pytest.approx(s.body.closest_mi)
    assert tl.center(0) == pytest.approx(s.body.start_mi)
    assert tl.center(s.impact_s) == pytest.approx(s.body.end_mi)
    assert tl.center(s.closest_s - 5) > tl.center(s.closest_s) < tl.center(s.closest_s + 5)
    for d in (5e7, 5e6, 5e5):
        assert tl.center(tl.time_at_center(d)) == pytest.approx(d, rel=1e-6)


def test_effects_are_sane(scenarios):
    for s in scenarios.values():
        tl = Timeline(s, 65)
        for t in range(0, 65):
            fx = tl.effects(float(t))
            assert -10 <= fx.water <= 30
            assert 0 <= fx.fade <= 1 and 0 <= fx.flash <= 1


def test_black_hole_numbers(scenarios):
    s = scenarios["black-hole"]
    tl = Timeline(s, 65)
    assert s.body.kind == "black_hole" and s.scene == "forest"
    # Earth crosses the horizon intact: its tidal breakup distance is far inside the horizon.
    breakup_mi = EARTH_RADIUS_MI * (2 * s.body.mass_earths) ** (1 / 3)
    assert breakup_mi < s.body.radius_mi / 10
    # The Sun loses Earth before the Moon is lost, and both happen before the horizon.
    assert 0 < tl.t_sun < tl.t_steal < s.impact_s
    assert tl.effects(s.impact_s + 1).fade == 1.0 and tl.effects(s.impact_s).flash == 0.0


def test_format_miles():
    from whatif.overlay import format_miles

    assert format_miles(24.9e12) == "24.90 trillion mi"
    assert format_miles(433.96e9) == "433.96 billion mi"
    assert format_miles(238_855) == "238,855 mi"


def test_terrain_has_a_lake_below_the_ridge():
    from whatif.forest import LAKE, terrain_height

    assert terrain_height(LAKE[0], LAKE[1]) < 0 < terrain_height(0.0, 320.0)


def test_comet_strike_readout(scenarios):
    s = scenarios["comet-strike"]
    tl = Timeline(s, 65)
    assert 0 < tl.t_entry < tl.tc < tl.t_blast < s.impact_s
    assert tl.tc - tl.t_entry == pytest.approx(3.16, abs=0.05)  # 62 miles at 45 degrees, 100,000 mph
    assert tl.readout(1.0)[0] == "DISTANCE TO COMET"
    assert tl.readout(tl.t_entry + 1)[0] == "ALTITUDE"
    label, value, details = tl.readout(tl.tc + 0.01)
    assert label == "AIR BLAST ARRIVES IN" and value == "18:20" and "300 MILES AWAY" in details
    assert tl.readout(tl.t_blast + 0.1)[1] == "0:00"


def test_soundtrack_is_stereo_and_quiet_enough():
    from whatif.audio import SR, soundtrack

    scenarios = load_scenarios()
    for sid in ("saturn-crash", "comet-strike"):
        s = scenarios[sid]
        audio = soundtrack(Timeline(s, 65), s.scene, 16.0, seed=1)
        assert audio.shape == (16 * SR, 2)
        assert np.abs(audio).max() <= 1.0
        rms = np.sqrt(np.mean(audio[2 * SR:14 * SR] ** 2))
        assert 0.02 < rms < 0.08  # about -26 dB: ambience, not a wall of noise


def test_new_bodies_load_with_their_looks(scenarios):
    uranus, neptune, star = scenarios["uranus-crash"], scenarios["neptune-crash"], scenarios["star-flyby"]
    assert uranus.body.ring_style == "uranus" and uranus.body.ring_open_deg == 68 and uranus.scene == "forest"
    assert neptune.body.ring_style == "neptune" and neptune.scene == "harbor"
    assert star.body.kind == "star" and star.mode == "flyby" and star.body.luminosity_suns == 0.001
    for s in (uranus, neptune, star):
        lines = Timeline(s, 65).lines()
        assert all(end - start >= 1.5 for start, end, _ in lines), s.id  # every subtitle stays up long enough to read


def test_star_light_scale():
    from whatif.video import star_light

    star = load_scenarios()["star-flyby"].body
    assert star_light(star, star.start_mi) == 0.0
    assert star_light(star, 3_000_000) == pytest.approx(1.0, abs=0.01)  # as bright as our Sun at closest


def test_escape_clock_and_readout(scenarios):
    from whatif.scenario import format_elapsed

    s = scenarios["earth-ejected"]
    tl = Timeline(s, 65)
    assert tl.elapsed_hours(0) == pytest.approx(1.0)
    assert tl.elapsed_hours(s.impact_s) / 8766 == pytest.approx(30e6, rel=1e-6)
    assert tl.temperature(tl.time_at_years(1 / 52)) == pytest.approx(0.0, abs=0.5)
    assert tl.temperature(tl.time_at_years(1.0)) == pytest.approx(-100.0, abs=0.5)
    # 1,000 km/s covers a light-year in about 300 years.
    assert tl.center(tl.time_at_years(300_000)) / 5.879e12 == pytest.approx(1000, rel=0.05)
    label, value, details = tl.readout(30.0)
    assert label == "DISTANCE FROM THE SUN" and "light-years" in value and details[1].startswith("SURFACE")
    assert format_elapsed(30) == "30 HOURS LATER" and format_elapsed(24 * 9) == "9 DAYS LATER"
    assert format_elapsed(8766 * 31_622) == "32,000 YEARS LATER"
    fx = tl.effects(tl.time_at_years(100_000))
    assert fx.daylight == 0.0 and fx.frost == 1.0 and fx.ice == 1.0 and fx.air_snow == 1.0


def test_hook_keeps_punctuation_tight():
    from whatif.overlay import balance, serif, split_hook

    lines = balance(split_hook("What if Earth was {flung out of the galaxy}?", (1, 2, 3)), serif(88, 560), 900)
    assert "".join(w for line in lines for w, _ in line).endswith("galaxy?")


def test_rocky_surface_and_elevation_load(tmp_path):
    from whatif.video import WhatIfScene
    yaml_text = """
- id: rocky-test
  title: t
  hook: "What if {Mars} crashed into Earth?"
  category: space
  scene: harbor
  body: {name: Mars, radius_mi: 2106, mass_earths: 0.107, start_mi: 34600000, surface: rocky, el_deg: 13}
  hud_label: DISTANCE TO MARS
  impact_s: 57
  beats: [{event: impact, label: IMPACT}]
  outro: q
  question: q
"""
    p = tmp_path / "s.yaml"
    p.write_text(yaml_text)
    s = load_scenarios(p)["rocky-test"]
    assert s.body.surface == "rocky" and s.body.el_deg == 13
    assert scenarios_default_surface() == "bands"
    scene = WhatIfScene(s, 65)
    frame, _, _ = scene.frame(10.0)
    assert frame.surface == 1.0
    assert np.degrees(np.arcsin(frame.body_dir[1])) == pytest.approx(13, abs=0.01)
    # Mars's tides match the Moon's ~491,000 mi out: farther than the Moon, since it's heavier.
    tl = Timeline(s, 65)
    assert tl.center(tl.time_at_tide(1)) == pytest.approx(4.91e5, rel=0.02)


def scenarios_default_surface():
    return load_scenarios()["saturn-crash"].body.surface
def test_pulsar_helpers():
    from whatif.scenario import format_factor, pulse_times, pulsar_light
    from whatif.video import pulsar_beam, _direction

    assert pulsar_light(1e14) == 0.0 and pulsar_light(1e5) == 1.0 and 0 < pulsar_light(3e9) < 1
    times = pulse_times(0.5, 10.0)
    assert times[0] == pytest.approx(0.3) and times[1] - times[0] == pytest.approx(2.0)
    s = _direction(2.0, 15.0)
    beam, pulse = pulsar_beam(s, times[2], 0.5)
    assert pulse == pytest.approx(1.0) and float(beam @ -s) == pytest.approx(1.0)  # the beam hits us on each pulse
    assert pulsar_beam(s, times[2] + 1.0, 0.5)[1] < 0.01  # and misses us half a turn later
    assert format_factor(1.04) == "1.0×" and format_factor(84624) == "84,624×"
    assert format_factor(3.04e7) == "30.4 MILLION×" and format_factor(8.3e12) == "8.3 TRILLION×"


def test_temperature_mode_clock_and_readout(scenarios):
    from whatif.scenario import format_temp
    hot, cold = scenarios["hotter-every-second"], scenarios["colder-every-second"]
    th, tc = Timeline(hot, 65), Timeline(cold, 65)
    assert th.temp_f(0) == 75 and th.temp_f(13.7) == pytest.approx(212)
    assert th.time_at_temp(451) == pytest.approx(37.6)
    assert tc.temp_f(60) == pytest.approx(-459.67)  # never below absolute zero
    assert tc.time_at_temp(-459.67) == pytest.approx(53.467, abs=0.01)
    assert format_temp(-258.4) == "−258°F" and format_temp(-459.67) == "−459.67°F"
    label, value, details = th.readout(14.0)
    assert label == "TEMPERATURE" and value == "215°F" and details == ["WATER BOILS"]


def test_falling_ramp():
    from whatif.scenario import ramp
    assert ramp(40, 32, -60) == 0 and ramp(-60, 32, -60) == 1 and 0 < ramp(-14, 32, -60) < 1


def test_street_scenes_build_frames(scenarios):
    from whatif.video import WhatIfScene, project
    for sid in ("hotter-every-second", "colder-every-second"):
        scene = WhatIfScene(scenarios[sid], 65)
        frame, dynamic, fx = scene.frame(1.0)
        assert frame.key_dir is not None and "limb" in dynamic  # lit by the sun, with a crowd out
        assert "limb" not in scene.frame(30.0)[1]  # by 30 s everyone has fled the heat or the cold
        x, y, depth = project(frame, scene.scene.sign_point)
        assert 0 < x < 1080 and 0 < y < 1920 and depth > 0  # the temperature board is in frame


def test_sun_approach_curve(scenarios):
    tl = Timeline(scenarios["sun-one-percent-closer"], 65)
    assert tl.sun_au(10) == pytest.approx(0.99 ** 10)
    assert tl.temp_f(0) == pytest.approx(59)
    assert tl.time_at_au(0.723) == pytest.approx(32.25, abs=0.05)  # Venus's orbit
    assert tl.time_at_temp(212) == pytest.approx(51.4, abs=0.1)
    label, value, details = tl.readout(0.0)
    assert label == "DISTANCE TO THE SUN" and value == "93,000,000 mi" and details[0] == "AVERAGE 59°F"
