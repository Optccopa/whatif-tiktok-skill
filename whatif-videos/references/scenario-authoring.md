# Writing a scenario

Contents: 1. Minimal entry · 2. All fields · 3. Pinning beats · 4. Physics · 5. Fact sheet · 6. Pitfalls · 7. Temperature videos

## 1. Minimal entry (impact mode)

Copy the closest catalog entry from `scripts/whatif/scenarios.yaml` and edit it. The skeleton:

```yaml
- id: mars-crash                       # lowercase-dashed, unique; becomes the file name
  title: What if Mars crashed into Earth?
  hook: "What if {Mars} crashed into Earth?"   # {braces} = accent color; keep the whole hook on 2 lines
  category: space
  scene: forest                        # harbor | forest
  accent: "#e8845a"                    # hook highlight + COMMENT BELOW; pick from the body's colors
  body:
    name: Mars
    radius_mi: 2106
    mass_earths: 0.107
    start_mi: 34600000                 # where the story starts (center to center); a meaningful real distance
    flattening: 0.006
    bands: ["#d9875a", "#a4502e", "#e9d8c8"]   # light band, dark band, pole caps
    light_color: "#ffb08a"             # light it throws on the scene when it's big
  hud_label: DISTANCE TO MARS
  impact_s: 57                         # must leave 7 s for the ending (65 s video)
  beats:
    - {at_s: 0, label: AS CLOSE AS IT EVER GETS}
    - {at_s: 3.8, line: "That red dot is Mars. It's heading straight for us."}
    # ... a line every ~4 s ...
    - {event: impact, label: IMPACT}
  final_line: "One line that reframes what we just saw."
  outro: What should hit Earth next?
  outro_small: COMMENT BELOW
  teaser: Watch the sky at the end 👀  # caption line 2
  question: What should hit Earth next?  # caption line 3
  hashtags: [whatif, mars, space, astronomy, science]
```

## 2. All fields

**Top level:** `id, title, hook, category, scene, accent, body, hud_label, impact_s, beats, final_line,
outro, outro_small, teaser, question, hashtags`, plus:
- `mode`: `impact` (default) | `flyby` | `strike` | `escape`
- `closest_s`: flyby only. Seconds at the closest point; must be less than `impact_s`. 38 works well.
- `strike:` (strike mode): `at_s` (impact second), `entry_alt_mi` (62, the Kármán line), `angle_deg` (45),
  `speed_mph`, `site_mi` (distance to the impact site), `blast_after_s` (screen seconds from impact to
  the air blast), `blast_real_s` (the same in real seconds; the HUD counts it down), `coma_mi`.
  Needs `at_s + blast_after_s < impact_s`.
- `escape:` (escape mode): `speed_mph` (2.2M), `start_hours` (1), `end_years` (3e7), `temperatures`
  (list of [years, °F] anchors, interpolated in log time).
- `temperature:` (temperature mode): `start_f` (75), `rate_f_per_s` (10, negative for colder), `floor_f` (−459.67).
  No `body:` needed; set `hud_label: TEMPERATURE` and `scene: boardwalk` (heat) or `plaza` (cold).

**`body:`**
| field | meaning |
|---|---|
| `name, radius_mi, mass_earths, start_mi` | required. `radius_mi` is the event horizon for black holes |
| `kind` | `planet` (default), `star`, `pulsar`, `black_hole`, `comet` |
| `flattening` | oblateness (Saturn 0.098, Jupiter 0.065) |
| `bands` | 3 hex colors: light band, dark band, poles |
| `rings` | `[inner, outer]` in body radii |
| `ring_style` | `saturn` (broad bright), `uranus` / `neptune` (thin dark) |
| `ring_color`, `ring_open_deg` (fixed tilt; omit to animate), `ring_roll_deg` | ring look |
| `spot`, `spot_color` | storm: `[lat°, lon°, size]`, e.g. Jupiter `[-22, -20, 0.09]` |
| `light_color` | the light it casts on the scene |
| `luminosity_suns`, `star_color` | stars only (harbor scene) |
| `closest_mi`, `end_mi` | flybys: center distance at the closest point, and when it leaves |
| `surface` | `bands` (default; gas/ice giants) or `rocky` (mottled terrain from `bands[0]` plains and `bands[1]` dark regions, small `bands[2]` polar caps; ignores `spot`). Use `rocky` for Mars, Venus-without-clouds, Mercury, the Moon |
| `el_deg` | impact mode: the body's elevation in degrees, overriding the scene (harbor 10, forest 8.5). Small bodies (Mars-sized) sit on the harbor's central peak at 10; 13 clears it |
| `spin_hz`, `spin_real_hz`, `beam_color` | pulsars (harbor): turns per second on screen (0.5 = a flash every 2 s), the real spin (adds a "SPIN SHOWN N× SLOWER" HUD note while there's room), beam color. Also uses `star_color`, `light_color`. The readout switches to light-years when far |

## 3. Pinning beats

Each beat sets a `line` (subtitle), a `label` (the milestone under the distance), or both. Pin it with exactly one of:

| key | when |
|---|---|
| `at_s: 12.0` | a fixed second |
| `gap_mi: 34600000` | when the surface-to-surface gap reaches this (impact/strike) or the distance from the Sun (escape) |
| `tide: 1` | when its tides are N× the Moon's (planets) |
| `years: 25000` | escape: when this much time has passed |
| `temp_f: 451` | temperature mode: when the air reaches this many °F |
| `event: …` | `steal` (it takes the Moon), `recede` (tide 10×: the sea pulls back), `surge` (300×: the wave comes), `flood` (3000×: the shore floods), `rings` (Earth enters the rings), `sun` (it out-pulls the Sun), `closest`, `impact`; strike mode adds `entry`, `strike`, `blast` |

Timing rules (the `check` command enforces them):
- The first line goes at **3.8 s**, after the hook fades. A label at `at_s: 0` sets the opening readout.
- Leave **3.5–4.5 s** between lines; a line shows for at most 4.2 s. Aim for 8–11 words per line.
- Lines pinned to distances land wherever the physics puts them. Run `check`, then slide the `at_s` lines
  around them so nothing gets squeezed.
- Events clamp into [0, impact_s]. For a light body, `steal` or `flood` may clamp to the impact second,
  meaning it never happens in time. Don't script an event that `check` shows at the very end
  unless that's the story.
- The scene effects are tied to the same events. In impact and flyby modes, the sea recedes, surges
  and floods at tide 10/300/3000×; volcanoes start ~1 s after the flood; fires and power cuts follow;
  people turn to look at 20–24 s; the Moon drifts toward the body after `steal`. Write lines that
  match those moments.

## 4. Physics (keep every number derivable)

Distances are center to center unless noted. Earth radius 3,959 mi, Moon at 238,855 mi, Moon mass
0.0123 Earths, Sun 93M mi away and 333,000 Earths.

- **Tides vs. the Moon:** `(M / 0.0123) × (238,855 / d)³`, with M in Earth masses. Each halving of the
  distance makes tides 8× stronger.
- **It steals the Moon** when its Hill radius reaches the Moon's orbit: `d = 238,855 × (3M)^(1/3)`
  (strictly that's when the Moon falls into its pull rather than Earth's).
- **It out-pulls the Sun** (Earth leaves its orbit): `d = 93M × (M / (3 × 333,000))^(1/3)`.
- **Looks bigger than the full Moon** (0.52° across) once `d < R / tan(0.26°) ≈ 220 R`.
- **Angular size:** `2 × asin(R/d)`; "N times wider than the full Moon" = that ÷ 0.52°.
- **Roche limit** (a fluid body is torn apart): `≈ 2.44 R_primary × (ρ_primary/ρ_satellite)^(1/3)`.
- **Black holes:** event horizon 2.95 km (1.83 mi) per Sun mass. Supermassive ones (above ~10⁸ Suns)
  have tides too weak at the horizon to tear Earth apart, so it crosses intact.
- **Light-year** 5.879 trillion mi. Light takes 8.3 min from the Sun.
- **Escape speed** from the Milky Way near the Sun is about 1.2M mph (~550 km/s).
- **Impacts:** comets hit at ~25–70 km/s (55,000–160,000 mph), asteroids at ~20 km/s. The Chicxulub
  impactor was ~6 mi wide. A blast wave travels at roughly sound speed, ~17 min per 300 mi.

To check a computed moment, run `check`: it prints the HUD (distance, tide factor, label) at every beat.

## 5. Fact sheet (NASA planetary fact sheets)

| body | radius mi | mass (Earths) | flattening | closest to Earth | notes |
|---|---|---|---|---|---|
| Moon | 1,080 | 0.0123 | 0.001 | 225,700 mi (perigee) | |
| Mercury | 1,516 | 0.055 | 0 | 48M mi | |
| Venus | 3,760 | 0.815 | 0 | 24M mi | hottest planet, 870°F |
| Mars | 2,106 | 0.107 | 0.006 | 34.6M mi | Olympus Mons 13.6 mi high; 2 moons; day 24h37m |
| Jupiter | 43,441 | 317.8 | 0.065 | 365M mi | Great Red Spot wider than Earth; 95 known moons |
| Saturn | 37,449 | 95.16 | 0.098 | 746M mi | density 0.69 g/cm³ (would float); rings 170,000 mi across, ~30 ft thick |
| Uranus | 15,759 | 14.54 | 0.023 | 1.6B mi | tilted 98°; 13 thin rings |
| Neptune | 15,299 | 17.15 | 0.017 | 2.7B mi | winds 1,200 mph; Voyager 2 in 1989 |
| Sun | 432,690 | 333,000 | 0 | 91.4M mi | |
| Neutron star / pulsar | ~7 (11 km) | ~466,000 (1.4 Suns) | | nearest known ~400 ly | spins up to 700×/s; a teaspoon weighs ~a billion tons; beams sweep like a lighthouse |
| M87* black hole | 11.9B (horizon) | 2.16e15 (6.5B Suns) | | 53M ly | first black hole imaged (2019) |

Cite only what you can back up. When unsure, round down and use "about".

## 6. Pitfalls we've hit

- A label saying "AS FAR AS THE NEAREST STAR" stayed on screen at 388 light-years. Word labels
  as thresholds ("PAST…", "CLOSER THAN…").
- A line about the red spot played before the spot was visible. Lines must match the frame.
- A black hole sat behind the peaks in the first render. Check stills: the hero must be above the skyline.
- Subtitles collided when a distance-pinned line landed right after a timed one. Use `check`.
- `impact_s` later than 58 leaves no room for the ending (the loader rejects it).
- Mars first rendered with gas-giant bands and looked like Jupiter. Rocky bodies need `surface: rocky`.
- Tides match the Moon's at `238,855 × (M / 0.0123)^(1/3)`: for any body heavier than the Moon that's
  *farther* than the Moon (Mars: 491,000 mi). Easy to invert by hand; trust `check`.
- Light bodies (Mars) are small until late: before ~28 s it's a dot, so early lines should be facts that
  don't need detail on screen, and the hero only fills the sky in the last ~8 s.

## 7. Temperature videos

The air changes by `rate_f_per_s` every second from `start_f`: at 10°F/s, 75°F becomes 645°F by 57 s,
or reaches absolute zero (−459.67°F) at 53.5 s, where it holds. Pin each beat to the real threshold
with `temp_f`, and make sure the scene shows it (the scenes key their effects to the same numbers):

| °F | what happens | scene effect |
|---|---|---|
| 134 | hottest air temperature ever recorded (Death Valley, 1913) | crowd gone by ~150 |
| 110–260 | grass and leaves dry to straw | `dry` shader |
| 212 | water boils (sea level) | lake foams, steam, then the level drops |
| 255–290 | wiring and transformers fail | lights, music, rides stop |
| 285–330 | balloons burst; plastic slumps (280–380) | balloons vanish, bins sag |
| 356 | popcorn pops (~356°F, 180°C) | the cart bursts |
| 451 | paper autoignites (the Fahrenheit 451 figure) | litter flames |
| 470–540 | canvas awnings and tents catch | awning and tent flames |
| 572 | wood autoignites (~570°F) | stalls, boards, trees burn; everything chars |
| 621 | lead melts | |
| 867 | Venus's average surface temperature (NASA) | |
| 32 | water freezes | fountain stops mid-spray; river ices over (to −40) |
| 20 to −90 | open water steams in colder air (sea smoke) | river steam, ice fog |
| −40 | Fahrenheit and Celsius meet | |
| −109.3 | carbon dioxide freezes out (deposition) | dry-ice frost falls |
| −128.6 | coldest natural air temperature recorded (Vostok, 1983) | |
| −166 | Jupiter's cloud tops (NASA) | |
| −297 | oxygen liquefies | pale blue rain |
| −320 | nitrogen liquefies: the air rains out | the sky turns black, the sound stops |
| −375 to −400 | Pluto's surface (NASA) | |
| −459.67 | absolute zero | the clock stops here |

Balloons in the cold shrink with absolute temperature (volume ∝ T, so radius ∝ T^⅓). That makes a
true and visible line: "The balloons shrivel as the helium inside contracts."

For a new temperature scene (for example a beach, a ski slope or a backyard), copy `boardwalk.py` or
`plaza.py`, keep the camera locked, and key every effect to `fx.temp_f` with `ramp(temp, a, b)` (which
also works for falling ranges, a > b).
