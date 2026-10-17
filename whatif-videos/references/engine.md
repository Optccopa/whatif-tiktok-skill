# The engine (read only when YAML can't express the request)

Contents: 1. Pipeline · 2. Modules · 3. Adding a body kind (worked example: pulsar) · 4. Other changes · 5. Performance and troubleshooting

## 1. Pipeline

```
scenarios.yaml ──load_scenarios──▶ Scenario ──▶ Timeline (distance, events, beats, readout, Effects per t)
                                                   │
          scene (Harbor | Forest): static Instances + dynamic(t, fx) Instances + camera(t, fx)
                                                   │
 video.WhatIfScene.frame(t) → LOOKS[scene](...) → gl.Frame (sky colors, body uniforms, lights)
                                                   │
 gl.Renderer.render(frame, dynamic) → RGB array → overlay.Overlay.apply(img, t) → ffmpeg (encode.py)
 audio.soundtrack(timeline, scene) → WAV ─────────────────────────────────────────▶ muxed AAC
```

Each frame is computed from `t` alone, with no state carried between frames, so `stills` renders
exactly what the video will show at those times.

## 2. Modules (`scripts/whatif/`)

| file | what it holds |
|---|---|
| `scenario.py` | dataclasses `Body`, `Strike`, `Escape`, `Scenario`, `Effects`; `Timeline` (center/gap/direction, the tide, event times, `effects(t)`, `readout(t)`, `lines()`); `load_scenarios` (validation and beat pinning); `format_miles` |
| `meshes.py` | low-poly prototypes (box, cyl, cone, ico, hull, prism...), transforms `tr/sc/rx/ry/rz/chain`, `rgb()` (hex→linear), material ids, `Instances` (24 floats per instance: mat4, rgb+emissive, material/seed/p1/p2) |
| `gl.py` | moderngl standalone EGL context; the GLSL for objects (materials: SOLID, WATER, GLOW, BUILDING with windows, STRIPES, MOUNTAIN lava, FLAME, GROUND, TERRAIN) and the sky shader; `Frame` (every per-frame uniform); `Renderer` (4× MSAA HDR, shadow map, bloom, ACES) |
| `harbor.py`, `forest.py`, `boardwalk.py`, `plaza.py` | the scenes: geometry built once, dynamic objects per frame, `camera()` (locked), `lights()`, `body_az_el`, `moon_az_el` |
| `video.py` | `harbor_look`, `forest_look`, `comet_look`, `escape_look` map (t, Effects) to a `Frame`; `WhatIfScene`; `render_frames`, `render_whatif`, `render_whatif_stills` |
| `props.py` | reusable street props: `Crowd` (strolling people who leave at set times toward an exit, optionally down stairs), stalls, string lights, lamp posts, benches, round trees, circus tents, popcorn cart, LED sign post, Ferris wheel, balloons, pigeons, flames, steam/smoke puffs |
| `overlay.py` | Pillow text sprites: hook, HUD, subtitles, outro; balanced line wrapping |
| `audio.py` | numpy-synthesized ambience: wind, drone, quake, the scene's beds, impact boom; `soundtrack()` |
| `encode.py` | the ffmpeg command, bitrate cap for the size limit, `shrink_to_fit` |

**Sky shader body types** (`Frame.body_type`, `u_body_type`): −1 none · 0 planet (ray-traced
oblate spheroid, bands, rings with shadows, storm spot) · 1 black hole (shadow, lensing, accretion disk,
photon ring) · 2 comet/strike (coma, tails, entry fireball, impact fireball and plume) · 3 star (disk,
limb darkening, halo, point at distance) · 4 pulsar (point, two beams along ±`u_beam_dir`, `u_pulse` flash). Shared inputs: `u_body_dir`, `u_body_d` (distance in body
radii; angular radius = asin(1/d)), `u_body_frame`, `u_px` (pixel size in radians), `u_time`, `u_glow`,
`u_glow_col`.

## 3. Adding a body kind: the pulsar example

(The pulsar is now implemented this way: `video.pulsar_beam`, the `pulsar` branch of `harbor_look`,
`u_body_type == 4` in `gl.py`, `scenario.pulsar_light` / `pulse_times`, and `audio._pulses`. Read them
as a worked example. Note the beam direction is computed in Python and passed as a uniform, so the
flash, the scene light and the audio tick all agree on when a beam crosses Earth.)

A pulsar is a neutron star: ~7 mi across, ~1.4 Suns, spinning, with two radiation beams sweeping
like a lighthouse. On screen, it's a tiny blinding point. Its beams sweep the sky, and its light
flashes across the scene at the spin rate (slowed down, then labeled TIME SLOWED). Gravity does
the rest: tides, Moon theft and impact all come from `mass_earths` automatically.

1. **Scenario:** `kind: pulsar` needs no loader change (`kind` is free text). Add fields to `Body` only
   if needed, e.g. `spin_hz: 1.0` (screen rate), `beam_color`. If you add a field, parse it in
   `load_scenarios` with a default.
2. **Shader:** in `gl.py`'s sky fragment shader, add `if (u_body_type == 4) { ... }` next to the star
   branch (`u_body_type == 3`). Reuse its disk and point code. For the beams, take the spin axis from
   `u_body_frame[1]`, a beam direction rotating around it with `u_time * spin`, and add a narrow cone
   glow `exp(-angle_to_beam² / w²)` along the great circle where the beam crosses the sky. Declare new
   uniforms, e.g. `uniform float u_spin; uniform vec3 u_beam_col;`.
3. **Bind the uniforms:** the renderer only sets uniforms it knows about. `Frame.sky_extra` is a dict
   of extras, but `Renderer.render` copies only the names listed in its two loops (floats, then vec3s,
   around `u_coma_r` / `u_tail_dir`). Add your names to those lists.
4. **Look:** in `video.py`, add a branch in the scene's look function (`harbor_look` handles stars and
   is the model; `forest_look` handles planets and black holes). Set `frame.body_type = 4`, put the
   uniforms in `frame.sky_extra`, and pulse the scene light with the spin:
   `frame.body_light = rgb(color, k * pulse(t))`. Tie sky and ambient colors to brightness, as the star
   branch does, so it doesn't tint the sky while it's still far away (a past mistake).
5. **Effects and audio** (optional): `Timeline.effects` already scales shake, flood and fire by tides.
   For a pulse sound, add a branch in `audio.soundtrack` (see the `black_hole` branches): e.g. a short
   tick at each beam pass.
6. **Test:** run `cd scripts && python3 -m pytest -q ../tests`, then render stills at the key times and look at
   them. Then add a test in `tests/test_engine.py` if the timeline logic changed.

Use the same pattern for any new visual: a branch in the shader, a few uniforms, and a look branch.

## 4. Other common changes

- **A body in a scene that doesn't support it** (a star in the forest): copy the `star` block from
  `harbor_look` into `forest_look`.
- **Body position:** `harbor.BODY_DIR` (az 0°, el 10°) and `forest.BODY_AZ_EL` (3°, 8.5°).
  Flybys and strikes take theirs from `Timeline.direction(t)`.
- **A new scene:** a class with `__init__(timeline, seed)`, `static` (Instances), `meshes` (dict, optional),
  `dynamic(t, fx) -> Instances`, `camera(t, fx) -> (pos, look, fov)` with a fixed position,
  `lights(t, fx, cam, power)`, `body_az_el` and `moon_az_el`. Register it in `video._scene_class` and
  `LOOKS`, and add an audio bed in `audio.soundtrack`.
- **Text layout:** the constants at the top of `overlay.py` (HOOK_TOP, HUD_X, SUB_CENTER...).
- **Readable subtitles on bright frames:** `Overlay._backing_alpha` measures the brightness behind a subtitle and fades in a soft dark backing above about 50% luminance (pulsar flashes, impact glow). Normal frames are untouched.

## 5. Performance and troubleshooting

- About 0.3 s per frame on CPU (Mesa llvmpipe), so 1,950 frames take 8–11 min. Stills take about 1–2 s each
  plus ~10 s of setup. Running two renders at once makes each one twice as slow.
- `moderngl.Error: ... EGL`: install `libegl1 libgl1-mesa-dri`. The context is created with
  `backend="egl"`, so no display is needed.
- The file is over the cap: `render` re-encodes automatically (`shrink_to_fit`). Busy particle
  scenes are the largest.
- A background render dies when the container restarts. Check that the output file exists and that
  `strip` shows 65.0 s before delivering; re-run if not. The `.part.mp4` file is the in-progress output.
- Banding in dark skies: keep `X264_PARAMS` (aq-mode 3) and the shader dither.

## 6. Street scenes (temperature mode)

`boardwalk.py` and `plaza.py` differ from the space scenes in three ways:
- **They own their look.** A scene with a `look(t, fx, pos, look_at, fov)` method returns its own `Frame`
  (sky palette, sun key light, shadows, fog), and `WhatIfScene` uses it instead of `LOOKS`.
- **They letter a sign.** `sign_point` (a world position) plus `sign_text(fx)`: `render_frames` projects
  the point with the frame's camera (`video.project`) and the overlay draws amber LED text there.
  `lens_frost(fx)` (0..1) fades ice crystals in from the frame edges.
- **Global material effects** in the object shader, set on the Frame: `dry` (greens fade to straw),
  `scorch` (everything chars in noisy patches), `boil` (water foams), plus the existing `frost`
  (white on upward faces) and `ice` (frozen water). These change thousands of static instances
  without rebuilding them.

Everything time-varying is keyed to `fx.temp_f` with `ramp(temp, a, b)`; people leave at
`timeline.time_at_temp(limit)` for a per-person limit.
