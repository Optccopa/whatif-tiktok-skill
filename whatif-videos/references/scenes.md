# Building a new scene

Read this before writing a new scene module. A new scene is the riskiest change in this skill.
The physics and text come out right almost by default. A scene can render correctly and still look
like a programmer's test level: one flat plane, grey boxes, an empty sky. Viewers decide in the
first second whether the video looks like the series they follow, so the scene's look matters as
much as its facts.

Contents: 1. What makes the existing scenes work · 2. Recipe · 3. Keep the scene · 4. Review loop · 5. Gotchas

## 1. What makes the existing scenes work

Open `boardwalk.py` (heat) or `harbor.py` (space) next to this. The same choices repeat in every scene:

- **One recognizable place with things that can change.** A boardwalk has people who can leave, grass
  that can dry, a lake that can boil and wood that can burn. Choose a place where the story's
  thresholds each have something to act on: water, plants, snow, people, lights, a landmark.
- **A leading line to a hero landmark.** The boardwalk runs straight toward a Ferris wheel. The plaza
  looks across a river to a truss bridge. The harbor looks over a bay to a pier and a wheel. The eye
  enters at the bottom and travels to the landmark, and then to the sky above it, where the action is.
- **Depth in four layers.** These layers are what make simple low-poly shapes read as a rich world:
  | layer | distance | boardwalk example |
  |---|---|---|
  | foreground | 0–30 m | planks with STRIPES, a bench, the sign post; nothing tall |
  | midground | 30–150 m | stalls, ~230 people, string lights, tents, the wheel |
  | background | 300–1,500 m | a low far-shore city (`BUILDING` with lit windows), sailboats |
  | far | 1,800–3,000 m | soft hills, colored between the sky's `mid` and `horizon` |
  The far layers fade toward the horizon color (aerial perspective) through `fog_density`
  (about 0.0007) and through their own hazier, cooler colors.
- **One time of day, low sun.** Golden hour, sunset or dusk. The sun 5–8° up throws long shadows and
  warm rim light, and the sky is a 3-stop gradient (`zenith`, `mid`, `horizon`). Avoid noon light,
  which flattens low-poly shapes.
- **A small, deliberate palette.** Each kind of thing picks from a list of 3–7 hex colors with
  `rng.choice` (see `props.AWNINGS`, `SHIRTS` and the hill colors). Colors are pastel and saturated.
  Whites are warm (`#f2efe8`), darks are blue-black (`#26262c`), and there is no large pure grey.
- **Density and variation.** Hundreds of instances, each with jittered size, yaw and color: about
  40 stalls, 230 people, 120 far buildings, 22 hills, 34 trees and 26 bushes on the boardwalk.
  Instancing makes this cheap: thousands of objects cost almost nothing on a GPU.
- **Small lights everywhere.** String lights, lamp globes, lit windows and ride bulbs (`GLOW`,
  emissive 4–9, `p2=1` so they follow `power`). They give the frame sparkle under bloom, and when the
  power fails the blackout reads instantly.
- **Something moving in the calm part.** A strolling crowd (`props.Crowd`), a turning wheel, bobbing
  boats, pigeons, waves. Without motion the first 20 s look like a still image.

## 2. Recipe: start from the template, change the layout

The kit (`scripts/whatif/kit.py`) already does the plumbing: the camera, the sky and light from a palette,
shadows, registration and sound. A new scene is its layout and its effects.

1. **Copy the template.** Run `whatif_cli.py new-scene beach`. It writes `whatif/beach.py`, a working copy
   of `lakeside.py` (a lakeside park at sunset, with heat and cold effects) with the class renamed. Set
   `scene: beach` in your YAML and it renders right away; nothing else needs registering. Render stills
   first to see the starting point, then change it piece by piece and keep it rendering. Rewrite the
   docstring first: what's where, in meters, and where the camera looks.
2. **Pick the look with class attributes** instead of writing a `look()`:
   - `palette`: `golden`, `sunset`, `dusk` or `day` (see `kit.PALETTES`), or a dict of your own.
   - `sun_az`: the sun's azimuth.
   - `camera_at = kit.framing(height, target)`: the house framing, fov 60, 15–40 m up, aimed slightly
     down at a landmark 120–600 m away. The camera never moves.
   - `shadow_center`, `shadow_size`: the shadow box, just covering the playfield.
   - `landmarks`: the points `stills` checks against the text zones.
   - `sound_bed`: an existing ambience (`boardwalk`, `plaza`, `harbor`, `forest`).
   - For space stories, set `space_look = "forest"` or `"harbor"`. The sky, the body, strikes and
     escapes then come from those looks, and the class supplies only layout and effects.
3. **Build in `build(rng)`, back to front.** Start with `kit.ground`, then `kit.far_hills` and
   `kit.far_city` (pass the palette's `hills` and `city` colors), the landmark, the leading line, the props
   and the people. Only then add `kit.railing` if you want a near edge. Use `props.py` before writing new
   meshes: `block`, `stall`, `round_tree`, `lamp_post`, `string_lights`, `bench`, `circus_tent`,
   `sign_post`, `Crowd`, `draw_ferris_wheel`, `draw_balloons`, `draw_pigeons`, `draw_flames`,
   `draw_puffs`. Use materials for texture: `STRIPES`, `GROUND`, `BUILDING` (`p1` = lit windows),
   `WATER`, `GLOW`, `FLAME`.
4. **Key the story in three small methods**, as `lakeside.py` does:
   - `sky(t, fx)`: `kit.blend(base, (HOT, k1), (SMOKE, k2), ...)`, which shifts the palette as the
     story turns.
   - `effects(t, fx)`: the global shader effects (`dry`, `boil`, `scorch`, `frost`, `ice`). They change
     every instance at once.
   - `dynamic(t, fx)`: water level, the crowd, leaves, steam and flames.

   Use `ramp(fx.temp_f, a, b)` at the real thresholds (see the table in `scenario-authoring.md`). Space
   scenes use `fx.water`, `shake`, `fire`, `power`, `lava`, `look` and `flee`. A scripted line with no
   visible change needs an effect, or the line goes.

   For a coast in a planet crash, copy the tide from `beach.py` instead of rebuilding it:
   - `seabed_y` and `_seabed_mesh`: a sloping seabed with rocks and tide pools, so a receding sea
     visibly exposes something.
   - `sea_level`: the water drains with `fx.water`.
   - `wave_front`, `wave_meshes` and `_draw_sea`: a ragged breaking wall with a dark face, a curling
     white lip, whitewater at the base and spray, which rolls in and floods.
   - The crowd: `Crowd` `stop`/`face` for staring, then `leave` at `fx.flee`.

   Draw a big surface like that wall as a SOLID color with a little emissive, not `WATER`; the water
   shader assumes a flat surface.
5. **Only if a new sound is really needed**, add a bed to `audio.soundtrack` and name it in
   `sound_bed`. Reusing the closest bed is usually fine.

`boardwalk.py`, `plaza.py`, `harbor.py` and `forest.py` predate the kit. Read them for ideas, but start
new scenes from the template.

## 3. Keep the scene

A scene you build is an asset for later videos. Keep `whatif/<name>.py` in the skill, add a short test to
`tests/test_engine.py` (copy `test_lakeside_scene`), and add the scene to the tables in `SKILL.md`.

## 4. Review loop (this is where the quality comes from)

`stills` takes a reference scenario and puts it in a second row of the sheet. It also prints a `WARN`
line when a landmark, the sign or the sun or body lands under the hook, the readout, the subtitles or
the TikTok buttons, or goes off screen.

```bash
whatif_cli.py --catalog my.yaml stills my-id 4 20 40 52 --ref hotter-every-second --sheet sheet.png
```

Open the sheet and critique your scene honestly against this list. Fix the worst problem,
re-render (each pass takes seconds) and repeat. Plan on at least three passes. A first pass that
looks fine almost always isn't.
- Placed next to the reference, would it pass as a frame from the same series?
- Is there one clear landmark and a line leading to it?
- Is about 40% or more of the frame sky, with the body or sun in open sky?
- Are all four depth layers there, with the far ones hazier and cooler?
- Is there any large flat single-color surface? Break it up with a material, props or a color change.
- Are there small lights, and are there people readable as people (~1.7 m)?
- Is the foreground clear, with nothing big within ~15 m of the camera?
- Does anything float, clip through the ground or stop at a visible edge of the world? Make the ground
  and water planes huge (±900 m and more), and make sure the far hills cover the horizon.
- At each scripted threshold, does the frame show what the line says?
- Zoom in on a full-size still (crop at 100%). Are there sawtooth edges, stair-stepped cliffs, or big
  flat slabs in the foreground? Rendering artifacts lose a side-by-side even when the composition wins.

## 5. Gotchas we've hit

- **Shaders** (strict drivers like AMD on Windows): don't name a GLSL function after a built-in
  (`noise1`–`noise4`; ours is `hnoise2`). Guard every division and `normalize()` that can reach
  zero. A NaN renders as a black block (the asteroid with `u_tail_len = 0`).
- **Terrain:** a heightfield on a coarse grid (10 m) turns steep cliffs into sawtooth. Build cliffs
  from stacked, slightly jittered slabs (`block`, `prism`, `hull`) for the strata, or use a grid of
  2–3 m where the ground is steep. In the shader, blend materials by slope over a wide band
  (`smoothstep(0.06, 0.75, steep)`): a tight band flips color from facet to facet and looks like sawtooth even on
  a fine mesh. Keep the foreground edge thin: a railing, with no wall or cap slabs
  filling the bottom of the frame, and nothing poking into the HUD or subtitle zones.
- **Stateless frames:** every frame is a pure function of `t`. Draw randomness in `__init__` from
  the seeded `rng`, or from a fixed-seed `np.random.default_rng(k)` inside `dynamic()`, so stills
  match the video.
- **`ramp(x, a, b)`** works for falling ranges too (a > b).
- **Files:** read and write text as UTF-8 (`encoding="utf-8"`). On Windows, set
  `PYTHONIOENCODING=utf-8` before running the CLI, because the captions contain emoji.
- **Tests:** run `cd scripts && python -m pytest -q ../tests` after any engine change.
