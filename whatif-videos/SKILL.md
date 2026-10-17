---
name: whatif-videos
description: Make 65-second vertical "What if…?" videos for TikTok, such as space disasters ("what if Mars crashed into Earth", "what if a black hole swallowed Earth") and everyday extremes ("what if it got 10 degrees hotter every second", "…colder every second"). They're low-poly 3D, rendered headless on the CPU, with a big serif hook, a distance readout, italic subtitles and a synthesized ambient soundtrack. Includes the full renderer, fonts, a 17-video catalog, the physics, the house style and the posting strategy. Use this skill whenever someone wants a what-if, space disaster, planet-crash, temperature, "what would happen if" or science-explainer short video, wants new scenario ideas, or wants a caption, posting schedule or cover advice for these videos, even if they don't say "skill" or "renderer".
---

# What-if videos

Each video is one locked shot of a calm place on Earth: a dusk **harbor**, a golden-hour **forest**
valley seen from a fire lookout, a sunset carnival **boardwalk** or a riverside city **plaza**.
Something changes, and everything on screen follows from real numbers. In the space videos, a body
approaches: its distance, its pull on our tides, when it steals the Moon, when the sea floods, the
impact. In the temperature videos, the air itself heats or cools every second, and each real
threshold shows up in the scene (water boils at 212°F, paper ignites at 451°F, oxygen liquefies at
−297°F). Text carries the story; there is no voice-over.

```
0.0–3.6 s   the hook ("What if {Saturn} crashed into Earth?"), also the cover frame
3.8–57 s    a subtitle every ~4 s; the left readout counts the distance (or temperature)
57 s        the event (impact, closest approach, horizon crossing...)
58–61 s     one final line, larger and centered
61.5–65 s   outro question + COMMENT BELOW
```

## Setup (once per machine)

```bash
pip install moderngl numpy pillow pyyaml          # Python 3.10+
apt-get install -y ffmpeg libegl1 libgl1-mesa-dri # headless OpenGL through Mesa (no GPU needed)
python3 <skill>/scripts/whatif_cli.py list        # should print the catalog
```

Everything runs through `scripts/whatif_cli.py` (call it by its full path from any directory).

## The workflow

Follow these steps in order. Rendering takes 8–11 minutes, so most of the work goes into the
cheap steps before it.

1. **Pick the mode and scene** (table below). Most requests are an existing mode with a new body:
   that's just a YAML entry, with no code.
2. **Write the scenario** in a YAML file. Use the bundled catalog `scripts/whatif/scenarios.yaml` when
   working in the repo that holds this skill. Otherwise use a local file passed with `--catalog my.yaml`;
   it can hold one or more entries in the same format. Read `references/scenario-authoring.md`
   for the fields, how to pin beats to distances and events, and the physics. Copy the closest
   existing entry and change it, rather than starting blank.
3. **Check:** `whatif_cli.py [--catalog f.yaml] check <id>` prints every beat with its time and the HUD
   at that moment, and warns about lines that are too fast, too long or too close together.
   Fix every warning unless you have a reason not to. Read the HUD column too: it must stay true
   for as long as it's on screen.
4. **Stills:** `whatif_cli.py stills <id> 2 20 35 48 55 59 63 --sheet sheet.png --out-dir stills`
   renders those frames in about 30 s. Look at the sheet. Check that:
   - the body is visible above the hills,
   - no text collides,
   - every line describes something the viewer can see at that moment,
   - the end card is clean.

   Iterate here, not after a render.
5. **Render:** `whatif_cli.py render <id> --out whatif-<id>.mp4` gives you 1080×1920, 30 fps, 65 s, AAC
   ambience, under 20 MB (`--max-mb`). Run it in the background and **render only one video at a
   time**. The renderer uses every CPU core; two renders at once each take twice as long, and both
   can be killed by a container restart.
6. **Verify:** `whatif_cli.py strip whatif-<id>.mp4` prints the duration, size and streams, and saves a
   frame strip. Look at it before delivering. Expect 65.0 s, video plus audio, and 10–17 MB.
7. **Deliver** the MP4 with its caption: `whatif_cli.py caption <id>` (format in `references/tiktok.md`).

When asked for several videos, queue them and render them one after another. Send each one as it
finishes, with its caption.

## Modes and scenes

| mode | story | distance model | examples |
|---|---|---|---|
| `impact` | it falls in and hits at `impact_s` | exponential: same time per halving | Saturn, Jupiter, Uranus, Neptune, black hole |
| `flyby` | closest at `closest_s`, gone by `impact_s` | log-distance, eased into closest | rogue gas giant, star |
| `strike` | a small body hits within sight, then the blast arrives | exponential, then 3 s straight through the air, then a countdown | comet |
| `escape` | Earth is thrown out of the galaxy | log clock from 1 hour to 30 million years | Earth ejected |
| `temperature` | the air heats or cools by a fixed amount every second | linear °F clock, held at absolute zero | hotter / colder every second |

| scene | look | supports |
|---|---|---|
| `harbor` | dusk bay, city lights, pier with Ferris wheel, balcony railing; tsunami, fires, power cuts | planets (rings), stars (`kind: star`) |
| `forest` | golden-hour valley, lake, snowy peaks, lookout railing; sunset into night | planets (rings, storm spot), black holes, strikes, escape |
| `boardwalk` | sunset carnival pier: crowds, striped stalls, string lights, Ferris wheel, lake, LED temperature sign | temperature (heat): people leave, grass dries, lake boils, power fails, popcorn pops, paper then wood ignite |
| `plaza` | riverside city square: fountain, truss bridge, brick buildings, tree, pigeons, subway, LED sign | temperature (cold): fountain and river freeze, sea smoke, dry-ice snow, liquid-air rain, black silent sky |

Body kinds: `planet` (oblate, banded or `surface: rocky`, optional rings and storm spot), `star` (glowing
disk; harbor only), `pulsar` (a blinding point with two sweeping lighthouse beams and flashes; harbor only),
`black_hole` (lensed shadow and disk; forest), `comet` (strike mode). Anything else, such as a white dwarf,
a magnetar or an asteroid swarm, needs engine work. Read `references/engine.md`,
which walks through adding a body kind.

Alternate scenes from video to video so a profile grid doesn't look like one video posted twice.

## The house style

These rules come from direct feedback. Breaking them got videos sent back, so keep them unless
the user changes them:

- **The camera never moves.** One locked viewpoint for the whole video. Shake rotates the view a
  little; the position stays put. No cuts, no pans, no zooms.
- **Clean over busy.** One composition and simple low-poly shapes. Leave space for the sky, where
  the action is.
- **Subtitles only, no TTS.** Italic serif, one idea per line, under ~70 characters, a new line every
  3.5–4.5 s. A gap of more than ~7 s with no new line feels dead.
- **Say only what's on screen.** Don't mention the Great Red Spot before it's visible, or the rings
  before you can see them. Re-time the line or cut it.
- **Every number is real.** Use NASA fact sheets for planets. Derive events from the physics in
  `references/scenario-authoring.md`. Round numbers honestly ("about 60,000 mph").
- **Readout labels stay true.** A label stays on screen until the next one replaces it, so word it
  to stay true as the distance keeps changing: "PAST THE NEAREST STAR", not "AS FAR AS…".
- **The body must be seen.** Place it high enough to clear the skyline, and make the hero moment
  fill the sky.
- **Temperature videos show each threshold.** When a line says "the lake boils", the lake must be
  boiling in that frame. Check the stills at every labeled temperature.
- **No watermark or handle** in the render (TikTok adds its own).
- 65 s, 1080×1920, under 20 MB.

More detail on the visual and sound language is in `references/style.md`.

## Writing a good script

The first 2 seconds decide whether anyone watches; the end decides whether they comment.

- **Hook:** "What if {X} crashed into Earth?" Braces mark the accent-colored words. Famous objects
  with a simple verb do best: "crashed into" beats "passed by" by a wide margin. For temperature
  videos: "What if it got 10 degrees {hotter} every second?"
- **Arc:** scale ("95 times heavier than Earth") → a surprising fact ("it would float in water") →
  distance milestones ("as close as Mars ever gets") → the physics turning ("its pull now matches
  the Moon's") → escalation (Moon stolen, sea drains, then floods, volcanoes) → the event → one
  final line that reframes it ("Saturn has no surface. Earth just sinks in.").
- **Outro:** a question that invites a comment, like "What should hit Earth next?"
- **Teaser** (caption line 2): point at the payoff without spoiling it: "Watch what happens to the
  Moon 🌙", "Wait until it leaves 👀".

## Reference files

- `references/scenario-authoring.md`: the YAML schema, beat pinning, events, physics formulas and
  ready-made facts. Read it before writing any scenario.
- `references/style.md`: the hook, readout and subtitle layout, scene palettes and soundtrack,
  plus a list of past mistakes to avoid.
- `references/tiktok.md`: the caption format, hashtags, posting schedule, covers, what tends to work,
  and the idea backlog.
- `references/engine.md`: how the renderer works and how to add a body kind, an effect or a sound.
  Read it only when a request needs something the YAML can't express.
- `tests/test_engine.py`: run with `cd scripts && python3 -m pytest -q ../tests` after any engine change.
