# moon-crash-village: notes

**What I built:** a new `village` scene (`scripts/whatif/village.py`). It is a snowy ski village at dusk, seen from a chalet balcony. It has a heightfield valley under snow with rocky steep faces, chalets with snowy roofs and lit windows, a church steeple and street lamps. Above the village there is a ski mountain with floodlit pistes and a moving gondola, and a forested peak opposite it.
- **Registration:** the scene is registered in `video._scene_class` and `LOOKS` through a new `village_look`. In that look the alpenglow slides off the peaks as the Sun sets behind us. Night falls, and then the snow brightens again under the falling Moon's light. That light is computed from its real apparent area: up to about 2,300 full Moons at contact.
- **Shaders (`gl.py`):** I added two effects.
  - `u_avalanche`: powder streaks pour down the slopes, top first.
  - `u_crack`: glowing seams on a rocky body once it is inside Earth's Roche limit, at a gap of 6,371 mi.
- **Audio:** a `village` bed in `audio.py` with dusk church bells, a gondola hum that dies with the power, and an avalanche roar.
- **Scenario:** in `moon-crash-village.yaml`. It passes `check` with no warnings, and the physics behind each number is noted in its comments. All 19 existing tests still pass.

**Review passes:** 2 full contact-sheet passes (9 frames each), plus the final stills.
- **Pass 1 found:**
  - The valley walls grew without limit and filled the sky.
  - The forest was a uniform carpet of trees.
  - The snow was too bright for white subtitles.
  - The chimney smoke looked like stacked balls.
  - Fog washed out the ending.
  - The lift sat in TikTok's button column.
- **Pass 2:** fixed all of these. The only later tweaks were faster crack glow and dimmer piste lights.

**What I'd still improve:**
- The avalanches read as soft light bands more than as rolling powder clouds. Billowing powder at the foot of the slopes would help.
- The near slope between the railing and the village is a plain white field.
- The scene brightens a lot in the last 10 s, so subtitles depend on the overlay's dark backing.
- I'd check frames at 44–48 s for the crack timing against the "tearing apart" line, and listen to the bells and avalanche mix in a real render.
- The scene only supports impact and flyby bodies, not strike, escape or temperature modes.
