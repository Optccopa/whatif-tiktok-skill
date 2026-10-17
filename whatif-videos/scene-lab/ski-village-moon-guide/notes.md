# moon-ski: the Moon crashes, seen from a ski village at dusk (stills only)

**What I built**
- A new `ski` scene (`skill/scripts/whatif/ski.py`). The camera is locked on a chalet balcony 22 m above a snowy village street. The street is lined with chalets that have lit windows, snowy roofs and chimney smoke, plus lamps, string lights and about 90 people, and it leads to a white church with a spire. A lit piste and chairlift run down the ski mountain on the right, a forested wall closes the left, and a low range ends the valley. The Moon rises in open sky over the valley end (az −3°, el 11.5°).
- A `ski_look` in `video.py`. It reuses the harbor's dusk palette, then adds pink alpenglow from the sun just set behind us, which fades into night. After that the Moon lights the snow, scaled by how many full moons bright it is (its apparent area). Registered in `_scene_class` and `LOOKS`.
- Story effects keyed to the timeline:
  - At 20–24 s people stop and stare at the Moon.
  - When the shaking starts, they run indoors (to chalet and church doors).
  - The lift stops and the lights cut out with the power.
  - Avalanche powder clouds slide down the visible slopes.
  - Chalet fires break out.
- An audio bed (`_ski`): church bell, street voices and lift hum, then screams, the avalanche roar and fire crackle.
- Small shared-engine changes, both backward compatible:
  - `props.Crowd` gained optional stop-and-stare.
  - `props.draw_flames` gained a `heat` knob.
- One new test. All 20 tests pass.
- Scenario: `moon-ski.yaml`. The physics is the same as the catalog `moon-crash`, but the script is rewritten so every line describes something visible from the mountains (no sea or coast lines). The new facts: brightness of 50/100 full moons, the GPS-orbit label, and the 22,000 mph impact speed (checked by free-fall calculation). `check` reports no warnings.

**Review passes:** 4 still passes after the harbor and forest reference sheets:
1. First look.
2. Fixed an avalanche that read as a white snake along the ridge, a brown eclipse-like halo on the Moon, and snow that was too hot under the subtitles.
3. Made avalanches visible by 46 s, when the line names them, and toned down fire lights that blew out the near chalets.
4. Removed avalanche puffs floating above the skyline.

After that I made a flame-color fix (the fires were white-hot cones), fixed exits so the crowd is indoors by 46 s, and rendered these final stills.

**What I'd still improve**
- The fires read as small orange cones. Smoke columns and a red glow on the snow around them would sell them better.
- The upper avalanche on the ski mountain looks like stacked discs. Puffs that fan out sideways and settle would look more natural.
- The balcony slab and railing take up much of the bottom third. A lower or thinner railing would show more of the street.
- The subtitles sit over bright snowy roofs at 4–20 s. They're readable, but a touch of backing or darker roofs in that band would help.
- Skiers and lift chairs are only a few pixels wide at 1–1.5 km. They read as motion only in the video.
- Not done because the task stops at stills: the full render and the audio check, and adding `ski` to the SKILL.md scene table once it ships.
