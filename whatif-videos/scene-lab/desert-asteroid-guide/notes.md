# Dinosaur asteroid: desert canyon overlook (stills only, no render)

**What I built.** A new `desert` scene (`skill/scripts/whatif/desert.py`). The camera is locked at a stone overlook wall on a canyon rim, looking down a red-rock canyon. The canyon floor sits 150 m below and carries a river and a two-lane highway with traffic: about 20 cars and trucks with headlights. They stop one by one about 1 to 3.5 s after the impact. About 2 km out the canyon opens onto a valley with buttes in the style of Monument Valley, then mesas fading into haze on the horizon. Ravens circle above the canyon. The aftermath (falling ejecta, the dust wall) works as in the forest. The dust wall stops 6 m short of the camera, so the railing stays visible at 52 s.

**Engine changes**, all in the copy:
- a `mat 9` desert-rock shader material in `gl.py`: sandstone strata on the cliffs, sand on the flats.
- `comet_look` in `video.py` can now take a per-scene `strike_look` palette, sun direction, fog and shadow box. The forest's values are the default, so the forest renders the same as before.
- the scene is registered in `video._scene_class` and `LOOKS`.
- a `_desert` audio bed in `audio.py`: dry wind, a hawk cry, crickets, and car hum that ends when the traffic stops. The strike layers come after it.

All 19 tests pass. The scenario is `dinosaur-asteroid-desert.yaml`, a copy of `dinosaur-asteroid` with these changes:
- a new line at 40.6 s: "The ground shakes. On the highway, every car stops."
- the crater line moves to 44.7 s and the ejecta line to 47.6 s.
- the final wind line now says "tear through the canyon" instead of "flatten the forest".

`check` shows no timing warnings.

**Review passes:** 5, compared against the forest version at the same times (in `review/`):
1. The overlook wall covered the bottom 30% of the frame, and the rail ran behind the subtitles.
2. I lowered and moved the wall, softened the cliffs, and made the headlight glows grow with distance.
3. The wall still had a large flat capstone, and no cars showed in the canyon. They were all bunched 3 to 6 km out.
4. I broke the capstones into separate blocks, kept the traffic in the first 3.2 km, and moved the rail posts off the road.
5. Final check after adding the audio bed.

**What I'd still improve:**
- The bench cliffs have a sawtooth edge where the 10 m terrain grid cuts across the curving canyon. A finer grid along the walls, or snapping the mesh to the walls, would fix it.
- The parapet is still chunky at 2.6 m.
- Cars only read as headlight dots past about 600 m.
- The 40 s frame is washed white by the existing fireball glow. The same happens in the forest version.
- The 52 s frame is a flat dust screen, as in the original.
- Spot fires where ejecta land would back up the "glowing hot" line.

**Rendering fix (after the blind test):** the camera moved out on a platform past the rim (z −24) behind a thin pipe railing, replacing the slab wall and the bare ground in the foreground. The terrain grid is now 2.5 m near the cliffs. The sand/rock blend in the shader is wider (smoothstep(0.06, 0.75, steep)), because a tight band flipped color facet by facet and made the cliffs look like sawtooth. The shadow box is tightened to the visible canyon.
