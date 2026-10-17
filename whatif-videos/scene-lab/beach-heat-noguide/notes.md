# Beach heat: notes at the stills stage

**What I built:** a new `beach` scene (`scripts/whatif/beach.py`). It's a tropical cove under a deep-blue afternoon sky, seen from a locked hotel-terrace camera. It has a sand heightfield with a curved waterline, a turquoise lagoon over a deep sea, a reef surf line, a jetty with four over-water bungalows, a lifeguard tower, a thatched beach bar with string lights and a neon sign, umbrellas and palapas, towels with paperbacks, loungers, beach balls and swim rings, kayaks, a catamaran, islands, palms framing both sides, a crowd including swimmers, and the LED temperature sign.

Every effect is keyed to °F:
- the crowd runs off between 95 and 150
- the palms dry from 110 to 260, then curl
- the sea foams at 213 (salt water) and steam rises
- the power dies from 255 to 290
- the floats swell, then burst between 312 and 348
- the loungers slump from 310 to 400
- the books ignite at 451
- polyester canopies melt and droop from 478 to 515
- wood smokes from 505
- everything wooden burns at 572 and chars, but the sand never chars

I registered it in `video._scene_class`. It has its own audio bed (`audio._beach`) with surf, gulls, crowd, a steel-pan tune that dies with the power, ball pops, boiling and fire. I pulled the shared wave, boil and burn code out into helpers, keeping the harbor and boardwalk random-draw order the same. I also made three small shared-engine tweaks:
- `draw_puffs` takes a height per source.
- Water uses emissive as light from a sandy bottom.
- GROUND with p2=1 opts out of scorch.

I added `test_beach_scene`, and all 20 tests pass. The scenario is `beach-heat.yaml` (id `hotter-every-second-beach`), and `check` gives no timing warnings.

**Review passes:** 5 rounds of stills.
1. The camera framed mostly empty, blown-out sand, and steam looked like giant cotton balls.
2. I fixed the framing and exposure, but the palms were out of frame and the fires were invisible in the haze.
3. Palms in frame, earlier smoke and char, bigger wood fires, and a flatter melted umbrella.
4. The steam became a solid wall.
5. I thinned the steam, softened the steam line to "rises", and added more sea mist.

**What I'd still improve:**
- Steam and wood smoke are still drawn as solid low-poly balls. They read as ping-pong balls, not vapor; a translucent or billboard mist would be better.
- The book fires at 451 are small glints at this distance.
- The loungers read as plain white blocks.
- The LED board is oversized compared with its lettering.
- The sun is behind the camera, so there's no visible sun disk in the sky.
- I haven't heard the audio yet. That needs a render.
