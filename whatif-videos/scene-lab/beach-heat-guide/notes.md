# hotter-beach: notes

**What I built:** a new `beach` scene (`skill/scripts/whatif/beach.py`): a tropical bay at sunset. The camera is locked 16 m up. The sea is on the left, with a sun glitter path, waders, buoys, sailboats, a palm island and purple volcanic peaks. The sand is on the right, with umbrellas, towels, loungers, kayaks, beach balls, ~200 people, a thatched beach bar with string lights and a lifeguard tower. Palms, a lamp-lit promenade and pastel hotels line the back of the beach. A curving shoreline leads to the hero landmark: a red-and-white lighthouse on a green headland. Every effect is keyed to °F:
- people leave at 110–175°F
- palms dry, brown and wilt, then burn at 572°F
- the sea boils from 213°F (seawater boils about 1°F above fresh water), and the steam haze thickens
- the power fails at 255–290°F
- beach balls burst and the plastic kayaks and loungers slump
- the paperbacks on the towels light at 451°F, then the umbrellas and the thatch, then the wood at 572°F
- the sand never chars, which the last two lines depend on.

Wiring:
- registered in `video._scene_class`
- a one-line shader guard in `gl.py`, so ground with p2=1 skips `scorch`
- an `_beach` audio bed in `audio.py`: waves, gulls, crowd, a steel-pan tune that dies with the power, the sea boiling, fire
- a test in `tests/test_engine.py` (20/20 pass)

The scenario is in `beach-heat.yaml` (id `hotter-beach`). `check` gives no timing warnings.

**Review passes:** 4 after the first render, compared with the boardwalk reference (`review/ref.png`, `review/p1`–`p4`):
1. Made the white foam strip that looked like a curb thinner, made the far peaks smaller and more purple so the sun sits in open sky, and cut the big steam blobs that covered the HUD down to a low mist.
2. Made the paperback and umbrella flames bigger, because the 451°F line wasn't readable.
3. Made people leave later, so the 10 s "everyone runs" line shows people running.
4. Moved the kayaks into view and made the beach balls bigger, so the 24.5 s line is visible.

**Still to improve:**
- The boiling foam on the water near the camera looks blocky. This is the shared shader, and the boardwalk has the same look.
- The steam is still sphere puffs; a shader mist would look better.
- The sea reads pale lilac and gold more than turquoise at this sun angle.
- "Lead melts" at 621°F is a HUD label only, with nothing on screen for it.
- Not rendered yet. Once you approve the look, run `render` and add `beach` to the scene table in SKILL.md.
