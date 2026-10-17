# Dinosaur asteroid, desert canyon version: stills before rendering

**What I built**
- A new `canyon` scene (`skill/scripts/whatif/canyon.py`). The camera is locked at a pipe railing on a canyon rim, 200 m up. You look down a red-rock canyon with layered cliffs on both sides and sandstone buttes. A straight two-lane highway runs to the vanishing point, with 140 cars, pickups and semis driving both ways. They brake hard right after the flash, and their brake lights come on. A muddy river winds beside the road, and two hawks circle. The lighting is golden hour.
- A new `DESERT` shader material (id 9 in `gl.py`): sandstone strata, desert-varnish streaks and sand with sage on the flats. The scene is registered in `video.py`. It reuses the forest's strike sky and look: flash, fireball, mushroom plume, falling ejecta and the dust wall.
- A canyon audio bed (`audio._canyon`): highway hum and cars passing until the traffic stops, hawk cries, sand hiss, then the shared strike sounds. There's also one terrain test, and all 20 tests pass.
- The scenario is `dinosaur-asteroid-canyon.yaml`. Same physics and numbers as the catalog entry. The new lines are tied to the highway: "about 600 times faster than the cars below" (45,000 / 75 mph), "Down on the highway, every car has stopped" at 40.6 s, and the winds "sweep up the canyon". The crater line moved to 47.9 s. `check` reports no timing warnings.

**Review passes:** 8 rounds of stills (in `review/p1`–`p8`), then the final set. Fixes along the way:
- the parapet was too tall, so I swapped it for a see-through railing
- the canyon read as a flat plain, so I narrowed it and raised the side rims
- a bug in the far wall buried the floor and hid the road
- the river was out of frame, then clamped flat, then had sawtooth banks

**What I'd still improve**
- The cars are honest scale, so they're only a few pixels. You can see the traffic stop in motion, but in a still you can't tell which lights are brake lights. They could be scaled up or given more light.
- The river reflects pale blue sky at this grazing angle, not brown. It pulls the eye left.
- t52 is solid dust, the same as the forest version, because the air blast covers everything. I could cap the dust front just past the railing so the railing stays visible in the brown-out.
- The buttes are plain 5-sided prisms. Talus skirts with some irregular faces would look better.
- The mushroom column at about 46 s is very wide and blocky. That comes from the shared shader.
