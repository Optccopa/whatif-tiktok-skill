# Scene lab

Scenes agents built during skill tests, kept so later videos can start from them instead of from scratch.
They are not wired into the engine yet. Each folder holds:

- the scene module (`beach.py`, `ski.py`, ...),
- the scenario YAML it was tested with,
- `engine.patch`: every engine change that run made (diff against the engine at the time; the scene module
  is in it too),
- `notes.md`: what the agent built and what it would still fix,
- `preview.png`: the frame at 20 s.

To use one: copy the module into `scripts/whatif/`, apply the shared-file hunks from `engine.patch` that
it needs (audio bed, materials, `video.py` registration; scenes built on `kit.Scene` need none), run the
tests and render stills. Then move it out of the lab: keep the module in `scripts/whatif/`, add a test, and
list it in `SKILL.md`.

`-guide` folders were built with the new-scene guide, and `-noguide` folders without it. In the blind test,
the guide versions of the beach and the ski village won. The desert lost because of rendering artifacts:
sawtooth terrain and a slab foreground.
