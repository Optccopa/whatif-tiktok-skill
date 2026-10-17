# What-if videos

A [Claude](https://claude.com) skill and a standalone Python renderer for 65-second vertical
**"What if…?" videos**: *What if Saturn crashed into Earth? What if a black hole swallowed us?
What if it got 10 degrees colder every second?*

![Six frames: Saturn's rings over a harbor, a black hole's disk over a forest, a burning boardwalk at 595°F, a frozen plaza at −365°F, a red supergiant filling the sky, the Moon about to hit](docs/showcase.jpg)

Each video is one locked shot of a calm place on Earth while something changes, and everything
on screen follows from real numbers: the body's distance, its tides, when it steals the Moon, when
the sea floods, the temperature at which water boils, paper ignites or oxygen turns to liquid.
Text carries the story (no voice-over), with a synthesized ambient soundtrack that follows the action.

- **Low-poly 3D**, rendered headless with OpenGL (moderngl). Works on a GPU, or on a CPU through Mesa.
- **1080×1920, 30 fps, 65 s, under 20 MB**, ready for TikTok, YouTube Shorts and Reels.
- **Five story modes:** impact, flyby, strike (a comet hits within sight), escape (Earth flung out of
  the galaxy) and temperature (the air heats or cools every second, or the Sun goes out).
- **Five scenes:** a dusk harbor, a golden-hour forest valley, a sunset carnival boardwalk, a riverside
  city plaza and a tropical beach (with a tsunami), each with crowds, lights and effects keyed to the physics.
  A lakeside template and a scene lab (beach, ski village, desert prototypes) give new scenes a head start.
- **22 ready-made scenarios**, from Saturn, Jupiter, Venus and the Moon to a pulsar, a white dwarf, the
  dinosaur-killing asteroid, a supermassive black hole, the red supergiant Stephenson 2-18, and "the Sun goes out".
- **Scenarios are YAML.** Most new videos are a new entry, with no code.

## Use it as a Claude skill

- **Claude Code:** copy the folder to `~/.claude/skills/whatif-videos` (all projects) or to
  `.claude/skills/whatif-videos` in a repository. Then ask for "a what if video about Venus crashing
  into Earth".

  ## Credits

- Fonts: [Source Serif 4](https://github.com/adobe-fonts/source-serif) (Adobe) and
  [Montserrat](https://github.com/JulietaUla/Montserrat), both under the SIL Open Font License;
  their licenses are in `whatif-videos/assets/fonts/`.

## License
MIT; see [LICENSE](LICENSE). The bundled fonts keep their own OFL licenses.
