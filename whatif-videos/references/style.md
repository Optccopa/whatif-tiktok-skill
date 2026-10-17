# Visual and sound style

The look is modeled on the "Ink Files" what-if videos: a calm low-poly diorama, serif typography,
and text that reads like a documentary. Calm is the point: the horror comes from the sky changing
while the view stays put.

## Frame layout (1080×1920; the overlay handles all of this)

| element | where | font |
|---|---|---|
| Hook | top, y≈300, centered, 2 lines, 80–88 px | Source Serif 4 semibold; the {braced} words in the accent color |
| HUD label | left, x 92, y 858 | Montserrat bold 25 px, letter-spaced, e.g. DISTANCE TO SATURN |
| HUD value | under the label | Source Serif 4, 88 px: "746,000,000 mi", "2.46 trillion mi", "388 light-years" |
| HUD details | under the value | Montserrat 22 px, soft white: tide factor, the current milestone, time and temperature (escape) |
| Subtitles | centered at (520, 1214), max 860 px wide | Source Serif 4 italic 54 px; the final line is 60 px at y 905 |
| Outro | center | serif 76 px question + accent COMMENT BELOW |

All text has a soft dark shadow and stays inside TikTok's safe area: nothing in the top ~220 px, the
right ~150 px (buttons) or the bottom ~420 px (caption). The hook doubles as the cover frame (frame 0),
so it must read at thumbnail size.

## Camera

One locked camera per scene, behind a wooden railing that frames the bottom of the shot:
- **harbor:** a hillside balcony over the bay at dusk. The city is lower right, the pier and Ferris wheel
  center, boats and people below. The body rises at azimuth 0°, elevation 10°.
- **forest:** a fire lookout on a ridge, the lake below, snowy peaks ~5–7° high on the skyline. The body sits
  at azimuth 3°, elevation 8.5°, so it clears the peaks.

- **boardwalk:** high over the start of a carnival pier, looking down it to a Ferris wheel on the lake
  under a low sun ahead-right (long shadows toward us). Stalls with pastel striped awnings line both
  sides; zigzags of string lights run overhead; the LED sign stands lower right.
- **plaza:** above a paved square, looking across a railing to the river, a blue truss bridge (left)
  and brick buildings with pale towers behind. The fountain sits lower left, the tree and subway right.

Shake only rotates the look direction, scaled by `Effects.shake`. The position never changes.
The user called out camera moves twice ("The pov shouldn't move location"), so treat this as a hard rule.

## Light and color

- harbor: deep-blue zenith, a pink twilight band, warm windows. It warms toward the body's
  `light_color` as it grows, then turns red-orange with fires and lava. Power flickers, then cuts after the flood.
- forest: golden hour (sun 7° up). In impact mode, the sun sets between 28 s and 46 s and night falls
  by ~50 s, so the climax is lit by the body itself. Stars come out at night.
- strike: golden hour holds, then a flash brighter than the sun, a burning sky, a mushroom plume and
  a dust wall rolling in.
- escape: a clear day dimming to black. The Sun shrinks to a star, frost and ice spread, the air
  snows out, the Milky Way band appears, and the spiral galaxy is seen from outside at the end.
- Rendering: HDR, bloom on bright things only, an ACES tonemap, a light vignette and dither so dark
  skies don't band.

## Sound (synthesized; no music unless the user supplies tracks with `--music-dir`)

The ambience matches the scene and builds with the danger:
- harbor: waves, gulls, crickets, wind; a low drone that rises with the tide factor; the sea pulling
  back, then a flood roar; a hiss for the rings; a quake rumble with thumps; the impact boom, then silence.
- forest: birds (song in the day, a flock's wingbeats around 15–22 s), crickets after sunset, wind
  in the trees, a rumble and creaking as the shake builds. A black hole cuts to dead silence at the horizon.
- strike: a silent flash (sound is slower than light), debris whistles, then the blast's roar and boom.
- The mix is normalized so the calm section (2–14 s) sits at −26 dB RMS, then soft-limited:
  quiet enough at the start that the climax hits harder.

## Text voice

Short, concrete, second person plural ("us", "we"). No exclamation marks; the facts do the work.
Present tense. Numbers spelled the way people say them ("95 times heavier", "hundreds of feet high").
Use "..." to split a two-beat moment across lines ("The sea drains away from the shore..." /
"...then comes back hundreds of feet high.").

## Past mistakes (don't repeat)

| what happened | fix |
|---|---|
| Multi-shot camera cuts felt busy | one locked shot |
| The camera drifted or dollied | fixed position; shake is rotation only |
| Trees and cottages right in front of the camera blocked the view | keep the foreground clear except for the railing |
| Smoke drawn as blobs looked fake | particles and soft shader effects instead |
| Water showed a grid pattern | directional waves |
| The comet tail looked like a laser beam; the coma whited out the screen | a curved, fading tail; a coma capped in size |
| The black hole was hidden behind the peaks | lower peaks, raise the body to 8.5° |
| The red star tinted the whole sky pink while still far away | scale its glow by its actual brightness |
| A space before the "?" after a highlighted word | the overlay keeps punctuation tight |
| The galaxy was too big and bright, with a seam in the Milky Way band | dimmer, smaller, seamless coordinates |
| Audio levels jumped from video to video | normalize on the calm section |
