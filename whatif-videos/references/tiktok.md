# TikTok: captions, posting and what has worked

These videos are built for TikTok (and work as Shorts and Reels): subtitles only, no voice-over.

## Caption format (generate it with `whatif_cli.py caption <id>`)

```
What if Saturn crashed into Earth? 🪐
Watch what happens to the Moon 🌙
What should hit Earth next? 👇
#whatif #fyp #saturn #space #astronomy
```

1. The title plus one emoji (🪐 planets, 🌌 galaxy or deep space, 🕳️ black holes, ☄️ comets, ⭐ stars).
2. The teaser: point at the payoff without giving it away.
3. The question that matches the outro, plus 👇, to pull comments.
4. Hashtags: always `#whatif #fyp`, then 3 that match the video (`#saturn #space #astronomy`, `#heatwave #temperature #science`). The CLI takes the first 3 of the scenario's `hashtags` list; change `CAPTION_TAGS` in `whatif_cli.py` for your own set.

## What tends to work

From early posts with this format (small samples; check your own analytics):
- **"X crashed into Earth"** with a famous body pulls far more views than "passed by". Prefer impacts of
  well-known objects: the Moon, Mars, Jupiter, Saturn, the Sun.
- **Don't post several videos at once.** TikTok tests each post on a small audience first; a batch splits
  that test.
- **Make covers distinct.** If every cover is the same calm first frame, the profile grid doesn't show what
  each video is. Pick the climax frame as the cover in the TikTok editor, or export one
  (`whatif_cli.py stills <id> 55`).

## Posting schedule

- **One video a day**, at a consistent time, early evening for the audience (about 6–9 pm). Two a day is fine
  if they're at least 6–8 h apart. Once your account has data, use TikTok Studio → Analytics →
  Followers for the real active hours.
- Keep a **buffer of 3–5 rendered videos**. Each takes ~10 min to render, so production is never the bottleneck.
- Post the closest follow-up to a hit next (after a Saturn video, Jupiter) to test whether the format,
  not luck, drove the views.
- Pin the best performer to the profile.
- After ~2 weeks, compare views by subject and mode, and make more of whatever wins.

## Idea backlog (ranked by expected pull)

Impacts: the Moon crashing into Earth · the Sun swallowing Earth · Venus · a neutron star or pulsar ·
a white dwarf · Mercury · Pluto · an asteroid (strike mode, like Chicxulub).
Temperature: 10 degrees hotter / colder every second (made) · 100 degrees per second · the Sun goes out · Earth gets 1% closer to the Sun every second.
Other: Earth gets rings · Earth stops spinning · a second Moon · a magnetar flyby · the Sun becomes a red giant.

Videos already made: saturn-crash, rogue-gas-giant, black-hole, comet-strike, uranus-crash,
neptune-crash, star-flyby, jupiter-crash, earth-ejected, mars-crash, pulsar-crash,
hotter-every-second, colder-every-second.
