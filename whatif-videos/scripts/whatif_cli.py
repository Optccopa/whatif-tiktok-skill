#!/usr/bin/env python3
"""What-if video tool. Run with the skill's scripts/ folder as the working directory or via its full path.

  whatif_cli.py list                                  scenarios in the catalog(s)
  whatif_cli.py check   <id>                          validate, print the beat sheet and timing warnings
  whatif_cli.py stills  <id> 5 30 50 [--sheet a.png]  render a few frames (fast) and tile them
  whatif_cli.py render  <id> [--out x.mp4]            full 65 s MP4 with sound, under --max-mb
  whatif_cli.py strip   <video.mp4> [5 30 50 62]      frame strip of a finished video, for review
  whatif_cli.py caption <id>                          the TikTok caption

Add --catalog my.yaml (repeatable) to load extra scenarios on top of the built-in catalog.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from whatif.scenario import CATALOG_PATH, LINE_MAX_SECONDS, Timeline, load_scenarios  # noqa: E402

SECONDS = 65.0
CAPTION_TAGS = ("whatif", "fyp")  # then the scenario's first 3 matching hashtags


def catalog(extra: list[str]) -> dict:
    out = load_scenarios(CATALOG_PATH, SECONDS)
    for path in extra or []:
        for sid, s in load_scenarios(Path(path), SECONDS).items():
            out[sid] = s  # a local file can override a built-in id
    return out


def pick(args):
    scenarios = catalog(args.catalog)
    if args.id not in scenarios:
        sys.exit(f"unknown scenario {args.id!r}; one of: {', '.join(scenarios)}")
    return scenarios[args.id]


def cmd_list(args):
    for sid, s in catalog(args.catalog).items():
        print(f"{sid:22} {s.mode:7} {s.scene:7} {s.title}")


def cmd_check(args):
    s = pick(args)
    tl = Timeline(s, SECONDS)
    print(f"{s.id}: mode={s.mode} scene={s.scene} story ends at {s.impact_s:.1f}s, outro at {tl.outro_start:.1f}s\n")
    warnings = []
    lines = tl.lines()
    for b in s.beats:
        hud = tl.readout(min(b.at + 0.01, tl.hud_end))
        print(f"{b.at:6.1f}s  {b.label or '':34} {b.line}")
        print(f"         HUD: {hud[1]}  {' / '.join(hud[2])}")
    if s.final_line:
        print(f"{tl.T + 1.2:6.1f}s  {'(final line)':34} {s.final_line}")
    for a0, a1, text in lines:
        shown = a1 - a0
        words = len(text.split())
        if shown < 1.8:
            warnings.append(f"{a0:.1f}s on screen only {shown:.1f}s: '{text}' (next line too soon)")
        if words / max(shown, 0.1) > 4.6:
            warnings.append(f"{a0:.1f}s {words} words in {shown:.1f}s is too fast to read: '{text}'")
        if len(text) > 70:
            warnings.append(f"{a0:.1f}s {len(text)} chars; keep lines under ~70: '{text}'")
    if lines and lines[0][0] < 3.6:
        warnings.append("first line starts before the hook has faded (3.6s)")
    for (a0, _, _), (b0, _, _) in zip(lines, lines[1:]):
        if b0 - a0 > LINE_MAX_SECONDS + 3.5:
            warnings.append(f"{a0:.1f}s→{b0:.1f}s: {b0 - a0:.1f}s with no new subtitle (feels dead on TikTok)")
    print("\n" + ("\n".join("WARN " + w for w in warnings) if warnings else "no timing warnings"))


def cmd_stills(args):
    from PIL import Image
    from whatif.video import render_whatif_stills

    s = pick(args)
    out = Path(args.out_dir)
    t0 = time.time()
    paths = render_whatif_stills(s, args.times, out, seconds=SECONDS, seed=args.seed)
    print(f"{len(paths)} frames in {time.time() - t0:.0f}s")
    for p in paths:
        print(p)
    if args.sheet:
        tile(paths, Path(args.sheet))
        print(args.sheet)


def tile(paths: list[Path], dest: Path, scale: float = 0.3) -> None:
    from PIL import Image

    ims = [Image.open(p) for p in paths]
    w, h = int(ims[0].width * scale), int(ims[0].height * scale)
    sheet = Image.new("RGB", (len(ims) * (w + 6) - 6, h), (255, 255, 255))
    for i, im in enumerate(ims):
        sheet.paste(im.resize((w, h)), (i * (w + 6), 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest)


def cmd_render(args):
    from whatif.video import render_whatif

    s = pick(args)
    out = Path(args.out or f"whatif-{s.id}.mp4")
    max_bytes = int(args.max_mb * 1e6) if args.max_mb else None
    render_whatif(s, out, fps=args.fps, seconds=SECONDS, preset=args.preset, crf=18, music_dir=args.music_dir or "",
                  max_bytes=max_bytes, seed=args.seed)
    print(out.resolve())


def cmd_strip(args):
    video = Path(args.video)
    times = args.times or [5, 30, 50, 62]
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for t in times:
            p = Path(tmp) / f"{t:05.1f}.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", str(video), "-frames:v", "1", str(p)],
                           check=True)
            paths.append(p)
        dest = Path(args.out or video.with_suffix(".strip.png"))
        tile(paths, dest, 0.25)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-of", "csv=p=0",
                            str(video)], capture_output=True, text=True).stdout.strip()
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                              str(video)], capture_output=True, text=True).stdout.split()
    duration, size = probe.split(",")
    print(f"{video.name}: {float(duration):.1f}s, {int(size) / 1e6:.1f} MB, streams: {', '.join(streams)}")
    print(dest)


def caption_text(s) -> str:
    emoji = {"planet": "🪐", "black_hole": "🕳️", "comet": "☄️", "star": "⭐", "pulsar": "⭐"}.get(s.body.kind, "")
    if s.body.is_moon:
        emoji = "🌕"
    if s.mode == "escape":
        emoji = "🌌"
    elif s.mode == "temperature":
        emoji = "🌑" if s.heat.end_years else "☀️" if s.heat.sun_factor_per_s else ("🔥" if s.heat.rate_f_per_s > 0 else "🥶")
    title = s.title + (" " + emoji if emoji else "")
    generic = {"whatif", "fyp"}
    topical = [h for h in s.hashtags if h.lower() not in generic][:3]
    tags = " ".join("#" + h for h in (*CAPTION_TAGS, *topical))
    return "\n".join(x for x in (title, s.teaser, f"{s.question} 👇", tags) if x)


def cmd_caption(args):
    print(caption_text(pick(args)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", action="append", default=[], help="extra scenarios YAML (repeatable)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    c = sub.add_parser("check")
    c.add_argument("id")
    c.set_defaults(fn=cmd_check)
    c = sub.add_parser("stills")
    c.add_argument("id")
    c.add_argument("times", type=float, nargs="+")
    c.add_argument("--out-dir", default="stills")
    c.add_argument("--sheet")
    c.add_argument("--seed", type=int, default=1)
    c.set_defaults(fn=cmd_stills)
    c = sub.add_parser("render")
    c.add_argument("id")
    c.add_argument("--out")
    c.add_argument("--max-mb", type=float, default=20.0, help="file size cap; 0 for none")
    c.add_argument("--fps", type=int, default=30)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--preset", default="medium")
    c.add_argument("--music-dir", help="folder of tracks to use instead of the synthesized ambience")
    c.set_defaults(fn=cmd_render)
    c = sub.add_parser("strip")
    c.add_argument("video")
    c.add_argument("times", type=float, nargs="*")
    c.add_argument("--out")
    c.set_defaults(fn=cmd_strip)
    c = sub.add_parser("caption")
    c.add_argument("id")
    c.set_defaults(fn=cmd_caption)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.fn(args)


if __name__ == "__main__":
    main()
