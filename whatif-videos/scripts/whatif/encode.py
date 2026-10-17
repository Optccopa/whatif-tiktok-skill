"""ffmpeg plumbing: raw RGB frames in, a TikTok-ready H.264/AAC MP4 out, optionally under a size cap."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

W, H = 1080, 1920
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".aac", ".wav", ".ogg", ".flac"}
SIZE_HEADROOM = 0.9  # aim for 90% of the upload limit


def bitrate_cap_kbps(max_bytes: int | None, seconds: float, audio_kbps: int) -> int | None:
    """Peak video bitrate that keeps the whole file under max_bytes."""
    if not max_bytes:
        return None
    total_kbps = max_bytes * 8 * SIZE_HEADROOM / seconds / 1000
    return max(300, int(total_kbps - audio_kbps))


def pick_music(music_dir: str, seed: str) -> Path | None:
    if not music_dir:
        return None
    tracks = sorted(p for p in Path(music_dir).glob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)
    return random.Random(seed).choice(tracks) if tracks else None


def ffmpeg_command(out: Path, fps: int, seconds: float, preset: str, crf: int, music: Path | None,
                   max_kbps: int | None = None, audio_kbps: int = 128, x264_params: str | None = None,
                   soundtrack: Path | None = None) -> list[str]:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
    ]
    if soundtrack:
        cmd += ["-i", str(soundtrack)]
    elif music:
        cmd += ["-stream_loop", "-1", "-i", str(music)]
    else:
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
    cmd += [
        "-map", "0:v", "-map", "1:a", "-t", f"{seconds:.3f}",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high",
        "-g", str(fps * 2),
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
        "-c:a", "aac", "-b:a", f"{audio_kbps}k",
    ]
    if x264_params:
        cmd += ["-x264-params", x264_params]
    if max_kbps:
        cmd += ["-maxrate", f"{max_kbps}k", "-bufsize", f"{max_kbps}k"]
    if music:
        cmd += ["-af", f"afade=t=in:d=0.5,afade=t=out:st={max(seconds - 1.5, 0):.2f}:d=1.5"]
    cmd += ["-movflags", "+faststart", str(out)]
    return cmd


def shrink_to_fit(path: Path, max_bytes: int, max_kbps: int, preset: str, crf: int, attempts: int = 2) -> None:
    """Re-encode at a lower peak bitrate in the rare case the capped encode still ran over."""
    for _ in range(attempts):
        if path.stat().st_size <= max_bytes:
            return
        max_kbps = int(max_kbps * 0.8)
        smaller = path.with_name(path.stem + ".fit.mp4")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
             "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high",
             "-maxrate", f"{max_kbps}k", "-bufsize", f"{max_kbps}k", "-pix_fmt", "yuv420p",
             "-c:a", "copy", "-movflags", "+faststart", str(smaller)],
            check=True, capture_output=True,
        )
        smaller.replace(path)
    if path.stat().st_size > max_bytes:
        raise RuntimeError(f"could not get {path.name} under {max_bytes / 1e6:.1f} MB")
