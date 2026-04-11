"""Interactive image debug viewer for inspecting tags and landmarks.

Uses ``kitten icat`` (Kitty terminal) to display images inline while showing
metadata (tags, landmarks, GPS, taken_at) below.  Navigate with arrow keys,
quit with ``q``.
"""

import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys
import termios
import tty
from pathlib import Path

from photo_tools.autotag import get_gps_coords, _parse_exif_datetime
from photo_tools.helpers import find_images, get_existing_keywords, read_exif

log = logging.getLogger("inspect")


# ---------------------------------------------------------------------------
# Image display backends
# ---------------------------------------------------------------------------

def _in_kitty() -> bool:
    """Return True if running inside a Kitty terminal."""
    return (os.environ.get("TERM", "") == "xterm-kitty"
            or os.environ.get("TERM_PROGRAM", "") == "kitty")


def _find_opener() -> list[str] | None:
    """Return the command for the system image opener, or None."""
    if platform.system() == "Darwin":
        return ["open"]
    if shutil.which("xdg-open"):
        return ["xdg-open"]
    return None


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def _load_metadata(path: Path) -> dict:
    """Read EXIF and return structured metadata for display."""
    exif = read_exif(path)
    gps = get_gps_coords(exif)
    taken_at = _parse_exif_datetime(exif)
    keywords = get_existing_keywords(exif)

    landmarks = sorted(t for t in keywords if t.startswith("landmark/"))
    tags = sorted(t for t in keywords if not t.startswith("landmark/"))

    return {
        "path": str(path.resolve()),
        "gps": f"{gps[0]:.6f}, {gps[1]:.6f}" if gps else "\u2014",
        "taken_at": taken_at.strftime("%Y-%m-%d %H:%M") if taken_at else "\u2014",
        "landmarks": ", ".join(landmarks) if landmarks else "\u2014",
        "tags": ", ".join(tags) if tags else "\u2014",
    }


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------

def _display_kitty(path: Path, meta: dict, index: int, total: int) -> None:
    """Display image inline via kitten icat and print metadata."""
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()

    subprocess.run(["kitten", "icat", "--clear"], capture_output=True)
    subprocess.run(["kitten", "icat", str(path)], capture_output=False)

    _print_meta(meta, index, total)


def _display_external(
    path: Path,
    meta: dict,
    index: int,
    total: int,
    opener: list[str],
    prev_proc: "subprocess.Popen | None",
) -> "subprocess.Popen | None":
    """Open image in an external viewer and print metadata in the terminal.

    Kills the previous viewer process (if any) before opening the new one.
    Returns the new Popen handle so the caller can track it.
    """
    if prev_proc is not None:
        try:
            prev_proc.terminate()
            prev_proc.wait(timeout=2)
        except Exception:
            try:
                prev_proc.kill()
            except Exception:
                pass

    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()

    proc = subprocess.Popen(
        opener + [str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    _print_meta(meta, index, total)
    return proc


def _print_meta(meta: dict, index: int, total: int) -> None:
    print()
    print(f"  [{index + 1}/{total}] {meta['path']}")
    print(f"  Taken:     {meta['taken_at']}")
    print(f"  GPS:       {meta['gps']}")
    print(f"  Landmarks: {meta['landmarks']}")
    print(f"  Tags:      {meta['tags']}")
    print()
    print("  [\u2190] prev   [\u2192] next   [q] quit")


# ---------------------------------------------------------------------------
# Keyboard input
# ---------------------------------------------------------------------------

def _read_key() -> str:
    """Read a single keypress and return a descriptive string.

    Recognises arrow-key escape sequences and single characters.
    Must be called while the terminal is in raw mode.
    """
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        seq = sys.stdin.read(1)
        if seq == "[":
            code = sys.stdin.read(1)
            if code == "C":
                return "right"
            if code == "D":
                return "left"
            # consume any remaining escape chars
            return "unknown"
        return "unknown"
    return ch


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def _interactive_loop(images: list[Path]) -> None:
    """Cycle through images with arrow keys, quit with q."""
    use_kitty = _in_kitty()
    opener = None if use_kitty else _find_opener()

    if not use_kitty and opener is None:
        log.error(
            "No image display method available. "
            "Run inside Kitty terminal, or install xdg-open."
        )
        sys.exit(1)

    total = len(images)
    idx = 0
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    viewer_proc: subprocess.Popen | None = None

    try:
        tty.setraw(fd)
        while True:
            meta = _load_metadata(images[idx])
            if use_kitty:
                _display_kitty(images[idx], meta, idx, total)
            else:
                viewer_proc = _display_external(
                    images[idx], meta, idx, total, opener, viewer_proc,
                )
            key = _read_key()
            if key == "q":
                break
            elif key == "right":
                idx = (idx + 1) % total
            elif key == "left":
                idx = (idx - 1) % total
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        if use_kitty:
            subprocess.run(["kitten", "icat", "--clear"], capture_output=True)
        elif viewer_proc is not None:
            try:
                viewer_proc.terminate()
                viewer_proc.wait(timeout=2)
            except Exception:
                try:
                    viewer_proc.kill()
                except Exception:
                    pass
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def build_inspect_parser(subparsers) -> argparse.ArgumentParser:
    """Register the 'inspect' subcommand."""
    sub = subparsers.add_parser(
        "inspect",
        help="Interactively view images with their tags, landmarks, GPS, and date.",
    )
    sub.add_argument("path", type=Path,
                     help="Image file or directory to inspect")
    sub.add_argument("-r", "--recursive", action="store_true",
                     help="Recurse into subdirectories")
    sub.set_defaults(func=run_inspect)
    return sub


def run_inspect(args) -> None:
    images = find_images(args.path, args.recursive)
    if not images:
        log.error("No supported images found at %s", args.path)
        sys.exit(1)

    log.info("Found %d image(s)", len(images))
    _interactive_loop(images)
