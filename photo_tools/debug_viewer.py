"""Interactive image debug viewer for inspecting tags and landmarks.

Uses ``kitten icat`` (Kitty terminal) to display images inline while showing
metadata (tags, landmarks, GPS, taken_at) below.  Navigate with arrow keys,
quit with ``q``.
"""

import argparse
import logging
import subprocess
import sys
import termios
import tty
from pathlib import Path

from photo_tools.autotag import get_gps_coords, _parse_exif_datetime
from photo_tools.helpers import find_images, get_existing_keywords, read_exif

log = logging.getLogger("inspect")


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

def _display(path: Path, meta: dict, index: int, total: int) -> None:
    """Clear screen, show image via kitten icat, and print metadata."""
    # Clear screen
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()

    # Display image via kitten icat
    subprocess.run(
        ["kitten", "icat", "--clear"],
        capture_output=True,
    )
    subprocess.run(
        ["kitten", "icat", str(path)],
        capture_output=False,
    )

    # Print metadata below the image
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
    total = len(images)
    idx = 0
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        while True:
            meta = _load_metadata(images[idx])
            _display(images[idx], meta, idx, total)
            key = _read_key()
            if key == "q":
                break
            elif key == "right":
                idx = (idx + 1) % total
            elif key == "left":
                idx = (idx - 1) % total
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        # Clear the icat image on exit
        subprocess.run(["kitten", "icat", "--clear"], capture_output=True)
        # Clear screen and reset cursor
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
