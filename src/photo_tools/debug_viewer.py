"""Interactive image debug viewer for inspecting tags and landmarks.

Opens images in the system viewer (``xdg-open`` / ``open``) while showing
metadata (tags, landmarks, GPS, taken_at) in the terminal.  Navigate with
arrow keys, quit with ``q``.
"""

import os
import platform
import shutil
import signal
import subprocess
import sys
import termios
import tty
from pathlib import Path

from photo_tools.autotag import _parse_exif_datetime, get_gps_coords
from photo_tools.helpers import find_images, get_existing_keywords, read_exif
from photo_tools.logging_setup import get_logger

log = get_logger("inspect")


# ---------------------------------------------------------------------------
# Image display backends
# ---------------------------------------------------------------------------

def _find_opener(geometry: str = "800x600") -> list[str] | None:
    """Return the command for the system image opener, or None.

    Prefers simple viewers that stay as a child process so we can kill
    them on navigation.  ``xdg-open`` and GNOME apps delegate via D-Bus,
    making the actual viewer unkillable.

    *geometry* is an X11 geometry string, e.g. ``"800x600"`` (size only),
    ``"+0+0"`` (position only), or ``"800x600+0+0"`` (both).
    """
    # Extract size portion (before any +) for ImageMagick -resize
    size_part = geometry.split("+")[0]  # "" when geometry is "+0+0"
    resize = (size_part + ">") if size_part else "800x600>"

    display_args = ["display", "-auto-orient", "-resize", resize]
    if "+" in geometry:
        display_args += ["-geometry", geometry]

    candidates: list[tuple[str, list[str]]] = [
        ("feh",         ["feh", "--scale-down", "--geometry", geometry]),
        ("imv",         ["imv", "-s", "shrink"]),
        ("imv-wayland", ["imv-wayland", "-s", "shrink"]),
        ("nsxiv",       ["nsxiv", "-g", geometry]),
        ("sxiv",        ["sxiv", "-g", geometry]),
        ("display",     display_args),
    ]
    for cmd, argv in candidates:
        if shutil.which(cmd):
            return argv
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

    landmarks = sorted(t for t in keywords if t.startswith("Landmarks/"))
    tags = sorted(t for t in keywords if not t.startswith("Landmarks/"))

    return {
        "path": str(path.resolve()),
        "gps": f"{gps[0]:.6f}, {gps[1]:.6f}" if gps else "\u2014",
        "taken_at": taken_at.strftime("%Y-%m-%d %H:%M") if taken_at else "\u2014",
        "landmarks": landmarks,
        "tags": tags,
    }


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------

def _kill_viewer(proc: subprocess.Popen) -> None:
    """Terminate the viewer process group, falling back to direct kill."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=2)
    except ProcessLookupError:
        pass
    except Exception:
        try:
            proc.kill()
        except ProcessLookupError:
            pass


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
        _kill_viewer(prev_proc)

    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()

    proc = subprocess.Popen(
        opener + [str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    _print_meta(meta, index, total)
    return proc


def _truncate(text: str, width: int) -> str:
    """Truncate *text* to *width* columns, adding ellipsis if needed."""
    if len(text) <= width:
        return text
    return text[: width - 1] + "\u2026"


def _print_meta(meta: dict, index: int, total: int) -> None:
    """Print image metadata, safe for raw-terminal mode."""
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80

    lines = [
        "",
        f"  [{index + 1}/{total}] {meta['path']}",
        f"  Taken:  {meta['taken_at']}",
        f"  GPS:    {meta['gps']}",
    ]
    if meta["landmarks"]:
        lines.append("  Landmarks:")
        for lm in meta["landmarks"]:
            lines.append(f"    {lm}")
    if meta["tags"]:
        lines.append("  Tags:")
        for tag in meta["tags"]:
            lines.append(f"    {tag}")
    lines += ["", "  [\u2190] prev   [\u2192] next   [q] quit"]

    for line in lines:
        sys.stdout.write(_truncate(line, cols) + "\r\n")
    sys.stdout.flush()


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
            if code == "A":
                return "up"
            if code == "B":
                return "down"
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
    opener = _find_opener()

    if opener is None:
        log.error(
            "No image viewer found. Install xdg-open (or run on macOS)."
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
        if viewer_proc is not None:
            _kill_viewer(viewer_proc)
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def add_inspect_subparser(tags_sub, parents=None) -> None:
    """Register 'inspect' as a sub-command of the 'tags' command."""
    p = tags_sub.add_parser(
        "inspect",
        parents=parents or [],
        help="Interactively view images with their tags, landmarks, GPS, and date.",
    )
    p.add_argument("path", type=Path,
                   help="Image file or directory to inspect")
    p.set_defaults(func=run_inspect)


def run_inspect(args) -> None:
    images = find_images(args.path)
    if not images:
        log.error("No supported images found at %s", args.path)
        sys.exit(1)

    log.info("Found %d image(s)", len(images))
    _interactive_loop(images)
