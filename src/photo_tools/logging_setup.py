"""Centralized logging configuration for photo-tools.

All photo-tools loggers live under the ``phototools.*`` hierarchy so
each subsystem (tagging, geocoding, ocr, ram, clip, landmarks, exif,
dates, duplicates, …) can be silenced or amplified independently via
``--log NAME=LEVEL,NAME=LEVEL`` or the ``PHOTOTOOLS_LOG`` env var.

The module also provides:

* ``get_logger(name)`` — returns ``phototools.<name>``.
* ``timed_step(name, …)`` — context manager that bookkeeps per-photo
  pipeline runs (timing + success/failure) for the per-photo summary.
* ``PhotoSummary`` — collects per-pipeline marks (·/✓/✗) for one photo.
* ``log_run_summary()`` — flushes the per-subsystem counters at the
  end of a ``tag run`` / ``tag fix`` invocation.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

ROOT = "phototools"

# ---------------------------------------------------------------------------
# Loggers and formatting
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return the ``phototools.<name>`` child logger."""
    return logging.getLogger(f"{ROOT}.{name}")


class _PhotoToolsFormatter(logging.Formatter):
    """``HH:MM:SS  LEVEL    [subsystem]   message`` — strips the ROOT prefix."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s  %(levelname)-7s  %(short_name)-13s  %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        name = record.name
        if name == ROOT:
            short = ""
        elif name.startswith(f"{ROOT}."):
            short = name[len(ROOT) + 1:]
        else:
            short = name  # external libraries
        record.short_name = f"[{short}]" if short else "[]"
        return super().format(record)


def setup_logging(verbose: bool = False, log_spec: str | None = None) -> None:
    """Install the photo-tools logging configuration.

    ``verbose`` toggles the whole ``phototools.*`` tree to DEBUG.
    ``log_spec`` is a comma-separated ``NAME=LEVEL`` list (CLI ``--log``);
    ``PHOTOTOOLS_LOG`` is consulted as a fallback. Per-subsystem overrides
    take precedence over the global verbose level.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler()
    handler.setFormatter(_PhotoToolsFormatter())
    root.addHandler(handler)
    # Keep third-party libraries quiet by default; raise individually if needed.
    root.setLevel(logging.WARNING)

    pt_root = logging.getLogger(ROOT)
    pt_root.setLevel(logging.DEBUG if verbose else logging.INFO)

    spec = log_spec if log_spec is not None else os.environ.get("PHOTOTOOLS_LOG", "")
    for entry in (s.strip() for s in spec.split(",") if s.strip()):
        if "=" not in entry:
            continue
        name, lvl = (p.strip() for p in entry.split("=", 1))
        if not name:
            continue
        level = getattr(logging, lvl.upper(), None)
        if not isinstance(level, int):
            continue
        get_logger(name).setLevel(level)

    # PaddleOCR is extremely chatty at INFO; silence its top-level loggers.
    for name in ("ppocr", "paddle", "paddlex"):
        logging.getLogger(name).setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Per-subsystem counters and timed steps
# ---------------------------------------------------------------------------

@dataclass
class _Counter:
    name: str
    runs: int = 0
    successes: int = 0
    errors: int = 0
    total_time: float = 0.0
    extras: dict[str, int | float] = field(default_factory=dict)

    def add(self, key: str, n: int | float = 1) -> None:
        self.extras[key] = self.extras.get(key, 0) + n


_counters: dict[str, _Counter] = {}


def get_counter(name: str) -> _Counter:
    c = _counters.get(name)
    if c is None:
        c = _counters[name] = _Counter(name=name)
    return c


@dataclass
class StepResult:
    """Outcome of a single pipeline run.

    Bodies set ``ran=True`` when work executed and ``ok=True`` on success.
    A step that exits without setting ``ran`` is treated as skipped and
    does not contribute to the subsystem's counters.
    """
    ran: bool = False
    ok: bool = False
    elapsed: float = 0.0


@contextmanager
def timed_step(name: str, *, photo: str | None = None, catch: bool = False):
    """Bracket one pipeline run for one photo.

    Usage::

        with timed_step("ocr", photo=path.name, catch=True) as s:
            phrases = run_ocr(path)
            s.ran = True
            s.ok = True

    When ``catch=True``, exceptions in the body are logged at WARNING and
    swallowed (the step is recorded as ran-but-failed). When ``catch=False``,
    exceptions still update the counter then re-raise.
    """
    log = get_logger(name)
    counter = get_counter(name)
    res = StepResult()
    t0 = time.perf_counter()
    failed = False
    try:
        yield res
    except Exception as e:
        res.ran = True
        failed = True
        if catch:
            log.warning("failed on %s: %s", photo or "?", e)
        else:
            res.elapsed = time.perf_counter() - t0
            counter.runs += 1
            counter.errors += 1
            counter.total_time += res.elapsed
            raise
    res.elapsed = time.perf_counter() - t0
    if res.ran or failed:
        counter.runs += 1
        counter.total_time += res.elapsed
        if failed or not res.ok:
            counter.errors += 1
        else:
            counter.successes += 1


# ---------------------------------------------------------------------------
# Per-photo summary line
# ---------------------------------------------------------------------------

def _tui_mark(name: str, status: str) -> None:
    """Forward a mark to the TUI when active. Lazy import to avoid cycles."""
    try:
        from photo_tools import tui
    except ImportError:
        return
    tui.mark(name, status)


class PhotoSummary:
    """One photo's per-pipeline marks for the trailing summary line.

    ``·`` = pipeline didn't run (skipped or filtered out).
    ``✓`` = pipeline ran and succeeded.
    ``✗`` = pipeline ran and failed.
    """

    DEFAULT_ORDER = ("gps", "ram", "landmarks", "ocr")

    def __init__(self) -> None:
        self.marks: dict[str, str] = {}

    def skip(self, name: str) -> None:
        self.marks[name] = "·"
        _tui_mark(name, "·")

    def ok(self, name: str) -> None:
        self.marks[name] = "✓"
        _tui_mark(name, "✓")

    def fail(self, name: str) -> None:
        self.marks[name] = "✗"
        _tui_mark(name, "✗")

    def record(self, name: str, ran: bool, ok: bool) -> None:
        if not ran:
            self.skip(name)
        elif ok:
            self.ok(name)
        else:
            self.fail(name)

    def render(self, order: tuple[str, ...] | None = None) -> str:
        names = order or self.DEFAULT_ORDER
        return " ".join(f"{n}{self.marks.get(n, '·')}" for n in names)


# ---------------------------------------------------------------------------
# End-of-run aggregate summary
# ---------------------------------------------------------------------------

def reset_counters() -> None:
    _counters.clear()


def log_run_summary(*, header: str = "Run summary:") -> None:
    """Emit the per-subsystem aggregate stats then reset counters."""
    if not _counters:
        return
    log = get_logger("tagging")
    log.info(header)
    for name in sorted(_counters):
        c = _counters[name]
        if c.runs == 0 and not c.extras:
            continue
        avg_ms = (c.total_time / c.runs * 1000.0) if c.runs else 0.0
        extras = ""
        if c.extras:
            extras = " | " + ", ".join(
                f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(c.extras.items())
            )
        log.info(
            "  %-12s %d ok, %d failed, avg %.0fms, total %.1fs%s",
            name, c.successes, c.errors, avg_ms, c.total_time, extras,
        )
    reset_counters()
