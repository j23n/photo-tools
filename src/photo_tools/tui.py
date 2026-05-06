"""Optional Rich-based TUI: log stream with persistent status panel.

When active, a status panel is pinned at the bottom of the terminal showing:

* current photo and ``[i/N]`` cursor
* per-pipeline marks for the in-flight photo (``?`` not started, ``·`` skipped,
  ``✓`` ok, ``✗`` failed)
* per-pipeline workload progress (for ``tag fix``)
* elapsed time and ETA

Logs print above the panel via the same Rich ``Console`` so the live region
stays anchored at the bottom and full scrollback is preserved.

The TUI is a no-op when stderr isn't a TTY, so piping to a file or running
under CI falls back to plain line-by-line logging without any branching at
the call sites.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field

# All state is module-level — only one tag run can be in flight at a time.
_state: "_TUIState | None" = None
_live = None
_console = None
_rich_handler: logging.Handler | None = None
_suppressed_handlers: list[tuple[logging.Logger, logging.Handler]] = []


@dataclass
class _TUIState:
    total: int = 0
    current_idx: int = 0
    current_photo: str = ""
    start_time: float = field(default_factory=time.time)
    photo_marks: dict[str, str] = field(default_factory=dict)
    workload: dict[str, list[int]] = field(default_factory=dict)  # name -> [done, total]
    header: str = ""


# Order pipelines are rendered in the per-photo summary line.
_PIPELINE_ORDER = ("gps", "ram", "landmarks", "ocr")


def is_active() -> bool:
    return _state is not None


def start(
    *,
    total: int,
    header: str = "tag run",
    workload: dict[str, int] | None = None,
    enabled: bool = True,
) -> None:
    """Begin TUI display.

    No-op when ``enabled`` is False or stderr isn't a TTY. Safe to call
    twice — a second call resets the per-run state but keeps the live
    region attached.
    """
    global _state, _live, _console, _rich_handler

    if not enabled or not sys.stderr.isatty():
        return

    if _state is not None:
        # Already running; reset for a new phase (rare — used by --watch).
        _state.total = total
        _state.current_idx = 0
        _state.current_photo = ""
        _state.start_time = time.time()
        _state.photo_marks.clear()
        _state.header = header
        _state.workload = (
            {k: [0, v] for k, v in workload.items()} if workload else {}
        )
        _refresh()
        return

    try:
        from rich.console import Console
        from rich.live import Live
    except ImportError:
        return

    _console = Console(stderr=True)
    _state = _TUIState(
        total=total,
        header=header,
        workload={k: [0, v] for k, v in workload.items()} if workload else {},
    )
    _live = Live(
        _render(),
        console=_console,
        refresh_per_second=4,
        transient=False,
        redirect_stdout=False,
        redirect_stderr=False,
    )
    _live.start()
    _install_log_handler()


def stop() -> None:
    """End TUI display and restore plain logging."""
    global _state, _live, _console, _rich_handler
    if _live is not None:
        try:
            _live.stop()
        except Exception:
            pass
    _restore_log_handlers()
    _state = None
    _live = None
    _console = None
    _rich_handler = None


def set_photo(idx: int, photo: str) -> None:
    if _state is None:
        return
    _state.current_idx = idx
    _state.current_photo = photo
    _state.photo_marks = {p: "?" for p in _PIPELINE_ORDER}
    _refresh()


def mark(pipeline: str, status: str) -> None:
    """Record a per-pipeline mark for the current photo.

    ``status`` is one of ``·`` skipped, ``✓`` ok, ``✗`` failed. When the
    pipeline appears in the workload dict, its ``done`` count is bumped
    only on real outcomes (``✓`` / ``✗``).
    """
    if _state is None:
        return
    _state.photo_marks[pipeline] = status
    if status in ("✓", "✗"):
        wl = _state.workload.get(pipeline)
        if wl is not None:
            wl[0] += 1
    _refresh()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt_dur(secs: float) -> str:
    secs = max(0, int(secs))
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _render():
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    s = _state
    if s is None:
        return Text("")

    elapsed = time.time() - s.start_time
    if s.current_idx > 0 and s.total > 0 and elapsed > 0:
        rate = s.current_idx / elapsed
        if rate > 0:
            remaining = (s.total - s.current_idx) / rate
            eta_str = f"  ETA {_fmt_dur(remaining)}"
        else:
            eta_str = ""
    else:
        eta_str = ""

    cursor = Text()
    if s.current_idx > 0:
        cursor.append(f"[{s.current_idx}/{s.total}] ", style="bold")
        cursor.append(s.current_photo, style="cyan")
        cursor.append("    ")
        for name in _PIPELINE_ORDER:
            m = s.photo_marks.get(name, "?")
            style = {
                "✓": "green",
                "✗": "red",
                "·": "dim",
                "?": "yellow",
            }.get(m, "")
            cursor.append(f"{name}{m}", style=style)
            cursor.append(" ")
    else:
        cursor.append(f"waiting (0/{s.total})", style="dim")

    rows = [cursor]
    if s.workload:
        wl_text = Text()
        wl_text.append("workload  ", style="dim")
        for name, (done, tot) in s.workload.items():
            wl_text.append(f"{name} {done}/{tot}  ")
        rows.append(wl_text)
    rows.append(Text(f"elapsed {_fmt_dur(elapsed)}{eta_str}", style="dim"))

    return Panel(Group(*rows), title=s.header, title_align="left",
                 border_style="blue", padding=(0, 1))


def _refresh() -> None:
    if _live is not None:
        _live.update(_render())


# ---------------------------------------------------------------------------
# Logging integration
# ---------------------------------------------------------------------------

class _RichLiveHandler(logging.Handler):
    """Emit formatted log records via a Rich Console.

    Routing through ``console.print`` lets ``rich.live.Live`` redraw the
    pinned status panel after each log line so the panel stays at the
    bottom and prior log lines remain in scrollback.
    """

    def __init__(self, console) -> None:
        super().__init__()
        self.console = console

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.console.print(msg, markup=False, highlight=False, soft_wrap=True)
        except Exception:
            self.handleError(record)


def _install_log_handler() -> None:
    """Mute the plain stderr StreamHandler and add a Rich-aware handler.

    The original handlers are stashed so ``stop()`` can restore them,
    making the TUI lifecycle reversible across multiple runs in the
    same process (e.g. tests).
    """
    global _rich_handler
    if _console is None:
        return

    from photo_tools.logging_setup import _PhotoToolsFormatter

    root = logging.getLogger()
    for h in list(root.handlers):
        # A console-bound StreamHandler would race with Live; suspend it.
        if isinstance(h, logging.StreamHandler) and not isinstance(h, _RichLiveHandler):
            root.removeHandler(h)
            _suppressed_handlers.append((root, h))

    handler = _RichLiveHandler(_console)
    handler.setFormatter(_PhotoToolsFormatter())
    root.addHandler(handler)
    _rich_handler = handler


def _restore_log_handlers() -> None:
    root = logging.getLogger()
    if _rich_handler is not None:
        try:
            root.removeHandler(_rich_handler)
        except Exception:
            pass
    while _suppressed_handlers:
        logger, handler = _suppressed_handlers.pop()
        if handler not in logger.handlers:
            logger.addHandler(handler)
