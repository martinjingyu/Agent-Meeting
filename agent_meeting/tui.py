"""Live multi-agent progress display for a running planning-rounds meeting.

Renders, while stdout is a real terminal: one overall progress bar for the
current round, and one row per participant below it showing that participant's
own agent-loop progress -- iteration/max_iterations of its GeneralAgent.run()
call, not the meeting round (the bar above already shows that) -- plus a single
live status line: either the tool call it's currently waiting on, or a preview
of what it just sent the model while waiting for a response.

Falls back to a no-op when stdout isn't a real terminal (piped output, log
capture, CI) -- rich's Live display assumes a real terminal to redraw in
place, and would otherwise either render nothing useful or spam partial
frames. TrajectoryUI's existing plain log() lines (runner.py's `verbose=`
plumbing) keep working unchanged in that case; MeetingProgress is a purely
additive presentation layer, never a replacement for the JSON trajectory
recorded by TurnRecorder/TrajectoryUI.
"""
from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from typing import Iterator, Literal

from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

Phase = Literal["idle", "waiting_llm", "tool_call", "done"]

_PREVIEW_LEN = 90

_PHASE_LABEL = {
    "idle": "[dim]idle[/dim]",
    "waiting_llm": "[yellow]thinking[/yellow]",
    "tool_call": "[cyan]tool call[/cyan]",
    "done": "[green]done[/green]",
}


def preview(text: str, limit: int = _PREVIEW_LEN) -> str:
    """Single-line, whitespace-collapsed preview -- used for both the tool-call
    args and the outgoing LLM message content, so a status line never wraps or
    breaks the live display's row layout."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


@dataclass
class _AgentRow:
    name: str
    iteration: int = 0
    max_iterations: int = 1
    phase: Phase = "idle"
    detail: str = ""


class MeetingProgress:
    """Shared, thread-safe live display for one meeting run.

    One instance per _run_planning_rounds() call. Participants update it
    concurrently from their own ThreadPoolExecutor worker threads (see
    runner.py's _run_planning_round), so every state mutation goes through
    `_lock`; rich's Live.update() is called with the lock released to avoid
    holding it across a render.
    """

    def __init__(self, max_rounds: int, agent_names: list[str]) -> None:
        self.max_rounds = max_rounds
        self.round_num = 0
        self._lock = threading.Lock()
        # max_iterations is per-participant and not known until its GeneralAgent is
        # actually constructed (role fallback in runner.py's _execute_turn), so this
        # placeholder is corrected by the first update_agent() call for each name --
        # a row is only visible once its turn has actually started anyway.
        self._rows: dict[str, _AgentRow] = {name: _AgentRow(name=name) for name in agent_names}
        self._console = Console()
        self._live: Live | None = None
        self.enabled = self._console.is_terminal

    def __enter__(self) -> "MeetingProgress":
        if self.enabled:
            self._live = Live(
                self._render(), console=self._console, refresh_per_second=4, transient=False,
            )
            self._live.__enter__()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._live is not None:
            self._live.__exit__(*exc_info)
            self._live = None

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        """Suspend rich's Live refresh for the duration of a blocking terminal prompt
        (input()) -- e.g. MeetingConfig.human_checkin's judge question. Left running,
        Live's own redraw (4/sec, in place via cursor movement) would land on top of
        whatever the caller prints/reads. rich's documented pattern for this is
        stop() before / start() after, not a bare context suspension -- start(refresh=
        True) forces an immediate redraw from the current state so the display
        doesn't sit stale until the next natural tick. No-op when never entered
        (self._live is None, e.g. non-terminal stdout)."""
        if self._live is None:
            yield
            return
        self._live.stop()
        try:
            yield
        finally:
            self._live.start(refresh=True)

    def set_round(self, round_num: int) -> None:
        with self._lock:
            self.round_num = round_num
        self._refresh()

    def update_agent(
        self, name: str, *, iteration: int, max_iterations: int, phase: Phase, detail: str = ""
    ) -> None:
        with self._lock:
            row = self._rows.setdefault(name, _AgentRow(name=name))
            row.iteration = iteration
            row.max_iterations = max_iterations
            row.phase = phase
            row.detail = detail
        self._refresh()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> Group:
        overall = Progress(
            TextColumn("[bold]Meeting[/bold]"),
            BarColumn(),
            TextColumn(f"round {self.round_num}/{self.max_rounds}"),
        )
        overall.add_task("round", total=max(1, self.max_rounds), completed=min(self.round_num, self.max_rounds))

        table = Table.grid(padding=(0, 1, 0, 0))
        table.add_column(width=22, no_wrap=True)
        table.add_column(width=18, no_wrap=True)
        table.add_column(no_wrap=True)
        with self._lock:
            rows = sorted(self._rows.values(), key=lambda r: r.name)
            for row in rows:
                iter_label = f"[{row.iteration}/{row.max_iterations}]"
                table.add_row(
                    f"[bold]{row.name}[/bold]",
                    f"{iter_label} {_PHASE_LABEL[row.phase]}",
                    preview(row.detail),
                )
        return Group(overall, table)
