"""Per-turn trajectory capture.

Two interception points feed one TurnRecorder:
  - TrajectoryUI: a ConsoleUI-compatible object passed as `ui=` to GeneralAgent.
    Captures tool_start/tool_done pairs (with timing) and no-ops everything else
    (no terminal output — this fully replaces ConsoleUI for participant runs).
  - LoggingLLMClient: wraps the participant's already-constructed LLMClient instance
    (swapped in via `agent.llm = LoggingLLMClient(agent.llm, recorder, ui)` after
    GeneralAgent construction). Captures the EXACT messages array sent on every
    individual LLM call plus the response, independent of whatever GeneralAgent's
    internal context management (compaction, history) produces that call.

Both write into the same TurnRecorder.events list, ordered by a shared `seq` counter.
A TurnRecorder is never touched by more than one thread at a time (one recorder per
participant turn), so no locking is required.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from typing import Any

_print_lock = threading.Lock()


def log(label: str, message: str) -> None:
    """Thread-safe progress line to stdout -- so a real run (minutes to tens of
    minutes, several participants in parallel) shows visible progress instead of
    going silent until the whole meeting finishes."""
    ts = datetime.now().strftime("%H:%M:%S")
    with _print_lock:
        print(f"[{ts}] [{label}] {message}", flush=True)


def _now() -> datetime:
    return datetime.now()


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="milliseconds") if dt else None


# Public aliases for cross-module use (orchestrator needs the same clock/formatting).
now = _now
iso = _iso


def _duration_ms(start: datetime | None, end: datetime | None) -> int | None:
    if start is None or end is None:
        return None
    return int((end - start).total_seconds() * 1000)


def _parse_success(raw: str) -> bool:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return bool(data.get("success", True))
    except (json.JSONDecodeError, TypeError):
        pass
    return True


def _preview(raw: str, max_len: int = 800) -> str:
    return raw if len(raw) <= max_len else raw[:max_len] + "…"


def summarize_turn_actions(turn: dict[str, Any], max_len_per_line: int = 150) -> str:
    """Deterministic (non-LLM) compaction of a completed turn's tool_call events into
    a compact action log -- built from the same events already captured by
    TrajectoryUI/LoggingLLMClient, not from asking the model to recall its own actions.
    Used when carrying a participant's own previous round forward into the next round's
    prompt, alongside the round's model-authored `changes_from_prior_round` (see
    round_tools.py) and its answer text."""
    lines: list[str] = []
    for e in turn.get("events") or []:
        if e.get("type") != "tool_call":
            continue
        mark = "ok" if e.get("success") else "FAILED"
        args_preview = _preview(json.dumps(e.get("args") or {}, ensure_ascii=False), 60)
        result_preview = _preview(str(e.get("result_preview") or ""), max_len_per_line)
        lines.append(f"- {e.get('tool')}({args_preview}) -> {mark}: {result_preview}")
    return "\n".join(lines) if lines else "(no tool calls this round)"


class TurnRecorder:
    """Accumulates one participant's full trajectory for one meeting turn."""

    def __init__(self, agent: str, round_num: int, decided_by: str) -> None:
        self.turn_id = f"trn_{uuid.uuid4().hex[:10]}"
        self.agent = agent
        self.round = round_num
        self.decided_by = decided_by
        self.available_tools: list[str] = []
        self.events: list[dict[str, Any]] = []
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.session_id: str | None = None
        self.session_path: str | None = None
        self.output: str = ""
        self._seq = 0

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq - 1

    def to_turn_dict(self) -> dict[str, Any]:
        events_sorted = sorted(self.events, key=lambda e: e["seq"])
        first_llm_call = next((e for e in events_sorted if e["type"] == "llm_call"), None)
        return {
            "turn_id": self.turn_id,
            "agent": self.agent,
            "round": self.round,
            "decided_by": self.decided_by,
            "available_tools": sorted(self.available_tools),
            "input": first_llm_call["request_messages"] if first_llm_call else None,
            "output": self.output,
            "session_id": self.session_id,
            "session_path": self.session_path,
            "start_time": _iso(self.start_time),
            "end_time": _iso(self.end_time),
            "duration_ms": _duration_ms(self.start_time, self.end_time),
            "events": events_sorted,
        }


class TrajectoryUI:
    """ConsoleUI-compatible callback target — captures tool calls into the recorder,
    and (unless verbose=False) prints a one-line progress update per event so a real
    run shows visible progress instead of going silent until the whole meeting ends."""

    def __init__(self, recorder: TurnRecorder, verbose: bool = True) -> None:
        self.recorder = recorder
        self.iteration = 0
        self.verbose = verbose
        self._open_tool: tuple[str, dict[str, Any], datetime] | None = None

    def _log(self, message: str) -> None:
        if self.verbose:
            log(self.recorder.agent, message)

    # ── callbacks GeneralAgent invokes ──────────────────────────────────
    def session_start(self, session_id: str, task_id: str) -> None:
        self.recorder.session_id = session_id
        self._log("started")

    def model_start(self, iteration: int) -> None:
        self.iteration = iteration
        self._log(f"iter {iteration}: thinking...")

    def tool_start(self, name: str, args: dict[str, Any]) -> None:
        self._open_tool = (name, args, _now())
        self._log(f"iter {self.iteration}: tool_call {name}({_preview(json.dumps(args, ensure_ascii=False), 100)})")

    def tool_done(self, name: str, result: str) -> None:
        if self._open_tool is None:
            return
        _, args, start = self._open_tool
        end = _now()
        success = _parse_success(result)
        self.recorder.events.append({
            "seq": self.recorder.next_seq(),
            "type": "tool_call",
            "iteration": self.iteration,
            "tool": name,
            "args": args,
            "result_preview": _preview(result),
            "success": success,
            "start_time": _iso(start),
            "end_time": _iso(end),
            "duration_ms": _duration_ms(start, end),
        })
        self._open_tool = None
        mark = "ok" if success else "FAILED"
        self._log(f"iter {self.iteration}: {name} done in {_duration_ms(start, end)}ms ({mark})")

    def llm_thinking(self, iteration: int, content: str) -> None:
        pass  # captured via LoggingLLMClient's llm_call events instead

    def compact(self, reason: str) -> None:
        self.recorder.events.append({
            "seq": self.recorder.next_seq(),
            "type": "compact",
            "iteration": self.iteration,
            "reason": reason,
            "start_time": _iso(_now()),
            "end_time": None,
            "duration_ms": None,
        })
        self._log(f"context compacted: {reason}")

    def event(self, label: str, detail: str = "") -> None:
        pass

    def interrupt(self) -> None:
        pass

    def final(self) -> None:
        pass

    def final_answer(self, text: str, iterations: int) -> None:
        # recorder.output is set by the orchestrator from agent.run()'s return value
        self._log(f"finished after {iterations} iteration(s)")

    def saved(self, path: str) -> None:
        self.recorder.session_path = path

    def self_review_start(self) -> None:
        pass

    def self_review_done(self) -> None:
        pass


def _redact_message_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Returns a copy of `messages` with any image_url data URIs (e.g. injected by
    research_agent's view_image tool) replaced by a short placeholder. The real bytes
    can be tens of KB to a few MB per image, and because the full accumulated message
    history gets resent on every LLM call within a turn, storing them verbatim in the
    turn's saved record (runs/<meeting_id>.json) would multiply that many times over
    for no benefit -- nothing here changes what's actually sent to the API, which
    always gets the original, unredacted `messages`; only the copy written into the
    meeting's saved record goes through this."""
    redacted: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            redacted.append(msg)
            continue
        new_content: list[Any] = []
        changed = False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                size = len(url) if isinstance(url, str) else 0
                new_content.append({
                    "type": "image_url",
                    "image_url": (
                        f"[redacted image data, {size} base64 chars -- not stored "
                        "in the meeting record to keep runs/<meeting_id>.json a "
                        "reasonable size]"
                    ),
                })
                changed = True
            else:
                new_content.append(part)
        redacted.append({**msg, "content": new_content} if changed else msg)
    return redacted


class LoggingLLMClient:
    """Wraps a real LLMClient instance to log the exact messages sent on every .chat() call."""

    def __init__(self, inner: Any, recorder: TurnRecorder, ui: TrajectoryUI) -> None:
        self._inner = inner
        self._recorder = recorder
        self._ui = ui

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def provider(self) -> str:
        return self._inner.provider

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        start = _now()
        response = self._inner.chat(messages, tools)
        end = _now()
        message = response.choices[0].message
        tool_calls: list[dict[str, Any]] = []
        for tc in getattr(message, "tool_calls", None) or []:
            args_raw = tc.function.arguments or "{}"
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = args_raw
            tool_calls.append({"name": tc.function.name, "arguments": args})
        self._recorder.events.append({
            "seq": self._recorder.next_seq(),
            "type": "llm_call",
            "iteration": self._ui.iteration,
            "request_messages": _redact_message_images(messages),
            "response_text": message.content,
            "response_tool_calls": tool_calls,
            "start_time": _iso(start),
            "end_time": _iso(end),
            "duration_ms": _duration_ms(start, end),
        })
        if self._ui.verbose:
            outcome = f"{len(tool_calls)} tool call(s) requested" if tool_calls else "produced final text"
            self._ui._log(f"iter {self._ui.iteration}: model responded in {_duration_ms(start, end)}ms ({outcome})")
        return response

    def complete_text(self, prompt: str) -> str:
        return self._inner.complete_text(prompt)
